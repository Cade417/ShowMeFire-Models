import unittest

import numpy as np
import pandas as pd

from fire_weather_ml import contract


class AssignEpisodesTests(unittest.TestCase):
    def test_consecutive_hours_within_the_same_episode_get_the_same_id(self):
        timestamps = pd.Series(["2020-01-01T00:00:00", "2020-01-01T06:00:00", "2020-01-02T00:00:00"])
        episodes = contract.assign_episodes(timestamps)
        self.assertEqual(episodes.iloc[0], episodes.iloc[1])
        self.assertEqual(episodes.iloc[1], episodes.iloc[2])

    def test_episode_boundary_is_exactly_5_days(self):
        anchor = contract.EPOCH + pd.Timedelta(days=50)  # exactly on a 5-day (120h) boundary
        last_hour_of_episode = anchor + pd.Timedelta(hours=119)
        first_hour_of_next_episode = anchor + pd.Timedelta(hours=120)
        timestamps = pd.Series([anchor, last_hour_of_episode, first_hour_of_next_episode])
        episodes = contract.assign_episodes(timestamps)
        self.assertEqual(episodes.iloc[0], episodes.iloc[1])
        self.assertNotEqual(episodes.iloc[1], episodes.iloc[2])

    def test_episode_id_is_stable_across_different_panels(self):
        a = contract.assign_episodes(pd.Series(["2020-06-15T00:00:00"]))
        b = contract.assign_episodes(pd.Series(["2019-01-01T00:00:00", "2020-06-15T00:00:00"]))
        self.assertEqual(a.iloc[0], b.iloc[1])

    def test_handles_tz_aware_timestamps_the_same_as_naive_ones(self):
        # The real historical panel's valid_time is UTC-aware
        # ("...+00:00") - this must not raise, and must agree with the
        # naive form of the same instant.
        naive = contract.assign_episodes(pd.Series(["2020-06-15T00:00:00"]))
        aware = contract.assign_episodes(pd.Series(["2020-06-15T00:00:00+00:00"]))
        self.assertEqual(naive.iloc[0], aware.iloc[0])


class AssignBlocksTests(unittest.TestCase):
    def test_produces_contiguous_chronological_blocks(self):
        episode_ids = pd.Series(range(100))
        blocks = contract.assign_blocks(episode_ids, n_blocks=5)
        self.assertTrue((blocks.diff().dropna() >= 0).all())

    def test_every_row_of_the_same_episode_gets_the_same_block(self):
        episode_ids = pd.Series([1, 1, 1, 2, 2, 3])
        blocks = contract.assign_blocks(episode_ids, n_blocks=3)
        self.assertEqual(blocks[episode_ids == 1].nunique(), 1)
        self.assertEqual(blocks[episode_ids == 2].nunique(), 1)

    def test_handles_fewer_unique_episodes_than_requested_blocks(self):
        episode_ids = pd.Series([1, 1, 2])
        blocks = contract.assign_blocks(episode_ids, n_blocks=5)
        self.assertEqual(blocks.nunique(), 2)


class CrossfitIndicesTests(unittest.TestCase):
    def setUp(self):
        timestamps = pd.date_range("2020-01-01", periods=240, freq="h").astype(str)
        self.panel = pd.DataFrame({"valid_time": timestamps})
        self.panel = contract.add_split_columns(self.panel)

    def test_yields_one_fold_per_block_with_disjoint_test_sets(self):
        seen_test_rows = set()
        fold_count = 0
        for train_index, test_index in contract.crossfit_indices(self.panel):
            fold_count += 1
            self.assertTrue(set(test_index).isdisjoint(set(train_index)))
            seen_test_rows.update(test_index.tolist())
        self.assertEqual(fold_count, self.panel["block"].nunique())
        self.assertEqual(seen_test_rows, set(self.panel.index))

    def test_raises_without_block_column(self):
        with self.assertRaises(ValueError):
            list(contract.crossfit_indices(pd.DataFrame({"valid_time": ["2020-01-01T00:00:00"]})))


class CreateManifestTests(unittest.TestCase):
    def test_manifest_is_deterministic_for_identical_input(self):
        timestamps = pd.date_range("2020-01-01", periods=240, freq="h").astype(str)
        panel = contract.add_split_columns(pd.DataFrame({"valid_time": timestamps}))
        manifest_a = contract.create_manifest(panel)
        manifest_b = contract.create_manifest(panel)
        self.assertEqual(manifest_a["manifest_sha256"], manifest_b["manifest_sha256"])

    def test_raises_without_split_columns(self):
        with self.assertRaises(ValueError):
            contract.create_manifest(pd.DataFrame({"valid_time": ["2020-01-01T00:00:00"]}))


class ManifestRoundTripTests(unittest.TestCase):
    def test_save_then_load_round_trips(self):
        import tempfile
        from pathlib import Path

        timestamps = pd.date_range("2020-01-01", periods=24, freq="h").astype(str)
        panel = contract.add_split_columns(pd.DataFrame({"valid_time": timestamps}), n_blocks=2)
        manifest = contract.create_manifest(panel, n_blocks=2)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "manifest.json"
            contract.save_manifest(manifest, path)
            reloaded = contract.load_manifest(path)
        self.assertEqual(reloaded["manifest_sha256"], manifest["manifest_sha256"])


if __name__ == "__main__":
    unittest.main()
