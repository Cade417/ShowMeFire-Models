import unittest

import numpy as np
import pandas as pd

from risk_fusion import risk_fusion_contract as contract


class AssignEpisodesTests(unittest.TestCase):
    def test_consecutive_days_within_the_same_episode_get_the_same_id(self):
        dates = pd.Series(["2020-01-01", "2020-01-02", "2020-01-03"])
        episodes = contract.assign_episodes(dates)
        self.assertEqual(episodes.iloc[0], episodes.iloc[1])
        self.assertEqual(episodes.iloc[1], episodes.iloc[2])

    def test_episode_boundary_is_exactly_14_days(self):
        # Find an actual episode boundary relative to the fixed EPOCH rather
        # than assuming an arbitrary date falls on one: the last day of
        # some episode N, and the first day of episode N+1, must differ.
        anchor = contract.EPOCH + pd.Timedelta(days=140)  # exactly on a 14-day boundary
        last_day_of_episode = anchor + pd.Timedelta(days=13)
        first_day_of_next_episode = anchor + pd.Timedelta(days=14)
        dates = pd.Series([anchor, last_day_of_episode, first_day_of_next_episode])
        episodes = contract.assign_episodes(dates)
        self.assertEqual(episodes.iloc[0], episodes.iloc[1])
        self.assertNotEqual(episodes.iloc[1], episodes.iloc[2])

    def test_episode_id_is_stable_across_different_panels(self):
        # Same calendar date must produce the same episode_id whether it
        # appears in a small test panel or the full real dataset - that's
        # what the fixed EPOCH (not "first date in this panel") guarantees.
        a = contract.assign_episodes(pd.Series(["2020-06-15"]))
        b = contract.assign_episodes(pd.Series(["2019-01-01", "2020-06-15"]))
        self.assertEqual(a.iloc[0], b.iloc[1])


class AssignBlocksTests(unittest.TestCase):
    def test_produces_contiguous_chronological_blocks(self):
        episode_ids = pd.Series(range(100))  # already-sorted, contiguous episode ids
        blocks = contract.assign_blocks(episode_ids, n_blocks=5)
        # Block assignment must be monotonically non-decreasing with episode_id.
        self.assertTrue((blocks.diff().dropna() >= 0).all())

    def test_every_row_of_the_same_episode_gets_the_same_block(self):
        episode_ids = pd.Series([1, 1, 1, 2, 2, 3])
        blocks = contract.assign_blocks(episode_ids, n_blocks=3)
        self.assertEqual(blocks[episode_ids == 1].nunique(), 1)
        self.assertEqual(blocks[episode_ids == 2].nunique(), 1)

    def test_block_sizes_are_roughly_balanced_by_episode_count(self):
        episode_ids = pd.Series(range(100))
        blocks = contract.assign_blocks(episode_ids, n_blocks=5)
        counts = blocks.value_counts()
        self.assertEqual(len(counts), 5)
        self.assertTrue((counts >= 18).all() and (counts <= 22).all())


class CrossfitIndicesTests(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range("2020-01-01", "2020-12-31", freq="D").strftime("%Y-%m-%d")
        self.panel = contract.add_split_columns(pd.DataFrame({"valid_local_date": dates}))

    def test_train_and_test_are_disjoint_and_cover_everything(self):
        for train_idx, test_idx in contract.crossfit_indices(self.panel, n_blocks=5):
            self.assertEqual(len(set(train_idx) & set(test_idx)), 0)
            self.assertEqual(len(train_idx) + len(test_idx), len(self.panel))

    def test_yields_exactly_n_blocks_folds(self):
        folds = list(contract.crossfit_indices(self.panel, n_blocks=5))
        self.assertEqual(len(folds), 5)

    def test_no_episode_is_split_across_train_and_test(self):
        for _, test_idx in contract.crossfit_indices(self.panel, n_blocks=5):
            test_episodes = set(self.panel.loc[test_idx, "episode_id"])
            train_episodes = set(self.panel.loc[~self.panel.index.isin(test_idx), "episode_id"])
            self.assertEqual(len(test_episodes & train_episodes), 0)

    def test_raises_without_block_column(self):
        with self.assertRaises(ValueError):
            list(contract.crossfit_indices(pd.DataFrame({"valid_local_date": ["2020-01-01"]})))


class ManifestTests(unittest.TestCase):
    def test_manifest_reports_correct_totals(self):
        dates = pd.date_range("2020-01-01", "2020-03-31", freq="D").strftime("%Y-%m-%d")
        panel = contract.add_split_columns(pd.DataFrame({
            "valid_local_date": dates, "region_id": [i % 3 for i in range(len(dates))],
        }))
        manifest = contract.create_manifest(panel)
        self.assertEqual(manifest["total_rows"], len(panel))
        self.assertEqual(sum(b["row_count"] for b in manifest["blocks"]), len(panel))
        self.assertEqual(manifest["split_version"], contract.SPLIT_VERSION)

    def test_manifest_requires_split_columns(self):
        with self.assertRaises(ValueError):
            contract.create_manifest(pd.DataFrame({"valid_local_date": ["2020-01-01"]}))

    def test_save_and_load_round_trip(self):
        import tempfile
        from pathlib import Path
        dates = pd.date_range("2020-01-01", "2020-01-31", freq="D").strftime("%Y-%m-%d")
        panel = contract.add_split_columns(pd.DataFrame({"valid_local_date": dates}))
        manifest = contract.create_manifest(panel)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            contract.save_manifest(manifest, path)
            loaded = contract.load_manifest(path)
        self.assertEqual(loaded["manifest_sha256"], manifest["manifest_sha256"])


if __name__ == "__main__":
    unittest.main()
