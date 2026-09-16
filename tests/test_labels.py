import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from risk_fusion import labels


def _write_csv_and_manifest(tmpdir, rows, min_tier="admin_reviewed"):
    csv_path = Path(tmpdir) / "labels.csv"
    frame = pd.DataFrame(rows)
    frame.to_csv(csv_path, index=False)
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    rows_by_tier = {"unverified": 0, "admin_reviewed": 0, "official_source_confirmed": 0}
    for row in rows:
        rows_by_tier[row["verification_tier"]] = rows_by_tier.get(row["verification_tier"], 0) + 1
    manifest = {
        "schema_version": "fire-labels-v1", "min_tier": min_tier, "row_count": len(rows),
        "rows_by_tier": rows_by_tier, "csv_sha256": digest,
    }
    manifest_path = Path(tmpdir) / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return csv_path, manifest_path


def _row(event_id, lat, lon, occurred_at, tier="official_source_confirmed",
         cause="wildfire", acres=1.0, county="29019"):
    return {
        "event_id": event_id, "source": "official", "verification_tier": tier,
        "label_weight": 1.0, "latitude": lat, "longitude": lon, "county_fips": county,
        "occurred_at": occurred_at, "occurred_at_precision": "minute", "cause_category": cause,
        "acres": acres, "acres_is_estimate": 0, "fuel_types": "", "frp": None,
        "confidence": None, "satellite": None, "label_revision": 1, "revised_at": None,
    }


class LoadFireLabelsTests(unittest.TestCase):
    def test_loads_valid_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path, manifest_path = _write_csv_and_manifest(tmp, [_row(1, 38.9, -92.3, "2020-07-01T18:00:00Z")])
            frame, manifest = labels.load_fire_labels(csv_path, manifest_path)
            self.assertEqual(len(frame), 1)
            self.assertEqual(manifest["schema_version"], "fire-labels-v1")

    def test_rejects_checksum_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path, manifest_path = _write_csv_and_manifest(tmp, [_row(1, 38.9, -92.3, "2020-07-01T18:00:00Z")])
            csv_path.write_text(csv_path.read_text() + "\n# tampered")
            with self.assertRaises(ValueError):
                labels.load_fire_labels(csv_path, manifest_path)

    def test_rejects_self_inconsistent_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [_row(1, 38.9, -92.3, "2020-07-01T18:00:00Z", tier="unverified")]
            csv_path, manifest_path = _write_csv_and_manifest(tmp, rows)
            manifest = json.loads(manifest_path.read_text())
            manifest["rows_by_tier"]["unverified"] = 0  # lie: csv actually has 1 unverified row
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                labels.load_fire_labels(csv_path, manifest_path)

    def test_drops_unverified_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [_row(1, 38.9, -92.3, "2020-07-01T18:00:00Z", tier="unverified"),
                   _row(2, 38.9, -92.3, "2020-07-01T18:00:00Z", tier="official_source_confirmed")]
            csv_path, manifest_path = _write_csv_and_manifest(tmp, rows)
            frame, _ = labels.load_fire_labels(csv_path, manifest_path)
            self.assertEqual(len(frame), 1)
            self.assertEqual(frame.iloc[0]["event_id"], 2)


class CauseFilterTests(unittest.TestCase):
    def test_excludes_prescribed_and_agricultural(self):
        frame = pd.DataFrame([
            _row(1, 38.9, -92.3, "2020-07-01T18:00:00Z", cause="prescribed"),
            _row(2, 38.9, -92.3, "2020-07-01T18:00:00Z", cause="agricultural"),
            _row(3, 38.9, -92.3, "2020-07-01T18:00:00Z", cause="wildfire"),
        ])
        filtered, excluded = labels.apply_cause_filter(frame)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(excluded, 2)
        self.assertEqual(filtered.iloc[0]["event_id"], 3)


class ClusterEventsTests(unittest.TestCase):
    def test_nearby_same_day_events_are_clustered(self):
        # ~1.5 km apart (roughly 0.0135 deg lat), 2 hours apart - inside both thresholds.
        frame = pd.DataFrame([
            _row(1, 38.9500, -92.3300, "2020-07-01T18:00:00Z"),
            _row(2, 38.9635, -92.3300, "2020-07-01T20:00:00Z"),
        ])
        clustered = labels.cluster_events(frame)
        self.assertEqual(clustered.iloc[0]["cluster_id"], clustered.iloc[1]["cluster_id"])
        self.assertEqual(clustered.iloc[0]["cluster_size"], 2)

    def test_far_apart_events_are_not_clustered(self):
        frame = pd.DataFrame([
            _row(1, 38.9500, -92.3300, "2020-07-01T18:00:00Z"),
            _row(2, 39.5000, -93.0000, "2020-07-01T18:05:00Z"),  # ~75km away
        ])
        clustered = labels.cluster_events(frame)
        self.assertNotEqual(clustered.iloc[0]["cluster_id"], clustered.iloc[1]["cluster_id"])
        self.assertTrue((clustered["cluster_size"] == 1).all())

    def test_same_location_but_too_far_apart_in_time_is_not_clustered(self):
        frame = pd.DataFrame([
            _row(1, 38.9500, -92.3300, "2020-07-01T00:00:00Z"),
            _row(2, 38.9500, -92.3300, "2020-07-02T12:00:00Z"),  # 36 hours later
        ])
        clustered = labels.cluster_events(frame)
        self.assertNotEqual(clustered.iloc[0]["cluster_id"], clustered.iloc[1]["cluster_id"])

    def test_empty_frame_does_not_raise(self):
        frame = pd.DataFrame(columns=["event_id", "latitude", "longitude", "occurred_at", "verification_tier"])
        clustered = labels.cluster_events(frame)
        self.assertEqual(len(clustered), 0)

    def test_representative_prefers_higher_tier(self):
        frame = pd.DataFrame([
            _row(1, 38.9500, -92.3300, "2020-07-01T18:00:00Z", tier="admin_reviewed"),
            _row(2, 38.9505, -92.3300, "2020-07-01T18:30:00Z", tier="official_source_confirmed"),
        ])
        clustered = labels.cluster_events(frame)
        representatives = labels.representative_events(clustered)
        self.assertEqual(len(representatives), 1)
        self.assertEqual(representatives.iloc[0]["event_id"], 2)

    def test_representative_tiebreaks_on_earliest_time(self):
        frame = pd.DataFrame([
            _row(1, 38.9500, -92.3300, "2020-07-01T20:00:00Z"),
            _row(2, 38.9505, -92.3300, "2020-07-01T18:00:00Z"),
        ])
        clustered = labels.cluster_events(frame)
        representatives = labels.representative_events(clustered)
        self.assertEqual(representatives.iloc[0]["event_id"], 2)

    def test_duplicate_collapse_rate_computation(self):
        frame = pd.DataFrame([
            _row(1, 38.9500, -92.3300, "2020-07-01T18:00:00Z"),
            _row(2, 38.9505, -92.3300, "2020-07-01T18:30:00Z"),  # duplicate of 1
            _row(3, 39.5000, -93.0000, "2020-07-05T18:00:00Z"),  # standalone
        ])
        clustered = labels.cluster_events(frame)
        rate = labels.duplicate_collapse_rate(clustered, tier="official_source_confirmed")
        self.assertAlmostEqual(rate, 2 / 3)


class CountyDayCountsTests(unittest.TestCase):
    def test_aggregates_events_per_county_and_local_date(self):
        frame = pd.DataFrame([
            _row(1, 38.95, -92.33, "2020-07-02T04:00:00Z", county="29019", acres=2.0),  # 2020-07-01 23:00 CDT
            _row(2, 38.96, -92.34, "2020-07-02T14:00:00Z", county="29019", acres=3.0),  # 2020-07-02 09:00 CDT
            _row(3, 39.10, -92.50, "2020-07-02T14:00:00Z", county="29027", acres=1.0),
        ])
        clustered = labels.cluster_events(frame)
        representatives = labels.representative_events(clustered)
        counts = labels.to_county_day_counts(representatives)
        boone_0701 = counts[(counts.county_fips == "29019") & (counts.valid_local_date == "2020-07-01")]
        boone_0702 = counts[(counts.county_fips == "29019") & (counts.valid_local_date == "2020-07-02")]
        self.assertEqual(len(boone_0701), 1)
        self.assertEqual(boone_0701.iloc[0]["event_count"], 1)
        self.assertEqual(len(boone_0702), 1)

    def test_excludes_prescribed_and_agricultural_from_counts(self):
        frame = pd.DataFrame([
            _row(1, 38.95, -92.33, "2020-07-01T18:00:00Z", cause="prescribed"),
        ])
        clustered = labels.cluster_events(frame)
        representatives = labels.representative_events(clustered)
        counts = labels.to_county_day_counts(representatives)
        self.assertEqual(len(counts), 0)

    def test_empty_input_returns_empty_frame_with_correct_columns(self):
        frame = pd.DataFrame(columns=["event_id", "latitude", "longitude", "occurred_at",
                                      "verification_tier", "cause_category", "acres", "county_fips"])
        counts = labels.to_county_day_counts(frame)
        self.assertEqual(len(counts), 0)
        self.assertIn("event_count", counts.columns)


class BuildLabelsTests(unittest.TestCase):
    def test_end_to_end_returns_expected_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [
                _row(1, 38.9500, -92.3300, "2020-07-01T18:00:00Z"),
                _row(2, 39.5000, -93.0000, "2020-07-05T18:00:00Z", tier="admin_reviewed"),
            ]
            csv_path, manifest_path = _write_csv_and_manifest(tmp, rows)
            result = labels.build_labels(csv_path, manifest_path)
            self.assertIn("primary_counts", result)
            self.assertIn("auxiliary_counts", result)
            self.assertEqual(result["raw_row_count"], 2)
            self.assertEqual(len(result["primary_counts"]), 1)
            self.assertEqual(len(result["auxiliary_counts"]), 1)

    def test_raises_when_duplicate_collapse_rate_exceeds_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            # 10 official events, 9 of which cluster into one duplicate blob -> 90% collapse rate.
            rows = [_row(1, 38.9500, -92.3300, "2020-07-01T18:00:00Z")]
            for i in range(2, 11):
                rows.append(_row(i, 38.9500 + i * 0.0001, -92.3300, "2020-07-01T18:05:00Z"))
            csv_path, manifest_path = _write_csv_and_manifest(tmp, rows)
            with self.assertRaises(ValueError):
                labels.build_labels(csv_path, manifest_path)


if __name__ == "__main__":
    unittest.main()
