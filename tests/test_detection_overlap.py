import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from detection_dataset import overlap


def _history():
    return pd.DataFrame({
        "external_id": ["a", "b", "c", "d"], "source": ["viirs", "viirs", "modis", "viirs"],
        "occurred_at": ["2026-01-10T18:00:00Z", "2026-01-10T18:00:00Z", "2026-01-12T18:00:00Z", "2027-05-01T18:00:00Z"],
        "latitude": [38.0, 38.5, 38.0, 38.0], "longitude": [-92.0, -92.0, -92.0, -92.0],
    })


def _neris():
    return pd.DataFrame([(38.001, -92.0, pd.Timestamp("2026-01-10"), 1, 1, 1, "", "A")],
                        columns=["lat", "lon", "call_date", "n_records", "unit_response", "duration", "mutual_aid", "dept"])


def test_only_detections_near_a_dispatch_in_space_and_time_overlap():
    out, summary = overlap.find_overlap(_history(), _neris(), radius_km=2, day_window=1)
    assert out["external_id"].tolist() == ["a"]            # b is too far, c is 2 days away, d is outside NERIS dates
    assert {"neris_distance_km", "neris_dt_days"} <= set(out.columns)
    assert summary["overlapping"] == 1
    assert summary["history_rows_inside_neris_dates"] == 2  # NERIS here spans one day (Jan 10); only a and b fall inside it
    assert summary["by_source"] == {"viirs": 1} and summary["by_month"] == {"2026-01": 1}


def test_wider_window_and_radius_find_more():
    out, _ = overlap.find_overlap(_history(), _neris(), radius_km=100, day_window=2)
    assert out["external_id"].tolist() == ["a", "b", "c"]
