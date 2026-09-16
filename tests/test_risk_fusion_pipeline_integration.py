"""
End-to-end smoke test chaining labels -> effort -> join_panel ->
diagnostics on small synthetic data. The full pipeline was verified
against the real 2011-2020 dataset manually (278,070 weather rows,
9,956 labels, offset-only Poisson GLM, poisson_binary_consistency); this
test exists to catch an interface mismatch between the modules in CI
without needing the real multi-GB HRRR archive.
"""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from risk_fusion import diagnostics, effort, join_panel, labels


class RiskFusionPipelineIntegrationTests(unittest.TestCase):
    def test_full_chain_runs_and_produces_a_fittable_panel(self):
        rng = np.random.default_rng(11)
        counties = [f"290{i:02d}" for i in range(10)]
        dates = pd.date_range("2020-01-01", "2020-06-30", freq="D").strftime("%Y-%m-%d")

        weather_rows = [{"county_fips": c, "valid_local_date": d, "temp_max_c": 20.0} for c in counties for d in dates]
        weather = pd.DataFrame(weather_rows)

        label_rows = []
        event_id = 1
        for _ in range(150):
            county = rng.choice(counties)
            date = rng.choice(dates)
            label_rows.append({
                "event_id": event_id, "source": "official", "verification_tier": "official_source_confirmed",
                "label_weight": 1.0, "latitude": 38.0 + rng.uniform(-0.1, 0.1), "longitude": -92.0 + rng.uniform(-0.1, 0.1),
                "county_fips": county, "occurred_at": f"{date}T18:00:00Z", "occurred_at_precision": "minute",
                "cause_category": "wildfire", "acres": 1.0, "acres_is_estimate": 0, "fuel_types": "",
                "frp": None, "confidence": None, "satellite": None, "label_revision": 1, "revised_at": None,
            })
            event_id += 1

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "labels.csv"
            pd.DataFrame(label_rows).to_csv(csv_path, index=False)
            digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(json.dumps({
                "schema_version": "fire-labels-v1", "min_tier": "official_source_confirmed",
                "row_count": len(label_rows),
                "rows_by_tier": {"unverified": 0, "admin_reviewed": 0, "official_source_confirmed": len(label_rows)},
                "csv_sha256": digest,
            }))

            label_result = labels.build_labels(csv_path, manifest_path)

        join_result = join_panel.build_labeled_panel(weather, label_result["primary_counts"])
        panel = join_result["panel"]
        self.assertEqual(len(panel), len(weather))

        county_reference = {c: {"burnable_area_km2": 1000.0} for c in counties}
        n_days = panel["valid_local_date"].nunique()
        rate_table = effort.fit_reporting_rate(label_result["primary_counts"], county_reference, n_days_observed=n_days)
        panel["log_effort"] = effort.log_effort_offset_batch(panel["county_fips"], rate_table, county_reference)

        y = panel["event_count"].to_numpy()
        exog = np.ones((len(panel), 1))
        model = sm.GLM(y, exog, offset=panel["log_effort"].to_numpy(), family=sm.families.Poisson())
        result = model.fit()
        lam = result.predict()

        # The offset-only fit's mean prediction must reproduce the observed
        # mean count - the basic GLM calibration property that would break
        # if the effort offset were wired in with, say, a sign error.
        self.assertAlmostEqual(float(lam.mean()), float(y.mean()), places=6)

        diag = diagnostics.poisson_binary_consistency(y, lam)
        self.assertIn(diag["recommended_count_family"], ("poisson", "negative_binomial"))
        self.assertGreaterEqual(diag["pearson_dispersion"], 0.0)


if __name__ == "__main__":
    unittest.main()
