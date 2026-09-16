"""
Builds a real per-county 1991-2020 mean-annual-precipitation climate normal
for every Missouri county in county_reference.json, closing the gap
documented in build_county_days.py and features.py's KBDI docstring (KBDI
requires a real climate normal; fabricating one would silently corrupt the
one cheap drought proxy this project has).

Source: NOAA NCEI's public "Climate at a Glance" County Time Series data
service (no API key required) - the same tool at
https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/county/time-series,
confirmed by inspecting its own network requests (this is the exact JSON
endpoint that page's chart calls, not a guessed URL):

    https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/county/
    time-series/{state_abbr}-{county_fips3}/pcp/12/12/data.json?raw=1

Returns {year: annual_total_precipitation_inches} for the full period of
record (the begyear/endyear query params do not restrict the response -
confirmed empirically - so this script slices to 1991-2020 itself). The
1991-2020 mean of that series IS the standard NOAA 30-year climate normal
period; this script computes it directly rather than depending on the
service to pre-average.

Every Missouri county FIPS in this repo (see county_reference.json) maps to
NOAA's county code as "MO-{last 3 digits of the 5-digit FIPS}", including
independent cities (verified: MO-510 for St. Louis city resolves).

Usage:
    python -m risk_fusion.build_precip_normals
    python -m risk_fusion.build_precip_normals --limit 5   # smoke test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

COUNTY_REFERENCE_PATH = REPO_ROOT / "risk_fusion" / "county_reference.json"
OUTPUT_PATH = REPO_ROOT / "risk_fusion" / "county_precip_normals.json"

BASE_URL = "https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/county/time-series"
STATE_ABBR = "MO"  # every county in county_reference.json is a Missouri FIPS (29xxx)
NORMAL_PERIOD_START = 1991
NORMAL_PERIOD_END = 2020
INCHES_TO_MM = 25.4
REQUEST_DELAY_SECONDS = 0.4  # be a good citizen against a public government server
MIN_YEARS_REQUIRED = 20  # out of a possible 30 - tolerate a few missing/short years, not silent full gaps


def _county_code(fips: str) -> str:
    """MO county FIPS (29xxx) -> NOAA CAG's 3-digit county code (last 3 digits)."""
    if len(fips) != 5 or not fips.startswith("29"):
        raise ValueError(f"Expected a 5-digit Missouri FIPS starting with '29', got {fips!r}")
    return fips[2:]


def fetch_annual_series(fips: str, timeout: float = 20.0) -> Dict[str, float]:
    """Fetches the full period-of-record annual precipitation series (inches) for one county."""
    url = f"{BASE_URL}/{STATE_ABBR}-{_county_code(fips)}/pcp/12/12/data.json?raw=1"
    request = urllib.request.Request(url, headers={"User-Agent": "showmefire-model-training/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def compute_normal(series: Dict[str, float]) -> Optional[Dict]:
    """Mean of the 1991-2020 subset, converted to mm. None if too few years are present."""
    years_used = {
        year: value for year, value in series.items()
        if NORMAL_PERIOD_START <= int(year) <= NORMAL_PERIOD_END and value is not None
    }
    if len(years_used) < MIN_YEARS_REQUIRED:
        return None
    mean_inches = sum(years_used.values()) / len(years_used)
    return {
        "mean_annual_precip_mm": round(mean_inches * INCHES_TO_MM, 2),
        "years_used": len(years_used),
        "period": f"{NORMAL_PERIOD_START}-{NORMAL_PERIOD_END}",
    }


def build(limit: Optional[int] = None) -> Dict:
    reference = json.loads(COUNTY_REFERENCE_PATH.read_text(encoding="utf-8"))
    counties = reference["counties"][:limit] if limit else reference["counties"]

    normals = {}
    failures = []
    for i, county in enumerate(counties):
        fips = county["fips"]
        try:
            series = fetch_annual_series(fips)
            normal = compute_normal(series)
            if normal is None:
                failures.append({"fips": fips, "name": county.get("name"), "reason": "fewer than "
                                  f"{MIN_YEARS_REQUIRED} years in {NORMAL_PERIOD_START}-{NORMAL_PERIOD_END}"})
            else:
                normals[fips] = {"name": county.get("name"), **normal}
                print(f"[{i+1}/{len(counties)}] {fips} {county.get('name')}: "
                      f"{normal['mean_annual_precip_mm']:.1f} mm ({normal['years_used']} years)")
        except Exception as exc:
            failures.append({"fips": fips, "name": county.get("name"), "reason": str(exc)})
            print(f"[{i+1}/{len(counties)}] {fips} {county.get('name')}: FAILED - {exc}")
        time.sleep(REQUEST_DELAY_SECONDS)

    output = {
        "schema": "county-precip-normals-v1",
        "source": "NOAA NCEI Climate at a Glance County Time Series (nClimGrid), pcp/12/12",
        "source_url_pattern": f"{BASE_URL}/{{state}}-{{county_fips3}}/pcp/12/12/data.json?raw=1",
        "normal_period": f"{NORMAL_PERIOD_START}-{NORMAL_PERIOD_END}",
        "retrieved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "county_count": len(normals),
        "failure_count": len(failures),
        "failures": failures,
        "counties": normals,
    }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, help="Only process the first N counties (smoke test)")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    output = build(limit=args.limit)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nWrote {args.output}: {output['county_count']} counties, {output['failure_count']} failures")
    if output["failures"]:
        print(f"Failures: {output['failures']}")


if __name__ == "__main__":
    main()
