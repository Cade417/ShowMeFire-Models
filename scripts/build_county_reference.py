"""
Build risk_fusion/county_reference.json: the per-county area, burnable-area
(placeholder until NLCD is downloaded - see county_geometry.py), and
spatial-region table consumed by risk_fusion_contract.py's region-stratified
folds and effort.py's log-effort offset.

Usage:
    python scripts/build_county_reference.py
"""
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from risk_fusion.county_geometry import GEOJSON_PATH, REGION_METHOD, build_county_table

OUTPUT_PATH = REPO_ROOT / "risk_fusion" / "county_reference.json"


def main():
    table = build_county_table()
    boundaries_sha256 = hashlib.sha256(GEOJSON_PATH.read_bytes()).hexdigest()

    output = {
        "schema": "county-reference-v1",
        "region_method": REGION_METHOD,
        "county_boundaries_sha256": boundaries_sha256,
        "county_count": len(table),
        "region_counts": {
            str(region): sum(1 for row in table if row["region_id"] == region)
            for region in sorted({row["region_id"] for row in table})
        },
        "counties": table,
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}: {len(table)} counties, regions={output['region_counts']}")


if __name__ == "__main__":
    main()
