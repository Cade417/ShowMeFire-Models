"""
One-time (re-run only if Missouri county boundaries officially change,
which is rare) export of the api repo's county shapefile into a portable
WGS84 GeoJSON vendored into THIS repo.

This is a build-time/dev-workspace tool, not a runtime dependency: the
training repo must never read api/'s filesystem at runtime (the two are
deployed separately - see paths.py's DB_PATH comment for the same
principle applied to the database). Run this once from a workspace
checkout that has both repos side by side, commit the output, and
risk_fusion/county_geometry.py loads only the vendored file afterward.

Usage:
    python scripts/export_county_geometry.py
    python scripts/export_county_geometry.py --api-shapefile ../api/maps/shapefiles/MO_County_Boundaries/MO_County_Boundaries.shp
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_API_SHAPEFILE = REPO_ROOT.parent / "api" / "maps" / "shapefiles" / "MO_County_Boundaries" / "MO_County_Boundaries.shp"
OUTPUT_DIR = REPO_ROOT / "risk_fusion"
OUTPUT_GEOJSON = OUTPUT_DIR / "county_boundaries.geojson"
OUTPUT_MANIFEST = OUTPUT_DIR / "county_boundaries_manifest.json"


def export(api_shapefile: Path) -> None:
    import shapefile
    from pyproj import Transformer
    from shapely.geometry import shape as shapely_shape, mapping
    from shapely.ops import transform as shapely_transform

    if not api_shapefile.exists():
        raise SystemExit(f"api shapefile not found: {api_shapefile}")

    # The source .prj is EPSG:3857 (Web Mercator, meters) - see
    # api/services/county_lookup.py's docstring for the same finding.
    to_wgs84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True).transform

    reader = shapefile.Reader(str(api_shapefile))
    fields = [field[0] for field in reader.fields[1:]]
    features = []
    for record, shp in zip(reader.records(), reader.shapes()):
        row = dict(zip(fields, record))
        fips = f"29{str(row.get('COUNTYFIPS') or '').zfill(3)}"
        name = str(row.get('COUNTYNAME') or '').strip()
        geom_3857 = shapely_shape(shp.__geo_interface__)
        geom_4326 = shapely_transform(to_wgs84, geom_3857)
        features.append({
            "type": "Feature",
            "properties": {"fips": fips, "name": name},
            "geometry": mapping(geom_4326),
        })

    geojson = {"type": "FeatureCollection", "features": features}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_GEOJSON.write_text(json.dumps(geojson), encoding="utf-8")

    digest = hashlib.sha256(OUTPUT_GEOJSON.read_bytes()).hexdigest()
    manifest = {
        "schema": "county-boundaries-v1",
        "crs": "EPSG:4326",
        "source": str(api_shapefile),
        "source_crs": "EPSG:3857",
        "county_count": len(features),
        "sha256": digest,
    }
    OUTPUT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT_GEOJSON} ({len(features)} counties, sha256={digest[:12]}...)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-shapefile", type=Path, default=DEFAULT_API_SHAPEFILE)
    args = parser.parse_args()
    export(args.api_shapefile)


if __name__ == "__main__":
    main()
