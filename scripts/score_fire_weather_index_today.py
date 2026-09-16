#!/usr/bin/env python3
"""
Ad-hoc local runner: scores fire_weather_index for a given HRRR cycle
(default: today's 12z) using whatever's already cached, printing a
per-county score/category table to the terminal. Not part of the shadow
pipeline (that only runs inside a live api/ forecast run) - this is for
checking the model locally without standing up the API.

By default this also renders a Missouri map PNG and prints its path - pass
--no-map to skip that and only print the table. The 'pixel' style (default)
deliberately matches the production Peak Fire Danger Forecast map's visual
identity exactly (see forecast/DailyForecast.py's create_base_map/
add_boundaries/add_title_and_branding in the api/ repo, which this mirrors):
same 5-category color palette, same title/subtitle/description placement
and fonts (Montserrat / Plus Jakarta Sans), same ShowMeFire logo and
"ShowMeFire.org" wordmark, same county/state boundary styling, same
vertical colorbar-style legend. Only the title text and the description's
criteria list differ, since this is a different model (a continuous
weighted score, not the if/then rule) - everything else is the same
visual language on purpose, so this reads as "the same family of product,"
not a placeholder.

Needs api/ checked out alongside this repo (M:/_Development/ShowMeFire/api)
for the county/state boundary shapefiles, fonts, and logo - model-training
has no copies of its own, since this rendering is a local dev convenience,
not part of the shipped pipeline. Logo rasterization uses resvg_py (a
prebuilt-wheel, no-native-deps SVG renderer) rather than the production
code's cairosvg - cairosvg needs a system libcairo that isn't set up on
this machine (confirmed: it isn't set up in api/'s own venv either).

Usage:
    python scripts/score_fire_weather_index_today.py
    python scripts/score_fire_weather_index_today.py --date 2026-09-14 --cycle 12
    python scripts/score_fire_weather_index_today.py --fetch    # fetch the HRRR run first if not cached
    python scripts/score_fire_weather_index_today.py --no-map   # table only, skip rendering
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parent.parent
API_ROOT = REPO_ROOT.parent / "api"
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

import paths
from fire_weather_index import build_county_days, factors, grid_score, model_bundle

CATEGORY_LABELS_ORDER = ("Low", "Moderate", "Elevated", "Critical", "Extreme")
NODATA_COLOR = "#CCCCCC"

# --- Exact production visual identity - see forecast/DailyForecast.py's
# "MAP 1: PEAK FIRE DANGER" section (api/ repo) for the source of every
# constant below. Colors/bins/labels copied verbatim. ---
PRODUCTION_COLORS = ["#90EE90", "#FFED4E", "#FFA500", "#FF0000", "#8B0000"]
PRODUCTION_LABELS = ["Low", "Moderate", "Elevated \nHigh", "Critical \n Very High", "Extreme"]
PRODUCTION_BINS = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
FIGURE_FACECOLOR = "#E8E8E8"
COUNTY_EDGE_COLOR = "#B6B6B6"
STATE_EDGE_COLOR = "#000000"


def _register_production_fonts():
    import matplotlib.font_manager as font_manager
    import matplotlib.pyplot as plt

    for font_path in (
        API_ROOT / "assets/Montserrat/static/Montserrat-Regular.ttf",
        API_ROOT / "assets/Plus_Jakarta_Sans/static/PlusJakartaSans-Regular.ttf",
        API_ROOT / "assets/Plus_Jakarta_Sans/static/PlusJakartaSans-Bold.ttf",
    ):
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
    plt.rcParams["font.family"] = "Montserrat"


def _add_logo(fig):
    """Same placement/size as production's add_title_and_branding (bottom-
    right, figure-fraction (0.99, 0.01), box_alignment=(1,0), zoom=0.03) -
    rasterized via resvg_py instead of cairosvg (see module docstring)."""
    import resvg_py
    import matplotlib.image as mpimg
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    from io import BytesIO

    svg_path = API_ROOT / "assets" / "LightBackGroundLogo.svg"
    if not svg_path.exists():
        return
    png_bytes = bytes(resvg_py.svg_to_bytes(svg_path=str(svg_path), zoom=2.0))
    image = mpimg.imread(BytesIO(png_bytes), format="png")
    imagebox = OffsetImage(image, zoom=0.03)
    ab = AnnotationBbox(imagebox, (0.99, 0.01), frameon=False, xycoords="figure fraction", box_alignment=(1, 0))
    fig.gca().add_artist(ab)


def add_title_and_branding(fig, title: str, subtitle: str, description: str):
    """Same fig.text calls (size/font/alignment) as
    forecast/DailyForecast.py::add_title_and_branding, plus the logo - but
    with vertical positions shifted down from production's when the title
    wraps to two lines (this model's name makes the parenthetical too long
    for one line at the same width production's shorter title used;
    production's own title is always one line, so it never needed this)."""
    _register_production_fonts()
    title_lines = title.count("\n") + 1
    subtitle_y = 0.97 - 0.075 * title_lines
    description_y = subtitle_y - 0.28
    fig.text(0.99, 0.97, title, fontsize=24, fontweight="bold", ha="right", va="top", fontname="Plus Jakarta Sans")
    fig.text(0.99, subtitle_y, subtitle, fontsize=16, ha="right", va="top", fontname="Montserrat")
    fig.text(0.99, description_y, description, fontsize=10, ha="right", va="top", linespacing=1.6, fontname="Montserrat")
    fig.text(0.02, 0.01, "ShowMeFire.org", fontsize=20, fontweight="bold", ha="left", va="bottom", fontname="Montserrat")
    _add_logo(fig)


def create_base_map(extent, map_crs, data_crs, pixel_width=2048, pixel_height=1152, dpi=144):
    """Exact same figure/axes construction as production's create_base_map: full-bleed axes, same facecolor."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(pixel_width / dpi, pixel_height / dpi), dpi=dpi, facecolor=FIGURE_FACECOLOR)
    ax = plt.axes([0, 0, 1, 1], projection=map_crs)
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_extent(extent, crs=data_crs)
    return fig, ax


def add_boundaries(ax, data_crs):
    """Exact same colors/widths/zorders as production's add_boundaries."""
    counties = _county_geometries()
    ax.add_geometries(counties.geometry, crs=data_crs, edgecolor=COUNTY_EDGE_COLOR, facecolor="none", linewidth=1, zorder=5)
    state = _state_geometries()
    ax.add_geometries(state.geometry, crs=data_crs, edgecolor=STATE_EDGE_COLOR, facecolor="none", linewidth=1.5, zorder=9)
    return state


def _county_geometries():
    """Missouri county boundaries in EPSG:4326, keyed by 5-digit FIPS - same
    join technique as api/services/burn_ban_map.py. Sourced from the api/
    repo checked out alongside this one - model-training has no shapefiles
    of its own."""
    import geopandas as gpd

    shp_path = API_ROOT / "maps" / "shapefiles" / "MO_County_Boundaries" / "MO_County_Boundaries.shp"
    if not shp_path.exists():
        raise FileNotFoundError(
            f"County boundary shapefile not found at {shp_path} - this rendering needs the api/ repo "
            "checked out alongside model-training/ (M:\\_Development\\ShowMeFire\\api). Use --no-map to skip."
        )
    counties = gpd.read_file(shp_path).to_crs("EPSG:4326")
    counties["fips"] = counties["COUNTYFIPS"].astype(str).str.zfill(3).radd("29")
    return counties


def _state_geometries():
    import geopandas as gpd

    shp_path = API_ROOT / "maps" / "shapefiles" / "MO_State_Boundary" / "MO_State_Boundary.shp"
    if not shp_path.exists():
        raise FileNotFoundError(f"State boundary shapefile not found at {shp_path}. Use --no-map to skip.")
    return gpd.read_file(shp_path).to_crs("EPSG:4326")


def _find_matching_cached_file(source: str, run_dt: datetime, max_delta_hours: float = 12.0) -> Path | None:
    """Finds `source`'s ("rrfs" or "fv3hires") own cached file whose parsed
    run time is CLOSEST to run_dt - not an exact-match-or-fail lookup,
    since RRFS only publishes 4x/day and FV3-HIRES capture cycles don't
    necessarily line up with HRRR's. Returns None (not an error, callers
    degrade to HRRR-only) if that source has nothing cached at all, OR if
    the closest file found is more than max_delta_hours away - without
    this cutoff, a request for a date this local cache has no real
    coverage for (this cache currently only spans a few days - RRFS/
    FV3-HIRES capture only started 2026-09-13/14) would silently blend in
    a completely unrelated day's weather instead of falling back, which
    is worse than not blending at all. Reuses fire_weather_index.
    build_county_days.SOURCES - the same cache dir/glob/filename-parsing
    already built for this exact lookup."""
    config = build_county_days.SOURCES[source]
    candidates = sorted(config["cache_dir"].glob(config["glob"]))
    best_path, best_delta = None, None
    for path in candidates:
        candidate_time = build_county_days._run_time_from_filename(path, config["filename_re"])
        if candidate_time is None:
            continue
        delta = abs((candidate_time.tz_localize(None) - pd.Timestamp(run_dt)).total_seconds())
        if best_delta is None or delta < best_delta:
            best_path, best_delta = path, delta
    if best_path is not None and best_delta > max_delta_hours * 3600:
        return None
    return best_path


def render_pixel_map(hrrr_path: Path, bundle: Dict, run_dt: datetime, out_path: Path) -> Path:
    """Smooth, per-grid-cell render matching the production Peak Fire
    Danger Forecast map's exact visual identity - computes the score
    directly on the HRRR grid via grid_score.py, BLENDED with whatever
    RRFS/FV3-HIRES cached files are closest in time (regridded onto HRRR's
    own grid - see grid_score.compute_blended_factor_grids for why that's
    needed instead of a direct pixel-to-pixel average), rescales the
    result onto production's 0-4 category-index bins, and renders with
    production's exact colors/fonts/boundaries/logo."""
    import cartopy.crs as ccrs
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from cartopy.mpl.patch import geos_to_path
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MplPath

    rrfs_path = _find_matching_cached_file("rrfs", run_dt)
    fv3hires_path = _find_matching_cached_file("fv3hires", run_dt)
    grids = grid_score.compute_blended_factor_grids(hrrr_path, rrfs_path=rrfs_path, fv3hires_path=fv3hires_path)

    # Real antecedent rain (prior cached HRRR days), same fix as the
    # county-day path - a place that got soaked yesterday with nothing
    # more forecast should read wetter than "zero rain", not bone-dry.
    antecedent = grid_score.compute_antecedent_precip_grid(hrrr_path)
    grids["precip_24h_mm"] = grids["precip_24h_mm"] + antecedent["antecedent_precip_mm"]
    antecedent_days = antecedent["antecedent_precip_days"]

    # Fuel-type-aware precip relief threshold, regridded straight onto
    # HRRR's own grid (None if the static bundle isn't available in this
    # environment - compute_score_grid then falls back to the flat default).
    fuel_threshold = grid_score.compute_fuel_threshold_grid(grids["lat"], grids["lon"])
    if fuel_threshold is not None:
        # Nearest-neighbor regridding of a fine-grained categorical fuel
        # raster onto HRRR's grid is checkerboard-noisy pixel to pixel
        # (confirmed live: real Missouri land cover alternates fields/
        # timber patches at a finer scale than either grid resolves
        # cleanly - values genuinely flip 5/10/18 from one cell to the
        # next across the whole domain, not just at one spot). sigma=3
        # was nowhere near enough to blend that out - fuel-type variation
        # is a regional characteristic (this project only classifies 3
        # coarse groups to begin with), not something meant to carry
        # native-pixel-resolution precision, so a much larger sigma here
        # is the more honest choice, not just a cosmetic smoothing hack.
        fuel_threshold = grid_score.smooth_score_grid(fuel_threshold, sigma=10.0)

    score = grid_score.compute_score_grid(grids, precip_relief_threshold=fuel_threshold)
    score = grid_score.smooth_score_grid(score, sigma=2.5)
    thresholds = bundle["category_thresholds"]["thresholds"]

    # Rescale the continuous [0,1] score onto a continuous 0-4 "category
    # index" via the calibrated thresholds as anchor points - this is what
    # lets us reuse production's exact bins=[-0.5..4.5]/BoundaryNorm/
    # contourf mechanism (built for a 0-4 category field) instead of
    # inventing a different-looking one for this model. Anchors: score=0 ->
    # index 0 (deep inside the Low bin), each threshold -> the exact
    # half-integer boundary between adjacent bins, score=1 -> index 4
    # (deep inside the Extreme bin).
    #
    # Since factors.RAW_SCORE_CEILING anchors the score to a real historic
    # event, the calibrated Extreme cutpoint (95th percentile) can legitimately
    # land AT 1.0 itself (score is clipped there, so enough days tie the
    # ceiling to make it the 95th percentile too - seen in practice: ~6% of
    # the production panel). A naive [..., thresholds[-1], 1.0] anchor list
    # would then have a zero-width final segment, collapsing the entire
    # Extreme band to a single unreachable point once gaussian smoothing
    # nudges every cell fractionally below 1.0. Guard against that: only add
    # a distinct 1.0 anchor when the last threshold is actually below it -
    # otherwise the final (previous threshold -> 1.0) segment itself IS the
    # approach-to-Extreme gradient, so its far end maps to a solidly-Extreme
    # index (4.0), not the Critical/Extreme boundary (3.5).
    anchor_scores = [0.0] + list(thresholds)
    anchor_indices = [0.0, 0.5, 1.5, 2.5, 3.5]
    if anchor_scores[-1] < 1.0:
        anchor_scores.append(1.0)
        anchor_indices.append(4.0)
    else:
        anchor_indices[-1] = 4.0
    category_index = np.interp(score, anchor_scores, anchor_indices)
    category_index = np.where(np.isnan(score), np.nan, category_index)

    lon = grids["lon"]
    lon = np.where(lon > 180, lon - 360, lon)
    lat = grids["lat"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    data_crs = ccrs.PlateCarree()
    map_crs = ccrs.LambertConformal(central_longitude=-92.45, central_latitude=38.3)
    extent = (-95.8, -89.1, 35.8, 40.8)
    fig, ax = create_base_map(extent, map_crs, data_crs)

    cmap = ListedColormap(PRODUCTION_COLORS)
    norm = BoundaryNorm(PRODUCTION_BINS, len(PRODUCTION_COLORS))
    contours = ax.contourf(lon, lat, category_index, transform=data_crs, levels=PRODUCTION_BINS,
                            cmap=cmap, norm=norm, alpha=0.7, zorder=7, antialiased=True)
    separators = ax.contour(lon, lat, category_index, transform=data_crs, levels=PRODUCTION_BINS[1:-1],
                             colors="black", linewidths=0.3, alpha=0.2, zorder=8)

    # Vector clip to Missouri's exact boundary (not a raster NaN mask -
    # that produced a visibly jagged/notched edge at HRRR's grid
    # resolution, confirmed live; a clip path is geometrically exact
    # regardless of grid resolution). BOTH artists need it - the fill and
    # the black category-boundary separator lines are two separate
    # matplotlib artists, and clipping only the fill left the separator
    # lines still drawn out to the full buffered HRRR domain, visibly
    # crossing into neighboring states past Missouri's own border
    # (confirmed live - that's the "colors extend beyond the map border"
    # bug this fixes).
    try:
        state = add_boundaries(ax, data_crs)
        state_geom = state.geometry.union_all()
        projected_geom = ax.projection.project_geometry(state_geom, data_crs)
        clip_path = PathPatch(MplPath.make_compound_path(*geos_to_path(projected_geom)), transform=ax.transData)
        for artist_set in (contours, separators):
            for collection in (getattr(artist_set, "collections", None) or [artist_set]):
                collection.set_clip_path(clip_path)
    except FileNotFoundError:
        pass  # boundary overlay/clip is cosmetic only - the pixel data itself doesn't need it

    ax.set_anchor("W")
    plt.subplots_adjust(left=0.05)

    # Same vertical colorbar-as-legend as production, same position.
    cax = fig.add_axes([0.02, 0.08, 0.02, 0.6])
    cbar = plt.colorbar(contours, cax=cax, label="Fire Danger Level")
    cbar.set_ticks([0, 1, 2, 3, 4])
    cbar.set_ticklabels(PRODUCTION_LABELS)

    rrfs_pct = 100.0 * grids.get("rrfs_coverage_fraction", 0.0)
    fv3hires_pct = 100.0 * grids.get("fv3hires_coverage_fraction", 0.0)
    add_title_and_branding(
        fig, "Missouri Peak Fire Danger Forecast\n(Fire Weather Severity Index)",
        f"Model Run: {run_dt.strftime('%Y-%m-%d %HZ')} | Valid: {run_dt.strftime('%Y-%m-%d')}",
        "LOCAL DIAGNOSTIC RENDER - NOT FOR OPERATIONS\n\n"
        "Fire Weather Severity Index: a continuous, weighted fire-weather danger\n"
        "score (RH, wind, VPD, precip relief), mapped to these 5 categories\n"
        "via calibrated score percentiles - not the same computation as\n"
        "the public if/then rule-based Fire Danger category.\n\n"
        f"Data Source: HRRR (backbone, every pixel) blended with RRFS\n"
        f"({rrfs_pct:.0f}% of pixels) and FV3-HIRES ({fv3hires_pct:.0f}% of pixels),\n"
        f"each regridded onto HRRR's own grid. Precip includes\n"
        f"{antecedent_days}/3 days of real antecedent rain, fuel-type-\n"
        "aware relief threshold | ShowMeFire fire_weather_index\n"
        "For More Info, Visit ShowMeFire.org",
    )

    _save_atomic(fig, out_path)
    return out_path


def render_county_map(out_df: pd.DataFrame, run_dt: datetime, out_path: Path) -> Path:
    """Discrete county choropleth (the alternate --style), same production visual identity."""
    import cartopy.crs as ccrs
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    label_to_id = {label: i for i, label in enumerate(CATEGORY_LABELS_ORDER)}
    category_by_fips = {row.county_fips: label_to_id.get(row.category) for row in out_df.itertuples()}

    data_crs = ccrs.PlateCarree()
    map_crs = ccrs.LambertConformal(central_longitude=-92.45, central_latitude=38.3)
    extent = (-95.8, -89.1, 35.8, 40.8)
    fig, ax = create_base_map(extent, map_crs, data_crs)

    counties = _county_geometries()
    counties["category"] = counties["fips"].map(category_by_fips)
    for category_id, color in enumerate(PRODUCTION_COLORS):
        subset = counties[counties["category"] == category_id]
        if not subset.empty:
            ax.add_geometries(subset.geometry, crs=data_crs, facecolor=color, edgecolor="none", alpha=0.7, zorder=6)
    no_data = counties[counties["category"].isna()]
    if not no_data.empty:
        ax.add_geometries(no_data.geometry, crs=data_crs, facecolor=NODATA_COLOR, edgecolor="none", zorder=6)
    add_boundaries(ax, data_crs)

    ax.set_anchor("W")
    plt.subplots_adjust(left=0.05)

    from matplotlib.patches import Patch
    cax = fig.add_axes([0.02, 0.08, 0.02, 0.6])
    cax.set_axis_off()
    legend_handles = [Patch(facecolor=color, label=label, alpha=0.7) for color, label in zip(PRODUCTION_COLORS, PRODUCTION_LABELS)]
    legend_handles.append(Patch(facecolor=NODATA_COLOR, label="No data"))
    cax.legend(handles=legend_handles, loc="center", fontsize=9, frameon=False)

    rrfs_pct = 100.0 * out_df["rrfs_available"].mean() if "rrfs_available" in out_df else 0.0
    fv3hires_pct = 100.0 * out_df["fv3hires_available"].mean() if "fv3hires_available" in out_df else 0.0
    add_title_and_branding(
        fig, "Missouri Peak Fire Danger Forecast\n(Fire Weather Severity Index)",
        f"Model Run: {run_dt.strftime('%Y-%m-%d %HZ')} | Valid: {run_dt.strftime('%Y-%m-%d')}",
        "LOCAL DIAGNOSTIC RENDER - NOT FOR OPERATIONS\n\n"
        "Fire Weather Severity Index: a continuous, weighted fire-weather danger\n"
        "score, aggregated to county-day and mapped to these 5 categories\n"
        "via calibrated score percentiles - not the same computation as\n"
        "the public if/then rule-based Fire Danger category.\n\n"
        f"Data Source: HRRR (backbone, 100% of counties) blended with\n"
        f"RRFS ({rrfs_pct:.0f}% of counties) and FV3-HIRES ({fv3hires_pct:.0f}% of\n"
        "counties) where cached history is available | ShowMeFire\n"
        "fire_weather_index\n"
        "For More Info, Visit ShowMeFire.org",
    )

    _save_atomic(fig, out_path)
    return out_path


def _save_atomic(fig, out_path: Path) -> None:
    import tempfile, os
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".fire_weather_index_local.", suffix=".png", dir=out_path.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        fig.savefig(temp_path, facecolor=fig.get_facecolor())
        plt.close(fig)
        temp_path.replace(out_path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        plt.close("all")


def _default_run_dt(cycle_hour: int = 12, publish_buffer_hours: int = 3) -> datetime:
    """The most recent `cycle_hour`z UTC that has plausibly been published
    already - NOT just "today at cycle_hour", which is wrong (and Herbie
    refuses outright - "date cannot be in the future") whenever the
    script runs before that cycle has actually happened yet today
    (confirmed live: running this before 12z UTC with no --date/--cycle
    override tried to fetch a still-in-the-future run). Falls back to
    yesterday's cycle if today's hasn't had time to publish yet."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    candidate = now.replace(hour=cycle_hour, minute=0, second=0, microsecond=0)
    if candidate + timedelta(hours=publish_buffer_hours) > now:
        candidate -= timedelta(days=1)
    return candidate


def main():
    default_run_dt = _default_run_dt()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=default_run_dt.strftime("%Y-%m-%d"),
                        help="HRRR run date, UTC, YYYY-MM-DD (default: most recent date whose 12z cycle "
                             "has plausibly published already)")
    parser.add_argument("--cycle", type=int, default=default_run_dt.hour,
                        help="HRRR cycle hour, UTC (default: 12, or yesterday's 12z if today's hasn't published yet)")
    parser.add_argument("--fetch", action="store_true", help="Fetch the HRRR run first if not already cached")
    parser.add_argument("--candidate-dir", type=Path, default=paths.FIRE_WEATHER_INDEX_CANDIDATE_DIR,
                        help="Registered bundle directory to score with (default: the local candidate dir "
                             "written by `python -m fire_weather_index.register_beta`)")
    parser.add_argument("--no-map", action="store_true", help="Skip rendering the PNG, print the table only")
    parser.add_argument("--style", choices=("pixel", "county"), default="pixel",
                        help="'pixel' (default): smooth per-grid-cell render matching the production Peak Fire "
                             "Danger map's style. 'county': discrete county choropleth (the original style).")
    parser.add_argument("--map-output", type=Path, default=None,
                        help="Where to write the map PNG (default: <FIRE_WEATHER_INDEX_DIR>/local_render/"
                             "fire_weather_index_<cycle>.png)")
    args = parser.parse_args()

    run_dt = datetime.strptime(f"{args.date} {args.cycle:02d}", "%Y-%m-%d %H")
    hrrr_target = paths.CACHE_HRRR_DIR / f"hrrr_{run_dt:%Y%m%d_%H}z_f04-15.nc"

    if not hrrr_target.exists():
        if not args.fetch:
            raise SystemExit(
                f"No cached HRRR run at {hrrr_target}. Re-run with --fetch to pull it now, "
                f"or fetch it yourself first: python -c \"from spatial.hrrr_capture import fetch_hrrr; "
                f"from datetime import datetime; fetch_hrrr(datetime({run_dt.year},{run_dt.month},{run_dt.day},{run_dt.hour}))\""
            )
        print(f"Fetching HRRR {run_dt:%Y-%m-%d %H}z...")
        from spatial.hrrr_capture import fetch_hrrr
        fetch_hrrr(run_dt)

    if args.fetch:
        # Pull THIS cycle's RRFS/FV3-HIRES directly, on demand - no
        # scheduled/background task required. Best-effort and non-fatal:
        # RRFS only publishes at 0/6/12/18z (spatial.rrfs_capture.
        # NA_PRODUCT_CYCLE_HOURS) and FV3-HIRES may not have posted yet for
        # a very recent cycle - either way, render_pixel_map's blending
        # already degrades gracefully (0% coverage, not an error) when a
        # source isn't available, same as before this fetch existed.
        print(f"Fetching RRFS {run_dt:%Y-%m-%d %H}z...")
        try:
            from spatial.rrfs_capture import fetch_rrfs
            fetch_rrfs(run_dt)
        except Exception as exc:
            print(f"  RRFS fetch skipped: {exc}")

        print(f"Fetching FV3-HIRES {run_dt:%Y-%m-%d %H}z...")
        try:
            from spatial.fv3hires_capture import fetch_fv3hires
            fetch_fv3hires(run_dt)
        except Exception as exc:
            print(f"  FV3-HIRES fetch skipped: {exc}")

    if not args.candidate_dir.exists():
        raise SystemExit(
            f"No registered bundle at {args.candidate_dir}. Run `python -m fire_weather_index.register_beta` first "
            "(needs `python -m fire_weather_index.evaluate` to have produced a passing report)."
        )

    print("Building the county-day panel (HRRR backbone + whatever RRFS/FV3-HIRES history is cached)...")
    panel = build_county_days.build()
    valid_local_date = run_dt.strftime("%Y-%m-%d")  # HRRR's own local-date bucketing may shift this by a day near cycle boundaries
    today = panel[panel["run_id"] == run_dt.strftime("%Y%m%d_%H")].copy()
    if today.empty:
        # run_id filter can miss if this cycle's rows got de-duplicated in favor of a later run for the same day -
        # fall back to date-based lookup, which is what actually matters for "what does today look like".
        today = panel[panel["valid_local_date"] == valid_local_date].copy()
    if today.empty:
        raise SystemExit(f"No county-day rows found for {run_dt:%Y-%m-%d %H}z after building the panel - "
                          "check that the HRRR file actually covers a valid afternoon lead window.")

    bundle = model_bundle.load(args.candidate_dir)
    thresholds = bundle["category_thresholds"]["thresholds"]
    labels = bundle["category_thresholds"]["category_labels"]

    rows = []
    for _, row in today.iterrows():
        row_factors = factors.compute_factors(row.to_dict())
        score = factors.compute_score(row_factors)
        category = model_bundle.score_to_category(score, thresholds) if score is not None else None
        rows.append({
            "county_fips": row["county_fips"], "score": score,
            "category": labels[category] if category is not None else None,
            "rrfs_available": row.get("rrfs_available"), "fv3hires_available": row.get("fv3hires_available"),
        })

    out = pd.DataFrame(rows).sort_values("score", ascending=False, na_position="last")
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(f"\nFire Weather Severity Index for {run_dt:%Y-%m-%d %H}z ({len(out)} counties):")
        print(out.to_string(index=False))
    print(f"\ncategory counts: {out['category'].value_counts(dropna=False).to_dict()}")
    print(f"rrfs_available: {int(out['rrfs_available'].sum())}/{len(out)}  "
          f"fv3hires_available: {int(out['fv3hires_available'].sum())}/{len(out)}")

    if not args.no_map:
        map_output = args.map_output or (paths.FIRE_WEATHER_INDEX_DIR / "local_render" /
                                          f"fire_weather_index_{run_dt:%Y%m%d_%H}z.png")
        try:
            if args.style == "pixel":
                written = render_pixel_map(hrrr_target, bundle, run_dt, map_output)
            else:
                written = render_county_map(out, run_dt, map_output)
            print(f"\nmap written to: {written.resolve()}")
        except Exception as exc:
            print(f"\nmap rendering failed (table above is still valid): {exc}")


if __name__ == "__main__":
    main()
