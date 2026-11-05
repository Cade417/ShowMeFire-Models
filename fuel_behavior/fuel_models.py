"""
Scott & Burgan (2005, RMRS-GTR-153) standard 40 fire behavior fuel models -
the reference table this project has never had. static_features/build_bundle.py
does NOT preserve the raw LANDFIRE FBFM40 code: it remaps whatever raw codes
appear in a given source raster tile to a sorted 1..N index, persisted only
in that bundle's own manifest (category_mappings). There is no lookup file
anywhere in this repo tying an encoded index - or even the raw LANDFIRE code -
back to a standard model like "GR1" or "TL8".

This table is keyed by the STANDARD Scott & Burgan code, not by anything
this project's pipeline currently produces. Using it against a real bundle
requires one more step this module does not do: read that bundle's
manifest to recover the raw LANDFIRE code for each encoded index, then map
raw LANDFIRE FBFM40 codes to these standard codes (LANDFIRE's own raw
codes are a numeric encoding of these same standard letter/number codes -
see LANDFIRE's FBFM40 product documentation). That mapping is deferred
until a real static bundle exists (training-data/static/bundles/ is
currently empty - nothing has ever been downloaded or built).

spread_base/intensity_base are a deliberately coarse 1-4 ordinal
simplification (low/moderate/high/very_high) of each model's published
fire behavior characteristics, not the precise Rothermel-derived fuel
load/SAV/bed-depth parameters Scott & Burgan actually tabulate. That
precision is not warranted for a first-pass RELATIVE index prototype with
no real Missouri data yet to calibrate against - see spread_index.py's
module docstring. Revisit with the full numeric parameter table if this
graduates beyond a relative index.
"""
from __future__ import annotations

from typing import Dict, NamedTuple, Optional


class FuelModel(NamedTuple):
    code: str
    group: str
    description: str
    spread_base: float  # 0.0 (non-burnable) or 1.0-4.0 (low..very_high)
    intensity_base: float  # 0.0 (non-burnable) or 1.0-4.0 (low..very_high)


# group meanings: NB=non-burnable, GR=grass, GS=grass-shrub, SH=shrub,
# TU=timber-understory, TL=timber-litter, SB=slash-blowdown.
FUEL_MODEL_TABLE: Dict[str, FuelModel] = {
    model.code: model
    for model in (
        FuelModel("NB1", "NB", "Urban/developed", 0.0, 0.0),
        FuelModel("NB2", "NB", "Snow/ice", 0.0, 0.0),
        FuelModel("NB3", "NB", "Agricultural", 0.0, 0.0),
        FuelModel("NB8", "NB", "Open water", 0.0, 0.0),
        FuelModel("NB9", "NB", "Bare ground/rock", 0.0, 0.0),

        FuelModel("GR1", "GR", "Short, sparse dry climate grass", 3.0, 1.0),
        FuelModel("GR2", "GR", "Low load, dry climate grass", 3.0, 2.0),
        FuelModel("GR3", "GR", "Low load, very coarse humid climate grass", 3.0, 2.0),
        FuelModel("GR4", "GR", "Moderate load, dry climate grass", 4.0, 2.0),
        FuelModel("GR5", "GR", "Low load, humid climate grass", 3.0, 2.0),
        FuelModel("GR6", "GR", "Moderate load, humid climate grass", 4.0, 3.0),
        FuelModel("GR7", "GR", "High load, dry climate grass", 4.0, 3.0),
        FuelModel("GR8", "GR", "High load, very coarse humid climate grass", 4.0, 4.0),
        FuelModel("GR9", "GR", "Very high load, humid climate grass", 4.0, 4.0),

        FuelModel("GS1", "GS", "Low load, dry climate grass-shrub", 2.0, 2.0),
        FuelModel("GS2", "GS", "Moderate load, dry climate grass-shrub", 3.0, 2.0),
        FuelModel("GS3", "GS", "Moderate load, humid climate grass-shrub", 3.0, 3.0),
        FuelModel("GS4", "GS", "High load, humid climate grass-shrub", 3.0, 4.0),

        FuelModel("SH1", "SH", "Low load, dry climate shrub", 1.0, 1.0),
        FuelModel("SH2", "SH", "Moderate load, dry climate shrub", 2.0, 2.0),
        FuelModel("SH3", "SH", "Moderate load, humid climate shrub", 2.0, 2.0),
        FuelModel("SH4", "SH", "Low load, humid climate shrub", 2.0, 3.0),
        FuelModel("SH5", "SH", "High load, dry climate shrub", 4.0, 4.0),
        FuelModel("SH6", "SH", "Low load, humid climate shrub", 3.0, 3.0),
        FuelModel("SH7", "SH", "Very high load, dry climate shrub", 3.0, 4.0),
        FuelModel("SH8", "SH", "High load, humid climate shrub", 3.0, 3.0),
        FuelModel("SH9", "SH", "Very high load, humid climate shrub", 3.0, 4.0),

        FuelModel("TU1", "TU", "Light load, dry climate timber-grass-shrub", 1.0, 1.0),
        FuelModel("TU2", "TU", "Moderate load, humid climate timber-shrub", 1.0, 2.0),
        FuelModel("TU3", "TU", "Moderate load, humid climate timber-grass-shrub", 2.0, 2.0),
        FuelModel("TU4", "TU", "Dwarf conifer with understory", 2.0, 3.0),
        FuelModel("TU5", "TU", "Very high load, dry climate timber-shrub", 2.0, 3.0),

        FuelModel("TL1", "TL", "Low load, compact conifer litter", 1.0, 1.0),
        FuelModel("TL2", "TL", "Low load broadleaf litter", 1.0, 1.0),
        FuelModel("TL3", "TL", "Moderate load conifer litter", 1.0, 1.0),
        FuelModel("TL4", "TL", "Small downed logs", 1.0, 2.0),
        FuelModel("TL5", "TL", "High load conifer litter", 1.0, 2.0),
        FuelModel("TL6", "TL", "High load broadleaf litter", 2.0, 2.0),
        FuelModel("TL7", "TL", "Large downed logs", 1.0, 2.0),
        FuelModel("TL8", "TL", "Long-needle litter", 2.0, 2.0),
        FuelModel("TL9", "TL", "Very high load broadleaf litter", 2.0, 3.0),

        FuelModel("SB1", "SB", "Low load activity fuel", 1.0, 2.0),
        FuelModel("SB2", "SB", "Moderate load activity/blowdown", 2.0, 3.0),
        FuelModel("SB3", "SB", "High load activity/blowdown", 2.0, 4.0),
        FuelModel("SB4", "SB", "High load blowdown", 3.0, 4.0),
    )
}


def lookup(fuel_model_code: str) -> Optional[FuelModel]:
    """Case-insensitive lookup; None for an unrecognized code (caller decides fallback, never silently defaults)."""
    return FUEL_MODEL_TABLE.get(fuel_model_code.strip().upper())
