# ShowMeFire Model Lab

Local Streamlit toolkit to **train, test, rank, compare, and preview beta forecasts** for both Fuel Moisture and Fire Danger.

## Setup

```powershell
cd M:\_Development\ShowMeFire\model-training
.venv\Scripts\Activate.ps1
$env:SMF_DATA_ROOT = "M:\_Development\ShowMeFire\training-data"
# Beta graphics use the deployed RAWS API by default:
# $env:SMF_API_BASE_URL = "http://localhost:8000"  # optional local override
python -m pip install -r requirements-lab.txt
# Station-sequence workshop also needs the spatial/torch stack:
# python -m pip install -r requirements-spatial.txt
# python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
```

## Run

```powershell
streamlit run model_lab/app.py
# or: .\model_lab\run.ps1
```

## Dashboard (start here)

The dashboard uses a small offline workflow. Select the product at the top, then
use the tabs in this order:

- **Fuel Moisture** ranks models by MAE and supports legacy XGBoost plus station sequence.
- **Fire Danger** ranks models by Macro F1 and class-support gates using the standalone fire-danger-model pipeline.

The local ladder is:

`Candidate → Challenger → Local Beta → Local Stable → Archived`

The ladder is local to `SMF_DATA_ROOT`. Its buttons never change the live API.

1. **Data** — download/unpack archived observations and click **Build FD dataset**
   to create and validate a separate Fire Danger training CSV.
2. **Train** — create a candidate without changing production.
3. **Test & compare** — review the locked holdout result against the control,
   then optionally compare generated forecast archives.
4. **Experiments** — inspect every saved run, gate, metric, and artifact path.

Forecast graphics, scheduling, registry browsing, and live API/CDN actions remain
available in the underlying modules but are intentionally not part of the primary
workflow.

The FD dataset builder reports station/date coverage, missingness, duplicates, and
support for all five danger classes. The staged FD trainer uses Low, Moderate,
and Elevated+ while preserving Critical/Extreme as rule-based severity detail.
Its test report still requires support for all three staged classes before a run
can pass, preventing a misleading score from a Low/Moderate-only test set.

## Training paths

| Action | What happens |
| --- | --- |
| **Train station sequence** | Trains a candidate on `aligned/station_leads.csv`, then (optionally) scores it on the locked holdout vs an XGB **incumbent control**. |
| **Train legacy XGB** | Fast point-model loop on `final_training_data.csv`; compares to registry stable/beta if present. |
| **Train fire-danger XGB** | Prepares `final_training_data.csv`, trains the standalone FD model, and evaluates Macro F1, class support, and baseline gates. |
| **Evaluate checkpoint** | Re-score any `models/experiments/*.pt` vs control. |
| **Experiment shelf** | All workshop runs with product-specific metrics and pass/fail. |

Gate rules for station sequence: MAE ≥5% better than control, bias not worse, critical MAE not worse, interval coverage, quantile order.

## Daily and beta forecasts

Daily results score `production`, `beta`, and tagged `beta_<tag>` archive series against observations. Fuel Moisture shows MAE/RMSE/bias; Fire Danger shows Macro F1 and accuracy.

The beta graphics action runs the existing ModelFD forecast with an injected FM or FD artifact and writes:

- `archive/forecasts/station_forecasts_beta_<tag>_YYYYMMDD_HH.json`
- `model_lab/beta_maps/<tag>/*.png`

CDN upload is forced off for workshop runs.

## Advanced tools

 | Tab | Role |
| --- | --- |
| Compare archives | Score production vs beta **forecast files** vs obs |
| Data sync | Pull CDN `data-archive/YYYYMMDD.zip` and unpack |
| Schedule & runtime | Production schedule reference + local cron sync/score |
| Registry / Eval reports / Load model | Browse artifacts and offline reports |

Artifacts land under `$SMF_DATA_ROOT/models/experiments/` and `$SMF_DATA_ROOT/reports/model_lab/workshop/`. The locked `final_station_evaluation.json` is never overwritten by the workshop.
