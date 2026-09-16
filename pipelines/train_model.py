import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.versioning import register_trained_model
import paths
from spatial.precipitation import PRECIPITATION_CONTRACT_SHA256, PRECIPITATION_CONTRACT_VERSION
from pipelines.experiment_log import append_experiment, compare_to_stable

# Small, deliberately modest grid - this is a rolling-origin search over a
# ~12k-row dataset, not a large-scale tuning job. Centered on the historical
# fixed defaults (n_estimators=200, learning_rate=0.1, max_depth=5) so a
# search that finds nothing better than the default is expected, not a bug.
DEFAULT_PARAM_GRID = {
    "n_estimators": [150, 200, 300],
    "max_depth": [4, 5, 6],
    "learning_rate": [0.05, 0.1],
}


def search_hyperparameters(X_train, y_train, param_grid=None, n_splits=4):
    """Rolling-origin (expanding-window) hyperparameter search.

    Uses TimeSeriesSplit rather than k-fold/shuffled CV because this is a
    time-ordered dataset with rolling-window features (temp_mean_3h etc.) -
    a shuffled split would let folds leak information across adjacent
    timestamps the same way the original 80/20 chronological split was
    chosen to avoid. Returns (best_params, best_mae, all_results).
    """
    param_grid = param_grid or DEFAULT_PARAM_GRID
    splitter = TimeSeriesSplit(n_splits=n_splits)

    results = []
    for params in ParameterGrid(param_grid):
        fold_maes = []
        for train_idx, val_idx in splitter.split(X_train):
            model = xgb.XGBRegressor(objective='reg:squarederror', **params)
            model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            preds = model.predict(X_train.iloc[val_idx])
            fold_maes.append(mean_absolute_error(y_train.iloc[val_idx], preds))
        avg_mae = sum(fold_maes) / len(fold_maes)
        results.append({"params": params, "cv_mae": avg_mae})

    results.sort(key=lambda r: r["cv_mae"])
    best = results[0]
    return best["params"], best["cv_mae"], results


def train_fuel_moisture_model(channel="beta", bump="patch", search=True):
    # 2. Load Data
    df = pd.read_csv(paths.TRAINING_DATA_DIR / 'final_training_data.csv', parse_dates=['obs_time'])
    df = df.sort_values('obs_time')

    # 3. Define Features and Target
    # Base features (always included)
    features_to_use = [
        'temp_c', 'rel_humidity', 'wind_speed_ms',
        'hour', 'month', 'emc_baseline',
        'temp_mean_3h', 'rh_mean_3h', 'temp_mean_6h', 'rh_mean_6h'
    ]

    # Add precipitation features if they exist in the dataset
    precip_features = ['precip_1h', 'precip_3h', 'precip_6h', 'precip_24h', 'hours_since_rain']
    available_precip_features = [f for f in precip_features if f in df.columns]

    if available_precip_features:
        features_to_use.extend(available_precip_features)
        print(f"✅ Including precipitation features: {available_precip_features}")
    else:
        print("⚠️  No precipitation features found in training data.")

    X = df[features_to_use]
    y = df['target_fm']

    # 4. Split into Train/Test sets chronologically (avoids leaking rolling-window
    # features across adjacent timestamps, which a random split would do).
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # 5. Pick hyperparameters (searched or fixed defaults), then fit on the
    # full chronological training split. The holdout evaluation below is
    # unaffected by whether search ran - same split, same metrics either way.
    if search:
        print("🔍 Running rolling-origin hyperparameter search...")
        best_params, best_cv_mae, all_results = search_hyperparameters(X_train, y_train)
        print(f"   Best CV MAE: {best_cv_mae:.3f} with params: {best_params}")
    else:
        best_params = {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 5}
        all_results = None

    model = xgb.XGBRegressor(objective='reg:squarederror', **best_params)
    model.fit(X_train, y_train)
    if available_precip_features:
        model.get_booster().set_attr(precipitation_contract_version=PRECIPITATION_CONTRACT_VERSION,
                                     precipitation_contract_sha256=PRECIPITATION_CONTRACT_SHA256)
    
    # 6. Evaluate Performance
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"📈 Model Performance:")
    print(f"   - Mean Absolute Error: {mae:.2f}%")
    print(f"   - R-Squared Score: {r2:.2f}")
    
    # 7. Save the model artifact to a scratch path, then hand it to the version registry
    scratch_model_path = paths.MODELS_DIR / f'.scratch_fuel_moisture_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    model.save_model(str(scratch_model_path))

    performance_metrics = {
        "mae": round(mae, 3),
        "r2_score": round(r2, 3),
        "training_samples": len(X_train),
        "test_samples": len(X_test)
    }

    # Compare against whatever is currently `stable` BEFORE registering this
    # run's beta candidate - registration only ever writes to `beta`/`stable`
    # via promote(), so this order doesn't matter for correctness, but doing
    # it first means the comparison line reads naturally as "new vs current".
    comparison_line, is_better = compare_to_stable("fuel_moisture", performance_metrics)

    version = register_trained_model(
        model_type="fuel_moisture",
        source_path=scratch_model_path,
        performance=performance_metrics,
        bump=bump,
        channel=channel,
        metadata=metadata,
    )
    scratch_model_path.unlink()
    print(f"✅ Registered fuel_moisture model as {channel} version {version}")
    if channel == "beta":
        print(f"   Promote it with: python pipelines/promote_model.py --model fuel_moisture --version {version}")

    print(f"📊 {comparison_line}")

    append_experiment(
        "fuel_moisture",
        metrics=performance_metrics,
        beta_version=version if channel == "beta" else None,
        params=best_params,
    )

    # 8. Feature Importance Visualization
    plt.figure(figsize=(10, 8))
    xgb.plot_importance(model)
    plt.tight_layout()
    plt.savefig(paths.PLOTS_DIR / 'feature_importance.png')

    return {
        "version": version,
        "channel": channel,
        "performance": performance_metrics,
        "params": best_params,
        "comparison": comparison_line,
        "is_better_than_stable": is_better,
    }

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the fuel moisture model")
    parser.add_argument("--channel", choices=["beta", "stable"], default="beta",
                         help="Channel to register the trained model under (default: beta)")
    parser.add_argument("--bump", choices=["major", "minor", "patch"], default="patch",
                         help="Version segment to bump (default: patch)")
    parser.add_argument("--no-search", action="store_true",
                         help="Skip hyperparameter search and use the historical fixed defaults")
    args = parser.parse_args()

    train_fuel_moisture_model(channel=args.channel, bump=args.bump, search=not args.no_search)
