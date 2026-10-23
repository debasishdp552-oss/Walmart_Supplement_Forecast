"""
train_pipeline.py
==================
Per-product demand forecasting with walk-forward (expanding-window) time
series cross-validation, benchmarking 3 approaches:

    1. ETS (Holt-Winters)  (statsmodels ExponentialSmoothing, per product,
                             additive trend + additive seasonality, period=52)
    2. Prophet              (per product, with Price/Discount regressors)
    3. XGBoost (pooled)     (one model across all products, product/category/
                             location as categorical features; forecast built
                             recursively since true future lags don't exist yet)

Each product is assigned its own backtest winner by mean walk-forward
MAPE, mirroring a real deployment where you don't force one algorithm on
every SKU. Winners + metrics + fitted artifacts are pickled to
artifacts/artifacts.pkl for the Streamlit app to load without retraining.

Run:
    python train_pipeline.py
"""

import os
import pickle
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

from preprocessing import build_features, add_calendar_features, SEASONAL_PERIOD, PROCESSED_PATH

N_SPLITS = 3          # expanding-window folds
TEST_HORIZON = 8       # weeks predicted per fold
MIN_TRAIN_SIZE = 104    # need >= 2 years before the first fold
ARTIFACT_PATH = "artifacts/artifacts.pkl"

FEATURE_COLS = [
    "Sales_Lag_1", "Sales_Lag_4", "Sales_Lag_12", "Sales_Lag_52",
    "Rolling_Mean_4", "Rolling_Std_4", "Rolling_Mean_12", "Rolling_Mean_52",
    "Price_Lag_1", "Discount_Lag_1",
    "Week_sin", "Week_cos", "Month", "Quarter", "Holiday_Season",
    "Category_Code", "Location_Code", "Product_Code",
]


def mape(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _naive_fallback(train: pd.Series, horizon: int, period: int = SEASONAL_PERIOD) -> np.ndarray:
    """Internal-only fallback (not a compared model) used when ETS/Prophet
    fail to converge on a short or degenerate series."""
    if len(train) >= period:
        return np.resize(train.iloc[-period:].values, horizon)
    return np.repeat(train.iloc[-1], horizon)


# ----------------------------------------------------------------------
# Individual model forecasters. Each takes a train slice (single product,
# sorted by date) and a horizon, and returns an array of point forecasts.
# ----------------------------------------------------------------------

def forecast_ets(train: pd.Series, horizon: int, period: int = SEASONAL_PERIOD) -> np.ndarray:
    """ETS / Holt-Winters exponential smoothing with additive trend and
    additive seasonality. Much cheaper to fit than a seasonal SARIMA at the
    same period, which is why it's the model of choice here when training
    speed matters (a full run trains ~4x per product across CV folds)."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    if len(train) < 2 * period:
        # not enough history for a full seasonal cycle -> fall back
        return _naive_fallback(train, horizon, period)
    try:
        model = ExponentialSmoothing(
            train, trend="add", seasonal="add", seasonal_periods=period,
            initialization_method="estimated",
        )
        res = model.fit(optimized=True)
        fc = res.forecast(horizon).values
        return np.clip(fc, a_min=0, a_max=None)  # demand can't be negative
    except Exception:
        return _naive_fallback(train, horizon, period)


def forecast_prophet(train_df: pd.DataFrame, horizon: int) -> np.ndarray:
    from prophet import Prophet

    pdf = train_df[["Date", "Units Sold", "Price", "Discount"]].rename(
        columns={"Date": "ds", "Units Sold": "y"}
    )
    try:
        m = Prophet(
            weekly_seasonality=False,
            yearly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode="additive",
        )
        m.add_regressor("Price")
        m.add_regressor("Discount")
        m.fit(pdf)

        future = m.make_future_dataframe(periods=horizon, freq="W-MON")
        # hold price/discount constant at last known values for the forecast window
        future["Price"] = pdf["Price"].iloc[-1]
        future["Discount"] = pdf["Discount"].iloc[-1]
        fc = m.predict(future)
        return fc["yhat"].values[-horizon:]
    except Exception:
        return _naive_fallback(train_df["Units Sold"], horizon)


def train_xgboost_pooled(feat_df: pd.DataFrame, cutoff_date):
    from xgboost import XGBRegressor

    train = feat_df[feat_df["Date"] < cutoff_date].dropna(subset=FEATURE_COLS + ["Units Sold"])
    model = XGBRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    )
    model.fit(train[FEATURE_COLS], train["Units Sold"])
    return model


def recursive_xgboost_forecast(model, full_df: pd.DataFrame, product: str, horizon: int) -> np.ndarray:
    """XGBoost has no naturally-generated future rows (lags depend on
    values we haven't seen yet), so we roll forward one week at a time,
    recomputing lag/rolling features from previously predicted values and
    calendar features from the actual future date of each step."""
    hist = full_df[full_df["Product Name"] == product].sort_values("Date").copy()
    preds = []
    last_price = hist["Price"].iloc[-1]
    last_discount = hist["Discount"].iloc[-1]
    future_dates = pd.date_range(hist["Date"].iloc[-1] + pd.Timedelta(weeks=1),
                                 periods=horizon, freq="W-MON")
    cal = add_calendar_features(pd.DataFrame({"Date": future_dates}))

    series = hist["Units Sold"].tolist()
    for step in range(horizon):
        row = {
            "Sales_Lag_1": series[-1],
            "Sales_Lag_4": series[-4] if len(series) >= 4 else series[0],
            "Sales_Lag_12": series[-12] if len(series) >= 12 else series[0],
            "Sales_Lag_52": series[-52] if len(series) >= 52 else series[0],
            "Rolling_Mean_4": np.mean(series[-4:]),
            "Rolling_Std_4": np.std(series[-4:]) if len(series) >= 2 else 0.0,
            "Rolling_Mean_12": np.mean(series[-12:]),
            "Rolling_Mean_52": np.mean(series[-52:]) if len(series) >= 52 else np.mean(series),
            "Price_Lag_1": last_price,
            "Discount_Lag_1": last_discount,
            "Week_sin": cal["Week_sin"].iloc[step],
            "Week_cos": cal["Week_cos"].iloc[step],
            "Month": cal["Month"].iloc[step],
            "Quarter": cal["Quarter"].iloc[step],
            "Holiday_Season": cal["Holiday_Season"].iloc[step],
            "Category_Code": hist["Category_Code"].iloc[-1],
            "Location_Code": hist["Location_Code"].iloc[-1],
            "Product_Code": hist["Product_Code"].iloc[-1],
        }
        x = pd.DataFrame([row])[FEATURE_COLS]
        yhat = float(model.predict(x)[0])
        preds.append(yhat)
        series.append(yhat)

    return np.array(preds)


# ----------------------------------------------------------------------
# Walk-forward CV per product
# ----------------------------------------------------------------------

@dataclass
class ProductResult:
    product: str
    mape_by_model: dict = field(default_factory=dict)
    best_model: str = ""
    best_mape: float = np.inf
    fold_details: list = field(default_factory=list)


def walk_forward_cv_for_product(product: str, full_df: pd.DataFrame) -> ProductResult:
    prod_df = full_df[full_df["Product Name"] == product].sort_values("Date").reset_index(drop=True)
    n = len(prod_df)

    if n < MIN_TRAIN_SIZE + TEST_HORIZON:
        result = ProductResult(product=product)
        result.mape_by_model = {"ETS": np.nan}
        result.best_model = "ETS"
        return result

    tscv = TimeSeriesSplit(n_splits=N_SPLITS, test_size=TEST_HORIZON, max_train_size=None)
    indices = np.arange(n)
    fold_scores = {"ETS": [], "Prophet": []}

    result = ProductResult(product=product)

    for train_idx, test_idx in tscv.split(indices):
        if len(train_idx) < MIN_TRAIN_SIZE:
            continue
        train_series = prod_df.iloc[train_idx]["Units Sold"]
        train_block = prod_df.iloc[train_idx]
        y_true = prod_df.iloc[test_idx]["Units Sold"].values
        cutoff_date = prod_df.iloc[test_idx]["Date"].iloc[0]

        preds = {
            "ETS": forecast_ets(train_series, len(test_idx)),
            "Prophet": forecast_prophet(train_block, len(test_idx)),
        }
        fold_row = {"cutoff": str(cutoff_date.date())}
        for name, yhat in preds.items():
            yhat = np.resize(yhat, len(y_true))
            score = mape(y_true, yhat)
            fold_scores[name].append(score)
            fold_row[name] = round(score, 2)
        result.fold_details.append(fold_row)

    for name, scores in fold_scores.items():
        clean = [s for s in scores if not np.isnan(s)]
        result.mape_by_model[name] = round(float(np.mean(clean)), 2) if clean else np.nan

    return result


def run_pipeline():
    print("Building features...")
    feat = build_features()
    full_df = feat.copy()
    os.makedirs("data", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)
    full_df.to_csv(PROCESSED_PATH, index=False)

    products = sorted(full_df["Product Name"].unique())
    print(f"Running walk-forward CV for {len(products)} products "
          f"({N_SPLITS} folds x {TEST_HORIZON}-week horizon)...")
    print("Models: ETS (Holt-Winters), Prophet, XGBoost (pooled)")

    results = {}
    for p in products:
        print(f"  -> {p}")
        results[p] = walk_forward_cv_for_product(p, full_df)

    # ---- Pooled XGBoost: train once on ALL history up to the final fold's
    #      cutoff so it's comparable to per-product backtests, then also
    #      score it per product on the same last fold for a fair 3-way compare.
    last_cutoff = full_df["Date"].max() - pd.Timedelta(weeks=TEST_HORIZON)
    xgb_model = train_xgboost_pooled(full_df, last_cutoff)

    for p in products:
        prod_df = full_df[full_df["Product Name"] == p].sort_values("Date")
        test_block = prod_df[prod_df["Date"] >= last_cutoff].iloc[:TEST_HORIZON]
        if len(test_block) == 0 or test_block[FEATURE_COLS].isna().any().any():
            results[p].mape_by_model["XGBoost (pooled)"] = np.nan
            continue
        yhat = xgb_model.predict(test_block[FEATURE_COLS])
        score = mape(test_block["Units Sold"].values, yhat)
        results[p].mape_by_model["XGBoost (pooled)"] = round(score, 2)

    # pick best model per product (lowest mean MAPE, ignoring NaNs)
    for p in products:
        valid = {k: v for k, v in results[p].mape_by_model.items() if v is not None and not np.isnan(v)}
        if valid:
            best = min(valid, key=valid.get)
            results[p].best_model = best
            results[p].best_mape = valid[best]
        else:
            results[p].best_model = "ETS"
            results[p].best_mape = np.nan

    # ---- Final production forecast: refit each product's winning model on
    #      ALL available history and forecast the next TEST_HORIZON weeks.
    forecasts = {}
    for p in products:
        prod_df = full_df[full_df["Product Name"] == p].sort_values("Date").reset_index(drop=True)
        winner = results[p].best_model
        h = TEST_HORIZON
        if winner == "ETS":
            fc = forecast_ets(prod_df["Units Sold"], h)
        elif winner == "Prophet":
            fc = forecast_prophet(prod_df, h)
        else:  # XGBoost pooled -> no future feature rows exist yet, refit-forecast recursively
            fc = recursive_xgboost_forecast(xgb_model, full_df, p, h)
        forecasts[p] = fc

    metrics_table = pd.DataFrame({p: results[p].mape_by_model for p in products}).T
    metrics_table["Best Model"] = [results[p].best_model for p in products]
    metrics_table["Best MAPE"] = [results[p].best_mape for p in products]

    # Store results as plain dicts (not dataclass instances) so the artifact
    # unpickles cleanly in app.py without needing train_pipeline's __main__
    # module context.
    results_plain = {
        p: {
            "product": r.product,
            "mape_by_model": r.mape_by_model,
            "best_model": r.best_model,
            "best_mape": r.best_mape,
            "fold_details": r.fold_details,
        }
        for p, r in results.items()
    }

    artifacts = {
        "results": results_plain,
        "metrics_table": metrics_table,
        "forecasts": forecasts,
        "xgb_model": xgb_model,
        "feature_cols": FEATURE_COLS,
        "test_horizon": TEST_HORIZON,
        "last_date": full_df["Date"].max(),
        "processed_data_path": PROCESSED_PATH,
    }
    with open(ARTIFACT_PATH, "wb") as f:
        pickle.dump(artifacts, f)

    print("\n=== Walk-forward MAPE by model (%) ===")
    print(metrics_table.round(2))
    print(f"\nSaved artifacts -> {ARTIFACT_PATH}")
    return artifacts


if __name__ == "__main__":
    run_pipeline()
