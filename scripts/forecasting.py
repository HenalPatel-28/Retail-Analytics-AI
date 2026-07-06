"""
forecasting.py

Purpose
-------
Forecasts weekly total sales using three models of increasing sophistication,
compares them on held-out test data using MAE, RMSE, and R^2, and saves the
best-performing model to disk.

Models
------
    1. Linear Regression  -- simple baseline; if fancier models can't beat
                              this, they aren't earning their complexity.
    2. Random Forest       -- captures non-linear patterns and interactions
                              between calendar features that linear
                              regression structurally cannot.
    3. Prophet              -- purpose-built for business time series;
                              models trend and seasonality natively rather
                              than through manually engineered features.

Critical Design Decision: Chronological Split, Never Random
--------------------------------------------------------------
We split train/test by DATE (earlier weeks = train, later weeks = test),
never with a random shuffle. A random split would let the model train on
weeks that come AFTER weeks it's being tested on -- this is data leakage
in its time-series form, the same concept introduced in Module 5, just
showing up in a different part of the pipeline.

Output
------
    reports/forecasting/model_comparison.png
    reports/forecasting_report.txt
    models/best_forecast_model.pkl        (Linear Regression or Random Forest)
    models/prophet_model.pkl              (Prophet, saved separately -- it
                                            doesn't pickle the same way as
                                            sklearn estimators)
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = Path("data/cleaned/retail_sales_features.csv")
CHARTS_DIR = Path("reports/forecasting")
REPORT_PATH = Path("reports/forecasting_report.txt")
MODELS_DIR = Path("models")

TEST_SIZE_FRACTION = 0.15  # last 15% of weeks held out as test data

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_weekly_sales() -> pd.DataFrame:
    """Load transaction data and aggregate to weekly total sales.

    We chose weekly granularity deliberately: daily is too noisy (large
    day-to-day swings with no real signal), monthly leaves too few data
    points (~48) for a meaningful train/test split.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found. Run scripts/feature_engineering.py first.")
    df = pd.read_csv(DATA_PATH, parse_dates=["order_date"])

    weekly = (
        df.set_index("order_date")
        .resample("W")["sales"]
        .sum()
        .reset_index()
        .rename(columns={"order_date": "week_start", "sales": "total_sales"})
    )
    logger.info("Aggregated to %d weekly data points.", len(weekly))
    return weekly


def add_calendar_features(weekly: pd.DataFrame) -> pd.DataFrame:
    """Engineer explicit calendar features for Linear Regression and Random Forest.

    Neither model understands dates natively -- we must translate 'when' into
    numeric signals: a linear trend index (time_index), seasonal indicators
    (month, quarter), and our known festive-season flag from Module 5's logic.
    """
    weekly = weekly.copy()
    weekly["time_index"] = np.arange(len(weekly))  # 0, 1, 2, ... captures overall trend
    weekly["month"] = weekly["week_start"].dt.month
    weekly["quarter"] = weekly["week_start"].dt.quarter
    weekly["week_of_year"] = weekly["week_start"].dt.isocalendar().week.astype(int)
    weekly["is_festive_season"] = weekly["month"].isin([10, 11]).astype(int)
    return weekly


def chronological_split(weekly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by DATE ORDER, never randomly.

    This is the single most important correctness rule in this script.
    weekly is already sorted by week_start (resample guarantees this), so
    a straightforward positional split is chronologically valid.
    """
    split_idx = int(len(weekly) * (1 - TEST_SIZE_FRACTION))
    train = weekly.iloc[:split_idx].reset_index(drop=True)
    test = weekly.iloc[split_idx:].reset_index(drop=True)
    logger.info(
        "Chronological split: %d train weeks (%s to %s), %d test weeks (%s to %s).",
        len(train), train["week_start"].min().date(), train["week_start"].max().date(),
        len(test), test["week_start"].min().date(), test["week_start"].max().date(),
    )
    return train, test


FEATURE_COLS = ["time_index", "month", "quarter", "week_of_year", "is_festive_season"]


def train_linear_regression(train: pd.DataFrame) -> LinearRegression:
    model = LinearRegression()
    model.fit(train[FEATURE_COLS], train["total_sales"])
    return model


def train_random_forest(train: pd.DataFrame) -> RandomForestRegressor:
    # n_estimators=200 and a fixed random_state: enough trees for stable
    # predictions, seeded for reproducibility (same habit as Module 2).
    model = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
    model.fit(train[FEATURE_COLS], train["total_sales"])
    return model


def train_prophet(train: pd.DataFrame):
    """Train a Prophet model.

    Prophet requires exactly two columns named 'ds' (date) and 'y' (value).
    We enable yearly seasonality (our Oct/Nov pattern repeats every year)
    and disable daily/weekly seasonality since our data is already
    aggregated to weekly points -- sub-weekly seasonality is meaningless here.
    """
    from prophet import Prophet  # imported here so the rest of the script
    # still works (Linear Regression, Random Forest) even in an environment
    # where Prophet isn't installed -- useful when isolating issues.

    prophet_df = train[["week_start", "total_sales"]].rename(columns={"week_start": "ds", "total_sales": "y"})
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(prophet_df)
    return model


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute the three standard regression metrics.

    MAE: average absolute error, in the same units as sales (Rs) -- easiest
         to explain to a non-technical stakeholder ("off by about Rs X on average").
    RMSE: penalizes large errors more heavily than MAE -- useful for knowing
         if the model occasionally makes very large mistakes.
    R^2: proportion of variance explained -- 1.0 is perfect, 0.0 means the
         model is no better than always predicting the mean.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "R2": round(r2, 4)}


def plot_comparison(test: pd.DataFrame, predictions: dict) -> None:
    """Plot actual vs each model's predicted values on the test period."""
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(test["week_start"], test["total_sales"], label="Actual", color="black", linewidth=2, marker="o")
    colors = {"Linear Regression": "#4C72B0", "Random Forest": "#55A868", "Prophet": "#C44E52"}
    for model_name, preds in predictions.items():
        ax.plot(test["week_start"], preds, label=model_name, linestyle="--", marker="x", color=colors.get(model_name))
    ax.set_title("Forecast Model Comparison -- Test Period (Actual vs Predicted)")
    ax.set_ylabel("Total Weekly Sales (Rs)")
    ax.legend()
    fig.tight_layout()
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHARTS_DIR / "model_comparison.png")
    plt.close(fig)
    logger.info("Saved model comparison chart.")


def main() -> None:
    weekly = load_weekly_sales()
    weekly = add_calendar_features(weekly)
    train, test = chronological_split(weekly)

    results = {}
    predictions = {}

    logger.info("Training Linear Regression...")
    lr_model = train_linear_regression(train)
    lr_preds = lr_model.predict(test[FEATURE_COLS])
    results["Linear Regression"] = evaluate(test["total_sales"], lr_preds)
    predictions["Linear Regression"] = lr_preds

    logger.info("Training Random Forest...")
    rf_model = train_random_forest(train)
    rf_preds = rf_model.predict(test[FEATURE_COLS])
    results["Random Forest"] = evaluate(test["total_sales"], rf_preds)
    predictions["Random Forest"] = rf_preds

    prophet_model = None
    try:
        logger.info("Training Prophet...")
        prophet_model = train_prophet(train)
        future = test[["week_start"]].rename(columns={"week_start": "ds"})
        forecast = prophet_model.predict(future)
        prophet_preds = forecast["yhat"].values
        results["Prophet"] = evaluate(test["total_sales"], prophet_preds)
        predictions["Prophet"] = prophet_preds
    except ImportError:
        logger.warning("Prophet not installed -- skipping Prophet model. Run: pip install prophet")
    except Exception as exc:
        logger.warning("Prophet failed to train (%s: %s) -- skipping Prophet and continuing. If this is a CmdStan error, run: python -m cmdstanpy.install_cmdstan", type(exc).__name__, exc)

    plot_comparison(test, predictions)

    # Pick the winner by lowest RMSE -- penalizes large misses more than MAE,
    # which matters more for business planning (a single very wrong week is
    # worse than several slightly-off weeks).
    best_model_name = min(results, key=lambda name: results[name]["RMSE"])
    logger.info("Best model by RMSE: %s", best_model_name)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if best_model_name == "Linear Regression":
        with open(MODELS_DIR / "best_forecast_model.pkl", "wb") as f:
            pickle.dump(lr_model, f)
    elif best_model_name == "Random Forest":
        with open(MODELS_DIR / "best_forecast_model.pkl", "wb") as f:
            pickle.dump(rf_model, f)
    elif best_model_name == "Prophet" and prophet_model is not None:
        with open(MODELS_DIR / "prophet_model.pkl", "wb") as f:
            pickle.dump(prophet_model, f)

    report_lines = [
        "=" * 70,
        "SALES FORECASTING -- MODEL COMPARISON REPORT",
        "=" * 70,
        f"Granularity: Weekly total sales",
        f"Train period: {train['week_start'].min().date()} to {train['week_start'].max().date()} ({len(train)} weeks)",
        f"Test period: {test['week_start'].min().date()} to {test['week_start'].max().date()} ({len(test)} weeks)",
        "",
        "-- Model Performance (on held-out test weeks) --",
    ]
    for name, metrics in results.items():
        report_lines.append(f"  {name}: MAE=Rs {metrics['MAE']:,.2f} | RMSE=Rs {metrics['RMSE']:,.2f} | R2={metrics['R2']}")
    report_lines.append("")
    report_lines.append(f"-- Winner: {best_model_name} (lowest RMSE) --")
    report_lines.append("")
    report_lines.append("-- Interpretation --")
    report_lines.append(
        "  MAE tells us the typical dollar-value error; RMSE penalizes any large individual "
        "misses more heavily. We select on RMSE because, for business planning, one badly "
        "wrong week (e.g., under-forecasting a festive spike) is more costly than several "
        "slightly-off ordinary weeks."
    )
    report_lines.append("=" * 70)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
    logger.info("Saved forecasting report to %s.", REPORT_PATH)

    print("\n" + "\n".join(report_lines))


if __name__ == "__main__":
    main()

