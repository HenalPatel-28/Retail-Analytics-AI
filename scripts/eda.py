"""
eda.py

Purpose
-------
Performs structured Exploratory Data Analysis on the engineered dataset
(data/cleaned/retail_sales_features.csv), following the standard analytical
sequence: descriptive statistics -> univariate analysis -> correlation
analysis -> categorical breakdowns -> time-series exploration.

Output
------
    reports/eda/*.png              -- exploratory charts
    reports/eda_summary_report.txt -- written findings, in plain business language

Note on scope
-------------
This module produces exploratory, "for the analyst's own understanding"
charts using matplotlib/seaborn. Module 8 (Visualization) builds polished,
presentation-ready interactive charts using Plotly for the actual
dashboards -- the two modules serve different purposes and audiences.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend -- we only save figures, never display them
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

DATA_PATH = Path("data/cleaned/retail_sales_features.csv")
CHARTS_DIR = Path("reports/eda")
REPORT_PATH = Path("reports/eda_summary_report.txt")

NUMERIC_COLS_OF_INTEREST = [
    "sales", "profit", "quantity", "discount", "unit_price",
    "profit_margin_pct", "days_to_ship",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found. Run scripts/feature_engineering.py first.")
    df = pd.read_csv(DATA_PATH, parse_dates=["order_date", "ship_date"])
    logger.info("Loaded %d rows, %d columns.", *df.shape)
    return df


def descriptive_statistics(df: pd.DataFrame) -> str:
    """Compute summary statistics: mean, median, std, skewness, kurtosis.

    Skewness and kurtosis matter specifically because sales/profit data in
    retail is almost never normally distributed -- it's typically right-skewed
    (many small orders, a few very large ones). Knowing this BEFORE modeling
    tells us to expect issues if we ever assume normality (e.g., certain
    statistical tests, or naive linear regression without transformation).
    """
    lines = ["-- Descriptive Statistics --\n"]
    desc = df[NUMERIC_COLS_OF_INTEREST].describe().round(2)
    lines.append(desc.to_string())
    lines.append("\n\n-- Skewness & Kurtosis --")
    for col in NUMERIC_COLS_OF_INTEREST:
        skew = stats.skew(df[col].dropna())
        kurt = stats.kurtosis(df[col].dropna())
        interpretation = "right-skewed" if skew > 0.5 else ("left-skewed" if skew < -0.5 else "roughly symmetric")
        lines.append(f"  {col}: skew={skew:.2f} ({interpretation}), kurtosis={kurt:.2f}")
    return "\n".join(lines)


def plot_distributions(df: pd.DataFrame) -> None:
    """Histogram + boxplot for each key numeric variable.

    We plot histogram AND boxplot side by side for sales/profit specifically
    because a histogram shows shape (skewness) while a boxplot makes outlier
    counts immediately visible -- together they tell a more complete story
    than either alone.
    """
    for col in ["sales", "profit", "quantity", "discount"]:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        sns.histplot(df[col], bins=40, kde=True, ax=axes[0], color="#4C72B0")
        axes[0].set_title(f"Distribution of {col}")
        sns.boxplot(x=df[col], ax=axes[1], color="#DD8452")
        axes[1].set_title(f"Boxplot of {col}")
        fig.tight_layout()
        fig.savefig(CHARTS_DIR / f"distribution_{col}.png")
        plt.close(fig)
    logger.info("Saved distribution plots for sales, profit, quantity, discount.")


def plot_correlation_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    """Correlation matrix of key numeric features.

    This is a genuinely useful step BEFORE Module 9's forecasting: if two
    features are highly correlated (e.g., sales and profit, which we'd
    expect), including both as independent predictors in a linear model
    can cause multicollinearity issues.
    """
    corr = df[NUMERIC_COLS_OF_INTEREST].corr().round(2)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax, vmin=-1, vmax=1)
    ax.set_title("Correlation Matrix -- Key Numeric Features")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "correlation_heatmap.png")
    plt.close(fig)
    logger.info("Saved correlation heatmap.")
    return corr


def plot_category_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Sales and profit margin by category -- a bar chart and a boxplot.

    We deliberately show profit_margin_pct BY category as a boxplot (not
    just a bar of averages) because Module 2's data generator gave
    Technology a much lower average margin AND much higher variance --
    an average alone would hide that variance.
    """
    summary = (
        df.groupby("category")
        .agg(total_sales=("sales", "sum"), avg_margin_pct=("profit_margin_pct", "mean"), order_count=("order_id", "nunique"))
        .round(2)
        .sort_values("total_sales", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    summary["total_sales"].plot(kind="bar", ax=ax, color="#55A868")
    ax.set_title("Total Sales by Category")
    ax.set_ylabel("Total Sales (Rs)")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "sales_by_category.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=df, x="category", y="profit_margin_pct", ax=ax)
    ax.set_title("Profit Margin % Distribution by Category")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "margin_by_category.png")
    plt.close(fig)

    logger.info("Saved category breakdown charts.")
    return summary


def plot_regional_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Sales by region -- both an absolute bar chart and a share-of-total pie chart."""
    region_sales = df.groupby("region")["sales"].sum().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    region_sales.plot(kind="bar", ax=ax, color="#8172B2")
    ax.set_title("Total Sales by Region")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "sales_by_region.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(region_sales, labels=region_sales.index, autopct="%1.1f%%", startangle=90)
    ax.set_title("Regional Share of Total Sales")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "regional_share_pie.png")
    plt.close(fig)

    logger.info("Saved regional breakdown charts.")
    return region_sales.round(2)


def plot_monthly_trend(df: pd.DataFrame) -> pd.Series:
    """Monthly sales trend line -- our first direct look at the seasonality
    we deliberately built into the data back in Module 2.
    """
    monthly = df.set_index("order_date").resample("ME")["sales"].sum()

    fig, ax = plt.subplots(figsize=(10, 4))
    monthly.plot(ax=ax, marker="o", color="#C44E52")
    ax.set_title("Monthly Sales Trend")
    ax.set_ylabel("Total Sales (Rs)")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "monthly_sales_trend.png")
    plt.close(fig)

    logger.info("Saved monthly sales trend chart.")
    return monthly.round(2)


def plot_scatter_discount_vs_profit(df: pd.DataFrame) -> None:
    """Scatter plot: discount vs profit margin.

    This is the single most useful exploratory chart for this dataset --
    it should visually confirm the "heavy discounts erode margin, sometimes
    into a loss" relationship we deliberately encoded in Module 2.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    sample = df.sample(min(2000, len(df)), random_state=42)  # sample for readability on large data
    sns.scatterplot(data=sample, x="discount", y="profit_margin_pct", hue="category", alpha=0.5, ax=ax)
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_title("Discount vs Profit Margin % (sampled)")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "discount_vs_margin_scatter.png")
    plt.close(fig)
    logger.info("Saved discount vs profit margin scatter plot.")


def build_report(
    df: pd.DataFrame,
    desc_text: str,
    corr: pd.DataFrame,
    category_summary: pd.DataFrame,
    region_sales: pd.Series,
    monthly_trend: pd.Series,
) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("EXPLORATORY DATA ANALYSIS -- SUMMARY REPORT")
    lines.append("=" * 70)
    lines.append(f"Dataset: {len(df)} rows, {df.shape[1]} columns")
    lines.append(f"Date range: {df['order_date'].min().date()} to {df['order_date'].max().date()}")
    lines.append("")
    lines.append(desc_text)
    lines.append("")
    lines.append("-- Correlation Highlights --")
    sales_corr = corr["sales"].drop("sales").sort_values(ascending=False)
    lines.append(f"  Strongest correlation with sales: {sales_corr.index[0]} ({sales_corr.iloc[0]})")
    lines.append(f"  Weakest correlation with sales: {sales_corr.index[-1]} ({sales_corr.iloc[-1]})")
    lines.append("")
    lines.append("-- Category Performance --")
    lines.append(category_summary.to_string())
    lines.append("")
    lines.append("-- Regional Sales --")
    lines.append(region_sales.to_string())
    lines.append("")
    lines.append("-- Monthly Sales Trend (highlights) --")
    peak_month = monthly_trend.idxmax()
    low_month = monthly_trend.idxmin()
    lines.append(f"  Peak month: {peak_month.strftime('%Y-%m')} (Rs {monthly_trend.max():,.0f})")
    lines.append(f"  Lowest month: {low_month.strftime('%Y-%m')} (Rs {monthly_trend.min():,.0f})")
    lines.append("")
    lines.append("-- Key Findings (plain language) --")
    lines.append(
        "  1. Sales and profit are right-skewed: most orders are modest in value, "
        "with a smaller number of large orders pulling the average upward. "
        "Median is more representative of a 'typical' order than the mean."
    )
    lines.append(
        "  2. Heavy discounting visibly erodes profit margin, and a meaningful "
        "share of highly-discounted orders fall below the profitability line "
        "(see discount_vs_margin_scatter.png)."
    )
    lines.append(
        "  3. Sales show clear seasonality, peaking in the Oct/Nov festive window -- "
        "this is a strong signal for Module 9's forecasting model to capture."
    )
    lines.append("=" * 70)
    return "\n".join(lines)


def main() -> None:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = load_data()

    desc_text = descriptive_statistics(df)
    plot_distributions(df)
    corr = plot_correlation_heatmap(df)
    category_summary = plot_category_breakdown(df)
    region_sales = plot_regional_breakdown(df)
    monthly_trend = plot_monthly_trend(df)
    plot_scatter_discount_vs_profit(df)

    report_text = build_report(df, desc_text, corr, category_summary, region_sales, monthly_trend)
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    logger.info("Saved EDA summary report to %s.", REPORT_PATH)

    print("\n" + report_text)


if __name__ == "__main__":
    main()
