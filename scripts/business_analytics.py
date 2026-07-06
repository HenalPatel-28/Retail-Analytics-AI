"""
business_analytics.py

Purpose
-------
Computes the standard set of retail business KPIs from the engineered
dataset (data/cleaned/retail_sales_features.csv), using explicit,
documented formulas -- matching how these metrics are actually defined
in retail finance/analytics, not ad-hoc approximations.

Output
------
    reports/kpi_summary_report.txt   -- headline KPIs in plain business language
    reports/kpi_monthly_trend.csv    -- month-over-month KPI trend table
    reports/kpi_top_products.csv     -- top products/categories by sales

A Note on KPI Definitions
---------------------------
Several KPIs here (especially Customer Lifetime Value) have more than one
legitimate industry definition. We use the simplest, most transparent
version and document it explicitly -- so anyone reading this code knows
exactly what assumption was made, rather than assuming their own
definition matches ours silently.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/cleaned/retail_sales_features.csv")
REPORT_PATH = Path("reports/kpi_summary_report.txt")
MONTHLY_TREND_PATH = Path("reports/kpi_monthly_trend.csv")
TOP_PRODUCTS_PATH = Path("reports/kpi_top_products.csv")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found. Run scripts/feature_engineering.py first.")
    df = pd.read_csv(DATA_PATH, parse_dates=["order_date", "ship_date"])
    logger.info("Loaded %d rows, %d columns.", *df.shape)
    return df


def compute_headline_kpis(df: pd.DataFrame) -> dict:
    """Compute the core, single-number KPIs that headline any executive report.

    Every formula here is deliberately explicit (not hidden behind a
    library function) so the definition is transparent and auditable --
    exactly what a finance team would expect to be able to verify by hand.
    """
    total_sales = df["sales"].sum()
    total_profit = df["profit"].sum()
    total_orders = df["order_id"].nunique()
    total_customers = df["customer_id"].nunique()

    profit_margin_pct = (total_profit / total_sales) * 100

    # Average Order Value: total revenue divided by the number of DISTINCT
    # orders, NOT the number of rows -- a common bug is dividing by row
    # count, which double-counts multi-item orders and understates AOV.
    average_order_value = total_sales / total_orders

    # Customer Lifetime Value (simple definition, explicitly documented):
    # total revenue generated divided by total unique customers, over the
    # observed period. This is NOT a predictive/future-looking CLV -- it's
    # a historical average. We call this out explicitly in the report too.
    customer_lifetime_value_simple = total_sales / total_customers

    # Repeat Customer Rate: percentage of customers who placed MORE THAN
    # ONE distinct order. We use customer_total_orders_to_date's final
    # value per customer (i.e., their true total), NOT the leakage-prone
    # column directly, to make the intent explicit here.
    orders_per_customer = df.groupby("customer_id")["order_id"].nunique()
    repeat_customers = (orders_per_customer > 1).sum()
    repeat_customer_rate_pct = (repeat_customers / total_customers) * 100

    return {
        "total_sales": total_sales,
        "total_profit": total_profit,
        "profit_margin_pct": profit_margin_pct,
        "total_orders": total_orders,
        "total_customers": total_customers,
        "average_order_value": average_order_value,
        "customer_lifetime_value_simple": customer_lifetime_value_simple,
        "repeat_customers": int(repeat_customers),
        "repeat_customer_rate_pct": repeat_customer_rate_pct,
    }


def compute_monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Month-over-month sales, profit, and growth rate.

    Growth % compares each month to the PREVIOUS month's total sales.
    The first month in the dataset has no prior month to compare against,
    so its growth is correctly left as NaN rather than 0 -- reporting 0%
    growth for the first month would be factually wrong (undefined, not zero).
    """
    monthly = (
        df.set_index("order_date")
        .resample("ME")
        .agg(total_sales=("sales", "sum"), total_profit=("profit", "sum"), order_count=("order_id", "nunique"))
    )
    monthly["mom_growth_pct"] = (monthly["total_sales"].pct_change() * 100).round(2)
    monthly = monthly.round(2)
    return monthly


def compute_top_products(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Top products by total sales -- answers 'which product generated the
    highest revenue?', a question our future AI Assistant (Module 12) needs
    to be able to answer directly from this kind of pre-computed table.
    """
    top = (
        df.groupby(["product_id", "product_name", "category"])
        .agg(total_sales=("sales", "sum"), total_profit=("profit", "sum"), units_sold=("quantity", "sum"))
        .round(2)
        .sort_values("total_sales", ascending=False)
        .head(top_n)
        .reset_index()
    )
    return top


def compute_regional_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """Regional sales share -- what % of total revenue comes from each region."""
    total_sales = df["sales"].sum()
    regional = (
        df.groupby("region")
        .agg(total_sales=("sales", "sum"), total_profit=("profit", "sum"), order_count=("order_id", "nunique"))
        .round(2)
    )
    regional["sales_share_pct"] = (regional["total_sales"] / total_sales * 100).round(2)
    return regional.sort_values("total_sales", ascending=False)


def build_report(kpis: dict, monthly: pd.DataFrame, regional: pd.DataFrame, top_products: pd.DataFrame) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("BUSINESS ANALYTICS -- KPI SUMMARY REPORT")
    lines.append("=" * 70)
    lines.append("")
    lines.append("-- Headline KPIs --")
    lines.append(f"  Total Sales                : Rs {kpis['total_sales']:,.2f}")
    lines.append(f"  Total Profit               : Rs {kpis['total_profit']:,.2f}")
    lines.append(f"  Overall Profit Margin      : {kpis['profit_margin_pct']:.2f}%")
    lines.append(f"  Total Orders                : {kpis['total_orders']:,}")
    lines.append(f"  Total Customers             : {kpis['total_customers']:,}")
    lines.append(f"  Average Order Value (AOV)   : Rs {kpis['average_order_value']:,.2f}")
    lines.append(
        f"  Customer Lifetime Value (CLV): Rs {kpis['customer_lifetime_value_simple']:,.2f}  "
        f"[NOTE: simple historical average = total sales / total customers over the "
        f"observed period. Not a predictive/future CLV model.]"
    )
    lines.append(f"  Repeat Customers            : {kpis['repeat_customers']:,} ({kpis['repeat_customer_rate_pct']:.2f}%)")
    lines.append("")
    lines.append("-- Monthly Trend (last 6 months) --")
    lines.append(monthly.tail(6).to_string())
    lines.append("")
    lines.append("-- Regional Performance --")
    lines.append(regional.to_string())
    lines.append("")
    lines.append(f"-- Top {len(top_products)} Products by Sales --")
    lines.append(top_products.to_string(index=False))
    lines.append("")
    lines.append("-- Business Interpretation --")
    best_month = monthly["total_sales"].idxmax()
    lines.append(
        f"  Peak sales month was {best_month.strftime('%B %Y')}, consistent with the "
        f"festive season pattern identified in Module 6's EDA."
    )
    top_region = regional.index[0]
    lines.append(
        f"  {top_region} contributes the largest regional share at "
        f"{regional.loc[top_region, 'sales_share_pct']:.1f}% of total sales."
    )
    lines.append(
        f"  A repeat customer rate of {kpis['repeat_customer_rate_pct']:.1f}% indicates "
        f"{'strong' if kpis['repeat_customer_rate_pct'] > 50 else 'moderate' if kpis['repeat_customer_rate_pct'] > 25 else 'weak'} "
        f"customer retention -- a key input for any customer segmentation work in Module 10."
    )
    lines.append("=" * 70)
    return "\n".join(lines)


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = load_data()

    kpis = compute_headline_kpis(df)
    monthly = compute_monthly_trend(df)
    regional = compute_regional_kpis(df)
    top_products = compute_top_products(df)

    monthly.to_csv(MONTHLY_TREND_PATH)
    logger.info("Saved monthly KPI trend to %s.", MONTHLY_TREND_PATH)

    top_products.to_csv(TOP_PRODUCTS_PATH, index=False)
    logger.info("Saved top products to %s.", TOP_PRODUCTS_PATH)

    report_text = build_report(kpis, monthly, regional, top_products)
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    logger.info("Saved KPI summary report to %s.", REPORT_PATH)

    print("\n" + report_text)


if __name__ == "__main__":
    main()
