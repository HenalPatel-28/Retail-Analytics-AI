"""
feature_engineering.py

Purpose
-------
Reads the normalized sales data (via the vw_order_summary view in MySQL)
and engineers a set of derived features used by later modules: EDA
(Module 6), business KPIs (Module 7), and ML forecasting/segmentation
(Modules 9-10).

Output
------
    data/cleaned/retail_sales_features.csv

A Note on Data Leakage
-----------------------
This script deliberately builds BOTH a leakage-safe and a leakage-prone
version of a customer-order-count feature, to make the distinction concrete:

    customer_total_orders_to_date  -- SAFE: cumulative count using only
                                       orders up to and including the
                                       current row's order_date. Any ML
                                       model trained on this could have
                                       computed the same value in real time
                                       at the moment the order was placed.

    customer_lifetime_order_count  -- LEAKY: total count using ALL of that
                                       customer's orders, including ones
                                       that happened AFTER the current row.
                                       A model trained on this is secretly
                                       being told information from the
                                       future relative to the row it's
                                       looking at.

We keep both, clearly labeled, so you can see the difference directly in
the output file rather than just reading about it abstractly.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from load_to_mysql import get_engine  # reuse the Module 4 connection logic

OUTPUT_PATH = Path("data/cleaned/retail_sales_features.csv")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_from_database() -> pd.DataFrame:
    """Pull the joined, analysis-ready view straight from MySQL.

    Reading from vw_order_summary (rather than re-joining CSVs in pandas)
    demonstrates a realistic pattern: once data lives in a proper relational
    database, downstream analytical scripts query it directly instead of
    re-implementing the joins themselves.
    """
    engine = get_engine()
    logger.info("Reading data from vw_order_summary...")
    df = pd.read_sql("SELECT * FROM vw_order_summary", con=engine, parse_dates=["order_date", "ship_date"])
    logger.info("Loaded %d rows, %d columns from the database.", *df.shape)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive calendar-based features from order_date.

    These are the foundation for nearly all time-based analysis in later
    modules -- Module 6's EDA and Module 9's Prophet forecasting both
    depend on being able to group by month/quarter/weekday directly,
    rather than re-parsing dates every time.
    """
    df = df.copy()
    df["order_year"] = df["order_date"].dt.year
    df["order_month"] = df["order_date"].dt.month
    df["order_quarter"] = df["order_date"].dt.quarter
    df["order_day_of_week"] = df["order_date"].dt.day_name()
    df["is_weekend"] = df["order_date"].dt.dayofweek.isin([5, 6])

    # This directly encodes the seasonal business logic we built into
    # Module 2's data generator (Oct/Nov Diwali spike) as an explicit,
    # queryable flag -- rather than making every later module re-derive
    # "which months count as festive season" from scratch.
    df["is_festive_season"] = df["order_month"].isin([10, 11])

    return df


def add_shipping_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive fulfillment-speed features."""
    df = df.copy()
    df["days_to_ship"] = (df["ship_date"] - df["order_date"]).dt.days
    return df


def add_financial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive normalized financial metrics.

    unit_price and profit_margin_pct let us compare a Rs 200 stationery
    item and a Rs 50,000 laptop on equal footing -- raw 'sales' and
    'profit' values alone can't support that kind of comparison.
    """
    df = df.copy()
    df["unit_price"] = (df["sales"] / df["quantity"]).round(2)
    df["profit_margin_pct"] = (df["profit"] / df["sales"] * 100).round(2)

    # Business-friendly discount tiers instead of raw decimals -- this is
    # the kind of transformation that makes a chart in Module 8 readable
    # to a non-technical stakeholder ("High discount orders" vs "0.4").
    def bucket_discount(discount: float) -> str:
        if discount == 0:
            return "No Discount"
        elif discount <= 0.2:
            return "Low (1-20%)"
        elif discount <= 0.35:
            return "Medium (21-35%)"
        else:
            return "High (36%+)"

    df["discount_bucket"] = df["discount"].apply(bucket_discount)
    return df


def add_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive customer-behavior features, clearly separating leakage-safe
    from leakage-prone versions.

    IMPORTANT: df must be sorted by order_date within each customer for the
    cumulative (leakage-safe) calculations to be correct -- an unsorted
    cumulative count would be meaningless.
    """
    df = df.copy()
    df = df.sort_values(["customer_id", "order_date"]).reset_index(drop=True)

    # customer_first_order_date: the earliest order date for each customer,
    # computed once and broadcast back to every row for that customer.
    first_order = df.groupby("customer_id")["order_date"].min().rename("customer_first_order_date")
    df = df.merge(first_order, on="customer_id", how="left")
    df["customer_tenure_days"] = (df["order_date"] - df["customer_first_order_date"]).dt.days

    # LEAKAGE-SAFE: cumulative distinct order count up to and including
    # the CURRENT order. At the moment this order was placed, this exact
    # number was knowable -- nothing from the future is included.
    unique_orders = (
        df[["customer_id", "order_id", "order_date"]]
        .drop_duplicates()
        .sort_values(["customer_id", "order_date"])
    )
    unique_orders["customer_total_orders_to_date"] = unique_orders.groupby("customer_id").cumcount() + 1
    df = df.merge(
        unique_orders[["customer_id", "order_id", "customer_total_orders_to_date"]],
        on=["customer_id", "order_id"],
        how="left",
    )

    # LEAKAGE-PRONE (deliberately included for teaching purposes):
    # total number of orders that customer EVER placed, including orders
    # that happened after the current row's order_date. Using this as a
    # predictive feature would let a model "see the future."
    lifetime_counts = (
        df[["customer_id", "order_id"]]
        .drop_duplicates()
        .groupby("customer_id")
        .size()
        .rename("customer_lifetime_order_count")
    )
    df = df.merge(lifetime_counts, on="customer_id", how="left")

    df["is_repeat_customer"] = df["customer_lifetime_order_count"] > 1

    return df


def add_order_level_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive features that summarize the whole order (across its line items).

    order_total_sales looks at ALL line items sharing the same order_id --
    this is different from the line-item-level 'sales' column, and is useful
    for basket-size analysis in Module 7.
    """
    df = df.copy()
    order_totals = df.groupby("order_id")["sales"].sum().rename("order_total_sales")
    df = df.merge(order_totals, on="order_id", how="left")

    def size_category(total: float) -> str:
        if total < 2000:
            return "Small"
        elif total < 15000:
            return "Medium"
        else:
            return "Large"

    df["order_size_category"] = df["order_total_sales"].apply(size_category)
    return df


def save_features(df: pd.DataFrame) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    logger.info("Saved engineered feature set to %s (%d rows, %d columns).", OUTPUT_PATH, *df.shape)


def main() -> None:
    df = load_from_database()

    df = add_time_features(df)
    df = add_shipping_features(df)
    df = add_financial_features(df)
    df = add_customer_features(df)
    df = add_order_level_features(df)

    logger.info("Feature engineering complete. Final shape: %d rows, %d columns.", *df.shape)
    logger.warning(
        "NOTE: 'customer_lifetime_order_count' and 'is_repeat_customer' are LEAKAGE-PRONE "
        "features -- do NOT use them as predictors in Module 9's forecasting model. "
        "Use 'customer_total_orders_to_date' instead for anything predictive."
    )

    save_features(df)


if __name__ == "__main__":
    main()
