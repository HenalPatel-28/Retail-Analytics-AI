"""
load_to_mysql.py

Purpose
-------
Reads the cleaned flat-file dataset (data/cleaned/retail_sales_cleaned.csv)
and loads it into the normalized MySQL schema defined in database/schema.sql
(customers, products, orders, order_items).

Why not just load the flat CSV directly into one table?
---------------------------------------------------------
That would defeat the entire point of Module 4. We deliberately decompose
the flat file into its normalized components here in Python, using pandas,
before writing to MySQL -- this mirrors exactly what a real ETL (Extract,
Transform, Load) job does: extract from a source, transform shape, load into
a normalized destination.

Usage
-----
    python scripts/load_to_mysql.py

Prerequisites
--------------
    1. database/schema.sql must already be executed against your MySQL server.
    2. data/cleaned/retail_sales_cleaned.csv must exist (run data_cleaning.py first).
    3. .env must contain valid DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, URL

CLEANED_PATH = Path("data/cleaned/retail_sales_cleaned.csv")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def get_engine() -> Engine:
    """Build a SQLAlchemy engine from environment variables in .env.

    Using environment variables (never hardcoded credentials) is a
    non-negotiable security practice -- this is the same .env file we set
    up in Module 1, now actually being put to use.
    """
    load_dotenv()

    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "3306")
    user = os.getenv("DB_USER", "root")
    password = os.getenv("DB_PASSWORD", "")
    database = os.getenv("DB_NAME", "retail_analytics")

    if not password:
        raise ValueError(
            "DB_PASSWORD is not set in .env -- refusing to connect with an empty password."
        )

    connection_url = URL.create(
        drivername="mysql+pymysql",
        username=user,
        password=password,
        host=host,
        port=int(port),
        database=database,
    )
    logger.info("Connecting to MySQL at %s:%s/%s as %s...", host, port, database, user)
    return create_engine(connection_url)


def load_cleaned_data() -> pd.DataFrame:
    if not CLEANED_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned data not found at {CLEANED_PATH}. Run scripts/data_cleaning.py first."
        )
    df = pd.read_csv(CLEANED_PATH, parse_dates=["order_date", "ship_date"])
    logger.info("Loaded %d rows from %s.", len(df), CLEANED_PATH)
    return df


def build_customers_table(df: pd.DataFrame) -> pd.DataFrame:
    """Extract one row per unique customer.

    We take the FIRST occurrence of each customer_id's attributes. In a real
    system with customers who might change address over time, you'd instead
    want a slowly-changing-dimension strategy -- but for this dataset,
    customer attributes are static, so first-occurrence is safe and correct.
    """
    customers = (
        df[["customer_id", "customer_name", "segment", "country", "city", "state", "region"]]
        .drop_duplicates(subset="customer_id", keep="first")
        .reset_index(drop=True)
    )
    logger.info("Built customers table: %d unique customers.", len(customers))
    return customers


def build_products_table(df: pd.DataFrame) -> pd.DataFrame:
    products = (
        df[["product_id", "product_name", "category", "sub_category"]]
        .drop_duplicates(subset="product_id", keep="first")
        .reset_index(drop=True)
    )
    logger.info("Built products table: %d unique products.", len(products))
    return products


def build_orders_table(df: pd.DataFrame) -> pd.DataFrame:
    """Extract one row per unique order (the header).

    Since a single order can have multiple line items (multiple rows in the
    flat file share the same order_id), we deduplicate on order_id -- the
    header-level fields (customer, dates, ship mode) are identical across
    all line items belonging to the same order by construction.
    """
    orders = (
        df[["order_id", "customer_id", "order_date", "ship_date", "ship_mode"]]
        .drop_duplicates(subset="order_id", keep="first")
        .reset_index(drop=True)
    )
    orders["order_date"] = orders["order_date"].dt.date
    orders["ship_date"] = orders["ship_date"].dt.date
    logger.info("Built orders table: %d unique orders.", len(orders))
    return orders


def build_order_items_table(df: pd.DataFrame) -> pd.DataFrame:
    """Extract the line-item facts: one row per product within an order."""
    order_items = df[["order_id", "product_id", "sales", "quantity", "discount", "profit"]].copy()
    logger.info("Built order_items table: %d line items.", len(order_items))
    return order_items


def write_table(engine: Engine, df: pd.DataFrame, table_name: str) -> None:
    """Write a DataFrame to MySQL, appending to the existing schema.

    We use if_exists='append' (not 'replace') because the table structure,
    constraints, and indexes were already created properly by schema.sql --
    letting pandas auto-create the table with to_sql(if_exists='replace')
    would silently drop all our carefully designed keys and constraints.
    """
    df.to_sql(table_name, con=engine, if_exists="append", index=False, method="multi", chunksize=500)
    logger.info("Loaded %d rows into '%s'.", len(df), table_name)


def clear_existing_data(engine: Engine) -> None:
    """Truncate tables before reloading, in dependency-safe order.

    This makes the script idempotent -- running it multiple times gives you
    the same end result instead of duplicate-key errors on the second run.
    We disable foreign key checks temporarily to allow truncation in any
    order, then re-enable them immediately after.
    """
    with engine.begin() as conn:
        conn.execute(text("SET FOREIGN_KEY_CHECKS = 0"))
        for table in ["order_items", "orders", "products", "customers"]:
            conn.execute(text(f"TRUNCATE TABLE {table}"))
        conn.execute(text("SET FOREIGN_KEY_CHECKS = 1"))
    logger.info("Cleared existing data from all tables (idempotent reload).")


def main() -> None:
    engine = get_engine()

    df = load_cleaned_data()

    customers = build_customers_table(df)
    products = build_products_table(df)
    orders = build_orders_table(df)
    order_items = build_order_items_table(df)

    clear_existing_data(engine)

    # Load order matters: customers and products first (referenced tables),
    # then orders (references customers), then order_items (references both).
    # Loading in the wrong order would violate foreign key constraints.
    write_table(engine, customers, "customers")
    write_table(engine, products, "products")
    write_table(engine, orders, "orders")
    write_table(engine, order_items, "order_items")

    logger.info("All tables loaded successfully.")


if __name__ == "__main__":
    main()

