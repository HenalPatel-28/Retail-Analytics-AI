"""
generate_data.py

Purpose
-------
Generates a realistic, synthetic retail sales transaction dataset for the
Retail Sales Analytics & Forecasting Platform project.

Why synthetic data?
--------------------
Real company transactional data is proprietary and rarely available for
learning projects. Instead of generating naive random numbers, this script
encodes real retail business logic:
    - Seasonal sales spikes (festive months, year-end)
    - Category-specific pricing and profit margins
    - Regional distribution weighted like real Indian population/demand patterns
    - Realistic discount-to-profit relationships (heavy discounts can cause losses)

This mirrors a legitimate industry practice: synthetic data generation is used
when real data is sensitive, unavailable, or when testing a pipeline before
production data exists.

Usage
-----
    python generate_data.py

Output
------
    data/raw/retail_sales_raw.csv
"""

from __future__ import annotations

import random
import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42
NUM_ORDERS: int = 5000          # number of distinct orders
START_DATE: date = date(2022, 1, 1)
END_DATE: date = date(2025, 12, 31)
OUTPUT_PATH: Path = Path("data/raw/retail_sales_raw.csv")

# Reproducibility: same seed -> same dataset every time this script runs.
# This is critical in real data engineering — without it, nobody (including
# future you) could reproduce a bug tied to specific data.
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

fake = Faker("en_IN")
Faker.seed(RANDOM_SEED)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reference / lookup data (mirrors how real retail master data is structured)
# ---------------------------------------------------------------------------

REGIONS_STATES_CITIES: dict[str, dict[str, list[str]]] = {
    "North": {
        "Delhi": ["New Delhi"],
        "Punjab": ["Amritsar", "Ludhiana"],
        "Haryana": ["Gurugram", "Faridabad"],
        "Uttar Pradesh": ["Lucknow", "Noida", "Kanpur"],
    },
    "South": {
        "Karnataka": ["Bengaluru", "Mysuru"],
        "Tamil Nadu": ["Chennai", "Coimbatore"],
        "Telangana": ["Hyderabad"],
        "Kerala": ["Kochi", "Thiruvananthapuram"],
    },
    "East": {
        "West Bengal": ["Kolkata", "Howrah"],
        "Odisha": ["Bhubaneswar"],
        "Bihar": ["Patna"],
    },
    "West": {
        "Maharashtra": ["Mumbai", "Pune", "Nagpur"],
        "Gujarat": ["Ahmedabad", "Surat", "Bilimora"],
        "Rajasthan": ["Jaipur", "Udaipur"],
    },
}

# Region demand weighting: West and South are historically the largest
# retail/e-commerce markets in India, so we weight sampling accordingly
# rather than treating all regions as equally likely — this is the kind
# of realistic bias real data actually has.
REGION_WEIGHTS: dict[str, float] = {"North": 0.22, "South": 0.30, "East": 0.13, "West": 0.35}

SEGMENTS: list[str] = ["Consumer", "Corporate", "Home Office"]
SEGMENT_WEIGHTS: list[float] = [0.55, 0.30, 0.15]

SHIP_MODES: list[str] = ["Standard Class", "Second Class", "First Class", "Same Day"]
SHIP_MODE_WEIGHTS: list[float] = [0.55, 0.25, 0.15, 0.05]

# category -> (sub_categories, base_price_range, base_margin_range)
# Margins differ realistically by category: Technology carries thinner
# margins than Furniture in most real retail businesses.
CATALOG: dict[str, dict] = {
    "Furniture": {
        "sub_categories": ["Chairs", "Tables", "Bookcases", "Furnishings"],
        "price_range": (800, 25000),
        "margin_range": (0.05, 0.18),
    },
    "Office Supplies": {
        "sub_categories": ["Storage", "Binders", "Paper", "Art", "Labels", "Fasteners"],
        "price_range": (50, 3000),
        "margin_range": (0.10, 0.35),
    },
    "Technology": {
        "sub_categories": ["Phones", "Machines", "Accessories", "Copiers"],
        "price_range": (1000, 60000),
        "margin_range": (0.03, 0.15),
    },
}

PRODUCT_NAME_ADJECTIVES = ["Premium", "Compact", "Deluxe", "Standard", "Pro", "Eco", "Classic"]


def weighted_choice(options: list[str], weights: list[float]) -> str:
    """Pick one option according to given probability weights.

    Using explicit weights (instead of uniform random.choice) is what makes
    this dataset realistic rather than arbitrary — real regions, segments,
    and shipping modes are never equally likely.
    """
    return random.choices(options, weights=weights, k=1)[0]


def random_date_between(start: date, end: date) -> date:
    """Return a random date between start and end (inclusive)."""
    delta_days = (end - start).days
    return start + timedelta(days=random.randint(0, delta_days))


def seasonal_multiplier(order_date: date) -> float:
    """Model realistic seasonal demand spikes.

    Real Indian retail sees strong spikes around:
        - October/November (Diwali, festive shopping season)
        - January (New Year, post-holiday clearance)
    This function returns a multiplier applied to base sales volume so that
    later EDA/forecasting modules have genuine seasonality to discover --
    without this, Prophet and other forecasting models would have nothing
    meaningful to learn in Module 9.
    """
    month = order_date.month
    if month in (10, 11):
        return 1.6
    if month == 1:
        return 1.3
    if month == 12:
        return 1.2
    return 1.0


def generate_customers(n: int) -> pd.DataFrame:
    """Generate a pool of unique customers to be reused across multiple orders.

    Real customers place multiple orders over time -- generating a fixed
    customer pool first (rather than a brand-new fake customer per order)
    is what makes repeat-customer analysis and CLV calculations in later
    modules possible at all.
    """
    customers = []
    for i in range(n):
        region = weighted_choice(list(REGIONS_STATES_CITIES.keys()), list(REGION_WEIGHTS.values()))
        state = random.choice(list(REGIONS_STATES_CITIES[region].keys()))
        city = random.choice(REGIONS_STATES_CITIES[region][state])
        customers.append(
            {
                "customer_id": f"CUST-{i + 1:05d}",
                "customer_name": fake.name(),
                "segment": weighted_choice(SEGMENTS, SEGMENT_WEIGHTS),
                "region": region,
                "state": state,
                "city": city,
                "country": "India",
            }
        )
    return pd.DataFrame(customers)


def generate_products() -> pd.DataFrame:
    """Generate a catalog of products, each tied to a category/sub-category
    with a realistic base price and margin range.
    """
    products = []
    product_counter = 1
    for category, meta in CATALOG.items():
        for sub_category in meta["sub_categories"]:
            # Multiple distinct products per sub-category, like a real catalog
            for _ in range(6):
                adjective = random.choice(PRODUCT_NAME_ADJECTIVES)
                products.append(
                    {
                        "product_id": f"PROD-{product_counter:05d}",
                        "category": category,
                        "sub_category": sub_category,
                        "product_name": f"{adjective} {sub_category[:-1] if sub_category.endswith('s') else sub_category}",
                        "base_price_min": meta["price_range"][0],
                        "base_price_max": meta["price_range"][1],
                        "margin_min": meta["margin_range"][0],
                        "margin_max": meta["margin_range"][1],
                    }
                )
                product_counter += 1
    return pd.DataFrame(products)


def generate_orders(num_orders: int, customers: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Generate order line items.

    Each 'order' becomes one or more line items (an order can contain
    multiple products), which mirrors how real order/order-line tables
    are structured in relational retail databases.
    """
    rows = []
    order_counter = 1

    for _ in range(num_orders):
        order_id = f"ORD-{order_counter:06d}"
        order_date = random_date_between(START_DATE, END_DATE)
        # Shipping always happens after the order date -- 1 to 7 days later,
        # which is realistic for standard retail fulfillment.
        ship_date = order_date + timedelta(days=random.randint(1, 7))
        ship_mode = weighted_choice(SHIP_MODES, SHIP_MODE_WEIGHTS)

        customer = customers.sample(1).iloc[0]

        # Each order has 1-4 line items (realistic basket size)
        num_line_items = random.choices([1, 2, 3, 4], weights=[0.5, 0.3, 0.15, 0.05], k=1)[0]
        chosen_products = products.sample(num_line_items)

        season_mult = seasonal_multiplier(order_date)

        for _, product in chosen_products.iterrows():
            quantity = random.randint(1, 8)
            base_unit_price = random.uniform(product["base_price_min"], product["base_price_max"])
            discount = round(random.choices(
                [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                weights=[0.35, 0.25, 0.2, 0.1, 0.07, 0.03],
                k=1,
            )[0], 2)

            sales = round(base_unit_price * quantity * season_mult * (1 - discount), 2)

            margin = random.uniform(product["margin_min"], product["margin_max"])
            # Heavy discounts erode margin realistically -- can even go negative,
            # which is true to real retail loss-leader behavior.
            effective_margin = margin - discount * 0.6
            profit = round(sales * effective_margin, 2)

            rows.append(
                {
                    "order_id": order_id,
                    "order_date": order_date.isoformat(),
                    "ship_date": ship_date.isoformat(),
                    "ship_mode": ship_mode,
                    "customer_id": customer["customer_id"],
                    "customer_name": customer["customer_name"],
                    "segment": customer["segment"],
                    "country": customer["country"],
                    "city": customer["city"],
                    "state": customer["state"],
                    "region": customer["region"],
                    "product_id": product["product_id"],
                    "category": product["category"],
                    "sub_category": product["sub_category"],
                    "product_name": product["product_name"],
                    "sales": sales,
                    "quantity": quantity,
                    "discount": discount,
                    "profit": profit,
                }
            )

        order_counter += 1

    return pd.DataFrame(rows)


def inject_realistic_data_quality_issues(df: pd.DataFrame) -> pd.DataFrame:
    """Deliberately introduce missing values and duplicates.

    Why on purpose? Module 3 (Data Cleaning) needs real problems to solve.
    A perfectly clean synthetic dataset would make the cleaning module
    pointless -- real retail data ALWAYS has quality issues: missing
    customer names from failed form submissions, duplicate rows from
    system retries, etc.
    """
    df = df.copy()
    rng = np.random.default_rng(RANDOM_SEED)

    # ~1.5% missing customer_name (simulates incomplete guest checkouts)
    missing_idx = rng.choice(df.index, size=int(len(df) * 0.015), replace=False)
    df.loc[missing_idx, "customer_name"] = np.nan

    # ~1% missing discount (simulates missing data entry)
    missing_idx = rng.choice(df.index, size=int(len(df) * 0.01), replace=False)
    df.loc[missing_idx, "discount"] = np.nan

    # ~0.5% duplicate rows (simulates system double-submission)
    dup_sample = df.sample(int(len(df) * 0.005), random_state=RANDOM_SEED)
    df = pd.concat([df, dup_sample], ignore_index=True)

    return df


def main() -> None:
    logger.info("Generating customer pool...")
    customers = generate_customers(n=800)

    logger.info("Generating product catalog...")
    products = generate_products()
    logger.info("Catalog size: %d products across %d categories", len(products), products["category"].nunique())

    logger.info("Generating %d orders...", NUM_ORDERS)
    orders = generate_orders(NUM_ORDERS, customers, products)
    logger.info("Generated %d order line items.", len(orders))

    logger.info("Injecting realistic data quality issues for Module 3...")
    orders = inject_realistic_data_quality_issues(orders)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    orders.to_csv(OUTPUT_PATH, index=False)
    logger.info("Saved raw dataset to %s (%d rows, %d columns)", OUTPUT_PATH, *orders.shape)


if __name__ == "__main__":
    main()
