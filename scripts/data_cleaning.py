"""
data_cleaning.py

Purpose
-------
Cleans the raw retail sales dataset (data/raw/retail_sales_raw.csv) and
produces an analysis-ready dataset (data/cleaned/retail_sales_cleaned.csv),
along with a human-readable cleaning report (reports/data_cleaning_report.txt).

Design Philosophy
------------------
Every cleaning decision here is deliberate and documented, not automatic.
Real data cleaning is a series of justified business assumptions, not
blind dropna()/fillna() calls. Where an assumption is made (e.g., treating
a missing discount as 0.0), it is logged explicitly so an analyst reviewing
this pipeline later can question or override it.

Usage
-----
    python scripts/data_cleaning.py

Input
-----
    data/raw/retail_sales_raw.csv

Output
------
    data/cleaned/retail_sales_cleaned.csv
    reports/data_cleaning_report.txt
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/retail_sales_raw.csv")
CLEANED_PATH = Path("data/cleaned/retail_sales_cleaned.csv")
REPORT_PATH = Path("reports/data_cleaning_report.txt")

CATEGORICAL_COLUMNS = [
    "ship_mode", "segment", "country", "city", "state",
    "region", "category", "sub_category",
]
DATE_COLUMNS = ["order_date", "ship_date"]
ID_COLUMNS = ["order_id", "customer_id", "product_id"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class CleaningReport:
    """Accumulates a record of every cleaning decision made.

    Why a dataclass instead of just print statements? Because this object
    becomes the actual audit trail -- it gets written to reports/ as a
    permanent, reviewable artifact, exactly like a real data team would
    produce for stakeholders or for their own future reference.
    """
    original_shape: tuple[int, int] = (0, 0)
    final_shape: tuple[int, int] = (0, 0)
    missing_before: dict = field(default_factory=dict)
    missing_filled: dict = field(default_factory=dict)
    duplicates_removed: int = 0
    dtype_conversions: list[str] = field(default_factory=list)
    business_rule_violations: dict = field(default_factory=dict)
    outliers_detected: dict = field(default_factory=dict)

    def to_text(self) -> str:
        lines = []
        lines.append("=" * 70)
        lines.append("DATA CLEANING REPORT")
        lines.append("=" * 70)
        lines.append(f"Original shape : {self.original_shape[0]} rows x {self.original_shape[1]} cols")
        lines.append(f"Final shape    : {self.final_shape[0]} rows x {self.final_shape[1]} cols")
        lines.append("")
        lines.append("-- Missing Values (before cleaning) --")
        for col, count in self.missing_before.items():
            lines.append(f"  {col}: {count} missing")
        lines.append("")
        lines.append("-- Missing Value Handling Decisions --")
        for col, decision in self.missing_filled.items():
            lines.append(f"  {col}: {decision}")
        lines.append("")
        lines.append(f"-- Duplicates Removed: {self.duplicates_removed} rows --")
        lines.append("")
        lines.append("-- Data Type Conversions --")
        for note in self.dtype_conversions:
            lines.append(f"  {note}")
        lines.append("")
        lines.append("-- Business Rule Validation --")
        if self.business_rule_violations:
            for rule, count in self.business_rule_violations.items():
                lines.append(f"  VIOLATION - {rule}: {count} rows affected")
        else:
            lines.append("  No violations found.")
        lines.append("")
        lines.append("-- Outlier Detection (reported, NOT removed) --")
        for col, info in self.outliers_detected.items():
            lines.append(f"  {col}: {info}")
        lines.append("=" * 70)
        return "\n".join(lines)


def load_data(path: Path) -> pd.DataFrame:
    """Load the raw CSV and fail loudly with a clear message if it's missing.

    Failing loudly (rather than letting pandas throw a generic FileNotFoundError)
    is a small but real production practice -- it tells the next engineer
    exactly what to do to fix it, rather than making them guess.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}. Run scripts/generate_data.py first."
        )
    logger.info("Loading raw data from %s...", path)
    df = pd.read_csv(path)
    logger.info("Loaded %d rows, %d columns.", *df.shape)
    return df


def handle_missing_values(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Fill missing values using explicit, documented business assumptions.

    We never use a blanket df.fillna(0) or df.dropna() -- each column gets
    its own justified decision, recorded in the report.
    """
    df = df.copy()

    missing_counts = df.isnull().sum()
    report.missing_before = {col: int(n) for col, n in missing_counts.items() if n > 0}

    if "customer_name" in df.columns and df["customer_name"].isnull().any():
        n = int(df["customer_name"].isnull().sum())
        df["customer_name"] = df["customer_name"].fillna("Guest Customer")
        report.missing_filled["customer_name"] = (
            f"Filled {n} missing values with 'Guest Customer' "
            f"(customer_id remains valid; treated as an incomplete guest checkout)"
        )

    if "discount" in df.columns and df["discount"].isnull().any():
        n = int(df["discount"].isnull().sum())
        df["discount"] = df["discount"].fillna(0.0)
        report.missing_filled["discount"] = (
            f"Filled {n} missing values with 0.0 "
            f"(ASSUMPTION: missing discount means no discount was applied -- "
            f"this should be validated against source system behavior in a real deployment)"
        )

    return df


def remove_duplicates(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Remove exact duplicate rows.

    We only remove EXACT full-row duplicates, never duplicates based on a
    partial key like order_id alone -- an order can legitimately have
    multiple line items, so order_id repeating is normal, not a duplicate.
    """
    before = len(df)
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    removed = before - len(df)
    report.duplicates_removed = removed
    if removed:
        logger.info("Removed %d exact duplicate rows.", removed)
    return df


def convert_data_types(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Convert columns to their proper, memory-efficient data types.

    Why this matters: dates stored as strings can't be used in time-series
    analysis (Module 6, Module 9) without conversion. Categorical columns
    stored as generic 'object' dtype use far more memory than the pandas
    'category' dtype, which matters once datasets scale into the millions
    of rows in a real production setting.
    """
    df = df.copy()

    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            report.dtype_conversions.append(f"{col}: object -> datetime64")

    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("category")
            report.dtype_conversions.append(f"{col}: object -> category")

    for col in ID_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("string")
            report.dtype_conversions.append(f"{col}: object -> string")

    numeric_cols = ["sales", "discount", "profit"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    if "quantity" in df.columns:
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").astype("Int64")

    return df


def validate_business_rules(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Check (and fix, where safe) violations of known business rules.

    We check rather than blindly trust the data, even though we generated
    it ourselves -- this is the habit that matters, since in a real job
    you will NEVER trust incoming data by default, synthetic or not.
    """
    df = df.copy()
    violations = {}

    if {"order_date", "ship_date"}.issubset(df.columns):
        bad_ship = df["ship_date"] < df["order_date"]
        violations["ship_date before order_date"] = int(bad_ship.sum())
        if bad_ship.any():
            # Fix: set ship_date = order_date + 1 day as a safe default
            df.loc[bad_ship, "ship_date"] = df.loc[bad_ship, "order_date"] + pd.Timedelta(days=1)

    if "sales" in df.columns:
        bad_sales = df["sales"] <= 0
        violations["sales <= 0"] = int(bad_sales.sum())
        df = df[~bad_sales]

    if "quantity" in df.columns:
        bad_qty = df["quantity"] <= 0
        violations["quantity <= 0"] = int(bad_qty.sum())
        df = df[~bad_qty]

    report.business_rule_violations = {k: v for k, v in violations.items() if v > 0}
    return df.reset_index(drop=True)


def detect_outliers(df: pd.DataFrame, report: CleaningReport) -> None:
    """Detect (but do NOT remove) outliers in sales and profit using the IQR method.

    Why detect without removing? A very large order or a large discount-driven
    loss is a genuine business event -- removing it would hide exactly the
    kind of finding a real business analytics module (Module 7) is supposed
    to surface. We only report outliers here for awareness; any removal
    decision would need a specific business reason, made explicitly, not
    an automatic statistical cutoff.
    """
    for col in ["sales", "profit"]:
        if col not in df.columns:
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        n_outliers = int(outlier_mask.sum())
        pct = round(n_outliers / len(df) * 100, 2)
        report.outliers_detected[col] = (
            f"{n_outliers} rows ({pct}%) outside [{round(lower_bound, 2)}, "
            f"{round(upper_bound, 2)}] using IQR method -- kept in dataset, "
            f"flagged for business review"
        )


def save_outputs(df: pd.DataFrame, report: CleaningReport) -> None:
    """Write the cleaned dataset and the cleaning report to disk."""
    CLEANED_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(CLEANED_PATH, index=False)
    logger.info("Saved cleaned dataset to %s (%d rows, %d cols).", CLEANED_PATH, *df.shape)

    REPORT_PATH.write_text(report.to_text(), encoding="utf-8")
    logger.info("Saved cleaning report to %s.", REPORT_PATH)


def main() -> None:
    report = CleaningReport()

    df = load_data(RAW_PATH)
    report.original_shape = df.shape

    df = handle_missing_values(df, report)
    df = remove_duplicates(df, report)
    df = convert_data_types(df, report)
    df = validate_business_rules(df, report)
    detect_outliers(df, report)

    report.final_shape = df.shape
    save_outputs(df, report)

    logger.info("Data cleaning complete.")
    print("\n" + report.to_text())


if __name__ == "__main__":
    main()
