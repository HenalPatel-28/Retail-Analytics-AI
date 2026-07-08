"""
recommendation.py

Purpose
-------
Builds a product recommendation system using Market Basket Analysis --
identifying which products are frequently purchased together, using the
classic support/confidence/lift framework, implemented directly with
pandas rather than a third-party library (baskets here are small --
1-4 items -- so a full Apriori implementation adds dependency risk
without adding real capability).

Output
------
    reports/recommendation/association_rules.csv   -- all qualifying rules
    reports/recommendation_report.txt               -- top findings, plain language
    models/product_recommendations.pkl              -- {product_id: [recommended_product_ids]}
                                                        lookup dict, ready for Module 12's
                                                        AI Assistant and Module 13's Flask app
"""

from __future__ import annotations

import logging
import pickle
from collections import Counter
from itertools import combinations
from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/cleaned/retail_sales_features.csv")
RULES_OUTPUT_PATH = Path("reports/recommendation/association_rules.csv")
REPORT_PATH = Path("reports/recommendation_report.txt")
MODELS_DIR = Path("models")

MIN_SUPPORT = 0.001    # pair must appear in at least 0.1% of all orders
MIN_LIFT = 1.2         # pair must be at least 20% more common than random chance
TOP_N_PER_PRODUCT = 5  # recommendations to store per product

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found. Run scripts/feature_engineering.py first.")
    df = pd.read_csv(DATA_PATH)
    logger.info("Loaded %d rows.", len(df))
    return df


def build_baskets(df: pd.DataFrame) -> tuple[list[list[str]], dict[str, str]]:
    """Group line items into per-order product baskets.

    Also builds a product_id -> product_name lookup, since rules are more
    useful reported by name than by ID alone.
    """
    baskets = (
        df.groupby("order_id")["product_id"]
        .apply(lambda x: sorted(set(x)))  # dedupe: same product twice in one order counts once for basket analysis
        .tolist()
    )
    name_lookup = df.drop_duplicates("product_id").set_index("product_id")["product_name"].to_dict()
    logger.info("Built %d order baskets.", len(baskets))
    return baskets, name_lookup


def compute_association_rules(baskets: list[list[str]]) -> pd.DataFrame:
    """Compute support, confidence, and lift for every product PAIR that
    co-occurs in at least one basket.

    We only consider pairs (not larger itemsets) -- with baskets of at most
    4 items, pairs already capture the vast majority of useful associations,
    and this keeps the implementation simple and fully transparent (no
    black-box library, every number traceable back to a basic count).
    """
    total_orders = len(baskets)

    # Count how many orders each individual product appears in
    product_counts = Counter()
    for basket in baskets:
        for product in basket:
            product_counts[product] += 1

    # Count how many orders each UNORDERED pair appears in together
    pair_counts = Counter()
    for basket in baskets:
        if len(basket) < 2:
            continue
        for pair in combinations(basket, 2):
            pair_counts[pair] += 1

    rules = []
    for (product_a, product_b), co_count in pair_counts.items():
        support = co_count / total_orders
        if support < MIN_SUPPORT:
            continue

        # Confidence and lift are DIRECTIONAL: "given A, how often is B also
        # bought" is a different number from "given B, how often is A also
        # bought" -- we compute and report both directions.
        confidence_a_to_b = co_count / product_counts[product_a]
        confidence_b_to_a = co_count / product_counts[product_b]
        support_b = product_counts[product_b] / total_orders
        support_a = product_counts[product_a] / total_orders
        lift = support / (support_a * support_b)

        if lift < MIN_LIFT:
            continue

        rules.append({
            "product_a": product_a, "product_b": product_b,
            "co_occurrence_count": co_count,
            "support": round(support, 5),
            "confidence_a_to_b": round(confidence_a_to_b, 4),
            "confidence_b_to_a": round(confidence_b_to_a, 4),
            "lift": round(lift, 3),
        })

    rules_df = pd.DataFrame(rules).sort_values("lift", ascending=False).reset_index(drop=True)
    logger.info("Found %d qualifying association rules (min_support=%.4f, min_lift=%.2f).", len(rules_df), MIN_SUPPORT, MIN_LIFT)
    return rules_df


def build_recommendation_lookup(rules_df: pd.DataFrame) -> dict[str, list[str]]:
    """Build a simple {product_id: [recommended product_ids]} dict for
    direct use in Module 12 (AI Assistant) and Module 13 (Flask app) --
    e.g. 'customers who bought this also bought...' on a product page.
    """
    lookup: dict[str, list[tuple[str, float]]] = {}

    for _, row in rules_df.iterrows():
        lookup.setdefault(row["product_a"], []).append((row["product_b"], row["lift"]))
        lookup.setdefault(row["product_b"], []).append((row["product_a"], row["lift"]))

    final_lookup = {}
    for product, candidates in lookup.items():
        # Highest lift first, deduplicated, capped at TOP_N_PER_PRODUCT
        seen = set()
        ranked = []
        for prod_id, lift in sorted(candidates, key=lambda x: x[1], reverse=True):
            if prod_id not in seen:
                ranked.append(prod_id)
                seen.add(prod_id)
            if len(ranked) >= TOP_N_PER_PRODUCT:
                break
        final_lookup[product] = ranked

    return final_lookup


def build_report(rules_df: pd.DataFrame, name_lookup: dict, total_orders: int, total_products: int) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("PRODUCT RECOMMENDATION -- MARKET BASKET ANALYSIS REPORT")
    lines.append("=" * 70)
    lines.append(f"Total orders analyzed: {total_orders}")
    lines.append(f"Total unique products: {total_products}")
    lines.append(f"Qualifying association rules found: {len(rules_df)} (min_support={MIN_SUPPORT}, min_lift={MIN_LIFT})")
    lines.append("")

    if rules_df.empty:
        lines.append(
            "-- No qualifying rules found --\n"
            "  This can legitimately happen with a large, diverse product catalog and small "
            "baskets (1-4 items each): most specific product PAIRS simply don't co-occur often "
            "enough to clear the support/lift thresholds. This is an honest result, not an error -- "
            "consider lowering MIN_SUPPORT/MIN_LIFT, or analyzing at the CATEGORY level instead "
            "of individual product level, if this happens on the real dataset."
        )
    else:
        lines.append("-- Top 10 Rules by Lift --")
        top10 = rules_df.head(10).copy()
        top10["product_a_name"] = top10["product_a"].map(name_lookup)
        top10["product_b_name"] = top10["product_b"].map(name_lookup)
        for _, row in top10.iterrows():
            lines.append(
                f"  {row['product_a_name']} + {row['product_b_name']}: "
                f"lift={row['lift']}, support={row['support']}, "
                f"confidence(A->B)={row['confidence_a_to_b']}, confidence(B->A)={row['confidence_b_to_a']}"
            )
        lines.append("")
        lines.append("-- Interpretation --")
        lines.append(
            "  Lift > 1 means these products are bought together MORE often than random chance "
            "would predict -- these are potential cross-sell opportunities, not just two popular "
            "items that happen to co-occur because both are popular individually."
        )
        lines.append("")
        lines.append("-- IMPORTANT STATISTICAL CAVEAT: Multiple Comparisons --")
        low_support_high_lift = rules_df[(rules_df["co_occurrence_count"] < 10)]
        lines.append(
            f"  {len(low_support_high_lift)} of {len(rules_df)} rules are based on FEWER THAN 10 "
            f"co-occurrences. With many products, we are testing thousands of possible pairs "
            f"simultaneously (e.g. {total_products} products = {total_products * (total_products - 1) // 2} "
            f"possible pairs) -- purely by random chance, some pairs will show elevated lift even with "
            f"ZERO real association, simply because we tested so many. This is the 'multiple comparisons' "
            f"problem. TREAT LOW-CO-OCCURRENCE, HIGH-LIFT RULES WITH CAUTION -- prioritize rules with "
            f"higher co_occurrence_count as more trustworthy, and consider this analysis exploratory "
            f"rather than confirmed business fact until validated against new data or a larger sample."
        )

    lines.append("=" * 70)
    return "\n".join(lines)


def main() -> None:
    RULES_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    baskets, name_lookup = build_baskets(df)

    rules_df = compute_association_rules(baskets)
    rules_df.to_csv(RULES_OUTPUT_PATH, index=False)
    logger.info("Saved association rules to %s.", RULES_OUTPUT_PATH)

    recommendation_lookup = build_recommendation_lookup(rules_df)
    with open(MODELS_DIR / "product_recommendations.pkl", "wb") as f:
        pickle.dump(recommendation_lookup, f)
    logger.info("Saved recommendation lookup for %d products to %s.", len(recommendation_lookup), MODELS_DIR)

    report_text = build_report(rules_df, name_lookup, len(baskets), df["product_id"].nunique())
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    logger.info("Saved recommendation report to %s.", REPORT_PATH)

    print("\n" + report_text)


if __name__ == "__main__":
    main()
