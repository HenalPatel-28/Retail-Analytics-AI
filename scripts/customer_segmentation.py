"""
customer_segmentation.py

Purpose
-------
Segments customers into behavioral groups using RFM (Recency, Frequency,
Monetary) analysis combined with K-Means clustering.

Why This Is Different From Module 9's Leakage Concerns
---------------------------------------------------------
Module 9 avoided using a customer's full order history to predict a specific
past point in time -- that would be leakage, since it uses information from
the future relative to the prediction target.

Here, we deliberately use each customer's COMPLETE order history, because
the task is different: this is a retrospective, present-tense snapshot
("as of today, which behavioral group is this customer in?"), not a
prediction about a specific past moment. There is no future to leak from --
we WANT the full picture to describe today's segments accurately.

Output
------
    data/cleaned/customer_segments.csv     -- one row per customer with RFM + segment
    reports/segmentation/elbow_plot.png
    reports/segmentation/silhouette_plot.png
    reports/segmentation/segment_scatter.png
    reports/segmentation_report.txt
    models/kmeans_customer_segments.pkl
    models/rfm_scaler.pkl
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
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("data/cleaned/retail_sales_features.csv")
SEGMENTS_OUTPUT_PATH = Path("data/cleaned/customer_segments.csv")
CHARTS_DIR = Path("reports/segmentation")
REPORT_PATH = Path("reports/segmentation_report.txt")
MODELS_DIR = Path("models")

K_RANGE = range(2, 9)  # candidate cluster counts to evaluate

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)
sns.set_style("whitegrid")


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found. Run scripts/feature_engineering.py first.")
    df = pd.read_csv(DATA_PATH, parse_dates=["order_date"])
    logger.info("Loaded %d rows.", len(df))
    return df


def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Recency, Frequency, Monetary per customer.

    Reference date is one day after the LAST order in the entire dataset --
    standard RFM convention, treating that as 'today' for recency purposes.
    """
    reference_date = df["order_date"].max() + pd.Timedelta(days=1)
    logger.info("Using reference date for recency calculation: %s", reference_date.date())

    rfm = df.groupby("customer_id").agg(
        last_order_date=("order_date", "max"),
        frequency=("order_id", "nunique"),
        monetary=("sales", "sum"),
    ).reset_index()

    rfm["recency_days"] = (reference_date - rfm["last_order_date"]).dt.days
    rfm = rfm.drop(columns=["last_order_date"])

    logger.info("Computed RFM for %d customers.", len(rfm))
    return rfm


def prepare_features(rfm: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """Log-transform (to reduce skew) and scale RFM features.

    Why log-transform first: monetary and frequency are typically heavily
    right-skewed (a few big spenders, many modest ones) -- feeding raw
    skewed values into K-Means (a distance-based algorithm) lets extreme
    values dominate the distance calculation, distorting clusters.

    Why scale at all: recency is in days (range ~0-1500), monetary is in
    rupees (range ~1,000-1,000,000+). Without scaling, K-Means would treat
    a 1000-rupee difference in monetary as equally significant as a
    1000-day difference in recency, which is meaningless -- scaling puts
    all three features on the same footing.
    """
    log_features = pd.DataFrame({
        "recency_log": np.log1p(rfm["recency_days"]),
        "frequency_log": np.log1p(rfm["frequency"]),
        "monetary_log": np.log1p(rfm["monetary"]),
    })

    scaler = StandardScaler()
    scaled = scaler.fit_transform(log_features)
    return scaled, scaler


def find_optimal_k(scaled_features: np.ndarray) -> tuple[dict, dict, int]:
    """Evaluate candidate cluster counts using the elbow method (inertia)
    and silhouette score, then select the k with the highest silhouette score.

    Neither metric alone is fully reliable: inertia always decreases as k
    increases (more clusters can only fit the data better), so the 'elbow'
    is subjective. Silhouette score directly measures how well-separated
    clusters are, giving us an objective number to select on -- we still
    plot the elbow curve too, since it's a standard part of how this
    decision gets communicated and reviewed.
    """
    inertias = {}
    silhouette_scores = {}

    for k in K_RANGE:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(scaled_features)
        inertias[k] = model.inertia_
        silhouette_scores[k] = silhouette_score(scaled_features, labels)
        logger.info("k=%d: inertia=%.1f, silhouette=%.3f", k, inertias[k], silhouette_scores[k])

    best_k = max(silhouette_scores, key=silhouette_scores.get)
    logger.info("Selected k=%d (highest silhouette score: %.3f)", best_k, silhouette_scores[best_k])
    return inertias, silhouette_scores, best_k


def plot_k_selection(inertias: dict, silhouette_scores: dict, best_k: int) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(inertias.keys()), list(inertias.values()), marker="o", color="#4C72B0")
    ax.axvline(best_k, color="red", linestyle="--", label=f"Selected k={best_k}")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Method")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "elbow_plot.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(silhouette_scores.keys()), list(silhouette_scores.values()), marker="o", color="#55A868")
    ax.axvline(best_k, color="red", linestyle="--", label=f"Selected k={best_k}")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Score by k")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "silhouette_plot.png")
    plt.close(fig)

    logger.info("Saved elbow and silhouette plots.")


def fit_final_model(scaled_features: np.ndarray, k: int) -> KMeans:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(scaled_features)
    return model


def label_segments(rfm: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign human-readable business labels to each numeric cluster.

    We label based on each cluster's RELATIVE position (rank) on recency,
    frequency, and monetary compared to other clusters -- not fixed
    thresholds -- since what counts as 'high frequency' is relative to
    this specific business's customer base, not a universal number.

    IMPORTANT: the label sets below are chosen PER-k, not from one fixed
    ordinal list. A generic list like [Champions, Loyal, Potential
    Loyalists, At Risk, ...] breaks down semantically at small k -- e.g.
    with only 3 clusters, "Potential Loyalists" would land on the WORST
    cluster, which makes no business sense. Each k gets a label set sized
    and worded specifically for that number of segments.
    """
    rfm = rfm.copy()
    cluster_profile = rfm.groupby("cluster")[["recency_days", "frequency", "monetary"]].mean()

    cluster_profile["recency_rank"] = cluster_profile["recency_days"].rank(ascending=True)
    cluster_profile["frequency_rank"] = cluster_profile["frequency"].rank(ascending=False)
    cluster_profile["monetary_rank"] = cluster_profile["monetary"].rank(ascending=False)
    cluster_profile["composite_rank"] = (
        cluster_profile["recency_rank"] + cluster_profile["frequency_rank"] + cluster_profile["monetary_rank"]
    )
    cluster_profile = cluster_profile.sort_values("composite_rank")

    n_clusters = len(cluster_profile)

    label_sets = {
        2: ["Champions", "Lost"],
        3: ["Champions", "At Risk", "Lost"],
        4: ["Champions", "Loyal Customers", "At Risk", "Lost"],
        5: ["Champions", "Loyal Customers", "Potential Loyalists", "At Risk", "Lost"],
        6: ["Champions", "Loyal Customers", "Potential Loyalists", "At Risk", "Hibernating", "Lost"],
        7: ["Champions", "Loyal Customers", "Potential Loyalists", "Needs Attention", "At Risk", "Hibernating", "Lost"],
        8: ["Champions", "Loyal Customers", "Potential Loyalists", "New Customers", "Needs Attention", "At Risk", "Hibernating", "Lost"],
    }
    labels_assigned = label_sets.get(n_clusters, [f"Segment {i+1}" for i in range(n_clusters)])

    cluster_to_label = dict(zip(cluster_profile.index, labels_assigned))
    rfm["segment"] = rfm["cluster"].map(cluster_to_label)

    cluster_profile = cluster_profile.assign(segment=[cluster_to_label[i] for i in cluster_profile.index])
    return rfm, cluster_profile


def plot_segment_scatter(rfm: pd.DataFrame) -> None:
    """2D scatter: frequency vs monetary, colored by segment, sized by recency (inverted).

    We use frequency x monetary as the two axes (rather than a 3D plot or
    PCA projection) because these two are the most business-intuitive to
    read directly on a chart -- 'how often' vs 'how much', with recency
    encoded as point size so all three dimensions are still visible.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    size = 300 / (rfm["recency_days"] + 10)
    sns.scatterplot(
        data=rfm, x="frequency", y="monetary", hue="segment", size=size,
        sizes=(20, 200), alpha=0.6, ax=ax, legend="brief",
    )
    ax.set_yscale("log")
    ax.set_title("Customer Segments -- Frequency vs Monetary (point size = recency, bigger = more recent)")
    ax.set_xlabel("Frequency (number of orders)")
    ax.set_ylabel("Monetary (total spend, log scale)")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "segment_scatter.png")
    plt.close(fig)
    logger.info("Saved segment scatter plot.")


def build_report(rfm: pd.DataFrame, cluster_profile: pd.DataFrame, best_k: int, silhouette_scores: dict) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("CUSTOMER SEGMENTATION -- RFM + K-MEANS REPORT")
    lines.append("=" * 70)
    lines.append(f"Total customers segmented: {len(rfm)}")
    lines.append(f"Number of segments (k): {best_k} (selected by highest silhouette score: {silhouette_scores[best_k]:.3f})")
    lines.append("")
    lines.append("-- Segment Profiles (average RFM per segment) --")
    display_cols = ["segment", "recency_days", "frequency", "monetary"]
    lines.append(cluster_profile.reset_index()[display_cols].round(1).to_string(index=False))
    lines.append("")
    lines.append("-- Segment Sizes --")
    lines.append(rfm["segment"].value_counts().to_string())
    lines.append("")
    lines.append("-- Business Recommendations --")
    lines.append("  Champions: highest value, most frequent, most recent -- reward with loyalty perks, early access to new products.")
    lines.append("  Loyal Customers: strong repeat behavior -- upsell/cross-sell opportunities.")
    lines.append("  At Risk / Needs Attention: declining recency despite past value -- targeted win-back campaigns before they churn.")
    lines.append("  Hibernating / Lost: long recency, low frequency -- low-cost re-engagement (email) rather than heavy investment.")
    lines.append("")
    lines.append("-- Important Caveat --")
    lines.append(
        "  This dataset's ~99% repeat-customer rate (noted in Module 7) is a known artifact "
        "of the synthetic data generation method (customers were resampled per order rather "
        "than modeling realistic customer acquisition/churn). In a real dataset with a "
        "genuine mix of one-time and repeat buyers, segment sizes and separation would look "
        "meaningfully different -- treat these specific proportions as illustrative of the "
        "METHOD, not as a real business finding."
    )
    lines.append("=" * 70)
    return "\n".join(lines)


def main() -> None:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    rfm = compute_rfm(df)

    scaled_features, scaler = prepare_features(rfm)
    inertias, silhouette_scores, best_k = find_optimal_k(scaled_features)
    plot_k_selection(inertias, silhouette_scores, best_k)

    final_model = fit_final_model(scaled_features, best_k)
    rfm["cluster"] = final_model.labels_

    rfm, cluster_profile = label_segments(rfm)
    plot_segment_scatter(rfm)

    SEGMENTS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rfm.to_csv(SEGMENTS_OUTPUT_PATH, index=False)
    logger.info("Saved customer segments to %s.", SEGMENTS_OUTPUT_PATH)

    with open(MODELS_DIR / "kmeans_customer_segments.pkl", "wb") as f:
        pickle.dump(final_model, f)
    with open(MODELS_DIR / "rfm_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    logger.info("Saved KMeans model and scaler to %s.", MODELS_DIR)

    report_text = build_report(rfm, cluster_profile, best_k, silhouette_scores)
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    logger.info("Saved segmentation report to %s.", REPORT_PATH)

    print("\n" + report_text)


if __name__ == "__main__":
    main()
