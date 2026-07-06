"""
visualization.py

Purpose
-------
Builds interactive, presentation-ready charts using Plotly, distinct from
Module 6's exploratory matplotlib charts. Each function here returns a
Plotly Figure object -- this is deliberate: these same functions get
imported directly by the Flask app in Module 13 to render charts on the
web dashboard, rather than being a run-once script that only saves images.

Usage
-----
As a library (how Module 13's Flask app will use it):

    from visualization import chart_monthly_sales_trend
    fig = chart_monthly_sales_trend(df)
    fig.show()  # or fig.to_html(), or embed via Flask/Dash

Standalone (for this module's own verification):

    python scripts/visualization.py

    This generates reports/dashboard_preview.html -- a single HTML file
    with every chart, so you can open it in a browser and check everything
    renders correctly before Module 13 wires it into Flask.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

DATA_PATH = Path("data/cleaned/retail_sales_features.csv")
PREVIEW_PATH = Path("reports/dashboard_preview.html")

# A consistent color palette across every chart -- small detail, but it's
# what makes a set of charts look like ONE dashboard instead of five
# unrelated plots slapped together.
CATEGORY_COLORS = {
    "Furniture": "#4C72B0",
    "Office Supplies": "#55A868",
    "Technology": "#C44E52",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found. Run scripts/feature_engineering.py first.")
    df = pd.read_csv(DATA_PATH, parse_dates=["order_date", "ship_date"])
    return df


def chart_monthly_sales_trend(df: pd.DataFrame) -> go.Figure:
    """Interactive line chart: monthly sales and profit over time.

    Uses a secondary y-axis for profit, since profit's scale (thousands)
    is much smaller than sales (hundreds of thousands) -- plotting them
    on the same axis would flatten profit into an invisible line.
    """
    monthly = (
        df.set_index("order_date")
        .resample("ME")
        .agg(total_sales=("sales", "sum"), total_profit=("profit", "sum"))
        .reset_index()
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly["order_date"], y=monthly["total_sales"],
        name="Total Sales", mode="lines+markers", line=dict(color="#4C72B0", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=monthly["order_date"], y=monthly["total_profit"],
        name="Total Profit", mode="lines+markers", line=dict(color="#C44E52", width=2),
        yaxis="y2",
    ))
    fig.update_layout(
        title="Monthly Sales & Profit Trend",
        xaxis_title="Month",
        yaxis=dict(title="Total Sales (Rs)"),
        yaxis2=dict(title="Total Profit (Rs)", overlaying="y", side="right"),
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def chart_sales_by_category(df: pd.DataFrame) -> go.Figure:
    """Interactive bar chart: total sales by category, colored consistently."""
    summary = df.groupby("category", as_index=False)["sales"].sum().sort_values("sales", ascending=True)
    fig = px.bar(
        summary, x="sales", y="category", orientation="h",
        color="category", color_discrete_map=CATEGORY_COLORS,
        title="Total Sales by Category",
        labels={"sales": "Total Sales (Rs)", "category": ""},
        template="plotly_white",
    )
    fig.update_layout(showlegend=False)
    return fig


def chart_regional_share(df: pd.DataFrame) -> go.Figure:
    """Interactive donut chart: regional share of total sales.

    A donut (pie with a hole) rather than a plain pie -- purely a design
    choice that reads slightly cleaner on a dashboard, with the same
    underlying data and interactivity (hover for exact %, click to isolate
    a slice in the legend).
    """
    regional = df.groupby("region", as_index=False)["sales"].sum()
    fig = px.pie(
        regional, values="sales", names="region", hole=0.4,
        title="Regional Share of Total Sales",
        template="plotly_white",
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return fig


def chart_top_products(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Interactive horizontal bar chart: top N products by total sales."""
    top = (
        df.groupby("product_name", as_index=False)
        .agg(total_sales=("sales", "sum"), category=("category", "first"))
        .sort_values("total_sales", ascending=True)
        .tail(top_n)
    )
    fig = px.bar(
        top, x="total_sales", y="product_name", orientation="h",
        color="category", color_discrete_map=CATEGORY_COLORS,
        title=f"Top {top_n} Products by Sales",
        labels={"total_sales": "Total Sales (Rs)", "product_name": ""},
        template="plotly_white",
    )
    return fig


def chart_discount_vs_margin(df: pd.DataFrame, sample_size: int = 2000) -> go.Figure:
    """Interactive scatter: discount vs profit margin %, colored by category.

    Sampled for the same reason as Module 6 -- with 8000+ points, an
    unsampled scatter becomes a solid, unreadable blob. Interactivity
    (zoom, hover for exact values) partially offsets this, but sampling
    still keeps the chart legible at first glance.
    """
    sample = df.sample(min(sample_size, len(df)), random_state=42)
    fig = px.scatter(
        sample, x="discount", y="profit_margin_pct", color="category",
        color_discrete_map=CATEGORY_COLORS,
        title="Discount vs Profit Margin % (sampled)",
        labels={"discount": "Discount", "profit_margin_pct": "Profit Margin %"},
        template="plotly_white",
        opacity=0.6,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="black", annotation_text="Break-even")
    return fig


def chart_sales_heatmap(df: pd.DataFrame) -> go.Figure:
    """Interactive heatmap: sales by category (rows) x month (columns).

    This is the visualization equivalent of a pivot table -- exactly the
    kind of chart a business stakeholder would ask for to spot which
    category drives the festive-season spike most strongly.
    """
    df = df.copy()
    df["order_month_name"] = df["order_date"].dt.strftime("%b")
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    pivot = df.pivot_table(
        index="category", columns="order_month_name", values="sales", aggfunc="sum", fill_value=0
    )
    pivot = pivot.reindex(columns=month_order)

    fig = px.imshow(
        pivot, labels=dict(x="Month", y="Category", color="Total Sales (Rs)"),
        title="Sales Heatmap -- Category x Month",
        color_continuous_scale="YlOrRd",
        aspect="auto",
        template="plotly_white",
    )
    return fig


def build_preview_html(df: pd.DataFrame) -> str:
    """Combine all charts into a single standalone HTML file for manual review.

    This is ONLY for this module's own verification -- Module 13's Flask
    app will call the chart_*() functions individually and embed each
    figure into its own page section, not use this combined preview file.
    """
    figs = [
        chart_monthly_sales_trend(df),
        chart_sales_by_category(df),
        chart_regional_share(df),
        chart_top_products(df),
        chart_discount_vs_margin(df),
        chart_sales_heatmap(df),
    ]

    html_parts = ["<html><head><title>Dashboard Preview</title></head><body>"]
    html_parts.append("<h1>Retail Analytics -- Dashboard Preview</h1>")
    for i, fig in enumerate(figs):
        # include_plotlyjs='cdn' on only the first chart avoids loading the
        # ~3MB plotly.js library once per chart -- one shared copy for the page.
        include_js = "cdn" if i == 0 else False
        html_parts.append(fig.to_html(full_html=False, include_plotlyjs=include_js))
    html_parts.append("</body></html>")
    return "\n".join(html_parts)


def main() -> None:
    df = load_data()
    logger.info("Loaded %d rows for visualization.", len(df))

    PREVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    html = build_preview_html(df)
    PREVIEW_PATH.write_text(html, encoding="utf-8")
    logger.info("Saved dashboard preview to %s -- open this in a browser to review.", PREVIEW_PATH)


if __name__ == "__main__":
    main()
