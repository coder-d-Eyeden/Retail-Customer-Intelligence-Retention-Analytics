from pathlib import Path
import pandas as pd
import numpy as np

from .data import PROC


def customer_features(df: pd.DataFrame, end_date=None) -> pd.DataFrame:
    """Build customer-level features from transaction data.

    Features per customer:
        - last_purchase, first_purchase: datetime bounds
        - frequency: number of unique invoices
        - monetary: total revenue
        - total_items: total quantity purchased
        - avg_order_value: mean revenue per invoice
        - recency: days since last purchase (relative to end_date)
        - customer_lifetime_days: span between first and last purchase
        - avg_days_between_orders: mean inter-purchase gap
    """
    end_date = end_date or df["InvoiceDate"].max() + pd.Timedelta(days=1)

    orders = (
        df.groupby(["CustomerID", "InvoiceNo"], as_index=False)
        .agg(
            order_date=("InvoiceDate", "min"),
            order_value=("Revenue", "sum"),
            items=("Quantity", "sum"),
        )
    )

    g = (
        orders.groupby("CustomerID")
        .agg(
            last_purchase=("order_date", "max"),
            first_purchase=("order_date", "min"),
            frequency=("InvoiceNo", "nunique"),
            monetary=("order_value", "sum"),
            total_items=("items", "sum"),
            avg_order_value=("order_value", "mean"),
        )
        .reset_index()
    )

    g["recency"] = (end_date - g["last_purchase"]).dt.days
    g["customer_lifetime_days"] = (g["last_purchase"] - g["first_purchase"]).dt.days

    gaps = (
        orders.sort_values(["CustomerID", "order_date"])
        .groupby("CustomerID")["order_date"]
        .diff()
        .dt.days
    )
    gap = (
        gaps.groupby(orders.loc[gaps.index, "CustomerID"]).mean().rename("avg_days_between_orders")
    )
    g = g.merge(gap, left_on="CustomerID", right_index=True, how="left")
    g["avg_days_between_orders"] = g["avg_days_between_orders"].fillna(0)

    PROC.mkdir(parents=True, exist_ok=True)
    g.to_csv(PROC / "customer_features.csv", index=False)
    return g


def rfm_segment(c: pd.DataFrame) -> pd.DataFrame:
    """Assign quintile-based RFM scores and named segments.

    Scores R, F, M each range 1-5. Segments:
        - Champion (15-18)
        - Loyal Customer (13-14)
        - Potential Loyalist (10-12)
        - Low Value (7-9)
        - At Risk/Lost (3-6)
    """
    c = c.copy()
    quantiles = c[["recency", "frequency", "monetary"]].quantile([0.2, 0.4, 0.6, 0.8])

    def score_val(value: float, col: str, lower_is_better: bool = False) -> int:
        """Score 1-5 based on which quintile the value falls into."""
        thresholds = quantiles[col].values  # [0.2, 0.4, 0.6, 0.8]
        rank = int(np.sum(value > thresholds)) + 1  # 1-5
        if lower_is_better:
            rank = 6 - rank  # invert: low recency -> high score
        return rank

    c["R"] = c["recency"].apply(lambda v: score_val(v, "recency", lower_is_better=True))
    c["F"] = c["frequency"].apply(lambda v: score_val(v, "frequency"))
    c["M"] = c["monetary"].apply(lambda v: score_val(v, "monetary"))
    c["RFM_Score"] = c[["R", "F", "M"]].sum(axis=1)

    c["Segment"] = pd.cut(
        c["RFM_Score"],
        bins=[0, 7, 10, 12, 15, 18],
        labels=["At Risk/Lost", "Low Value", "Potential Loyalist", "Loyal Customer", "Champion"],
        include_lowest=True,
    )

    PROC.mkdir(parents=True, exist_ok=True)
    c.to_csv(PROC / "rfm_segments.csv", index=False)
    return c


def cohort_retention(df: pd.DataFrame) -> pd.DataFrame:
    """Compute monthly cohort retention rates."""
    x = df.copy()
    x["order_month"] = x["InvoiceDate"].dt.to_period("M")

    first = x.groupby("CustomerID")["order_month"].min().rename("cohort_month")
    x = x.merge(first, on="CustomerID")
    x["cohort_index"] = (
        (x["order_month"].dt.year - x["cohort_month"].dt.year) * 12
        + (x["order_month"].dt.month - x["cohort_month"].dt.month)
        + 1
    )

    active = (
        x.groupby(["cohort_month", "cohort_index"])["CustomerID"]
        .nunique()
        .reset_index()
    )
    base = (
        active[active["cohort_index"] == 1][["cohort_month", "CustomerID"]]
        .rename(columns={"CustomerID": "base"})
    )
    active = active.merge(base, on="cohort_month")
    active["retention"] = active["CustomerID"] / active["base"]

    PROC.mkdir(parents=True, exist_ok=True)
    active.to_csv(PROC / "cohort_retention.csv", index=False)
    return active
