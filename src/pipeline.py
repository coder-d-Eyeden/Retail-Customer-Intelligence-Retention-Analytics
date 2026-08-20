"""Pipeline orchestrator: loads data, builds features, runs models."""

from .data import load_raw, save_clean, PROC, MODEL
from .features import customer_features, rfm_segment, cohort_retention
from .modeling import kmeans_cluster, temporal_churn


def run_pipeline():
    PROC.mkdir(parents=True, exist_ok=True)
    MODEL.mkdir(parents=True, exist_ok=True)

    print("Loading and cleaning transactions...")
    df = load_raw()
    save_clean(df)
    print(f"  Clean transactions: {len(df):,}")

    print("Building customer features...")
    c = customer_features(df)

    print("RFM segmentation...")
    c = rfm_segment(c)

    print("K-Means clustering...")
    c = kmeans_cluster(c)

    print("Cohort retention analysis...")
    cohort_retention(df)

    print("Temporal churn prediction...")
    temporal_churn(df)

    print(f"\nSummary: {len(df):,} transactions; {c.CustomerID.nunique():,} customers; £{df.Revenue.sum():,.2f} revenue")
    print("Done. Run: streamlit run app.py")


if __name__ == "__main__":
    run_pipeline()
