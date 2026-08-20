from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent
P = ROOT / "data" / "processed"

st.set_page_config(page_title="Retail Customer Intelligence", layout="wide", page_icon="📊")

st.title("Retail Customer Intelligence & Retention Analytics")
st.caption("UCI Online Retail Dataset | End-to-end customer analytics pipeline")

if not (P / "clean_transactions.csv").exists():
    st.warning("Pipeline outputs not found. Run `python run_pipeline.py` first.")
    st.stop()

# Load data
df = pd.read_csv(P / "clean_transactions.csv", parse_dates=["InvoiceDate"])
rfm = pd.read_csv(P / "rfm_segments.csv")
clusters = pd.read_csv(P / "customer_clusters.csv")
cohort = pd.read_csv(P / "cohort_retention.csv")
fi = pd.read_csv(P / "feature_importance.csv")
churn = pd.read_csv(P / "churn_dataset.csv") if (P / "churn_dataset.csv").exists() else None

# ============================================================
# KPI HEADER
# ============================================================
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Revenue", f"£{df.Revenue.sum():,.0f}")
col2.metric("Customers", f"{df.CustomerID.nunique():,}")
col3.metric("Orders", f"{df.InvoiceNo.nunique():,}")
col4.metric("AOV", f"£{df.groupby('InvoiceNo').Revenue.sum().mean():,.2f}")
col5.metric("Countries", f"{df.Country.nunique()}")

st.divider()

# ============================================================
# TABS
# ============================================================
tab_overview, tab_segmentation, tab_clustering, tab_retention, tab_churn, tab_recs = st.tabs([
    "Overview", "RFM Segmentation", "Customer Clustering", "Cohort Retention", "Churn Prediction", "Recommendations"
])

# ── TAB 1: OVERVIEW ────────────────────────────────────────
with tab_overview:
    st.subheader("Revenue Trends")

    monthly = (
        df.assign(Month=df.InvoiceDate.dt.to_period("M").astype(str))
        .groupby("Month", as_index=False)
        .agg(Revenue=("Revenue", "sum"), Orders=("InvoiceNo", "nunique"), Customers=("CustomerID", "nunique"))
    )
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=monthly["Month"], y=monthly["Revenue"], name="Revenue (£)", marker_color="steelblue"), secondary_y=False)
    fig.add_trace(go.Scatter(x=monthly["Month"], y=monthly["Customers"], name="Customers", line=dict(color="coral", width=2)), secondary_y=True)
    fig.update_layout(height=400, legend=dict(orientation="h", yanchor="bottom", y=1.02))
    fig.update_yaxes(title_text="Revenue (£)", secondary_y=False)
    fig.update_yaxes(title_text="Unique Customers", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.markdown("**Revenue by Country (Top 10)**")
        co = df.groupby("Country", as_index=False).agg(Revenue=("Revenue", "sum"), Customers=("CustomerID", "nunique"))
        co = co.nlargest(10, "Revenue")
        st.plotly_chart(px.bar(co, x="Revenue", y="Country", orientation="h", color="Revenue",
                               color_continuous_scale="Blues"), use_container_width=True)

    with right:
        st.markdown("**Daily Transaction Volume**")
        daily = df.set_index("InvoiceDate").resample("D").agg(
            Transactions=("InvoiceNo", "nunique"), Revenue=("Revenue", "sum")
        )
        daily["Rev_MA7"] = daily["Revenue"].rolling(7).mean()
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig2.add_trace(go.Bar(x=daily.index, y=daily["Transactions"], name="Transactions", marker_color="lightblue", opacity=0.5), secondary_y=False)
        fig2.add_trace(go.Scatter(x=daily.index, y=daily["Rev_MA7"], name="Revenue (7d MA)", line=dict(color="coral")), secondary_y=True)
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**Top 20 Products by Revenue**")
    top_prods = (
        df.groupby(["StockCode", "Description"], as_index=False)
        .agg(Revenue=("Revenue", "sum"), Qty=("Quantity", "sum"))
        .nlargest(20, "Revenue")
    )
    st.dataframe(top_prods[["Description", "Revenue", "Qty"]].reset_index(drop=True).style.format({"Revenue": "£{:,.2f}", "Qty": "{:,}"}),
                 use_container_width=True, height=400)

# ── TAB 2: RFM SEGMENTATION ────────────────────────────────
with tab_segmentation:
    st.subheader("RFM Segmentation")
    st.markdown("Customers scored on **Recency** (1=recent, 5=old), **Frequency** and **Monetary** (1=low, 5=high), then grouped into 5 segments.")

    col_a, col_b = st.columns(2)
    with col_a:
        seg_counts = rfm["Segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Customers"]
        st.plotly_chart(px.bar(seg_counts, x="Segment", y="Customers", color="Segment",
                               color_discrete_sequence=px.colors.qualitative.Set2), use_container_width=True)

    with col_b:
        seg_rev = rfm.groupby("Segment", observed=False).agg(
            Customers=("CustomerID", "count"),
            Avg_Revenue=("monetary", "mean"),
            Total_Revenue=("monetary", "sum")
        ).reset_index()
        seg_rev["Pct_Revenue"] = (seg_rev["Total_Revenue"] / seg_rev["Total_Revenue"].sum() * 100).round(1)
        st.plotly_chart(px.bar(seg_rev, x="Segment", y="Total_Revenue", color="Segment",
                               color_discrete_sequence=px.colors.qualitative.Set2, text="Pct_Revenue",
                               labels={"Total_Revenue": "Revenue (£)", "Pct_Revenue": "% of Total"}),
                        use_container_width=True)

    st.markdown("**Segment Profiles**")
    seg_profile = rfm.groupby("Segment", observed=False).agg(
        Customers=("CustomerID", "count"),
        Avg_Recency_Days=("recency", "mean"),
        Avg_Frequency=("frequency", "mean"),
        Avg_Revenue=("monetary", "mean"),
        Total_Revenue=("monetary", "sum"),
    ).round(1)
    seg_profile["Pct_Revenue"] = (seg_profile["Total_Revenue"] / seg_profile["Total_Revenue"].sum() * 100).round(1)
    st.dataframe(seg_profile, use_container_width=True)

    st.markdown("**Radar Chart: Segment Comparison**")
    radar_cols = ["recency", "frequency", "monetary"]
    radar_data = rfm.groupby("Segment", observed=False)[radar_cols].mean()
    radar_norm = (radar_data - radar_data.min()) / (radar_data.max() - radar_data.min())
    fig_radar = go.Figure()
    for seg in radar_norm.index:
        vals = radar_norm.loc[seg].tolist()
        vals.append(vals[0])
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=radar_cols + [radar_cols[0]], fill="toself", name=seg
        ))
    fig_radar.update_layout(height=450, polar=dict(radialaxis=dict(visible=True)))
    st.plotly_chart(fig_radar, use_container_width=True)

# ── TAB 3: CLUSTERING ──────────────────────────────────────
with tab_clustering:
    st.subheader("K-Means Customer Clustering")
    n_clusters = clusters["Cluster"].nunique()
    st.info(f"Optimal K = {n_clusters} clusters (selected by silhouette score)")

    clust_profile = clusters.groupby("Cluster").agg(
        Customers=("CustomerID", "count"),
        Avg_Recency=("recency", "mean"),
        Avg_Frequency=("frequency", "mean"),
        Avg_Monetary=("monetary", "mean"),
        Avg_AOV=("avg_order_value", "mean"),
        Total_Revenue=("monetary", "sum"),
    ).round(1)
    clust_profile["Pct_Revenue"] = (clust_profile["Total_Revenue"] / clust_profile["Total_Revenue"].sum() * 100).round(1)
    st.dataframe(clust_profile, use_container_width=True)

    cluster_cols = ["recency", "frequency", "monetary", "avg_order_value", "total_items", "customer_lifetime_days"]
    fig_box = make_subplots(rows=2, cols=3, subplot_titles=cluster_cols)
    for i, col in enumerate(cluster_cols):
        r, c = divmod(i, 3)
        for cl in sorted(clusters["Cluster"].unique()):
            data = clusters[clusters["Cluster"] == cl][col]
            fig_box.add_trace(go.Box(y=data.clip(upper=data.quantile(0.95)), name=f"C{cl}", showlegend=(i == 0)), row=r + 1, col=c + 1)
    fig_box.update_layout(height=600, title="Feature Distributions by Cluster")
    st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("**Cluster Revenue Contribution**")
    fig_pie = px.pie(clust_profile.reset_index(), values="Total_Revenue", names="Cluster",
                     title="Revenue Share by Cluster", hole=0.4)
    st.plotly_chart(fig_pie, use_container_width=True)

# ── TAB 4: COHORT RETENTION ────────────────────────────────
with tab_retention:
    st.subheader("Cohort Retention Analysis")

    cohort_pivot = cohort.pivot(index="cohort_month", columns="cohort_index", values="retention")
    fig_heat = px.imshow(
        cohort_pivot.values,
        labels=dict(x="Months Since First Purchase", y="Cohort Month", color="Retention Rate"),
        x=[str(int(c)) for c in cohort_pivot.columns],
        y=[str(c) for c in cohort_pivot.index],
        color_continuous_scale="YlGnBu",
        aspect="auto",
        text_auto=".0%"
    )
    fig_heat.update_layout(height=600, title="Monthly Cohort Retention Heatmap")
    st.plotly_chart(fig_heat, use_container_width=True)

    avg_ret = cohort.groupby("cohort_index")["retention"].mean().reset_index()
    st.plotly_chart(
        px.bar(avg_ret, x="cohort_index", y="retention", labels={"cohort_index": "Month", "retention": "Avg Retention"},
               title="Average Retention Decay", text_auto=".0%"),
        use_container_width=True
    )

    st.markdown("**Key Insight:** Retention drops sharply in the first 1-2 months, then stabilizes. Post-acquisition engagement in the first 30 days is critical.")

# ── TAB 5: CHURN PREDICTION ────────────────────────────────
with tab_churn:
    st.subheader("Customer Inactivity Prediction")
    st.markdown("Churn is defined temporally: features are built from an observation window (first 75% of dates), and the target is whether the customer makes zero purchases in the remaining 25%.")

    if churn is not None:
        churn_rate = churn["churn"].mean()
        col_x, col_y = st.columns(2)
        with col_x:
            st.metric("Churn Rate", f"{churn_rate:.1%}")
        with col_y:
            st.metric("At-Risk Customers", f"{churn['churn'].sum():,}")

        col_l, col_r = st.columns(2)
        with col_l:
            st.plotly_chart(
                px.pie(churn, names=churn["churn"].map({0: "Active", 1: "Inactive"}).rename("Status"),
                       title="Churn Distribution", hole=0.4, color_discrete_sequence=["steelblue", "coral"]),
                use_container_width=True
            )
        with col_r:
            st.plotly_chart(
                px.bar(fi.sort_values("importance", ascending=True), x="importance", y="feature", orientation="h",
                       title="Feature Importance", color="importance", color_continuous_scale="Blues"),
                use_container_width=True
            )

        st.markdown("**Churned vs Active Customer Profiles**")
        compare_cols = ["recency", "frequency", "monetary"]
        fig_comp = make_subplots(rows=1, cols=3, subplot_titles=[c.title() for c in compare_cols])
        for i, col in enumerate(compare_cols):
            for label, color, grp in [("Active", "steelblue", 0), ("Inactive", "coral", 1)]:
                data = churn[churn["churn"] == grp][col].clip(upper=churn[col].quantile(0.95))
                fig_comp.add_trace(go.Histogram(x=data, name=label, opacity=0.6, marker_color=color, showlegend=(i == 0)), row=1, col=i + 1)
        fig_comp.update_layout(barmode="overlay", height=350)
        st.plotly_chart(fig_comp, use_container_width=True)
    else:
        st.warning("Churn dataset not available.")

# ── TAB 6: RECOMMENDATIONS ─────────────────────────────────
with tab_recs:
    st.subheader("Business Recommendations")

    st.markdown("""
    ### Revenue Intelligence
    - The **top ~20% of customers generate 80% of revenue** (Pareto). Protect this base with dedicated account management and loyalty programs.
    - **UK dominates** revenue. Growth opportunity in Germany, France, and EIRE for international expansion.

    ### Segment-Specific Actions

    | Segment | Action |
    |---------|--------|
    | **Champions** | Reward loyalty; early access to new products; referral programs |
    | **Loyal Customers** | Upsell/cross-sell; membership tiers; personalized recommendations |
    | **Potential Loyalists** | Onboarding sequences; first-purchase follow-ups; volume discounts |
    | **Low Value** | Re-engagement campaigns; limited-time offers; automated email sequences |
    | **At Risk / Lost** | Win-back campaigns with significant incentives; exit surveys |

    ### Churn Prevention
    - **Top predictors**: Recency, frequency, and customer lifetime are the strongest churn signals.
    - **Early warning system**: Customers with declining order frequency over 2-3 months should trigger automated outreach.
    - **ROI optimization**: High-value at-risk customers warrant human intervention; low-value at-risk get automated campaigns.

    ### Retention Tactics
    - Cohort analysis shows **sharp first-month drop-off**. The critical window is the first 30 days.
    - Invest in **post-purchase engagement**: thank-you emails, product tutorials, reorder reminders.
    - **Subscription/loyalty programs** could flatten the retention decay curve.
    """)

    st.divider()
    st.caption("Built with Python, Pandas, Scikit-learn, Streamlit | Data: UCI Online Retail Dataset (CC BY 4.0)")
