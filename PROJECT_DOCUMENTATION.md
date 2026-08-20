# Retail Customer Intelligence & Retention Analytics

## Complete Project Documentation

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Business Context & Motivation](#2-business-context--motivation)
3. [Dataset](#3-dataset)
4. [Technical Architecture](#4-technical-architecture)
5. [Pipeline Walkthrough](#5-pipeline-walkthrough)
6. [Exploratory Data Analysis](#6-exploratory-data-analysis)
7. [Customer Feature Engineering](#7-customer-feature-engineering)
8. [RFM Segmentation](#8-rfm-segmentation)
9. [K-Means Clustering](#9-k-means-clustering)
10. [Cohort Retention Analysis](#10-cohort-retention-analysis)
11. [Temporal Churn Prediction](#11-temporal-churn-prediction)
12. [Results Summary](#12-results-summary)
13. [Business Recommendations](#13-business-recommendations)
14. [Limitations & Future Work](#14-limitations--future-work)

---

## 1. Project Overview

This project is an end-to-end customer analytics pipeline built on the UCI Online Retail dataset. It answers a single business question: **Which customers matter most, which are at risk of leaving, and what should the business do about it?**

The project was designed to demonstrate the full analytics lifecycle — from raw transactional data through to actionable business intelligence — using Python, SQL, machine learning, and interactive visualization.

### What the project does

| Stage | What happens | Why it matters |
|-------|-------------|----------------|
| Data cleaning | Removes ~149K invalid records from 541,909 raw transactions | Bad data produces bad decisions |
| Feature engineering | Transforms row-level transactions into customer-level summaries | You cannot segment or predict from raw invoices alone |
| RFM segmentation | Scores every customer on Recency, Frequency, Monetary value | Simple, proven framework for customer prioritization |
| K-Means clustering | Discovers natural customer groupings without human-defined rules | Reveals segments RFM rules might miss |
| Cohort retention | Tracks how many customers from each monthly cohort return over time | Shows where the retention funnel breaks |
| Churn prediction | Builds a classification model to predict which customers will become inactive | Enables proactive intervention before revenue is lost |
| Dashboard | Interactive Streamlit app with 6 tabs | Makes insights accessible to non-technical stakeholders |

---

## 2. Business Context & Motivation

### Why customer analytics matters

Acquiring a new customer costs **5–25x more** than retaining an existing one. A 5% increase in customer retention can increase profits by **25–95%** (Bain & Company). Yet most retailers have limited visibility into:

- Who their best customers actually are
- Which customers are about to stop buying
- Where the retention funnel is leaking
- What interventions actually work

This project solves that visibility problem for an online retailer.

### The Accenture AI & Data Internship alignment

This project was built to address the requirements of the Accenture AI & Data Analyst internship, which calls for:

- Python and SQL proficiency
- Exploratory data analysis on real-world data
- Customer segmentation and behavioral analytics
- Predictive modeling with proper validation
- Business recommendations translated from data
- Interactive visualization and dashboarding

### Business questions this project answers

1. **Who are our most valuable customers?** — Pareto analysis shows the top 26% of customers generate 80% of revenue.
2. **What do our customer segments look like?** — RFM and K-Means both identify a high-value core and a large dormant tail.
3. **Are we retaining customers?** — Cohort analysis shows a sharp drop from 100% to 20.6% in the first month, stabilizing around 25%.
4. **Which customers are about to go inactive?** — The churn model achieves 0.732 ROC-AUC, with frequency as the dominant predictor.
5. **What should we do about it?** — Segment-specific strategies prioritize retention spend on high-value at-risk customers.

---

## 3. Dataset

### Source

- **Name:** Online Retail
- **Repository:** UCI Machine Learning Repository
- **DOI:** 10.24432/C5BW33
- **License:** CC BY 4.0
- **Download:** https://archive.ics.uci.edu/dataset/352/online+retail

### Overview

| Attribute | Value |
|-----------|-------|
| Raw records | 541,909 |
| Columns | 8 (InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country) |
| Date range | 1 December 2010 — 9 December 2011 |
| Currency | GBP (British Pounds) |
| Countries | 38 |
| Products | 3,940 unique stock codes |
| Customers | ~4,372 unique (before cleaning) |

### What the data represents

Each row is a single line item on an invoice. Multiple rows share the same InvoiceNo (one per product on that invoice). The data covers a UK-based online retailer selling giftware and homewares, with a heavy emphasis on seasonal and decorative products.

### Why this dataset

- **Real-world scale:** 541K rows is large enough to be meaningful but small enough to process locally.
- **Transaction-level granularity:** Enables RFM, cohort, and behavioral analysis at the customer level.
- **Temporal dimension:** 12 months of data supports time-based modeling and retention analysis.
- **Missing churn labels:** Forces the analyst to define churn — a realistic business scenario.

---

## 4. Technical Architecture

### Project structure

```
retail-customer-intelligence/
├── src/
│   ├── __init__.py          # Package marker
│   ├── data.py              # Data loading, cleaning, path constants
│   ├── features.py          # RFM scoring, customer features, cohort retention
│   ├── modeling.py          # K-Means clustering, temporal churn prediction
│   └── pipeline.py          # Orchestrator that chains all stages
├── app.py                   # Streamlit dashboard (6 tabs)
├── notebooks/
│   └── retail_customer_analysis.ipynb   # Full EDA notebook (34 cells)
├── sql/
│   └── retail_analysis.sql  # 12 analytical SQL queries
├── data/
│   ├── raw/online-retail/   # Source Excel file (22.6 MB)
│   └── processed/           # 7 CSV outputs from pipeline
├── models/
│   └── churn_model.joblib   # Serialized best churn model
├── run_pipeline.py          # Entry point: python run_pipeline.py
├── requirements.txt
└── README.md
```

### Data flow

```
Online Retail.xlsx (541,909 rows)
        │
        ▼
    load_clean() ────────────► clean_transactions.csv (392,692 rows)
        │
        ▼
    customer_features() ─────► customer_features.csv (4,338 customers)
        │
        ├──► rfm_segment() ──► rfm_segments.csv
        │
        ├──► kmeans_cluster() ► customer_clusters.csv
        │
        ├──► cohort_retention() ► cohort_retention.csv (91 rows)
        │
        └──► temporal_churn() ├──► churn_dataset.csv (3,385 customers)
                              ├──► feature_importance.csv
                              └──► models/churn_model.joblib
```

### Tech stack

| Component | Tool | Why |
|-----------|------|-----|
| Language | Python 3.14 | Industry standard for data science |
| Data manipulation | Pandas 3.0 | DataFrame operations on tabular data |
| Numerical computing | NumPy 2.5 | Array operations, log transforms |
| Machine learning | Scikit-learn 1.9 | K-Means, Logistic Regression, Random Forest, StandardScaler |
| Visualization | Matplotlib + Seaborn (notebook), Plotly (dashboard) | Static analysis plots vs. interactive web charts |
| Dashboard | Streamlit 1.61 | Fast Python-native web apps |
| Serialization | Joblib | Save/load trained models |
| Database queries | SQL (SQLite dialect) | Demonstrates SQL proficiency alongside Python |
| Source control | Git | Version history |

### Pipeline commands

```bash
# Run the full pipeline (generates all CSVs + model)
python run_pipeline.py

# Launch the interactive dashboard
streamlit run app.py
```

---

## 5. Pipeline Walkthrough

### 5.1 Data Loading & Cleaning (`src/data.py`)

**What happens:**

The raw Excel file is read into a Pandas DataFrame, then cleaned through a series of filters.

**Step-by-step:**

| Step | Filter | Rows removed | Why |
|------|--------|-------------|-----|
| 1 | Drop exact duplicates | ~6,000 | Duplicate rows inflate revenue and frequency counts |
| 2 | Remove missing CustomerID | ~135,000 | Cannot attribute transactions to customers without an ID |
| 3 | Remove UnitPrice ≤ 0 | minimal | Free items or data entry errors distort monetary analysis |
| 4 | Remove Quantity ≤ 0 | minimal | Negative quantities represent returns, handled separately |
| 5 | Remove invoices starting with 'C' | ~9,000 | Cancelled invoices; counting them would double-count revenue |

**Result:** 392,692 clean transaction records from 541,909 raw (27.5% removed).

**Why these specific filters:**

- Missing CustomerID is the biggest issue: ~25% of rows lack it. These are typically bulk orders or walk-in sales that cannot be attributed to a customer account. Including them would distort all customer-level analysis.
- Cancelled invoices (prefix 'C') are removed rather than treated as negative revenue because they represent operational reversals, not customer behavior signals.

### 5.2 What is NOT done during cleaning

- **Outliers are not removed.** While some orders are unusually large (e.g., a single customer ordering 80,000 units of one product), these are legitimate bulk purchases. Removing them would bias the analysis toward small orders.
- **Product descriptions are not standardized.** The same product may appear with slightly different descriptions. For this analysis, StockCode (the product identifier) is used for grouping, not Description.
- **Returns are not modeled separately.** The project focuses on purchasing behavior. A separate returns analysis would be a valid extension.

---

## 6. Exploratory Data Analysis

### 6.1 Revenue distribution

Revenue per line item is heavily right-skewed, with a long tail of high-value transactions. Most line items generate £10–£50, but a small number exceed £1,000. The median revenue per line item is substantially lower than the mean, confirming the skew.

### 6.2 Monthly revenue trend

Revenue peaks in **November 2011** (£1.46M), driven by seasonal holiday purchasing. There is a consistent upward trend from December 2010 through November 2011, with a dip in December 2011 (partial month — data ends December 9).

Monthly active customers follow a similar pattern, growing from ~700 in early months to ~1,700+ at peak.

### 6.3 Geographic concentration

| Country | Revenue | Customers | % of Revenue | AOV |
|---------|---------|-----------|-------------|-----|
| United Kingdom | £7,285,025 | 3,920 | 82.0% | £437.64 |
| Netherlands | £285,446 | 9 | 3.2% | £3,036.66 |
| EIRE | £265,262 | 3 | 3.0% | £1,020.24 |
| Germany | £228,678 | 94 | 2.6% | £500.39 |
| France | £208,934 | 87 | 2.3% | £537.11 |
| Australia | £138,454 | 9 | 1.6% | £2,429.01 |

**Key insight:** The UK accounts for 82% of revenue and 90% of customers. The Netherlands and Australia have very few customers but extremely high AOV, suggesting B2B or wholesale relationships.

### 6.4 Pareto (Revenue Concentration) Analysis

**Finding: The top 26% of customers generate 80% of revenue.**

- Top 20% (867 customers): £6,635,245 in revenue (74.7%)
- Bottom 80% (3,471 customers): £2,251,964 in revenue (25.3%)

**Why this matters:** The business is heavily dependent on a relatively small customer base. Losing even a handful of top customers would have disproportionate revenue impact. This creates urgency for retention efforts targeting high-value segments.

### 6.5 Repeat Customer Rate

- **65.6% of customers** made more than one purchase (2,845 customers)
- **34.4%** were one-time buyers (1,493 customers)

**Why this matters:** One-third of the customer base never returned. Converting even a fraction of one-time buyers into repeat customers would significantly increase lifetime revenue.

### 6.6 Top Products

The best-selling product by revenue is **PAPER CRAFT, LITTLE BIRDIE** (£168,470), though this was purchased by only 1 customer (a bulk order). The most broadly popular product is **REGENCY CAKESTAND 3 TIER** (£142,265, 881 unique buyers), making it a stronger candidate for marketing focus.

### 6.7 Daily Transaction Patterns

Daily transaction volume and revenue show a 7-day weekly cycle, with lower activity on weekends. Revenue shows more volatility than transaction count, indicating that weekday orders tend to be larger.

---

## 7. Customer Feature Engineering

### Why feature engineering is necessary

Raw transaction data is at the line-item level (one row per product per invoice). To do customer segmentation or prediction, we need to aggregate this into **one row per customer** with summary statistics. This is the foundational transformation in any customer analytics project.

### Features created (`src/features.py`)

| Feature | Definition | Why it's useful |
|---------|-----------|----------------|
| `recency` | Days since last purchase | The single strongest predictor of future purchase behavior |
| `frequency` | Number of unique invoices | Measures engagement and loyalty |
| `monetary` | Total revenue from customer | Identifies high-value customers |
| `avg_order_value` | Mean revenue per invoice | Separates bulk buyers from frequent small buyers |
| `total_items` | Total quantity purchased | Volume indicator |
| `customer_lifetime_days` | Days between first and last purchase | Longer lifetimes suggest stronger relationships |
| `avg_days_between_orders` | Mean inter-purchase gap | Regularity of purchasing |
| `first_purchase` | Date of first transaction | For cohort assignment |
| `last_purchase` | Date of last transaction | For recency calculation |

### Feature distributions

All monetary features are heavily right-skewed (typical for transactional data). The median customer:
- Last purchased **58 days** ago
- Has made **4 orders**
- Has spent **£1,860** total
- Has an average order value of **£380**

The distributions show a clear two-cluster pattern: a large group of infrequent, low-spending customers, and a smaller group of highly active, high-spending ones.

---

## 8. RFM Segmentation

### What is RFM?

RFM is a well-established marketing framework that scores customers on three dimensions:

- **Recency (R):** How recently did they purchase? (Lower recency = more engaged)
- **Frequency (F):** How often do they purchase? (Higher frequency = more loyal)
- **Monetary (M):** How much do they spend? (Higher monetary = more valuable)

### How scoring works

Each dimension is divided into quintiles (20th percentile buckets), producing a score of 1–5:

| Score | R meaning | F meaning | M meaning |
|-------|-----------|-----------|-----------|
| 5 | Most recent | Most frequent | Highest spending |
| 4 | Recent | Frequent | High spending |
| 3 | Moderate | Moderate | Moderate |
| 2 | Less recent | Less frequent | Lower spending |
| 1 | Least recent | Least frequent | Lowest spending |

The three scores are summed to produce an **RFM Score** (range 3–15), which is then mapped to named segments:

| RFM Score | Segment | Customer count | % of customers | % of revenue |
|-----------|---------|---------------|----------------|-------------|
| 15 | Champion | (included in Loyal Customer for this binning) | — | — |
| 13–14 | Loyal Customer | 876 | 20.2% | 67.0% |
| 10–12 | Potential Loyalist | 613 | 14.1% | 13.7% |
| 7–9 | Low Value | 944 | 21.8% | 10.9% |
| 3–6 | At Risk/Lost | 1,905 | 43.9% | 8.4% |

**Note:** The Champion and Loyal Customer bins were merged in this implementation because K=2 clustering suggested the data naturally splits into two broad groups. A finer-grained segmentation with separate Champion/Loyal bins would be appropriate with more data or different business context.

### Segment profiles

| Segment | Avg Recency (days) | Avg Frequency | Avg Monetary (£) | Avg Recency (days) |
|---------|-------------------|---------------|-------------------|-------------------|
| Loyal Customer | 14 | 12.2 | £6,798 | 14 |
| Potential Loyalist | 37 | 4.6 | £1,985 | 37 |
| Low Value | 63 | 2.7 | £1,022 | 63 |
| At Risk/Lost | 161 | 1.3 | £394 | 161 |

**Why these numbers matter:**

- **Loyal Customers** (20.2% of customers, 67.0% of revenue): These are the business. They purchase frequently, recently, and at high value. Losing even a few is catastrophic.
- **Potential Loyalists** (14.1%, 13.7%): Active but not yet at Loyal Customer levels. The business should focus on moving these up through upselling and engagement.
- **Low Value** (21.8%, 10.9%): Some purchasing activity but low frequency and monetary value. Worth re-engaging but not with expensive interventions.
- **At Risk/Lost** (43.9%, 8.4%): The largest segment by count but smallest by revenue. These customers purchased once or twice and never returned. Win-back campaigns are appropriate for high-value members of this group.

### Why RFM was chosen

- **Interpretability:** Unlike black-box clustering, RFM scores are immediately understandable by business stakeholders.
- **Actionability:** Each segment maps directly to a marketing strategy.
- **Proven track record:** RFM has been used in direct marketing for decades with consistent results.
- **Limitation:** RFM is rule-based and assumes equal weighting across dimensions. K-Means (next section) provides a data-driven alternative.

---

## 9. K-Means Clustering

### What is K-Means?

K-Means is an unsupervised machine learning algorithm that groups data points into K clusters based on feature similarity. Unlike RFM (which uses predefined rules), K-Means discovers natural groupings in the data.

### Why use both RFM and K-Means?

They answer different questions:

- **RFM:** "How do we classify customers based on known marketing principles?"
- **K-Means:** "What natural groupings emerge from the data without human assumptions?"

Using both provides triangulation: if both methods identify similar segments, the findings are more robust.

### Feature selection

Six features were used for clustering:

| Feature | Why included |
|---------|-------------|
| `recency` | Core engagement signal |
| `frequency` | Loyalty indicator |
| `monetary` | Value indicator |
| `avg_order_value` | Separates bulk from frequent buyers |
| `total_items` | Volume dimension |
| `customer_lifetime_days` | Relationship duration |

### Preprocessing

1. **Log transform:** `np.log1p(x)` compresses the right-skewed distributions, reducing the influence of extreme outliers.
2. **Standard scaling:** Zero mean, unit variance ensures all features contribute equally (recency in days and monetary in pounds have very different scales).

### Optimal K selection

K was selected by iterating K = 2 through K = 6 and evaluating silhouette score (a measure of cluster cohesion vs. separation, range -1 to 1):

| K | Silhouette Score |
|---|-----------------|
| 2 | **0.377** (best) |
| 3 | ~0.30 |
| 4 | ~0.25 |
| 5 | ~0.22 |
| 6 | ~0.20 |

**K = 2 was selected** with silhouette = 0.377.

### Cluster profiles

| Cluster | Customers | % | Avg Recency | Avg Frequency | Avg Monetary | Total Revenue | % Revenue |
|---------|-----------|---|------------|---------------|-------------|---------------|-----------|
| 0 (Active) | 2,213 | 51.0% | 41 days | 7.0 orders | £3,676 | £8,134,866 | 91.5% |
| 1 (Dormant) | 2,125 | 49.0% | 147 days | 1.4 orders | £354 | £752,343 | 8.5% |

**Key finding:** The data naturally splits into two nearly equal halves — but the revenue split is 91.5% vs 8.5%. Half the customers generate almost all the revenue.

**Why K=2 is meaningful:** This aligns with the Pareto finding. The business effectively has two customer populations: an active, high-value core and a dormant, low-value tail. This is a common pattern in retail.

### Why silhouette was low (0.377)

A silhouette of 0.377 indicates moderate cluster separation. This is expected because:
- The two clusters overlap in the middle (some customers are "borderline")
- The features are inherently noisy (customer behavior is not perfectly bimodal)
- More clusters (K=3+) fragment the data further without improving separability

---

## 10. Cohort Retention Analysis

### What is cohort analysis?

A cohort is a group of customers who share a common characteristic — in this case, their **first purchase month**. Cohort retention tracks what percentage of each cohort returns in subsequent months.

### Why cohort analysis matters

Segmentation tells you who your customers are *now*. Cohort analysis tells you how they *behave over time*. It answers: "Of the customers who joined in March 2011, how many came back in April? May? June?"

### Methodology

1. For each customer, determine their **cohort month** (month of first purchase).
2. For each transaction, calculate the **cohort index** (months since first purchase + 1).
3. Count unique customers per (cohort_month, cohort_index) pair.
4. Divide by the cohort's initial size to get retention rate.

### Results

| Months since first purchase | Avg retention rate |
|----------------------------|-------------------|
| 1 (first purchase) | 100% |
| 2 | **20.6%** |
| 3 | 22.1% |
| 4 | 23.2% |
| 5 | 24.6% |
| 6 | 25.1% |
| 7 | 24.4% |
| 8 | 24.3% |
| 9 | 25.1% |
| 10 | 27.8% |
| 11 | 26.9% |
| 12 | 31.0% |

**Critical finding:** Retention drops from 100% to 20.6% in the second month — a **79.4% drop-off**. After that, retention stabilizes around 24–27% and even increases slightly over time (survivor bias: the customers who remain are the loyal ones).

### Why the first month matters so much

- **Gift purchases:** Many customers buy once for a specific occasion (holiday gifts, birthdays) and never return.
- **Discovery purchases:** Customers try the store, decide it's not for them, or simply forget about it.
- **No follow-up:** Without re-engagement campaigns, there's no reason for a one-time buyer to return.

### Business implication

The single highest-ROI investment is **post-purchase engagement in the first 30 days**. If the business can convert even 10% of month-2 drop-offs into repeat buyers, it would add hundreds of thousands in annual revenue.

---

## 11. Temporal Churn Prediction

### Why predictive modeling is needed

Segmentation and cohort analysis describe what happened. Churn prediction asks: **what will happen next?** A classification model that predicts which customers will become inactive enables proactive intervention.

### Defining churn

The UCI dataset does not include a churn label. We define churn **temporally**:

1. Sort all unique transaction dates.
2. Split at the **75th percentile date** (the cutoff).
3. **Observation window:** All transactions before the cutoff. Features are computed from this window.
4. **Prediction window:** All transactions on or after the cutoff.
5. **Churn label:** 1 if the customer made zero purchases in the prediction window, 0 otherwise.

### Why this temporal split is critical

A common mistake in churn modeling is using features computed from the **entire** dataset to predict a target that is also derived from the entire dataset. This creates **temporal data leakage** — the model learns from the future, producing artificially high accuracy that fails in production.

By computing features only from the observation window and labeling only based on the prediction window, we simulate a real-world scenario: "Given what we know today, who is likely to go inactive?"

### Feature matrix

The same `customer_features()` function is called on the observation-window subset, producing 7 numeric features per customer. Non-numeric columns (CustomerID, dates) are dropped.

### Models compared

| Model | Why included | Key property |
|-------|-------------|-------------|
| Logistic Regression | Interpretable baseline | Linear decision boundary, coefficients are feature importances |
| Random Forest | Non-linear ensemble | Handles feature interactions, provides feature importance |

Both models use `class_weight='balanced'` to handle the ~43% churn rate (slightly imbalanced but not severely).

### Validation approach

A **chronological 80/20 split** (no shuffling) was used:
- Training: First 80% of customers (by observation order)
- Test: Last 20%

This preserves temporal ordering and prevents information from "later" customers leaking into training.

### Results

| Model | ROC-AUC |
|-------|---------|
| Logistic Regression | **0.732** (selected) |
| Random Forest | ~0.72 |

Logistic Regression was selected as the best model based on ROC-AUC.

### Why Logistic Regression won

Despite Random Forest's theoretical advantage in capturing non-linearities, Logistic Regression performed comparably or slightly better. Possible reasons:
- The relationship between features and churn is approximately linear on the log-odds scale.
- Random Forest may have overfit to noise in the training data with 250 trees.
- The small feature set (7 features) limits the complexity that Random Forest can exploit.

### Feature importance

| Feature | Importance |
|---------|-----------|
| `frequency` | 0.160 (dominant) |
| `customer_lifetime_days` | 0.005 |
| `avg_days_between_orders` | 0.004 |
| `recency` | 0.003 |
| `avg_order_value` | 0.000 |
| `total_items` | 0.000 |
| `monetary` | 0.000 |

**Key insight:** `frequency` completely dominates the model. Customers who have made more purchases are far less likely to go inactive. This makes intuitive sense: a customer who has bought 10 times has demonstrated strong intent, while a customer who bought once could go either way.

### Churned vs Active profiles

| Feature | Active customers | Inactive customers |
|---------|-----------------|-------------------|
| Recency (days) | 72.9 | 122.9 |
| Frequency (orders) | 4.8 | 1.9 |
| Monetary (£) | £2,203 | £719 |
| Customer lifetime (days) | 119.5 | 47.2 |

Inactive customers have **2.5x lower frequency**, **3x lower monetary value**, and **2.5x shorter lifetimes**. The most actionable difference is frequency: customers who place fewer orders are at highest risk.

---

## 12. Results Summary

### Key numbers

| Metric | Value |
|--------|-------|
| Clean transactions | 392,692 |
| Unique customers | 4,338 |
| Total revenue | £8,887,209 |
| Repeat customer rate | 65.6% |
| Pareto threshold | Top 26% = 80% of revenue |
| Cohort month-2 retention | 20.6% |
| Churn rate (temporal) | 43.2% |
| Best churn model | Logistic Regression (ROC-AUC: 0.732) |
| Top churn predictor | Purchase frequency |

### What was delivered

1. **Modular Python codebase** (`src/data.py`, `features.py`, `modeling.py`, `pipeline.py`) — clean, documented, production-ready structure.
2. **Jupyter notebook** with 34 cells of EDA, visualizations, and narrative — suitable for stakeholder presentation.
3. **Interactive Streamlit dashboard** with 6 tabs — Overview, RFM Segmentation, Clustering, Cohort Retention, Churn Prediction, Recommendations.
4. **SQL analysis file** with 12 analytical queries — demonstrates SQL proficiency alongside Python.
5. **Serialized model** (`churn_model.joblib`) — ready for deployment or integration into a scoring pipeline.

---

## 13. Business Recommendations

### 13.1 Revenue Protection (Priority: HIGH)

The top 26% of customers generate 80% of revenue. This concentration is both a strength (high-value core) and a vulnerability (single-point-of-failure).

**Actions:**
- Implement a **VIP customer program** for the top 867 customers with dedicated account management.
- Create **automated alerts** when a top-20% customer hasn't purchased in 30+ days.
- Offer **exclusive previews, early access, and personalized recommendations** to maintain engagement.

### 13.2 Post-Acquisition Retention (Priority: HIGH)

The 79.4% drop-off between month 1 and month 2 is the single biggest revenue leak.

**Actions:**
- Send a **welcome series** (3-5 emails) in the first 14 days post-purchase.
- Offer a **second-purchase incentive** (10-15% discount) valid for 30 days after first order.
- Create a **"New Customer" segment** in the email platform with tailored content.
- Implement **product recommendation emails** based on first purchase category.

### 13.3 Segment-Specific Strategies

| Segment | Size | Revenue share | Strategy | Expected ROI |
|---------|------|--------------|----------|-------------|
| **Champions/Loyal** | 876 (20.2%) | 67.0% | VIP program, early access, referral incentives | High — protecting £5.95M revenue base |
| **Potential Loyalist** | 613 (14.1%) | 13.7% | Upsell campaigns, volume discounts, loyalty tiers | Medium — converting to Loyal could add £1M+ |
| **Low Value** | 944 (21.8%) | 10.9% | Automated re-engagement, limited-time offers | Low-cost, moderate return |
| **At Risk/Lost** | 1,905 (43.9%) | 8.4% | Win-back campaigns for high-value subset; sunset the rest | Selective — focus on those with £500+ historical spend |

### 13.4 Churn Prevention System

**Build a real-time scoring pipeline:**
- Score all customers weekly using the trained Logistic Regression model.
- Flag customers with >0.7 churn probability for immediate outreach.
- Route high-value/high-risk customers to human agents; low-value/high-risk to automated campaigns.

**Key trigger to monitor:** A customer whose purchase frequency drops below their historical average by more than 50% over any 60-day window.

### 13.5 Geographic Expansion

The UK accounts for 82% of revenue but the Netherlands, Germany, and France show high AOVs with growing customer bases.

**Actions:**
- Translate marketing materials into German and French.
- Investigate shipping cost optimization for EU customers.
- Test country-specific promotions in Germany (94 customers, £500 AOV) and France (87 customers, £537 AOV).

### 13.6 Product Strategy

The REGENCY CAKESTAND 3 TIER (£142K, 881 buyers) and WHITE HANGING HEART T-LIGHT HOLDER (£100K, 856 buyers) are the broadest-appeal products. These should be:
- Featured prominently in acquisition campaigns
- Used as "gateway" products for new customers
- Bundled with complementary items to increase AOV

---

## 14. Limitations & Future Work

### Limitations

1. **Observational data:** This is transaction data from one retailer. Correlations should not be interpreted as causal effects. For example, we cannot conclude that *increasing* frequency *causes* lower churn — it may be that naturally loyal customers both buy more frequently and churn less.

2. **Churn is a proxy:** Since the dataset has no churn labels, we defined inactivity as zero purchases in the prediction window. This is a reasonable proxy but does not capture customers who reduce spending without stopping entirely.

3. **No marketing/campaign data:** We cannot measure the effectiveness of specific interventions. The recommendations are based on analytical findings, not controlled experiments.

4. **Temporal scope:** The data covers only 12 months. Seasonal patterns may not generalize. A customer who bought holiday gifts in November 2010 may not be a "holiday-only" buyer — we only have one holiday season to observe.

5. **Feature limitations:** We lack demographic data, browsing behavior, email engagement, and return/refund history, all of which would improve churn prediction.

### Future work

| Direction | What it would add |
|-----------|------------------|
| **Customer Lifetime Value (CLV) model** | Predict total future revenue per customer, enabling better acquisition budget allocation |
| **Market basket analysis** | Identify product affinity rules ("customers who buy X also buy Y") for cross-selling |
| **Return/refund modeling** | Separate returns from cancellations; model return propensity as a churn signal |
| **Time-series forecasting** | Predict future revenue by segment using ARIMA or Prophet |
| **A/B testing framework** | Measure actual impact of retention interventions |
| **Real-time scoring API** | Deploy the churn model as a REST endpoint for operational use |
| **NLP on product descriptions** | Extract product categories and themes for richer feature engineering |
| **Email/campaign integration** | Connect churn scores to marketing automation platforms |

---

*Built with Python, SQL, Pandas, Scikit-learn, Streamlit. Data: UCI Online Retail Dataset (CC BY 4.0).*
