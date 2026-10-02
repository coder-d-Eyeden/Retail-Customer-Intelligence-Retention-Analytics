# Retail Customer Intelligence & Retention Analytics

End-to-end customer analytics project built on the real **UCI Online Retail** dataset. 
## Dataset
UCI Online Retail: https://archive.ics.uci.edu/dataset/352/online+retail
541,909 transaction records covering 01/12/2010–09/12/2011. License: CC BY 4.0.

## What this project answers
- Which customers generate the most value?
- What customer segments exist?
- Which customers are at risk of inactivity?
- What behaviors predict inactivity?
- How strong is customer retention across cohorts?
- What should the business do?

## Pipeline
`UCI Excel -> cleaning -> customer features -> RFM -> K-Means -> cohort retention -> temporal churn model -> business insights -> Streamlit dashboard`

## Quick start
1. Download `Online Retail.xlsx` from the UCI page above and put it in `data/raw/`.
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `python run_pipeline.py`
4. Launch dashboard: `streamlit run app.py`

## Important methodology
Churn is not supplied by UCI. We define inactivity temporally: features are calculated from an observation window and the target is whether the customer purchases in a later prediction window. This avoids future leakage.

## Limitations
This is observational transaction data from one retailer and period. Churn is a business-defined inactivity proxy, not a contractual customer cancellation. Predictive associations should not be interpreted as causal effects.
