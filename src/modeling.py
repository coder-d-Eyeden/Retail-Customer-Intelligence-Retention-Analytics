from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib

from .data import PROC, MODEL
from .features import customer_features


def kmeans_cluster(c: pd.DataFrame) -> pd.DataFrame:
    """Apply K-Means clustering on log-transformed customer features.

    Selects best K (2-6) by silhouette score.
    """
    cols = [
        "recency", "frequency", "monetary",
        "avg_order_value", "total_items", "customer_lifetime_days",
    ]
    X = np.log1p(c[cols].clip(lower=0))
    X = StandardScaler().fit_transform(X)

    best_km, best_score, best_labels = None, -1, None
    for k in range(2, 7):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        s = silhouette_score(X, labels)
        if s > best_score:
            best_km, best_score, best_labels = km, s, labels

    c = c.copy()
    c["Cluster"] = best_labels

    PROC.mkdir(parents=True, exist_ok=True)
    c.to_csv(PROC / "customer_clusters.csv", index=False)
    print(f"Best K={best_km.n_clusters}; silhouette={best_score:.3f}")
    return c


def temporal_churn(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Predict customer inactivity using a temporally valid split.

    - Observation window: first 75% of unique dates
    - Prediction window: remaining 25%
    - Target: whether the customer made zero purchases in prediction window

    Returns (churn_features_df, results_dict).
    """
    dates = sorted(df["InvoiceDate"].dt.normalize().unique())
    cutoff = pd.Timestamp(dates[int(len(dates) * 0.75)])

    obs = df[df["InvoiceDate"] < cutoff].copy()
    future = df[df["InvoiceDate"] >= cutoff].copy()

    feat = customer_features(obs, cutoff)

    future_buy = (
        future.groupby("CustomerID")["InvoiceNo"]
        .nunique()
        .rename("future_orders")
    )
    feat = feat.merge(future_buy, left_on="CustomerID", right_index=True, how="left")
    feat["future_orders"] = feat["future_orders"].fillna(0)
    feat["churn"] = (feat["future_orders"] == 0).astype(int)

    PROC.mkdir(parents=True, exist_ok=True)
    feat.to_csv(PROC / "churn_dataset.csv", index=False)

    # Prepare feature matrix
    drop_cols = ["CustomerID", "last_purchase", "first_purchase", "future_orders", "churn"]
    X = feat.drop(columns=[c for c in drop_cols if c in feat.columns], errors="ignore")
    y = feat["churn"]
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.select_dtypes(include=np.number)

    # Chronological split (no shuffling to prevent temporal leakage)
    split = int(len(X) * 0.8)
    Xtr, Xte = X.iloc[:split], X.iloc[split:]
    ytr, yte = y.iloc[:split], y.iloc[split:]

    models = {
        "logistic": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=250, random_state=42, class_weight="balanced", n_jobs=-1
        ),
    }

    results = {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        preds = m.predict(Xte)
        proba = m.predict_proba(Xte)[:, 1]
        results[name] = {
            "roc_auc": roc_auc_score(yte, proba),
            "report": classification_report(yte, preds, output_dict=True),
        }

    # Select best model, refit on full data, save
    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    best = models[best_name]
    best.fit(X, y)

    MODEL.mkdir(parents=True, exist_ok=True)
    joblib.dump((best, X.columns.tolist()), MODEL / "churn_model.joblib")

    # Feature importance
    importances = getattr(
        best,
        "feature_importances_",
        np.abs(best.coef_[0]) if hasattr(best, "coef_") else np.zeros(len(X.columns)),
    )
    (
        pd.DataFrame({"feature": X.columns, "importance": importances})
        .sort_values("importance", ascending=False)
        .to_csv(PROC / "feature_importance.csv", index=False)
    )

    print(f"Best churn model: {best_name}; ROC-AUC: {results[best_name]['roc_auc']:.3f}")
    return feat, results
