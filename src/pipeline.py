"""
Customer Segmentation Pipeline
================================
Loads raw online retail transaction data, cleans it, engineers RFM
(Recency, Frequency, Monetary) features per customer, clusters customers
with K-Means, and writes small processed CSVs used by the Streamlit app.

Usage:
    python src/pipeline.py
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

RAW_PATH = "data/raw/online_retail.csv"
OUT_DIR = "data/processed"


def load_and_clean(path: str) -> pd.DataFrame:
    """Load raw transactions and remove missing / duplicate / invalid rows."""
    df = pd.read_csv(path, encoding="ISO-8859-1")

    # Drop rows without a customer ID (can't attribute the purchase) or description
    df = df.dropna(subset=["Customer ID", "Description"])
    df = df.drop_duplicates()

    # Keep genuine sales only: positive quantity and positive price.
    # (Negative quantities are returns/cancellations - excluded from the
    # segmentation base, but could be analysed separately.)
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)].copy()

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="mixed", dayfirst=True)
    df["Revenue"] = df["Quantity"] * df["Price"]
    df["Customer ID"] = df["Customer ID"].astype(int)

    return df


def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate cleaned transactions into one Recency/Frequency/Monetary row per customer."""
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("Customer ID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("Invoice", "nunique"),
        Monetary=("Revenue", "sum"),
        Country=("Country", "first"),
    ).reset_index()

    # Drop the very small number of customers with zero/negative net spend
    rfm = rfm[rfm["Monetary"] > 0].reset_index(drop=True)
    return rfm


def choose_k(X_scaled: np.ndarray, k_range=range(2, 9)) -> tuple[list, list]:
    """Return inertia and silhouette scores for a range of k, to justify the chosen k."""
    inertias, sil_scores = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, labels))
    return list(k_range), inertias, sil_scores


def cluster_customers(rfm: pd.DataFrame, k: int = 4) -> pd.DataFrame:
    """Scale RFM features and cluster with K-Means."""
    features = rfm[["Recency", "Frequency", "Monetary"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    rfm["Cluster"] = km.fit_predict(X_scaled)

    # 2D projection for visualisation
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    rfm["PCA1"], rfm["PCA2"] = coords[:, 0], coords[:, 1]

    sil = silhouette_score(X_scaled, rfm["Cluster"])
    return rfm, sil


def label_clusters(rfm: pd.DataFrame) -> pd.DataFrame:
    """Assign business-friendly names based on each cluster's actual RFM profile,
    rather than a fixed order (a cluster ranked 3rd by revenue could still be
    recently-active, or could be long-dormant - the label should reflect that)."""
    profile = rfm.groupby("Cluster").agg(
        Monetary=("Monetary", "mean"),
        Recency=("Recency", "mean"),
        Frequency=("Frequency", "mean"),
    )

    n = profile.shape[0]
    money_rank = profile["Monetary"].rank(ascending=False)  # 1 = highest spend
    top_half = money_rank[money_rank <= n / 2].index      # high-value clusters
    bottom_half = money_rank[money_rank > n / 2].index    # low-value clusters

    label_map = {}

    # High-value clusters: the most-recent-and-frequent of the two is "Champions"
    top_sorted = profile.loc[top_half].sort_values("Monetary", ascending=False).index
    for i, cluster_id in enumerate(top_sorted):
        label_map[cluster_id] = "Champions (VIP)" if i == 0 else "Loyal Customers"

    # Low-value clusters: split by recency *within this subset* - the one that
    # hasn't purchased in longest is "At Risk / Lost", the more recent one is "New / Low-Value"
    bottom_sorted = profile.loc[bottom_half].sort_values("Recency", ascending=True).index
    for i, cluster_id in enumerate(bottom_sorted):
        label_map[cluster_id] = "New / Low-Value" if i == 0 else "At Risk / Lost"

    rfm["Segment"] = rfm["Cluster"].map(label_map)
    return rfm


def build_summary(rfm: pd.DataFrame) -> pd.DataFrame:
    summary = rfm.groupby("Segment").agg(
        Customers=("Customer ID", "count"),
        Total_Revenue=("Monetary", "sum"),
        Avg_Revenue=("Monetary", "mean"),
        Avg_Recency=("Recency", "mean"),
        Avg_Frequency=("Frequency", "mean"),
    ).round(2).reset_index().sort_values("Total_Revenue", ascending=False)
    return summary


def main():
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading & cleaning raw data...")
    df = load_and_clean(RAW_PATH)
    print(f"  {len(df):,} clean transaction rows")

    print("Computing RFM features...")
    rfm = compute_rfm(df)
    print(f"  {len(rfm):,} customers")

    print("Selecting k (elbow + silhouette)...")
    features = rfm[["Recency", "Frequency", "Monetary"]]
    X_scaled = StandardScaler().fit_transform(features)
    k_range, inertias, sil_scores = choose_k(X_scaled)
    k_selection = pd.DataFrame({"k": k_range, "inertia": inertias, "silhouette": sil_scores})
    k_selection.to_csv(f"{OUT_DIR}/k_selection.csv", index=False)
    best_k = k_range[int(np.argmax(sil_scores))]
    print(f"  Best silhouette score at k={best_k}")

    print("Clustering customers...")
    rfm, sil = cluster_customers(rfm, k=4)  # 4 kept for interpretability (Champions/Loyal/At-Risk/New)
    rfm = label_clusters(rfm)
    print(f"  Silhouette score (k=4): {sil:.4f}")

    summary = build_summary(rfm)

    rfm.to_csv(f"{OUT_DIR}/customer_segments.csv", index=False)
    summary.to_csv(f"{OUT_DIR}/segment_summary.csv", index=False)

    with open(f"{OUT_DIR}/metrics.txt", "w") as f:
        f.write(f"customers={len(rfm)}\n")
        f.write(f"total_revenue={rfm['Monetary'].sum():.2f}\n")
        f.write(f"silhouette_k4={sil:.4f}\n")
        f.write(f"best_k_by_silhouette={best_k}\n")

    print("Done. Outputs written to", OUT_DIR)


if __name__ == "__main__":
    main()
