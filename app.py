"""
Customer Segmentation Dashboard
=================================
Interactive Streamlit app for exploring RFM-based customer segments
built from the UCI Online Retail dataset.

Run locally:
    streamlit run app.py
"""

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide", page_icon="🛍️")

DATA_DIR = "data/processed"


@st.cache_data
def load_data():
    customers = pd.read_csv(f"{DATA_DIR}/customer_segments.csv")
    summary = pd.read_csv(f"{DATA_DIR}/segment_summary.csv")
    k_selection = pd.read_csv(f"{DATA_DIR}/k_selection.csv")
    return customers, summary, k_selection


customers, summary, k_selection = load_data()

st.title("🛍️ Customer Segmentation Dashboard")
st.caption(
    "RFM analysis + K-Means clustering on the UCI Online Retail transaction dataset "
    "(~4,300 customers, ~393K cleaned transactions)."
)

# ---- Sidebar filters ----
st.sidebar.header("Filters")
segments = st.sidebar.multiselect(
    "Segment", options=summary["Segment"].tolist(), default=summary["Segment"].tolist()
)
countries = st.sidebar.multiselect(
    "Country", options=sorted(customers["Country"].unique()), default=[]
)

filtered = customers[customers["Segment"].isin(segments)]
if countries:
    filtered = filtered[filtered["Country"].isin(countries)]

# ---- KPI row ----
col1, col2, col3, col4 = st.columns(4)
col1.metric("Customers", f"{filtered.shape[0]:,}")
col2.metric("Total Revenue", f"£{filtered['Monetary'].sum():,.0f}")
col3.metric("Avg Order Frequency", f"{filtered['Frequency'].mean():.1f}")
col4.metric("Avg Recency (days)", f"{filtered['Recency'].mean():.0f}")

st.divider()

# ---- Segment revenue chart ----
left, right = st.columns([1, 1])

with left:
    st.subheader("Revenue by Segment")
    seg_view = summary[summary["Segment"].isin(segments)]
    fig_rev = px.bar(
        seg_view.sort_values("Total_Revenue", ascending=True),
        x="Total_Revenue", y="Segment", orientation="h",
        text_auto=".2s", color="Segment",
    )
    fig_rev.update_layout(showlegend=False, xaxis_title="Total Revenue (£)", yaxis_title="")
    st.plotly_chart(fig_rev, use_container_width=True)

with right:
    st.subheader("Customer Count by Segment")
    fig_cnt = px.pie(seg_view, values="Customers", names="Segment", hole=0.45)
    st.plotly_chart(fig_cnt, use_container_width=True)

# ---- PCA scatter of clusters ----
st.subheader("Customer Clusters (PCA projection)")
fig_pca = px.scatter(
    filtered, x="PCA1", y="PCA2", color="Segment",
    hover_data=["Customer ID", "Recency", "Frequency", "Monetary", "Country"],
    opacity=0.6,
)
st.plotly_chart(fig_pca, use_container_width=True)

# ---- k selection justification ----
with st.expander("Why k = 4? (elbow + silhouette analysis)"):
    fig_k = px.line(k_selection, x="k", y=["inertia", "silhouette"], markers=True)
    st.plotly_chart(fig_k, use_container_width=True)
    st.write(
        "k=4 was chosen for business interpretability (Champions / Loyal / "
        "New-Low-Value / At-Risk) while keeping a strong silhouette score."
    )

# ---- Segment summary table ----
st.subheader("Segment Summary")
st.dataframe(seg_view.set_index("Segment"), use_container_width=True)

st.divider()
st.caption(
    "Segments are derived from Recency, Frequency and Monetary value per customer, "
    "clustered with K-Means (StandardScaler-scaled features, k=4, random_state=42)."
)
