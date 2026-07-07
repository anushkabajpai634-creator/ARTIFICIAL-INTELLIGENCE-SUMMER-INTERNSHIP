import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

st.set_page_config(
    page_title="Income Clustering",
    page_icon="📊",
    layout="wide"
)

st.title("📊 K-Means Income Clustering")

@st.cache_data
def load_data():
    return pd.read_csv("income.csv")

try:
    df = load_data()

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Original Data")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["Age"], df["Income($)"])
    ax.set_xlabel("Age")
    ax.set_ylabel("Income ($)")
    ax.set_title("Age vs Income")
    st.pyplot(fig)

    n_clusters = st.slider(
        "Select Number of Clusters",
        min_value=2,
        max_value=10,
        value=3
    )

    km = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    df["Cluster"] = km.fit_predict(df[["Age", "Income($)"]])

    st.subheader("Clustered Data")
    st.dataframe(df)

    fig2, ax2 = plt.subplots(figsize=(8, 5))

    scatter = ax2.scatter(
        df["Age"],
        df["Income($)"],
        c=df["Cluster"]
    )

    centers = km.cluster_centers_

    ax2.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="X",
        s=200
    )

    ax2.set_xlabel("Age")
    ax2.set_ylabel("Income ($)")
    ax2.set_title("K-Means Clustering Result")

    st.pyplot(fig2)

    st.subheader("Cluster Centers")

    centers_df = pd.DataFrame(
        centers,
        columns=["Age", "Income($)"]
    )

    st.dataframe(centers_df)

except Exception as e:
    st.error(f"Error: {e}")
