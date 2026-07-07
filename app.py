import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Google Play Store Dashboard", layout="wide")

st.title("📊 Google Play Store Data Visualization Dashboard")

uploaded_file = st.file_uploader(
    "Upload googleplaystore_v2.csv",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset")
    st.write(df.head())

    st.subheader("Dataset Shape")
    st.write(df.shape)

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    # Data Cleaning
    df = df[df["Rating"].notnull()]

    if "Price" in df.columns:
        df["Price"] = df["Price"].astype(str).str.replace("$", "", regex=False)
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce").fillna(0)

    if "Reviews" in df.columns:
        df["Reviews"] = pd.to_numeric(df["Reviews"], errors="coerce")

    if "Installs" in df.columns:
        df["Installs"] = (
            df["Installs"]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("+", "", regex=False)
        )
        df["Installs"] = pd.to_numeric(df["Installs"], errors="coerce")

    st.sidebar.header("Charts")

    chart = st.sidebar.selectbox(
        "Choose Visualization",
        (
            "Rating Distribution",
            "Price Distribution",
            "Content Rating",
            "Top Categories",
            "Rating vs Reviews"
        )
    )

    if chart == "Rating Distribution":
        fig, ax = plt.subplots(figsize=(8,5))
        sns.histplot(df["Rating"], bins=20, kde=True, ax=ax)
        st.pyplot(fig)

    elif chart == "Price Distribution":
        fig, ax = plt.subplots(figsize=(8,5))
        sns.histplot(df["Price"], bins=20, ax=ax)
        st.pyplot(fig)

    elif chart == "Content Rating":
        fig, ax = plt.subplots(figsize=(8,5))
        df["Content Rating"].value_counts().plot(
            kind="bar",
            ax=ax
        )
        ax.set_ylabel("Count")
        st.pyplot(fig)

    elif chart == "Top Categories":
        fig, ax = plt.subplots(figsize=(10,5))
        df["Category"].value_counts().head(10).plot(
            kind="bar",
            ax=ax
        )
        ax.set_ylabel("Apps")
        st.pyplot(fig)

    elif chart == "Rating vs Reviews":
        fig, ax = plt.subplots(figsize=(8,5))
        sns.scatterplot(
            data=df,
            x="Reviews",
            y="Rating",
            ax=ax
        )
        st.pyplot(fig)

    st.subheader("Summary Statistics")
    st.write(df.describe())

else:
    st.info("Please upload the googleplaystore_v2.csv file.")
