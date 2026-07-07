import streamlit as st
import pandas as pd

st.set_page_config(page_title="Outlier Detection Using Percentile", layout="wide")

st.title("Outlier Detection Using Percentile Method")

uploaded_file = st.file_uploader("Upload AB_NYC_2019.csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Original Dataset")
    st.write(df.head())

    st.subheader("Price Statistics")
    st.write(df["price"].describe())

    min_threshold, max_threshold = df["price"].quantile([0.01, 0.999])

    st.write(f"**1st Percentile:** {min_threshold}")
    st.write(f"**99.9th Percentile:** {max_threshold}")

    outliers = df[(df["price"] < min_threshold) | (df["price"] > max_threshold)]

    st.subheader("Detected Outliers")
    st.write(outliers)

    df_clean = df[
        (df["price"] > min_threshold)
        & (df["price"] < max_threshold)
    ]

    st.subheader("Dataset After Removing Outliers")
    st.write(df_clean.head())

    st.write("Original Shape:", df.shape)
    st.write("Cleaned Shape:", df_clean.shape)

    st.subheader("Cleaned Price Statistics")
    st.write(df_clean["price"].describe())
