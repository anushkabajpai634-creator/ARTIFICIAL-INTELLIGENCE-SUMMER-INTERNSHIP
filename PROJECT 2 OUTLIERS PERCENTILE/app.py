import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Outlier Detection using Percentile", layout="wide")

st.title("📊 Outlier Detection Using Percentile Method")

uploaded_file = st.file_uploader(
    "Upload AB_NYC_2019.csv",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Original Dataset")
    st.write(df.head())

    if "price" not in df.columns:
        st.error("Dataset must contain a 'price' column.")
    else:

        st.subheader("Price Statistics Before Removing Outliers")
        st.write(df["price"].describe())

        fig1 = px.box(
            df,
            y="price",
            title="Price Distribution Before Removing Outliers"
        )
        st.plotly_chart(fig1, use_container_width=True)

        min_threshold, max_threshold = df["price"].quantile([0.01, 0.999])

        st.write(f"**1st Percentile:** {min_threshold:.2f}")
        st.write(f"**99.9th Percentile:** {max_threshold:.2f}")

        df_clean = df[
            (df["price"] > min_threshold) &
            (df["price"] < max_threshold)
        ]

        st.subheader("Dataset After Removing Outliers")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Original Rows", len(df))

        with col2:
            st.metric("Rows After Cleaning", len(df_clean))

        st.write(df_clean.head())

        st.subheader("Price Statistics After Removing Outliers")
        st.write(df_clean["price"].describe())

        fig2 = px.box(
            df_clean,
            y="price",
            title="Price Distribution After Removing Outliers"
        )
        st.plotly_chart(fig2, use_container_width=True)

        csv = df_clean.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download Cleaned Dataset",
            data=csv,
            file_name="cleaned_airbnb_data.csv",
            mime="text/csv"
        )
