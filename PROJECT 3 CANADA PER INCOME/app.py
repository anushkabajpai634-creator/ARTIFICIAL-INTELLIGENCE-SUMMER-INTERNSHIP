import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(
    page_title="Canada Income Prediction",
    page_icon="📈",
    layout="centered"
)

st.title("📈 Canada Per Capita Income Prediction")

try:
    df = pd.read_csv("canada_per_capita_income.csv")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    X = df[['year']]
    y = df['per capita income (US$)']

    model = LinearRegression()
    model.fit(X, y)

    st.success("Model Trained Successfully!")

    year = st.number_input(
        "Enter Year",
        min_value=int(df['year'].min()),
        max_value=2100,
        value=2025
    )

    if st.button("Predict Income"):
        prediction = model.predict([[year]])[0]

        st.metric(
            "Predicted Per Capita Income",
            f"${prediction:,.2f}"
        )

    st.subheader("Model Information")
    st.write("Coefficient:", model.coef_[0])
    st.write("Intercept:", model.intercept_)

except FileNotFoundError:
    st.error(
        "Place 'canada_per_capita_income.csv' in the same GitHub repository as app.py"
    )
except Exception as e:
    st.error(f"Error: {e}")
