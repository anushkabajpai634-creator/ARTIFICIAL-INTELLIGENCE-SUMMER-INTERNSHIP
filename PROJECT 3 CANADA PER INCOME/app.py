import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Canada Income Prediction", page_icon="📈")

st.title("📈 Canada Per Capita Income Prediction")

uploaded_file = st.file_uploader(
    "Upload canada_per_capita_income.csv",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    X = df[['year']]
    y = df['per capita income (US$)']

    model = LinearRegression()
    model.fit(X, y)

    st.success("Model trained successfully!")

    st.subheader("Predict Income")

    year = st.number_input(
        "Enter Year",
        min_value=1970,
        max_value=2100,
        value=2025
    )

    if st.button("Predict"):
        prediction = model.predict([[year]])[0]

        st.metric(
            label="Predicted Per Capita Income (US$)",
            value=f"${prediction:,.2f}"
        )

    st.subheader("Model Details")
    st.write(f"Coefficient: {model.coef_[0]:.2f}")
    st.write(f"Intercept: {model.intercept_:.2f}")

else:
    st.info("Please upload the dataset to continue.")
