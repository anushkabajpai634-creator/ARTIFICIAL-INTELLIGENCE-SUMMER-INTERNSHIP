import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

st.title("Employee Retention Prediction")

# Upload dataset
uploaded_file = st.file_uploader("Upload HR_comma_sep.csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Feature selection
    subdf = df[['satisfaction_level',
                'average_montly_hours',
                'promotion_last_5years',
                'salary']]

    # One-hot encoding
    salary_dummies = pd.get_dummies(subdf.salary, prefix="salary")
    df_with_dummies = pd.concat([subdf, salary_dummies], axis='columns')
    df_with_dummies.drop('salary', axis='columns', inplace=True)

    X = df_with_dummies
    y = df['left']

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.3, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    st.success(f"Model Accuracy: {accuracy:.2f}")

    st.header("Predict Employee Retention")

    satisfaction = st.slider(
        "Satisfaction Level", 0.0, 1.0, 0.5
    )
    monthly_hours = st.number_input(
        "Average Monthly Hours", min_value=50, max_value=400, value=200
    )
    promotion = st.selectbox(
        "Promotion in Last 5 Years", [0, 1]
    )
    salary = st.selectbox(
        "Salary Level", ["low", "medium", "high"]
    )

    salary_low = 1 if salary == "low" else 0
    salary_medium = 1 if salary == "medium" else 0
    salary_high = 1 if salary == "high" else 0

    input_data = pd.DataFrame([[
        satisfaction,
        monthly_hours,
        promotion,
        salary_high,
        salary_low,
        salary_medium
    ]], columns=X.columns)

    if st.button("Predict"):
        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.error("Employee is likely to leave the company.")
        else:
            st.success("Employee is likely to stay in the company.")
