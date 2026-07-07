import streamlit as st
import pandas as pd
# Assuming 'df' is your input DataFrame containing 'product_mng'
df = pd.get_dummies(df, columns=['department']) # Replace 'department' with your actual column name
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Employee Retention Predictor",
    page_icon="👨‍💼",
    layout="wide"
)

st.title("👨‍💼 Employee Retention Prediction")
st.write("Predict whether an employee will leave the company.")

@st.cache_data
def load_data():
    return pd.read_csv("PROJECT 4 LOGISTIC REGRESSION/HR_comma_sep.csv")

try:
    df = load_data()

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Convert categorical columns
    df_model = pd.get_dummies(
        df,
        columns=["salary"],
        drop_first=True
    )

    X = df_model.drop("left", axis=1)
    y = df_model["left"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    st.success(f"Model Accuracy: {accuracy:.2%}")

    st.subheader("Enter Employee Details")

    satisfaction_level = st.slider(
        "Satisfaction Level", 0.0, 1.0, 0.5
    )

    last_evaluation = st.slider(
        "Last Evaluation", 0.0, 1.0, 0.5
    )

    number_project = st.number_input(
        "Number of Projects", 1, 10, 3
    )

    average_montly_hours = st.number_input(
        "Average Monthly Hours", 50, 350, 200
    )

    time_spend_company = st.number_input(
        "Years at Company", 1, 15, 3
    )

    Work_accident = st.selectbox(
        "Work Accident",
        [0, 1]
    )

    promotion_last_5years = st.selectbox(
        "Promotion in Last 5 Years",
        [0, 1]
    )

    salary = st.selectbox(
        "Salary Level",
        ["low", "medium", "high"]
    )

    salary_medium = 1 if salary == "medium" else 0
    salary_high = 1 if salary == "high" else 0

    if st.button("Predict"):

        employee = [[
            satisfaction_level,
            last_evaluation,
            number_project,
            average_montly_hours,
            time_spend_company,
            Work_accident,
            promotion_last_5years,
            salary_medium,
            salary_high
        ]]

        prediction = model.predict(employee)[0]
        probability = model.predict_proba(employee)[0][1]

        if prediction == 1:
            st.error(
                f"Employee is likely to leave.\nProbability: {probability:.2%}"
            )
        else:
            st.success(
                f"Employee is likely to stay.\nProbability of leaving: {probability:.2%}"
            )

except Exception as e:
    st.error(f"Error: {e}")
