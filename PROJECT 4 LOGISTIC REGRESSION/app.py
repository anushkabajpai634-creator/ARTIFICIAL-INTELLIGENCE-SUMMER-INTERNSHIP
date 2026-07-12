import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

st.set_page_config(
    page_title="Employee Retention Predictor",
    page_icon="👨‍💼",
    layout="centered"
)

st.title("👨‍💼 Employee Retention Prediction")
st.write("Predict whether an employee is likely to leave the company.")

# Load Dataset
@st.cache_data
def load_data():
    return pd.read_csv("PROJECT 4 LOGISTIC REGRESSION/HR_comma_sep.csv")

df = load_data()

# Data preprocessing
subdf = df[['satisfaction_level',
            'average_montly_hours',
            'promotion_last_5years',
            'salary']]

salary_dummies = pd.get_dummies(subdf['salary'], prefix='salary')

X = pd.concat([
    subdf.drop('salary', axis=1),
    salary_dummies
], axis=1)

y = df['left']

# Train Model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

st.header("Enter Employee Details")

satisfaction = st.slider(
    "Satisfaction Level",
    0.0, 1.0, 0.5
)

hours = st.slider(
    "Average Monthly Hours",
    80,
    320,
    200
)

promotion = st.selectbox(
    "Promotion in Last 5 Years",
    [0, 1]
)

salary = st.selectbox(
    "Salary Level",
    ["low", "medium", "high"]
)

# Create input dataframe
input_df = pd.DataFrame({
    'satisfaction_level': [satisfaction],
    'average_montly_hours': [hours],
    'promotion_last_5years': [promotion],
    'salary_low': [1 if salary == 'low' else 0],
    'salary_medium': [1 if salary == 'medium' else 0],
    'salary_high': [1 if salary == 'high' else 0]
})

# Ensure column order matches training data
input_df = input_df[X.columns]

if st.button("Predict"):

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    if prediction == 1:
        st.error("⚠️ Employee is likely to Leave the Company.")
    else:
        st.success("✅ Employee is likely to Stay with the Company.")

    st.subheader("Prediction Probability")
    st.write(f"Stay : **{probability[0]*100:.2f}%**")
    st.write(f"Leave: **{probability[1]*100:.2f}%**")

st.markdown("---")
st.caption("Built with Streamlit & Scikit-learn")
