import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Employee Retention Prediction", page_icon="📊")

st.title("📊 Employee Retention Prediction")
st.write("Predict whether an employee is likely to leave the company.")

# Load Dataset
@st.cache_data
def load_data():
    return pd.read_csv("PROJECT 5/HR_comma_sep.csv")

df = load_data()

# Prepare Data
subdf = df[['satisfaction_level',
            'average_montly_hours',
            'promotion_last_5years',
            'salary']]

salary_dummies = pd.get_dummies(subdf['salary'], prefix='salary')

X = pd.concat(
    [subdf.drop('salary', axis=1), salary_dummies],
    axis=1
)

y = df['left']

# Train Model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

st.sidebar.header("Employee Details")

satisfaction = st.sidebar.slider(
    "Satisfaction Level",
    0.0, 1.0, 0.5
)

hours = st.sidebar.slider(
    "Average Monthly Hours",
    50, 350, 200
)

promotion = st.sidebar.selectbox(
    "Promotion in Last 5 Years",
    [0, 1]
)

salary = st.sidebar.selectbox(
    "Salary Level",
    ["low", "medium", "high"]
)

salary_low = 1 if salary == "low" else 0
salary_medium = 1 if salary == "medium" else 0
salary_high = 1 if salary == "high" else 0

input_data = pd.DataFrame({
    'satisfaction_level': [satisfaction],
    'average_montly_hours': [hours],
    'promotion_last_5years': [promotion],
    'salary_high': [salary_high],
    'salary_low': [salary_low],
    'salary_medium': [salary_medium]
})

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    if prediction == 1:
        st.error("⚠️ Employee is likely to leave the company.")
    else:
        st.success("✅ Employee is likely to stay with the company.")

    st.write(f"Probability of Staying: **{probability[0]:.2%}**")
    st.write(f"Probability of Leaving: **{probability[1]:.2%}**")

st.subheader("Dataset Preview")
st.dataframe(df.head())
