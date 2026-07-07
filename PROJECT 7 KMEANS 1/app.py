import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Iris K-Means Clustering", layout="wide")

st.title("🌸 Iris Flower Clustering using K-Means")
st.write(
    "This app uses the Iris dataset and performs K-Means clustering "
    "using Petal Length and Petal Width features."
)

# Load dataset
iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# Select required features
df = df[['petal length (cm)', 'petal width (cm)']]

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Scaling
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

df_scaled = pd.DataFrame(
    scaled_data,
    columns=['petal length', 'petal width']
)

# Select K
k = st.slider("Select Number of Clusters (K)", 2, 10, 3)

# KMeans
km = KMeans(n_clusters=k, random_state=42)
clusters = km.fit_predict(df_scaled)

df_scaled["Cluster"] = clusters

# Cluster Plot
st.subheader("K-Means Clustering")

fig, ax = plt.subplots(figsize=(8, 6))

scatter = ax.scatter(
    df_scaled["petal length"],
    df_scaled["petal width"],
    c=df_scaled["Cluster"]
)

ax.scatter(
    km.cluster_centers_[:, 0],
    km.cluster_centers_[:, 1],
    marker='*',
    s=250
)

ax.set_xlabel("Petal Length")
ax.set_ylabel("Petal Width")
ax.set_title(f"K-Means Clustering (K={k})")

st.pyplot(fig)

# Elbow Method
st.subheader("Elbow Plot")

sse = []

K = range(1, 11)

for i in K:
    model = KMeans(n_clusters=i, random_state=42)
    model.fit(df_scaled[['petal length', 'petal width']])
    sse.append(model.inertia_)

fig2, ax2 = plt.subplots(figsize=(8, 5))

ax2.plot(K, sse, marker="o")
ax2.set_xlabel("Number of Clusters (K)")
ax2.set_ylabel("SSE")
ax2.set_title("Elbow Method")

st.pyplot(fig2)

st.success("Optimal K is typically around 3 for the Iris dataset.")
