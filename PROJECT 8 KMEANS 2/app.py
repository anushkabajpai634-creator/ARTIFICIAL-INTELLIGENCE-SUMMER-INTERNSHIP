import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

st.title("K-Means Clustering App")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("Dataset")
    st.write(data.head())

    numeric_data = data.select_dtypes(include=['number'])

    n_clusters = st.slider("Select Number of Clusters", 2, 10, 3)

    if st.button("Run K-Means"):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(numeric_data)

        data["Cluster"] = clusters

        st.subheader("Clustered Data")
        st.write(data)

        if numeric_data.shape[1] >= 2:
            fig, ax = plt.subplots()
            ax.scatter(
                numeric_data.iloc[:, 0],
                numeric_data.iloc[:, 1],
                c=clusters
            )
            ax.scatter(
                kmeans.cluster_centers_[:, 0],
                kmeans.cluster_centers_[:, 1],
                marker='X',
                s=200
            )

            ax.set_xlabel(numeric_data.columns[0])
            ax.set_ylabel(numeric_data.columns[1])

            st.pyplot(fig)
