import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# 🚀 App header
st.title("Customer Segmentation App")
st.write("Enter customer details below to predict their segment.")

# ✅ Load saved models
try:
    kmeans = joblib.load("kmeans_model.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# ✅ Load PCA data (optional for plotting)
pca_df = None
if os.path.exists("pca_clusters.csv"):
    try:
        pca_df = pd.read_csv("pca_clusters.csv")
    except Exception:
        st.warning("⚠️ PCA CSV exists but could not be loaded.")

# ✅ Cluster descriptions
cluster_descriptions = {
    0: "High-income, frequent buyers, low recency",
    1: "Low-income, occasional buyers, high recency",
    2: "Moderate-income, regular buyers",
    3: "New customers, low activity",
    4: "Loyal customers, high web/store purchases",
    5: "Inactive users, high recency, low spending"
}

# ✅ Input fields
age = st.number_input("Age", 18, 100, 35)
income = st.number_input("Income", 0, 200000, 50000)
total_spending = st.number_input("Total Spending", 0, 5000, 1000)
num_web_purchases = st.number_input("Web Purchases", 0, 100, 10)
num_store_purchases = st.number_input("Store Purchases", 0, 100, 10)
num_web_visits = st.number_input("Web Visits per Month", 0, 50, 30)
recency = st.number_input("Recency (days since last purchase)", 0, 365, 30)

# ✅ Create input DataFrame
input_data = pd.DataFrame([{
    "Age": age,
    "Income": income,
    "Total_Spending": total_spending,
    "NumWebPurchases": num_web_purchases,
    "NumStorePurchases": num_store_purchases,
    "NumWebVisitsMonth": num_web_visits,
    "Recency": recency
}])

# ✅ Scale input
input_scaled = scaler.transform(input_data)

# ✅ Prediction logic
if st.button("Predict Segment"):
    cluster = kmeans.predict(input_scaled)[0]
    description = cluster_descriptions.get(cluster, "No description available")

    st.success(f"Predicted Segment: Cluster {cluster}")
    st.info(f"Segment Description: {description}")

    # Save prediction
    input_data["PredictedCluster"] = cluster
    if os.path.exists("predictions.csv"):
        pd.concat([pd.read_csv("predictions.csv"), input_data], ignore_index=True).to_csv("predictions.csv", index=False)
    else:
        input_data.to_csv("predictions.csv", index=False)
    st.success("Prediction saved to predictions.csv")

    # ✅ PCA Visualization
    if pca_df is not None:
        try:
            # Recalculate PCA
            full_data = pca_df.drop(columns=["PCA1", "PCA2"], errors='ignore')
            full_scaled = scaler.transform(full_data.drop(columns=["Cluster"], errors='ignore'))
            pca_model = PCA(n_components=2)
            pca_result = pca_model.fit_transform(full_scaled)

            full_data["PCA1"] = pca_result[:, 0]
            full_data["PCA2"] = pca_result[:, 1]
            full_data["Cluster"] = pca_df["Cluster"]

            # Transform current input
            new_pca = pca_model.transform(input_scaled)

            # Plot
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=full_data, x="PCA1", y="PCA2", hue="Cluster", palette="Set1", alpha=0.6)
            plt.scatter(new_pca[0, 0], new_pca[0, 1], color="black", s=120, label="New Customer")
            plt.title("Customer Segmentation Map")
            plt.legend()
            st.pyplot(plt)
        except Exception as e:
            st.warning(f"PCA visualization failed: {e}")
