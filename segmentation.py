import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Load models
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

# Optional: Load PCA data if used for plotting
try:
    pca_df = pd.read_csv("pca_clusters.csv")  # DataFrame with PCA1, PCA2, Cluster columns
except FileNotFoundError:
    pca_df = None

# Define cluster descriptions (customize for your model)
cluster_descriptions = {
    0: "High-income, frequent buyers, low recency",
    1: "Low-income, occasional buyers, high recency",
    2: "Moderate-income, regular buyers",
    3: "New customers, low activity",
    4: "Loyal customers, high web/store purchases",
    5: "Inactive users, high recency, low spending"
}

# Streamlit UI
st.title("Customer Segmentation App")
st.write("Enter customer details below to predict their segment.")

# Input fields
age = st.number_input("Age", min_value=18, max_value=100, value=35)
income = st.number_input("Income", min_value=0, max_value=200000, value=50000)
total_spending = st.number_input("Total Spending", min_value=0, max_value=5000, value=1000)
num_web_purchases = st.number_input("Web Purchases", min_value=0, max_value=100, value=10)
num_store_purchases = st.number_input("Store Purchases", min_value=0, max_value=100, value=10)
num_web_visits = st.number_input("Web Visits per Month", min_value=0, max_value=50, value=30)
recency = st.number_input("Recency (days since last purchase)", min_value=0, max_value=365, value=30)

# Create DataFrame from input
input_data = pd.DataFrame({
    "Age": [age],
    "Income": [income],
    "Total_Spending": [total_spending],
    "NumWebPurchases": [num_web_purchases],
    "NumStorePurchases": [num_store_purchases],
    "NumWebVisitsMonth": [num_web_visits],
    "Recency": [recency]
})

# Scale input
input_scaled = scaler.transform(input_data)

# Predict Segment
if st.button("Predict Segment"):
    cluster = kmeans.predict(input_scaled)[0]
    description = cluster_descriptions.get(cluster, "No description available")
    
    st.success(f"Predicted Segment: Cluster {cluster}")
    st.info(f"Segment Description: {description}")

    # Save prediction
    input_data["PredictedCluster"] = cluster
    if os.path.exists("predictions.csv"):
        existing = pd.read_csv("predictions.csv")
        updated = pd.concat([existing, input_data], ignore_index=True)
        updated.to_csv("predictions.csv", index=False)
    else:
        input_data.to_csv("predictions.csv", index=False)
    st.success("Prediction saved to predictions.csv")

    # Optional: Plot on PCA
    if pca_df is not None:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.decomposition import PCA

        # Add PCA to the current input for plotting
        full_data = pca_df.copy()
        pca_model = PCA(n_components=2)
        full_scaled = scaler.transform(full_data.drop(columns=["Cluster", "PCA1", "PCA2"], errors='ignore'))
        pca_result = pca_model.fit_transform(full_scaled)
        full_data["PCA1"], full_data["PCA2"] = pca_result[:, 0], pca_result[:, 1]

        # PCA of current input
        new_pca = pca_model.transform(input_scaled)

        # Plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=full_data, x="PCA1", y="PCA2", hue="Cluster", palette="Set1", alpha=0.6)
        plt.scatter(new_pca[0, 0], new_pca[0, 1], color="black", s=100, label="New Customer")
        plt.title("Customer Segmentation (with Input)")
        plt.legend()
        st.pyplot(plt)


    