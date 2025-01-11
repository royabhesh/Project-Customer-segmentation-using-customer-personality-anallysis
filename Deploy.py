import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Set up page configuration
st.set_page_config(page_title="Customer Segmentation", layout="wide")

# Custom CSS for background image
st.markdown(
    """
    <style>
    .main {
        background-image: url('https://th.bing.com/th/id/OIP.6rHjdwhwrL_VCpWWTh1m_gHaHa?pid=ImgDet&w=172&h=172&c=7&dpr=1.1');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Predefined preprocessing logic
def preprocess_data():
    # Example data for preprocessing
    data = {
        "Education": ["Post Graduate", "Under Graduate", "Post Graduate", "Under Graduate"],
        "Marital_Status": ["Relationship", "Single", "Single", "Relationship"],
        "Income": [58000, 43000, 72000, 65000],
        "Kids": [1, 0, 2, 1],
        "Expenses": [1500, 2000, 1200, 1800],
        "TotalAcceptedCmp": [1, 0, 3, 2],
        "NumTotalPurchases": [12, 8, 15, 10],
        "customer_Age": [35, 28, 42, 39]
    }
    df = pd.DataFrame(data)

    # Encode categorical variables with predefined classes
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(["Under Graduate", "Post Graduate"])
    df['Education'] = label_encoder.transform(df['Education'])

    marital_label_encoder = LabelEncoder()
    marital_label_encoder.classes_ = np.array(["Relationship", "Single"])
    df['Marital_Status'] = marital_label_encoder.transform(df['Marital_Status'])

    # Scale numerical columns
    col_scale = ['Income', 'Kids', 'Expenses', 'TotalAcceptedCmp', 'NumTotalPurchases', 'customer_Age']
    scaler = StandardScaler()
    df[col_scale] = scaler.fit_transform(df[col_scale])

    return df, scaler, label_encoder, marital_label_encoder, col_scale


# Load and preprocess data
data, scaler, label_encoder, marital_label_encoder, col_scale = preprocess_data()

# PCA and KMeans clustering
feature_columns = ['Education', 'Marital_Status'] + col_scale
X_scaled = data[feature_columns]

# Apply PCA
pca = PCA(n_components=3)
PCA_data = pca.fit_transform(X_scaled)

# Fit KMeans
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(PCA_data)

# Streamlit 
st.title("Customer Segmentation")

st.write("### Enter New Customer Details:")
Education = st.selectbox("Education Level", ["Under Graduate", "Post Graduate"])
Marital_Status = st.selectbox("Marital Status", ["Relationship", "Single"])
Income = st.number_input("Income", min_value=0, value=50000)
Kids = st.number_input("Number of Kids", min_value=0, value=0)
Expenses = st.number_input("Total Yearly Expenses", min_value=0, value=1000)
TotalAcceptedCmp = st.number_input("Total Accepted Campaigns", min_value=0, value=0)
NumTotalPurchases = st.number_input("Total Number of Purchases", min_value=0, value=1)
customer_Age = st.number_input("Customer Age", min_value=18, value=35)



# Predict cluster for new customer
if st.button("Predict Cluster"):
    try:
        # Preprocess the input
        new_customer = pd.DataFrame({
            "Education": [label_encoder.transform([Education])[0]],
            "Marital_Status": [marital_label_encoder.transform([Marital_Status])[0]],
            "Income": [Income],
            "Kids": [Kids],
            "Expenses": [Expenses],
            "TotalAcceptedCmp": [TotalAcceptedCmp],
            "NumTotalPurchases": [NumTotalPurchases],
            "customer_Age": [customer_Age],
        })

        # Scale only the numerical columns
        new_customer_scaled = new_customer.copy()
        new_customer_scaled[col_scale] = scaler.transform(new_customer[col_scale])

        # Apply PCA
        new_customer_pca = pca.transform(new_customer_scaled[feature_columns])

        # Predict cluster
        predicted_cluster = kmeans.predict(new_customer_pca)
        cluster_name = f"Cluster {predicted_cluster[0] + 1}"

        st.success(f"The new customer belongs to: {cluster_name} ")
    except Exception as e:
        st.error(f"Error: {e}")
