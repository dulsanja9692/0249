import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ----------------------------
# Load model and scaler
# ----------------------------
with open('best_model .pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load dataset
df = pd.read_csv("data/diabetes.csv")



# ----------------------------
# Streamlit App
# ----------------------------
st.set_page_config(page_title="Diabetes Prediction App", layout="wide")

# Title & Description
st.title("🩺 Diabetes Prediction App")
st.markdown("""
This application allows you to:
- Explore the diabetes dataset
- Visualise trends and patterns
- Predict diabetes probability for new inputs
- View model performance and comparisons
""")

# Sidebar Navigation
menu = st.sidebar.radio("Navigation", ["Data Exploration", "Visualisation", "Model Prediction", "Model Performance"])

# ----------------------------
# Data Exploration Section
# ----------------------------
if menu == "Data Exploration":
    st.subheader("Dataset Overview")
    st.write("**Shape:**", df.shape)
    st.write("**Columns:**", df.columns.tolist())
    st.write("**Data Types:**")
    st.write(df.dtypes)

    st.subheader("Sample Data")
    st.dataframe(df.head())

    st.subheader("Interactive Filtering")
    column_to_filter = st.selectbox("Select column to filter", df.columns)
    if df[column_to_filter].dtype in [np.int64, np.float64]:
        min_val, max_val = st.slider("Select value range", float(df[column_to_filter].min()), float(df[column_to_filter].max()), (float(df[column_to_filter].min()), float(df[column_to_filter].max())))
        filtered_df = df[(df[column_to_filter] >= min_val) & (df[column_to_filter] <= max_val)]
    else:
        unique_vals = df[column_to_filter].unique()
        selected_vals = st.multiselect("Select values", unique_vals, default=unique_vals)
        filtered_df = df[df[column_to_filter].isin(selected_vals)]
    st.write(filtered_df)

# ----------------------------
# Visualisation Section
# ----------------------------
elif menu == "Visualisation":
    st.subheader("data Visualisations")
    
    # Chart 1: Outcome distribution
    fig, ax = plt.subplots()
    sns.countplot(x='Outcome', data=df, ax=ax)
    st.pyplot(fig)
    
    # Chart 2: Correlation heatmap
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    
    # Chart 3: BMI vs Glucose scatter
    fig, ax = plt.subplots()
    sns.scatterplot(x='BMI', y='Glucose', hue='Outcome', data=df, ax=ax)
    st.pyplot(fig)

# ----------------------------
# Model Prediction Section
# ----------------------------
elif menu == "Model Prediction":
    st.subheader("Make a Prediction")
    st.markdown("Enter feature values below:")
    
    input_data = {}
    for col in df.drop('Outcome', axis=1).columns:
        val = st.number_input(f"Enter {col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
        input_data[col] = val

    if st.button("Predict"):
        try:
            input_df = pd.DataFrame([input_data])
            scaled_input = scaler.transform(input_df)
            prediction = model.predict(scaled_input)[0]
            probability = model.predict_proba(scaled_input)[0][prediction] if hasattr(model, 'predict_proba') else None
            
            st.write("**Prediction:**", "Diabetic" if prediction == 1 else "Not Diabetic")
            if probability is not None:
                st.write(f"**Confidence:** {probability:.2f}")
        except Exception as e:
            st.error(f"Error making prediction: {e}")

# ----------------------------
# Model Performance Section
# ----------------------------
elif menu == "Model Performance":
    st.subheader("Model Evaluation Metrics")
    
    # Prepare data
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    
    acc = accuracy_score(y, y_pred)
    st.write("**Accuracy:**", acc)
    
    st.write("**Classification Report:**")
    st.text(classification_report(y, y_pred))
    
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'], ax=ax)
    st.pyplot(fig)
