import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt

# Function to generate synthetic data
def generate_synthetic_data(n_samples=5000):
    np.random.seed(42)
    heights = np.random.normal(160, 10, n_samples)
    weights = np.random.normal(60, 15, n_samples)
    favorite_colors = np.random.choice(['red', 'blue', 'green', 'yellow', 'pink', 'purple'], n_samples)
    favorite_subjects = np.random.choice(['math', 'science', 'english', 'history', 'art', 'sports'], n_samples)
    genders = np.random.choice(['Boy', 'Girl'], n_samples)
    
    # New features
    shoe_sizes = np.where(genders == 'Boy', np.random.normal(42, 2, n_samples), np.random.normal(38, 2, n_samples))
    hair_lengths = np.where(genders == 'Boy', np.random.normal(5, 2, n_samples), np.random.normal(15, 5, n_samples))
    
    data = pd.DataFrame({
        'height': heights,
        'weight': weights,
        'favorite_color': favorite_colors,
        'favorite_subject': favorite_subjects,
        'shoe_size': shoe_sizes,
        'hair_length': hair_lengths,
        'gender': genders
    })
    
    return data

# Generate synthetic data
data = generate_synthetic_data()

# Feature engineering
data['bmi'] = data['weight'] / ((data['height'] / 100) ** 2)
data['height_to_shoe_size_ratio'] = data['height'] / data['shoe_size']

# Preprocess the data
X = pd.get_dummies(data.drop('gender', axis=1))
y = data['gender']

# Standardize numerical features
scaler = StandardScaler()
numerical_features = ['height', 'weight', 'shoe_size', 'hair_length', 'bmi', 'height_to_shoe_size_ratio']
X[numerical_features] = scaler.fit_transform(X[numerical_features])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Calculate feature importances
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
feature_importance = feature_importance.sort_values('importance', ascending=False).reset_index(drop=True)

# Streamlit App
st.title("Advanced Intelligent Student Gender Predictor")

# Input fields
height = st.number_input("Enter Height (cm):", min_value=50, max_value=250, step=1)
weight = st.number_input("Enter Weight (kg):", min_value=20, max_value=150, step=1)
favorite_color = st.selectbox("Select Favorite Color:", ['red', 'blue', 'green', 'yellow', 'pink', 'purple'])
favorite_subject = st.selectbox("Select Favorite Subject:", ['math', 'science', 'english', 'history', 'art', 'sports'])
shoe_size = st.number_input("Enter Shoe Size (EU):", min_value=20, max_value=50, step=1)
hair_length = st.number_input("Enter Hair Length (cm):", min_value=0, max_value=100, step=1)

# Prediction
input_data = pd.DataFrame({
    'height': [height],
    'weight': [weight],
    'favorite_color': [favorite_color],
    'favorite_subject': [favorite_subject],
    'shoe_size': [shoe_size],
    'hair_length': [hair_length]
})

# Feature engineering for input data
input_data['bmi'] = input_data['weight'] / ((input_data['height'] / 100) ** 2)
input_data['height_to_shoe_size_ratio'] = input_data['height'] / input_data['shoe_size']

input_data = pd.get_dummies(input_data)
input_data = input_data.reindex(columns=X.columns, fill_value=0)

# Standardize numerical features of input data
input_data[numerical_features] = scaler.transform(input_data[numerical_features])

predicted_gender = model.predict(input_data)[0]
prediction_proba = model.predict_proba(input_data)[0]

# Display results
st.success(f"The predicted gender is: {predicted_gender}")
st.info(f"Probability of Boy: {prediction_proba[0]:.2%}, Probability of Girl: {prediction_proba[1]:.2%}")

# Show model performance metrics
st.subheader("Model Performance")
st.write(f"Model accuracy (on test data): {accuracy:.2%}")
st.write("Confusion Matrix:")
st.write(conf_matrix)
st.write("Classification Report:")
st.text(class_report)

# Show feature importances
st.subheader("Feature Importances")
fig, ax = plt.subplots()
feature_importance[:10].plot(x='feature', y='importance', kind='bar', ax=ax)
plt.title("Top 10 Most Important Features")
plt.tight_layout()
st.pyplot(fig)

# SHAP values for model interpretability
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(input_data)

st.subheader("SHAP Feature Importance")
fig, ax = plt.subplots()
shap.summary_plot(shap_values[1], input_data, plot_type="bar", show=False)
plt.title("SHAP Feature Importance")
st.pyplot(fig)

# Additional Information
st.info("Note: This prediction is based on a synthetic dataset and a Random Forest model. The model's performance and feature importances provide insights into the prediction process.")