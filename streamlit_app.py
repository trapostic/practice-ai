import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import random

# Function to generate synthetic data
def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)
    heights = np.random.normal(160, 10, n_samples)
    weights = np.random.normal(60, 15, n_samples)
    favorite_colors = np.random.choice(['red', 'blue', 'green', 'yellow', 'pink', 'purple'], n_samples)
    favorite_subjects = np.random.choice(['math', 'science', 'english', 'history', 'art', 'sports'], n_samples)
    genders = np.random.choice(['Boy', 'Girl'], n_samples)
    
    data = pd.DataFrame({
        'height': heights,
        'weight': weights,
        'favorite_color': favorite_colors,
        'favorite_subject': favorite_subjects,
        'gender': genders
    })
    
    return data

# Generate synthetic data
data = generate_synthetic_data()

# Preprocess the data
X = pd.get_dummies(data.drop('gender', axis=1))
y = data['gender']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple Decision Tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Evaluate the model (just for reference)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Streamlit App
st.title("Intelligent Student Gender Predictor")

# Input fields
height = st.number_input("Enter Height (cm):", min_value=50, max_value=250, step=1)
weight = st.number_input("Enter Weight (kg):", min_value=20, max_value=150, step=1)
favorite_color = st.selectbox("Select Favorite Color:", ['red', 'blue', 'green', 'yellow', 'pink', 'purple'])
favorite_subject = st.selectbox("Select Favorite Subject:", ['math', 'science', 'english', 'history', 'art', 'sports'])

# Prediction
input_data = pd.DataFrame({
    'height': [height],
    'weight': [weight],
    'favorite_color': [favorite_color],
    'favorite_subject': [favorite_subject]
})
input_data = pd.get_dummies(input_data)
input_data = input_data.reindex(columns=X.columns, fill_value=0)

predicted_gender = model.predict(input_data)[0]

# Language Model Insight (using random responses for demo purposes)
llm_responses = {
    'Boy': "Based on the input features, it appears that the student is likely a boy.",
    'Girl': "Given the data, the student is probably a girl."
}

# Display results
st.success(f"The predicted gender is: {predicted_gender}")
st.info(llm_responses[predicted_gender])

# Show model accuracy (for reference)
st.write(f"Model accuracy (on test data): {accuracy:.2%}")

# Additional Information
st.info("Note: This prediction is based on a synthetic dataset and a simple decision tree model.")
