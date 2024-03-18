import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from collections import Counter

# Read the dataset
DATA_PATH = "dataset/Training.csv"
data = pd.read_csv(DATA_PATH).dropna(axis=1)

# Encode the target variable
encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])

# Split the data
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

# Initialize models
final_svm_model = SVC()
final_nb_model = GaussianNB()
final_rf_model = RandomForestClassifier(random_state=18)
final_gbm_model = GradientBoostingClassifier(random_state=42)  # Add GBM model

# Train the models
final_svm_model.fit(X_train, y_train)
final_nb_model.fit(X_train, y_train)
final_rf_model.fit(X_train, y_train)
final_gbm_model.fit(X_train, y_train)  # Train GBM model

# Create a symptom index dictionary
symptoms = X.columns.values
symptom_index = {symptom.capitalize(): index for index, symptom in enumerate(symptoms)}

# Custom mode function for strings
def mode_str(data):
    counter = Counter(data)
    max_count = max(counter.values())
    return [key for key, val in counter.items() if val == max_count]

# Define the function to predict disease
def predictDisease(symptoms):
    symptoms = symptoms.split(",")
    input_data = [0] * len(symptom_index)
    for symptom in symptoms:
        index = symptom_index.get(symptom.capitalize(), None)
        if index is not None:
            input_data[index] = 1
    input_data = np.array(input_data).reshape(1, -1)

    rf_prediction = encoder.inverse_transform([final_rf_model.predict(input_data)])[0]
    nb_prediction = encoder.inverse_transform([final_nb_model.predict(input_data)])[0]
    svm_prediction = encoder.inverse_transform([final_svm_model.predict(input_data)])[0]
    gbm_prediction = encoder.inverse_transform([final_gbm_model.predict(input_data)])[0]  # Predict using GBM

    final_prediction = mode_str([rf_prediction, nb_prediction, svm_prediction, gbm_prediction])[0]

    predictions = {
        "rf_model_prediction": rf_prediction,
        "naive_bayes_prediction": nb_prediction,
        "svm_model_prediction": svm_prediction,
        "gbm_model_prediction": gbm_prediction,  # Include GBM prediction
        "final_prediction": final_prediction
    }
    return predictions

# Define the Streamlit app
def main():
    st.title("Disease Prediction App")
    symptoms_input = st.text_input("Enter symptoms separated by commas:")
    if st.button("Predict"):
        predictions = predictDisease(symptoms_input)
        st.write("### Predictions:")
        st.write(f"RF Model Prediction: {predictions['rf_model_prediction']}")
        st.write(f"Naive Bayes Model Prediction: {predictions['naive_bayes_prediction']}")
        st.write(f"SVM Model Prediction: {predictions['svm_model_prediction']}")
        st.write(f"GBM Model Prediction: {predictions['gbm_model_prediction']}")  # Display GBM prediction
        st.write(f"Final Prediction: {predictions['final_prediction']}")

if __name__ == "__main__":
    main()
