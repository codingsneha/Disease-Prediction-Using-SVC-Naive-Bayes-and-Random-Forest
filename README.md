
---

# Disease Prediction Machine Learning Model

This python project implements a robust machine-learning model that efficiently predicts human diseases based on symptoms.

## Table of Contents
- [Deployment](#deployment)
- [Introduction](#introduction)
- [Approach](#approach)
- [Implementation](#implementation)

## Deployment
The model is deployed using Streamlit and accessible at [this link](https://codingsneha-disease-prediction-ml.streamlit.app/).

## Introduction
The goal of this project is to develop a machine learning model capable of accurately predicting diseases based on the symptoms exhibited by patients. The model utilizes a [dataset](https://www.kaggle.com/kaushil268/disease-prediction-using-machine-learning/) from Kaggle containing information on various symptoms and corresponding disease prognoses.

## Approach
- **Gathering the Data:** Utilized a dataset from Kaggle consisting of two CSV files for training and testing, containing 133 columns including symptoms and disease prognoses.
- **Cleaning the Data:** Processed the data by removing null values and encoding the target column into numerical form using a label encoder.
- **Model Building:** Trained Support Vector Classifier, Naive Bayes Classifier, and Random Forest Classifier using the cleaned data. Evaluated models using cross-validation and combined predictions for robustness.
- **Inference:** Predicted diseases for input symptoms by combining predictions from all three models. Defined a function to take symptoms as input, predict diseases, and return predictions in JSON format.

## Implementation
### Libraries Used
- `numpy`
- `pandas`
- `scipy.stats`
- `matplotlib`
- `seaborn`
- `sklearn`

### Code Snippets in the .ipynb File
- Importing libraries and reading the dataset.
- Checking dataset balance and encoding the target value.
- Splitting the data for training and testing.
- Model building using Support Vector Classifier, Naive Bayes Classifier, and Random Forest Classifier.
- Combining predictions for robust classification.
- Defining a function to predict diseases based on input symptoms.

---
