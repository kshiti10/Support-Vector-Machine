# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 16:08:41 2024

@author: Kshitija
""
Problem Statement:
 Prepare a classification model using the Naive Bayes algorithm for the salary dataset. Train 
and test datasets are given separately. Use both for model
building. 
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
##
salary_data_train=pd.read_csv("C:/11-Naive Bayes Classifier/SalaryData_Test.csv",encoding="ISO-8859-1")
salary_data_test=pd.read_csv("C:/11-Naive Bayes Classifier/SalaryData_Train.csv",encoding="ISO-8859-1")

# Separate features (X) and target variable (y) for both datasets
X_train = salary_data_train.drop(columns=['Salary'])  # Features for training data
y_train = salary_data_train['Salary']  # Target variable for training data
X_test = salary_data_test.drop(columns=['Salary'])  # Features for testing data

# Initialize LabelEncoder to convert categorical variables to numerical values
label_encoder = LabelEncoder()

# Encode categorical variables in both training and testing datasets
for col in X_train.select_dtypes(include=['object']).columns:
    X_train[col] = label_encoder.fit_transform(X_train[col])  # Transform and encode training data
    X_test[col] = label_encoder.transform(X_test[col])  # Transform testing data based on training data's encoding

# Train-test split for validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Initialize and train the Naive Bayes classifier
# Initialize Naive Bayes classifier
nb_classifier = GaussianNB()  
# Train the classifier using training data
nb_classifier.fit(X_train, y_train) 

# Predictions on the validation set
y_pred = nb_classifier.predict(X_val)  # Make predictions on the validation set

# Evaluate classifier performance (e.g., accuracy)
# Calculate accuracy by comparing predicted values with actual values
accuracy = (y_pred == y_val).mean() 
# Print the accuracy of the classifier on the validation set
print("Accuracy:", accuracy)  