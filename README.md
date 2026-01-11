💳 Credit Card Fraud Detection using Logistic Regression

This project uses the fraudTrain.csv dataset to detect whether a credit card transaction is fraudulent (1) or genuine (0) based on transaction and customer details.
A Logistic Regression model is trained and evaluated using Confusion Matrix, Classification Report, and ROC-AUC Score.

📌 Project Overview

Dataset: fraudTrain.csv

Target variable: is_fraud (0 = Genuine, 1 = Fraud)

Filtering rule: only transactions with amt > 50

Feature Engineering:

hour = extracted from transaction time (trans_date_trans_time)

age = computed using dob and transaction date

Categorical Encoding:

Label Encoding for category, gender, job

Model used:

Logistic Regression (class_weight='balanced') to handle imbalanced data

🛠 Tools & Libraries Used

Python

Pandas / NumPy → data cleaning & preprocessing

Scikit-learn → train-test split, scaling, model training, evaluation metrics

Matplotlib → fraud location visualization (lat-long scatter plot)

📊 Results

ROC-AUC Score: 0.9774

Accuracy: 96%

The model achieved high recall for fraud transactions (93%), meaning it detects most fraud cases effectively.

A Lat-Long scatter plot was used to visualize transaction locations and highlight fraud patterns.
