# Student Performance Prediction using Logistic Regression

## Project Overview

This project uses Logistic Regression to classify students into Pass and Fail categories based on their academic and demographic information.

## Dataset

The project uses the Students Performance in Exams dataset.

The dataset contains 1000 student records and the following columns:

* gender
* race/ethnicity
* parental level of education
* lunch
* test preparation course
* math score
* reading score
* writing score

## Objective

The objective of this project is to implement a complete machine learning classification workflow using Logistic Regression.

## Project Workflow

1. Data Loading
2. Data Cleaning
3. Exploratory Data Analysis
4. Feature Engineering
5. Target Variable Creation
6. Categorical Encoding
7. Train-Test Split
8. Feature Scaling
9. Logistic Regression Model Training
10. Prediction
11. Model Evaluation

## Data Preprocessing

The dataset was checked for:

* Missing values
* Duplicate records
* Unique categorical values
* Descriptive statistics

An Average score was calculated using the Math, Reading, and Writing scores.

The target variable was created using the following rule:

* Average score >= 50 → Pass (1)
* Average score < 50 → Fail (0)

## Feature Encoding

Label Encoding was applied to:

* gender
* lunch

One-Hot Encoding was applied to:

* race/ethnicity
* parental level of education
* test preparation course

After encoding, the dataset contained 15 input features.

## Model

Logistic Regression was used for binary classification.

The dataset was divided into:

* 80% training data
* 20% testing data

StandardScaler was used for feature scaling.

## Model Evaluation

The model was evaluated using:

* Accuracy
* Confusion Matrix
* Precision
* Recall
* F1-score

Actual and predicted values were also compared.

## Important Note

The target variable was created from the Average score, and the Average score was calculated from the Math, Reading, and Writing scores.

Since these three scores are also used as input features, this introduces target leakage.

Therefore, this project is mainly intended as a Logistic Regression classification and machine learning practice project rather than a realistic future student-performance prediction system.

## Project Files

```text
student-performance-prediction/
│
├── Student_Performance_Prediction.ipynb
├── StudentsPerformance.csv
├── student_performance_logistic_model.pkl
├── student_performance_scaler.pkl
└── README.md
```

## Libraries Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib

## Author

Govardhan Goud

B.Tech in Artificial Intelligence and Data Science
