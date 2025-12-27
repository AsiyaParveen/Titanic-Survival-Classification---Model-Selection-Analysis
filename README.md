# Titanic-Survival-Classification---Model-Selection-Analysis
Project Title: Titanic Survival Classification - Model Selection Analysis

This project demonstrates a comprehensive workflow for evaluating and selecting the best classification model for the Titanic dataset.



Workflow Overview:


Exploratory Data Analysis (EDA): Visualized survival counts by sex and passenger class, and analyzed age and fare distributions.


Data Wrangling: Imputed missing values for 'age' (mean) and 'embarked' (mode), and dropped the 'deck' column due to excessive missing data.

Model Training: implemented five different classifiers using scikit-learn:

Logistic Regression

Random Forest Classifier

Decision Tree Classifier

K-Neighbors Classifier

Support Vector Classifier (SVC) 

Performance Results:
Accuracy: Logistic Regression & Random Forest (~0.81).
Precision: Logistic Regression led with ~0.80.
Recall: Logistic Regression & Random Forest led with ~0.716.
F1-Score: Random Forest achieved the highest score of ~0.76.
Conclusion: For this dataset, Logistic Regression and Random Forest proved to be the most robust models across multiple metrics. Confusion matrices were also generated to visualize the true vs. false positives/negatives for each model
Tools Used: Python, Pandas, Seaborn, Matplotlib, Scikit-learn
