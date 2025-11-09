# Ethical Analysis of Loan Approval Dataset

## Project Overview

This project analyzes a loan approval dataset to identify and quantify potential ethical biases in the decision-making process. The primary goal is to determine if predictive models, even when not explicitly trained on a sensitive attribute like gender, exhibit bias in their outcomes.

This analysis is conducted in [loan_notebook.ipynb](./loan_notebook.ipynb).

## Analysis Workflow

* Feature Engineering: A gender feature is engineered using the gender_guesser library to serve as a sensitive attribute for auditing.
* Exploratory Data Analysis (EDA): Investigates the dataset for initial signs of bias, including differences in approval rates and financial feature distributions (e.g., income, credit score) between genders.
* Model Training: Three models (SVM, Random Forest, Logistic Regression) are trained on the data without the gender feature.
* Bias & Fairness Analysis: The trained models are audited for fairness using the test set's sensitive attribute. The analysis measures:
* Disparate Impact (Statistical Parity): Do approval rates differ between groups?
* Disparate Mistreatment (Accuracy): Do error rates differ between groups?
* Feature Importance: The models are analyzed to understand which features (e.g., loan_to_points, dti_ratio) were the most powerful predictors, explaining how bias might be learned through proxy features.

## Key Findings

* Performance: The Logistic Regression model achieved the highest overall accuracy (99%).
* Fairness: The Logistic Regression model also proved to be the fairest, demonstrating the smallest gaps in both approval rates (Disparate Impact) and error rates (Disparate Mistreatment) between genders.

## Dependencies

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* gender_guesser
* kagglehub