# House-Price-Prediction
 Dataset: Boston Housing Dataset (or any housing dataset). Goal: Predict the price of houses given features like rooms, area, etc.

Project Overview

This project builds regression models to predict house prices (median value) using the classic Boston Housing dataset.We use features of homes (e.g. crime rate, room count) to predict the median home value (MEDV). The goal is to compare different regression algorithms on this data, illustrating supervised learning for predicting continuous targets. The Boston Housing data comes from the UCI Machine Learning Repository (506 samples),making it a standard benchmark for regression tasks.

Dataset Description

The Boston Housing dataset contains 506 examples of homes in the Boston area Each sample has 13 numeric features and one target value. Key features include

CRIM: per capita crime rate by town

RM: average number of rooms per dwelling

PTRATIO: pupil–teacher ratio by town

The target variable is MEDV – the median value of owner-occupied homes in $1000’s

In summary, this dataset has 506 samples, 13 features, and 1 continuous target

Preprocessing

Before training, we preprocess the data as follows:

Missing values: The original Boston dataset has no missing entries
If new missing values appear (e.g. in extended data), we would handle them by imputation (such as filling with the mean of that feature) or by dropping rows with missing data.

Feature scaling: We standardize all numeric features to have zero mean and unit variance. Standardization is a common step because many algorithms (like linear regression) perform better when features are normalized In practice we use StandardScaler from scikit-learn to achieve this.

Model Training

We train three different regression models on the preprocessed data:

Linear Regression: Ordinary least squares linear model. It fits a linear function minimizing the sum of squared residuals between predicted and actual values

Decision Tree Regressor: A tree-based model that splits the data into regions based on feature values (using squared-error criterion by default).

Random Forest Regressor: An ensemble of decision trees. It builds many trees on random subsets and averages their predictions to improve accuracy and reduce overfitting

Each model is trained on the training set and tuned (using default settings or simple parameter choices) to predict the MEDV target.

Evaluation Metrics

We evaluate each model using two common regression metrics:

Mean Squared Error (MSE): the average of the squared differences between predicted and actual values A lower MSE indicates better fit (zero is ideal).

R² Score: the coefficient of determination. It measures the proportion of variance in the target explained by the model
An R² of 1.0 means perfect prediction, while 0.0 means the model does no better than predicting the mean.

By comparing MSE and R² on a held-out test set, we assess how well each model predicts house prices.

Installation Guide

1. Install Python 3: Ensure you have Python 3.x installed (download from python.org if needed).
2. Create a virtual environment (recommended):
   python3 -m venv env
   source env/bin/activate   # On Windows use: env\Scripts\activate

4. Install dependencies: Use pip to install required packages from requirements.txt:
   pip install --upgrade pip
   pip install -r requirements.txt


   How to Run: Prepare the data: (If a preprocessing script is provided) run: python preprocess.py

   Train the models: Run the training script python train_models.py

   Evaluate the models: Run the evaluation script python evaluate.py

Requirements

pandas
numpy
scikit-learn
matplotlib
scipy









