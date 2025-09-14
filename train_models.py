# train_models.py

import os
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

def main():
    # Load preprocessed training data
    # We only need the training portion here (but load both for simplicity)
    X_train, X_test, y_train, y_test = joblib.load('data/preprocessed_data.pkl')

    # Create output directory for models if it doesn't exist
    os.makedirs('models', exist_ok=True)

    # Initialize the regression models
    lin_reg = LinearRegression()                       # Ordinary Least Squares linear regression
    tree_reg = DecisionTreeRegressor(random_state=42)  # Decision Tree regressor
    rf_reg = RandomForestRegressor(random_state=42)    # Random Forest regressor

    # Train each model on the training data
    lin_reg.fit(X_train, y_train)
    tree_reg.fit(X_train, y_train)
    rf_reg.fit(X_train, y_train)

    # Save trained models to disk
    joblib.dump(lin_reg, 'models/linear_regression_model.pkl')
    joblib.dump(tree_reg, 'models/decision_tree_model.pkl')
    joblib.dump(rf_reg, 'models/random_forest_model.pkl')

    print("Training complete. Models saved to 'models/' directory.")

if __name__ == "__main__":
    main()
