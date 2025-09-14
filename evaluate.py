# evaluate.py

import joblib
from sklearn.metrics import mean_squared_error, r2_score

def main():
    # Load preprocessed data (including test set)
    X_train, X_test, y_train, y_test = joblib.load('data/preprocessed_data.pkl')

    # Load each trained model
    lin_reg = joblib.load('models/linear_regression_model.pkl')
    tree_reg = joblib.load('models/decision_tree_model.pkl')
    rf_reg = joblib.load('models/random_forest_model.pkl')

    # Create a list of (model_name, model) for easy iteration
    models = [
        ("Linear Regression", lin_reg),
        ("Decision Tree", tree_reg),
        ("Random Forest", rf_reg)
    ]

    # Evaluate each model on the test set
    print("Model Evaluation on Test Data:")
    for name, model in models:
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)   # Mean squared error:contentReference[oaicite:6]{index=6}
        r2 = r2_score(y_test, y_pred)             # R^2 score (coefficient of determination):contentReference[oaicite:7]{index=7}
        print(f"{name} -> MSE: {mse:.2f}, R^2: {r2:.2f}")

if __name__ == "__main__":
    main()
