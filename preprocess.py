# preprocess.py

import os
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib

def main():
    # Load the Boston Housing dataset
    # Note: load_boston is deprecated in newer sklearn; using here for illustration.
    data = load_boston()
    features = data.data      # shape (506, 13)
    target = data.target      # shape (506,)

    # Split into training and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )  # train_test_split shuffles and splits the data:contentReference[oaicite:2]{index=2}

    # Handle missing values (placeholder): impute missing values with column mean
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)  # fit on train data
    X_test = imputer.transform(X_test)        # apply same transformation to test data

    # Scale features to mean=0, variance=1 using StandardScaler:contentReference[oaicite:3]{index=3}
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # compute mean/scale on training data
    X_test_scaled = scaler.transform(X_test)        # apply same scaling to test data

    # Ensure the output directory exists
    os.makedirs('data', exist_ok=True)

    # Save the preprocessed data (training and test sets) to disk
    joblib.dump((X_train_scaled, X_test_scaled, y_train, y_test), 'data/preprocessed_data.pkl')
    print("Preprocessing complete. Data saved to 'data/preprocessed_data.pkl'.")

if __name__ == "__main__":
    main()
