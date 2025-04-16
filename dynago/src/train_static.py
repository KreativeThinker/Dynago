import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib


def train():
    df = pd.read_csv("dynago/data/normalized.csv")

    X = df.iloc[:, :-1].values  # All columns except the last one
    y = df.iloc[:, -1].values  # Last column is the gesture label

    # Ensure at least 2 classes exist
    if len(set(y)) < 2:
        raise ValueError("Not enough classes to train. Check your dataset.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train the SVM model
    svm_model = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
    svm_model.fit(X_train, y_train)

    # Test the model
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Save the trained model
    MODEL_PATH = "dynago/models/gesture_svm.pkl"
    joblib.dump(svm_model, MODEL_PATH)
    print(f"Model saved at {MODEL_PATH}")


if __name__ == "__main__":
    train()
