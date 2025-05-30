import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib


def train():
    # Load dataset
    df = pd.read_csv("dynago/data/swipe.csv", header=None)
    df.columns = [
        "gesture",
        "direction",
        "p1",
        "p2",
        "p3",
        "p4",
        "p5",
        "p6",
        "p7",
        "p8",
        "p9",
        "p10",
    ]

    # Encode gesture and direction labels
    le_gesture = LabelEncoder()
    le_direction = LabelEncoder()
    y = pd.DataFrame(
        {
            "gesture": le_gesture.fit_transform(df["gesture"]),
            "direction": le_direction.fit_transform(df["direction"]),
        }
    )

    # Process 3D keypoint lists into flat arrays
    def parse_and_flatten(row):
        coords = []
        for i in ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10"]:
            try:
                parsed = literal_eval(row[i])
            except:
                print(f"Skipping {i}")
            coords.extend([coord for point in parsed for coord in point])
        return coords

    X = np.array(df.apply(parse_and_flatten, axis=1).to_list())

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Choose model
    base_model = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
    model = MultiOutputRegressor(base_model)

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Calculate accuracy for each output
    gesture_accuracy = accuracy_score(
        y_test["gesture"], y_pred[:, 0].round().astype(int)
    )
    direction_accuracy = accuracy_score(
        y_test["direction"], y_pred[:, 1].round().astype(int)
    )
    overall_accuracy = (gesture_accuracy + direction_accuracy) / 2

    # Decode results if needed
    decoded_preds = pd.DataFrame(
        {
            "gesture": le_gesture.inverse_transform(y_pred[:, 0].round().astype(int)),
            "direction": le_direction.inverse_transform(
                y_pred[:, 1].round().astype(int)
            ),
        }
    )

    MODEL_PATH = "dynago/models/swipe_svm.pkl"
    joblib.dump(model, MODEL_PATH)

    print(decoded_preds.head())
    print(f"\nGesture Accuracy: {gesture_accuracy:.2f}")
    print(f"Direction Accuracy: {direction_accuracy:.2f}")
    print(f"Overall Accuracy: {overall_accuracy:.2f}")

    accuracy = {
        "gesture_accuracy": gesture_accuracy,
        "direction_accuracy": direction_accuracy,
        "overall_accuracy": overall_accuracy,
    }

    __import__("pprint").pprint(accuracy)


# Call the training function
if __name__ == "__main__":
    train()
