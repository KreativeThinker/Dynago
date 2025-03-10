import joblib
import numpy as np
from dynago.src.capture import from_video

MODEL_PATH = "dynago/models/gesture_svm.pkl"


def predict_gesture(input_data):
    svm_model = joblib.load(MODEL_PATH)
    input_data = np.array(input_data).reshape(1, -1)
    prediction = svm_model.predict(input_data)

    return prediction[0]


def main():
    sample_input = from_video()
    gesture = predict_gesture(sample_input)
    print(f"Predicted Gesture: {gesture}")
