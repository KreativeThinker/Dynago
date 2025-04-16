import math
import csv
import multiprocessing
import os
import joblib
import numpy as np
from collections import deque

from dynago.config import VEL_THRESHOLD

model = joblib.load("dynago/models/swipe_svm.pkl")

# Data structures
mean_landmark_history = deque(maxlen=10)
landmark_history = deque(maxlen=10)

# CSV file setup
DATA_DIR = "dynago/data"
DATA_FILE = os.path.join(DATA_DIR, "swipe.csv")
os.makedirs(DATA_DIR, exist_ok=True)

# Queue for parallel data saving
save_data_queue = multiprocessing.Queue()


def data_saver_process():
    """Background process to save gesture data efficiently"""
    with open(DATA_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        while True:
            data = save_data_queue.get()
            if data is None:  # Termination signal
                break

            writer.writerow(
                [data["gesture_id"], data["swipe_direction"], *data["landmarks"]]
            )


# Start the saver process
saver = multiprocessing.Process(target=data_saver_process)
saver.start()


def get_tracking_point(raw_landmarks, indices):
    """Get tracking points and update history"""
    points = [raw_landmarks[i][:2] for i in indices if i < len(raw_landmarks)]
    if points:
        landmark_history.append(raw_landmarks)  # Store full landmarks
        return np.mean(points, axis=0)
    return raw_landmarks[0][:2]


# def calculate_swipe_direction(gesture_id):
#     flat_landmarks = []
#     for landmarks in landmark_history:
#         for point in landmarks:
#             flat_landmarks.extend(point)  # Only use x and y coordinates
#
#     # Ensure you're flattening the correct number of features
#     if len(flat_landmarks) != 630:  # If it's not 630, this will raise a flag
#         # print(f"Warning: Features mismatch! Expected 630, got {len(flat_landmarks)}")
#         return None
#
#     # Make prediction using the trained model
#     X = np.array(flat_landmarks).reshape(1, -1)  # Reshape for single prediction
#     prediction = model.predict(X)
#
#     # Decode the gesture prediction
#     predicted_gesture = prediction[0]  # Assuming single output from the model
#
#     if predicted_gesture[0] != gesture_id:
#         print(f"Geseture Mismatch: Got {predicted_gesture[0]} expected {gesture_id}")
#
#     # Print or log the prediction for debugging
#     print(f"Predicted Gesture: {predicted_gesture}")
#     landmark_history.clear()
#     return predicted_gesture[1]


def calculate_swipe_direction(gesture_id):
    """Calculate direction and store data for training"""
    if len(mean_landmark_history) < 2:
        return None

    # Calculate direction
    start, end = mean_landmark_history[0], mean_landmark_history[-1]
    dx, dy = end[0] - start[0], end[1] - start[1]
    magnitude = math.sqrt(dx**2 + dy**2)

    if magnitude < VEL_THRESHOLD:
        return None

    angle = math.degrees(math.atan2(dy, dx))
    direction = None

    if 150 <= angle <= 180 or -180 <= angle <= -150:
        direction = 0  # Left
    elif -30 <= angle <= 30:
        direction = 1  # Right
    elif 60 < angle < 120:
        direction = 2  # Down
    elif -120 < angle < -60:
        direction = 3  # Up

    if direction is not None and landmark_history:
        # Send data to saver process
        data = {
            "gesture_id": gesture_id,
            "swipe_direction": direction,
            "landmarks": list(landmark_history),
        }
        save_data_queue.put(data)
        landmark_history.clear()

    return direction


def cleanup():
    """Call this when exiting the application"""
    save_data_queue.put(None)
    saver.join()
