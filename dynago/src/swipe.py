import math
import csv
import multiprocessing
import os
import numpy as np
from collections import deque
from dynago.config import VEL_THRESHOLD

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
            # Flatten landmarks and convert to strings
            flat_landmarks = [
                f"{coord:.6f}" for point in data["landmarks"] for coord in point
            ]
            writer.writerow(
                [data["gesture_id"], data["swipe_direction"], *flat_landmarks]
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
        save_data_queue.put(
            {
                "gesture_id": gesture_id,
                "swipe_direction": direction,
                "landmarks": landmark_history[-1],  # Most recent full landmarks
            }
        )

    return direction


def cleanup():
    """Call this when exiting the application"""
    save_data_queue.put(None)
    saver.join()
