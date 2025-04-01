import math
import numpy as np
from collections import deque
from dynago.config import VEL_THRESHOLD

landmark_history = deque(maxlen=10)


def get_tracking_point(raw_landmarks, indices):
    points = [raw_landmarks[i][:2] for i in indices if i < len(raw_landmarks)]
    if points:
        return np.mean(points, axis=0)
    return raw_landmarks[0][:2]


def calculate_swipe_direction():
    if len(landmark_history) < 2:
        return None
    start, end = landmark_history[0], landmark_history[-1]
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    magnitude = math.sqrt(dx**2 + dy**2)
    if magnitude < VEL_THRESHOLD:
        return None  # Movement too small
    angle = math.degrees(math.atan2(dy, dx))
    if 150 <= angle <= 180 or -180 <= angle <= -150:
        return 0
    elif -30 <= angle <= 30:
        return 1
    elif 60 < angle < 120:
        return 2
    elif -120 < angle < -60:
        return 3
    return None
