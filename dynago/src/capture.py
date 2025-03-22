import cv2
import os
import mediapipe as mp
import numpy as np
import joblib
import json
import time
from dynago.config import N_FRAMES
from dynago.src.swipe import (
    get_tracking_point,
    calculate_swipe_direction,
    landmark_history,
)

MODEL_PATH = "dynago/models/gesture_svm.pkl"
GESTURE_MAP_PATH = "dynago/data/gesture_map.json"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()


with open(GESTURE_MAP_PATH, "r") as f:
    gesture_map = json.load(f)
    gesture_map = {int(k): v for k, v in gesture_map.items()}


def normalize_landmarks(landmarks):
    num_landmarks = 21
    landmarks = np.array(landmarks).reshape(num_landmarks, 3)
    wrist = landmarks[0]
    landmarks -= wrist
    max_dist = np.max(np.linalg.norm(landmarks, axis=1))
    landmarks /= max_dist if max_dist > 0 else 1
    return landmarks.flatten()


def predict_gesture(input_data):
    svm_model = joblib.load(MODEL_PATH)
    input_data = np.array(input_data).reshape(1, -1)
    prediction = svm_model.predict(input_data)
    return prediction[0]


def capture_landmarks():
    cap = cv2.VideoCapture(0)
    frame_count = 0
    tracking_motion = False
    current_tracking_indices = None
    current_static_gesture = ""
    cooldown_until = time.time()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        if time.time() < cooldown_until:
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                raw_landmarks = [(lm.x, lm.y, lm.z) for lm in landmarks.landmark]
                norm_landmarks = normalize_landmarks(raw_landmarks)

                if tracking_motion:
                    if current_tracking_indices is not None:
                        tracking_point = get_tracking_point(
                            raw_landmarks, current_tracking_indices
                        )
                    else:
                        tracking_point = raw_landmarks[0][:2]
                    landmark_history.append(tracking_point)
                    swipe = calculate_swipe_direction()
                    if swipe:
                        print(
                            f'{current_static_gesture} swipe "{swipe}"', flush=True
                        )  # Fix swipe output format
                        cooldown_until = (
                            time.time() + 2
                        )  # Set cooldown period (2 second)
                        tracking_motion = False
                        current_tracking_indices = None
                        landmark_history.clear()
                elif frame_count % N_FRAMES == 0:
                    gesture = predict_gesture(norm_landmarks)
                    if gesture == 0:
                        print(".", end="", flush=True)  # Corrected to print a dot
                        tracking_motion = False
                    else:
                        mapping = gesture_map.get(int(gesture), None)
                        if mapping:
                            current_static_gesture = mapping.get("name", "Unknown")
                            current_tracking_indices = mapping.get("landmarks", [0])
                        else:
                            current_static_gesture = str(gesture)
                            current_tracking_indices = [0]

                        print(
                            f'Detected: "{current_static_gesture}"'
                        )  # Fix gesture detection output
                        tracking_motion = True
                        landmark_history.clear()
                        tracking_point = get_tracking_point(
                            raw_landmarks, current_tracking_indices
                        )
                        landmark_history.append(tracking_point)

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
