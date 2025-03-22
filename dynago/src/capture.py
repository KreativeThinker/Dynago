import cv2
import mediapipe as mp
import numpy as np
import joblib
import collections
import math
import json
import time

MODEL_PATH = "dynago/models/gesture_svm.pkl"
GESTURE_MAP_PATH = "dynago/data/gesture_map.json"
N_FRAMES = 10  # Static gesture classification every N frames
VEL_THRESHOLD = 0.05  # Minimum movement for swipe detection
DISPLAY_TIME = 1  # Seconds to display swipe direction

# Load gesture mapping from JSON and convert keys to int
with open(GESTURE_MAP_PATH, "r") as f:
    gesture_map = json.load(f)
    gesture_map = {int(k): v for k, v in gesture_map.items()}

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Motion tracking variables
landmark_history = collections.deque(maxlen=N_FRAMES)
tracking_motion = False  # Whether we're currently tracking motion
swipe_text = ""
swipe_time = 0
current_tracking_indices = None  # Landmark indices to track for the current gesture
current_static_gesture = ""


def predict_gesture(input_data):
    svm_model = joblib.load(MODEL_PATH)
    input_data = np.array(input_data).reshape(1, -1)
    prediction = svm_model.predict(input_data)
    return prediction[0]


def normalize_landmarks(landmarks):
    num_landmarks = 21
    landmarks = np.array(landmarks).reshape(num_landmarks, 3)
    wrist = landmarks[0]
    landmarks -= wrist
    max_dist = np.max(np.linalg.norm(landmarks, axis=1))
    landmarks /= max_dist if max_dist > 0 else 1
    return landmarks.flatten()


def get_tracking_point(raw_landmarks, indices):
    """
    Compute the average (x, y) coordinate for the given landmark indices.
    """
    points = [raw_landmarks[i][:2] for i in indices if i < len(raw_landmarks)]
    if points:
        return np.mean(points, axis=0)
    return raw_landmarks[0][:2]  # Fallback to wrist


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
    if -30 <= angle <= 30:
        return "Right Swipe"
    elif 150 <= angle <= 180 or -180 <= angle <= -150:
        return "Left Swipe"
    elif 60 < angle < 120:
        return "Up Swipe"
    elif -120 < angle < -60:
        return "Down Swipe"
    return None


def capture_landmarks():
    global tracking_motion, swipe_text, swipe_time, current_tracking_indices, current_static_gesture
    cap = cv2.VideoCapture(0)
    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
                raw_landmarks = [(lm.x, lm.y, lm.z) for lm in landmarks.landmark]
                norm_landmarks = normalize_landmarks(raw_landmarks)

                if tracking_motion:
                    # In motion tracking mode: track the average position of the relevant landmarks
                    if current_tracking_indices is not None:
                        tracking_point = get_tracking_point(
                            raw_landmarks, current_tracking_indices
                        )
                    else:
                        tracking_point = raw_landmarks[0][:2]
                    landmark_history.append(tracking_point)
                    swipe = calculate_swipe_direction()
                    if swipe:
                        swipe_text = swipe
                        swipe_time = time.time()
                        tracking_motion = False  # Reset motion tracking after detection
                        current_tracking_indices = None
                        landmark_history.clear()
                elif frame_count % N_FRAMES == 0:
                    # Classify static gesture every N frames
                    gesture = predict_gesture(norm_landmarks)
                    if gesture == 0:
                        current_static_gesture = "No Gesture"
                        tracking_motion = False
                    else:
                        # Use gesture mapping to get the tracking indices
                        mapping = gesture_map.get(int(gesture), None)
                        if mapping:
                            current_static_gesture = mapping.get("name", "Unknown")
                            current_tracking_indices = mapping.get("landmarks", [0])
                        else:
                            current_static_gesture = str(gesture)
                            current_tracking_indices = [0]
                        tracking_motion = True
                        landmark_history.clear()
                        tracking_point = get_tracking_point(
                            raw_landmarks, current_tracking_indices
                        )
                        landmark_history.append(tracking_point)

                # Always display static gesture on screen
                cv2.putText(
                    frame,
                    f"Gesture: {current_static_gesture}",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

        # Display swipe direction if detected for DISPLAY_TIME seconds
        if time.time() - swipe_time < DISPLAY_TIME and swipe_text:
            cv2.putText(
                frame,
                f"Swipe: {swipe_text}",
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2,
            )

        frame_count += 1
        cv2.imshow("Hand Gesture Recognition", frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_landmarks()
