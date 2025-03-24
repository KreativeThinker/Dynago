import cv2
import mediapipe as mp
import numpy as np
import joblib
import json
import time
import pyautogui
from pynput.mouse import Controller as MouseController
from dynago.config import N_FRAMES
from dynago.src.swipe import (
    get_tracking_point,
    calculate_swipe_direction,
    landmark_history,
)
from dynago.src.command import execute_command

MODEL_PATH = "dynago/models/gesture_svm.pkl"
GESTURE_MAP_PATH = "dynago/data/gesture_map.json"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

with open(GESTURE_MAP_PATH, "r") as f:
    gesture_map = json.load(f)
    gesture_map = {int(k): v for k, v in gesture_map.items()}

BUFFER_SIZE = 5
gesture_buffer = []

mouse = MouseController()
screen_width, screen_height = pyautogui.size()
smoothed_position = None
alpha = 0.2  # Smaller alpha gives more smoothing


def update_smoothed_position(new_position):
    global smoothed_position
    if smoothed_position is None:
        smoothed_position = new_position
    else:
        smoothed_position = alpha * np.array(new_position) + (1 - alpha) * np.array(
            smoothed_position
        )
    return smoothed_position


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
    current_gesture_id = None
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
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

                # If gesture 6 ("point") is detected, control cursor live.
                if frame_count % N_FRAMES == 0:
                    gesture = predict_gesture(norm_landmarks)
                    if gesture == 6:
                        current_gesture_id = 6
                        current_static_gesture = gesture_map.get(6, {}).get(
                            "name", "point"
                        )
                        # Use index finger tip (landmark 8) as pointer reference
                        point = raw_landmarks[8][:2]
                        smoothed = update_smoothed_position(point)
                        mouse_x = int(smoothed[0] * screen_width)
                        mouse_y = int(smoothed[1] * screen_height)
                        mouse.position = (mouse_x, mouse_y)
                        print(
                            f'Detected: "{current_static_gesture}" - Cursor moved to ({mouse_x}, {mouse_y})',
                            flush=True,
                        )
                        # Do not enter motion tracking mode; update cursor continuously.
                        tracking_motion = False
                        continue

                # For non-point gestures, use motion tracking and swipe detection.
                if tracking_motion:
                    if current_tracking_indices is not None:
                        tracking_point = get_tracking_point(
                            raw_landmarks, current_tracking_indices
                        )
                    else:
                        tracking_point = raw_landmarks[0][:2]
                    landmark_history.append(tracking_point)
                    swipe = calculate_swipe_direction()
                    if swipe is not None:
                        print(f'{current_static_gesture} swipe "{swipe}"', flush=True)
                        execute_command(current_gesture_id, swipe)
                        cooldown_until = time.time() + 2  # 2-second cooldown
                        tracking_motion = False
                        current_tracking_indices = None
                        landmark_history.clear()
                elif frame_count % N_FRAMES == 0:
                    gesture = predict_gesture(norm_landmarks)
                    if gesture == 0:
                        print(".", end="", flush=True)
                        tracking_motion = False
                        current_gesture_id = None
                        gesture_buffer.clear()
                    else:
                        # If gesture changes, cancel previous tracking.
                        if (
                            current_gesture_id is not None
                            and gesture != current_gesture_id
                        ):
                            tracking_motion = False
                            landmark_history.clear()
                            print(
                                "Gesture changed. Cancelling previous tracking.",
                                flush=True,
                            )
                        current_gesture_id = gesture
                        mapping = gesture_map.get(int(gesture), None)
                        if mapping:
                            current_static_gesture = mapping.get("name", "Unknown")
                            current_tracking_indices = mapping.get("landmarks", [0])
                        else:
                            current_static_gesture = str(gesture)
                            current_tracking_indices = [0]
                        print(f'Detected: "{current_static_gesture}"', flush=True)
                        tracking_motion = True
                        landmark_history.clear()
                        tracking_point = get_tracking_point(
                            raw_landmarks, current_tracking_indices
                        )
                        landmark_history.append(tracking_point)
        frame_count += 1
        # cv2.imshow("Hand Gesture Recognition", frame)
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_landmarks()
# import cv2
# import mediapipe as mp
# import numpy as np
# import joblib
# import json
# import time
# from dynago.config import N_FRAMES
# from dynago.src.swipe import (
#     get_tracking_point,
#     calculate_swipe_direction,
#     landmark_history,
# )
# from dynago.src.command import execute_command
#
# MODEL_PATH = "dynago/models/gesture_svm.pkl"
# GESTURE_MAP_PATH = "dynago/data/gesture_map.json"
#
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands()
#
# with open(GESTURE_MAP_PATH, "r") as f:
#     gesture_map = json.load(f)
#     gesture_map = {int(k): v for k, v in gesture_map.items()}
#
#
# def normalize_landmarks(landmarks):
#     num_landmarks = 21
#     landmarks = np.array(landmarks).reshape(num_landmarks, 3)
#     wrist = landmarks[0]
#     landmarks -= wrist
#     max_dist = np.max(np.linalg.norm(landmarks, axis=1))
#     landmarks /= max_dist if max_dist > 0 else 1
#     return landmarks.flatten()
#
#
# def predict_gesture(input_data):
#     svm_model = joblib.load(MODEL_PATH)
#     input_data = np.array(input_data).reshape(1, -1)
#     prediction = svm_model.predict(input_data)
#     return prediction[0]
#
#
# def capture_landmarks():
#     cap = cv2.VideoCapture(0)
#     frame_count = 0
#     tracking_motion = False
#     current_tracking_indices = None
#     current_static_gesture = ""
#     current_gesture_id = None
#     cooldown_until = time.time()
#
#     while cap.isOpened():
#         success, frame = cap.read()
#         if not success:
#             continue
#
#         if time.time() < cooldown_until:
#             continue
#
#         frame = cv2.flip(frame, 1)
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = hands.process(rgb_frame)
#
#         if results.multi_hand_landmarks:
#             for landmarks in results.multi_hand_landmarks:
#                 raw_landmarks = [(lm.x, lm.y, lm.z) for lm in landmarks.landmark]
#                 norm_landmarks = normalize_landmarks(raw_landmarks)
#
#                 if tracking_motion:
#                     # Cancel tracking if static gesture changes
#                     new_gesture = predict_gesture(norm_landmarks)
#                     if new_gesture != current_gesture_id:
#                         tracking_motion = False
#                         current_tracking_indices = None
#                         landmark_history.clear()
#                         print("Gesture changed. Cancelling swipe tracking.")
#                         continue
#
#                     if current_tracking_indices is not None:
#                         tracking_point = get_tracking_point(
#                             raw_landmarks, current_tracking_indices
#                         )
#                     else:
#                         tracking_point = raw_landmarks[0][:2]
#                     landmark_history.append(tracking_point)
#                     swipe = calculate_swipe_direction()
#                     if swipe is not None:
#                         print(f'{current_static_gesture} swipe "{swipe}"', flush=True)
#                         execute_command(current_gesture_id, swipe)
#                         cooldown_until = time.time() + 1
#                         tracking_motion = False
#                         current_tracking_indices = None
#                         landmark_history.clear()
#                 elif frame_count % N_FRAMES == 0:
#                     gesture = predict_gesture(norm_landmarks)
#                     if gesture == 0:
#                         print(".", end="", flush=True)
#                         tracking_motion = False
#                         current_gesture_id = None
#                     else:
#                         # If a new gesture is detected while tracking, abort previous tracking.
#                         if (
#                             current_gesture_id is not None
#                             and gesture != current_gesture_id
#                         ):
#                             tracking_motion = False
#                             landmark_history.clear()
#                             print("Gesture changed. Cancelling previous tracking.")
#                         current_gesture_id = gesture
#                         mapping = gesture_map.get(int(gesture), None)
#                         if mapping:
#                             current_static_gesture = mapping.get("name", "Unknown")
#                             current_tracking_indices = mapping.get("landmarks", [0])
#                         else:
#                             current_static_gesture = str(gesture)
#                             current_tracking_indices = [0]
#                         print(f'Detected: "{current_static_gesture}"')
#                         tracking_motion = True
#                         landmark_history.clear()
#                         tracking_point = get_tracking_point(
#                             raw_landmarks, current_tracking_indices
#                         )
#                         landmark_history.append(tracking_point)
#
#         frame_count += 1
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()
#
#
# if __name__ == "__main__":
#     capture_landmarks()
