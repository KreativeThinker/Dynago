import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
import pyautogui
from pynput.mouse import Controller as MouseController
from dynago.config import N_FRAMES, GESTURE_MAP
from dynago.src.swipe import (
    get_tracking_point,
    calculate_swipe_direction,
    landmark_history,
)
from dynago.src.command import execute_command

MODEL_PATH = "dynago/models/gesture_svm.pkl"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils


BUFFER_SIZE = 10
gesture_buffer = []

mouse = MouseController()
screen_width, screen_height = pyautogui.size()
smoothed_position = None
alpha = 0.2  # Smoothing factor


def update_smoothed_position(new_position, speed_factor=1.0):
    global smoothed_position
    adaptive_alpha = min(0.1 + 0.3 * speed_factor, 0.5)  # Adjust smoothing dynamically
    if smoothed_position is None:
        smoothed_position = new_position
    else:
        smoothed_position = adaptive_alpha * np.array(new_position) + (
            1 - adaptive_alpha
        ) * np.array(smoothed_position)
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


def predict_gesture_with_voting(input_data):
    global gesture_buffer
    prediction = predict_gesture(input_data)
    gesture_buffer.append(prediction)
    if len(gesture_buffer) > BUFFER_SIZE:
        gesture_buffer.pop(0)
    return max(set(gesture_buffer), key=gesture_buffer.count)


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

                # Live cursor control for point gesture (gesture 6)
                if frame_count % N_FRAMES == 0:
                    gesture = predict_gesture(norm_landmarks)
                    if gesture == 6:
                        current_gesture_id = 6
                        current_static_gesture = GESTURE_MAP.get(6, {}).get(
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
                        tracking_motion = False
                        continue

                # For non-point gestures, use stable gesture detection and motion tracking
                if frame_count % N_FRAMES == 0:
                    new_gesture = predict_gesture(norm_landmarks)
                    # If not already tracking, start tracking if a valid gesture is detected.
                    if not tracking_motion:
                        if new_gesture == 0:
                            print(".", end="", flush=True)
                            current_gesture_id = None
                        else:
                            current_gesture_id = new_gesture
                            mapping = GESTURE_MAP.get(int(new_gesture), {})
                            current_static_gesture = mapping.get("name", "Unknown")
                            current_tracking_indices = mapping.get("landmarks", [0])
                            print(f'Detected: "{current_static_gesture}"', flush=True)
                            tracking_motion = True
                            landmark_history.clear()
                            tracking_point = get_tracking_point(
                                raw_landmarks, current_tracking_indices
                            )
                            landmark_history.append(tracking_point)
                    else:
                        # Already tracking: abort if gesture changed
                        if new_gesture != current_gesture_id:
                            tracking_motion = False
                            current_gesture_id = None
                            current_tracking_indices = None
                            landmark_history.clear()
                            print("Gesture changed. Cancelling tracking.", flush=True)
                        else:
                            tracking_point = get_tracking_point(
                                raw_landmarks, current_tracking_indices
                            )
                            landmark_history.append(tracking_point)
                            swipe = calculate_swipe_direction()
                            if swipe is not None:
                                print(
                                    f'{current_static_gesture} swipe "{swipe}"',
                                    flush=True,
                                )
                                execute_command(current_gesture_id, swipe)
                                cooldown_until = time.time() + 2
                                tracking_motion = False
                                current_gesture_id = None
                                current_tracking_indices = None
                                landmark_history.clear()

        frame_count += 1
        cv2.imshow("Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_landmarks()
