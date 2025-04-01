import cv2
import mediapipe as mp
import numpy as np
import joblib
import pyautogui
import multiprocessing as mp_proc
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
mp_drawing = mp.solutions.drawing_utils

BUFFER_SIZE = 10
screen_width, screen_height = pyautogui.size()


def init_worker():
    """Initialize worker-specific resources"""
    global mouse
    mouse = MouseController()


def normalize_landmarks(landmarks):
    num_landmarks = 21
    landmarks = np.array(landmarks).reshape(num_landmarks, 3)
    wrist = landmarks[0]
    landmarks -= wrist
    max_dist = np.max(np.linalg.norm(landmarks, axis=1))
    landmarks /= max_dist if max_dist > 0 else 1
    return landmarks.flatten()


def predict_gesture(input_data, model):
    input_data = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_data)
    return prediction[0]


def process_frame(frame, hands, model, state):
    """Process a single frame and return results"""
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    output = {
        "frame": frame,
        "results": results,
        "command": None,
        "mouse_pos": None,
        "gesture_name": None,
    }

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            raw_landmarks = [(lm.x, lm.y, lm.z) for lm in landmarks.landmark]
            norm_landmarks = normalize_landmarks(raw_landmarks)
            mp_drawing.draw_landmarks(
                output["frame"], landmarks, mp_hands.HAND_CONNECTIONS
            )

            # Predict gesture
            gesture = predict_gesture(norm_landmarks, model)

            # Handle point gesture (direct mouse control)
            if gesture == 6:
                output["gesture_name"] = GESTURE_MAP.get(6, {}).get("name", "point")
                point = raw_landmarks[8][:2]  # index finger tip
                smoothed = update_smoothed_position(point, state["smoothed_position"])
                mouse_x = int(smoothed[0] * screen_width)
                mouse_y = int(smoothed[1] * screen_height)
                output["mouse_pos"] = (mouse_x, mouse_y)
                state["tracking_motion"] = False
                return output

            # Handle other gestures
            if state["frame_count"] % N_FRAMES == 0:
                if not state["tracking_motion"]:
                    if gesture == 0:
                        state["current_gesture_id"] = None
                    else:
                        state["current_gesture_id"] = gesture
                        mapping = GESTURE_MAP.get(int(gesture), {})
                        output["gesture_name"] = mapping.get("name", "Unknown")
                        state["tracking_indices"] = mapping.get("landmarks", [0])
                        state["tracking_motion"] = True
                        landmark_history.clear()
                        tracking_point = get_tracking_point(
                            raw_landmarks, state["tracking_indices"]
                        )
                        landmark_history.append(tracking_point)
                else:
                    if gesture != state["current_gesture_id"]:
                        state["tracking_motion"] = False
                        state["current_gesture_id"] = None
                        state["tracking_indices"] = None
                        landmark_history.clear()
                    else:
                        tracking_point = get_tracking_point(
                            raw_landmarks, state["tracking_indices"]
                        )
                        landmark_history.append(tracking_point)
                        swipe = calculate_swipe_direction()
                        if swipe is not None:
                            output["command"] = (state["current_gesture_id"], swipe)
                            state["tracking_motion"] = False
                            state["current_gesture_id"] = None
                            state["tracking_indices"] = None
                            landmark_history.clear()

    return output


def update_smoothed_position(new_position, prev_position, speed_factor=1.0):
    adaptive_alpha = min(0.1 + 0.3 * speed_factor, 0.5)
    if prev_position is None:
        return new_position
    return adaptive_alpha * np.array(new_position) + (1 - adaptive_alpha) * np.array(
        prev_position
    )


def capture_landmarks(cmd_queue):
    """Main capture process with improved resource management"""
    cap = cv2.VideoCapture(0)

    # Initialize state dictionary
    state = {
        "frame_count": 0,
        "tracking_motion": False,
        "tracking_indices": None,
        "current_gesture_id": None,
        "smoothed_position": None,
    }

    # Load model once at start
    model = joblib.load(MODEL_PATH)

    # Initialize hands detector
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    ) as hands:

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            # Process frame
            result = process_frame(frame, hands, model, state)

            # Handle results
            if result["mouse_pos"] is not None:
                mouse.position = result["mouse_pos"]

            if result["command"] is not None:
                cmd_queue.put(result["command"])

            # Update smoothed position if we have mouse movement
            if result["mouse_pos"] is not None:
                state["smoothed_position"] = update_smoothed_position(
                    result["mouse_pos"], state["smoothed_position"]
                )

            # Update frame count
            state["frame_count"] += 1

            # Display frame
            cv2.imshow("Gesture Recognition", result["frame"])
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


def command_worker(cmd_queue):
    """Command processing worker with initialization"""
    init_worker()
    while True:
        command = cmd_queue.get()
        if command is None:  # Sentinel value to stop
            break
        gesture_id, swipe = command
        execute_command(gesture_id, swipe)


if __name__ == "__main__":
    # Use spawn context for better compatibility
    mp = mp_proc.get_context("spawn")

    # Create command queue
    cmd_queue = mp.Queue(maxsize=10)  # Prevent queue from growing too large

    try:
        # Start command worker
        worker = mp.Process(
            target=command_worker,
            args=(cmd_queue,),
            daemon=True,  # Worker will terminate when main process ends
        )
        worker.start()

        # Run capture in main process (avoids issues with OpenCV in child processes)
        capture_landmarks(cmd_queue)

    finally:
        # Cleanup
        cmd_queue.put(None)  # Signal worker to stop
        worker.join(timeout=1)  # Wait for worker to finish
        if worker.is_alive():
            worker.terminate()  # Force terminate if needed
        cmd_queue.close()  # Clean up queue
