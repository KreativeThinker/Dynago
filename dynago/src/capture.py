import multiprocessing
from collections import deque

import cv2
import joblib
import mediapipe as mp
import numpy as np

from dynago.config import ENABLE_MOUSE, GESTURE_MAP, N_FRAMES
from dynago.src.command import execute_command
from dynago.src.mouse import GestureMouse
from dynago.src.swipe import (
    calculate_swipe_direction,
    cleanup,
    get_tracking_point,
    landmark_history,
    mean_landmark_history,
)

MODEL_PATH = "dynago/models/gesture_svm.pkl"

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

BUFFER_SIZE = 10
PREDICTION_BUFFER_SIZE = 5


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


def process_frame(frame, hands, model, state, mouse_controller):
    """Process a single frame and return results."""
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR_RGB)
    results = hands.process(rgb_frame)

    output = {
        "frame": frame,
        "results": results,
        "command": None,
        "gesture_name": "...", # Default text
        "in_mouse_mode": False,
    }
    
    # Store the currently confirmed gesture
    last_confirmed_gesture = state["current_gesture_id"]
    
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            raw_landmarks = [(lm.x, lm.y, lm.z) for lm in landmarks.landmark]
            norm_landmarks = normalize_landmarks(raw_landmarks)
            mp_drawing.draw_landmarks(
                output["frame"], landmarks, mp_hands.HAND_CONNECTIONS
            )

            raw_gesture = predict_gesture(norm_landmarks, model)

            state["prediction_buffer"].append(raw_gesture)

            confirmed_gesture = None
            if len(state["prediction_buffer"]) == state["prediction_buffer"].maxlen:
                first_pred = state["prediction_buffer"][0]
                if all(p == first_pred for p in state["prediction_buffer"]):
                    confirmed_gesture = first_pred
            
            if confirmed_gesture is not None and confirmed_gesture != state["current_gesture_id"]:
                # A new gesture has been confirmed!
                state["current_gesture_id"] = confirmed_gesture
                # Clear swipe history on ANY state change
                mean_landmark_history.clear()
                landmark_history.clear()

            current_state_id = state["current_gesture_id"]
            
            if current_state_id is None:
                output["gesture_name"] = "..." # Waiting for confirmation
            
            elif current_state_id == 0:
                output["gesture_name"] = "No Gesture"
                # This is the idle/stop state, do nothing.
            
            elif current_state_id == 4 and ENABLE_MOUSE:
                output["gesture_name"] = GESTURE_MAP.get(5, {}).get("name", "point")
                output["in_mouse_mode"] = True
                finger_tip_pos = raw_landmarks[8][:2]  # index finger tip (x,y)
                mouse_controller.update(finger_tip_pos)
            
            elif current_state_id is not None:
                # This is a swipe-tracking state (e.g., fist, palm)
                mapping = GESTURE_MAP.get(int(current_state_id), {})
                output["gesture_name"] = mapping.get("name", "Unknown")
                tracking_indices = mapping.get("landmarks", [0])

                # Continue tracking swipe motion
                tracking_point = get_tracking_point(
                    raw_landmarks, tracking_indices
                )
                mean_landmark_history.append(tracking_point)
                swipe = calculate_swipe_direction(current_state_id)

                if swipe is not None:
                    # Swipe detected! Send command
                    output["command"] = (current_state_id, swipe)
                    # Clear history to wait for the next swipe
                    mean_landmark_history.clear()
                    landmark_history.clear()

    else:
        # No hand detected.
        # Force a "No Gesture" prediction into the buffer
        state["prediction_buffer"].append(0) 
        
        # Check if buffer is full of 0s
        if len(state["prediction_buffer"]) == state["prediction_buffer"].maxlen:
            if all(p == 0 for p in state["prediction_buffer"]):
                if state["current_gesture_id"] != 0:
                    # Hand lost, confirm "No Gesture"
                    state["current_gesture_id"] = 0
                    mean_landmark_history.clear()
                    landmark_history.clear()
        
        if state["current_gesture_id"] == 0:
            output["gesture_name"] = "No Gesture"


    # If we are in a transition (buffer not unanimous)
    # keep displaying the *last* confirmed gesture name for visual stability
    if confirmed_gesture is None and last_confirmed_gesture is not None:
        if last_confirmed_gesture == 0:
             output["gesture_name"] = "No Gesture"
        elif last_confirmed_gesture is not None:
            # Get name from GESTURE_MAP, default to "..."
            mapping = GESTURE_MAP.get(int(last_confirmed_gesture), {})
            output["gesture_name"] = mapping.get("name", "...")

    return output


def capture_landmarks(cmd_queue):
    """Main capture process with improved resource management."""
    cap = cv2.VideoCapture(0)
    mouse_controller = GestureMouse()
    in_mouse_mode = False  # Track if we're in mouse control mode

    # Initialize state dictionary
    state = {
        "frame_count": 0,
        "tracking_motion": False,
        "tracking_indices": None,
        "current_gesture_id": None,
    }

    # Load model once at start
    model = joblib.load(MODEL_PATH)

    # Initialize hands detector
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    ) as hands:

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            # Process frame - now passing mouse_controller and in_mouse_mode
            result = process_frame(frame, hands, model, state, mouse_controller)
            in_mouse_mode = result["in_mouse_mode"]

            # Skip command processing if in mouse mode
            if not in_mouse_mode and result["command"] is not None:
                cmd_queue.put(result["command"])

            # Update frame count
            state["frame_count"] += 1

            # Display frame
            cv2.imshow("Gesture Recognition", result["frame"])
            # cv2.moveWindow("Gesture Recognition", 100, 100)  # Set position
            # cv2.resizeWindow("Gesture Recognition", 32, 18)  # Set dimensions
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


def command_worker(cmd_queue):
    """Command processing worker."""
    while True:
        command = cmd_queue.get()
        if command is None:  # Sentinel value to stop
            break
        gesture_id, swipe = command
        execute_command(gesture_id, swipe)


if __name__ == "__main__":
    # Use spawn context for better compatibility
    mp = multiprocessing.get_context("spawn")

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

        # Run capture in main process
        capture_landmarks(cmd_queue)

    finally:
        # Cleanup
        cleanup()
        cmd_queue.put(None)
        worker.join(timeout=1)
        if worker.is_alive():
            worker.terminate()
        cmd_queue.close()
