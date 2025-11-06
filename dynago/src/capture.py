import multiprocessing

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
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    output = {
        "frame": frame,
        "results": results,
        "command": None,
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

            if gesture == 4 and ENABLE_MOUSE:
                output["gesture_name"] = GESTURE_MAP.get(5, {}).get("name", "point")
                output["in_mouse_mode"] = True
                finger_tip_pos = raw_landmarks[8][:2]  # index finger tip (x,y)
                mouse_controller.update(finger_tip_pos)
                return output  # Handle gestures
            elif not state["tracking_motion"]:
                # When not tracking motion, check for new gestures every N frames
                if state["frame_count"] % N_FRAMES == 0:
                    if gesture == 0:  # No gesture detected
                        state["current_gesture_id"] = None
                    else:
                        # New gesture detected
                        state["current_gesture_id"] = gesture
                        mapping = GESTURE_MAP.get(int(gesture), {})
                        output["gesture_name"] = mapping.get("name", "Unknown")
                        state["tracking_indices"] = mapping.get("landmarks", [0])
                        state["tracking_motion"] = True
                        mean_landmark_history.clear()
                        landmark_history.clear()
                        tracking_point = get_tracking_point(
                            raw_landmarks, state["tracking_indices"]
                        )
                        mean_landmark_history.append(tracking_point)
            else:
                # When tracking motion, process every frame (bypass N_FRAMES filter)
                if gesture != state["current_gesture_id"]:
                    # Gesture changed while tracking
                    state["tracking_motion"] = False
                    state["current_gesture_id"] = None
                    state["tracking_indices"] = None
                    mean_landmark_history.clear()
                    landmark_history.clear()
                else:
                    # Continue tracking same gesture
                    tracking_point = get_tracking_point(
                        raw_landmarks, state["tracking_indices"]
                    )
                    mean_landmark_history.append(tracking_point)
                    swipe = calculate_swipe_direction(gesture)
                    if swipe is not None:
                        # Swipe detected
                        output["command"] = (state["current_gesture_id"], swipe)
                        state["tracking_motion"] = False
                        state["current_gesture_id"] = None
                        state["tracking_indices"] = None
                        mean_landmark_history.clear()
                        landmark_history.clear()

    output["in_mouse_mode"] = False
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
