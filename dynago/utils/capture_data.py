import cv2
import mediapipe as mp
import numpy as np
import csv
import os
from dynago.config import GESTURE_MAP

# Constants
CSV_PATH = "dynago/data/raw.csv"

# Initialize Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils


def save_landmark_data(landmarks, gesture_index):
    """Save flattened landmarks + gesture index to CSV."""
    data = np.array(landmarks).flatten().tolist()
    data.append(gesture_index)

    file_exists = os.path.isfile(CSV_PATH)

    with open(CSV_PATH, mode="a") as f:
        writer = csv.writer(f)

        if not file_exists:
            header = [f"x{i+1}, y{i+1}, z{i+1}" for i in range(21)] + ["gesture_index"]
            writer.writerow(header)

        writer.writerow(data)

    print(f"✅ Saved gesture index '{gesture_index}' to {CSV_PATH}")


def capture():
    """Capture hand landmarks and store them with a gesture index."""
    gesture_map = GESTURE_MAP
    if not gesture_map:
        return

    # Print available gesture mappings
    print("\n📌 Available Gestures:")
    for idx, name in gesture_map.items():
        print(f"  {idx}: {name}")

    # Take gesture index input
    gesture_index = input("\nEnter the gesture index: ").strip()

    if gesture_index not in gesture_map:
        print("❌ Invalid gesture index!")
        return

    cap = cv2.VideoCapture(0)
    print("📷 Press SPACE to capture landmarks, 'q' to quit.")

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

        cv2.imshow("Gesture Recognition", frame)

        key = cv2.waitKey(1)
        if key == 32:  # SPACE key to capture
            if results.multi_hand_landmarks:
                for landmarks in results.multi_hand_landmarks:
                    save_landmark_data(
                        [(lm.x, lm.y, lm.z) for lm in landmarks.landmark], gesture_index
                    )
            else:
                print("❌ No hand detected!")

        if key & 0xFF == ord("q"):  # Quit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    capture()
