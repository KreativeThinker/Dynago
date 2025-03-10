import cv2
import mediapipe as mp
import numpy as np
import csv
import os

# Constants
# Initialize Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils


def save_landmark_data(landmarks, gesture_name, csv_path="dynago/data/raw.csv"):
    """Save flattened landmarks + gesture name to CSV."""
    data = np.array(landmarks).flatten().tolist()
    data.append(gesture_name)

    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode="a") as f:
        writer = csv.writer(f)

        if not file_exists:
            header = [f"x{i+1}, y{i+1}, z{i+1}" for i in range(21)] + ["gesture"]
            writer.writerow(header)

        writer.writerow(data)

    print(f"✅ Saved '{gesture_name}' to {csv_path}")


def capture():
    gesture_name = input("Enter gesture name: ").strip()

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
        if key == 32:
            if results.multi_hand_landmarks:
                for landmarks in results.multi_hand_landmarks:
                    save_landmark_data(
                        [(lm.x, lm.y, lm.z) for lm in landmarks.landmark], gesture_name
                    )
            else:
                print("❌ No hand detected!")

        if key & 0xFF == ord("q"):  # Quit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    capture()
