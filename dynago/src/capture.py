import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils


def normalize_landmarks(landmarks):
    num_landmarks = 21
    landmarks = np.array(landmarks).reshape(num_landmarks, 3)

    wrist = landmarks[0]
    landmarks -= wrist

    max_dist = np.max(np.linalg.norm(landmarks, axis=1))
    landmarks /= max_dist if max_dist > 0 else 1

    return landmarks.flatten()


def capture_landmarks():
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

        cv2.imshow("Hand Capture", frame)

        key = cv2.waitKey(1)
        if key == 32:
            if results.multi_hand_landmarks:
                for landmarks in results.multi_hand_landmarks:
                    cap.release()
                    cv2.destroyAllWindows()
                    raw_landmarks = [(lm.x, lm.y, lm.z) for lm in landmarks.landmark]
                    return normalize_landmarks(raw_landmarks)

            print("❌ No hand detected!")

        # Quit
        if key & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return None


def from_video():
    landmarks = capture_landmarks()
    if landmarks is not None:
        return normalize_landmarks(landmarks)
    return None
