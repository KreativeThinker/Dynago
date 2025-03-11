import cv2
import mediapipe as mp
import numpy as np
import joblib

MODEL_PATH = "dynago/models/gesture_svm.pkl"


mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils


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


def capture_landmarks():
    cap = cv2.VideoCapture(0)

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

                # Extract & normalize landmarks
                raw_landmarks = [(lm.x, lm.y, lm.z) for lm in landmarks.landmark]
                norm_landmarks = normalize_landmarks(raw_landmarks)

                # Predict gesture
                gesture = predict_gesture(norm_landmarks)

                # Display prediction
                cv2.putText(
                    frame,
                    f"Gesture: {gesture}",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

        cv2.imshow("Hand Gesture Recognition", frame)

        key = cv2.waitKey(1)
        if key & 0xFF == ord("q"):  # Quit on 'q'
            break

    cap.release()
    cv2.destroyAllWindows()
