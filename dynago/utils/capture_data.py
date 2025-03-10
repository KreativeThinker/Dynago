import cv2
import mediapipe


def capture():
    mediapipe_hands = mediapipe.solutions.hands
    hands = mediapipe_hands.Hands()
    mediapipe_drawing = mediapipe.solutions.drawing_utils

    capture = cv2.VideoCapture(0)

    while capture.isOpened():
        _, frame = capture.read()
        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        results = hands.process(frame)
        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                # handedness = (
                #     results.multi_handedness[
                #         results.multi_hand_landmarks.index(landmarks)
                #     ]
                #     .classification[0]
                #     .label
                # )

                mediapipe_drawing.draw_landmarks(
                    frame, landmarks, mediapipe_hands.HAND_CONNECTIONS
                )

        cv2.imshow("Gesture Recognition", frame)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    capture()
