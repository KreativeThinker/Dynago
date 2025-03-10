import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

mp_drawing = mp.solutions.drawing_utils

screen_width = 1000
screen_height = 1000

cap = cv2.VideoCapture(0)
prev_x = None
prev_y = None

while cap.isOpened():
    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            handedness = (
                results.multi_handedness[results.multi_hand_landmarks.index(landmarks)]
                .classification[0]
                .label
            )

            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
            # draws hand landmarks and connections on the frame
            # index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            # index_mid = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
            #
            # if handedness == "Left":  # left mouse
            #
            #     # if initial_dist is None:
            #     #     initial_dist = index_tip.y - index_mid.y
            #
            #     mcp_x = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x
            #     mcp_y = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y
            #
            #     cursor_x = int(mcp_x * screen_width)
            #     cursor_y = int(mcp_y * screen_height)
            #
            #     # current_dist = index_tip.y - index_mid.y
            #     if index_tip.y >= index_mid.y:
            #         print("click")
            #
            # elif handedness == "Right":  # right keyboard
            #     x, y = int(index_tip.x * screen_width), int(index_tip.y * screen_height)
            #     if prev_x is not None and prev_y is not None:
            #         dx = x - prev_x
            #         dy = y - prev_y
            #
            #         if abs(dx) > abs(dy):
            #             if dx > 50:  # right
            #                 print("right")
            #             elif dx < -50:  # left
            #                 print("left")
            #         else:  # Vertical swipe
            #             if dy > 50:  # down
            #                 print("down")
            #             elif dy < -50:  # up
            #                 print("up")
            #
            #     prev_x = x
            #     prev_y = y
            #
    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
