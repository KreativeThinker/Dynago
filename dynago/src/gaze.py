import cv2
import mediapipe as mp
import pyautogui
from pprint import pprint

# Init screen size
screen_w, screen_h = pyautogui.size()
w, h = pyautogui.size()

# MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, refine_landmarks=True
)

# Eye indices
right_socket_indices = [33, 133, 153, 157, 160, 163]
pupil_index = 468

# Webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # Get socket center
        socket_pts = [(landmarks[i].x, landmarks[i].y) for i in right_socket_indices]
        socket_center = tuple(sum(c) / len(c) for c in zip(*socket_pts))

        # Pupil center
        pupil_landmark = landmarks[pupil_index]
        pupil_center = (pupil_landmark.x, pupil_landmark.y)

        # Offset (pupil relative to socket)
        offset_x = socket_center[0] - pupil_center[0]
        offset_y = socket_center[1] - pupil_center[1]

        # Normalize offset
        eye_w = landmarks[33].x - landmarks[133].x
        eye_h = landmarks[159].y - landmarks[145].y
        norm_x = offset_x / eye_w
        norm_y = offset_y / eye_h

        # Scale to screen
        screen_x = screen_w // 2 + int(norm_x * screen_w)
        screen_y = screen_h // 2 + int(norm_y * screen_h)

        print(f"{offset_x},{offset_y}\t| ", end="")
        print(f"{eye_w},{eye_h}\t| ", end="")
        print(f"{norm_x},{norm_y}\t| ", end="")
        print(f"{screen_x},{screen_y}\n", end="")
        # print("\n")

        pyautogui.moveTo(screen_x, screen_y)

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
