import cv2
import mediapipe as mp
import numpy as np
import time


# Thresholds (tune these values)
GAZE_THRESHOLD_X = 0.04  # Horizontal sensitivity
GAZE_THRESHOLD_Y = 0.03  # Vertical sensitivity
BLINK_THRESHOLD = 0.28   # Lower = more sensitive to blinks


PRINT_INTERVAL = 0.5  # 0.5 seconds
BLINK_COOLDOWN = 15   # Frames to wait between clicks


LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473

RIGHT_SOCKET_INDICES = [33, 133, 153, 157, 160, 163]
LEFT_SOCKET_INDICES = [362, 263, 380, 384, 387, 390]

RIGHT_EYE_TOP = 159
RIGHT_EYE_BOTTOM = 145
RIGHT_EYE_LEFT_CORNER = 133
RIGHT_EYE_RIGHT_CORNER = 33
LEFT_EYE_TOP = 386
LEFT_EYE_BOTTOM = 374
LEFT_EYE_LEFT_CORNER = 362
LEFT_EYE_RIGHT_CORNER = 263

calibrated = False
center_gaze_ratio = (0, 0)
last_print_time = 0
blink_cooldown_counter = 0

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)

def get_gaze_ratio(landmarks, socket_indices, iris_index):
    """
    Calculates the normalized position of the iris relative to the eye socket.
    Returns (x_ratio, y_ratio)
    """
    try:
        socket_points = np.array(
            [(landmarks[i].x, landmarks[i].y) for i in socket_indices]
        )
        socket_center = np.mean(socket_points, axis=0)

        iris_center = np.array(
            [(landmarks[iris_index].x, landmarks[iris_index].y)]
        )

        eye_width = np.linalg.norm(socket_points[0] - socket_points[1])
        eye_height = np.max(socket_points[:, 1]) - np.min(socket_points[:, 1])

        offset_x = (iris_center[0][0] - socket_center[0]) / eye_width
        offset_y = (iris_center[0][1] - socket_center[1]) / eye_height
        
        return (offset_x, offset_y)
    except Exception as e:
        return (0, 0)

def get_blink_ratio(landmarks, top_idx, bottom_idx, left_idx, right_idx):
    """
    Calculates the ratio of vertical to horizontal eye distance (EAR).
    """
    try:
        top_lid = np.array([landmarks[top_idx].x, landmarks[top_idx].y])
        bottom_lid = np.array([landmarks[bottom_idx].x, landmarks[bottom_idx].y])
        left_corner = np.array([landmarks[left_idx].x, landmarks[left_idx].y])
        right_corner = np.array([landmarks[right_idx].x, landmarks[right_idx].y])

        vertical_dist = np.linalg.norm(top_lid - bottom_lid)
        horizontal_dist = np.linalg.norm(left_corner - right_corner)

        ratio = vertical_dist / horizontal_dist
        return ratio
    except Exception as e:
        return 0.5 


while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    if not calibrated:
        cv2.putText(frame, "Look at screen center and press 'c'", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Tracking... Press 'q' to quit", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        right_gaze_ratio = get_gaze_ratio(landmarks, RIGHT_SOCKET_INDICES, RIGHT_IRIS_CENTER)
        left_gaze_ratio = get_gaze_ratio(landmarks, LEFT_SOCKET_INDICES, LEFT_IRIS_CENTER)
        
        avg_gaze_ratio = (
            (right_gaze_ratio[0] + left_gaze_ratio[0]) / 2,
            (right_gaze_ratio[1] + left_gaze_ratio[1]) / 2
        )

        right_blink_ratio = get_blink_ratio(
            landmarks, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, 
            RIGHT_EYE_LEFT_CORNER, RIGHT_EYE_RIGHT_CORNER
        )
        left_blink_ratio = get_blink_ratio(
            landmarks, LEFT_EYE_TOP, LEFT_EYE_BOTTOM, 
            LEFT_EYE_LEFT_CORNER, LEFT_EYE_RIGHT_CORNER
        )
        avg_blink_ratio = (right_blink_ratio + left_blink_ratio) / 2

        if calibrated:
            current_time = time.time()
            
            if blink_cooldown_counter > 0:
                blink_cooldown_counter -= 1

            if avg_blink_ratio < BLINK_THRESHOLD and blink_cooldown_counter == 0:
                print("CLICK!")
                blink_cooldown_counter = BLINK_COOLDOWN 

            if current_time - last_print_time > PRINT_INTERVAL:
                dx = avg_gaze_ratio[0] - center_gaze_ratio[0]
                dy = avg_gaze_ratio[1] - center_gaze_ratio[1]

                dir_x = "Center"
                if dx > GAZE_THRESHOLD_X:
                    dir_x = "Right"
                elif dx < -GAZE_THRESHOLD_X:
                    dir_x = "Left"
                
                dir_y = "Center"
                if dy > GAZE_THRESHOLD_Y:
                    dir_y = "Down"
                elif dy < -GAZE_THRESHOLD_Y:
                    dir_y = "Up"
                
                print(f"Gaze: {dir_y}, {dir_x}")
                last_print_time = current_time

    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break
    if key == ord('c') and not calibrated and results.multi_face_landmarks:
        center_gaze_ratio = avg_gaze_ratio
        calibrated = True
        last_print_time = time.time()
        print(f"Calibrated! Center ratio: {center_gaze_ratio}")

    
    cv2.imshow('Gaze Calibration Test', frame)


cap.release()
cv2.destroyAllWindows()
face_mesh.close()