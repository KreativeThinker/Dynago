import cv2
import mediapipe as mp
import numpy as np
import time


# Thresholds (tune these values)
BLINK_THRESHOLD = 0.28   # Lower = more sensitive to blinks
DIRECTION_THRESHOLD = 0.3  # Gaze must be >70% or <30% to trigger a direction


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

CALIBRATION_STEPS = [
    "CENTER", "TOP-LEFT", "TOP-RIGHT", "BOTTOM-LEFT", "BOTTOM-RIGHT"
]

calibration_state = 0  # 0-4 for steps, 5 for calibrated
gaze_ratios_calib = {step: (0, 0) for step in CALIBRATION_STEPS}
gaze_range_x = (0, 0)  # min, max (Left, Right)
gaze_range_y = (0, 0)  # min, max (Up, Down)
last_print_time = 0
blink_cooldown_counter = 0

# --- MediaPipe Setup ---
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

        eye_width = np.max(socket_points[:, 0]) - np.min(socket_points[:, 0])
        eye_height = np.max(socket_points[:, 1]) - np.min(socket_points[:, 1])
        
        if eye_width == 0 or eye_height == 0:
            return (0, 0)

        offset_x = (iris_center[0][0] - socket_center[0]) / (eye_width / 2) 
        offset_y = (iris_center[0][1] - socket_center[1]) / (eye_height / 2) 
        
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
        
        if horizontal_dist == 0:
            return 0.5 

        ratio = vertical_dist / horizontal_dist
        return ratio
    except Exception as e:
       
        return 0.5 


# --- Main Loop ---
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    
    if calibration_state < len(CALIBRATION_STEPS):
        step_name = CALIBRATION_STEPS[calibration_state]
        cv2.putText(frame, f"Look at {step_name} and press 'c'", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Calibrated! Tracking... Press 'q' to quit", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # --- Gaze Calculation ---
        right_gaze_ratio = get_gaze_ratio(landmarks, RIGHT_SOCKET_INDICES, RIGHT_IRIS_CENTER)
        left_gaze_ratio = get_gaze_ratio(landmarks, LEFT_SOCKET_INDICES, LEFT_IRIS_CENTER)
        
        avg_gaze_ratio = (
            (right_gaze_ratio[0] + left_gaze_ratio[0]) / 2,
            (right_gaze_ratio[1] + left_gaze_ratio[1]) / 2
        )

        # --- Blink Calculation ---
        right_blink_ratio = get_blink_ratio(
            landmarks, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, 
            RIGHT_EYE_LEFT_CORNER, RIGHT_EYE_RIGHT_CORNER
        )
        left_blink_ratio = get_blink_ratio(
            landmarks, LEFT_EYE_TOP, LEFT_EYE_BOTTOM, 
            LEFT_EYE_LEFT_CORNER, LEFT_EYE_RIGHT_CORNER
        )
        avg_blink_ratio = (right_blink_ratio + left_blink_ratio) / 2

        # --- State Logic ---
        if calibration_state == len(CALIBRATION_STEPS): # If calibrated
            current_time = time.time()
            
            # 1. Blink Detection
            if blink_cooldown_counter > 0:
                blink_cooldown_counter -= 1

            if avg_blink_ratio < BLINK_THRESHOLD and blink_cooldown_counter == 0:
                print("CLICK!")
                blink_cooldown_counter = BLINK_COOLDOWN 

            # 2. Gaze Direction
            if current_time - last_print_time > PRINT_INTERVAL:
                curr_x, curr_y = avg_gaze_ratio
                
                norm_x = np.interp(curr_x, gaze_range_x, [0, 1])
                norm_y = np.interp(curr_y, gaze_range_y, [0, 1])
                
                norm_x = np.clip(norm_x, 0, 1)
                norm_y = np.clip(norm_y, 0, 1)

                dir_x = "Center"
                if norm_x < DIRECTION_THRESHOLD:
                    dir_x = "Left"
                elif norm_x > (1 - DIRECTION_THRESHOLD):
                    dir_x = "Right"
                
                dir_y = "Center"
                if norm_y < DIRECTION_THRESHOLD:
                    dir_y = "Up"
                elif norm_y > (1 - DIRECTION_THRESHOLD):
                    dir_y = "Down"
                
                print(f"Gaze: {dir_y}, {dir_x}")
                last_print_time = current_time

    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break
    
    if key == ord('c') and calibration_state < len(CALIBRATION_STEPS) and results.multi_face_landmarks:
        step_name = CALIBRATION_STEPS[calibration_state]
        gaze_ratios_calib[step_name] = avg_gaze_ratio
        print(f"Calibrated {step_name}: {avg_gaze_ratio}")
        
        calibration_state += 1
        
        if calibration_state == len(CALIBRATION_STEPS):
            
            gaze_left = (gaze_ratios_calib["TOP-LEFT"][0] + gaze_ratios_calib["BOTTOM-LEFT"][0]) / 2
            gaze_right = (gaze_ratios_calib["TOP-RIGHT"][0] + gaze_ratios_calib["BOTTOM-RIGHT"][0]) / 2
            
            gaze_top = (gaze_ratios_calib["TOP-LEFT"][1] + gaze_ratios_calib["TOP-RIGHT"][1]) / 2
            gaze_bottom = (gaze_ratios_calib["BOTTOM-LEFT"][1] + gaze_ratios_calib["BOTTOM-RIGHT"][1]) / 2

            gaze_range_x = (gaze_left, gaze_right)
            gaze_range_y = (gaze_top, gaze_bottom) 

            print("--- CALIBRATION COMPLETE ---")
            print(f"X-Range (Left, Right): ({gaze_left:.4f}, {gaze_right:.4f})")
            print(f"Y-Range (Up, Down): ({gaze_top:.4f}, {gaze_bottom:.4f})")
            print(f"Center: {gaze_ratios_calib['CENTER']}")
            last_print_time = time.time()

    
    cv2.imshow('Gaze Calibration Test', frame)

cap.release()
cv2.destroyAllWindows()
face_mesh.close()