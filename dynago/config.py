import json

with open("dynago/data/gesture_map.json", "r") as file:
    GESTURE_MAP = json.load(file)
    GESTURE_MAP = {int(k): v for k, v in GESTURE_MAP.items()}

N_FRAMES = 6  # Static gesture classification every N frames
VEL_THRESHOLD = 0.25  # Minimum movement for swipe detection
DISPLAY_TIME = 1  # Seconds to display swipe direction
ENABLE_MOUSE = False
