import os
import json
import threading

# Load gesture map globally
GESTURE_MAP_PATH = "dynago/data/gesture_map.json"
with open(GESTURE_MAP_PATH, "r") as f:
    GESTURE_MAP = json.load(f)
    GESTURE_MAP = {int(k): v for k, v in GESTURE_MAP.items()}


def move_workspace(direction):
    os.system(f"bspc desktop --focus -f {['next', 'prev'][direction]}")


def scroll(direction):
    scroll_cmds = [
        "xdotool click 6",
        "xdotool click 7",
        "xdotool click 4",
        "xdotool click 5",
    ]
    for _ in range(5):
        os.system(scroll_cmds[direction])
    # 6 = Scroll Left, 7 = Scroll Right, 4 = Scroll Up, 5 = Scroll Down


def page_scroll(direction):
    cmds = ["xdotool key Page_Up", "xdotool key Page_Down"]
    os.system(cmds[direction - 2])


def adjust_volume(direction):
    os.system(f"pactl set-sink-volume @DEFAULT_SINK@ {['-5%', '+5%'][direction-2]}")


def adjust_brightness(direction):
    os.system(f"light {['-U','-A'][direction]} 10")


def toggle_scratchpad(_):
    os.system("staticpad 0")  # Custom scratchpad script


# Define a function mapping
FUNCTION_MAP = {
    1: move_workspace,
    2: scroll,
    3: adjust_volume,
    4: adjust_brightness,
    5: lambda direction: print(f"Cursor select {direction}"),
    6: lambda direction: print(f"Drawing {direction}"),
    9: page_scroll,
}


def execute_command(gesture_id, direction):
    if gesture_id not in GESTURE_MAP:
        print("Gesture not found in map:", gesture_id)
        return

    function_id = GESTURE_MAP[gesture_id]["function"][direction]
    if function_id is None or function_id == 0:
        print(
            f"No function assigned for {direction} swipe with {GESTURE_MAP[gesture_id]['name']}"
        )
        return

    print(
        f"Executing function {function_id} for {GESTURE_MAP[gesture_id]['name']} ({direction} swipe)"
    )

    # Run the function in a separate thread to avoid blocking
    if function_id in FUNCTION_MAP:
        threading.Thread(target=FUNCTION_MAP[function_id], args=(direction,)).start()
    else:
        print(f"Function {function_id} not defined!")
