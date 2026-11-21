# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Dynago is a Linux desktop gesture controller. It uses a webcam to capture hand gestures via MediaPipe, classifies them with a trained SVM, then maps gesture+direction combos to system commands (workspace switching, scrolling, volume, brightness, etc.).

All scripts must be run from the **project root** — paths like `dynago/data/raw.csv` and `dynago/models/gesture_svm.pkl` are relative to it.

## Commands

This project uses `uv` with `poetry-core` as the build backend. Scripts are defined in `pyproject.toml`:

```bash
# Run the main application (gesture recognition + command execution)
uv run dev

# Test gesture recognition visually (no command execution, shows labels on screen)
uv run test

# Training pipeline (run in order)
uv run capture        # Capture raw landmark data (interactive, press SPACE per sample)
uv run normalize      # Normalize raw.csv → normalized.csv (incremental)
uv run train_static   # Train gesture SVM → dynago/models/gesture_svm.pkl

# Data analysis utilities
uv run visualize      # Plot dataset distribution
uv run analyze        # Generate performance metrics and plots
```

## Architecture

### Two-stage recognition pipeline

1. **Static gesture classifier** — every `N_FRAMES` (default 6), 21 normalized hand landmarks are fed into `gesture_svm.pkl`. Returns a gesture ID (0 = no gesture / fist).
2. **Swipe direction detector** — once a gesture is identified, the system switches to motion tracking mode. It tracks the mean position of the gesture's configured landmarks across frames and computes direction from start→end displacement angle. Requires movement above `VEL_THRESHOLD` (default 0.25 normalized units).

### Multiprocessing architecture (`src/main.py`, `src/capture.py`)

- **Main process**: runs `capture_landmarks()` — OpenCV camera loop, MediaPipe inference, gesture classification, swipe detection
- **Worker process**: runs `command_worker()` — receives `(gesture_id, direction)` tuples from a `Queue` and calls `execute_command()`
- The two processes communicate via a bounded `multiprocessing.Queue(maxsize=10)` to prevent backpressure

### Gesture → command mapping

`dynago/data/gesture_map.json` is the central configuration:
```json
{ "gesture_id": { "name": "...", "landmarks": [indices_to_track], "function": [left, right, down, up] } }
```
The `function` array maps each of the 4 swipe directions to a function ID. Function IDs are resolved in `src/command.py`'s `FUNCTION_MAP` to actual OS commands (via `xdotool`, `pactl`, `bspc`, `light`).

`dynago/config.py` is where global tunables live: `N_FRAMES`, `VEL_THRESHOLD`, `DISPLAY_TIME`, `ENABLE_MOUSE`.

### Mouse control mode

When `ENABLE_MOUSE = True` in `config.py`, a "point" gesture (gesture ID 4) activates `GestureMouse` (`src/mouse.py`) — the index finger tip drives the cursor with exponential smoothing, bypassing swipe detection entirely.

### Training data flow

```
capture → dynago/data/raw.csv
normalize → dynago/data/normalized.csv   (incremental: only processes new rows)
train_static → dynago/models/gesture_svm.pkl
```

Raw data: 21 landmarks × 3 coords (x, y, z) = 63 features + gesture label.
Normalization: center on wrist (landmark 0), scale to unit sphere.

### Key files

| File | Role |
|------|------|
| `dynago/config.py` | Global tunables and gesture map loader |
| `dynago/data/gesture_map.json` | Gesture ID → name, tracking landmarks, command functions |
| `dynago/src/capture.py` | Camera loop, landmark normalization, SVM inference, swipe state machine |
| `dynago/src/swipe.py` | Swipe direction calculation from landmark history deques |
| `dynago/src/command.py` | Function ID → OS command dispatch (xdotool, pactl, bspc, light) |
| `dynago/src/mouse.py` | `GestureMouse` class — threaded mouse control via pynput |
| `dynago/src/gaze.py` | Standalone experimental eye-tracking script (not integrated into main) |
| `dynago/src/test.py` | Standalone visual tester — shows gesture name and swipe text on-screen |

## System Dependencies

The command functions in `src/command.py` call Linux-specific tools: `bspc` (BSPWM), `xdotool`, `pactl` (PulseAudio), `light` (backlight), and a custom `staticpad` script. These are not Python dependencies and must be installed separately on the host.
