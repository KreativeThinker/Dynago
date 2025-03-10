# Dynago

Dynamic and Natural Gesture Operations

# Installation:

```bash
git Clone https://github.com/KreativeThinker/Dynago

cd Dynago
```

Init venv and install packages

```bash
python3.12 -m venv .venv
# OR
python -m venv .venv

# Activate venv
source .venv/bin/activate

# Install packages
pip install poetry
poetry install
```

# Usage

| Command                | Task                                                                                                        |
| ---------------------- | ----------------------------------------------------------------------------------------------------------- |
| `poetry run capture`   | Opens camera after taking gesture name as input. Press space to capture frames and landmark data. q to exit |
| `poetry run normalize` | Normalizes data and stores it in `dynago/data/normalized.csv`                                               |
| `poetry run train`     | Train model and store it for future use                                                                     |
| `poetry run dev`       | Run a simple single frame capture and prediction model                                                      |
