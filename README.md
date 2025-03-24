# 🚀 Dynago - Dynamic and Natural Gesture Operations

## 🎯 Overview
**Dynago** is an AI-powered gesture recognition system designed for **Linux-based computing**. Using **computer vision** and **machine learning**, it allows users to control their system effortlessly with hand gestures detected via a standard webcam. This eliminates the reliance on traditional input devices like keyboards and mice, enhancing accessibility and user experience.

### ✨ Key Features
✅ **Full system control** using AI-powered gestures  
✅ **Mediapipe & SVM-based classification** for accuracy  
✅ **No additional hardware required** – just a webcam!  
✅ **Lightweight & optimized for low-resource devices**  
✅ **Applications in medical, industrial, and assistive tech**  

---

## 🛠 Installation

Clone the repository:
```bash
git clone https://github.com/KreativeThinker/Dynago
cd Dynago
```

Initialize a virtual environment and install dependencies:
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

---

## 🚀 Usage

| Command                | Task                                                                                                        |
|-----------------------|------------------------------------------------------------------------------------------------------------|
| `poetry run capture`   | Opens camera, takes gesture name as input, captures frames & landmark data. Press 'space' to capture, 'q' to exit. |
| `poetry run normalize` | Normalizes collected data and stores it in `dynago/data/normalized.csv`.                                   |
| `poetry run train`     | Trains the AI model and saves it for future use.                                                           |
| `poetry run dev`       | Runs a simple single-frame capture and prediction model.                                                   |

---

## 🏗 System Architecture

📌 **How Dynago Works:**

![System Architecture](architecture.png)

1️⃣ **Initialization**: Load system libraries, initialize the webcam, and set parameters.  
2️⃣ **Base Gesture Detection**: Capture frames, extract hand landmarks via **Mediapipe**, convert to feature vectors, and classify static gestures using **SVM**.  
3️⃣ **Dynamic Gesture Recognition**: Detect motion by computing velocity vectors between frames, classify gestures dynamically with **SVM & motion thresholds**.  
4️⃣ **Action Execution**: Trigger corresponding system actions based on recognized gestures.  

🔹 **Designed for Efficiency & Accessibility** – Works on standard webcams without external hardware, making gesture-based computing widely accessible.  

---

## 📌 Future Enhancements
🚀 Improve gesture recognition accuracy with deep learning (CNNs/RNNs).  
🚀 Enhance real-time responsiveness with GPU acceleration.  
🚀 Expand gesture mappings for additional Linux system controls.  

---

## 🤝 Contributing
We welcome contributions! Feel free to **fork** this repository, create a new **branch**, and submit a **pull request** with improvements.  

---

## 📜 License
Dynago is **open-source** and available under the **MIT License**.  

For more details, refer to the project documentation or research references in the provided presentation.  

🚀 *Let's make gesture-based computing the future!*
