# Adaptive Multimodal AI for Emotion Consistency & Deception Detection

##  Overview

This project is a real-time AI system designed to detect suspicious or deceptive behavior using multiple signals.

The system analyzes:

* Eye blink rate
* Eye gaze
* Head pose
* Shoulder movement
* Heart rate (via Arduino)

By combining these signals, the system improves reliability compared to single-method approaches.

---

##  Technologies Used

* Python
* OpenCV
* MediaPipe
* Tkinter
* Arduino
* SQLite

---

##  How It Works

1. The system captures live video using a webcam
2. Extracts facial and behavioral features
3. Reads heart rate data from Arduino
4. Performs baseline calibration
5. Detects deviations indicating suspicious behavior

---

##  How to Run

```bash
pip install -r requirements.txt
python src/main.py
```

---

##  Project Structure

* `src/` → Main Python source code
* `hardware/` → Arduino code
* `data/` → Calibration data

---

##  Future Scope

* Deep learning integration
* Speech analysis
* Web/mobile deployment
* Improved accuracy using large datasets

---


