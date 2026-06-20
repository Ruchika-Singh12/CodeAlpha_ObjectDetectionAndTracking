# 🎥 Task 4: Object Detection and Tracking
### Real-Time Detection (YOLOv8) + Multi-Object Tracking (SORT)

---

## 📌 Project Overview

This project detects objects in a live webcam feed (or video file) using a
**pre-trained YOLOv8 model**, then tracks each detected object across frames
using the **SORT (Simple Online and Realtime Tracking)** algorithm — assigning
a consistent **tracking ID** to every object as it moves.

| Component          | Technique                                   |
|---------------------|----------------------------------------------|
| Video Input         | OpenCV (`cv2.VideoCapture`) — webcam or file |
| Object Detection     | YOLOv8 (pre-trained, Ultralytics)            |
| Bounding Boxes       | Drawn per frame from YOLO output             |
| Object Tracking      | SORT (Kalman Filter + Hungarian Algorithm)   |
| Output Display       | Live window with labels + tracking IDs + FPS |

---

## 📁 Project Structure

```
object_detection/
├── app.py              ← Main script (detection + tracking + display)
├── sort.py             ← SORT tracking algorithm implementation
├── requirements.txt    ← Python dependencies
├── README.md           ← This file
└── output/
    └── output.mp4       ← Saved result video (auto-generated)
```

---

## ⚙️ Setup Instructions

### Step 1: Open this folder in VS Code
```
File → Open Folder → object_detection
```
Make sure the terminal is opened **inside this folder** (right-click folder → "Open in Integrated Terminal").

### Step 2: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the app
```bash
python app.py
```

The **YOLOv8n model (`yolov8n.pt`)** will auto-download on first run (~6 MB).

### Step 4: Controls
- Live window opens showing detection + tracking.
- Press **`q`** to quit.
- Result is also saved to `output/output.mp4`.

---

## 🧠 How It Works

```
Video Frame (Webcam/File)
        ↓
YOLOv8 Detection → bounding boxes + class + confidence
        ↓
SORT Tracker → matches detections to existing tracks
        ↓             (Kalman Filter predicts motion,
        ↓              Hungarian Algorithm matches IDs)
Draw boxes + Track ID + Label + FPS
        ↓
Display + Save Output Video
```

---

## 🎛️ Configuration (edit in `app.py`)

| Variable        | Description                                   |
|------------------|------------------------------------------------|
| `VIDEO_SOURCE`   | `0` for webcam, or `"video.mp4"` for a file file |
| `MODEL_PATH`     | YOLO model — `yolov8n.pt` (fast) up to `yolov8x.pt` (accurate) |
| `CONF_THRESH`    | Minimum detection confidence (default `0.4`)   |
| `SAVE_OUTPUT`    | `True`/`False` — save result video              |

**To use a video file instead of webcam:**
```python
VIDEO_SOURCE = "input_video.mp4"
```
Place your video file in the project folder.

---

## 📚 Technologies Used

- **Python 3.x**
- **OpenCV** – video capture, drawing, display
- **Ultralytics YOLOv8** – pre-trained object detection model
- **SORT** – Kalman Filter + Hungarian Algorithm for multi-object tracking
- **SciPy** – linear assignment (Hungarian algorithm) for SORT
- **NumPy** – numerical operations

---

## ✅ Task Requirements Covered

- ✔ Real-time video input using webcam or video file (OpenCV)
- ✔ Pre-trained model (YOLOv8) for object detection
- ✔ Each frame processed — bounding boxes drawn
- ✔ Object tracking applied using SORT algorithm
- ✔ Output displayed live with labels + tracking IDs + FPS counter

---

*Built as Task 4: Object Detection and Tracking — Computer Vision Project*
