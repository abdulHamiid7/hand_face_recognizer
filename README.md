# Hand & Face Recognizer

A real-time computer vision app built with Python, OpenCV and MediaPipe. It uses your webcam to:

- Detect your **face** and draw a red rectangle around it, with a confidence score
- Detect up to **two hands**, draw their 21 landmarks, and label each one as **Left** or **Right**
- Show a live **FPS** counter

## Requirements

- Python 3.8 – 3.12
- A webcam
- Packages:
  - `opencv-python`
  - `mediapipe==0.10.14`

## Installation

```bash
# (optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python hand_face_recognizer.py
```

A window opens showing your camera feed with the detections drawn on it.

### Controls

| Key        | Action                  |
|------------|-------------------------|
| `q` / `ESC` | Quit the app           |
| `s`        | Save a screenshot (`screenshot_0.png`, `screenshot_1.png`, ...) |

## Configuration

Change these in `hand_face_recognizer.py`:

| Setting | Where | Description |
|---------|-------|-------------|
| `camera_index` | `main()` | Which camera to use (`0` is the default webcam, try `1` for an external one) |
| `width`, `height` | `main()` | Capture resolution. Lower values give a higher FPS |
| `max_num_hands` | `mp_hands.Hands(...)` | Maximum number of hands to track |
| `model_complexity` | `mp_hands.Hands(...)` | `0` is faster, `1` is more accurate |
| `model_selection` | `mp_face.FaceDetection(...)` | `0` for faces within ~2 m, `1` for faces up to ~5 m |
| `min_detection_confidence` | both detectors | Raise to reduce false detections |

## How It Works

1. Each frame is read from the webcam and **mirrored**, so it behaves like a mirror.
2. The frame is converted from BGR to RGB, as MediaPipe expects.
3. **MediaPipe Face Detection** returns a bounding box, which is drawn as a red rectangle.
4. **MediaPipe Hands** returns 21 landmarks and a handedness label per hand. The landmarks are drawn, and the label is placed above each hand.
5. The FPS is calculated from the time between frames and drawn in the corner.

## Troubleshooting

- **Camera does not open:** close other apps using the camera, or change `camera_index`.
- **Left and Right look swapped:** remove the `cv2.flip(frame, 1)` line in `main()`.
- **Low FPS:** lower the resolution, or set `model_complexity=0`.
- **`mp.solutions` errors:** make sure you installed the pinned version, `mediapipe==0.10.14`.

## Project Structure

```
.
├── hand_face_recognizer.py
└── README.md
```

## Developed by Eng "AbdulHamiid"
