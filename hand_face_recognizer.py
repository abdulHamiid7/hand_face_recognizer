"""
Hand Recognizer (Left/Right + landmarks) and Face Rectangle
Uses MediaPipe Hands + Face Detection with OpenCV.

Controls:
    q / ESC -> quit
    s       -> save a screenshot
"""

import time
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

RED = (0, 0, 255)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)

HAND_POINT = mp_drawing.DrawingSpec(color=RED, thickness=3, circle_radius=3)
HAND_LINE = mp_drawing.DrawingSpec(color=WHITE, thickness=2)


def draw_face_boxes(frame, detections):
    """Draw a red rectangle around each detected face."""
    if not detections:
        return
    h, w, _ = frame.shape
    for det in detections:
        box = det.location_data.relative_bounding_box
        x, y = int(box.xmin * w), int(box.ymin * h)
        bw, bh = int(box.width * w), int(box.height * h)
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), RED, 3)
        score = int(det.score[0] * 100)
        cv2.putText(frame, f"Face {score}%", (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)


def draw_hands(frame, results):
    """Draw hand landmarks and a Left/Right label near each wrist."""
    if not results.multi_hand_landmarks:
        return
    h, w, _ = frame.shape
    for landmarks, handedness in zip(results.multi_hand_landmarks,
                                     results.multi_handedness):
        mp_drawing.draw_landmarks(
            frame, landmarks, mp_hands.HAND_CONNECTIONS,
            HAND_POINT, HAND_LINE)

        label = handedness.classification[0].label  # "Left" or "Right"

        # Place the label above the hand's bounding area
        xs = [lm.x for lm in landmarks.landmark]
        ys = [lm.y for lm in landmarks.landmark]
        tx = int(min(xs) * w)
        ty = max(int(min(ys) * h) - 15, 30)
        cv2.putText(frame, label, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, RED, 3)


def main(camera_index=0, width=640, height=480):
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}")

    prev = time.time()
    shots = 0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    ) as hands, mp_face.FaceDetection(
        model_selection=0,            # 0 = within ~2m, 1 = up to ~5m
        min_detection_confidence=0.6,
    ) as faces:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            # Mirror so Left/Right match what you see on screen
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False

            hand_results = hands.process(rgb)
            face_results = faces.process(rgb)

            draw_face_boxes(frame, face_results.detections)
            draw_hands(frame, hand_results)

            now = time.time()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now
            cv2.putText(frame, f"{int(fps)} FPS", (10, 30),
                        cv2.FONT_HERSHEY_PLAIN, 2, GREEN, 2)

            cv2.imshow("Hand & Face Recognizer", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                cv2.imwrite(f"screenshot_{shots}.png", frame)
                print(f"Saved screenshot_{shots}.png")
                shots += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
