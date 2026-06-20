"""
Optional helper: generates a simple synthetic test video with moving shapes.
Useful for testing the detection/tracking pipeline without a webcam.
NOTE: YOLO is trained on real objects, so for a true test, use a real webcam
or download a sample video (e.g. a street/traffic clip) and set VIDEO_SOURCE
in app.py to that file's path.
"""

import cv2
import numpy as np

def generate_test_video(path="test_video.mp4", frames=150, w=640, h=480):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 30, (w, h))

    for i in range(frames):
        frame = np.full((h, w, 3), 30, dtype=np.uint8)
        x = int(50 + (i * 3) % (w - 100))
        cv2.rectangle(frame, (x, 100), (x + 80, 180), (0, 200, 0), -1)
        cv2.circle(frame, (w - x, 300), 40, (0, 0, 200), -1)
        writer.write(frame)

    writer.release()
    print(f"Test video saved to: {path}")
    print("NOTE: This contains simple shapes, not real-world objects.")
    print("For meaningful YOLO detections, use a real webcam or a real video clip.")


if __name__ == "__main__":
    generate_test_video()
