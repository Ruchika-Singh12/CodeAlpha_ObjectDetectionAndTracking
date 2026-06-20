"""
Task 4: Object Detection and Tracking
Real-time video input (webcam/file) -> YOLO detection -> SORT tracking -> live display
"""

import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

VIDEO_SOURCE = 0          # 0 = webcam, or path to a video file e.g. "input.mp4"
MODEL_PATH   = "yolov8n.pt"   # pre-trained YOLOv8 nano model (auto-downloads)
CONF_THRESH  = 0.4
SAVE_OUTPUT  = True
OUTPUT_PATH  = "output/output.mp4"

# Random colors per track ID (so each tracked object keeps a consistent color)
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")


def get_color(track_id):
    return tuple(int(c) for c in COLORS[int(track_id) % len(COLORS)])


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Loading YOLOv8 model...")
    model = YOLO(MODEL_PATH)
    class_names = model.names
    print("  Model loaded successfully!")
    print("=" * 55)

    tracker = Sort(max_age=15, min_hits=3, iou_threshold=0.3)

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("ERROR: Could not open video source.")
        return

    fps_in    = cap.get(cv2.CAP_PROP_FPS) or 30
    width     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if SAVE_OUTPUT:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps_in, (width, height))

    prev_time = cv2.getTickCount()
    frame_count = 0

    print("Press 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream.")
            break

        frame_count += 1

        # ── STEP 1: Run YOLO detection on this frame ──
        results = model(frame, conf=CONF_THRESH, verbose=False)[0]

        detections = []
        labels_map = {}
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            detections.append([x1, y1, x2, y2, conf])
            labels_map[(round(x1, 1), round(y1, 1))] = class_names[cls_id]

        dets_np = np.array(detections) if detections else np.empty((0, 5))

        # ── STEP 2: Update SORT tracker with detections ──
        tracked_objects = tracker.update(dets_np)

        # ── STEP 3: Draw bounding boxes + labels + tracking IDs ──
        for *box, track_id in tracked_objects:
            x1, y1, x2, y2 = map(int, box)
            color = get_color(track_id)

            # Find nearest detection label (best-effort class name lookup)
            label = "object"
            best_dist = float("inf")
            for (lx, ly), cname in labels_map.items():
                dist = abs(lx - x1) + abs(ly - y1)
                if dist < best_dist:
                    best_dist = dist
                    label = cname

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"ID {int(track_id)} | {label}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, text, (x1 + 3, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # ── FPS counter ──
        curr_time = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Objects Tracked: {len(tracked_objects)}", (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Object Detection & Tracking | Task 4", frame)

        if writer is not None:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting...")
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print(f"\nProcessed {frame_count} frames.")
    if SAVE_OUTPUT:
        print(f"Output saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
