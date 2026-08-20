"""
TASK 4: Object Detection and Tracking
--------------------------------------
- Real-time video input (webcam or video file) via OpenCV
- Pre-trained YOLOv8 model for object detection
- Per-frame processing with bounding boxes
- Object tracking (ByteTrack, SORT-family algorithm) with persistent IDs
- Real-time display with labels + tracking IDs

Install dependencies first:
    pip install ultralytics opencv-python

Usage:
    python object_detection_tracking.py --source 0            # webcam
    python object_detection_tracking.py --source video.mp4    # video file
"""

import argparse
import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time object detection and tracking")
    parser.add_argument(
        "--source", type=str, default="0",
        help="Video source: webcam index (e.g. 0) or path to a video file"
    )
    parser.add_argument(
        "--model", type=str, default="yolov8n.pt",
        help="Pre-trained YOLO model to use (yolov8n/s/m/l/x.pt). "
             "'n' (nano) is fastest and downloads automatically on first run."
    )
    parser.add_argument(
        "--tracker", type=str, default="bytetrack.yaml",
        choices=["bytetrack.yaml", "botsort.yaml"],
        help="Tracking algorithm config (ByteTrack or BoT-SORT, both SORT-family trackers)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.4,
        help="Confidence threshold for detections"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # --- Set up video input (webcam index or file path) ---
    source = int(args.source) if args.source.isdigit() else args.source

    # --- Load pre-trained detection model ---
    model = YOLO(args.model)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of stream or cannot read frame.")
            break

        # --- Process each frame: detect + track in one call ---
        # model.track() runs detection then feeds boxes into the tracker
        # (ByteTrack/BoT-SORT), assigning a persistent ID to each object.
        results = model.track(
            frame,
            persist=True,          # keep track IDs consistent across frames
            tracker=args.tracker,
            conf=args.conf,
            verbose=False,
        )

        result = results[0]
        boxes = result.boxes

        if boxes is not None and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            track_ids = boxes.id.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)

            for box, track_id, conf, cls_id in zip(xyxy, track_ids, confs, cls_ids):
                x1, y1, x2, y2 = map(int, box)
                label = model.names[cls_id]

                # --- Draw bounding box ---
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # --- Draw label + tracking ID ---
                text = f"{label} ID:{track_id} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 255, 0), -1)
                cv2.putText(frame, text, (x1 + 2, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # --- Display output in real time ---
        cv2.imshow("Object Detection & Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()