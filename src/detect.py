"""Vehicle detection and tracking logic using YOLOv8."""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# COCO class IDs for vehicles
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# BGR colors per class for bounding boxes
CLASS_COLORS = {
    "car":        (0, 255, 0),    # green
    "motorcycle": (255, 165, 0),  # orange
    "bus":        (0, 0, 255),    # red
    "truck":      (255, 0, 255),  # magenta
}

def load_model(weights: str = "models/yolov8m.pt") -> YOLO:
    return YOLO(weights)


def open_video(path: str | Path) -> tuple:
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    meta = {
        "width":        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height":       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps":          cap.get(cv2.CAP_PROP_FPS) or 25.0,
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    return cap, meta


def make_writer(path: str | Path, width: int, height: int, fps: float) -> cv2.VideoWriter:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, fps, (width, height))


def track_vehicles(model: YOLO, frame: np.ndarray, conf: float = 0.35) -> object:
    """
    Run YOLOv8 + ByteTrack on a single frame.

    imgsz=INFERENCE_SIZE downscales 4K frames before inference so the model
    sees vehicles at a scale closer to its training distribution (640px).
    Bboxes are automatically rescaled back to original frame dimensions.
    """
    results = model.track(
        frame,
        persist=True,
        classes=list(VEHICLE_CLASSES.keys()),
        conf=conf,
        verbose=False,
    )[0]
    return results


def draw_tracks(frame: np.ndarray, results) -> None:
    """Draw bounding boxes, labels, and track IDs on frame (in-place)."""
    if results.boxes is None or len(results.boxes) == 0:
        return

    has_ids = results.boxes.id is not None

    for i, box in enumerate(results.boxes.xyxy):
        x1, y1, x2, y2 = map(int, box)
        cls_id     = int(results.boxes.cls[i])
        conf       = float(results.boxes.conf[i])
        class_name = results.names[cls_id]
        color      = CLASS_COLORS.get(class_name, (200, 200, 200))

        if has_ids:
            track_id = int(results.boxes.id[i])
            label = f"#{track_id} {class_name} {conf:.2f}"
        else:
            label = f"{class_name} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - baseline - 4), (x1 + tw, y1), color, -1)
        cv2.putText(
            frame, label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (0, 0, 0), 1, cv2.LINE_AA,
        )


# ── legacy detection-only pipeline ───────────────────────────────────────────

def detect_vehicles(model, frame, conf_threshold=0.35):
    results = model(frame, verbose=False)[0]
    detections = []
    for box in results.boxes:
        class_id = int(box.cls[0])
        if class_id not in VEHICLE_CLASSES:
            continue
        conf = float(box.conf[0])
        if conf < conf_threshold:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detections.append({
            "class_id":   class_id,
            "class_name": VEHICLE_CLASSES[class_id],
            "confidence": conf,
            "bbox":       (x1, y1, x2, y2),
        })
    return detections


def draw_detections(frame, detections):
    annotated = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = f"{det['class_name']} {det['confidence']:.2f}"
        color = CLASS_COLORS.get(det["class_name"], (200, 200, 200))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(annotated, (x1, y1 - th - baseline - 4), (x1 + tw, y1), color, -1)
        cv2.putText(annotated, label, (x1, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return annotated


def process_video(input_path, output_path, weights="yolov8n.pt", conf_threshold=0.35):
    input_path  = Path(input_path)
    output_path = Path(output_path)
    model = load_model(weights)
    cap, meta = open_video(input_path)
    writer = make_writer(output_path, meta["width"], meta["height"], meta["fps"])
    stats = {"total_frames": 0, "total_detections": 0, "detections_per_class": {}}
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            detections = detect_vehicles(model, frame, conf_threshold)
            annotated  = draw_detections(frame, detections)
            writer.write(annotated)
            stats["total_frames"] += 1
            stats["total_detections"] += len(detections)
            for det in detections:
                name = det["class_name"]
                stats["detections_per_class"][name] = stats["detections_per_class"].get(name, 0) + 1
    finally:
        cap.release()
        writer.release()
    print(f"[detect] {stats['total_frames']} frames → {output_path}")
    print(f"[detect] detections: {stats['total_detections']}  by class: {stats['detections_per_class']}")
    return stats


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python detect.py <input_video> [output_video]")
        sys.exit(1)
    process_video(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "data/processed/output_annotated.mp4")