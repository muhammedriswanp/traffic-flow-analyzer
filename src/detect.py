"""Vehicle detection logic using YOLOv8."""

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


def load_model(weights: str = "models/yolov8n.pt") -> YOLO:
    """Load a YOLOv8 model. Downloads pretrained weights on first run."""
    model = YOLO(weights)
    return model


def detect_vehicles(
    model: YOLO,
    frame: np.ndarray,
    conf_threshold: float = 0.3,
) -> list[dict]:
    """
    Run YOLOv8 inference on a single frame and return vehicle detections.

    Args:
        model: Loaded YOLO model.
        frame: BGR image array (H x W x 3).
        conf_threshold: Minimum confidence to keep a detection.

    Returns:
        List of dicts with keys: class_id, class_name, confidence, bbox (x1,y1,x2,y2).
    """
    results = model(frame, verbose=False)[0]        #Show detailed information while the program is running.
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


def draw_detections(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    """
    Draw bounding boxes and labels on a copy of the frame.

    Args:
        frame: Original BGR frame.
        detections: List returned by detect_vehicles().

    Returns:
        Annotated BGR frame.
    """
    annotated = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = f"{det['class_name']} {det['confidence']:.2f}"
        color = CLASS_COLORS.get(det["class_name"], (200, 200, 200))

        # Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Label background
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(annotated, (x1, y1 - th - baseline - 4), (x1 + tw, y1), color, -1)

        # Label text
        cv2.putText(
            annotated, label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (0, 0, 0), 1, cv2.LINE_AA,
        )

    return annotated


def process_video(
    input_path: str | Path,
    output_path: str | Path,
    weights: str = "yolov8n.pt",
    conf_threshold: float = 0.3,
) -> dict:
    """
    Run vehicle detection on every frame of a video and save annotated output.

    Args:
        input_path:  Path to the source video.
        output_path: Path to write the annotated video (MP4).
        weights:     YOLOv8 model weights file.
        conf_threshold: Minimum detection confidence.

    Returns:
        Summary dict: total_frames, total_detections, detections_per_class.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = load_model(weights)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_path}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")        # #4-letter code that identifies the video codec.
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

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
                stats["detections_per_class"][name] = (
                    stats["detections_per_class"].get(name, 0) + 1
                )

    finally:
        cap.release()
        writer.release()

    print(f"[detect] Processed {stats['total_frames']} frames → {output_path}")
    print(f"[detect] Total detections: {stats['total_detections']}")
    print(f"[detect] By class: {stats['detections_per_class']}")

    return stats

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python detect.py <input_video> [output_video]")
        sys.exit(1)

    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else "data/processed/output_annotated.mp4"
    process_video(src, dst)