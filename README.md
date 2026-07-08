# traffic-flow-analyzer

An AI-powered traffic video analytics system that detects, tracks, and counts vehicles from surveillance videos using YOLOv8 and OpenCV, generating traffic reports and visualizations for traffic engineering analysis.

## Project Structure

```
traffic-flow-analyzer/
├── main.py                 # Entry point (WIP)
├── requirements.txt        # Python dependencies
├── models/                 # YOLO weights directory
├── data/                   # Input/output video data
├── src/
│   ├── detect.py           # Vehicle detection using YOLOv8
│   ├── counter.py          # Line-crossing vehicle counter
│   ├── visualize.py        # Plots and dashboard visualizations
│   └── report.py           # CSV and statistics report generator
└── README.md
```

## Detection Module (`src/detect.py`)

- Uses [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for object detection.
- Detects **4 vehicle classes**: car, motorcycle, bus, truck (COCO dataset IDs 2, 3, 5, 7).
- Each class is drawn with a distinct BGR color (green, orange, red, magenta).
- Default confidence threshold: **0.3**.
- `process_video(input_path, output_path, weights, conf_threshold)` — processes every frame of a video, draws bounding boxes with class labels and confidence, and writes an annotated MP4 output. Returns a summary dict with `total_frames`, `total_detections`, and `detections_per_class`.

### Sample Output

![Detection Example](data/processed/Screenshot%202026-07-07%20172538.png)

### CLI Usage

```bash
python src/detect.py data/input_video.mp4 data/processed/output.mp4
```

If no output path is specified, the result is saved to `data/processed/output_annotated.mp4`.

## Setup

```bash
pip install -r requirements.txt
```

The YOLO weights (`yolov8n.pt`) are downloaded automatically on first run.
