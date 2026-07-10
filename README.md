# Traffic Flow Analyzer

A vehicle detection and counting system built on YOLOv8 (deep learning) and OpenCV,
tracking vehicles across frames and reporting flow rates from traffic footage.

---

## What it does

- Detects vehicles (car, truck, bus, motorcycle) using YOLOv8n
- Tracks each vehicle across frames using ByteTrack (built into Ultralytics)
- Counts vehicles crossing a horizontal line, split by direction (IN / OUT)
- Exports a CSV event log, plain-text summary, and a 2×2 dashboard PNG

---

## Project Structure

```
traffic-flow-analyzer/
├── main.py               # Entry point
├── requirements.txt
├── models/               # YOLOv8 weights (auto-downloaded on first run)
├── data/
│   ├── raw/             # Input videos (not tracked in git)
│   ├──processed/        # Output video
│   └──reports           # CSV, report, dashboard
└── src/
    ├── detect.py         # YOLOv8 detection + tracking
    ├── counter.py        # Line-crossing counter
    ├── report.py         # CSV and text report generation
    └── visualize.py      # Dashboard PNG
```

---

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

YOLOv8 weights (`yolov8n.pt`) are downloaded automatically on first run.

---

## Usage

```bash
python main.py data/raw/your_video.mp4
```

### Outputs (written to `data/processed/`)

| File | Description |
|---|---|
| `output_annotated.mp4` | Video with bounding boxes, track IDs, and count line |
| `crossings.csv` | Per-vehicle crossing events (frame, track_id, class, direction) |
| `report.txt` | Plain-text summary |
| `dashboard.png` | 2×2 chart dashboard |

---

## Sample Results

Tested on a highway video (4509 frames, ~2.5 min):

```
Total vehicles : 139
IN             : 115
OUT            : 24
Flow rate      : 45.9 veh/min
```

---

## Stack

| Tool | Purpose |
|---|---|
| [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) | Detection + ByteTrack tracking |
| OpenCV | Video I/O and frame annotation |
| pandas | Crossing event log |
| matplotlib | Dashboard charts |

---

## Known Limitations

- YOLOv8s (small) is used for speed — occasional misclassification between similar classes (e.g. car vs truck) is expected. Swappable with `yolov8m.pt` for better accuracy.
- Counting line is placed at the vertical midpoint of the frame by default. Videos where vehicles don't cross the midline will show zero counts.
- Optimised for standard traffic camera angles. Very high resolution (4K+) or fisheye footage may reduce detection quality.

---

## Future Scope

### Adaptive Signal Timing

Most traffic signals run on a fixed timer — green for 60 seconds, red for 60 seconds, regardless of actual traffic volume. This system lays the groundwork for a smarter approach: measuring the actual vehicle count per direction and allocating green time proportionally to the busier lanes, reducing unnecessary wait times and improving junction throughput.