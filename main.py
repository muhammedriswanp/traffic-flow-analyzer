"""Entry point for the traffic flow analyzer pipeline."""

import argparse
import sys
from pathlib import Path

import cv2

from src.detect import load_model, detect_vehicles, draw_detections
from src.counter import CounterConfig, VehicleCounter, draw_counter_overlay
from src.report import (
    build_summary,
    save_crossings_csv,
    save_summary_json,
    save_summary_txt,
    log_to_mlflow,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Traffic Flow Analyzer — detect, count, and report vehicles in video."
    )
    p.add_argument("input",  help="Path to source video file")
    p.add_argument(
        "--output", "-o",
        default=None,
        help="Path for annotated output video (default: data/processed/<input_stem>_counted.mp4)",
    )
    p.add_argument(
        "--weights", "-w",
        default="yolov8n.pt",
        help="YOLOv8 weights file (default: yolov8n.pt)",
    )
    p.add_argument(
        "--conf", "-c",
        type=float, default=0.3,
        help="Detection confidence threshold (default: 0.3)",
    )
    p.add_argument(
        "--line-y", "-l",
        type=int, default=None,
        help="Y-pixel for counting line (default: middle of frame)",
    )
    p.add_argument(
        "--report-dir", "-r",
        default="data/reports",
        help="Directory to save CSV / JSON / TXT reports (default: data/reports)",
    )
    p.add_argument(
        "--mlflow", action="store_true",
        help="Log results to MLflow (requires mlflow installed)",
    )
    p.add_argument(
        "--no-video", action="store_true",
        help="Skip writing the annotated output video (faster, reports only)",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> dict:
    """
    Full pipeline: open video → detect → count → annotate → save → report.

    Returns the summary dict.
    """
    input_path  = Path(args.input)
    report_dir  = Path(args.report_dir)
    stem        = input_path.stem

    output_path = Path(
        args.output or f"data/processed/{stem}_counted.mp4"
    )

    # ── Open video ──────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"[main] ERROR: cannot open video: {input_path}", file=sys.stderr)
        sys.exit(1)

    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[main] Input  : {input_path}  ({width}×{height}, {fps:.1f} fps, ~{total_frames} frames)")

    # ── Counter setup ────────────────────────────────────────────────────────
    line_y = args.line_y if args.line_y is not None else height // 2
    cfg    = CounterConfig(line_y=line_y, frame_height=height)
    counter = VehicleCounter(cfg)
    print(f"[main] Counting line at Y={line_y}")

    # ── Video writer ─────────────────────────────────────────────────────────
    writer = None
    if not args.no_video:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        print(f"[main] Output : {output_path}")

    # ── Model ────────────────────────────────────────────────────────────────
    model = load_model(args.weights)
    print("[main] Model loaded ✓  — starting processing…\n")

    # ── Frame loop ───────────────────────────────────────────────────────────
    frame_idx      = 0
    total_detected = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detect_vehicles(model, frame, conf_threshold=args.conf)
        new_crossings = counter.update(detections, frame_idx)
        total_detected += len(detections)

        if new_crossings:
            for ev in new_crossings:
                print(
                    f"  [frame {frame_idx:>5}] {ev.class_name:<12} crossed "
                    f"{'↓ IN' if ev.direction == 'in' else '↑ OUT'}  "
                    f"(track #{ev.track_id})"
                )

        if writer is not None:
            annotated = draw_detections(frame, detections)
            draw_counter_overlay(annotated, counter)
            writer.write(annotated)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"[main] …{frame_idx}/{total_frames} frames processed")

    cap.release()
    if writer:
        writer.release()

    # ── Reports ──────────────────────────────────────────────────────────────
    print(f"\n[main] Processed {frame_idx} frames  |  {total_detected} detections  |  {counter.total} crossings")
    print(f"[main] Breakdown: {counter.counts}\n")

    report_dir.mkdir(parents=True, exist_ok=True)

    csv_path  = report_dir / f"{stem}_crossings.csv"
    json_path = report_dir / f"{stem}_summary.json"
    txt_path  = report_dir / f"{stem}_report.txt"

    save_crossings_csv(counter.crossings, csv_path)

    summary = build_summary(
        counter,
        video_name=input_path.name,
        fps=fps,
        total_frames=frame_idx,
    )
    save_summary_json(summary, json_path)
    save_summary_txt(summary, txt_path)

    # ── MLflow ───────────────────────────────────────────────────────────────
    if args.mlflow:
        artifacts = [csv_path, json_path]
        if not args.no_video and output_path.exists():
            artifacts.append(output_path)
        log_to_mlflow(
            summary,
            run_name=stem,
            artifacts=artifacts,
        )

    print("\n[main] ✅ Done.")
    return summary


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()