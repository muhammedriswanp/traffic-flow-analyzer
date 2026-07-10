"""Traffic Flow Analyzer — entry point."""

import argparse
from pathlib import Path

from src.detect import load_model, open_video, make_writer, track_vehicles, draw_tracks
from src.counter import CounterConfig, VehicleCounter
from src.report import build_summary, save_crossings_csv, save_summary_txt
from src.visualize import save_dashboard


def run(input_path: Path, no_dashboard: bool = False) -> None:
    # ── output paths ─────────────────────────────────────────────────────
    out_dir   = Path("data/reports")
    video_out = Path("data/processed/output_annotated.mp4")
    csv_out   = out_dir / "crossings.csv"
    txt_out   = out_dir / "report.txt"
    dash_out  = out_dir / "dashboard.png"

    # ── setup ────────────────────────────────────────────────────────────
    model      = load_model()
    cap, meta  = open_video(input_path)
    writer     = make_writer(video_out, meta["width"], meta["height"], meta["fps"])

    config  = CounterConfig(line_y=int(meta["height"] * 0.65))
    counter = VehicleCounter(config)

    # ── frame loop ───────────────────────────────────────────────────────
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = track_vehicles(model, frame)
        counter.update(results, frame_idx=frame_idx)
        draw_tracks(frame, results)
        counter.draw_overlay(frame)
        writer.write(frame)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"[main] Frame {frame_idx}  |  IN: {counter.total_in}  OUT: {counter.total_out}")

    cap.release()
    writer.release()
    print(f"[main] Done — {frame_idx} frames → {video_out}")

    # ── reports ──────────────────────────────────────────────────────────
    summary = build_summary(
        counter,
        video_name=input_path.name,
        fps=meta["fps"],
        total_frames=frame_idx,
    )

    save_crossings_csv(counter.crossings, csv_out)
    save_summary_txt(summary, txt_out)

    if not no_dashboard:
        save_dashboard(summary, counter.crossings, dash_out)

    # ── terminal summary ─────────────────────────────────────────────────
    print()
    print(f"  Total : {summary['total']}")
    print(f"  IN    : {summary['total_in']}")
    print(f"  OUT   : {summary['total_out']}")
    if "flow_rate_per_min" in summary:
        print(f"  Flow  : {summary['flow_rate_per_min']} veh/min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traffic Flow Analyzer")
    parser.add_argument("input", help="Path to input video")
    parser.add_argument("--no-dashboard", action="store_true", help="Skip dashboard PNG")
    args = parser.parse_args()

    run(Path(args.input), no_dashboard=args.no_dashboard)