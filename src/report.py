"""CSV event log and text summary report generator."""

import json
from pathlib import Path

import pandas as pd


def build_summary(counter, video_name: str, fps: float, total_frames: int) -> dict:
    """
    Build a flat summary dict from a finished VehicleCounter.

    Args:
        counter:      VehicleCounter instance after processing.
        video_name:   Source video filename (for the report header).
        fps:          Video frame rate.
        total_frames: Number of frames processed.

    Returns:
        Dict with total counts, per-class breakdown, and flow rate.
    """
    duration_s = total_frames / fps if fps > 0 else 0

    summary = {
        "video":             video_name,
        "total_frames":      total_frames,
        "duration_s":        round(duration_s, 2),
        "fps":               round(fps, 2),
        "total_in":          counter.total_in,
        "total_out":         counter.total_out,
        "total":             counter.total_in + counter.total_out,
    }

    # Per-class breakdown
    for cls, dirs in counter.count.items():
        summary[f"{cls}_in"]  = dirs["in"]
        summary[f"{cls}_out"] = dirs["out"]

    # Flow rate (vehicles per minute, inbound)
    if duration_s > 0:
        summary["flow_rate_per_min"] = round(counter.total_in / duration_s * 60, 1)

    return summary


def save_crossings_csv(crossings: list[dict], path: str | Path) -> None:
    """
    Save the per-vehicle crossing event log as a CSV file.

    Columns: frame, track_id, class, direction
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if crossings:
        df = pd.DataFrame(crossings)
    else:
        df = pd.DataFrame(columns=["frame", "track_id", "class", "direction"])

    df.to_csv(path, index=False)
    print(f"[report] Crossings CSV → {path}  ({len(df)} events)")


def save_summary_txt(summary: dict, path: str | Path) -> None:
    """
    Save a human-readable plain-text summary report.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "=" * 44,
        "  TRAFFIC FLOW ANALYZER — SUMMARY REPORT",
        "=" * 44,
        f"  Video          : {summary.get('video', 'N/A')}",
        f"  Duration       : {summary.get('duration_s', 0):.1f} s  ({summary.get('fps', 0)} fps)",
        f"  Frames         : {summary.get('total_frames', 0)}",
        "-" * 44,
        f"  Total vehicles : {summary.get('total', 0)}",
        f"  Inbound  (IN)  : {summary.get('total_in', 0)}",
        f"  Outbound (OUT) : {summary.get('total_out', 0)}",
    ]

    if "flow_rate_per_min" in summary:
        lines.append(f"  Flow rate      : {summary['flow_rate_per_min']} veh/min (inbound)")

    lines.append("-" * 44)
    lines.append("  Per-class breakdown:")

    vehicle_classes = ["car", "motorcycle", "bus", "truck"]
    for cls in vehicle_classes:
        in_count  = summary.get(f"{cls}_in",  0)
        out_count = summary.get(f"{cls}_out", 0)
        if in_count + out_count > 0:
            lines.append(f"    {cls:<12} IN: {in_count:<4} OUT: {out_count}")

    lines.append("=" * 44)

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[report] Summary TXT  → {path}")


def save_summary_json(summary: dict, path: str | Path) -> None:
    """Save summary as JSON (useful for downstream tooling)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[report] Summary JSON → {path}")