import csv, json
from datetime import datetime
from pathlib import Path

CROSSINGS_HEADER = ["frame", "track_id", "class_name", "direction", "center_y"]

def save_crossings_csv(crossings: list[dict], output_path: str | Path) -> Path:
    """
    Write every crossing event to a CSV file.
 
    Args:
        crossings:   List of dicts from VehicleCounter.crossings.
        output_path: Where to save the CSV.
 
    Returns:
        Path to the written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CROSSINGS_HEADER)
        writer.writeheader()
        writer.writerows(crossings)

    print(f"[report] Crossings CSV → {output_path}  ({len(crossings)} rows)")
    return output_path

def build_summary(counter, *, video_name: str = "", fps: float = 25.0, total_frames: int = 0) -> dict:
    """
    Build a summary dict from the counter's final state.
 
    Args:
        counter:      VehicleCounter instance after processing.
        video_name:   Name of the source video file.
        fps:          Frames per second of the video.
        total_frames: Total number of frames processed.
 
    Returns:
        Dict with totals, per-class counts, and video metadata.
    """

    duration_s = total_frames / fps if fps else 0

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "video":        video_name,
        "total_frames": total_frames,
        "duration_s":   round(duration_s, 2),
        "fps":          fps,
        **counter.summer(),
    }

    if duration_s > 0 :
        summary["flow_rate_per_min"] = round(summary["total"] / (duration_s / 60), 2)
    
    return summary

def save_summary_txt(summary: dict, output_path: str | Path) -> Path:
    """Write a human-readable plain-text traffic report."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    VEHICLE_CLASSES = ["car", "truck", "bus", "motorcycle"]
    
    lines = [
        "=" * 52,
        "  TRAFFIC FLOW ANALYSIS REPORT",
        "=" * 52,
        f"  Generated : {summary['generated_at']}",
        f"  Video     : {summary.get('video', 'N/A')}",
        f"  Duration  : {summary['duration_s']} s"
        f"  ({summary['total_frames']} frames @ {summary['fps']} fps)",
        "-" * 52,
        f"  Total crossings : {summary['total']}",
        f"    ↓ IN           : {summary['total_in']}",
        f"    ↑ OUT          : {summary['total_out']}",
    ]

    if "flow_rate_per_min" in summary:
        lines.append(f"  Flow rate       : {summary['flow_rate_per_min']} veh/min")
    
    lines += ["-" * 52, "  By class:", ""]

    for cls in VEHICLE_CLASSES:
        in_count  = summary.get(f"{cls}_in",  0)
        out_count = summary.get(f"{cls}_out", 0)
        if in_count or out_count:
            total_cls = in_count + out_count
            lines.append(
                f"  {cls:<12}  total={total_cls:>4}   in={in_count:>4}   out={out_count:>4}"
            )

    lines += ["", "=" * 52]

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[report] Text report → {output_path}")
    return output_path
