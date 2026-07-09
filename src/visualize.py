"""Plots and dashboard visualizations."""
 
import matplotlib
matplotlib.use("agg")   # Save plots without opening a window

import pandas as pd 
from pathlib import Path
import matplotlib.pyplot as plt

def save_dashboard(summary: dict, crossings: list[dict], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(crossings) if crossings else pd.DataFrame(columns=["frame", "class", "direction"])
    fps = summary.get("fps", 25.0)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Traffic Flow — {summary.get('video', '')}", fontsize=13, fontweight="bold")

    # ── 1. Bar chart: IN vs OUT per class ────────────────────────────────
    ax = axes[0][0]
    if not df.empty:
        classes = sorted(df["class"].unique())
        in_vals = [len(df[(df["class"]==c) & (df["direction"] == "in")] ) for c in classes]
        out_vals  = [len(df[(df["class"]==c) & (df["direction"]=="out")]) for c in classes]
        x = range(len(classes))
        ax.bar([i - 0.2 for i in x], in_vals,  0.4, label="IN",  color="#2ecc71")
        ax.bar([i + 0.2 for i in x], out_vals, 0.4, label="OUT", color="#e74c3c")
        ax.set_xticks(list(x))
        ax.set_xticklabels([c.capitalize() for c in classes])
        ax.legend()
    ax.set_title("Vehicles by Class")
    ax.set_ylabel("Count")

# ── 2. Timeline: crossings per minute ────────────────────────────────
    ax = axes[0][1]
    if not df.empty:
        df["minute"] = (df["frame"] / fps / 60).astype(int)
        mins = range(df["minute"].max() + 1)
        in_m = df[df["direction"]=="in"].groupby("minute").size().reindex(mins, fill_value=0)
        out_m = df[df["direction"]=="out"].groupby("minute").size().reindex(mins, fill_value=0)
        ax.plot(list(mins), in_m.values, color="#2ecc71", marker="o", label="IN")
        ax.plot(list(mins), out_m.values, color="#e74c3c", marker="o", label="OUT")
        ax.legend()
    ax.set_title("Crossings per Minute")
    ax.set_xlabel("Minute")
    ax.set_ylabel("Count")
 
    # ── 3. Donut: class share ─────────────────────────────────────────────
    ax = axes[1][0]
    if not df.empty:
        counts = df["class"].value_counts()
        ax.pie(counts.values, labels=[c.capitalize() for c in counts.index],
               autopct='%1.0f%%', startangle=90,
               wedgeprops={"width": 0.5})   
    ax.set_title("Class Distribution")
 
    # ── 4. Stats panel ────────────────────────────────────────────────────
    ax = axes[1][1]
    ax.axis("off")
    ax.set_title("Summary")
    lines = [
        ("Total",     summary.get("total", 0)),
        ("IN",        summary.get("total_in", 0)),
        ("OUT",       summary.get("total_out", 0)),
        ("Duration",  f"{summary.get('duration_s', 0):.0f} s"),
        ("Flow rate", f"{summary.get('flow_rate_per_min', 0):.1f} veh/min"),
        ("Frames",    summary.get("total_frames", 0)),
    ]

    for i, (label, value) in enumerate(lines):
        y = 0.85 - i * 0.13
        ax.text(0.1,  y, label, transform=ax.transAxes, fontsize=10, color="#555")
        ax.text(0.9,  y, str(value), transform=ax.transAxes, fontsize=10, ha="right", fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] Dashboard → {output_path}")
 




