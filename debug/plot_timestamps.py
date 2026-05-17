"""Sensor timeline: shows when each stream starts/ends and the overlap window."""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.euroc import EuRoCSequence

MAV0 = Path(__file__).parent.parent / "vicon_room1/V1_01_easy/mav0"
seq = EuRoCSequence(MAV0)

streams = {
    "IMU":         seq.imu.timestamps_ns,
    "Cam0":        seq.cam0.timestamps_ns,
    "Cam1":        seq.cam1.timestamps_ns,
    "Ground truth": seq.groundtruth.timestamps_ns,
}

# common reference: earliest timestamp across all streams
t_ref = min(ts[0] for ts in streams.values())
streams_s = {k: (v - t_ref) / 1e9 for k, v in streams.items()}

# overlap window
t_start = max(v[0] for v in streams_s.values())
t_end   = min(v[-1] for v in streams_s.values())

fig, ax = plt.subplots(figsize=(12, 3))
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

for i, (name, ts) in enumerate(streams_s.items()):
    ax.barh(i, ts[-1] - ts[0], left=ts[0], height=0.5,
            color=colors[i], alpha=0.8, label=name)
    ax.text(ts[0], i + 0.32, f"{len(ts)} samples", fontsize=7, va="bottom")

# overlap shading
ax.axvspan(t_start, t_end, color="green", alpha=0.1, label=f"Overlap  {t_end - t_start:.2f} s")
ax.axvline(t_start, color="green", lw=1, ls="--")
ax.axvline(t_end,   color="green", lw=1, ls="--")

ax.set_yticks(range(len(streams)))
ax.set_yticklabels(streams_s.keys())
ax.set_xlabel("Time [s] from earliest timestamp")
ax.legend(loc="lower right", fontsize=8)
ax.grid(axis="x", lw=0.3)
plt.suptitle("Sensor stream timeline — V1_01_easy")
plt.tight_layout()
plt.savefig("debug_timestamps.png", dpi=150)
print(f"Overlap window: {t_start:.3f} s → {t_end:.3f} s  ({t_end - t_start:.2f} s)")
print("Saved debug_timestamps.png")
