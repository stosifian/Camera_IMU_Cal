"""Ground truth trajectory: 3D position path and quaternion components over time."""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.euroc import EuRoCSequence

MAV0 = Path(__file__).parent.parent / "vicon_room1/V1_01_easy/mav0"
seq = EuRoCSequence(MAV0)

t  = (seq.groundtruth.timestamps_ns - seq.groundtruth.timestamps_ns[0]) / 1e9
pos = seq.groundtruth.position
ori = seq.groundtruth.orientation

fig = plt.figure(figsize=(14, 5))

# --- 3D trajectory ---
ax3d = fig.add_subplot(121, projection="3d")
ax3d.plot(pos[:, 0], pos[:, 1], pos[:, 2], lw=0.8)
ax3d.scatter(*pos[0],  color="green", s=40, zorder=5, label="start")
ax3d.scatter(*pos[-1], color="red",   s=40, zorder=5, label="end")
ax3d.set_xlabel("x [m]"); ax3d.set_ylabel("y [m]"); ax3d.set_zlabel("z [m]")
ax3d.set_title("3D position")
ax3d.legend(fontsize=8)

# --- quaternion over time ---
ax_q = fig.add_subplot(122)
for i, lbl in enumerate(["qw", "qx", "qy", "qz"]):
    ax_q.plot(t, ori[:, i], lw=0.7, label=lbl)
ax_q.set_xlabel("Time [s]")
ax_q.set_ylabel("Quaternion component")
ax_q.set_title("Orientation (quaternion)")
ax_q.legend(loc="upper right", fontsize=8)
ax_q.grid(True, lw=0.3)

plt.suptitle("Ground truth — V1_01_easy")
plt.tight_layout()
plt.savefig("debug_trajectory.png", dpi=150)
print("Saved debug_trajectory.png")
