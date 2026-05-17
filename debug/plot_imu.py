"""IMU raw data: gyro and accel traces with per-axis noise stats."""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.euroc import EuRoCSequence

MAV0 = Path(__file__).parent.parent / "vicon_room1/V1_01_easy/mav0"
seq = EuRoCSequence(MAV0)

t = (seq.imu.timestamps_ns - seq.imu.timestamps_ns[0]) / 1e9
gyro = seq.imu.angular_velocity
accel = seq.imu.linear_acceleration
labels = ["x", "y", "z"]

fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

for i, lbl in enumerate(labels):
    axes[0].plot(t, gyro[:, i], lw=0.6, label=lbl)
    axes[1].plot(t, accel[:, i], lw=0.6, label=lbl)

# noise stats in legend
for i, lbl in enumerate(labels):
    std_g = np.std(gyro[:, i])
    std_a = np.std(accel[:, i])
    axes[0].get_lines()[i].set_label(f"{lbl}  σ={std_g:.4f} rad/s")
    axes[1].get_lines()[i].set_label(f"{lbl}  σ={std_a:.4f} m/s²")

axes[0].set_ylabel("Angular velocity [rad/s]")
axes[1].set_ylabel("Linear acceleration [m/s²]")
axes[1].set_xlabel("Time [s]")
for ax in axes:
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, lw=0.3)

plt.suptitle("IMU raw data — V1_01_easy")
plt.tight_layout()
plt.savefig("debug_imu.png", dpi=150)
print("Saved debug_imu.png")
