"""Quick sanity check: print dataset stats, plot IMU, show first camera frame."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.euroc import EuRoCSequence

MAV0 = Path(__file__).parent.parent / "vicon_room1/V1_01_easy/mav0"

seq = EuRoCSequence(MAV0)

# --- stats ---
t0 = min(seq.imu.timestamps_ns[0], seq.cam0.timestamps_ns[0])
t1 = max(seq.imu.timestamps_ns[-1], seq.cam0.timestamps_ns[-1])
duration = (t1 - t0) / 1e9

print(f"Duration : {duration:.2f} s")
print(f"IMU      : {len(seq.imu)} samples @ {seq.imu_params.rate_hz} Hz")
print(f"Cam0     : {len(seq.cam0)} frames  @ {seq.cam0_params.rate_hz} Hz")
print(f"Cam1     : {len(seq.cam1)} frames  @ {seq.cam1_params.rate_hz} Hz")
print(f"Cam0 K   :\n{seq.cam0_params.K}")
print(f"Cam0 dist: {seq.cam0_params.distortion_coefficients}")

# --- IMU plot ---
t = (seq.imu.timestamps_ns - seq.imu.timestamps_ns[0]) / 1e9
fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
for i, label in enumerate(["x", "y", "z"]):
    axes[0].plot(t, seq.imu.angular_velocity[:, i], label=label)
    axes[1].plot(t, seq.imu.linear_acceleration[:, i], label=label)
axes[0].set_ylabel("Angular vel [rad/s]")
axes[1].set_ylabel("Linear accel [m/s²]")
axes[1].set_xlabel("Time [s]")
for ax in axes:
    ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig("imu_plot.png", dpi=150)
print("\nSaved imu_plot.png")

# --- first camera frame ---
img = seq.cam0.load_image(0)
print(f"Frame 0  : shape={img.shape}, dtype={img.dtype}")
fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.imshow(img, cmap="gray")
ax2.set_title("cam0 frame 0")
ax2.axis("off")
plt.tight_layout()
plt.savefig("cam0_frame0.png", dpi=150)
print("Saved cam0_frame0.png")
