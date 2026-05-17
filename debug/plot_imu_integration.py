"""Compare IMU-integrated rotation angles vs ground truth — shows gyro drift."""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.euroc import EuRoCSequence


def skew(v):
    return np.array([[ 0,    -v[2],  v[1]],
                     [ v[2],  0,    -v[0]],
                     [-v[1],  v[0],  0   ]])


def rodrigues(v):
    theta = np.linalg.norm(v)
    if theta < 1e-10:
        return np.eye(3) + skew(v)
    K = skew(v / theta)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def quat_to_rot(q):
    """q = [qw, qx, qy, qz]"""
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
    ])


def rot_to_euler(R):
    """ZYX Euler angles [roll, pitch, yaw] in degrees."""
    pitch = np.arcsin(-R[2, 0])
    roll  = np.arctan2(R[2, 1], R[2, 2])
    yaw   = np.arctan2(R[1, 0], R[0, 0])
    return np.degrees([roll, pitch, yaw])


MAV0 = Path(__file__).parent.parent / "vicon_room1/V1_01_easy/mav0"
seq = EuRoCSequence(MAV0)

# --- integrate IMU gyro ---
ts  = seq.imu.timestamps_ns
gyro = seq.imu.angular_velocity
dt_all = np.diff(ts) / 1e9

R = np.eye(3)
imu_euler = [rot_to_euler(R)]
for i in range(len(dt_all)):
    R = R @ rodrigues(gyro[i] * dt_all[i])
    imu_euler.append(rot_to_euler(R))

imu_euler = np.array(imu_euler)
t_imu = (ts - ts[0]) / 1e9

# --- ground truth euler angles ---
gt_euler = np.array([rot_to_euler(quat_to_rot(q)) for q in seq.groundtruth.orientation])
t_gt = (seq.groundtruth.timestamps_ns - ts[0]) / 1e9

# --- plot ---
fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
angle_labels = ["Roll [deg]", "Pitch [deg]", "Yaw [deg]"]

for i, lbl in enumerate(angle_labels):
    axes[i].plot(t_imu, imu_euler[:, i], lw=0.7, label="IMU integrated", color="#4C72B0")
    axes[i].plot(t_gt,  gt_euler[:, i],  lw=0.7, label="Ground truth",   color="#DD8452")
    axes[i].set_ylabel(lbl)
    axes[i].legend(loc="upper right", fontsize=8)
    axes[i].grid(True, lw=0.3)

axes[-1].set_xlabel("Time [s]")
plt.suptitle("IMU integrated rotation vs ground truth — V1_01_easy")
plt.tight_layout()
plt.savefig("debug_imu_integration.png", dpi=150)
print("Saved debug_imu_integration.png")
