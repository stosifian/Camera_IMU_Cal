"""
Debug script for explore.py residual function.

Checks per window:
  - How many IMU samples are used
  - IMU integrated rotation angle vs ground truth rotation angle
  - IMU integrated translation norm vs ground truth translation norm
  - Hand-eye residual (rotation + translation parts) at identity guess
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.euroc import EuRoCSequence

# -----------------------------------------------------------------------
# Helpers (mirrors explore.py)
# -----------------------------------------------------------------------

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

def vicon_pose_at_index(vicon_data, idx):
    pos  = vicon_data.ds_position[idx, :]
    quat = vicon_data.ds_orientation[idx, :]
    R = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = pos
    return T

def imu_pose(imu_data_R, imu_data_P):
    T = np.eye(4)
    T[:3, :3] = rodrigues(imu_data_R)
    T[:3, 3]  = imu_data_P
    return T

def invert_transform(T):
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3]  = -R.T @ t
    return T_inv

def hand_eye_residual(T_cam_imu, dT_imu, dT_cam):
    E = (T_cam_imu @ dT_imu) @ invert_transform(dT_cam @ T_cam_imu)
    rot_vec = Rotation.from_matrix(E[:3, :3]).as_rotvec()
    return np.concatenate([rot_vec, E[:3, 3]])

# -----------------------------------------------------------------------
# Load + downsample (mirrors explore.py)
# -----------------------------------------------------------------------

MAV0 = Path(__file__).parent.parent / "vicon_room1/V1_01_easy/mav0"
seq  = EuRoCSequence(MAV0)

seq.groundtruth.ds_timestamps_ns = seq.groundtruth.timestamps_ns[::10]
seq.groundtruth.ds_orientation   = seq.groundtruth.orientation[::10, :]
seq.groundtruth.ds_position      = seq.groundtruth.position[::10, :]

T_cam_imu = np.eye(4)   # identity guess — same as x0 = np.zeros(7) in explore.py

# -----------------------------------------------------------------------
# Per-window diagnostics
# -----------------------------------------------------------------------

n_windows    = len(seq.groundtruth.ds_timestamps_ns) - 1
imu_counts   = []
imu_rot_deg  = []
gt_rot_deg   = []
imu_trans_m  = []
gt_trans_m   = []
res_rot      = []   # first 3 components of residual
res_trans    = []   # last 3 components
skipped      = []

# -----------------
# Timing plot
# ------------------


fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=False)

# groundtruth ds_position vs ds_timestamps
t_gt = (seq.groundtruth.ds_timestamps_ns - seq.groundtruth.ds_timestamps_ns[0]) / 1e9
for i, label in enumerate("xyz"):
    axes[0].plot(t_gt, seq.groundtruth.ds_position[:, i], label=label)
axes[0].set(title="groundtruth ds_position", xlabel="time (s)", ylabel="m")
axes[0].legend()

# imu timestamps — just show the intervals to verify regularity
t_imu = (seq.imu.timestamps_ns - seq.imu.timestamps_ns[0]) / 1e9
dt_imu = np.diff(t_imu) * 1e3  # ms
axes[1].plot(dt_imu)
axes[1].set(title="IMU timestamp intervals", xlabel="sample", ylabel="dt (ms)")

plt.tight_layout()
plt.savefig("debug_timeseries.png", dpi=120)
plt.show()







for i in range(n_windows):
    mask = ((seq.imu.timestamps_ns >= seq.groundtruth.ds_timestamps_ns[i]) &
            (seq.imu.timestamps_ns <= seq.groundtruth.ds_timestamps_ns[i + 1]))

    int_angular_velocity  = seq.imu.angular_velocity[mask]
    int_accel = seq.imu.linear_acceleration[mask]
    int_ts    = seq.imu.timestamps_ns[mask] / 1e9

    imu_counts.append(mask.sum())

    if mask.sum() < 2:
        skipped.append(i)
        imu_rot_deg.append(np.nan)
        gt_rot_deg.append(np.nan)
        imu_trans_m.append(np.nan)
        gt_trans_m.append(np.nan)
        res_rot.append([np.nan] * 3)
        res_trans.append([np.nan] * 3)
        continue



    # Get world frame
    T_i = vicon_pose_at_index(seq.groundtruth, i)
    R_world = T_i[:3, :3]
    R_start = R_world

    # DEBUG
    #world_z_in_marker = R_start.T @ np.array([0, 0, 1])
    #print(f"World up direction in marker frame: {world_z_in_marker}")

    # integrate angular velocity and rotate and apply for grav comp
    dt         = np.mean(np.diff(int_ts))
    delta_Rot  = np.sum(int_angular_velocity  * dt, axis=0)
    
    '''
    # DEBUG
    if i == 21:
        print(f"GT translation in window: {gt_trans_m[i-1]:.6f} m")
        print(f"GT rotation in window:    {gt_rot_deg[i-1]:.4f} deg")
        print(f"Gyro mean: {seq.imu.angular_velocity[mask].mean(axis=0)}")
        print(f"Gyro std:  {seq.imu.angular_velocity[mask].std(axis=0)}")
        print(f"Accel mean (IMU frame):    {int_accel.mean(axis=0)}")
        print(f"Accel magnitude:           {np.linalg.norm(int_accel.mean(axis=0)):.3f}")
        print(f"Accel rotated to 'world':  {(int_accel @ R_start.T).mean(axis=0)}")
        print(f"R_start:\n{R_start}")
    
    # DEBUG: Find the most stationary windows
    stationary_score = gt_trans_m + np.radians(gt_rot_deg)  # combined metric
    stationary_score = np.where(np.isnan(stationary_score), np.inf, stationary_score)
    best = np.argsort(stationary_score)[:10]

    print("Most stationary windows:")
    for i in best:
        print(f"  window {i:4d}: trans={gt_trans_m[i]:.5f} m, rot={gt_rot_deg[i]:.4f} deg")
    '''


    # world coordinate math, gravity correction
    g_imu = np.array([9.05, -0.04, -3.67])
    int_accel_corrected = int_accel - g_imu
    int_accel_world = int_accel_corrected @ R_start.T
    delta_Vel_world  = np.cumsum(int_accel_world * dt, axis=0)
    delta_Pos_world  = np.sum(delta_Vel_world * dt, axis=0)

    # Rotate translation back into IMU/marker frame at t_i
    delta_Pos = R_start.T @ delta_Pos_world

    

    dT_imu     = imu_pose(delta_Rot, delta_Pos)

    T_i   = vicon_pose_at_index(seq.groundtruth, i)
    T_i1  = vicon_pose_at_index(seq.groundtruth, i + 1)
    dT_gt = np.linalg.inv(T_i) @ T_i1

    imu_rot_deg.append(np.degrees(np.linalg.norm(delta_Rot)))
    gt_rot_deg.append(np.degrees(np.linalg.norm(
        Rotation.from_matrix(dT_gt[:3, :3]).as_rotvec())))

    imu_trans_m.append(np.linalg.norm(delta_Pos))
    gt_trans_m.append(np.linalg.norm(dT_gt[:3, 3]))

    err = hand_eye_residual(T_cam_imu, dT_imu, dT_gt)
    res_rot.append(err[:3])
    res_trans.append(err[3:])

imu_counts  = np.array(imu_counts)
imu_rot_deg = np.array(imu_rot_deg)
gt_rot_deg  = np.array(gt_rot_deg)
imu_trans_m = np.array(imu_trans_m)
gt_trans_m  = np.array(gt_trans_m)
res_rot     = np.array(res_rot)
res_trans   = np.array(res_trans)
window_idx  = np.arange(n_windows)

# -----------------------------------------------------------------------
# Summary printout
# -----------------------------------------------------------------------

print(f"Total windows : {n_windows}")
print(f"Skipped       : {len(skipped)}  (< 2 IMU samples)")
print(f"IMU samples/window — min: {imu_counts.min()}  max: {imu_counts.max()}  mean: {imu_counts.mean():.1f}")
print(f"Rotation error  norm — mean: {np.nanmean(np.linalg.norm(res_rot,   axis=1)):.4f} rad")
print(f"Translation error norm — mean: {np.nanmean(np.linalg.norm(res_trans, axis=1)):.4f} m")

# -----------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------

fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

# 1. IMU samples per window
axes[0].bar(window_idx, imu_counts, width=1.0, color="#4C72B0", alpha=0.8)
if skipped:
    axes[0].bar(skipped, [imu_counts.max()] * len(skipped),
                width=1.0, color="red", alpha=0.4, label="skipped")
    axes[0].legend(fontsize=8)
axes[0].set_ylabel("IMU samples")
axes[0].set_title("IMU samples per integration window")

# 2. Rotation angle comparison
axes[1].plot(window_idx, imu_rot_deg, lw=0.8, label="IMU integrated", color="#4C72B0")
axes[1].plot(window_idx, gt_rot_deg,  lw=0.8, label="Ground truth",   color="#DD8452")
axes[1].set_ylabel("Rotation [deg]")
axes[1].set_title("Per-window rotation angle: IMU vs ground truth")
axes[1].legend(fontsize=8)

# 3. Translation norm comparison
axes[2].plot(window_idx, imu_trans_m, lw=0.8, label="IMU integrated", color="#4C72B0")
axes[2].plot(window_idx, gt_trans_m,  lw=0.8, label="Ground truth",   color="#DD8452")
axes[2].set_ylabel("Translation [m]")
axes[2].set_title("Per-window translation norm: IMU vs ground truth")
axes[2].legend(fontsize=8)

# 4. Residual norms
rot_norms   = np.linalg.norm(res_rot,   axis=1)
trans_norms = np.linalg.norm(res_trans, axis=1)
axes[3].plot(window_idx, rot_norms,   lw=0.8, label="Rotation residual [rad]",  color="#4C72B0")
axes[3].plot(window_idx, trans_norms, lw=0.8, label="Translation residual [m]", color="#DD8452")
axes[3].set_ylabel("Residual norm")
axes[3].set_title("Hand-eye residual per window (identity guess)")
axes[3].set_xlabel("Window index")
axes[3].legend(fontsize=8)

for ax in axes:
    ax.grid(True, lw=0.3)

plt.suptitle("Residual function diagnostics — V1_01_easy")
plt.tight_layout()
plt.savefig("debug_residuals.png", dpi=150)
print("\nSaved debug_residuals.png")
