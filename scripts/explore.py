import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from scipy.linalg import svd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.euroc import EuRoCSequence

''' Helper function definitions '''
def skew(v):
    return np.array([[ 0,    -v[2],  v[1]],
                     [ v[2],  0,    -v[0]],
                     [-v[1],  v[0],  0   ]])

def rodrigues(v):
    theta = np.linalg.norm(v)
    if theta < 1e-10:
        return np.eye(3) + skew(v)
    axis = v / theta
    K = skew(axis)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

def vicon_pose_at_index(vicon_data, idx):
    pos = vicon_data.ds_position[idx, :]
    qua = vicon_data.ds_orientation[idx, :]
    quasp = qua[[1, 2, 3, 0]]

    R = Rotation.from_quat(quasp).as_matrix()

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T

def imu_pose(imu_data_R, imu_data_P):
    pos = imu_data_P
    R = rodrigues(imu_data_R)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T

def relative_motion(T_a, T_b):
    return np.linalg.inv(T_a) @ T_b


def invert_transform(T):
    ''' Note that you can just do np.linalg.inv(T), but 
    this saves time since a closed-form inverse exists'''
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3]  = -R.T @ t
    return T_inv


def hand_eye_residual(T_cam_imu, dT_imu, dT_cam):
    # LHS of constraint
    left = T_cam_imu @ dT_imu

    # RHS of constraint
    right = dT_cam @ T_cam_imu

    # Error transform: how much do they disagree?
    E = left @ invert_transform(right)

    # Decompose E into a 6-vector
    R_err = E[:3, :3]
    t_err = E[:3, 3]

    # Rotation as axis-angle (3-vector)
    rot_vec = Rotation.from_matrix(R_err).as_rotvec()

    return np.concatenate([rot_vec, t_err])

def unpack_params(params):
    """
    Convert the optimizer's flat 7-vector into geometric forms

    params layout:
    [0:3] rotation vector (axis-angle) for T_marker_imu
    [3:6] translation for T_marker_imu
    [6]   time offset t_d (seconds)

    """
    rotvec = params[0:3]
    trans  = params[3:6]
    t_d    = params[6]     

    # Build the 4x4 transform
    R = Rotation.from_rotvec(rotvec).as_matrix()
    T_cam_imu = np.eye(4)
    T_cam_imu[:3, :3] = R
    T_cam_imu[:3, 3] = trans

    return T_cam_imu, t_d



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
debug_plot = 0
if debug_plot:
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

    # --- GT Plot ---
    t = (seq.groundtruth.timestamps_ns - seq.groundtruth.timestamps_ns[0]) / 1e9
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    for i, label in enumerate(["x", "y", "z"]):
        axes[0].plot(t, seq.groundtruth.position[:, i], label=label)

    axes[0].set_ylabel("Position [m]")
    axes[0].set_xlabel("Time [s]")

    for i, label in enumerate(["w", "x", "y", "z"]):
        axes[1].plot(t, seq.groundtruth.orientation[:, i], label=label)

    axes[1].set_ylabel("Quaternion [rad]")
    axes[1].set_xlabel("Time [s]")


    for ax in axes:
        ax.legend(loc="upper right")
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig("GT_plot.png", dpi=150)
    print("\nSaved GT_plot.png")




# --- Interpolate ground truth ---
seq.groundtruth.offset = seq.groundtruth.timestamps_ns[0]
seq.groundtruth.ds_timestamps_ns = seq.groundtruth.timestamps_ns[::10] # downsample from 100 Hz to 10 Hz
seq.groundtruth.ds_orientation = seq.groundtruth.orientation[::10,:]
seq.groundtruth.ds_position = seq.groundtruth.position[::10,:]

# --- Integrate IMU between each downsampled GT value, set up residual function ---
def residuals(params, seq):
    
    # Optimizer parameters as Rotation matrix f
    R_cam_imu = Rotation.from_rotvec(params[:3]).as_matrix()
    errors = []
    gt_rot_deg = []
    window_idx = []
    cnt = 0
    # Loop over each time window (which is the 10 Hz rate)
    # Note: IMU timestamp starts later than groundtruth, so first 10-15 windows from GT not used
    for i in range(len(seq.groundtruth.ds_timestamps_ns)-1):
        
        # Mask samples so they're sliced from the respective time window
        int_indices = np.where((seq.imu.timestamps_ns >= seq.groundtruth.ds_timestamps_ns[i]) & (seq.imu.timestamps_ns <= seq.groundtruth.ds_timestamps_ns[i+1]))
        int_angular_velocity = seq.imu.angular_velocity[int_indices]
        int_linear_accel = seq.imu.linear_acceleration[int_indices]
        int_timestamps_s = seq.imu.timestamps_ns[int_indices] / 1e9

        # Skip initial part of sequence where time offset exists between IMU and GT
        if len(int_timestamps_s) < 2:
            #print(i)
            continue
    

        dt = np.mean(np.diff(int_timestamps_s)) # simply averaging to a scalar to not deal with array mult with n-1 diff value lol
        delta_Rot = np.sum(int_angular_velocity * dt, axis=0) # This only works because small angle changes over sample (paraxial approx.)
        delta_Vel = np.cumsum(int_linear_accel * dt, axis=0)
        delta_Pos = np.sum(delta_Vel * dt, axis=0)
        delta_R_imu = rodrigues(delta_Rot)

        # Return integrated imu values over time slice as 4x4 homogeneous transform matrix
        # IGNORE UNTIL YOU START USING TRANSLATION
        #dT_imu = imu_pose(delta_Rot, delta_Pos)

        # Cam-marker coordinates
        T_i = vicon_pose_at_index(seq.groundtruth, i)
        T_i1 = vicon_pose_at_index(seq.groundtruth, i+1)
        dT_cam = relative_motion(T_i, T_i1)
        delta_R_cam = dT_cam[:3, :3]

        gt_rot_deg.append(np.degrees(np.linalg.norm(
        Rotation.from_matrix(delta_R_cam).as_rotvec())))

        #dT.append(dT_cam)

        # Debug
        #print(f"dT_imu translation norm: {np.linalg.norm(dT_imu[:3, 3]):.4f}")
        #print(f"dT_imu rotation angle: {np.linalg.norm(Rotation.from_matrix(dT_imu[:3,:3]).as_rotvec()):.4f}")
        #print(f"dT_marker translation norm: {np.linalg.norm(dT_cam[:3, 3]):.4f}")
        #print(f"dT_marker rotation angle: {np.linalg.norm(Rotation.from_matrix(dT_cam[:3,:3]).as_rotvec()):.4f}")
        
        window_idx.append(cnt)
        cnt =+ 1

        # Solving for T_imu_cam (formatted as 3-vec)
        R_err = R_cam_imu @ delta_R_imu @ (delta_R_cam @ R_cam_imu).T
        errors.append(Rotation.from_matrix(R_err).as_rotvec())

    return np.concatenate(errors)


x0 = np.zeros(3)
err = residuals(x0, seq)
print(f"Residual shape: {err.shape}")
print(f"Residual norm at identity guess: {np.linalg.norm(err)}")
print(f"First 3 residuals: {err[:3]}")

# -- Optimize --
result = least_squares(residuals, x0=x0, args=(seq,), method='lm', verbose=2)
res_rot = result.fun.reshape(-1, 3)
print(f"Recovered rotvec: {result.x[:3]}")
print(f"Recovered rotation (deg): {np.degrees(np.linalg.norm(result.x[:3])):.4f}")
print(f"Final cost: {result.cost}")

# -- Calculate covariance -- #

# Jacobian at the solution: each row is one residual, each column is one
# parameter. J[i, k] = ∂residual_i / ∂param_k, evaluated at result.x.
# scipy computed this numerically during optimization (finite differences)
# and stores the final one in result.jac. Shape: (n_residuals, n_params).
J = result.jac # shape: (n_residuals, n_params)

# Residual variance: sum of squared residuals divided by degrees of freedom
n_residuals = J.shape[0]
n_params = J.shape[1]

# result.fun is the residual vector evaluated at the final parameters —
# i.e., what residuals(result.x, ...) returned. Shape: (n_residuals,).
residuals_final = result.fun

# σ² (residual variance): how big the "leftover" errors are on average,
# after the optimizer did its best. The denominator (n_residuals - n_params)
# is the "degrees of freedom" correction — same idea as dividing by N-1
# instead of N when computing a sample variance. It accounts for the fact
# that fitting parameters absorbs some of the variance.
#
# Intuition: a model that fits the data perfectly (σ² → 0) implies very
# precise parameter estimates. A model with large residuals (σ² big)
# implies looser estimates.
sigma_sq = np.sum(residuals_final**2) / (n_residuals - n_params)

# Covariance via pseudoinverse (more numerically stable than direct inv)
# SVD decomposes J = U · diag(s) · Vᵀ
#   U:  (n_residuals, n_params) — orthonormal columns
#   s:  (n_params,)             — singular values, sorted descending
#   Vt: (n_params, n_params)    — orthonormal rows (Vᵀ)
#
# full_matrices=False uses the "thin" SVD: faster and gives shapes that
# match our use case (we don't need the full n_residuals × n_residuals U).
U, s, Vt = svd(J, full_matrices=False)

# Treat signular values as effectively zero as they represent directions in parameter space 
# where data is uninformative
threshold = np.finfo(float).eps * max(J.shape) * s[0]

# Invert the singular values, but set inversions to zero for any
# singular value below threshold. This is what makes this a *pseudoinverse*
# rather than a true inverse — it gracefully handles rank deficiency.
# For well-constrained directions: s_inv[k] = 1/s[k] (normal inversion).
# For near-zero directions: s_inv[k] = 0 (don't blow up to infinity).
s_inv = np.where(s > threshold, 1/s, 0)

# Reconstruct (Jᵀ J)⁻¹ from the SVD and scale by σ² to get the covariance.
#
# The math: J = U diag(s) Vᵀ implies Jᵀ J = V diag(s²) Vᵀ, so
# (Jᵀ J)⁻¹ = V diag(1/s²) Vᵀ. In code: Vᵀ has shape (n_params, n_params),
# so we transpose it back to V via Vt.T, then sandwich diag(s_inv²)
# between V and Vᵀ.
#
# Final shape of cov: (n_params, n_params).
#   Diagonal entries: variance of each parameter
#   Off-diagonals: covariance between pairs of parameters
cov = (Vt.T @ np.diag(s_inv**2) @ Vt) * sigma_sq

sigma_per_param = np.sqrt(np.diag(cov))
print(f"Recovered rotvec:        {result.x}")
print(f"1-sigma uncertainty:     {sigma_per_param}")
print(f"1-sigma (degrees):       {np.degrees(sigma_per_param)}")
print(f"Correlation matrix:")
print(cov / np.outer(sigma_per_param, sigma_per_param))
print(f"Condition number J^T*J: {s.min()/s.max()}")



# ----- Checking optimized result ----- #
# First get max rotation window by calculating gt_rot_deg
gt_rot_deg = []
window_idx = []
cnt = 0
for i in range(len(seq.groundtruth.ds_timestamps_ns)-1):
        
    # Mask samples so they're sliced from the respective time window
    int_indices = np.where((seq.imu.timestamps_ns >= seq.groundtruth.ds_timestamps_ns[i]) & (seq.imu.timestamps_ns <= seq.groundtruth.ds_timestamps_ns[i+1]))
    int_angular_velocity = seq.imu.angular_velocity[int_indices]
    int_linear_accel = seq.imu.linear_acceleration[int_indices]
    int_timestamps_s = seq.imu.timestamps_ns[int_indices] / 1e9

    # Skip initial part of sequence where time offset exists between IMU and GT
    if len(int_timestamps_s) < 2:
        #print(i)
        continue
    

    # Cam-marker coordinates
    T_i = vicon_pose_at_index(seq.groundtruth, i)
    T_i1 = vicon_pose_at_index(seq.groundtruth, i+1)
    dT_cam = relative_motion(T_i, T_i1)
    delta_R_cam = dT_cam[:3, :3]

    gt_rot_deg.append(np.degrees(np.linalg.norm(
    Rotation.from_matrix(delta_R_cam).as_rotvec())))
    window_idx.append(cnt)
    cnt += 1

R_recovered = Rotation.from_rotvec(result.x[:3]).as_matrix()

# Find the window with significant rotation
i = np.argmax(gt_rot_deg)  # window with most rotation
print(f"Window {i}: GT rotation = {gt_rot_deg[i]:.2f} deg")

# Apply the mask
mask = np.where((seq.imu.timestamps_ns >= seq.groundtruth.ds_timestamps_ns[i]) & (seq.imu.timestamps_ns <= seq.groundtruth.ds_timestamps_ns[i+1]))
omega = seq.imu.angular_velocity[mask]
int_timestamps_s = seq.imu.timestamps_ns[mask] / 1e9
dt = np.mean(np.diff(int_timestamps_s))
delta_Rot = np.sum(omega * dt, axis=0)
R_imu_delta = Rotation.from_rotvec(delta_Rot).as_matrix()

# Cam-marker coordinates
T_i = vicon_pose_at_index(seq.groundtruth, i)
T_i1 = vicon_pose_at_index(seq.groundtruth, i+1)
dT_cam = relative_motion(T_i, T_i1)
delta_R_cam = dT_cam[:3, :3]

# Check the hand-eye constraint with recovered R
# Constraint: R · R_imu = R_marker · R
LHS = R_recovered @ R_imu_delta
RHS = delta_R_cam @ R_recovered
err = LHS @ RHS.T  # should be identity if R is correct
err_angle_deg = np.degrees(np.linalg.norm(Rotation.from_matrix(err).as_rotvec()))
print(f"Constraint error with recovered R: {err_angle_deg:.4f} deg")

# Compare with identity
R_id = np.eye(3)
LHS_id = R_id @ R_imu_delta
RHS_id = delta_R_cam @ R_id
err_id = LHS_id @ RHS_id.T
err_id_angle_deg = np.degrees(np.linalg.norm(Rotation.from_matrix(err_id).as_rotvec()))
print(f"Constraint error with identity:    {err_id_angle_deg:.4f} deg")

# Find a window with rotation primarily around one Vicon axis
# Compute axis-angle for each window's GT rotation
gt_axes = []
for i in range(len(gt_rot_deg)):
    if gt_rot_deg[i] > 3:  # significant rotation
        T_i  = vicon_pose_at_index(seq.groundtruth, i)
        T_i1 = vicon_pose_at_index(seq.groundtruth, i+1)
        dT = relative_motion(T_i, T_i1)
        rotvec = Rotation.from_matrix(dT[:3, :3]).as_rotvec()
        axis = rotvec / np.linalg.norm(rotvec)
        gt_axes.append((i, axis, np.linalg.norm(rotvec)))

# Find one where rotation is primarily about a single axis
for i, axis, angle in gt_axes[:20]:
    dominant = np.argmax(np.abs(axis))
    if np.abs(axis[dominant]) > 0.9:  # >90% on one axis
        print(f"Window {i}: GT axis ~{['x','y','z'][dominant]} ({axis}), angle {np.degrees(angle):.2f}°")
        
        # Compute IMU axis for same window
        mask = (seq.imu.timestamps_ns >= seq.groundtruth.ds_timestamps_ns[i]) & \
               (seq.imu.timestamps_ns <= seq.groundtruth.ds_timestamps_ns[i+1])
        omega = seq.imu.angular_velocity[mask]
        ts = seq.imu.timestamps_ns[mask] / 1e9
        dt = np.mean(np.diff(ts))
        delta_Rot = np.sum(omega * dt, axis=0)
        imu_axis = delta_Rot / np.linalg.norm(delta_Rot)
        imu_dominant = np.argmax(np.abs(imu_axis))
        print(f"           IMU axis ~{['x','y','z'][imu_dominant]} ({imu_axis})")
        print()
        if i > gt_axes[15][0]:  # show a few
            break



# -- Result Plots -- #
res_norms = np.linalg.norm(res_rot, axis=1)

# Make sure gt_rot_deg is aligned to the same windows as res_norms
# (drop any skipped windows from both, or use np.nan masking)
gt_rot_deg = np.array(gt_rot_deg)

# Scatter with color encoding rotation magnitude
fig, ax = plt.subplots(figsize=(10, 5))
sc = ax.scatter(
    np.arange(len(res_norms)),
    res_norms,
    c=gt_rot_deg,
    cmap='viridis',  # or 'plasma', 'magma' — perceptually uniform
    s=15,
    alpha=0.7,
)

cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('GT rotation (deg)')

ax.set_xlabel('Window index')
ax.set_ylabel('Residual norm (rad)')
ax.set_title('Per-window residual after convergence, colored by rotation magnitude')

plt.tight_layout()
plt.show()






# -- Ground Truth Check -- #

# From YAML file
T_BS_vicon = np.array([
    [ 0.33638, -0.01749,  0.94156],
    [-0.02078, -0.99972, -0.01114],
    [ 0.94150, -0.01582, -0.33665]
])

# YAML gives T_body_vicon. We want T_marker_imu = T_vicon_body @ T_body_imu = T_vicon_body
R_marker_imu_yaml = T_BS_vicon.T  # inverse of a rotation is its transpose

rotvec_yaml = Rotation.from_matrix(R_marker_imu_yaml).as_rotvec()
rotvec_recovered = np.array([-2.55373041, -0.02244341, -1.79270892])

print(f"YAML rotvec:      {rotvec_yaml}")
print(f"Recovered rotvec: {rotvec_recovered}")
print(f"YAML angle:       {np.degrees(np.linalg.norm(rotvec_yaml)):.2f}°")
print(f"Recovered angle:  {np.degrees(np.linalg.norm(rotvec_recovered)):.2f}°")

# Direct comparison: error between the two rotations
R_recovered = Rotation.from_rotvec(rotvec_recovered).as_matrix()
R_diff = R_marker_imu_yaml @ R_recovered.T
err_deg = np.degrees(np.linalg.norm(Rotation.from_matrix(R_diff).as_rotvec()))
print(f"Difference between recovered and YAML: {err_deg:.3f}°")