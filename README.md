# Camera–IMU Hand-Eye Calibration

Rotation-only hand-eye calibration between a Vicon marker body and an IMU, using the [EuRoC MAV dataset](https://rpg.ifi.uzh.ch/docs/IJRR17_Burri.pdf).

## Method

Ground-truth rotations are derived from the 100 Hz Vicon stream, downsampled to 10 Hz keyframes. IMU gyroscope measurements are integrated over each 100 ms window (small-angle approximation) to produce a paired rotation delta. A hand-eye constraint is formed per window and the 3-parameter marker-to-IMU rotation is recovered with `scipy.optimize.least_squares`. Covariance is estimated from the Jacobian SVD at the solution.

Translation calibration is out of scope for this version — per-window IMU translation requires initial velocity, which cannot be recovered from gyro/accel alone over short windows.

## Results

Calibration was run across three V1 sequences of increasing motion difficulty:

| Sequence | Angle (°) | Error vs YAML (°) | σx (°) | σy (°) | σz (°) | Final Cost |
|---|---|---|---|---|---|---|
| V1_01 easy     | 178.78 | 2.58 | 0.45 | 0.83 | 0.47 | 0.093 |
| V1_02 medium   | 179.59 | 2.58 | 0.29 | 0.47 | 0.31 | 0.053 |
| V1_03 difficult| 180.11 | 3.49 | 0.41 | 0.43 | 0.49 | 0.16  |

See `report.html` for figures and full discussion.

## Structure

```
scripts/explore.py   — main calibration script
src/euroc.py         — EuRoC dataset reader
debug/               — standalone diagnostic plot scripts
report.html          — self-contained report with embedded figures
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Place EuRoC sequences under `vicon_room1/` following the standard `mav0/` layout, then run:

```bash
python scripts/explore.py
```
