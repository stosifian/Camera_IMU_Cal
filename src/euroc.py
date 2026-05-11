from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml


@dataclass
class ImuParams:
    T_BS: np.ndarray            # (4,4) body-to-sensor transform
    rate_hz: float
    gyro_noise_density: float   # rad/s/sqrt(Hz)
    gyro_random_walk: float     # rad/s^2/sqrt(Hz)
    accel_noise_density: float  # m/s^2/sqrt(Hz)
    accel_random_walk: float    # m/s^3/sqrt(Hz)


@dataclass
class CameraParams:
    T_BS: np.ndarray                  # (4,4) body-to-sensor transform
    rate_hz: float
    resolution: tuple                 # (width, height)
    intrinsics: np.ndarray            # [fu, fv, cu, cv]
    distortion_model: str
    distortion_coefficients: np.ndarray

    @property
    def K(self) -> np.ndarray:
        """3x3 camera matrix."""
        fu, fv, cu, cv = self.intrinsics
        return np.array([[fu, 0, cu],
                         [0, fv, cv],
                         [0,  0,  1]], dtype=np.float64)


@dataclass
class ImuData:
    timestamps_ns: np.ndarray        # (N,) int64, nanoseconds
    angular_velocity: np.ndarray     # (N,3) rad/s  [x, y, z]
    linear_acceleration: np.ndarray  # (N,3) m/s^2  [x, y, z]

    def __len__(self) -> int:
        return len(self.timestamps_ns)


@dataclass
class CameraData:
    timestamps_ns: np.ndarray  # (M,) int64, nanoseconds
    image_paths: list           # length M, Path objects

    def load_image(self, idx: int, grayscale: bool = True) -> np.ndarray:
        flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        return cv2.imread(str(self.image_paths[idx]), flag)

    def __len__(self) -> int:
        return len(self.timestamps_ns)


class EuRoCSequence:
    """Loads one EuRoC mav0 sequence from disk."""

    def __init__(self, mav0_path):
        self.root = Path(mav0_path)
        self.imu = self._load_imu_data()
        self.cam0 = self._load_camera_data("cam0")
        self.cam1 = self._load_camera_data("cam1")
        self.imu_params = self._load_imu_params()
        self.cam0_params = self._load_camera_params("cam0")
        self.cam1_params = self._load_camera_params("cam1")

    # ------------------------------------------------------------------
    # Data loaders
    # ------------------------------------------------------------------

    def _load_imu_data(self) -> ImuData:
        df = pd.read_csv(
            self.root / "imu0" / "data.csv",
            comment="#", header=0,
            names=["ts", "wx", "wy", "wz", "ax", "ay", "az"],
        )
        return ImuData(
            timestamps_ns=df["ts"].to_numpy(dtype=np.int64),
            angular_velocity=df[["wx", "wy", "wz"]].to_numpy(dtype=np.float64),
            linear_acceleration=df[["ax", "ay", "az"]].to_numpy(dtype=np.float64),
        )

    def _load_camera_data(self, cam: str) -> CameraData:
        cam_dir = self.root / cam
        df = pd.read_csv(
            cam_dir / "data.csv",
            comment="#", header=0,
            names=["ts", "filename"],
        )
        paths = [cam_dir / "data" / fn for fn in df["filename"]]
        return CameraData(
            timestamps_ns=df["ts"].to_numpy(dtype=np.int64),
            image_paths=paths,
        )

    # ------------------------------------------------------------------
    # Param loaders
    # ------------------------------------------------------------------

    def _load_imu_params(self) -> ImuParams:
        y = self._read_yaml(self.root / "imu0" / "sensor.yaml")
        return ImuParams(
            T_BS=np.array(y["T_BS"]["data"]).reshape(4, 4),
            rate_hz=float(y["rate_hz"]),
            gyro_noise_density=float(y["gyroscope_noise_density"]),
            gyro_random_walk=float(y["gyroscope_random_walk"]),
            accel_noise_density=float(y["accelerometer_noise_density"]),
            accel_random_walk=float(y["accelerometer_random_walk"]),
        )

    def _load_camera_params(self, cam: str) -> CameraParams:
        y = self._read_yaml(self.root / cam / "sensor.yaml")
        return CameraParams(
            T_BS=np.array(y["T_BS"]["data"]).reshape(4, 4),
            rate_hz=float(y["rate_hz"]),
            resolution=tuple(y["resolution"]),
            intrinsics=np.array(y["intrinsics"], dtype=np.float64),
            distortion_model=y["distortion_model"],
            distortion_coefficients=np.array(y["distortion_coefficients"], dtype=np.float64),
        )

    @staticmethod
    def _read_yaml(path: Path) -> dict:
        with open(path) as f:
            return yaml.safe_load(f)
