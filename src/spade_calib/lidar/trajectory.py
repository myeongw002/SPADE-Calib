"""GLIM T_world_lidar poses with explicit timestamp and SE(3) conventions."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class GlimTrajectory:
    """Epoch seconds, position[m] and xyzw quaternion of T_world_lidar.

    Query times are evaluated relative to the first timestamp to preserve
    accuracy with large Unix epoch seconds. Extrapolation always raises.
    """
    timestamps: NDArray[np.float64]
    positions: NDArray[np.float64]
    quaternions_xyzw: NDArray[np.float64]

    def __post_init__(self):
        ts = np.asarray(self.timestamps, dtype=np.float64)
        p = np.asarray(self.positions, dtype=np.float64)
        q = np.asarray(self.quaternions_xyzw, dtype=np.float64)
        if ts.ndim != 1 or len(ts) < 2 or p.shape != (len(ts), 3) or q.shape != (len(ts), 4):
            raise ValueError('Invalid GLIM trajectory shapes (requires >=2 poses)')
        if not (np.isfinite(ts).all() and np.isfinite(p).all() and np.isfinite(q).all()):
            raise ValueError('Non-finite GLIM trajectory entry')
        if np.any(np.diff(ts) <= 0):
            raise ValueError('GLIM timestamps must increase strictly')
        if np.any(np.linalg.norm(q, axis=1) < 1e-12):
            raise ValueError('GLIM has zero-norm quaternions')
        object.__setattr__(self, 'timestamps', ts)
        object.__setattr__(self, 'positions', p)
        object.__setattr__(self, 'quaternions_xyzw', q / np.linalg.norm(q, axis=1)[:, None])

    @classmethod
    def from_array(cls, values: NDArray) -> 'GlimTrajectory':
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] < 8:
            raise ValueError('GLIM rows must contain timestamp x y z qx qy qz qw')
        return cls(arr[:, 0], arr[:, 1:4], arr[:, 4:8])

    def at(self, timestamps_seconds: NDArray | float) -> tuple[NDArray, NDArray]:
        """Return (rotation matrices, xyz) for each query time; no extrapolation."""
        from scipy.spatial.transform import Rotation, Slerp
        query = np.asarray(timestamps_seconds, dtype=np.float64)
        scalar = query.ndim == 0
        query = np.atleast_1d(query)
        tq = query - self.timestamps[0]
        return self.at_offset(tq[0] if scalar else tq)

    def at_offset(self, seconds_from_first_pose: NDArray | float) -> tuple[NDArray, NDArray]:
        """Interpolate using small-offset seconds; best for nanosecond stamps."""
        from scipy.spatial.transform import Rotation, Slerp
        tq = np.asarray(seconds_from_first_pose, dtype=np.float64)
        scalar = tq.ndim == 0
        tq = np.atleast_1d(tq)
        relative = self.timestamps - self.timestamps[0]
        if not np.isfinite(tq).all() or np.any(tq < relative[0]) or np.any(tq > relative[-1]):
            raise ValueError('GLIM pose query would extrapolate')
        r = Slerp(relative, Rotation.from_quat(self.quaternions_xyzw))(tq).as_matrix()
        p = np.stack([np.interp(tq, relative, self.positions[:, k]) for k in range(3)], axis=-1)
        return (r[0], p[0]) if scalar else (r, p)
