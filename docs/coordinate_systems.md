# Coordinate systems and transformation conventions

These conventions must be checked at every data import/export boundary.

## Extrinsic transform

Use `T_camera_lidar` (equivalently `T_C_L`) for a 4x4 rigid transform:

```text
p_camera = R_camera_lidar @ p_lidar + t_camera_lidar
T_camera_lidar = [[R_camera_lidar, t_camera_lidar], [0, 0, 0, 1]]
```

Translation is in meters. R is proper orthonormal, det(R)=+1. The inverse `T_lidar_camera` must not be confused with `T_camera_lidar`.

**Accumulated cloud caveat:** a GLIM-accumulated cloud may live in a `lidar_reference` or local map frame rather than the live LiDAR sensor frame. Its `frame_id` must match the *source frame* of the transform applied to it. A sensor-to-camera transform cannot directly project map-frame points without a map-to-sensor transform.

## Camera model

- Camera image coordinates: u right, v down (OpenCV convention).
- Camera 3D coordinates: x right, y down, z forward.
- `K` uses pixels, `distortion` follows a documented OpenCV camera model.
- Depth maps are defined as **camera Z in meters**, not Euclidean ray length.
- Undistortion must be applied consistently when backprojecting pixels.

For a normalized undistorted ray `r = [x_n, y_n, 1]`, a camera-Z depth `z` gives `p_C = z*r`.

## Plane representation

Store a unit normal `n` and signed offset `d` satisfying:

```text
n.T @ p + d = 0
```

When transforming an LiDAR-frame plane to camera coordinates:

```text
n_C = R_C_L @ n_L
d_C = d_L - n_C.T @ t_C_L
```

Normals are sign-ambiguous: `(n,d)` and `(-n,-d)` describe the same plane. Sign alignment must apply to both components.

## Trajectory and timestamps

- Use explicit timestamps (prefer `int64` nanoseconds) with known clock domain.
- Define pose directions (e.g. `T_world_lidar`) and reference timestamp before applying deskew.
- Monocular SfM translation has unknown scale until alignment.
- Hand–Eye `AX=XB` motion convention must explicitly state which of A/B is metric and how scale is applied.

Do not infer the convention from filename alone.
