"""
depth_registration.py

Registers a depth image to the color image of an RGB-D camera. No ROS imports.

Same method as librealsense align to color: the two corners (u -/+ 0.5, v -/+ 0.5) of each depth pixel go through
the depth intrinsics, the depth->color extrinsics and the color intrinsics (OpenCV undistortPoints / projectPoints).
The depth value fills each color pixel in that box. If two depth pixels fill one color pixel, the nearer one stays.

Extrinsics: p_color = R @ p_depth + t (as rs2_extrinsics depth_to_color).
"""

from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation


def camera_matrix(info: Dict) -> np.ndarray:
    """3x3 K of a camera_info dict (camera_info_to_dict)."""
    return np.asarray(info["k"], dtype=np.float64).reshape(3, 3)


def distortion(info: Dict) -> np.ndarray:
    """Distortion coefficients of a camera_info dict (plumb_bob order)."""
    return np.asarray(info.get("d") or [], dtype=np.float64)


def rotation_from_rotvec(rv) -> np.ndarray:
    """Rotation vector (rad) -> 3x3 rotation matrix."""
    return Rotation.from_rotvec(np.asarray(rv, dtype=np.float64)).as_matrix()


def register_depth_to_color(
    depth: np.ndarray,
    K_depth: np.ndarray,
    K_color: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    color_shape: Tuple[int, int],
    D_depth: Optional[np.ndarray] = None,
    D_color: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Depth image registered to the color image (dtype and units of depth, shape color_shape).

    depth           (H_d, W_d) depth (for example uint16 mm); 0 = no data
    K_depth/K_color 3x3 intrinsics; D_depth/D_color distortion coefficients (None: no distortion)
    R, t            depth->color extrinsics; t in the units of depth
    """
    H_c, W_c = int(color_shape[0]), int(color_shape[1])
    v, u = np.nonzero(depth > 0)
    z = depth[v, u].astype(np.float64)
    if z.size == 0:
        return np.zeros((H_c, W_c), dtype=depth.dtype)
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3)

    def project_corner(d: float):
        """Color pixel (u, v) and color-frame z of the pixel corner (u + d, v + d) of each depth pixel."""
        px = np.stack([u + d, v + d], axis=1).astype(np.float64).reshape(-1, 1, 2)
        ray = cv2.undistortPoints(px, K_depth, D_depth).reshape(-1, 2)
        P = np.column_stack([ray * z[:, None], z]) @ R.T + t
        uv, _ = cv2.projectPoints(P, np.zeros(3), np.zeros(3), K_color, D_color)
        return uv.reshape(-1, 2), P[:, 2]

    uv0, zc0 = project_corner(-0.5)
    uv1, zc1 = project_corner(+0.5)
    ok = (zc0 > 0) & (zc1 > 0) & np.isfinite(uv0).all(axis=1) & np.isfinite(uv1).all(axis=1)
    # Pixel index = round(coordinate), as librealsense align
    u0, v0 = np.floor(uv0[ok] + 0.5).astype(np.int64).T
    u1, v1 = np.floor(uv1[ok] + 0.5).astype(np.int64).T
    zval = z[ok]

    # Fill each box (inclusive) with the depth; keep the minimum (z-buffer)
    out = np.full(H_c * W_c, np.inf)
    for dv in range(int(np.max(v1 - v0, initial=-1)) + 1):
        for du in range(int(np.max(u1 - u0, initial=-1)) + 1):
            uu, vv = u0 + du, v0 + dv
            m = (uu <= u1) & (vv <= v1) & (uu >= 0) & (uu < W_c) & (vv >= 0) & (vv < H_c)
            np.minimum.at(out, vv[m] * W_c + uu[m], zval[m])
    out[~np.isfinite(out)] = 0
    return out.reshape(H_c, W_c).astype(depth.dtype)


def load_extrinsics_yaml(path: str) -> Tuple[np.ndarray, np.ndarray, dict]:
    """(R, t [m], data) of a calibrate_depth_to_color yaml."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    R = np.asarray(data["rotation"], dtype=np.float64).reshape(3, 3)
    t = np.asarray(data["translation_m"], dtype=np.float64).reshape(3)
    return R, t, data


def extrinsics_from_realsense_msg(msg) -> Tuple[np.ndarray, np.ndarray]:
    """realsense2_camera_msgs/Extrinsics -> (R, t). The rotation in the message is column-major."""
    R = np.asarray(msg.rotation, dtype=np.float64).reshape(3, 3).T
    t = np.asarray(msg.translation, dtype=np.float64).reshape(3)
    return R, t


def camera_info_to_dict(msg) -> Optional[Dict]:
    """sensor_msgs/CameraInfo -> dict for yaml (None -> None)."""
    if msg is None:
        return None
    return {
        "frame_id": msg.header.frame_id,
        "width": int(msg.width),
        "height": int(msg.height),
        "distortion_model": msg.distortion_model,
        "k": [float(x) for x in msg.k],
        "d": [float(x) for x in msg.d],
        "p": [float(x) for x in msg.p],
    }
