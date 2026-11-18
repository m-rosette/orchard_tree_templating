"""
depth_registration.py

Registers a depth image to the color camera of an RGB-D camera (pinhole model, no distortion). This module does
not import ROS.

The method is the same as librealsense align (rs2::align to color): the two corners (u -/+ 0.5, v -/+ 0.5) of each
depth pixel go through the depth intrinsics, the depth->color extrinsics and the color intrinsics. The depth value
fills each color pixel in that box. When two depth pixels fill the same color pixel, the nearest one stays.

Extrinsics convention: p_color = R @ p_depth + t (the same as rs2_extrinsics, depth_to_color).
"""

from typing import Dict, Optional, Tuple

import numpy as np
import yaml


def camera_matrix(info: Dict) -> np.ndarray:
    """Return the 3x3 K from a camera_info dict (key "k" or "K", 9 values in row order)."""
    k = info.get("k", info.get("K"))
    return np.asarray(k, dtype=np.float64).reshape(3, 3)


def check_no_distortion(info: Dict, tol: float = 1e-6) -> bool:
    """Return True if the camera_info has no distortion (this module does not model distortion)."""
    d = info.get("d", info.get("D")) or []
    return all(abs(float(x)) <= tol for x in d)


def rotation_from_rotvec(rv) -> np.ndarray:
    """Rodrigues formula: rotation vector (rad) -> 3x3 rotation matrix."""
    rv = np.asarray(rv, dtype=np.float64)
    th = float(np.linalg.norm(rv))
    if th < 1e-12:
        return np.eye(3)
    k = rv / th
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * kx + (1 - np.cos(th)) * kx @ kx


def register_depth_to_color(
    depth: np.ndarray,
    K_depth: np.ndarray,
    K_color: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    color_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Return the depth image registered to the color image (same dtype and units as depth, shape color_shape).

    depth        (H_d, W_d) depth image (for example uint16 mm). 0 = no data.
    K_depth      3x3 depth intrinsics
    K_color      3x3 color intrinsics
    R, t         depth->color extrinsics, p_color = R @ p_depth + t. t in the same units as depth.
    color_shape  (H_c, W_c)
    """
    H_c, W_c = int(color_shape[0]), int(color_shape[1])
    v, u = np.nonzero(depth > 0)
    z = depth[v, u].astype(np.float64)
    if z.size == 0:
        return np.zeros((H_c, W_c), dtype=depth.dtype)

    fx_d, fy_d, cx_d, cy_d = K_depth[0, 0], K_depth[1, 1], K_depth[0, 2], K_depth[1, 2]
    fx_c, fy_c, cx_c, cy_c = K_color[0, 0], K_color[1, 1], K_color[0, 2], K_color[1, 2]
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3)

    def project_corner(du: float):
        x = (u + du - cx_d) / fx_d * z
        y = (v + du - cy_d) / fy_d * z
        P = np.stack([x, y, z])                         # (3, N)
        Pc = R @ P + t[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            uc = fx_c * Pc[0] / Pc[2] + cx_c
            vc = fy_c * Pc[1] / Pc[2] + cy_c
        return uc, vc, Pc[2]

    u0f, v0f, zc0 = project_corner(-0.5)
    u1f, v1f, zc1 = project_corner(+0.5)
    ok = (zc0 > 0) & (zc1 > 0) & np.isfinite(u0f) & np.isfinite(u1f)
    # Pixel index = round(coordinate); the same rounding as librealsense align
    u0 = np.floor(u0f[ok] + 0.5).astype(np.int64)
    v0 = np.floor(v0f[ok] + 0.5).astype(np.int64)
    u1 = np.floor(u1f[ok] + 0.5).astype(np.int64)
    v1 = np.floor(v1f[ok] + 0.5).astype(np.int64)
    zval = z[ok]

    out = np.full(H_c * W_c, np.inf)
    span_u = int(np.max(u1 - u0)) + 1 if u0.size else 0
    span_v = int(np.max(v1 - v0)) + 1 if v0.size else 0
    for dv in range(max(span_v, 0)):
        for du in range(max(span_u, 0)):
            uu, vv = u0 + du, v0 + dv
            m = (uu <= u1) & (vv <= v1) & (uu >= 0) & (uu < W_c) & (vv >= 0) & (vv < H_c)
            np.minimum.at(out, vv[m] * W_c + uu[m], zval[m])

    out[~np.isfinite(out)] = 0
    return out.reshape(H_c, W_c).astype(depth.dtype)


def registration_error(
    depth: np.ndarray, aligned_ref: np.ndarray, K_depth, K_color, R, t, max_abs: float = 100.0
) -> Tuple[float, float, int]:
    """
    Compare register_depth_to_color(depth) with a reference aligned image (for example the camera driver output).
    Return (median |diff|, fraction of pixels with |diff| <= max_abs, number of pixels with data in both).
    """
    reg = register_depth_to_color(depth, K_depth, K_color, R, t, aligned_ref.shape[:2]).astype(np.float64)
    ref = aligned_ref.astype(np.float64)
    both = (reg > 0) & (ref > 0)
    n = int(both.sum())
    if n == 0:
        return float("inf"), 0.0, 0
    diff = np.abs(reg[both] - ref[both])
    return float(np.median(diff)), float(np.mean(diff <= max_abs)), n


def load_extrinsics_yaml(path: str) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Read a depth->color extrinsics yaml (written by calibrate_depth_to_color).
    Return (R 3x3, t (3,) in metres, the full dict).
    """
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
    """sensor_msgs/CameraInfo -> plain dict for yaml."""
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
