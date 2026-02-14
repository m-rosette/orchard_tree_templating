#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import utm
import yaml

# -----------------------------
# Optional dependency (Optuna)
# -----------------------------
try:
    import optuna
except Exception as e:
    optuna = None  # type: ignore[assignment]
    _OPTUNA_IMPORT_ERR = e
else:
    _OPTUNA_IMPORT_ERR = None

# ============================================================
# Shell helpers
# ============================================================

def run_cmd(
    cmd: List[str],
    *,
    check: bool = True,
    capture: bool = True,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    timeout_s: Optional[float] = None,
) -> subprocess.CompletedProcess:
    kwargs: Dict[str, Any] = {
        "text": True,
        "cwd": str(cwd) if cwd else None,
        "env": env,
    }
    if capture:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
    else:
        kwargs["stdout"] = None
        kwargs["stderr"] = None

    try:
        return subprocess.run(cmd, check=check, timeout=timeout_s, **kwargs)
    except subprocess.CalledProcessError as e:
        if capture:
            sys.stderr.write(f"\n[CMD FAIL] {' '.join(cmd)}\n")
            sys.stderr.write(f"stdout:\n{e.stdout}\n")
            sys.stderr.write(f"stderr:\n{e.stderr}\n")
        raise
    except subprocess.TimeoutExpired as e:
        sys.stderr.write(f"\n[CMD TIMEOUT] {' '.join(cmd)}\n")
        if hasattr(e, "stdout") and e.stdout:
            sys.stderr.write(f"stdout:\n{e.stdout}\n")
        if hasattr(e, "stderr") and e.stderr:
            sys.stderr.write(f"stderr:\n{e.stderr}\n")
        raise


def popen_cmd(
    cmd: List[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    log_path: Optional[Path] = None,
) -> subprocess.Popen:
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        f = open(log_path, "w", buffering=1)
        p = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            preexec_fn=os.setsid,
        )
        p._log_file = f  # type: ignore[attr-defined]
        return p

    return subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid,
    )


def terminate_process_group(p: subprocess.Popen, *, sig=signal.SIGINT, timeout_s: float = 10.0) -> None:
    if p.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(p.pid), sig)
    except Exception:
        pass

    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if p.poll() is not None:
            break
        time.sleep(0.1)

    if p.poll() is None:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception:
            pass

    f = getattr(p, "_log_file", None)
    if f is not None:
        try:
            f.close()
        except Exception:
            pass


def ros2_node_list(*, env: Optional[Dict[str, str]] = None) -> List[str]:
    p = run_cmd(["ros2", "node", "list"], check=True, capture=True, env=env)
    return [ln.strip() for ln in (p.stdout or "").splitlines() if ln.strip()]


def wait_for_node(node_name: str, timeout_s: float = 30.0, *, env: Optional[Dict[str, str]] = None) -> bool:
    node_name = node_name if node_name.startswith("/") else f"/{node_name}"
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            if node_name in ros2_node_list(env=env):
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def wait_for_service(service: str, timeout_s: float = 30.0, *, env: Optional[Dict[str, str]] = None) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            p = run_cmd(["ros2", "service", "list"], check=True, capture=True, env=env)
            svcs = [ln.strip() for ln in (p.stdout or "").splitlines() if ln.strip()]
            if service in svcs:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def ros2_param_set(node_name: str, param: str, value: Any, *, env: Optional[Dict[str, str]] = None) -> None:
    if isinstance(value, bool):
        v = "true" if value else "false"
    elif isinstance(value, (int, float)):
        if isinstance(value, float) and (not math.isfinite(value)):
            raise ValueError(f"Non-finite param value for {param}: {value}")
        v = f"{value}"
    else:
        v = str(value)

    node_arg = node_name if node_name.startswith("/") else f"/{node_name}"

    try:
        run_cmd(["ros2", "param", "set", node_arg, param, v], check=True, capture=True, env=env)
        return
    except Exception:
        run_cmd(["ros2", "param", "set", node_name.lstrip("/"), param, v], check=True, capture=True, env=env)


# ============================================================
# GT loading (class==0 + row_num)
# ============================================================

def _f(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def mean_latlon(position_estimates: Any) -> Tuple[Optional[float], Optional[float], int]:
    if not isinstance(position_estimates, list) or len(position_estimates) == 0:
        return None, None, 0
    lats: List[float] = []
    lons: List[float] = []
    for p in position_estimates:
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            continue
        lat = _f(p[0])
        lon = _f(p[1])
        if lat is None or lon is None:
            continue
        lats.append(lat)
        lons.append(lon)
    if not lats:
        return None, None, 0
    return float(np.mean(lats)), float(np.mean(lons)), len(lats)


def get_base_utm_from_tree(tree: dict) -> Tuple[float, float]:
    if "position_estimates" in tree:
        lat_m, lon_m, n_used = mean_latlon(tree.get("position_estimates"))
        if lat_m is not None and lon_m is not None and n_used > 0:
            e, n, _, _ = utm.from_latlon(lat_m, lon_m)
            return float(e), float(n)

    v = tree.get("position_estimate", None)
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        e = _f(v[0])
        n = _f(v[1])
        if e is not None and n is not None:
            return float(e), float(n)

    raise ValueError("Could not find usable GPS/UTM position in this entry.")


def load_ground_truth(gt_json_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with gt_json_path.open("r") as f:
        data = json.load(f)

    obj_nums: List[int] = []
    pts: List[np.ndarray] = []
    row_nums: List[int] = []

    for tree in data:
        cls = tree.get("class", None)
        if cls is None:
            continue
        try:
            if float(cls) != 0.0:
                continue
        except Exception:
            continue

        rn = tree.get("row_num", None)
        if rn is None:
            continue

        obj = int(tree.get("object_num", -1))
        e, n = get_base_utm_from_tree(tree)
        p = np.array([e, n], dtype=np.float64)

        adj = tree.get("position_adjustment", [0.0, 0.0])
        if isinstance(adj, (list, tuple)) and len(adj) >= 2:
            p[0] += float(_f(adj[0]) or 0.0)
            p[1] += float(_f(adj[1]) or 0.0)

        obj_nums.append(obj)
        pts.append(p)
        row_nums.append(int(rn))

    if not pts:
        raise RuntimeError("No GT objects found with class==0.0 and row_num present.")

    return np.asarray(obj_nums, dtype=int), np.vstack(pts), np.asarray(row_nums, dtype=int)


def get_gt_anchor(
    gt_obj_nums: np.ndarray,
    gt_pts: np.ndarray,
    gt_row_nums: np.ndarray,
    anchor_obj: int
) -> Tuple[np.ndarray, int]:
    idx = np.where(gt_obj_nums == int(anchor_obj))[0]
    if idx.size == 0:
        raise ValueError(
            f"GT anchor object_num {anchor_obj} not found AFTER filtering to class==0.0. "
            "Pick an anchor that is a tree."
        )
    j = int(idx[0])
    return gt_pts[j, :].copy(), int(gt_row_nums[j])


def pca_row_axis(points_EN: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if points_EN.shape[0] < 2:
        u = np.array([1.0, 0.0], dtype=np.float64)
        v = np.array([0.0, 1.0], dtype=np.float64)
        return u, v
    mu = points_EN.mean(axis=0)
    X = points_EN - mu
    C = (X.T @ X) / max(points_EN.shape[0] - 1, 1)
    _, V = np.linalg.eigh(C)
    u = V[:, 1]
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.array([-u[1], u[0]], dtype=np.float64)
    return u, v


def extract_gt_row_by_row_num(
    gt_obj_nums: np.ndarray,
    gt_pts: np.ndarray,
    gt_row_nums: np.ndarray,
    anchor_object_number: int,
) -> Tuple[np.ndarray, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gt_anchor, anchor_row_num = get_gt_anchor(gt_obj_nums, gt_pts, gt_row_nums, anchor_object_number)

    keep = (gt_row_nums == anchor_row_num)
    row_pts = gt_pts[keep]
    row_obj = gt_obj_nums[keep]

    u, v = pca_row_axis(row_pts)

    s = (row_pts - gt_anchor[None, :]) @ u
    order = np.argsort(s)
    row_pts = row_pts[order]
    row_obj = row_obj[order]

    return gt_anchor, anchor_row_num, row_obj, row_pts, u, v


# ============================================================
# Local YAML projection (anchored)
# ============================================================

def wrap_angle(theta: float) -> float:
    return (theta + math.pi) % (2.0 * math.pi) - math.pi


def rot2(yaw_rad: float) -> np.ndarray:
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def parse_initial_gps_utm(local_data: Dict[str, Any]) -> Tuple[float, float]:
    gps0 = local_data.get("initial_gps_fix", None)
    if not isinstance(gps0, (list, tuple)) or len(gps0) < 2:
        raise ValueError("Local YAML missing initial_gps_fix.")
    e0, n0, _, _ = utm.from_latlon(float(gps0[0]), float(gps0[1]))
    return float(e0), float(n0)


def parse_final_gps_utm(local_data: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    fgps = local_data.get("final_gps_fix", None)
    if isinstance(fgps, (list, tuple)) and len(fgps) >= 2:
        try:
            e, n, _, _ = utm.from_latlon(float(fgps[0]), float(fgps[1]))
            return float(e), float(n)
        except Exception:
            return None
    return None


def yaw_from_gps_baseline(e0: float, n0: float, final_utm: Optional[Tuple[float, float]]) -> float:
    if final_utm is None:
        raise ValueError("Need final_gps_fix for GPS yaw.")
    dE = float(final_utm[0] - e0)
    dN = float(final_utm[1] - n0)
    if abs(dE) < 1e-9 and abs(dN) < 1e-9:
        raise ValueError("final_gps_fix too close to initial for yaw.")
    return wrap_angle(float(math.atan2(dN, dE)))


def select_yaw(
    yaw_source: str,
    manual_yaw_rad: Optional[float],
    ignore_yaml_yaw: bool,
    local_data: Dict[str, Any],
    e0: float,
    n0: float,
    final_utm: Optional[Tuple[float, float]],
) -> float:
    yaw_yaml = local_data.get("initial_yaw_correction_rad", None)

    if manual_yaw_rad is not None:
        return float(manual_yaw_rad)

    if yaw_source == "yaml":
        if yaw_yaml is None:
            raise ValueError("yaw_source=yaml but initial_yaw_correction_rad missing.")
        return float(yaw_yaml)

    if yaw_source == "gps":
        return yaw_from_gps_baseline(e0, n0, final_utm)

    if yaw_source == "gps_plus_yaml":
        if yaw_yaml is None:
            raise ValueError("yaw_source=gps_plus_yaml but initial_yaw_correction_rad missing.")
        return wrap_angle(yaw_from_gps_baseline(e0, n0, final_utm) - float(yaw_yaml))

    if not ignore_yaml_yaw and yaw_yaml is not None:
        return float(yaw_yaml)
    return yaw_from_gps_baseline(e0, n0, final_utm)


def parse_initial_odom_xy(local_data: Dict[str, Any]) -> Tuple[float, float]:
    odom0 = local_data.get("initial_odom_correction", [0.0, 0.0, 0.0])
    if not isinstance(odom0, (list, tuple)) or len(odom0) < 2:
        return 0.0, 0.0
    return float(odom0[0]), float(odom0[1])


def local_xy_to_rotated_dEN(
    x_raw: float,
    y_raw: float,
    *,
    use_initial_odom_correction: bool,
    ox: float,
    oy: float,
    local_frame: str,
    R2: np.ndarray,
) -> np.ndarray:
    if use_initial_odom_correction:
        x_raw = x_raw - ox
        y_raw = y_raw - oy

    if local_frame == "EN":
        dE0, dN0 = x_raw, y_raw
    elif local_frame == "NE":
        dE0, dN0 = y_raw, x_raw
    elif local_frame == "WN":
        dE0, dN0 = -x_raw, y_raw
    else:
        raise ValueError(f"Unhandled local_frame {local_frame}")

    return R2 @ np.array([dE0, dN0], dtype=np.float64)


def project_local_yaml_to_utm(
    local_data: Dict[str, Any],
    *,
    gt_obj_nums: np.ndarray,
    gt_pts: np.ndarray,
    gt_row_nums: np.ndarray,
    anchor_object_number: int,
    anchor_local_tree_id: Optional[int],
    use_initial_odom_correction: bool,
    local_frame: str,
    manual_yaw_rad: Optional[float],
    yaw_source: str,
    ignore_yaml_yaw: bool,
) -> Tuple[np.ndarray, List[int], float]:
    if not isinstance(local_data, dict):
        raise ValueError("Local YAML must be a mapping.")

    e0, n0 = parse_initial_gps_utm(local_data)
    final_utm = parse_final_gps_utm(local_data)
    yaw = select_yaw(yaw_source, manual_yaw_rad, ignore_yaml_yaw, local_data, e0, n0, final_utm)
    R2 = rot2(yaw)
    ox, oy = parse_initial_odom_xy(local_data)

    trees = local_data.get("trees", None)
    if not isinstance(trees, dict) or len(trees) == 0:
        raise ValueError("Local YAML missing trees.")

    ids_sorted = sorted(int(k) for k in trees.keys())
    if anchor_local_tree_id is None:
        anchor_local_tree_id = ids_sorted[0]
    else:
        anchor_local_tree_id = int(anchor_local_tree_id)

    if str(anchor_local_tree_id) not in trees and anchor_local_tree_id not in trees:
        raise ValueError(f"Anchor local_tree_id {anchor_local_tree_id} not present in YAML trees keys.")

    anchor_xy = trees.get(anchor_local_tree_id, trees.get(str(anchor_local_tree_id)))
    if not isinstance(anchor_xy, (list, tuple)) or len(anchor_xy) < 2:
        raise ValueError(f"Bad anchor trees[{anchor_local_tree_id}] entry: {anchor_xy}")

    d_anchor = local_xy_to_rotated_dEN(
        float(anchor_xy[0]), float(anchor_xy[1]),
        use_initial_odom_correction=use_initial_odom_correction,
        ox=ox, oy=oy,
        local_frame=local_frame,
        R2=R2,
    )

    gt_anchor, _anchor_row = get_gt_anchor(gt_obj_nums, gt_pts, gt_row_nums, anchor_object_number)

    e0a = float(gt_anchor[0] - d_anchor[0])
    n0a = float(gt_anchor[1] - d_anchor[1])

    pts = np.zeros((len(ids_sorted), 2), dtype=np.float64)
    for i, tid in enumerate(ids_sorted):
        vv = trees.get(tid, trees.get(str(tid)))
        if not isinstance(vv, (list, tuple)) or len(vv) < 2:
            raise ValueError(f"Bad trees[{tid}] entry: {vv}")
        d = local_xy_to_rotated_dEN(
            float(vv[0]), float(vv[1]),
            use_initial_odom_correction=use_initial_odom_correction,
            ox=ox, oy=oy,
            local_frame=local_frame,
            R2=R2,
        )
        pts[i, 0] = e0a + float(d[0])
        pts[i, 1] = n0a + float(d[1])

    return pts, ids_sorted, float(yaw)


# ============================================================
# Strict one-to-one SD scoring
# ============================================================

def indexwise_sd_score_from_anchor_forward_strict(
    local_pts: np.ndarray,
    gt_anchor: np.ndarray,
    gt_row_obj: np.ndarray,
    gt_row_pts: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    *,
    forward_sign: int,
    eps: float = 1e-6,
    mismatch_penalty_per_tree: float = 10.0,
) -> Dict[str, Any]:
    sG = (gt_row_pts - gt_anchor[None, :]) @ u
    sL = (local_pts - gt_anchor[None, :]) @ u

    keepG = (forward_sign * sG) >= -eps
    keepL = (forward_sign * sL) >= -eps

    G = gt_row_pts[keepG]
    G_obj = gt_row_obj[keepG]
    sGk = sG[keepG]

    L = local_pts[keepL]
    sLk = sL[keepL]

    oG = np.argsort(forward_sign * sGk)
    oL = np.argsort(forward_sign * sLk)

    G = G[oG]
    G_obj = G_obj[oG]
    L = L[oL]

    nG = int(G.shape[0])
    nL = int(L.shape[0])
    n_pairs = int(min(nG, nL))
    if n_pairs <= 0:
        raise ValueError("No pairs to score after forward-from-anchor filtering.")

    Gp = G[:n_pairs]
    Lp = L[:n_pairs]

    err = (Lp - Gp).astype(np.float64)
    err_s = err @ u
    err_d = err @ v

    rmse_s = float(np.sqrt(np.mean(err_s * err_s)))
    rmse_d = float(np.sqrt(np.mean(err_d * err_d)))
    rmse_sd = float(np.sqrt(np.mean((err_s * err_s) + (err_d * err_d))))

    missing = abs(nG - nL)
    penalty = float(mismatch_penalty_per_tree) * float(missing)

    return {
        "n_pairs": n_pairs,
        "n_gt_forward": nG,
        "n_local_forward": nL,
        "missing_count_abs": int(missing),
        "mismatch_penalty_m": penalty,
        "rmse_s_m": rmse_s,
        "rmse_d_m": rmse_d,
        "rmse_sd_m": rmse_sd,
        "rmse_s_with_penalty_m": rmse_s + penalty,
        "rmse_d_with_penalty_m": rmse_d + penalty,
        "rmse_sd_with_penalty_m": rmse_sd + penalty,
        "paired_gt_object_nums": G_obj[:n_pairs].tolist(),
        "err_s_m": err_s.tolist(),
        "err_d_m": err_d.tolist(),
    }


# ============================================================
# Robot motion sign
# ============================================================

def _extract_robot_xy_list(local_data: Dict[str, Any]) -> Optional[List[Tuple[float, float]]]:
    for key in ("robot_poses", "robot_positions", "robot_path", "poses", "odometry", "robot_pose_list"):
        v = local_data.get(key, None)
        if not v:
            continue

        out: List[Tuple[float, float]] = []

        if isinstance(v, list):
            for item in v:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    out.append((float(item[0]), float(item[1])))
                elif isinstance(item, dict):
                    if "x" in item and "y" in item:
                        out.append((float(item["x"]), float(item["y"])))
                    elif "position" in item and isinstance(item["position"], dict):
                        pos = item["position"]
                        if "x" in pos and "y" in pos:
                            out.append((float(pos["x"]), float(pos["y"])))
            return out if out else None

        if isinstance(v, dict):
            for _, item in v.items():
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    out.append((float(item[0]), float(item[1])))
                elif isinstance(item, dict) and "x" in item and "y" in item:
                    out.append((float(item["x"]), float(item["y"])))
            return out if out else None

    return None


def motion_sign_along_row_u(
    local_data: Dict[str, Any],
    *,
    gt_obj_nums: np.ndarray,
    gt_pts: np.ndarray,
    gt_row_nums: np.ndarray,
    anchor_object_number: int,
    anchor_local_tree_id: Optional[int],
    use_initial_odom_correction: bool,
    local_frame: str,
    manual_yaw_rad: Optional[float],
    yaw_source: str,
    ignore_yaml_yaw: bool,
    u: np.ndarray,
) -> int:
    if not isinstance(local_data, dict):
        return +1

    xy_list = _extract_robot_xy_list(local_data)
    if not xy_list or len(xy_list) < 2:
        return +1

    e0, n0 = parse_initial_gps_utm(local_data)
    final_utm = parse_final_gps_utm(local_data)
    yaw = select_yaw(yaw_source, manual_yaw_rad, ignore_yaml_yaw, local_data, e0, n0, final_utm)
    R2 = rot2(yaw)
    ox, oy = parse_initial_odom_xy(local_data)

    trees = local_data.get("trees", None)
    if not isinstance(trees, dict) or len(trees) == 0:
        return +1
    ids_sorted = sorted(int(k) for k in trees.keys())
    if anchor_local_tree_id is None:
        anchor_local_tree_id = ids_sorted[0]
    anchor_xy = trees.get(anchor_local_tree_id, trees.get(str(anchor_local_tree_id)))
    if not isinstance(anchor_xy, (list, tuple)) or len(anchor_xy) < 2:
        return +1

    d_anchor = local_xy_to_rotated_dEN(
        float(anchor_xy[0]), float(anchor_xy[1]),
        use_initial_odom_correction=use_initial_odom_correction,
        ox=ox, oy=oy,
        local_frame=local_frame,
        R2=R2,
    )
    gt_anchor, _ = get_gt_anchor(gt_obj_nums, gt_pts, gt_row_nums, anchor_object_number)
    e0a = float(gt_anchor[0] - d_anchor[0])
    n0a = float(gt_anchor[1] - d_anchor[1])

    x0, y0 = xy_list[0]
    x1, y1 = xy_list[-1]
    d0 = local_xy_to_rotated_dEN(
        x0, y0,
        use_initial_odom_correction=use_initial_odom_correction,
        ox=ox, oy=oy,
        local_frame=local_frame,
        R2=R2,
    )
    d1 = local_xy_to_rotated_dEN(
        x1, y1,
        use_initial_odom_correction=use_initial_odom_correction,
        ox=ox, oy=oy,
        local_frame=local_frame,
        R2=R2,
    )
    p0 = np.array([e0a + d0[0], n0a + d0[1]], dtype=np.float64)
    p1 = np.array([e0a + d1[0], n0a + d1[1]], dtype=np.float64)

    ds = float((p1 - p0) @ u)
    return +1 if ds >= 0.0 else -1


# ============================================================
# Dump service parsing
# ============================================================

def parse_written_path_from_dump(stdout: str) -> Path:
    if not stdout:
        raise RuntimeError("Dump service returned empty stdout; cannot parse written_path.")

    m = re.search(r"written_path:\s*([^\s]+)", stdout)
    if m:
        return Path(m.group(1).strip().strip('"').strip("'")).expanduser()

    m = re.search(r"written_path\s*=\s*'([^']+)'", stdout)
    if m:
        return Path(m.group(1).strip()).expanduser()

    m = re.search(r'written_path\s*=\s*"([^"]+)"', stdout)
    if m:
        return Path(m.group(1).strip()).expanduser()

    raise RuntimeError(f"Could not parse written_path from dump service output.\n---\n{stdout}\n---")


# ============================================================
# Worker env
# ============================================================

def build_worker_env(*, worker_id: int, domain_id_base: int, ros_log_dir: Path) -> Dict[str, str]:
    env = dict(os.environ)
    env["ROS_DOMAIN_ID"] = str(int(domain_id_base) + int(worker_id))
    env["ROS_LOG_DIR"] = str(ros_log_dir)
    return env


# ============================================================
# IO helpers
# ============================================================

def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2))


def load_json_if_exists(p: Path) -> Optional[dict]:
    if not p.exists():
        return None
    try:
        obj = json.loads(p.read_text())
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


# ============================================================
# Optuna search space (TPE)
# ============================================================

def suggest_params(trial) -> Dict[str, Any]:  # "optuna.trial.Trial"
    params: Dict[str, Any] = {}

    params["slot_spacing"] = trial.suggest_float("slot_spacing", 0.9, 1.2)
    params["slot_s_gate"] = trial.suggest_float("slot_s_gate", 0.35, 0.55)

    params["meas_std_x_fwd"] = trial.suggest_float("meas_std_x_fwd", 0.25, 0.70)
    params["meas_std_y_lat"] = trial.suggest_float("meas_std_y_lat", 0.25, 0.80)

    params["motion_noise.a_trans"] = trial.suggest_float("motion_noise.a_trans", 0.05, 0.35)
    params["motion_noise.b_trans"] = trial.suggest_float("motion_noise.b_trans", 0.05, 0.35)
    params["motion_noise.c_lat"] = trial.suggest_float("motion_noise.c_lat", 0.05, 0.30)
    params["motion_noise.a_rot"] = trial.suggest_float("motion_noise.a_rot", 0.03, 0.12)
    params["motion_noise.b_rot"] = trial.suggest_float("motion_noise.b_rot", 0.03, 0.15)

    params["prior_sigma_s"] = trial.suggest_float("prior_sigma_s", 0.15, 0.90)
    params["prior_sigma_d"] = trial.suggest_float("prior_sigma_d", 0.15, 0.90)

    params["template_prior.sigma_s"] = trial.suggest_float("template_prior.sigma_s", 0.30, 0.80)
    params["template_prior.sigma_d"] = trial.suggest_float("template_prior.sigma_d", 0.10, 0.30)
    params["template_prior.w"] = trial.suggest_float("template_prior.w", 1.0, 5.0)
    params["template_prior.decay_k"] = trial.suggest_float("template_prior.decay_k", 0.5, 3.0)
    params["template_prior.max_seen"] = trial.suggest_int("template_prior.max_seen", 4, 16)

    params["max_back_assoc"] = trial.suggest_categorical("max_back_assoc", [0, 1])
    params["max_fwd_assoc"] = trial.suggest_categorical("max_fwd_assoc", [1, 2, 3])

    params["maha_gate_median"] = trial.suggest_categorical("maha_gate_median", [0.0, 4.0, 6.0, 8.0])

    return params



# ============================================================
# One Optuna trial runner
# ============================================================

def run_one_optuna_trial(
    *,
    trial,  # "optuna.trial.Trial"
    args_ns: argparse.Namespace,
    gt_obj_nums: np.ndarray,
    gt_pts: np.ndarray,
    gt_row_nums: np.ndarray,
    gt_anchor: np.ndarray,
    anchor_row_num: int,
    gt_row_obj: np.ndarray,
    gt_row_pts: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    env: Dict[str, str],
    worker_id: int,
) -> Tuple[float, float]:
    out_root = Path(args_ns.out_dir).expanduser().resolve()
    dataset_dir = Path(args_ns.dataset).expanduser().resolve()

    trial_tag = f"trial_{trial.number:06d}_w{worker_id:02d}"
    trial_dir = out_root / "optuna_trials" / trial_tag
    trial_dir.mkdir(parents=True, exist_ok=True)

    params = suggest_params(trial)
    write_json(trial_dir / "params.json", params)

    slam_log = trial_dir / "slam_launch.log"
    slam_pkg, slam_launch_file = args_ns.slam_launch.split()
    slam_cmd = ["ros2", "launch", slam_pkg, slam_launch_file] + list(args_ns.slam_launch_args)

    slam_p = popen_cmd(slam_cmd, log_path=slam_log, env=env)

    try:
        if not wait_for_node(args_ns.slam_node, timeout_s=float(args_ns.wait_node_timeout_s), env=env):
            raise RuntimeError(f"Timed out waiting for node {args_ns.slam_node}. See {slam_log}")

        set_log_lines: List[str] = []
        for k, v in params.items():
            ros2_param_set(args_ns.slam_node, k, v, env=env)
            set_log_lines.append(f"{k}={v}")
        (trial_dir / "param_set.txt").write_text("\n".join(set_log_lines) + "\n")

        replay_cmd = [
            "ros2", "run", "tree_template", "bag_data_replayer", "--",
            "--dataset", str(dataset_dir),
            "--speed", str(args_ns.speed),
        ]
        (trial_dir / "replayer_cmd.txt").write_text(" ".join(replay_cmd) + "\n")
        run_cmd(replay_cmd, check=True, capture=False, timeout_s=None, env=env)

        if not wait_for_service(args_ns.dump_service, timeout_s=float(args_ns.dump_timeout_s), env=env):
            raise RuntimeError(f"Dump service not found: {args_ns.dump_service}")

        trial_yaml_base = trial_dir / "slam_trunks.yaml"
        append_ts = "true" if bool(args_ns.dump_append_timestamp) else "false"
        payload = "{output_path: '" + str(trial_yaml_base) + "', append_timestamp: " + append_ts + "}"

        dump_cmd = ["ros2", "service", "call", args_ns.dump_service, args_ns.dump_service_type, payload]
        p_dump = run_cmd(dump_cmd, check=True, capture=True, timeout_s=float(args_ns.dump_timeout_s), env=env)
        (trial_dir / "dump_service_cmd.txt").write_text(" ".join(dump_cmd) + "\n")
        (trial_dir / "dump_service_out.txt").write_text((p_dump.stdout or "") + "\n" + (p_dump.stderr or ""))

        time.sleep(float(args_ns.settle_s))

        yaml_path = parse_written_path_from_dump(p_dump.stdout or "")
        if not yaml_path.exists():
            raise FileNotFoundError(f"Dump service reported written_path but file not found: {yaml_path}")

        copied_yaml = trial_dir / yaml_path.name
        if yaml_path.resolve() != copied_yaml.resolve():
            shutil.copy2(yaml_path, copied_yaml)

        local_data = yaml.safe_load(copied_yaml.read_text())

        local_pts, _local_ids, yaw_used = project_local_yaml_to_utm(
            local_data,
            gt_obj_nums=gt_obj_nums,
            gt_pts=gt_pts,
            gt_row_nums=gt_row_nums,
            anchor_object_number=int(args_ns.anchor_object_number),
            anchor_local_tree_id=args_ns.anchor_local_tree_id,
            use_initial_odom_correction=bool(args_ns.use_initial_odom_correction),
            local_frame=str(args_ns.local_frame),
            manual_yaw_rad=args_ns.manual_yaw_rad,
            yaw_source=str(args_ns.yaw_source),
            ignore_yaml_yaw=bool(args_ns.ignore_yaml_yaw),
        )

        forward_sign = motion_sign_along_row_u(
            local_data,
            gt_obj_nums=gt_obj_nums,
            gt_pts=gt_pts,
            gt_row_nums=gt_row_nums,
            anchor_object_number=int(args_ns.anchor_object_number),
            anchor_local_tree_id=args_ns.anchor_local_tree_id,
            use_initial_odom_correction=bool(args_ns.use_initial_odom_correction),
            local_frame=str(args_ns.local_frame),
            manual_yaw_rad=args_ns.manual_yaw_rad,
            yaw_source=str(args_ns.yaw_source),
            ignore_yaml_yaw=bool(args_ns.ignore_yaml_yaw),
            u=u,
        )

        score = indexwise_sd_score_from_anchor_forward_strict(
            local_pts, gt_anchor, gt_row_obj, gt_row_pts, u, v,
            forward_sign=forward_sign,
            mismatch_penalty_per_tree=float(args_ns.mismatch_penalty_per_tree),
        )

        score["forward_sign"] = int(forward_sign)
        score["yaw_used_rad"] = float(yaw_used)
        score["yaml"] = str(copied_yaml)
        score["trial"] = trial_tag
        score["params"] = params
        score["anchor_row_num"] = int(anchor_row_num)

        write_json(trial_dir / "score.json", score)

        trial.set_user_attr("trial_dir", str(trial_dir))
        trial.set_user_attr("yaml", str(copied_yaml))
        trial.set_user_attr("rmse_s_with_penalty_m", float(score["rmse_s_with_penalty_m"]))
        trial.set_user_attr("rmse_d_with_penalty_m", float(score["rmse_d_with_penalty_m"]))
        trial.set_user_attr("rmse_sd_with_penalty_m", float(score["rmse_sd_with_penalty_m"]))
        trial.set_user_attr("missing_count_abs", int(score["missing_count_abs"]))
        trial.set_user_attr("n_pairs", int(score["n_pairs"]))
        trial.set_user_attr("n_gt_forward", int(score["n_gt_forward"]))
        trial.set_user_attr("n_local_forward", int(score["n_local_forward"]))

        return float(score["rmse_s_with_penalty_m"]), float(score["rmse_d_with_penalty_m"])

    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        (trial_dir / "error.txt").write_text(err + "\n")
        trial.set_user_attr("failed", True)
        trial.set_user_attr("error", err)
        return 1.0e9, 1.0e9

    finally:
        terminate_process_group(slam_p)


# ============================================================
# Best selection + aggregation
# ============================================================

def pick_best_min_sd(scores: List[dict]) -> Optional[dict]:
    """
    Pick best trial by minimizing rmse_sd_with_penalty_m.
    Tie-breaks (in order): missing_count_abs, rmse_s_with_penalty_m, rmse_d_with_penalty_m.
    """
    best: Optional[dict] = None
    best_sd = float("inf")
    best_miss = float("inf")
    best_s = float("inf")
    best_d = float("inf")

    for sc in scores:
        try:
            sd = float(sc.get("rmse_sd_with_penalty_m", float("inf")))
            miss = float(sc.get("missing_count_abs", float("inf")))
            s = float(sc.get("rmse_s_with_penalty_m", float("inf")))
            d = float(sc.get("rmse_d_with_penalty_m", float("inf")))
        except Exception:
            continue

        # primary: minimize sd
        if sd < best_sd - 1e-12:
            best = sc
            best_sd, best_miss, best_s, best_d = sd, miss, s, d
            continue

        # tie-breaks
        if abs(sd - best_sd) <= 1e-12:
            if miss < best_miss - 1e-12:
                best = sc
                best_sd, best_miss, best_s, best_d = sd, miss, s, d
            elif abs(miss - best_miss) <= 1e-12 and s < best_s - 1e-12:
                best = sc
                best_sd, best_miss, best_s, best_d = sd, miss, s, d
            elif abs(miss - best_miss) <= 1e-12 and abs(s - best_s) <= 1e-12 and d < best_d - 1e-12:
                best = sc
                best_sd, best_miss, best_s, best_d = sd, miss, s, d

    return best


def write_trials_csv(out_root: Path, scores_by_trial: Dict[str, dict]) -> Path:
    import csv
    trials_csv = out_root / "trials.csv"
    tmp = out_root / "trials.csv.tmp"

    rows = sorted(scores_by_trial.items(), key=lambda kv: kv[0])
    with tmp.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "trial", "yaml",
            "rmse_s_with_penalty_m", "rmse_d_with_penalty_m", "rmse_sd_with_penalty_m",
            "rmse_s_m", "rmse_d_m", "rmse_sd_m",
            "missing_count_abs", "mismatch_penalty_m",
            "n_pairs", "n_gt_forward", "n_local_forward",
            "yaw_used_rad",
            "anchor_row_num",
            "params_json",
        ])
        for trial_tag, sc in rows:
            trial_dir = out_root / "optuna_trials" / trial_tag
            params_json_name = "params.json" if (trial_dir / "params.json").exists() else ""
            w.writerow([
                trial_tag, Path(str(sc.get("yaml", ""))).name,
                f"{float(sc.get('rmse_s_with_penalty_m', 0.0)):.6f}",
                f"{float(sc.get('rmse_d_with_penalty_m', 0.0)):.6f}",
                f"{float(sc.get('rmse_sd_with_penalty_m', 0.0)):.6f}",
                f"{float(sc.get('rmse_s_m', 0.0)):.6f}",
                f"{float(sc.get('rmse_d_m', 0.0)):.6f}",
                f"{float(sc.get('rmse_sd_m', 0.0)):.6f}",
                int(sc.get("missing_count_abs", 0)),
                f"{float(sc.get('mismatch_penalty_m', 0.0)):.6f}",
                int(sc.get("n_pairs", 0)),
                int(sc.get("n_gt_forward", 0)),
                int(sc.get("n_local_forward", 0)),
                f"{float(sc.get('yaw_used_rad', 0.0)):.10f}",
                int(sc.get("anchor_row_num", -1)),
                params_json_name,
            ])

    tmp.replace(trials_csv)
    return trials_csv


def aggregate_scores(out_root: Path) -> Tuple[List[dict], Dict[str, dict]]:
    scores: List[dict] = []
    scores_by_trial: Dict[str, dict] = {}
    trial_dirs = sorted((out_root / "optuna_trials").glob("trial_*"))
    for td in trial_dirs:
        sc = load_json_if_exists(td / "score.json")
        if sc is None:
            continue
        scores.append(sc)
        scores_by_trial[td.name] = sc
    return scores, scores_by_trial


def _best_history_next_index(best_hist_dir: Path) -> int:
    best_hist_dir.mkdir(parents=True, exist_ok=True)
    mx = 0
    for p in best_hist_dir.glob("best_*"):
        m = re.match(r"best_(\d+)$", p.name)
        if m:
            mx = max(mx, int(m.group(1)))
    return mx + 1


def snapshot_best(out_root: Path, best: dict, *, kind: str) -> None:
    best_trial = str(best.get("trial", "")).strip()
    if not best_trial:
        return

    trial_dir = out_root / "optuna_trials" / best_trial
    if not trial_dir.exists():
        return

    best_yaml_path = Path(str(best.get("yaml", "")))
    if not best_yaml_path.exists():
        ymls = list(trial_dir.glob("*.yml")) + list(trial_dir.glob("*.yaml"))
        if ymls:
            best_yaml_path = ymls[0]

    params_path = trial_dir / "params.json"
    score_path = trial_dir / "score.json"

    if kind == "current":
        dst = out_root / "best_current"
        if dst.exists():
            shutil.rmtree(dst, ignore_errors=True)
        dst.mkdir(parents=True, exist_ok=True)
    else:
        hist = out_root / "best_history"
        idx = _best_history_next_index(hist)
        dst = hist / f"best_{idx:04d}"
        dst.mkdir(parents=True, exist_ok=True)

    if best_yaml_path.exists():
        shutil.copy2(best_yaml_path, dst / best_yaml_path.name)

    if params_path.exists():
        shutil.copy2(params_path, dst / "best_params.json")
    else:
        (dst / "best_params.json").write_text(json.dumps(best.get("params", {}), indent=2))

    if score_path.exists():
        shutil.copy2(score_path, dst / "best_score.json")
    else:
        (dst / "best_score.json").write_text(json.dumps(best, indent=2))

    meta = {
        "trial": best_trial,
        "yaml": best_yaml_path.name if best_yaml_path else "",
        "rmse_s_with_penalty_m": float(best.get("rmse_s_with_penalty_m", float("nan"))),
        "rmse_d_with_penalty_m": float(best.get("rmse_d_with_penalty_m", float("nan"))),
        "rmse_sd_with_penalty_m": float(best.get("rmse_sd_with_penalty_m", float("nan"))),
        "missing_count_abs": int(best.get("missing_count_abs", 0)),
        "written_at_unix": time.time(),
    }
    (dst / "best_meta.json").write_text(json.dumps(meta, indent=2))


def best_signature(best: dict) -> str:
    t = str(best.get("trial", ""))
    s = float(best.get("rmse_s_with_penalty_m", float("inf")))
    d = float(best.get("rmse_d_with_penalty_m", float("inf")))
    sd = float(best.get("rmse_sd_with_penalty_m", float("inf")))
    return f"{t}|{s:.12f}|{d:.12f}|{sd:.12f}"


# ============================================================
# Parallel worker (Optuna)
# ============================================================

def worker_process_main(payload: Dict[str, Any]) -> None:
    if optuna is None:
        raise RuntimeError(f"Optuna not available: {_OPTUNA_IMPORT_ERR}")

    worker_id = int(payload["worker_id"])
    domain_id_base = int(payload["domain_id_base"])

    args_dict = payload["args"]
    args_ns = argparse.Namespace(**args_dict)

    out_root = Path(args_ns.out_dir).expanduser().resolve()
    worker_root = out_root / f"_worker_{worker_id:02d}"
    worker_root.mkdir(parents=True, exist_ok=True)
    ros_log_dir = worker_root / "ros_logs"
    ros_log_dir.mkdir(parents=True, exist_ok=True)

    env = build_worker_env(worker_id=worker_id, domain_id_base=domain_id_base, ros_log_dir=ros_log_dir)

    gt_json_path = Path(args_ns.gt_json).expanduser().resolve()
    gt_obj_nums, gt_pts, gt_row_nums = load_ground_truth(gt_json_path)
    gt_anchor, anchor_row_num, gt_row_obj, gt_row_pts, u, v = extract_gt_row_by_row_num(
        gt_obj_nums, gt_pts, gt_row_nums, int(args_ns.anchor_object_number)
    )

    sampler = optuna.samplers.TPESampler(seed=int(args_ns.seed), multivariate=True, constant_liar=True)

    study = optuna.create_study(
        study_name=str(args_ns.study_name),
        storage=str(args_ns.optuna_storage),
        load_if_exists=True,
        directions=["minimize", "minimize"],
        sampler=sampler,
    )

    n_local_trials = int(payload["n_trials"])
    for _ in range(n_local_trials):
        trial = study.ask()
        rmse_s, rmse_d = run_one_optuna_trial(
            trial=trial,
            args_ns=args_ns,
            gt_obj_nums=gt_obj_nums,
            gt_pts=gt_pts,
            gt_row_nums=gt_row_nums,
            gt_anchor=gt_anchor,
            anchor_row_num=anchor_row_num,
            gt_row_obj=gt_row_obj,
            gt_row_pts=gt_row_pts,
            u=u,
            v=v,
            env=env,
            worker_id=worker_id,
        )
        study.tell(trial, (rmse_s, rmse_d))


# ============================================================
# Main
# ============================================================

def split_counts(total_trials: int, workers: int) -> List[int]:
    workers = max(1, int(workers))
    base = total_trials // workers
    rem = total_trials % workers
    return [base + (1 if i < rem else 0) for i in range(workers)]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Optuna TPE sweep runner for RowFastSLAM with incremental trials.csv and best snapshots."
    )

    ap.add_argument("--dataset", required=True)
    ap.add_argument("--speed", type=float, default=6.0)

    ap.add_argument("--slam_launch", default="tree_template trunk_to_template_pub.launch.py",
                    help='Launch target as "package launch_file.py"')
    ap.add_argument("--slam_launch_args", nargs="*", default=["use_sim_time:=true", "replay:=true"])

    ap.add_argument("--settle_s", type=float, default=1.0,
        help="Seconds to wait after dump service returns before reading/scoring the written YAML.")

    ap.add_argument("--dump_service", default="/save_slam_results")
    ap.add_argument("--dump_service_type", default="tree_template_interfaces/srv/SaveSlamResults")
    ap.add_argument("--dump_append_timestamp", action="store_true")
    ap.add_argument("--dump_timeout_s", type=float, default=30.0)

    ap.add_argument("--slam_node", default="row_fast_slam")
    ap.add_argument("--wait_node_timeout_s", type=float, default=40.0)

    ap.add_argument("--gt_json", required=True)
    ap.add_argument("--anchor_object_number", type=int, default=572)
    ap.add_argument("--anchor_local_tree_id", type=int, default=None)
    ap.add_argument("--use_initial_odom_correction", action="store_true")
    ap.add_argument("--local_frame", choices=["EN", "NE", "WN"], default="EN")
    ap.add_argument("--yaw_source", choices=["auto", "yaml", "gps", "gps_plus_yaml"], default="auto")
    ap.add_argument("--ignore_yaml_yaw", action="store_true")
    ap.add_argument("--manual_yaw_rad", type=float, default=None)

    ap.add_argument("--mismatch_penalty_per_tree", type=float, default=10.0)

    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out_dir", required=True)

    # incremental write/snapshot behavior
    ap.add_argument("--keep_best", action="store_true",
                    help="Writes best_current/ during the run and at the end.")
    ap.add_argument("--best_update_every_s", type=float, default=30.0,
                    help="How often to update trials.csv + best_current during the run.")
    ap.add_argument("--best_snapshot_history", action="store_true",
                    help="If set, every time best improves create best_history/best_XXXX/")

    # crash / exit behavior
    ap.add_argument("--kill_on_error", action="store_true",
                    help="If set, any worker nonzero exitcode aborts the sweep. Otherwise we still aggregate what worked.")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--domain_id_base", type=int, default=30)

    ap.add_argument("--study_name", default="row_fastslam_tpe_sd_moo")
    ap.add_argument("--optuna_storage", default=None)

    args = ap.parse_args()

    if optuna is None:
        raise RuntimeError(
            f"Optuna is not installed or failed to import.\n"
            f"Error: {_OPTUNA_IMPORT_ERR}\n"
            f"Install: pip install optuna"
        )

    out_root = Path(args.out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "optuna_trials").mkdir(parents=True, exist_ok=True)

    dataset_dir = Path(args.dataset).expanduser().resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset dir not found: {dataset_dir}")

    gt_json_path = Path(args.gt_json).expanduser().resolve()
    if not gt_json_path.exists():
        raise FileNotFoundError(f"GT JSON not found: {gt_json_path}")

    gt_obj_nums0, gt_pts0, gt_row_nums0 = load_ground_truth(gt_json_path)
    _gt_anchor_tmp, anchor_row_tmp = get_gt_anchor(gt_obj_nums0, gt_pts0, gt_row_nums0, int(args.anchor_object_number))

    if args.optuna_storage is None:
        db_path = out_root / "optuna_study.db"
        args.optuna_storage = f"sqlite:///{db_path}"

    print(f"[INFO] Output dir: {out_root}")
    print(f"[INFO] Trials: {args.trials} | Workers: {args.workers} | domain_id_base: {args.domain_id_base}")
    print(f"[INFO] Anchor object_num: {args.anchor_object_number} | Anchor row_num (GT): {anchor_row_tmp}")
    print(f"[INFO] Optuna study: {args.study_name}")
    print(f"[INFO] Optuna storage: {args.optuna_storage}")

    sampler = optuna.samplers.TPESampler(seed=int(args.seed), multivariate=True, constant_liar=True)
    _ = optuna.create_study(
        study_name=str(args.study_name),
        storage=str(args.optuna_storage),
        load_if_exists=True,
        directions=["minimize", "minimize"],
        sampler=sampler,
    )

    workers = max(1, int(args.workers))
    counts = split_counts(int(args.trials), workers)

    import multiprocessing as mp
    ctx = mp.get_context("spawn")

    args_dict = vars(args).copy()
    processes: List[mp.Process] = []

    failures_log = out_root / "worker_failures.log"
    last_agg_t = 0.0
    last_best_sig = ""

    def do_incremental_aggregate(*, force: bool = False) -> None:
        nonlocal last_agg_t, last_best_sig
        now = time.time()
        if (not force) and (now - last_agg_t) < float(args.best_update_every_s):
            return
        last_agg_t = now

        scores, scores_by_trial = aggregate_scores(out_root)

        if scores_by_trial:
            trials_csv = write_trials_csv(out_root, scores_by_trial)
            atomic_write_text(out_root / "status.txt",
                              f"updated_at_unix={now}\nnum_scored_trials={len(scores_by_trial)}\ntrials_csv={trials_csv}\n")

        if not scores:
            return

        best = pick_best_min_sd(scores)
        if not best:
            return

        if bool(args.keep_best):
            snapshot_best(out_root, best, kind="current")

        sig = best_signature(best)
        if sig != last_best_sig:
            last_best_sig = sig
            if bool(args.keep_best) and bool(args.best_snapshot_history):
                snapshot_best(out_root, best, kind="history")

    try:
        for wid, ntr in enumerate(counts):
            if ntr <= 0:
                continue
            payload = {
                "worker_id": wid,
                "domain_id_base": int(args.domain_id_base),
                "args": args_dict,
                "n_trials": int(ntr),
            }
            p = ctx.Process(target=worker_process_main, args=(payload,), daemon=False)
            p.start()
            processes.append(p)

        # Monitor and keep writing partial results even if some workers crash.
        while True:
            do_incremental_aggregate(force=False)
            if not any(p.is_alive() for p in processes):
                break
            time.sleep(0.5)

        do_incremental_aggregate(force=True)

        bad = [p for p in processes if p.exitcode not in (0, None)]
        if bad:
            codes = ", ".join(str(p.exitcode) for p in bad)
            msg = f"[WARN] one or more workers exited non-zero. exitcodes=[{codes}]\n"
            with failures_log.open("a") as f:
                f.write(msg)
            print(msg.strip())
            if bool(args.kill_on_error):
                raise RuntimeError(f"One or more workers exited non-zero. exitcodes=[{codes}]")

    except KeyboardInterrupt:
        print("\n[WARN] Ctrl-C: terminating workers...")
        for p in processes:
            try:
                p.terminate()
            except Exception:
                pass
        raise
    finally:
        for p in processes:
            if p.is_alive():
                try:
                    p.terminate()
                except Exception:
                    pass

    # Final aggregate / print
    scores, scores_by_trial = aggregate_scores(out_root)
    trials_csv = write_trials_csv(out_root, scores_by_trial) if scores_by_trial else None

    print("\n=== DONE ===")
    if trials_csv:
        print(f"Wrote: {trials_csv}")
    else:
        print("No trials.csv written (no score.json files found).")

    if not scores:
        print("No successful trials (check per-trial error.txt and slam_launch.log).")
        return

    best = pick_best_min_sd(scores)
    if best is None:
        print("No valid scores to select best from.")
        return

    best_trial = str(best.get("trial", ""))
    best_yaml_path = Path(str(best.get("yaml", "")))
    best_s = float(best.get("rmse_s_with_penalty_m", float("inf")))
    best_d = float(best.get("rmse_d_with_penalty_m", float("inf")))
    best_sd = float(best.get("rmse_sd_with_penalty_m", float("inf")))
    miss = int(best.get("missing_count_abs", 0))

    print(
        f"Best: rmse_s={best_s:.3f} m | rmse_d={best_d:.3f} m | rmse_sd={best_sd:.3f} m "
        f"| missing_abs={miss} | {best_yaml_path.name} in {best_trial}"
    )

    if bool(args.keep_best) and best is not None:
        snapshot_best(out_root, best, kind="current")
        if bool(args.best_snapshot_history):
            snapshot_best(out_root, best, kind="history")


if __name__ == "__main__":
    main()
