#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

import csv
import os
from collections import defaultdict, deque

import numpy as np
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from nav_msgs.msg import Odometry

from tree_template_interfaces.msg import TrunkInfo, TrunkRegistry


# ============= Utility / math helpers =============

def wrap_angle(theta: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (theta + np.pi) % (2.0 * np.pi) - np.pi



def inv_and_logdet_2x2(a: float, b: float, c: float, d: float) -> Tuple[bool, float, float, float, float, float]:
    """
    Closed-form inverse and log(det) for a 2x2 matrix [[a,b],[c,d]].
    Returns (ok, inv00, inv01, inv10, inv11, logdet).
    """
    det = a * d - b * c
    if det <= 0.0 or (not np.isfinite(det)):
        return (False, 0.0, 0.0, 0.0, 0.0, 0.0)
    inv_det = 1.0 / det
    inv00 =  d * inv_det
    inv01 = -b * inv_det
    inv10 = -c * inv_det
    inv11 =  a * inv_det
    return (True, inv00, inv01, inv10, inv11, float(np.log(det)))

def predict_parts(pose: np.ndarray, mu_lm: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Return (c, s, x_fwd_pred, y_lat_pred) for landmark mu_lm observed from pose.
    Uses the same measurement model as _predict_z_Hland_Jpose, but avoids array allocations.
    """
    x, y, th = float(pose[0]), float(pose[1]), float(pose[2])
    c = float(np.cos(th)); s = float(np.sin(th))
    dx = float(mu_lm[0] - x)
    dy = float(mu_lm[1] - y)
    x_fwd = c * dx + s * dy
    y_lat = -s * dx + c * dy
    return c, s, x_fwd, y_lat

def se2_compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compose SE(2) poses/deltas: a ⊕ b (apply body-frame delta b to pose a)."""
    ax, ay, ath = float(a[0]), float(a[1]), float(a[2])
    bx, by, bth = float(b[0]), float(b[1]), float(b[2])
    c = float(np.cos(ath)); s = float(np.sin(ath))
    x = ax + c * bx - s * by
    y = ay + s * bx + c * by
    th = wrap_angle(ath + bth)
    return np.array([x, y, th], dtype=float)

def se2_inverse(d: np.ndarray) -> np.ndarray:
    """Inverse of an SE(2) body-frame delta."""
    dx, dy, dth = float(d[0]), float(d[1]), float(d[2])
    c = float(np.cos(dth)); s = float(np.sin(dth))
    x = -(c * dx + s * dy)
    y = -(-s * dx + c * dy)
    th = wrap_angle(-dth)
    return np.array([x, y, th], dtype=float)

def se2_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return delta d such that a ⊕ d = b (delta expressed in frame of a)."""
    ax, ay, ath = float(a[0]), float(a[1]), float(a[2])
    bx, by, bth = float(b[0]), float(b[1]), float(b[2])
    dxw = bx - ax
    dyw = by - ay
    c = float(np.cos(ath)); s = float(np.sin(ath))
    dx = c * dxw + s * dyw
    dy = -s * dxw + c * dyw
    dth = wrap_angle(bth - ath)
    return np.array([dx, dy, dth], dtype=float)

def se2_from_odom(msg: Odometry) -> np.ndarray:
    """Extract [x, y, yaw] from nav_msgs/Odometry"""
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    yaw = R.from_quat([q.x, q.y, q.z, q.w]).as_euler("zyx", degrees=False)[0]
    return np.array([float(p.x), float(p.y), float(yaw)], dtype=float)

def se2_compose_many(poses: np.ndarray, delta: np.ndarray, out: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Vectorized SE(2) compose for many poses:
        out[i] = poses[i] ⊕ delta
    where delta is a body-frame delta applied at each pose[i].

    poses: (N,3) [x,y,yaw]
    delta: (3,)  [dx,dy,dyaw]
    out:   optional (N,3) preallocated output
    """
    if out is None:
        out = np.empty_like(poses)

    dx = float(delta[0]); dy = float(delta[1]); dth = float(delta[2])

    th = poses[:, 2]
    c = np.cos(th)
    s = np.sin(th)

    out[:, 0] = poses[:, 0] + c * dx - s * dy
    out[:, 1] = poses[:, 1] + s * dx + c * dy
    out[:, 2] = (poses[:, 2] + dth + np.pi) % (2.0 * np.pi) - np.pi
    return out

def se2_compose_single(pose: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Small helper: pose ⊕ delta, returns new array (no in-place)."""
    return se2_compose(pose, delta)

def quat_xyzw_from_yaw(yaw: float) -> np.ndarray:
    """Return quaternion [x, y, z, w] from yaw."""
    return R.from_euler("z", float(yaw)).as_quat()

def safe_nanmean(arr):
    a = arr[np.isfinite(arr)]
    return float(np.mean(a)) if a.size else float("nan")

# ============= Landmark EKF container =============

@dataclass
class LandmarkEKF:
    mu: np.ndarray        # (2,) landmark position in world frame [x, y]
    Sigma: np.ndarray     # (2,2) covariance
    seen_count: int = 0
    width_sum: float = 0.0
    width_count: int = 0

    @property
    def width_mean(self) -> float:
        if self.width_count > 0:
            return self.width_sum / self.width_count
        return float("nan")

    def update_width(self, w_meas: Optional[float]):
        if w_meas is None:
            return
        self.width_sum += float(w_meas)
        self.width_count += 1


# ============= Particle container =============

@dataclass
class Particle:
    """
    Particle holds EKF landmark states, while pose/weight live in shared NumPy arrays
    owned by the RowFastSLAMNode for fast vectorized updates and resampling.
    """
    idx: int
    landmarks: Dict[int, LandmarkEKF]     # slot index -> landmark EKF

    _poses: np.ndarray                    # (N,3) shared [x,y,yaw]
    _weights: np.ndarray                  # (N,) shared weights

    @property
    def pose(self) -> np.ndarray:
        # View into shared pose array (no copy)
        return self._poses[self.idx]

    @property
    def weight(self) -> float:
        return float(self._weights[self.idx])

    @weight.setter
    def weight(self, v: float) -> None:
        self._weights[self.idx] = float(v)



class RowFastSLAMNode(Node):
    """
    FastSLAM (particle filter over robot pose, EKF per landmark) specialized to an orchard row.

    Key behavior:
      - Slot association is based on the best particle's projection into the row basis
      - NEW: measurement-informed pose proposal (FastSLAM 2.0 style) to reduce wheel slip issues
    """
    def __init__(self):
        super().__init__("row_fast_slam")

        # ---------- Parameters ----------
        self.declare_parameter("num_particles", 500)
        self.declare_parameter("measurement_topic", "trunk_measurements_raw")
        self.declare_parameter("odom_topic", "/odometry/filtered")
        self.declare_parameter("registry_topic", "fastslam_registry")
        self.declare_parameter("odom_best_topic", "odom_slam_best")

        # Row structure (slot indexing along the row)
        self.declare_parameter("slot_spacing", 1.25)          # [m]
        self.declare_parameter("row_origin_s", 0.6)          # [m] s coordinate of slot 0
        self.declare_parameter("slot_s_gate", 0.8)           # [m] max |s_meas - s_slot| to accept
        self.declare_parameter("snap_downstream_spacing", False)
        self.declare_parameter("downstream_snap_mode", "unseen_only")  # "unseen_only" or "all_downstream"

        # Template-map initialization
        self.declare_parameter("use_template_map", False)
        self.declare_parameter("publish_unseen_template_landmarks", False) # If False, we will NOT publish template landmarks until they've been seen at least N times.
        self.declare_parameter("min_seen_count_to_publish", 1)
        self.declare_parameter("use_global_row_yaw", True)
        self.declare_parameter("num_slots", 48) # Marcus manually counted 68 trunk in Jostan's dataset for row 12 (not sure which are "polinators")
        self.declare_parameter("exceed_slot_num", False) # Allow dynamic extension beyond num_slots (when template map is enabled)
        self.declare_parameter("max_extra_slots", 0)  # 0 disables cap
        self.declare_parameter("row_origin_x", float("nan"))          # if NaN, use first odom pose
        self.declare_parameter("row_origin_y", float("nan"))         # if NaN, use first odom pose
        self.declare_parameter("row_dir_sign", 1)            # (+1 => +X, -1 => -X)
        self.declare_parameter("max_back_assoc", 0)
        self.declare_parameter("max_fwd_assoc", 2)
        
        self.declare_parameter("start_side", "near")       # slot 0 side (for template parity)
        self.declare_parameter("init_d_near", 0.0)
        self.declare_parameter("init_d_far", 0.0)
        self.declare_parameter("prior_sigma_s", 0.6)
        self.declare_parameter("prior_sigma_d", 0.6)
        self.declare_parameter("side_mode", "semantic")    # "fixed" or "geometry" or "semantic"

        # Measurement noise (std dev) in robot frame
        self.declare_parameter("meas_std_y_lat", 0.35)        # left/right
        self.declare_parameter("meas_std_x_fwd", 0.75)        # forward/back - heading

        # Motion noise parameters (interpreted in body frame)
        self.declare_parameter("motion_noise.a_trans", 0.55)
        self.declare_parameter("motion_noise.b_trans", 0.5)
        self.declare_parameter("motion_noise.c_lat", 1.0)
        self.declare_parameter("motion_noise.a_rot", 0.8)
        self.declare_parameter("motion_noise.b_rot", 2.5)

        # Resampling
        self.declare_parameter("resample_interval", 20) # 0 disables
        self.declare_parameter("neff_ratio_threshold", 0.3)

        # Semantic side memory
        self.declare_parameter("semantic_side_min_votes", 3)
        self.declare_parameter("semantic_side_fallback", "fixed")  # "fixed" or "unknown"

        # Innovation gating (optional): reject measurement updates if median Mahalanobis is too large.
        # Set <= 0.0 to disable.
        self.declare_parameter("maha_gate_median", 0.0)  # dimensionless (Mahalanobis^2)
        self.declare_parameter("maha_gate_min_particles", 50)
        
        # Landmark covariance floor (optional): prevents EKF landmarks from becoming unrealistically overconfident.
        # Set <= 0.0 to disable a component.
        self.declare_parameter("landmark_sigma_floor_x", 0.15)  # meters (world X)
        self.declare_parameter("landmark_sigma_floor_y", 0.15)  # meters (world Y)

        # Measurement-informed pose proposal (FastSLAM 2.0)
        self.declare_parameter("proposal.enable", False)
        self.declare_parameter("proposal.min_pose_std_xy", 0.08)     # m
        self.declare_parameter("proposal.min_pose_std_yaw", 0.06)    # rad

        # Debug
        self.declare_parameter("debug.log_every_n_meas", 0)          # 0 disables
        self.declare_parameter("debug.print_measurement_stamp", True)

        self.declare_parameter("row_yaw_init_window", 30)

        # Template prior (soft anchor toward template slot position)
        self.declare_parameter("template_prior.enable", True)
        self.declare_parameter("template_prior.sigma_s", 0.8)     # m along-row
        self.declare_parameter("template_prior.sigma_d", 0.6)     # m lateral
        self.declare_parameter("template_prior.decay_k", 3.0)     # hits; larger = prior lasts longer
        self.declare_parameter("template_prior.max_seen", 8)      # stop applying after this many hits
        self.declare_parameter("template_prior.w", 1.0)           # weight multiplier

        self.declare_parameter("meas_max_lag_sec", 0.5)

        # Odom history for delayed measurement alignment (seconds).
        self.declare_parameter("odom_history_sec", 20.0)
        self.meas_max_lag_sec = float(self.get_parameter("meas_max_lag_sec").value)
        self.odom_history_sec = float(self.get_parameter("odom_history_sec").value)

        self.row_yaw_init_window = int(self.get_parameter("row_yaw_init_window").value)

        self._row_yaw_samples = []
        self._row_axis_initialized = False

        # Template origin anchoring
        self.declare_parameter("template_origin_from_first_measurement", False)
        self.template_origin_from_first_measurement = bool(
            self.get_parameter("template_origin_from_first_measurement").value
        )
        self._template_origin_anchored = False

        # Rolling odom history: deque of (t_sec, pose_se2=[x,y,yaw]). Used to time-align delayed measurements.
        self._odom_hist = deque()

        # ---------- Resolve params ----------
        self.num_particles = int(self.get_parameter("num_particles").value)
        self.measurement_topic = str(self.get_parameter("measurement_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.registry_topic = str(self.get_parameter("registry_topic").value)
        self.odom_best_topic = str(self.get_parameter("odom_best_topic").value)

        self.slot_spacing = float(self.get_parameter("slot_spacing").value)
        self.row_origin_s = float(self.get_parameter("row_origin_s").value)
        self.slot_s_gate = float(self.get_parameter("slot_s_gate").value)

        self.snap_downstream_spacing = bool(self.get_parameter("snap_downstream_spacing").value)
        self.downstream_snap_mode = str(self.get_parameter("downstream_snap_mode").value)

        self.use_template_map = bool(self.get_parameter("use_template_map").value)
        self.publish_unseen_template_landmarks = bool(self.get_parameter("publish_unseen_template_landmarks").value)
        self.min_seen_count_to_publish = int(self.get_parameter("min_seen_count_to_publish").value)
        if self.min_seen_count_to_publish < 0:
            self.min_seen_count_to_publish = 0
        self.use_global_row_yaw = bool(self.get_parameter("use_global_row_yaw").value)
        self.num_slots = int(self.get_parameter("num_slots").value)
        self.exceed_slot_num = bool(self.get_parameter("exceed_slot_num").value)
        self.max_extra_slots = int(self.get_parameter("max_extra_slots").value)
        if self.max_extra_slots < 0:
            self.max_extra_slots = 0


        self.row_origin_xy = np.array(
            [
                float(self.get_parameter("row_origin_x").value),
                float(self.get_parameter("row_origin_y").value),
            ],
            dtype=float,
        )

        self.row_dir_sign = int(self.get_parameter("row_dir_sign").value)
        if self.row_dir_sign not in (-1, 1):
            self.row_dir_sign = 1

        self.max_back_assoc = int(self.get_parameter("max_back_assoc").value)
        self.max_fwd_assoc = int(self.get_parameter("max_fwd_assoc").value)

        self.start_side = str(self.get_parameter("start_side").value).strip().lower()
        self.init_d_near = float(self.get_parameter("init_d_near").value)
        self.init_d_far = float(self.get_parameter("init_d_far").value)
        self.prior_sigma_s = float(self.get_parameter("prior_sigma_s").value)
        self.prior_sigma_d = float(self.get_parameter("prior_sigma_d").value)
        self.side_mode = str(self.get_parameter("side_mode").value).strip().lower()

        self.meas_std = np.array(
            [
                float(self.get_parameter("meas_std_x_fwd").value),
                float(self.get_parameter("meas_std_y_lat").value),
            ],
            dtype=float,
        )
        self.R = np.diag(self.meas_std ** 2)
        self.R00 = float(self.R[0, 0])
        self.R11 = float(self.R[1, 1])

        self.a_trans = float(self.get_parameter("motion_noise.a_trans").value)
        self.b_trans = float(self.get_parameter("motion_noise.b_trans").value)
        self.c_lat = float(self.get_parameter("motion_noise.c_lat").value)
        self.a_rot = float(self.get_parameter("motion_noise.a_rot").value)
        self.b_rot = float(self.get_parameter("motion_noise.b_rot").value)

        self.resample_interval = int(self.get_parameter("resample_interval").value)
        self.neff_ratio_threshold = float(self.get_parameter("neff_ratio_threshold").value)

        self.semantic_side_min_votes = int(self.get_parameter("semantic_side_min_votes").value)
        self.semantic_side_fallback = str(self.get_parameter("semantic_side_fallback").value).strip().lower()
        if self.semantic_side_fallback not in ("fixed", "unknown"):
            self.semantic_side_fallback = "fixed"

        self.maha_gate_median = float(self.get_parameter("maha_gate_median").value)
        self.maha_gate_min_particles = max(1, int(self.get_parameter("maha_gate_min_particles").value))

        sx = float(self.get_parameter("landmark_sigma_floor_x").value)
        sy = float(self.get_parameter("landmark_sigma_floor_y").value)
        self._lm_sigma_floor_var_x = (sx * sx) if (sx > 0.0) else 0.0
        self._lm_sigma_floor_var_y = (sy * sy) if (sy > 0.0) else 0.0

        self.proposal_enable = bool(self.get_parameter("proposal.enable").value)
        self.proposal_min_pose_std_xy = float(self.get_parameter("proposal.min_pose_std_xy").value)
        self.proposal_min_pose_std_yaw = float(self.get_parameter("proposal.min_pose_std_yaw").value)

        self.debug_log_every_n_meas = int(self.get_parameter("debug.log_every_n_meas").value)
        self.debug_print_measurement_stamp = bool(self.get_parameter("debug.print_measurement_stamp").value)

        self.template_prior_enable = bool(self.get_parameter("template_prior.enable").value)
        self.template_prior_sigma_s = float(self.get_parameter("template_prior.sigma_s").value)
        self.template_prior_sigma_d = float(self.get_parameter("template_prior.sigma_d").value)
        self.template_prior_decay_k = float(self.get_parameter("template_prior.decay_k").value)
        self.template_prior_max_seen = int(self.get_parameter("template_prior.max_seen").value)
        self.template_prior_w = float(self.get_parameter("template_prior.w").value)

        if self.template_prior_decay_k <= 1e-6:
            self.template_prior_decay_k = 1e-6
        if self.template_prior_max_seen < 0:
            self.template_prior_max_seen = 0

        # ---------- Row basis ----------
        self.row_yaw_est: Optional[float] = None

        self.t_hat = np.array([float(self.row_dir_sign), 0.0], dtype=float)
        self.t_hat /= (np.linalg.norm(self.t_hat) + 1e-12)
        self.n_hat = np.array([-self.t_hat[1], self.t_hat[0]], dtype=float)

        # Association basis placeholders; will be overwritten and locked in _maybe_init_row_axis_from_odom()
        self.t_hat_assoc = self.t_hat.copy()
        self.n_hat_assoc = self.n_hat.copy()
        self.assoc_axis_locked = False

        # ---------- State ----------
        self.particles: List[Particle] = []
        self.last_odom_pose: Optional[np.ndarray] = None
        self._last_odom_time: Optional[float] = None

        # ---------- Work buffers (avoid per-callback allocations) ----------
        self._work_N: int = 0
        self._work_logw: np.ndarray = np.zeros((0,), dtype=np.float64)
        self._work_maha: np.ndarray = np.zeros((0,), dtype=np.float64)
        self._work_logdet: np.ndarray = np.zeros((0,), dtype=np.float64)
        self._work_poses: np.ndarray = np.zeros((0, 3), dtype=np.float64)
        self._work_cos: np.ndarray = np.zeros((0,), dtype=np.float64)
        self._work_sin: np.ndarray = np.zeros((0,), dtype=np.float64)
        self._work_poses_meas: np.ndarray = np.zeros((0, 3), dtype=np.float64)
        self.measurement_count = 0
        self._slot_side_votes: Dict[int, Dict[str, int]] = {}
        self._last_odom_msg: Optional[Odometry] = None
        self._boot_odom_pose: Optional[np.ndarray] = None
        self._pf_initialized = False

        # Store last motion noise (body-frame) from odom updates
        self._last_motion_std_body = np.array([0.2, 0.2, 0.1], dtype=float)

        # ---------- Pub/Sub ----------
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.registry_pub = self.create_publisher(TrunkRegistry, self.registry_topic, qos)
        self.odom_best_pub = self.create_publisher(Odometry, self.odom_best_topic, qos)

        self.cb_odom = MutuallyExclusiveCallbackGroup()
        self.cb_meas = MutuallyExclusiveCallbackGroup()

        odom_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=50,  # odom can be higher
            reliability=ReliabilityPolicy.BEST_EFFORT,  # bag replay friendly; ok for odom
            durability=DurabilityPolicy.VOLATILE,
        )

        # meas_qos = QoSProfile(
        #     history=HistoryPolicy.KEEP_LAST,
        #     depth=1,  # key change: do NOT backlog detections
        #     reliability=ReliabilityPolicy.BEST_EFFORT,
        #     durability=DurabilityPolicy.VOLATILE,
        # )
        meas_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, odom_qos, callback_group=self.cb_odom)
        self.meas_sub = self.create_subscription(TrunkInfo, self.measurement_topic, self.measurement_callback, meas_qos, callback_group=self.cb_meas)

        self.get_logger().info(
            "RowFastSLAMNode started.\n"
            f"  num_particles={self.num_particles}\n"
            f"  odom_topic={self.odom_topic}\n"
            f"  measurement_topic={self.measurement_topic}\n"
            f"  registry_topic={self.registry_topic}\n"
            f"  odom_best_topic={self.odom_best_topic}\n"
            f"  use_template_map={self.use_template_map}, use_global_row_yaw={self.use_global_row_yaw}\n"
            f"  num_slots={self.num_slots}, slot_spacing={self.slot_spacing:.2f}, slot_s_gate={self.slot_s_gate:.2f}\n"
            f"  row_dir_sign={self.row_dir_sign}, side_mode={self.side_mode}\n"
            f"  proposal_enable={self.proposal_enable}"
        )

        # -----------------------------
        # Debug CSV logging (tuning)
        # -----------------------------
        self.declare_parameter("debug.csv_enable", True)
        self.declare_parameter("debug.csv_path", "/home/marcus/apple_harvest_ws/src/orchard_tree_templating/tree_template/debug_data/fastslam_debug.csv")
        self.declare_parameter("debug.csv_flush_every_n", 1)

        self.debug_csv_enable = bool(self.get_parameter("debug.csv_enable").value)
        self.debug_csv_path = str(self.get_parameter("debug.csv_path").value)
        self.debug_csv_flush_every_n = int(self.get_parameter("debug.csv_flush_every_n").value)

        self._dbg_reject_counts = defaultdict(int)
        self._dbg_csv_fp = None
        self._dbg_csv = None
        self._dbg_csv_rows_since_flush = 0

        if self.debug_csv_enable:
            os.makedirs(os.path.dirname(self.debug_csv_path), exist_ok=True)

            # Always overwrite file on startup
            self._dbg_csv_fp = open(self.debug_csv_path, "w", newline="")
            self._dbg_csv = csv.writer(self._dbg_csv_fp)

            # Always write header (because "w" truncates any existing file)
            self._dbg_csv.writerow([
                "t_sec",
                "meas_idx",
                "slot_j",
                "z_x_fwd",
                "z_y_lat",
                "range_r",
                "proposal_enable",
                "neff",
                "w_max",
                "w_min",
                "w_std",
                "logw_max",
                "logw_mean",
                "logw_std",
                "maha_mean",
                "maha_p95",
                "logdet_mean",
                "logdet_p95",
                "reject_reason_counts",
            ])
            self._dbg_csv_fp.flush()

    def _dbg_reject(self, key: str) -> None:
        if getattr(self, "debug_csv_enable", False):
            self._dbg_reject_counts[key] += 1

    def _dbg_close(self) -> None:
        fp = getattr(self, "_dbg_csv_fp", None)
        if fp is not None:
            try:
                fp.close()
            except Exception:
                pass
            self._dbg_csv_fp = None
            self._dbg_csv = None

    def _best_particle(self) -> Optional[Particle]:
        if not self.particles:
            return None
        return self.particles[int(np.argmax([p.weight for p in self.particles]))]

    def _side_for_index(self, i: int) -> str:
        if self.start_side not in ("near", "far"):
            return "near"
        if i % 2 == 0:
            return self.start_side
        return "far" if self.start_side == "near" else "near"

    def _template_d_for_slot(self, j: int) -> float:
        side = self._side_for_index(int(j))
        return float(self.init_d_near) if side == "near" else float(self.init_d_far)

    def _sigma_world_from_row_sigmas(self) -> np.ndarray:
        t = self.t_hat_assoc.reshape(2, 1)
        n = self.n_hat_assoc.reshape(2, 1)
        return (self.prior_sigma_s ** 2) * (t @ t.T) + (self.prior_sigma_d ** 2) * (n @ n.T)

    def _update_semantic_side_votes(self, slot_j: int, side: str):
        side = (side or "").strip().lower()
        if side not in ("near", "far"):
            return
        if slot_j not in self._slot_side_votes:
            self._slot_side_votes[slot_j] = {"near": 0, "far": 0}
        self._slot_side_votes[slot_j][side] += 1

    def _semantic_side_for_slot(self, slot_j: int) -> str:
        votes = self._slot_side_votes.get(slot_j)
        if not votes:
            return self._side_for_index(int(slot_j)) if self.semantic_side_fallback == "fixed" else "unknown"

        n_near = int(votes.get("near", 0))
        n_far = int(votes.get("far", 0))
        total = n_near + n_far

        if total < max(self.semantic_side_min_votes, 1):
            return self._side_for_index(int(slot_j)) if self.semantic_side_fallback == "fixed" else "unknown"

        if n_near == n_far:
            return self._side_for_index(int(slot_j)) if self.semantic_side_fallback == "fixed" else "unknown"

        return "near" if n_near > n_far else "far"

    def _maybe_init_row_axis_from_odom(self, yaw: float) -> None:
        if self._row_axis_initialized:
            return

        # If we are NOT using a global row yaw estimate, initialize immediately from 1 sample.
        needed = 1 if (not self.use_global_row_yaw) else int(self.row_yaw_init_window)
        needed = max(1, needed)

        self._row_yaw_samples.append(float(yaw))

        # Keep buffer bounded
        if len(self._row_yaw_samples) > needed:
            self._row_yaw_samples.pop(0)

        # Only initialize once we have enough samples
        if len(self._row_yaw_samples) < needed:
            return

        # Circular mean over the collected samples
        ys = np.array(self._row_yaw_samples, dtype=float)
        mean_yaw = float(np.arctan2(np.sin(ys).mean(), np.cos(ys).mean()))
        mean_yaw = wrap_angle(mean_yaw)

        # Set row yaw + basis (only here)
        self.row_yaw_est = mean_yaw

        self.t_hat = np.array([np.cos(mean_yaw), np.sin(mean_yaw)], dtype=float) * float(self.row_dir_sign)
        self.t_hat /= (np.linalg.norm(self.t_hat) + 1e-12)
        self.n_hat = np.array([-self.t_hat[1], self.t_hat[0]], dtype=float)

        # Freeze association basis immediately
        self.t_hat_assoc = self.t_hat.copy()
        self.n_hat_assoc = self.n_hat.copy()
        self.assoc_axis_locked = True

        # Convenience aliases
        self.row_axis = self.t_hat.copy()
        self.lateral_dir = self.n_hat.copy()

        self._row_axis_initialized = True

        self.get_logger().info(
            f"Row axis initialized from {len(self._row_yaw_samples)} odom samples "
            f"(needed={needed}): yaw={mean_yaw:.3f} rad, "
            f"t_hat=[{self.t_hat[0]:.3f}, {self.t_hat[1]:.3f}]"
        )

    def _anchor_template_origin_to_first_meas(self, mu_world_first: np.ndarray) -> None:
        """
        Re-anchor row_origin_xy so that template slot 0 lands exactly at mu_world_first.
        This is intended to run ONCE, early, before any meaningful landmark updates.
        """
        if self._template_origin_anchored:
            return
        if not self.use_template_map:
            return

        mu_world_first = np.asarray(mu_world_first, dtype=float).reshape(2,)

        # Slot-0 lateral offset from template (near/far)
        d0 = float(self._template_d_for_slot(0))

        # We want:
        #   mu0 = row_origin_xy + row_origin_s * t_hat_assoc + d0 * n_hat_assoc = mu_world_first
        # => row_origin_xy = mu_world_first - row_origin_s*t_hat_assoc - d0*n_hat_assoc
        old_origin = self.row_origin_xy.copy()

        self.row_origin_xy = (
            mu_world_first
            - float(self.row_origin_s) * self.t_hat_assoc
            - d0 * self.n_hat_assoc
        ).astype(float)

        # Use this as the stable association origin too
        self.assoc_origin_xy = self.row_origin_xy.copy()
        self.assoc_origin_s = float(self.row_origin_s)
        self.assoc_last_slot = 0

        # Re-seed ALL template landmarks to match the new origin.
        # Safe because we run this only once (early).
        Sigma0 = self._sigma_world_from_row_sigmas()
        for p in self.particles:
            for j, lm in p.landmarks.items():
                j = int(j)
                d_j = float(self._template_d_for_slot(j))
                s_j = float(self.row_origin_s) + float(j) * float(self.slot_spacing)
                mu_j = self.row_origin_xy + s_j * self.t_hat_assoc + d_j * self.n_hat_assoc
                lm.mu = mu_j.astype(float)
                # Keep covariance reasonable for unseen landmarks
                if lm.seen_count <= 0:
                    lm.Sigma = Sigma0.copy()

        self._template_origin_anchored = True

        self.get_logger().info(
            "Anchored template origin to first measurement: "
            f"old_origin=[{old_origin[0]:.3f},{old_origin[1]:.3f}] -> "
            f"new_origin=[{self.row_origin_xy[0]:.3f},{self.row_origin_xy[1]:.3f}]"
        )
    
    def _extend_template_slots_to(self, new_max_slot: int) -> None:
        """
        Ensure slots [0..new_max_slot] exist in every particle when use_template_map is enabled.
        Extends self.num_slots accordingly.
        """
        if (not self.use_template_map) or (not self.particles):
            return

        new_max_slot = int(new_max_slot)
        if new_max_slot < self.num_slots:
            return

        # Optional cap
        if self.max_extra_slots > 0:
            allowed_max = (self.num_slots - 1) + self.max_extra_slots
            if new_max_slot > allowed_max:
                new_max_slot = allowed_max

        Sigma0 = self._sigma_world_from_row_sigmas()

        start_j = int(self.num_slots)
        end_j = int(new_max_slot)

        for p in self.particles:
            for j in range(start_j, end_j + 1):
                if j in p.landmarks:
                    continue
                d0 = self._template_d_for_slot(j)
                s0 = self.row_origin_s + j * self.slot_spacing
                mu0 = self.row_origin_xy + s0 * self.t_hat_assoc + d0 * self.n_hat_assoc
                p.landmarks[j] = LandmarkEKF(mu=mu0.copy(), Sigma=Sigma0.copy())

        self.num_slots = end_j + 1

        self.get_logger().info(
            f"Extended template slots: num_slots={self.num_slots} (added {start_j}..{end_j})"
        )

    def _template_mu0_for_slot(self, j: int) -> np.ndarray:
        """
        Expected (template) world position for slot j, based on current row_origin_xy, row_origin_s, and row basis.
        """
        j = int(j)
        s0 = float(self.row_origin_s) + float(j) * float(self.slot_spacing)
        d0 = float(self._template_d_for_slot(j))
        return (self.row_origin_xy + s0 * self.t_hat_assoc + d0 * self.n_hat_assoc).astype(float)

    def _template_prior_alpha(self, seen_count: int) -> float:
        """
        Decay schedule: strong early, fades with seen_count.
        Using exp(-k/decay_k). You can swap to 1/(k+1) if you prefer.
        """
        k = float(max(int(seen_count), 0))
        if self.template_prior_max_seen > 0 and k >= float(self.template_prior_max_seen):
            return 0.0
        return float(np.exp(-k / float(self.template_prior_decay_k)))

    def _apply_template_prior_logw(self, lm: LandmarkEKF, j: int) -> float:
        """
        Return a log-weight increment (<= 0) based on how far landmark lm is from template position for slot j.
        Penalizes in (s,d) coordinates with sigmas.
        """
        if not (self.use_template_map and self.template_prior_enable):
            return 0.0

        alpha = self._template_prior_alpha(lm.seen_count)
        if alpha <= 0.0:
            return 0.0

        mu0 = self._template_mu0_for_slot(j)
        e = (lm.mu - mu0).astype(float)  # world error (2,)

        # Convert error into row coordinates (s along row, d lateral)
        e_s = float(np.dot(e, self.t_hat_assoc))
        e_d = float(np.dot(e, self.n_hat_assoc))

        sig_s = max(float(self.template_prior_sigma_s), 1e-6)
        sig_d = max(float(self.template_prior_sigma_d), 1e-6)

        # Gaussian penalty (drop constants; they just shift all weights equally)
        pen = 0.5 * ((e_s / sig_s) ** 2 + (e_d / sig_d) ** 2)

        return -float(self.template_prior_w) * float(alpha) * float(pen)


    # =========================================================
    #  Motion update
    # =========================================================

    def _dt_from_stamp(self, stamp) -> float:
        t_cur = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        if self._last_odom_time is None:
            self._last_odom_time = t_cur
            return 1e-3
        dt = max(t_cur - self._last_odom_time, 1e-3)
        self._last_odom_time = t_cur
        return dt
    
    def _stamp_to_sec(self, stamp) -> float:
        return float(stamp.sec) + 1e-9 * float(stamp.nanosec)

    def _meas_time_sec(self, msg: TrunkInfo) -> float:
        # Prefer TrunkInfo.stamp
        if hasattr(msg, "stamp"):
            try:
                t = self._stamp_to_sec(msg.stamp)
                if t > 1e-6:
                    return t
            except Exception:
                pass

        # Prefer header.stamp
        if hasattr(msg, "header"):
            try:
                t = self._stamp_to_sec(msg.header.stamp)
                if t > 1e-6:
                    return t
            except Exception:
                pass

        # Fallback: last odom time (bag-time consistent)
        if self._last_odom_time is not None:
            return float(self._last_odom_time)

        # Absolute last resort
        return float(self.get_clock().now().nanoseconds) * 1e-9

    def _ensure_work_buffers(self, n: int) -> None:
        """Ensure per-callback work arrays are allocated for n particles."""
        if n == self._work_N:
            return
        self._work_N = int(n)
        self._work_logw = np.zeros((n,), dtype=np.float64)
        self._work_maha = np.full((n,), np.nan, dtype=np.float64)
        self._work_logdet = np.full((n,), np.nan, dtype=np.float64)
        self._work_poses = np.zeros((n, 3), dtype=np.float64)
        self._work_cos = np.zeros((n,), dtype=np.float64)
        self._work_sin = np.zeros((n,), dtype=np.float64)
        self._work_classic_mask = np.zeros((n,), dtype=bool)
        self._work_poses_meas = np.zeros((n, 3), dtype=np.float64)

    def _prune_odom_hist(self, t_now: float) -> None:
        T = float(self.odom_history_sec)
        while self._odom_hist and (t_now - self._odom_hist[0][0]) > T:
            self._odom_hist.popleft()

    def _lookup_odom_pose(self, t_query: float) -> Optional[np.ndarray]:
        """Interpolate odom pose at time t_query. Returns [x,y,yaw] or None if out of range."""
        if len(self._odom_hist) < 2:
            return None
        t0 = float(self._odom_hist[0][0])
        tN = float(self._odom_hist[-1][0])
        if t_query < t0 or t_query > tN:
            return None

        # Linear search is fine for ~1000 samples; can be upgraded to binary search if needed.
        for k in range(len(self._odom_hist) - 1):
            ta, pa = self._odom_hist[k]
            tb, pb = self._odom_hist[k + 1]
            if ta <= t_query <= tb:
                if (tb - ta) < 1e-9:
                    return pa.copy()
                u = (t_query - ta) / (tb - ta)

                x = (1.0 - u) * float(pa[0]) + u * float(pb[0])
                y = (1.0 - u) * float(pa[1]) + u * float(pb[1])

                ya = float(pa[2]); yb = float(pb[2])
                dy = wrap_angle(yb - ya)
                yaw = wrap_angle(ya + u * dy)

                return np.array([x, y, yaw], dtype=float)
        return None

    def _ensure_initialized_from_odom(self, cur_pose: np.ndarray) -> bool:
        if self._pf_initialized:
            return False

        if not np.isfinite(self.row_origin_xy).all():
            self.row_origin_xy = cur_pose[0:2].copy()

        self.last_odom_pose = cur_pose.copy()
        self._init_particles(cur_pose)
        self._pf_initialized = True

        if self.use_template_map:
            if self.template_origin_from_first_measurement:
                self.assoc_origin_xy = None
                self.assoc_origin_s = None
                self.assoc_last_slot = None
            else:
                self.assoc_origin_xy = self.row_origin_xy.copy()
                self.assoc_origin_s  = float(self.row_origin_s)
                self.assoc_last_slot = 0

        self.get_logger().info(
            f"Initialized particles at odom pose x={cur_pose[0]:.2f}, y={cur_pose[1]:.2f}, yaw={cur_pose[2]:.2f} rad"
        )

        if self.use_template_map:
            self._publish_registry_from_best()
        self._publish_odom_from_best()
        return True

    def _motion_noise_std(self, v: float, omega: float, dt: float) -> np.ndarray:
        dx_expected = abs(v) * dt
        dtheta_expected = abs(omega) * dt

        x_floor = 1e-4
        y_floor = 1e-4
        theta_floor = 1e-4

        sigma_fwd = (self.a_trans * dx_expected) + (self.b_trans * dtheta_expected) + x_floor
        sigma_lat = (self.c_lat * dx_expected) + y_floor
        sigma_theta = (self.a_rot * dtheta_expected) + (self.b_rot * dx_expected) + theta_floor
        return np.array([sigma_fwd, sigma_lat, sigma_theta], dtype=float)

    def odom_callback(self, msg: Odometry):
        self._last_odom_msg = msg
        cur_pose = se2_from_odom(msg)
        dt = self._dt_from_stamp(msg.header.stamp)

        # Cache odom for delayed measurement alignment
        t_odom = self._stamp_to_sec(msg.header.stamp)
        self._odom_hist.append((t_odom, cur_pose.copy()))
        self._prune_odom_hist(t_odom)

        # self.get_logger().warn(f"Odom stamp: {msg.header.stamp}")

        self._maybe_init_row_axis_from_odom(cur_pose[2])

        # While bootstrapping row axis, just cache odom and do nothing else
        if not self._row_axis_initialized:
            self._boot_odom_pose = cur_pose.copy()
            return

        # Once row axis is ready, initialize PF once
        if self._ensure_initialized_from_odom(cur_pose):
            return

        prev_odom = self.last_odom_pose.copy()
        dx_w = float(cur_pose[0] - prev_odom[0])
        dy_w = float(cur_pose[1] - prev_odom[1])
        dyaw = wrap_angle(float(cur_pose[2] - prev_odom[2]))

        c0 = float(np.cos(prev_odom[2]))
        s0 = float(np.sin(prev_odom[2]))
        d_fwd = c0 * dx_w + s0 * dy_w
        d_lat = -s0 * dx_w + c0 * dy_w

        self.last_odom_pose = cur_pose.copy()

        if not self.particles:
            self._init_particles(cur_pose)
            self._publish_odom_from_best()
            return

        v = float(msg.twist.twist.linear.x)
        omega = float(msg.twist.twist.angular.z)
        noise_std = self._motion_noise_std(v, omega, dt)  # [fwd, lat, yaw]
        self._last_motion_std_body = noise_std.copy()

        # Vectorized motion update across all particles (poses live in shared array)
        poses = getattr(self, "_poses", None)
        if poses is None:
            # Backward safety: fall back to per-particle update if poses not initialized
            for p in self.particles:
                eps = noise_std * np.random.randn(3)
                df = float(d_fwd + eps[0])
                dl = float(d_lat + eps[1])
                dth = float(dyaw + eps[2])

                cy = float(np.cos(p.pose[2]))
                sy = float(np.sin(p.pose[2]))

                p.pose[0] += cy * df - sy * dl
                p.pose[1] += sy * df + cy * dl
                p.pose[2] = wrap_angle(float(p.pose[2] + dth))
        else:
            N = poses.shape[0]

            eps = np.random.randn(N, 3) * noise_std.reshape(1, 3)
            df = d_fwd + eps[:, 0]
            dl = d_lat + eps[:, 1]
            dth = dyaw + eps[:, 2]

            th = poses[:, 2]
            c = np.cos(th)
            s = np.sin(th)

            poses[:, 0] += c * df - s * dl
            poses[:, 1] += s * df + c * dl
            poses[:, 2] = (th + dth + np.pi) % (2.0 * np.pi) - np.pi

        self._publish_odom_from_best()
    
    def _init_particles(self, init_pose: np.ndarray):
        # Lock association axis on first initialization so slot coordinates don't drift mid-run
        if not getattr(self, "assoc_axis_locked", False):
            self.t_hat_assoc = self.t_hat.copy()
            self.n_hat_assoc = self.n_hat.copy()
            self.assoc_axis_locked = True

        Sigma0 = self._sigma_world_from_row_sigmas()
        landmarks_template: Dict[int, LandmarkEKF] = {}

        if self.use_template_map:
            if not np.isfinite(self.row_origin_xy).all():
                self.row_origin_xy = init_pose[0:2].copy()

            for j in range(self.num_slots):
                d0 = self._template_d_for_slot(j)
                s0 = self.row_origin_s + j * self.slot_spacing
                mu0 = self.row_origin_xy + s0 * self.t_hat_assoc + d0 * self.n_hat_assoc
                landmarks_template[j] = LandmarkEKF(mu=mu0.copy(), Sigma=Sigma0.copy())

        # Shared pose/weight arrays (fast vectorized propagation + resampling)
        self._poses = np.zeros((self.num_particles, 3), dtype=np.float64)
        self._weights = np.full((self.num_particles,), 1.0 / float(self.num_particles), dtype=np.float64)

        # Initialize poses with small jitter
        self._poses[:] = init_pose.reshape(1, 3)
        self._poses[:, 0:2] += np.random.randn(self.num_particles, 2) * 0.01
        self._poses[:, 2] = (self._poses[:, 2] + np.random.randn(self.num_particles) * 0.005 + np.pi) % (2.0 * np.pi) - np.pi

        self.particles = []
        for i in range(self.num_particles):
            lm_dict = {
                j: LandmarkEKF(mu=lm.mu.copy(), Sigma=lm.Sigma.copy(),
                            seen_count=lm.seen_count,
                            width_sum=lm.width_sum, width_count=lm.width_count)
                for j, lm in landmarks_template.items()
            }
            self.particles.append(Particle(idx=i, landmarks=lm_dict, _poses=self._poses, _weights=self._weights))

        self._normalize_weights()

    # =========================================================
    #  Measurement update
    # =========================================================

    def measurement_callback(self, msg: TrunkInfo):
        t_meas_sec = self._meas_time_sec(msg)
        if getattr(self, "debug_print_measurement_stamp", False):
            self.get_logger().info(f"Measurement stamp: {t_meas_sec:.6f}")

        if not self.particles or self.last_odom_pose is None:
            self._dbg_reject("uninitialized")
            return

        # ---- Time alignment: compute deltas but do NOT write-back rewind/fast-forward ----
        pose_odom_meas = self._lookup_odom_pose(t_meas_sec)
        if pose_odom_meas is None:
            self._dbg_reject("meas_too_old")
            return

        pose_odom_now = self.last_odom_pose.copy()
        d_meas_to_now = se2_between(pose_odom_meas, pose_odom_now)
        d_now_to_meas = se2_inverse(d_meas_to_now)

        # No pose mutation here anymore.

        # ---- Parse measurement (robot frame) ----
        x_fwd = float(msg.pose.position.x)
        y_lat = float(msg.pose.position.y)

        if not np.isfinite(y_lat) or not np.isfinite(x_fwd):
            self._dbg_reject("nan_measurement")
            return

        z = np.array([x_fwd, y_lat], dtype=float)
        r = float(np.linalg.norm(z))
        if r < 1e-3 or r > 15.0:
            self._dbg_reject("range_gate")
            return

        w_meas = float(msg.width) if np.isfinite(msg.width) else None

        # ---- Slot association at MEASUREMENT time (pass d_now_to_meas) ----
        slot_j, mu_world_approx = self._data_association_slot(z, d_now_to_meas=d_now_to_meas)
        if slot_j is None or mu_world_approx is None:
            self._dbg_reject("assoc_failed")
            return
        slot_j = int(slot_j)

        if (not self.use_template_map) and (getattr(self, "assoc_origin_s", None) is None):
            self.assoc_origin_xy = self.row_origin_xy.copy()

            # Lock origin_s so that THIS measurement lands at slot 0
            s_meas0 = float(np.dot(mu_world_approx - self.assoc_origin_xy, self.t_hat))
            self.assoc_origin_s = float(s_meas0)

            # Initialize hysteresis state
            self.assoc_last_slot = 0
            slot_j = 0

        # Template-map bounds check (should only happen if max_extra_slots cap hit)
        if self.use_template_map and slot_j >= self.num_slots:
            self._dbg_reject("assoc_oob_template")
            return

        self._update_semantic_side_votes(slot_j, msg.side)

        # ---- FastSLAM update: weights + (optional) measurement-informed proposal ----
        n_particles = len(self.particles)
        self._ensure_work_buffers(n_particles)

        particles = self.particles  # local alias

        # Work arrays (reused each callback)
        logw = self._work_logw
        logw[:n_particles].fill(0.0)

        maha_arr = self._work_maha
        maha_arr[:n_particles].fill(np.nan)

        logdet_arr = self._work_logdet
        logdet_arr[:n_particles].fill(np.nan)

        # Cache poses at "now" then compute measurement-time poses vectorized
        poses_now = self._work_poses
        for i, p in enumerate(particles):
            poses_now[i, 0] = p.pose[0]
            poses_now[i, 1] = p.pose[1]
            poses_now[i, 2] = p.pose[2]

        poses_meas = self._work_poses_meas
        se2_compose_many(poses_now[:n_particles], d_now_to_meas, out=poses_meas[:n_particles])

        ths = poses_meas[:n_particles, 2]
        np.cos(ths, out=self._work_cos[:n_particles])
        np.sin(ths, out=self._work_sin[:n_particles])

        # ------------------------------------------------------------
        # Build per-particle landmark refs and decide classic vs proposal
        # ------------------------------------------------------------
        classic_mask = self._work_classic_mask
        classic_mask[:n_particles].fill(False)

        lm_list: List[Optional[LandmarkEKF]] = [None] * n_particles

        for i, p in enumerate(particles):
            lm = p.landmarks.get(slot_j, None)

            if self.use_template_map:
                if lm is None:
                    lm_list[i] = None
                    continue
            else:
                if lm is None:
                    # init uses particle pose; that's fine (world-from-robot not time-dependent),
                    # but for correctness under lag you could use pose_meas. We'll keep it as-is.
                    lm = self._init_landmark_from_measurement_at_pose(poses_meas[i], z)
                    p.landmarks[slot_j] = lm

            # Template soft anchor prior
            if self.use_template_map and self.template_prior_enable:
                logw[i] += self._apply_template_prior_logw(lm, slot_j)

            lm_list[i] = lm

            # Classic likelihood if proposal disabled OR landmark not yet reliable
            if (not self.proposal_enable) or (lm.seen_count <= 0):
                classic_mask[i] = True

        # ------------------------------------------------------------
        # Vectorized classic scoring across particles (at measurement-time poses)
        # ------------------------------------------------------------
        idx = np.nonzero(classic_mask[:n_particles])[0]
        if idx.size > 0:
            # Gather landmark state for these particles
            mu0 = np.fromiter((float(lm_list[i].mu[0]) for i in idx), dtype=np.float64, count=int(idx.size))
            mu1 = np.fromiter((float(lm_list[i].mu[1]) for i in idx), dtype=np.float64, count=int(idx.size))

            sxx = np.fromiter((float(lm_list[i].Sigma[0, 0]) for i in idx), dtype=np.float64, count=int(idx.size))
            sxy = np.fromiter((float(lm_list[i].Sigma[0, 1]) for i in idx), dtype=np.float64, count=int(idx.size))
            syy = np.fromiter((float(lm_list[i].Sigma[1, 1]) for i in idx), dtype=np.float64, count=int(idx.size))

            px = poses_meas[idx, 0]
            py = poses_meas[idx, 1]
            c = self._work_cos[idx]
            s = self._work_sin[idx]

            dx = mu0 - px
            dy = mu1 - py

            xpred = c * dx + s * dy
            ypred = -s * dx + c * dy

            e0 = float(z[0]) - xpred
            e1 = float(z[1]) - ypred

            # S = H*Sigma*H^T + R, H = [[c,s],[-s,c]]
            q00 = c * c * sxx + 2.0 * c * s * sxy + s * s * syy
            q01 = -c * s * sxx + (c * c - s * s) * sxy + c * s * syy
            q11 = s * s * sxx - 2.0 * c * s * sxy + c * c * syy

            S00 = q00 + self.R00
            S01 = q01
            S11 = q11 + self.R11

            det = S00 * S11 - S01 * S01
            ok = det > 0.0

            inv_det = np.zeros_like(det)
            inv_det[ok] = 1.0 / det[ok]

            i00 = np.zeros_like(det)
            i11 = np.zeros_like(det)
            i01 = np.zeros_like(det)

            i00[ok] =  S11[ok] * inv_det[ok]
            i11[ok] =  S00[ok] * inv_det[ok]
            i01[ok] = -S01[ok] * inv_det[ok]

            maha = np.full_like(det, np.nan)
            maha[ok] = (e0[ok] * (i00[ok] * e0[ok] + i01[ok] * e1[ok]) +
                        e1[ok] * (i01[ok] * e0[ok] + i11[ok] * e1[ok]))

            logdet = np.full_like(det, np.nan)
            logdet[ok] = np.log(det[ok])

            maha_arr[idx] = maha
            logdet_arr[idx] = logdet
            logw[idx] += -0.5 * (maha + logdet)

            if not np.all(ok):
                self._dbg_reject("S_nonposdef")

        # ------------------------------------------------------------
        # Proposal scoring (per-particle; done at measurement-time pose)
        # After proposal, we store p.pose back at "now" via ⊕ d_meas_to_now.
        # ------------------------------------------------------------
        if self.proposal_enable:
            for i, p in enumerate(particles):
                if classic_mask[i]:
                    continue
                lm = lm_list[i]
                if lm is None:
                    continue
                if lm.seen_count <= 0:
                    continue

                pose_meas_i = poses_meas[i].copy()

                # measurement-informed proposal over pose (FastSLAM 2.0)
                z_pred, H_lm, J_x = self._predict_z_Hland_Jpose(pose_meas_i, lm.mu)

                Qeff = H_lm @ lm.Sigma @ H_lm.T + self.R
                Pbar = self._pose_prior_cov_world(float(pose_meas_i[2]))
                S = J_x @ Pbar @ J_x.T + Qeff

                innov = z - z_pred

                # Fast 2x2 inverse/logdet for S
                S00 = float(S[0, 0]); S01 = float(S[0, 1])
                S10 = float(S[1, 0]); S11 = float(S[1, 1])
                ok, i00, i01, i10, i11, logdet = inv_and_logdet_2x2(S00, S01, S10, S11)
                if not ok:
                    self._dbg_reject("S_nonposdef")
                    continue

                e0 = float(innov[0]); e1 = float(innov[1])
                maha = e0 * (i00 * e0 + i01 * e1) + e1 * (i10 * e0 + i11 * e1)

                maha_arr[i] = maha
                logdet_arr[i] = logdet
                logw[i] += -0.5 * (maha + logdet)

                Sinv = np.array([[i00, i01], [i10, i11]], dtype=float)
                K = Pbar @ J_x.T @ Sinv

                dx = (K @ innov)
                dx[2] = wrap_angle(float(dx[2]))

                # Clamp proposal correction magnitude
                dx[0] = float(np.clip(dx[0], -0.25, 0.25))
                dx[1] = float(np.clip(dx[1], -0.25, 0.25))
                dx[2] = float(np.clip(dx[2], -0.20, 0.20))

                xhat = pose_meas_i + dx
                xhat[2] = wrap_angle(float(xhat[2]))

                Phat = (np.eye(3) - K @ J_x) @ Pbar
                Phat = 0.5 * (Phat + Phat.T)

                try:
                    xnew_meas = np.random.multivariate_normal(mean=xhat, cov=Phat)
                except Exception:
                    xnew_meas = xhat

                xnew_meas[2] = wrap_angle(float(xnew_meas[2]))

                # Store particle pose back in "now" coordinates
                p.pose[:] = se2_compose_single(np.asarray(xnew_meas, dtype=float), d_meas_to_now)

        # ---- Innovation gating (median particle) ----
        skip_update = False
        maha_f = maha_arr[np.isfinite(maha_arr)]
        maha_median = float(np.median(maha_f)) if maha_f.size else float("nan")
        if self.maha_gate_median > 0.0:
            if (not np.isfinite(maha_median)) or (maha_f.size < self.maha_gate_min_particles):
                self._dbg_reject("maha_gate_insufficient")
                skip_update = True
            elif maha_median > self.maha_gate_median:
                self._dbg_reject("maha_gate")
                skip_update = True

        if not skip_update:
            # ---- Apply weights safely ----
            maxlog = float(np.max(logw))
            w = np.exp(logw - maxlog)

            for i, p in enumerate(self.particles):
                p.weight *= float(w[i])
            self._normalize_weights()

        # ---- Debug CSV row ----
        if getattr(self, "debug_csv_enable", False) and (self._dbg_csv is not None):
            neff = float(self._effective_sample_size())
            wts = np.array([p.weight for p in self.particles], dtype=float)

            def _p95(a: np.ndarray) -> float:
                a2 = a[np.isfinite(a)]
                return float(np.percentile(a2, 95)) if a2.size else float("nan")

            finite_logw = logw[np.isfinite(logw)]
            reject_snapshot = ";".join([f"{k}:{v}" for k, v in sorted(self._dbg_reject_counts.items())])

            self._dbg_csv.writerow([
                float(t_meas_sec),
                int(self.measurement_count),
                int(slot_j),
                float(z[0]),  # z_x_fwd
                float(z[1]),  # z_y_lat
                float(r),
                int(self.proposal_enable),
                neff,
                float(np.max(wts)),
                float(np.min(wts)),
                float(np.std(wts)),
                float(np.max(logw)),
                float(np.mean(finite_logw)) if finite_logw.size else float("nan"),
                float(np.std(finite_logw)) if finite_logw.size else float("nan"),
                safe_nanmean(maha_arr),
                _p95(maha_arr),
                safe_nanmean(logdet_arr),
                _p95(logdet_arr),
                reject_snapshot,
            ])

            self._dbg_csv_rows_since_flush += 1
            if self._dbg_csv_rows_since_flush >= max(int(self.debug_csv_flush_every_n), 1):
                try:
                    self._dbg_csv_fp.flush()
                except Exception:
                    pass
                self._dbg_csv_rows_since_flush = 0

        # ---- Landmark EKF update AFTER pose proposal ----
        if not skip_update:
            for i, p in enumerate(self.particles):
                lm = p.landmarks.get(slot_j, None)
                if self.use_template_map:
                    if lm is None:
                        continue
                else:
                    if lm is None:
                        lm = self._init_landmark_from_measurement_at_pose(poses_meas[i], z)
                        p.landmarks[slot_j] = lm

                # Use measurement-time pose for update.
                # For proposal particles we recompute pose_meas from the updated "now" pose.
                if self.proposal_enable and (not classic_mask[i]) and (lm is not None) and (lm.seen_count > 0):
                    pose_meas_i = se2_compose_single(p.pose, d_now_to_meas)
                else:
                    pose_meas_i = poses_meas[i]

                self._ekf_update_landmark_at_pose(pose_meas_i, lm, z, self.R)
                lm.update_width(w_meas)

        # ---- Downstream spacing anchor update ----
        if self.snap_downstream_spacing:
            origin_xy = self.assoc_origin_xy if getattr(self, "assoc_origin_xy", None) is not None else self.row_origin_xy
            s_meas_best = float(np.dot(mu_world_approx - origin_xy, self.t_hat_assoc))

            self.row_origin_s = (float(self.assoc_origin_s) if getattr(self, "assoc_origin_s", None) is not None else self.row_origin_s)

            lm_best = self._best_particle().landmarks.get(slot_j) if self._best_particle() else None
            seen = int(lm_best.seen_count) if lm_best is not None else 0

            if seen < 2:
                alpha = 0.15
                target = s_meas_best - slot_j * self.slot_spacing
                self.row_origin_s = (1.0 - alpha) * float(self.row_origin_s) + alpha * float(target)
                self._apply_downstream_spacing(anchor_j=slot_j)

        # ---- Resampling ----
        self.measurement_count += 1
        if self.resample_interval > 0 and (self.measurement_count % self.resample_interval) == 0:
            self._maybe_resample()

        # ---- Publish ----
        self._publish_registry_from_best()
        self._publish_odom_from_best()

    def _world_from_robot(self, pose: np.ndarray, z_robot: np.ndarray) -> np.ndarray:
        rx, ry, th = float(pose[0]), float(pose[1]), float(pose[2])
        c, s = float(np.cos(th)), float(np.sin(th))

        x_fwd = float(z_robot[0])
        y_lat = float(z_robot[1])

        dx_w = c * x_fwd - s * y_lat
        dy_w = s * x_fwd + c * y_lat
        return np.array([rx + dx_w, ry + dy_w], dtype=float)

    def _data_association_slot(self, z_robot: np.ndarray, d_now_to_meas: Optional[np.ndarray] = None) -> Tuple[Optional[int], Optional[np.ndarray]]:
        """
        Returns:
        slot_j (int) or None if association fails
        mu_world_approx: the approximate world position of this measurement (2,)
        """

        if not self.particles:
            return None, None
        if self.last_odom_pose is None:
            return None, None

        # ---- Pose reference for association ----
        if self.proposal_enable:
            pb = self._best_particle()
            if pb is None:
                return None, None
            pose_ref = pb.pose.copy()

        else:
            # ---- Use weighted-mean pose for stable association ----
            w = np.array([p.weight for p in self.particles], dtype=float)
            sw = float(np.sum(w))
            if sw <= 1e-12:
                w[:] = 1.0 / float(len(w))
            else:
                w /= sw

            self._ensure_work_buffers(len(self.particles))
            poses = self._work_poses[:len(self.particles)]
            for i, p_i in enumerate(self.particles):
                poses[i, 0] = p_i.pose[0]
                poses[i, 1] = p_i.pose[1]
                poses[i, 2] = p_i.pose[2]

            xy = np.sum(poses[:, 0:2] * w[:, None], axis=0)
            c = float(np.sum(np.cos(poses[:, 2]) * w))
            s = float(np.sum(np.sin(poses[:, 2]) * w))
            yaw = float(np.arctan2(s, c))

            pose_ref = np.array([xy[0], xy[1], yaw], dtype=float)

        # If we're doing delayed measurement alignment, evaluate association at measurement time.
        if d_now_to_meas is not None:
            pose_ref = se2_compose_single(pose_ref, d_now_to_meas)

        mu_world_approx = self._world_from_robot(pose_ref, z_robot)  # (2,)

        # ---- Stable association origin (xy anchor) ----
        # Set once from the first time you initialized from odom.
        if getattr(self, "assoc_origin_xy", None) is None:
            # fall back to row_origin_xy if present
            if hasattr(self, "row_origin_xy") and np.isfinite(self.row_origin_xy).all():
                self.assoc_origin_xy = self.row_origin_xy.copy()
            else:
                # last resort: use current reference pose xy
                self.assoc_origin_xy = pose_ref[:2].copy()

        origin_xy = self.assoc_origin_xy
        if origin_xy is None or (not np.isfinite(origin_xy).all()):
            return None, None

        # ---- Row coordinate for this measurement ----
        s_meas = float(np.dot(mu_world_approx - origin_xy, self.t_hat_assoc))

        if getattr(self, "assoc_origin_s", None) is None:
            # If template map is on, we normally expect assoc_origin_s to already be set.
            # But if we're anchoring template origin from the first measurement, we initialize here.
            if self.use_template_map and self.template_origin_from_first_measurement:
                # Anchor template origin so slot 0 lands on THIS first measurement
                self._anchor_template_origin_to_first_meas(mu_world_approx)
                return 0, mu_world_approx

            if self.use_template_map:
                return None, None

            self.assoc_origin_s = float(s_meas)
            self.assoc_last_slot = 0
            return 0, mu_world_approx

        origin_s = float(self.assoc_origin_s)

        # ---- Candidate slot index (avoid round() boundary flips) ----
        u = (s_meas - origin_s) / float(self.slot_spacing)
        j_cand = int(np.floor(u + 0.5))  # "round" but via floor to keep consistent with hysteresis

        # ---- Hysteresis / monotonic-ish clamp (debug-friendly) ----
        last = getattr(self, "assoc_last_slot", None)
        if last is None:
            j_idx = j_cand
        else:
            # allow small backward correction, avoid large back-jumps
            j_idx = j_cand
            if j_idx < last - self.max_back_assoc:
                j_idx = last - self.max_back_assoc
            if j_idx > last + self.max_fwd_assoc:
                j_idx = last + self.max_fwd_assoc

        # ---- Gate in s ----
        s_slot = origin_s + float(j_idx) * float(self.slot_spacing)

        if abs(s_meas - s_slot) > float(self.slot_s_gate):
            return None, None

        # ---- Template bounds (optionally extend) ----
        if self.use_template_map:
            if j_idx < 0:
                return None, None

            if j_idx >= self.num_slots:
                if self.exceed_slot_num:
                    self._extend_template_slots_to(j_idx)
                    # If we hit the cap, extension may not reach j_idx
                    if j_idx >= self.num_slots:
                        return None, None
                else:
                    return None, None

        self.assoc_last_slot = int(j_idx)
        return int(j_idx), mu_world_approx

    # ------------- Landmark EKF update + proposal helpers -------------

    def _predict_z_and_H(self, pose: np.ndarray, mu_lm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x, y, th = float(pose[0]), float(pose[1]), float(pose[2])
        c, s = float(np.cos(th)), float(np.sin(th))

        dx = float(mu_lm[0] - x)
        dy = float(mu_lm[1] - y)

        x_fwd = c * dx + s * dy
        y_lat = -s * dx + c * dy

        # z = [x_fwd, y_lat]
        z_pred = np.array([x_fwd, y_lat], dtype=float)

        # H_lm = d z / d [lm_x, lm_y]
        H = np.array([[ c,  s],
                    [-s,  c]], dtype=float)
        return z_pred, H

    def _predict_z_Hland_Jpose(self, pose: np.ndarray, mu_lm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x, y, th = float(pose[0]), float(pose[1]), float(pose[2])
        c, s = float(np.cos(th)), float(np.sin(th))

        dx = float(mu_lm[0] - x)
        dy = float(mu_lm[1] - y)

        x_fwd = c * dx + s * dy
        y_lat = -s * dx + c * dy

        # z = [x_fwd, y_lat]
        z_pred = np.array([x_fwd, y_lat], dtype=float)

        # d z / d landmark
        H_lm = np.array([[ c,  s],
                        [-s,  c]], dtype=float)

        J_x = np.array([[-c, -s,  y_lat],
                        [ s, -c, -x_fwd]], dtype=float)

        return z_pred, H_lm, J_x

    def _pose_prior_cov_world(self, yaw: float) -> np.ndarray:
        std = np.array(self._last_motion_std_body, dtype=float).copy()

        std_xy_floor = float(self.proposal_min_pose_std_xy)
        std_yaw_floor = float(self.proposal_min_pose_std_yaw)
        std[0] = max(float(std[0]), std_xy_floor)
        std[1] = max(float(std[1]), std_xy_floor)
        std[2] = max(float(std[2]), std_yaw_floor)

        Pf = float(std[0] * std[0])
        Pl = float(std[1] * std[1])
        Pth = float(std[2] * std[2])

        c, s = float(np.cos(yaw)), float(np.sin(yaw))
        R2 = np.array([[c, -s],
                       [s,  c]], dtype=float)

        Pxy = R2 @ np.diag([Pf, Pl]) @ R2.T
        P = np.zeros((3, 3), dtype=float)
        P[0:2, 0:2] = Pxy
        P[2, 2] = Pth
        return P

    def _init_landmark_from_measurement(self, p: Particle, z: np.ndarray) -> LandmarkEKF:
        mu_init = self._world_from_robot(p.pose, z)
        Sigma_init = np.diag([0.5 ** 2, 0.5 ** 2])
        return LandmarkEKF(mu=mu_init, Sigma=Sigma_init)
    
    def _init_landmark_from_measurement_at_pose(self, pose: np.ndarray, z: np.ndarray) -> LandmarkEKF:
        mu_init = self._world_from_robot(pose, z)
        Sigma_init = np.diag([0.5 ** 2, 0.5 ** 2])
        return LandmarkEKF(mu=mu_init, Sigma=Sigma_init)

    def _ekf_update_landmark(self, p: Particle, lm: LandmarkEKF, z: np.ndarray, R_meas: np.ndarray):
        z_pred, H = self._predict_z_and_H(p.pose, lm.mu)

        S = H @ lm.Sigma @ H.T + R_meas

        S00 = float(S[0, 0]); S01 = float(S[0, 1])
        S10 = float(S[1, 0]); S11 = float(S[1, 1])
        ok, i00, i01, i10, i11, _logdet = inv_and_logdet_2x2(S00, S01, S10, S11)
        if not ok:
            return

        Sinv = np.array([[i00, i01], [i10, i11]], dtype=float)
        K = lm.Sigma @ H.T @ Sinv

        innov = z - z_pred
        lm.mu = lm.mu + K @ innov
        lm.Sigma = (np.eye(2) - K @ H) @ lm.Sigma

        # --- Covariance floor to avoid overconfident landmarks (helps prevent huge Mahalanobis spikes) ---
        if getattr(self, "_lm_sigma_floor_var_x", 0.0) > 0.0:
            lm.Sigma[0, 0] = max(float(lm.Sigma[0, 0]), float(self._lm_sigma_floor_var_x))
        if getattr(self, "_lm_sigma_floor_var_y", 0.0) > 0.0:
            lm.Sigma[1, 1] = max(float(lm.Sigma[1, 1]), float(self._lm_sigma_floor_var_y))

        lm.Sigma = 0.5 * (lm.Sigma + lm.Sigma.T)
        lm.seen_count += 1

    def _ekf_update_landmark_at_pose(self, pose: np.ndarray, lm: LandmarkEKF, z: np.ndarray, R_meas: np.ndarray):
        z_pred, H = self._predict_z_and_H(pose, lm.mu)

        S = H @ lm.Sigma @ H.T + R_meas

        S00 = float(S[0, 0]); S01 = float(S[0, 1])
        S10 = float(S[1, 0]); S11 = float(S[1, 1])
        ok, i00, i01, i10, i11, _logdet = inv_and_logdet_2x2(S00, S01, S10, S11)
        if not ok:
            return

        Sinv = np.array([[i00, i01], [i10, i11]], dtype=float)
        K = lm.Sigma @ H.T @ Sinv

        innov = z - z_pred
        lm.mu = lm.mu + K @ innov
        lm.Sigma = (np.eye(2) - K @ H) @ lm.Sigma

        if getattr(self, "_lm_sigma_floor_var_x", 0.0) > 0.0:
            lm.Sigma[0, 0] = max(float(lm.Sigma[0, 0]), float(self._lm_sigma_floor_var_x))
        if getattr(self, "_lm_sigma_floor_var_y", 0.0) > 0.0:
            lm.Sigma[1, 1] = max(float(lm.Sigma[1, 1]), float(self._lm_sigma_floor_var_y))

        lm.Sigma = 0.5 * (lm.Sigma + lm.Sigma.T)
        lm.seen_count += 1

    # =========================================================
    #  Snapping logic
    # =========================================================

    def _apply_downstream_spacing(self, anchor_j: int):
        for p in self.particles:
            for k, lm in p.landmarks.items():
                if k <= anchor_j:
                    continue
                if self.downstream_snap_mode == "unseen_only" and lm.seen_count > 0:
                    continue

                s_k = self.row_origin_s + int(k) * self.slot_spacing
                d_k = self._template_d_for_slot(int(k)) if lm.seen_count == 0 else float(np.dot(lm.mu - self.row_origin_xy, self.n_hat_assoc))
                lm.mu = self.row_origin_xy + s_k * self.t_hat_assoc + d_k * self.n_hat_assoc

    # =========================================================
    #  Resampling
    # =========================================================

    def _normalize_weights(self):
        if not self.particles:
            return
        w = getattr(self, "_weights", None)
        if w is None:
            # Create shared weights if missing (shouldn't happen in normal use)
            w = np.array([float(p.weight) for p in self.particles], dtype=np.float64)
            self._weights = w
            for i, p in enumerate(self.particles):
                p._weights = w
                p.idx = i

        s = float(np.sum(w))
        if s < 1e-12 or (not np.isfinite(s)):
            w.fill(1.0 / float(len(self.particles)))
            return
        w *= (1.0 / s)

    def _effective_sample_size(self) -> float:
        if not self.particles:
            return 0.0
        w = getattr(self, "_weights", None)
        if w is None:
            w = np.array([float(p.weight) for p in self.particles], dtype=np.float64)
            self._weights = w
        return 1.0 / float(np.sum(w * w) + 1e-12)

    def _maybe_resample(self):
        if not self.particles:
            return
        N = len(self.particles)

        neff = self._effective_sample_size()
        if (neff / float(N)) > self.neff_ratio_threshold:
            return

        poses = getattr(self, "_poses", None)
        w = getattr(self, "_weights", None)

        # Create shared pose/weight arrays if missing (shouldn't happen in normal use)
        if poses is None:
            poses = np.zeros((N, 3), dtype=np.float64)
            for i, p in enumerate(self.particles):
                poses[i, :] = np.array(p.pose, dtype=np.float64)
            self._poses = poses

        if w is None:
            w = np.array([float(p.weight) for p in self.particles], dtype=np.float64)
            self._weights = w

        # Systematic resample
        cdf = np.cumsum(w)
        step = 1.0 / float(N)
        start = np.random.uniform(0.0, step)
        u = start + step * np.arange(N)
        parents = np.searchsorted(cdf, u)

        # Resample poses via indexed copy (fast)
        poses_new = poses[parents, :].copy()
        poses[:, :] = poses_new

        # Reset weights uniform
        w.fill(1.0 / float(N))

        # Resample landmarks (deep copy)
        old_particles = self.particles
        new_particles: List[Particle] = []
        for i in range(N):
            src = old_particles[int(parents[i])]
            lm_new = {
                j: LandmarkEKF(mu=lm.mu.copy(), Sigma=lm.Sigma.copy(),
                            seen_count=lm.seen_count,
                            width_sum=lm.width_sum, width_count=lm.width_count)
                for j, lm in src.landmarks.items()
            }
            new_particles.append(Particle(idx=i, landmarks=lm_new, _poses=poses, _weights=w))

        self.particles = new_particles

    # =========================================================
    #  Publishing
    # =========================================================

    def _publish_odom_from_best(self):
        p_best = self._best_particle()
        if p_best is None:
            return

        odom_out = Odometry()

        if self._last_odom_msg is not None:
            odom_out.header.stamp = self._last_odom_msg.header.stamp
            odom_out.twist = self._last_odom_msg.twist
        else:
            odom_out.header.stamp = self.get_clock().now().to_msg()

        odom_out.header.frame_id = "map"
        odom_out.child_frame_id = "amiga__base"

        odom_out.pose.pose.position.x = float(p_best.pose[0])
        odom_out.pose.pose.position.y = float(p_best.pose[1])
        odom_out.pose.pose.position.z = 0.0

        q = quat_xyzw_from_yaw(float(p_best.pose[2]))
        odom_out.pose.pose.orientation.x = float(q[0])
        odom_out.pose.pose.orientation.y = float(q[1])
        odom_out.pose.pose.orientation.z = float(q[2])
        odom_out.pose.pose.orientation.w = float(q[3])

        self.odom_best_pub.publish(odom_out)

    def _publish_registry_from_best(self):
        p_best = self._best_particle()
        if p_best is None or not p_best.landmarks:
            return

        items = sorted(p_best.landmarks.items(), key=lambda kv: kv[0])

        row_yaw = self.row_yaw_est if self.row_yaw_est is not None else self._estimate_row_yaw_from_best()
        row_q = quat_xyzw_from_yaw(row_yaw) if (row_yaw is not None) else np.array([0.0, 0.0, 0.0, 1.0])

        if self.side_mode == "fixed":
            side_by_j = {int(j): self._side_for_index(int(j)) for j, _ in items}
        elif self.side_mode == "semantic":
            side_by_j = {int(j): self._semantic_side_for_slot(int(j)) for j, _ in items}
        elif self.side_mode == "geometry":
            side_by_j = self._geometry_sides_for_items(items, p_best.pose[0:2])
        else:
            side_by_j = {int(j): self._side_for_index(int(j)) for j, _ in items}

        trunks: List[TrunkInfo] = []
        for j, lm in items:
            # --- Template prior "strength" knob: do not publish unseen template slots ---
            if self.use_template_map and (not self.publish_unseen_template_landmarks):
                if lm.seen_count < self.min_seen_count_to_publish:
                    continue

            ti = TrunkInfo()
            if self._last_odom_msg is not None:
                ti.stamp = self._last_odom_msg.header.stamp
            else:
                ti.stamp = self.get_clock().now().to_msg()
            ti.pose.position.x = float(lm.mu[0])
            ti.pose.position.y = float(lm.mu[1])
            ti.pose.position.z = 0.0

            if self.use_global_row_yaw and row_yaw is not None:
                ti.pose.orientation.x = float(row_q[0])
                ti.pose.orientation.y = float(row_q[1])
                ti.pose.orientation.z = float(row_q[2])
                ti.pose.orientation.w = float(row_q[3])
            else:
                ti.pose.orientation.w = 1.0

            side = side_by_j[int(j)]
            if side not in ("near", "far"):
                side = self._side_for_index(int(j))
            ti.side = side

            ti.width = lm.width_mean if np.isfinite(lm.width_mean) else float("nan")
            trunks.append(ti)

        reg = TrunkRegistry()
        reg.trunks = trunks
        self.registry_pub.publish(reg)

    def _geometry_sides_for_items(self, items, robot_xy: np.ndarray) -> Dict[int, str]:
        d_lats = np.array([float(np.dot(self.n_hat, lm.mu)) for _, lm in items], dtype=float)
        if d_lats.size < 2:
            return {int(j): "near" for j, _ in items}

        center_lat = float(np.mean(d_lats))
        d_robot = float(np.dot(self.n_hat, robot_xy))
        robot_side_sign = 1.0 if abs(d_robot - center_lat) < 1e-6 else np.sign(d_robot - center_lat)

        out: Dict[int, str] = {}
        for (j, _), d_lat in zip(items, d_lats):
            offset = float(d_lat - center_lat)
            same_side = (np.sign(offset) == robot_side_sign) if abs(offset) >= 1e-6 else True
            out[int(j)] = "near" if same_side else "far"
        return out

    def _estimate_row_yaw_from_best(self) -> Optional[float]:
        p_best = self._best_particle()
        if p_best is None:
            return None

        pts = np.array([lm.mu for _, lm in sorted(p_best.landmarks.items())], dtype=float)
        if pts.shape[0] < 2:
            return None

        mu = pts.mean(axis=0)
        X = pts - mu
        C = (X.T @ X) / max(pts.shape[0] - 1, 1)

        eigvals, eigvecs = np.linalg.eigh(C)
        v = eigvecs[:, int(np.argmax(eigvals))]

        yaw = float(np.arctan2(v[1], v[0]))

        if self.row_dir_sign == -1:
            if np.cos(yaw) > 0:
                yaw = wrap_angle(yaw + np.pi)
        else:
            if np.cos(yaw) < 0:
                yaw = wrap_angle(yaw + np.pi)

        return yaw


def main(args=None):
    rclpy.init(args=args)
    node = RowFastSLAMNode()

    executor = MultiThreadedExecutor(num_threads=4)

    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down RowFastSLAMNode...")
    finally:
        try:
            node._dbg_close()
        except Exception:
            pass
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except rclpy.exceptions.RCLError:
                pass


if __name__ == "__main__":
    main()