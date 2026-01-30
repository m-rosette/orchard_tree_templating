#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

import csv
import os
from collections import defaultdict

import numpy as np
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry

from tree_template_interfaces.msg import TrunkInfo, TrunkRegistry


# ============= Utility / math helpers =============

def wrap_angle(theta: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (theta + np.pi) % (2.0 * np.pi) - np.pi

def se2_from_odom(msg: Odometry) -> np.ndarray:
    """Extract [x, y, yaw] from nav_msgs/Odometry"""
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    yaw = R.from_quat([q.x, q.y, q.z, q.w]).as_euler("zyx", degrees=False)[0]
    return np.array([float(p.x), float(p.y), float(yaw)], dtype=float)

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
    pose: np.ndarray                      # (3,) [x, y, yaw] in world frame
    landmarks: Dict[int, LandmarkEKF]     # slot index -> landmark EKF
    weight: float = 1.0


# ============= FastSLAM Node =============

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
        self.declare_parameter("row_datum_pose_topic", "row_datum_pose")
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

        # Measurement-informed pose proposal (FastSLAM 2.0)
        self.declare_parameter("proposal.enable", False)
        self.declare_parameter("proposal.min_pose_std_xy", 0.08)     # m
        self.declare_parameter("proposal.min_pose_std_yaw", 0.06)    # rad

        # Debug
        self.declare_parameter("debug.log_every_n_meas", 0)          # 0 disables
        self._dbg_meas_counter = 0

        self.declare_parameter("init_row_yaw_avg_count", 200)  # number of initial odom yaws to average

        # Template prior (soft anchor toward template slot position)
        self.declare_parameter("template_prior.enable", True)
        self.declare_parameter("template_prior.sigma_s", 0.8)     # m along-row
        self.declare_parameter("template_prior.sigma_d", 0.6)     # m lateral
        self.declare_parameter("template_prior.decay_k", 3.0)     # hits; larger = prior lasts longer
        self.declare_parameter("template_prior.max_seen", 8)      # stop applying after this many hits
        self.declare_parameter("template_prior.w", 1.0)           # weight multiplier

        self.init_row_yaw_avg_count = max(1, int(self.get_parameter("init_row_yaw_avg_count").value))

        # Track whether row origin came from odom (NaN params) and yaw bootstrapping state
        self._row_origin_from_odom = False
        self._yaw_bootstrap_done = False
        self._yaw_bootstrap_n = 0
        self._yaw_bootstrap_cos = 0.0
        self._yaw_bootstrap_sin = 0.0

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

        self.row_datum_pose_topic = str(self.get_parameter("row_datum_pose_topic").value)
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
                float(self.get_parameter("meas_std_y_lat").value),
                float(self.get_parameter("meas_std_x_fwd").value),
            ],
            dtype=float,
        )
        self.R = np.diag(self.meas_std ** 2)

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

        self.proposal_enable = bool(self.get_parameter("proposal.enable").value)
        self.proposal_min_pose_std_xy = float(self.get_parameter("proposal.min_pose_std_xy").value)
        self.proposal_min_pose_std_yaw = float(self.get_parameter("proposal.min_pose_std_yaw").value)

        self.debug_log_every_n_meas = int(self.get_parameter("debug.log_every_n_meas").value)

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
        self.row_yaw_alpha = 0.2

        self.t_hat = np.array([float(self.row_dir_sign), 0.0], dtype=float)
        self.t_hat /= (np.linalg.norm(self.t_hat) + 1e-12)
        self.n_hat = np.array([-self.t_hat[1], self.t_hat[0]], dtype=float)

        self.row_axis = self.t_hat.copy()
        self.lateral_dir = self.n_hat.copy()

        # ---------- State ----------
        self.particles: List[Particle] = []
        self.last_odom_pose: Optional[np.ndarray] = None
        self._last_odom_time: Optional[float] = None
        self.measurement_count = 0
        self._slot_side_votes: Dict[int, Dict[str, int]] = {}
        self._last_odom_msg: Optional[Odometry] = None

        # Store last motion noise (body-frame) from odom updates
        self._last_motion_std_body = np.array([0.2, 0.2, 0.1], dtype=float)

        # ---------- Pub/Sub ----------
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.registry_pub = self.create_publisher(TrunkRegistry, self.registry_topic, qos)
        self.odom_best_pub = self.create_publisher(Odometry, self.odom_best_topic, qos)

        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 50)
        self.meas_sub = self.create_subscription(TrunkInfo, self.measurement_topic, self.measurement_callback, 50)
        self.row_datum_sub = self.create_subscription(PoseStamped, self.row_datum_pose_topic, self.row_datum_pose_callback, 10)

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
                "z_y_lat",
                "z_x_fwd",
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
                "rowprior_pen_mean",
                "rowprior_pen_p95",
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
        t = self.t_hat.reshape(2, 1)
        n = self.n_hat.reshape(2, 1)
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

    def _set_row_axis_from_yaw(self, th: float):
        """Update row basis vectors from a yaw angle."""
        th = float(th)
        self.row_yaw_est = th

        self.t_hat = np.array([np.cos(th), np.sin(th)], dtype=float) * float(self.row_dir_sign)
        self.t_hat /= (np.linalg.norm(self.t_hat) + 1e-12)
        self.n_hat = np.array([-self.t_hat[1], self.t_hat[0]], dtype=float)

        self.row_axis = self.t_hat.copy()
        self.lateral_dir = self.n_hat.copy()

    def _bootstrap_row_yaw_from_odom(self, yaw_meas: float):
        """
        If the row origin is taken from odom (row_origin_x/y are NaN), set row yaw as the
        circular mean of the first N odom yaw measurements.
        """
        if not self.use_global_row_yaw:
            return
        if not self._row_origin_from_odom:
            return
        if self._yaw_bootstrap_done:
            return

        # accumulate circular mean stats
        y = float(yaw_meas)
        self._yaw_bootstrap_cos += float(np.cos(y))
        self._yaw_bootstrap_sin += float(np.sin(y))
        self._yaw_bootstrap_n += 1

        # compute current mean (even before we have all N, so it stabilizes smoothly)
        th_mean = float(np.arctan2(self._yaw_bootstrap_sin, self._yaw_bootstrap_cos))
        self._set_row_axis_from_yaw(th_mean)

        if self._yaw_bootstrap_n >= self.init_row_yaw_avg_count:
            self._yaw_bootstrap_done = True
            self.get_logger().info(
                f"Row yaw bootstrap complete using first {self._yaw_bootstrap_n} odom yaws: yaw={th_mean:.3f} rad"
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
                mu0 = self.row_origin_xy + s0 * self.t_hat + d0 * self.n_hat
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
        return (self.row_origin_xy + s0 * self.t_hat + d0 * self.n_hat).astype(float)

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
        e_s = float(np.dot(e, self.t_hat))
        e_d = float(np.dot(e, self.n_hat))

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

    def _ensure_initialized_from_odom(self, cur_pose: np.ndarray) -> bool:
        if self.last_odom_pose is not None:
            return False

        # --- Determine whether origin comes from odom (NaN params) ---
        if not np.isfinite(self.row_origin_xy).all():
            self.row_origin_xy = cur_pose[0:2].copy()
            self._row_origin_from_odom = True
        else:
            self._row_origin_from_odom = False

        # --- Initialize/Bootstrap row axis (yaw) ---
        if self.row_yaw_est is None:
            # Default: use initial odom yaw
            yaw_init = float(cur_pose[2])

            # If we used odom as origin and we want a global row yaw, start the circular-mean bootstrap
            if self.use_global_row_yaw and self._row_origin_from_odom:
                # initialize bootstrap accumulators on the first call
                self._yaw_bootstrap_cos = float(np.cos(yaw_init))
                self._yaw_bootstrap_sin = float(np.sin(yaw_init))
                self._yaw_bootstrap_n = 1
                self._yaw_bootstrap_done = (self._yaw_bootstrap_n >= int(self.init_row_yaw_avg_count))

                th = float(np.arctan2(self._yaw_bootstrap_sin, self._yaw_bootstrap_cos))
                self.row_yaw_est = th
            else:
                th = yaw_init
                self.row_yaw_est = th

            # Build row basis vectors from th
            self.t_hat = np.array([np.cos(th), np.sin(th)], dtype=float) * float(self.row_dir_sign)
            self.t_hat /= (np.linalg.norm(self.t_hat) + 1e-12)
            self.n_hat = np.array([-self.t_hat[1], self.t_hat[0]], dtype=float)

            self.row_axis = self.t_hat.copy()
            self.lateral_dir = self.n_hat.copy()

            if self.use_global_row_yaw and self._row_origin_from_odom:
                self.get_logger().info(
                    f"Initialized row axis from odom yaw bootstrap (n={self._yaw_bootstrap_n}/{int(self.init_row_yaw_avg_count)}): "
                    f"yaw={th:.3f} rad, t_hat=[{self.t_hat[0]:.3f}, {self.t_hat[1]:.3f}]"
                )
            else:
                self.get_logger().info(
                    f"Initialized row axis from initial odom yaw: yaw={th:.3f} rad, "
                    f"t_hat=[{self.t_hat[0]:.3f}, {self.t_hat[1]:.3f}]"
                )

        # --- Particle initialization ---
        self.last_odom_pose = cur_pose.copy()
        self._init_particles(cur_pose)

        # --- Template association bookkeeping ---
        if self.use_template_map:
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

        # keep refining the initial yaw mean for the first N odom samples (only if origin-from-odom)
        self._bootstrap_row_yaw_from_odom(float(cur_pose[2]))

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

        self._publish_odom_from_best()

    def _init_particles(self, init_pose: np.ndarray):
        Sigma0 = self._sigma_world_from_row_sigmas()
        landmarks_template: Dict[int, LandmarkEKF] = {}

        if self.use_template_map:
            if not np.isfinite(self.row_origin_xy).all():
                self.row_origin_xy = init_pose[0:2].copy()

            for j in range(self.num_slots):
                d0 = self._template_d_for_slot(j)
                s0 = self.row_origin_s + j * self.slot_spacing
                mu0 = self.row_origin_xy + s0 * self.t_hat + d0 * self.n_hat
                landmarks_template[j] = LandmarkEKF(mu=mu0.copy(), Sigma=Sigma0.copy())

        self.particles = []
        for _ in range(self.num_particles):
            pose = init_pose.copy()
            pose[0:2] += np.random.randn(2) * 0.01
            pose[2] = wrap_angle(pose[2] + np.random.randn() * 0.005)

            lm_dict = {j: LandmarkEKF(mu=lm.mu.copy(), Sigma=lm.Sigma.copy())
                       for j, lm in landmarks_template.items()}

            self.particles.append(Particle(pose=pose, landmarks=lm_dict, weight=1.0))

        self._normalize_weights()

    # =========================================================
    #  Measurement update
    # =========================================================

    def measurement_callback(self, msg: TrunkInfo):
        if not self.particles or self.last_odom_pose is None:
            self._dbg_reject("uninitialized")
            return

        # ---- Parse measurement (robot frame) ----
        y_lat = float(msg.pose.position.y)
        x_fwd = float(msg.pose.position.x)

        if not np.isfinite(y_lat) or not np.isfinite(x_fwd):
            self._dbg_reject("nan_measurement")
            return
        # if x_fwd < 0.0:
        #     self._dbg_reject("x_fwd_negative")
        #     return

        z = np.array([y_lat, x_fwd], dtype=float)
        r = float(np.linalg.norm(z))
        if r < 1e-3 or r > 15.0:
            self._dbg_reject("range_gate")
            return

        w_meas = float(msg.width) if np.isfinite(msg.width) else None

        # ---- Slot association (using improved _data_association_slot) ----
        slot_j, mu_world_approx = self._data_association_slot(z)
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

        # Template-map bounds check
        if self.use_template_map and slot_j >= self.num_slots:
            # Should only happen if we hit a max_extra_slots cap
            self._dbg_reject("assoc_oob_template")
            return

        self._update_semantic_side_votes(slot_j, msg.side)

        # ---- FastSLAM update: weights + (optional) measurement-informed proposal ----
        n_particles = len(self.particles)

        # Key: start logw at 0 so "no update" means neutral (and you avoid systematic weight collapse)
        logw = np.zeros(n_particles, dtype=float)

        maha_arr = np.full(n_particles, np.nan, dtype=float)
        logdet_arr = np.full(n_particles, np.nan, dtype=float)

        for i, p in enumerate(self.particles):
            lm = p.landmarks.get(slot_j, None)

            if self.use_template_map:
                if lm is None:
                    continue
            else:
                if lm is None:
                    lm = self._init_landmark_from_measurement(p, z)
                    p.landmarks[slot_j] = lm

            # --- Template soft anchor prior (encourages slot j to stay near its template position early) ---
            if self.use_template_map and self.template_prior_enable:
                logw[i] += self._apply_template_prior_logw(lm, slot_j)

            # Classic likelihood if proposal disabled OR landmark not yet reliable
            if (not self.proposal_enable) or (lm.seen_count <= 0):
                z_pred, H_lm, _Jx = self._predict_z_Hland_Jpose(p.pose, lm.mu)
                S = H_lm @ lm.Sigma @ H_lm.T + self.R
                innov = z - z_pred

                try:
                    Sinv = np.linalg.inv(S)
                except np.linalg.LinAlgError:
                    self._dbg_reject("S_singular")
                    continue

                maha = float(innov.T @ Sinv @ innov)
                sign, logdet = np.linalg.slogdet(S)
                if sign <= 0:
                    self._dbg_reject("S_nonposdef")
                    continue

                maha_arr[i] = maha
                logdet_arr[i] = logdet
                logw[i] += -0.5 * (maha + logdet)

            else:
                # measurement-informed proposal over pose (FastSLAM 2.0)
                z_pred, H_lm, J_x = self._predict_z_Hland_Jpose(p.pose, lm.mu)

                Qeff = H_lm @ lm.Sigma @ H_lm.T + self.R
                Pbar = self._pose_prior_cov_world(float(p.pose[2]))
                S = J_x @ Pbar @ J_x.T + Qeff

                innov = z - z_pred
                try:
                    Sinv = np.linalg.inv(S)
                except np.linalg.LinAlgError:
                    self._dbg_reject("S_singular")
                    continue

                maha = float(innov.T @ Sinv @ innov)
                sign, logdet = np.linalg.slogdet(S)
                if sign <= 0:
                    self._dbg_reject("S_nonposdef")
                    continue

                maha_arr[i] = maha
                logdet_arr[i] = logdet
                logw[i] += -0.5 * (maha + logdet)

                K = Pbar @ J_x.T @ Sinv

                # --- proposal correction (clamped) ---
                dx = (K @ innov)
                dx[2] = wrap_angle(float(dx[2]))

                # Clamp proposal correction magnitude (prevents teleport-to-wrong-slot collapse)
                dx[0] = float(np.clip(dx[0], -0.25, 0.25))   # meters
                dx[1] = float(np.clip(dx[1], -0.25, 0.25))   # meters
                dx[2] = float(np.clip(dx[2], -0.20, 0.20))   # radians (~11 deg)

                xhat = p.pose + dx
                xhat[2] = wrap_angle(float(xhat[2]))

                Phat = (np.eye(3) - K @ J_x) @ Pbar
                Phat = 0.5 * (Phat + Phat.T)

                try:
                    xnew = np.random.multivariate_normal(mean=xhat, cov=Phat)
                except Exception:
                    xnew = xhat

                xnew[2] = wrap_angle(float(xnew[2]))
                p.pose = xnew

        # ---- Apply weights safely (log-sum-exp style) ----
        maxlog = float(np.max(logw))
        w = np.exp(logw - maxlog)

        for i, p in enumerate(self.particles):
            p.weight *= float(w[i])
        self._normalize_weights()

        # ---- Debug CSV row ----
        if getattr(self, "debug_csv_enable", False) and (self._dbg_csv is not None):
            if getattr(self, "_last_odom_msg", None) is not None:
                st = self._last_odom_msg.header.stamp
                t_sec = float(st.sec) + 1e-9 * float(st.nanosec)
            else:
                t_sec = float(self.get_clock().now().nanoseconds) * 1e-9

            neff = float(self._effective_sample_size())
            wts = np.array([p.weight for p in self.particles], dtype=float)

            def _p95(a: np.ndarray) -> float:
                a2 = a[np.isfinite(a)]
                return float(np.percentile(a2, 95)) if a2.size else float("nan")

            finite_logw = logw[np.isfinite(logw)]
            reject_snapshot = ";".join([f"{k}:{v}" for k, v in sorted(self._dbg_reject_counts.items())])

            self._dbg_csv.writerow([
                t_sec,
                int(self.measurement_count),
                int(slot_j),
                float(z[0]),
                float(z[1]),
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
        for p in self.particles:
            lm = p.landmarks.get(slot_j, None)
            if self.use_template_map:
                if lm is None:
                    continue
            else:
                if lm is None:
                    lm = self._init_landmark_from_measurement(p, z)
                    p.landmarks[slot_j] = lm

            self._ekf_update_landmark(p, lm, z, self.R)
            lm.update_width(w_meas)

        # ---- Downstream spacing anchor update (use assoc origin if you want consistency) ----
        if self.snap_downstream_spacing:
            origin_xy = self.assoc_origin_xy if getattr(self, "assoc_origin_xy", None) is not None else self.row_origin_xy
            s_meas_best = float(np.dot(mu_world_approx - origin_xy, self.t_hat))

            # Keep row_origin_s aligned with association anchor when snapping
            self.row_origin_s = (float(self.assoc_origin_s) if getattr(self, "assoc_origin_s", None) is not None else self.row_origin_s)

            lm_best = self._best_particle().landmarks.get(slot_j) if self._best_particle() else None
            seen = int(lm_best.seen_count) if lm_best is not None else 0

            # Only allow snapping early, before this slot is well-established
            if seen < 2:
                alpha = 0.15  # 0.05–0.2 is typical
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

    # ------------- Data association by slot index -------------

    def _world_from_robot(self, pose: np.ndarray, z_robot: np.ndarray) -> np.ndarray:
        rx, ry, th = float(pose[0]), float(pose[1]), float(pose[2])
        c, s = float(np.cos(th)), float(np.sin(th))

        y_lat = float(z_robot[0])
        x_fwd = float(z_robot[1])

        dx_w = c * x_fwd - s * y_lat
        dy_w = s * x_fwd + c * y_lat
        return np.array([rx + dx_w, ry + dy_w], dtype=float)

    def _data_association_slot(self, z_robot: np.ndarray) -> Tuple[Optional[int], Optional[np.ndarray]]:
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

            poses = np.stack([p.pose for p in self.particles], axis=0)
            xy = np.sum(poses[:, 0:2] * w[:, None], axis=0)
            c = float(np.sum(np.cos(poses[:, 2]) * w))
            s = float(np.sum(np.sin(poses[:, 2]) * w))
            yaw = float(np.arctan2(s, c))

            pose_ref = np.array([xy[0], xy[1], yaw], dtype=float)

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
        s_meas = float(np.dot(mu_world_approx - origin_xy, self.t_hat))

        if getattr(self, "assoc_origin_s", None) is None:
            # If template map is on, assoc_origin_s should already be set at init.
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

        z_pred = np.array([y_lat, x_fwd], dtype=float)

        H = np.array([[-s,  c],
                      [ c,  s]], dtype=float)
        return z_pred, H

    def _predict_z_Hland_Jpose(self, pose: np.ndarray, mu_lm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x, y, th = float(pose[0]), float(pose[1]), float(pose[2])
        c, s = float(np.cos(th)), float(np.sin(th))

        dx = float(mu_lm[0] - x)
        dy = float(mu_lm[1] - y)

        x_fwd = c * dx + s * dy
        y_lat = -s * dx + c * dy
        z_pred = np.array([y_lat, x_fwd], dtype=float)

        H_lm = np.array([[-s,  c],
                         [ c,  s]], dtype=float)

        J_x = np.array([[ s, -c, -x_fwd],
                        [-c, -s,  y_lat]], dtype=float)

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

    def _ekf_update_landmark(self, p: Particle, lm: LandmarkEKF, z: np.ndarray, R_meas: np.ndarray):
        z_pred, H = self._predict_z_and_H(p.pose, lm.mu)

        S = H @ lm.Sigma @ H.T + R_meas
        try:
            Sinv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return
        K = lm.Sigma @ H.T @ Sinv

        innov = z - z_pred
        lm.mu = lm.mu + K @ innov
        lm.Sigma = (np.eye(2) - K @ H) @ lm.Sigma
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
                d_k = self._template_d_for_slot(int(k)) if lm.seen_count == 0 else float(np.dot(lm.mu - self.row_origin_xy, self.n_hat))
                lm.mu = self.row_origin_xy + s_k * self.t_hat + d_k * self.n_hat

    def _resnap_unseen_template_landmarks(self):
        if not self.use_template_map or not self.particles:
            return
        for p in self.particles:
            for k, lm in p.landmarks.items():
                if lm.seen_count > 0:
                    continue
                s_k = self.row_origin_s + int(k) * self.slot_spacing
                d_k = self._template_d_for_slot(int(k))
                lm.mu = self.row_origin_xy + s_k * self.t_hat + d_k * self.n_hat

    # =========================================================
    #  Resampling
    # =========================================================

    def _normalize_weights(self):
        if not self.particles:
            return
        s = float(sum(p.weight for p in self.particles))
        if s < 1e-12:
            w0 = 1.0 / float(len(self.particles))
            for p in self.particles:
                p.weight = w0
            return
        inv = 1.0 / s
        for p in self.particles:
            p.weight *= inv

    def _effective_sample_size(self) -> float:
        w = np.array([p.weight for p in self.particles], dtype=float)
        return 1.0 / float(np.sum(w * w) + 1e-12)

    def _maybe_resample(self):
        if not self.particles:
            return
        neff = self._effective_sample_size()
        if (neff / float(len(self.particles))) > self.neff_ratio_threshold:
            return

        w = np.array([p.weight for p in self.particles], dtype=float)
        cdf = np.cumsum(w)

        N = len(self.particles)
        step = 1.0 / N
        start = np.random.uniform(0.0, step)
        u = start + step * np.arange(N)

        idxs = np.searchsorted(cdf, u)

        new_particles: List[Particle] = []
        w0 = 1.0 / float(N)

        for idx in idxs:
            src = self.particles[int(idx)]
            lm_new = {
                j: LandmarkEKF(mu=lm.mu.copy(), Sigma=lm.Sigma.copy(),
                               seen_count=lm.seen_count,
                               width_sum=lm.width_sum, width_count=lm.width_count)
                for j, lm in src.landmarks.items()
            }
            new_particles.append(Particle(pose=src.pose.copy(), landmarks=lm_new, weight=w0))

        self.particles = new_particles
        self._normalize_weights()

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

    # =========================================================
    #  Row yaw callback
    # =========================================================

    def row_datum_pose_callback(self, msg: PoseStamped):
        q = msg.pose.orientation
        yaw_meas = float(R.from_quat([q.x, q.y, q.z, q.w]).as_euler("zyx")[0])

        if self.row_yaw_est is None:
            self.row_yaw_est = yaw_meas
        else:
            a = float(self.row_yaw_alpha)
            c = a * np.cos(self.row_yaw_est) + (1.0 - a) * np.cos(yaw_meas)
            s = a * np.sin(self.row_yaw_est) + (1.0 - a) * np.sin(yaw_meas)
            self.row_yaw_est = float(np.arctan2(s, c))

        th = float(self.row_yaw_est)

        self.t_hat = np.array([np.cos(th), np.sin(th)], dtype=float) * float(self.row_dir_sign)
        self.t_hat /= (np.linalg.norm(self.t_hat) + 1e-12)
        self.n_hat = np.array([-self.t_hat[1], self.t_hat[0]], dtype=float)

        self.row_axis = self.t_hat.copy()
        self.lateral_dir = self.n_hat.copy()

        self._resnap_unseen_template_landmarks()
        self._publish_registry_from_best()


def main(args=None):
    rclpy.init(args=args)
    node = RowFastSLAMNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down RowFastSLAMNode...")
    finally:
        try:
            node._dbg_close()
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except rclpy.exceptions.RCLError:
                pass


if __name__ == "__main__":
    main()
