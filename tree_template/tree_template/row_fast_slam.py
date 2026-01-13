#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry

from tf2_ros import TransformBroadcaster
from tree_template_interfaces.msg import TrunkInfo, TrunkRegistry


# ============= Utility / math helpers =============

def wrap_angle(theta: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (theta + np.pi) % (2.0 * np.pi) - np.pi

def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9

def se2_from_odom(msg: Odometry) -> np.ndarray:
    """Extract [x, y, yaw] from nav_msgs/Odometry"""
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    yaw = R.from_quat([q.x, q.y, q.z, q.w]).as_euler("zyx", degrees=False)[0]
    return np.array([float(p.x), float(p.y), float(yaw)], dtype=float)

def quat_xyzw_from_yaw(yaw: float) -> np.ndarray:
    # returns [x, y, z, w]
    return R.from_euler("z", float(yaw)).as_quat()

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
    """
    def __init__(self):
        super().__init__("row_fast_slam_node")

        # ---------- Parameters ----------
        self.declare_parameter("num_particles", 1000)
        self.declare_parameter("measurement_topic", "trunk_measurements")
        self.declare_parameter("odom_topic", "odom")
        self.declare_parameter("registry_topic", "fastslam_registry")
        self.declare_parameter("pose_topic", "fastslam_pose")
        self.declare_parameter("odom_best_topic", "odom_slam_best")

        # Row structure (slot indexing along the row)
        self.declare_parameter("slot_spacing", 0.75)   # [m]
        self.declare_parameter("row_origin_s", 0.6)    # [m] s coordinate of slot 0
        self.declare_parameter("slot_s_gate", 0.25)     # [m] max |s_meas - s_slot| to accept
        self.declare_parameter("snap_downstream_spacing", True)
        self.declare_parameter("downstream_snap_mode", "unseen_only")  # "unseen_only" or "all_downstream"

        # Template-map initialization
        self.declare_parameter("use_template_map", True)
        self.declare_parameter("use_global_row_yaw", True)
        self.declare_parameter("num_slots", 50)
        self.declare_parameter("row_origin_x", 0.6)   # if NaN, use first odom pose
        self.declare_parameter("row_origin_y", -1.8)    # if NaN, use first odom pose
        self.declare_parameter("row_dir_sign", 1) # (+1 => +X, -1 => -X)
        self.declare_parameter("row_datum_pose_topic", "row_datum_pose")

        self.declare_parameter("start_side", "near")   # slot 0 side
        self.declare_parameter("init_d_near", 0.0)
        self.declare_parameter("init_d_far", 0.0)
        self.declare_parameter("prior_sigma_s", 0.6)
        self.declare_parameter("prior_sigma_d", 0.6)
        self.declare_parameter("side_mode", "semantic")   # "fixed" or "geometry" or "semantic"

        # Measurement noise (std dev) in robot frame
        self.declare_parameter("meas_std_x", 0.10) # left/right
        self.declare_parameter("meas_std_y", 0.3) # forward/back

        # Motion model noise parameters
        self.declare_parameter("motion_noise.a_trans", 0.1)
        self.declare_parameter("motion_noise.b_trans", 0.001)
        self.declare_parameter("motion_noise.c_lat", 0.5)
        self.declare_parameter("motion_noise.a_rot", 0.1)
        self.declare_parameter("motion_noise.b_rot", 0.005)

        # Resampling
        self.declare_parameter("resample_interval", 10)
        self.declare_parameter("neff_ratio_threshold", 0.3)

        # Semantic side memory (from TrunkClusterToTemplateNode side classification)
        self.declare_parameter("semantic_side_min_votes", 3)          # votes needed before we "trust" it
        self.declare_parameter("semantic_side_fallback", "fixed")     # "fixed" or "unknown"

        # ---------- Resolve params ----------
        self.num_particles = int(self.get_parameter("num_particles").value)
        self.measurement_topic = str(self.get_parameter("measurement_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.registry_topic = str(self.get_parameter("registry_topic").value)
        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.odom_best_topic = str(self.get_parameter("odom_best_topic").value)

        self.slot_spacing = float(self.get_parameter("slot_spacing").value)
        self.row_origin_s = float(self.get_parameter("row_origin_s").value)
        self.slot_s_gate = float(self.get_parameter("slot_s_gate").value)

        self.snap_downstream_spacing = bool(self.get_parameter("snap_downstream_spacing").value)
        self.downstream_snap_mode = str(self.get_parameter("downstream_snap_mode").value)

        self.use_template_map = bool(self.get_parameter("use_template_map").value)
        self.use_global_row_yaw = bool(self.get_parameter("use_global_row_yaw").value)
        self.num_slots = int(self.get_parameter("num_slots").value)

        self.row_origin_xy = np.array(
            [
                float(self.get_parameter("row_origin_x").value),
                float(self.get_parameter("row_origin_y").value),
            ],
            dtype=float,
        )

        self.row_dir_sign = int(self.get_parameter("row_dir_sign").value)
        if self.row_dir_sign not in (-1, 1):
            self.row_dir_sign = -1
        self.row_datum_pose_topic = str(self.get_parameter("row_datum_pose_topic").value)

        self.start_side = str(self.get_parameter("start_side").value).strip().lower()
        self.init_d_near = float(self.get_parameter("init_d_near").value)
        self.init_d_far = float(self.get_parameter("init_d_far").value)
        self.prior_sigma_s = float(self.get_parameter("prior_sigma_s").value)
        self.prior_sigma_d = float(self.get_parameter("prior_sigma_d").value)
        self.side_mode = str(self.get_parameter("side_mode").value)

        # FIXED row axis: world X only (positive or negative)
        self.row_axis = np.array([float(self.row_dir_sign), 0.0], dtype=float)
        # FIXED lateral direction: world +Y (keeps near/far “left-right” stable)
        self.lateral_dir = np.array([0.0, 1.0], dtype=float)

        self.meas_std = np.array(
            [
                float(self.get_parameter("meas_std_x").value),
                float(self.get_parameter("meas_std_y").value),
            ],
            dtype=float,
        )
        self.R = np.diag(self.meas_std ** 2)

        self.a_trans = float(self.get_parameter("motion_noise.a_trans").value)
        self.b_trans = float(self.get_parameter("motion_noise.b_trans").value)
        self.c_lat   = float(self.get_parameter("motion_noise.c_lat").value)
        self.a_rot   = float(self.get_parameter("motion_noise.a_rot").value)
        self.b_rot   = float(self.get_parameter("motion_noise.b_rot").value)

        self.resample_interval = int(self.get_parameter("resample_interval").value)
        self.neff_ratio_threshold = float(self.get_parameter("neff_ratio_threshold").value)

        self.semantic_side_min_votes = int(self.get_parameter("semantic_side_min_votes").value)
        self.semantic_side_fallback = str(self.get_parameter("semantic_side_fallback").value).strip().lower()
        if self.semantic_side_fallback not in ("fixed", "unknown"):
            self.semantic_side_fallback = "fixed"

        # ---------- Fixed geometry (precompute once) ----------
        # Row axis: world ±X
        self.row_axis = np.array([float(self.row_dir_sign), 0.0], dtype=float)
        self.row_axis /= (np.linalg.norm(self.row_axis) + 1e-12)

        # Lateral: world +Y
        self.lateral_dir = np.array([0.0, 1.0], dtype=float)

        self.row_yaw_est: Optional[float] = None
        self.row_yaw_alpha = 0.2  # smoothing (0.0 = no smoothing, 0.9 = heavy smoothing)

        # Cached basis
        self.t_hat = self.row_axis.copy()
        self.n_hat = self.lateral_dir.copy()

        # ---------- State ----------
        self.particles: List[Particle] = []
        self.last_odom_pose: Optional[np.ndarray] = None
        self._last_odom_time: Optional[float] = None
        self.measurement_count = 0
        self._slot_side_votes: Dict[int, Dict[str, int]] = {}
        self._last_odom_msg: Optional[Odometry] = None

        # ---------- Pub/Sub ----------
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.registry_pub = self.create_publisher(TrunkRegistry, self.registry_topic, qos)
        self.pose_pub = self.create_publisher(PoseStamped, self.pose_topic, qos)
        self.odom_best_pub = self.create_publisher(Odometry, self.odom_best_topic, qos)

        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 50)
        self.meas_sub = self.create_subscription(TrunkInfo, self.measurement_topic, self.measurement_callback, 50)
        self.row_datum_sub = self.create_subscription(PoseStamped, self.row_datum_pose_topic, self.row_datum_pose_callback, 10)
        
        self.tf_broadcaster = TransformBroadcaster(self)

        self.get_logger().info(
            "RowFastSLAMNode started.\n"
            f"  num_particles={self.num_particles}\n"
            f"  odom_topic={self.odom_topic}\n"
            f"  measurement_topic={self.measurement_topic}\n"
            f"  registry_topic={self.registry_topic}\n"
            f"  pose_topic={self.pose_topic}\n"
            f"  odom_best_topic={self.odom_best_topic}\n"
            f"  use_template_map={self.use_template_map}, use_global_row_yaw={self.use_global_row_yaw}\n"
            f"  num_slots={self.num_slots}, slot_spacing={self.slot_spacing:.2f}, slot_s_gate={self.slot_s_gate:.2f}\n"
            f"  row_dir_sign={self.row_dir_sign}, side_mode={self.side_mode}"
        )

    # ---------- Helpers (row structure) ----------

    def _best_particle(self) -> Optional[Particle]:
        if not self.particles:
            return None
        return self.particles[int(np.argmax([p.weight for p in self.particles]))]

    def _stamp_now(self):
        return self._last_odom_msg.header.stamp if self._last_odom_msg is not None else self.get_clock().now().to_msg()

    def _side_for_index(self, i: int) -> str:
        if self.start_side not in ("near", "far"):
            return "near"
        if i % 2 == 0:
            return self.start_side
        return "far" if self.start_side == "near" else "near"

    def _sigma_world_from_row_sigmas(self) -> np.ndarray:
        # Sigma = sigma_s^2 * t t^T + sigma_d^2 * n n^T (with fixed t,n)
        t = self.row_axis.reshape(2, 1)
        n = self.lateral_dir.reshape(2, 1)
        return (self.prior_sigma_s ** 2) * (t @ t.T) + (self.prior_sigma_d ** 2) * (n @ n.T)

    def _row_basis_2d(self):
        """Return (t_hat, n_hat) in world XY."""
        t_hat = self.row_axis.astype(float)
        t_hat /= (np.linalg.norm(t_hat) + 1e-12)
        n_hat = self.lateral_dir.astype(float)
        n_hat /= (np.linalg.norm(n_hat) + 1e-12)
        return t_hat, n_hat
    
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
            # no semantic info yet
            if self.semantic_side_fallback == "fixed":
                return self._side_for_index(int(slot_j))
            return "unknown"

        n_near = int(votes.get("near", 0))
        n_far = int(votes.get("far", 0))
        total = n_near + n_far

        # If we haven't accumulated enough evidence, fall back (optional)
        if total < max(self.semantic_side_min_votes, 1):
            if self.semantic_side_fallback == "fixed":
                return self._side_for_index(int(slot_j))
            return "unknown"

        if n_near == n_far:
            # tie-breaker: either fixed parity or unknown
            if self.semantic_side_fallback == "fixed":
                return self._side_for_index(int(slot_j))
            return "unknown"

        return "near" if n_near > n_far else "far"

    # =========================================================
    #  Motion update (FastSLAM step 2)
    # =========================================================

    def _dt_from_stamp(self, stamp) -> float:
        t_cur = stamp.sec + stamp.nanosec * 1e-9
        if self._last_odom_time is None:
            self._last_odom_time = t_cur
            return 1e-3
        dt = max(t_cur - self._last_odom_time, 1e-3)
        self._last_odom_time = t_cur
        return dt

    def _ensure_initialized_from_odom(self, cur_pose: np.ndarray) -> bool:
        if self.last_odom_pose is not None:
            return False

        if not np.isfinite(self.row_origin_xy).all():
            self.row_origin_xy = cur_pose[0:2].copy()

        self.last_odom_pose = cur_pose.copy()
        self._init_particles(cur_pose)

        self.get_logger().info(
            f"Initialized particles at odom pose x={cur_pose[0]:.2f}, y={cur_pose[1]:.2f}, yaw={cur_pose[2]:.2f} rad"
        )

        if self.use_template_map:
            self._publish_registry_from_best()
        self._publish_pose_from_best()
        self._publish_odom_from_best()
        return True

    def _motion_noise_std(self, v: float, omega: float, dt: float) -> np.ndarray:
        dx_expected = abs(v) * dt
        dtheta_expected = abs(omega) * dt

        sigma_x = self.a_trans * dx_expected + self.b_trans
        sigma_y = self.c_lat * sigma_x
        sigma_theta = self.a_rot * dtheta_expected + self.b_rot
        return np.array([sigma_x, sigma_y, sigma_theta], dtype=float)

    def odom_callback(self, msg: Odometry):
        self._last_odom_msg = msg
        cur_pose = se2_from_odom(msg)
        dt = self._dt_from_stamp(msg.header.stamp)

        if self._ensure_initialized_from_odom(cur_pose):
            return

        self.last_odom_pose = cur_pose.copy()

        if not self.particles:
            self._init_particles(cur_pose)
            self._publish_pose_from_best()
            self._publish_odom_from_best()
            return

        v = float(msg.twist.twist.linear.x)
        omega = float(msg.twist.twist.angular.z)
        noise_std = self._motion_noise_std(v, omega, dt)

        # Vectorized-ish sampling (still per particle object)
        for p in self.particles:
            p.pose = cur_pose + noise_std * np.random.randn(3)
            p.pose[2] = wrap_angle(p.pose[2])

        self._publish_pose_from_best()
        self._publish_odom_from_best()

    def _init_particles(self, init_pose: np.ndarray):
        Sigma0 = self._sigma_world_from_row_sigmas()
        landmarks_template: Dict[int, LandmarkEKF] = {}

        if self.use_template_map:
            if not np.isfinite(self.row_origin_xy).all():
                self.row_origin_xy = init_pose[0:2].copy()

            for j in range(self.num_slots):
                side = self._side_for_index(j)
                d0 = self.init_d_near if side == "near" else self.init_d_far
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
    #  Measurement update (FastSLAM step 3)
    # =========================================================

    def measurement_callback(self, msg: TrunkInfo):
        """
        Measurement update for a single observed trunk.

        Measurement is a trunk position in the ROBOT FRAME.
        Uses (x, z) as planar (forward, left).
        """
        if not self.particles or self.last_odom_pose is None:
            return

        z = np.array([msg.pose.position.x, msg.pose.position.z], dtype=float)
        if np.linalg.norm(z) < 1e-3:
            return

        w_meas = msg.width if np.isfinite(msg.width) else None

        slot_j, mu_world_approx = self._data_association_slot(z)
        if slot_j is None or mu_world_approx is None:
            return
        
        self._update_semantic_side_votes(slot_j, msg.side)

        # EKF updates per particle
        R_meas = self.R
        for p in self.particles:
            lm = p.landmarks.get(slot_j, None)

            if lm is None:
                lm = self._init_landmark_from_measurement(p, z)
                p.landmarks[slot_j] = lm

            self._ekf_update_landmark(p, lm, z, self.R)
            lm.update_width(w_meas)

        # Update downstream spacing
        if self.snap_downstream_spacing:
            s_meas = float(np.dot(mu_world_approx - self.row_origin_xy, self.t_hat))
            self.row_origin_s = s_meas - int(slot_j) * self.slot_spacing
            self._apply_downstream_spacing(anchor_j=int(slot_j))

        # Importance weighting
        self._update_weights_after_measurement(z, slot_j, R_meas)

        # Resampling
        self.measurement_count += 1
        if self.resample_interval > 0 and self.measurement_count % self.resample_interval == 0:
            self._maybe_resample()

        # Publish
        self._publish_registry_from_best()
        self._publish_pose_from_best()
        self._publish_odom_from_best()

    # ------------- Data association by slot index -------------

    def _world_from_robot(self, pose: np.ndarray, z_robot: np.ndarray) -> np.ndarray:
        # mu_world = [x,y] + R(yaw) * z_robot
        x, y, th = float(pose[0]), float(pose[1]), float(pose[2])
        c, s = np.cos(th), np.sin(th)
        dx_w = c * z_robot[0] - s * z_robot[1]
        dy_w = s * z_robot[0] + c * z_robot[1]
        return np.array([x + dx_w, y + dy_w], dtype=float)

    def _data_association_slot(self, z_robot: np.ndarray) -> Tuple[Optional[int], Optional[np.ndarray]]:
        p_best = self._best_particle()
        if p_best is None:
            return None, None

        mu_world_approx = self._world_from_robot(p_best.pose, z_robot)

        s_meas = float(np.dot(mu_world_approx - self.row_origin_xy, self.row_axis))
        j_idx = int(round((s_meas - self.row_origin_s) / self.slot_spacing))

        s_slot = self.row_origin_s + j_idx * self.slot_spacing
        if abs(s_meas - s_slot) > self.slot_s_gate:
            self.get_logger().debug(
                f"Measurement s={s_meas:.2f} too far from slot s_j={s_slot:.2f}; rejecting."
            )
            return None, None

        if self.use_template_map:
            j_idx = int(np.clip(j_idx, 0, self.num_slots - 1))

        return j_idx, mu_world_approx

    # ------------- Landmark EKF update -------------

    def _predict_z_and_H(self, pose: np.ndarray, mu_lm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        z_pred = R(-yaw) * (mu - [x,y])
        H = R(-yaw)
        """
        x, y, th = float(pose[0]), float(pose[1]), float(pose[2])
        c, s = np.cos(th), np.sin(th)

        dx = float(mu_lm[0] - x)
        dy = float(mu_lm[1] - y)

        z_pred = np.array([c * dx + s * dy, -s * dx + c * dy], dtype=float)
        H = np.array([[c, s],
                      [-s, c]], dtype=float)
        return z_pred, H

    def _init_landmark_from_measurement(self, p: Particle, z: np.ndarray) -> LandmarkEKF:
        mu_init = self._world_from_robot(p.pose, z)
        Sigma_init = np.diag([0.5 ** 2, 0.5 ** 2])
        return LandmarkEKF(mu=mu_init, Sigma=Sigma_init)

    def _ekf_update_landmark(self, p: Particle, lm: LandmarkEKF, z: np.ndarray, R_meas: np.ndarray):
        z_pred, H = self._predict_z_and_H(p.pose, lm.mu)

        S = H @ lm.Sigma @ H.T + R_meas
        K = lm.Sigma @ H.T @ np.linalg.inv(S)

        innov = z - z_pred
        lm.mu = lm.mu + K @ innov
        lm.Sigma = (np.eye(2) - K @ H) @ lm.Sigma
        lm.seen_count += 1

    def _apply_downstream_spacing(self, anchor_j: int):
        for p in self.particles:
            for k, lm in p.landmarks.items():
                if k <= anchor_j:
                    continue
                if self.downstream_snap_mode == "unseen_only" and lm.seen_count > 0:
                    continue

                s_k = self.row_origin_s + int(k) * self.slot_spacing
                d_k = float(np.dot(lm.mu - self.row_origin_xy, self.n_hat))
                lm.mu = self.row_origin_xy + s_k * self.t_hat + d_k * self.n_hat

    # ------------- Importance weighting -------------

    def _update_weights_after_measurement(self, z: np.ndarray, slot_j: int, R_meas: np.ndarray):
        logw = np.full(len(self.particles), -50.0, dtype=float)

        for i, p in enumerate(self.particles):
            lm = p.landmarks.get(slot_j)
            if lm is None:
                continue

            # z_pred: predicted measurement for this landmark (position in the robot frame)
            # H: measurement Jacobian (2x2) wrt landmark position in world frame
            z_pred, H = self._predict_z_and_H(p.pose, lm.mu)

            # lm.Sigma: landmark EKF covariance for this particle/slot
            # R_meas: diagonal measurement noise covariance which is the square of self.meas_std for x,y
            # S: covariance
            S = H @ lm.Sigma @ H.T + R_meas

            # innov: innovation = difference between actual and predicted measurement
            innov = z - z_pred
            try:
                Sinv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                continue

            # Mahalanobis distance: how big is the innovation compared to expected uncertainty
            #    small maha -> measurement matches prediction well
            #    large maha -> measurement is unlikely given prediction
            maha = float(innov.T @ Sinv @ innov)
            # Log-determinant of S: encapsulates normalized measurement uncertainty
            #    log -> numerical stability
            #    determinate -> volume of uncertainty
            #    sign -> should be positive for valid covariance
            sign, logdet = np.linalg.slogdet(S)
            if sign <= 0:
                continue

            # Log-weight update: log-likelihood of measurement given prediction (multivariate Gaussian)
            #    this implements only the exponent part (normalization constant omitted)
            logw[i] = -0.5 * (maha + logdet)

        # maxlog: numerical stability
        maxlog = float(np.max(logw))

        # Convert log-weights to normal weights (unnormalized)
        w = np.exp(logw - maxlog)

        # Apply weights to particles
        for i, p in enumerate(self.particles):
            p.weight *= float(w[i])

        self._normalize_weights()

    def _normalize_weights(self):
        if not self.particles:
            return
        # Sum of weights
        s = float(sum(p.weight for p in self.particles))
        # Avoid division by zero
        if s < 1e-12:
            # reset to uniform weights
            w0 = 1.0 / float(len(self.particles))
            for p in self.particles:
                p.weight = w0
            return
        # Normalize
        inv = 1.0 / s
        for p in self.particles:
            p.weight *= inv

    # ------------- Resampling -------------

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

    # ------------- Publishing -------------

    def _publish_pose_from_best(self):
        p_best = self._best_particle()
        if p_best is None:
            return

        ps = PoseStamped()
        ps.header.frame_id = "odom_slam"
        ps.header.stamp = self._stamp_now()

        ps.pose.position.x = float(p_best.pose[0])
        ps.pose.position.y = float(p_best.pose[1])
        ps.pose.position.z = 0.0

        q = quat_xyzw_from_yaw(float(p_best.pose[2]))
        ps.pose.orientation.x = float(q[0])
        ps.pose.orientation.y = float(q[1])
        ps.pose.orientation.z = float(q[2])
        ps.pose.orientation.w = float(q[3])

        self.pose_pub.publish(ps)

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

        odom_out.header.frame_id = "odom_slam"
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

        # Decide side per slot index once
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

            ti.side = side_by_j[int(j)]
            ti.width = lm.width_mean if np.isfinite(lm.width_mean) else float("nan")
            trunks.append(ti)

        reg = TrunkRegistry()
        reg.trunks = trunks
        self.registry_pub.publish(reg)

    def _geometry_sides_for_items(self, items, robot_xy: np.ndarray) -> Dict[int, str]:
        d_lats = np.array([float(np.dot(self.lateral_dir, lm.mu)) for _, lm in items], dtype=float)
        if d_lats.size < 2:
            return {int(j): "near" for j, _ in items}

        center_lat = float(np.mean(d_lats))
        d_robot = float(np.dot(self.lateral_dir, robot_xy))
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

        # enforce direction convention
        if self.row_dir_sign == -1:
            if np.cos(yaw) > 0:
                yaw = wrap_angle(yaw + np.pi)
        else:
            if np.cos(yaw) < 0:
                yaw = wrap_angle(yaw + np.pi)

        return yaw

    def row_datum_pose_callback(self, msg: PoseStamped):
        q = msg.pose.orientation
        yaw = float(R.from_quat([q.x, q.y, q.z, q.w]).as_euler("zyx")[0])

        # angle-safe exponential smoothing
        if self.row_yaw_est is None:
            self.row_yaw_est = yaw
        else:
            a = float(self.row_yaw_alpha)
            c = a * np.cos(self.row_yaw_est) + (1.0 - a) * np.cos(yaw)
            s = a * np.sin(self.row_yaw_est) + (1.0 - a) * np.sin(yaw)
            self.row_yaw_est = float(np.arctan2(s, c))

        th = float(self.row_yaw_est)

        # Update row basis used by slotting/template/snapping
        self.t_hat = np.array([np.cos(th), np.sin(th)], dtype=float) * float(self.row_dir_sign)
        self.t_hat /= (np.linalg.norm(self.t_hat) + 1e-12)
        self.n_hat = np.array([-self.t_hat[1], self.t_hat[0]], dtype=float)

        # If you still use these elsewhere, keep them consistent:
        self.row_axis = self.t_hat.copy()
        self.lateral_dir = self.n_hat.copy()


def main(args=None):
    rclpy.init(args=args)
    node = RowFastSLAMNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down RowFastSLAMNode...")
    finally:
        node.destroy_node()
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except rclpy.exceptions.RCLError:
                pass


if __name__ == "__main__":
    main()
