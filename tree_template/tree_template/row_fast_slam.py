#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped

from tf2_ros import TransformBroadcaster
from tree_template_interfaces.msg import TrunkInfo, TrunkRegistry


# ============= Utility / math helpers =============

def wrap_angle(theta: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


def se2_from_odom(msg: Odometry) -> np.ndarray:
    """Extract [x, y, yaw] from nav_msgs/Odometry."""
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation

    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    yaw = float(np.arctan2(siny_cosp, cosy_cosp))

    return np.array([float(p.x), float(p.y), yaw], dtype=float)


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

    Updates in this version:
      - Publishes a best-particle Odometry on `odom_slam_best` (configurable).
      - DOES NOT pivot/rotate landmark map based on global row yaw.
      - Assumes the trunk template axis is world X (either +X or -X).
        (Row direction encoded by row_dir_sign.)
    """

    def __init__(self):
        super().__init__("row_fast_slam_node")

        # ---------- Parameters ----------
        self.declare_parameter("num_particles", 1000)
        self.declare_parameter("measurement_topic", "trunk_measurements")
        self.declare_parameter("odom_topic", "odom")
        self.declare_parameter("registry_topic", "fastslam_registry")
        self.declare_parameter("pose_topic", "fastslam_pose")

        # NEW: publish best SLAM odom
        self.declare_parameter("odom_best_topic", "odom_slam_best")

        # Row structure (slot indexing along the row)
        self.declare_parameter("slot_spacing", 1.0)   # [m]
        self.declare_parameter("row_origin_s", 0.0)    # [m] s coordinate of slot 0
        self.declare_parameter("slot_s_gate", 1.0)     # [m] max |s_meas - s_slot| to accept
        self.declare_parameter("snap_downstream_spacing", True)
        self.declare_parameter("downstream_snap_mode", "unseen_only")  # "unseen_only" or "all_downstream"

        # Template-map initialization
        self.declare_parameter("use_template_map", True)
        self.declare_parameter("num_slots", 50)
        self.declare_parameter("row_origin_x", -0.6)   # if NaN, use first odom pose
        self.declare_parameter("row_origin_y", 1.8)    # if NaN, use first odom pose

        # NEW: row direction sign along X axis. (+1 => +X, -1 => -X)
        self.declare_parameter("row_dir_sign", -1)

        self.declare_parameter("start_side", "near")   # slot 0 side
        self.declare_parameter("init_d_near", 0.0)
        self.declare_parameter("init_d_far", 0.0)
        self.declare_parameter("prior_sigma_s", 1.0)
        self.declare_parameter("prior_sigma_d", 1.0)
        self.declare_parameter("side_mode", "fixed")   # "fixed" or "geometry"

        # Measurement noise (std dev) in robot frame
        self.declare_parameter("meas_std_x", 0.05)
        self.declare_parameter("meas_std_y", 0.05)

        # Resampling
        self.declare_parameter("resample_interval", 3)
        self.declare_parameter("neff_ratio_threshold", 0.5)

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

        self.resample_interval = int(self.get_parameter("resample_interval").value)
        self.neff_ratio_threshold = float(self.get_parameter("neff_ratio_threshold").value)

        # ---------- State ----------
        self.particles: List[Particle] = []
        self.last_odom_pose: Optional[np.ndarray] = None
        self._last_odom_time: Optional[float] = None
        self.measurement_count = 0

        # Keep the last incoming odom message so we can mirror frame_ids / twist if desired
        self._last_odom_msg: Optional[Odometry] = None

        # ---------- Pub/Sub ----------
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.registry_pub = self.create_publisher(TrunkRegistry, self.registry_topic, qos)
        self.pose_pub = self.create_publisher(PoseStamped, self.pose_topic, qos)

        # NEW: best-odom publisher
        self.odom_best_pub = self.create_publisher(Odometry, self.odom_best_topic, qos)

        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 50)
        self.meas_sub = self.create_subscription(TrunkInfo, self.measurement_topic, self.measurement_callback, 50)

        self.tf_broadcaster = TransformBroadcaster(self)

        self.get_logger().info(
            "RowFastSLAMNode started.\n"
            f"  num_particles={self.num_particles}\n"
            f"  odom_topic={self.odom_topic}\n"
            f"  measurement_topic={self.measurement_topic}\n"
            f"  registry_topic={self.registry_topic}\n"
            f"  pose_topic={self.pose_topic}\n"
            f"  odom_best_topic={self.odom_best_topic}\n"
            f"  use_template_map={self.use_template_map}, num_slots={self.num_slots}, slot_spacing={self.slot_spacing:.2f}\n"
            f"  row_axis_fixed=[{self.row_axis[0]:.0f}, {self.row_axis[1]:.0f}] (template assumed on world X)\n"
            f"  side_mode={self.side_mode}"
        )

    # ---------- Helpers (row structure) ----------

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

    # =========================================================
    #  Motion update (FastSLAM step 2)
    # =========================================================

    def odom_callback(self, msg: Odometry):
        """
        Motion model anchored on incoming odom.

        Treat the odom pose as the mean, and sample particle poses
        around it with noise scaled by twist and dt.
        """
        self._last_odom_msg = msg
        cur_pose = se2_from_odom(msg)  # [x, y, yaw]

        t_cur = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self._last_odom_time is None:
            self._last_odom_time = t_cur
        dt = max(t_cur - self._last_odom_time, 1e-3)
        self._last_odom_time = t_cur

        if self.last_odom_pose is None:
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
            return

        self.last_odom_pose = cur_pose.copy()

        if not self.particles:
            self._init_particles(cur_pose)
            return

        # Build noise std from twist & dt
        v = msg.twist.twist.linear.x
        omega = msg.twist.twist.angular.z

        a_trans = 0.1
        b_trans = 0.005
        c_lat   = 0.3
        a_rot   = 0.1
        b_rot   = 0.002

        dx_expected = abs(v) * dt
        dtheta_expected = abs(omega) * dt

        sigma_x = a_trans * dx_expected + b_trans
        sigma_y = c_lat * sigma_x
        sigma_theta = a_rot * dtheta_expected + b_rot

        noise_std = np.array([sigma_x, sigma_y, sigma_theta], dtype=float)

        # Sample particle poses around odom pose
        for p in self.particles:
            noise = noise_std * np.random.randn(3)
            p.pose = cur_pose + noise
            p.pose[2] = wrap_angle(p.pose[2])

    def _init_particles(self, init_pose: np.ndarray):
        """Initialize particle set around an initial pose (and optionally seed a template landmark map)."""
        Sigma0 = self._sigma_world_from_row_sigmas()
        landmarks_template: Dict[int, LandmarkEKF] = {}

        if self.use_template_map:
            if not np.isfinite(self.row_origin_xy).all():
                self.row_origin_xy = init_pose[0:2].copy()

            t_hat, n_hat = self._row_basis_2d()

            for j in range(self.num_slots):
                side = self._side_for_index(j)
                d0 = self.init_d_near if side == "near" else self.init_d_far
                s0 = self.row_origin_s + j * self.slot_spacing

                # FIXED template: along world X (+/-), lateral world Y
                mu0 = self.row_origin_xy + s0 * t_hat + d0 * n_hat
                landmarks_template[j] = LandmarkEKF(mu=mu0.copy(), Sigma=Sigma0.copy())

        self.particles = []
        for _ in range(self.num_particles):
            pose = init_pose.copy()
            pose[0:2] += np.random.randn(2) * 0.01
            pose[2] = wrap_angle(pose[2] + np.random.randn() * 0.005)

            lm_dict: Dict[int, LandmarkEKF] = {}
            for j, lm in landmarks_template.items():
                lm_dict[j] = LandmarkEKF(mu=lm.mu.copy(), Sigma=lm.Sigma.copy())

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

        z = np.array([-msg.pose.position.x, msg.pose.position.z], dtype=float)
        if np.linalg.norm(z) < 1e-3:
            return

        w_meas = msg.width if np.isfinite(msg.width) else None

        slot_j, mu_world_approx = self._data_association_slot(z)
        if slot_j is None or mu_world_approx is None:
            return

        # EKF updates per particle
        R_meas = self.R
        for p in self.particles:
            lm = p.landmarks.get(slot_j, None)

            if lm is None:
                th = p.pose[2]
                c = np.cos(th)
                s = np.sin(th)

                dx_w = c * z[0] - s * z[1]
                dy_w = s * z[0] + c * z[1]
                mu_init = np.array([p.pose[0] + dx_w, p.pose[1] + dy_w], dtype=float)

                Sigma_init = np.diag([0.5 ** 2, 0.5 ** 2])
                lm = LandmarkEKF(mu=mu_init, Sigma=Sigma_init)
                p.landmarks[slot_j] = lm

            self._ekf_update_landmark(p, lm, z, R_meas)
            lm.update_width(w_meas)

        # --- spacing grid snap is still OK (it only shifts s-origin / translates slots) ---
        if self.snap_downstream_spacing:
            t_hat, _ = self._row_basis_2d()
            s_meas = float(np.dot(mu_world_approx - self.row_origin_xy, t_hat))

            self.row_origin_s = s_meas - slot_j * self.slot_spacing
            self._apply_downstream_spacing(anchor_j=slot_j)

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

    def _data_association_slot(self, z_robot: np.ndarray) -> Tuple[Optional[int], Optional[np.ndarray]]:
        """
        Map a measurement to the nearest slot index along the FIXED row axis (world X).

        Returns (slot_j, mu_world_approx).
        """
        if not self.particles:
            return None, None

        weights = np.array([p.weight for p in self.particles], dtype=float)
        best_idx = int(np.argmax(weights))
        p_best = self.particles[best_idx]

        th = p_best.pose[2]
        c = np.cos(th)
        s = np.sin(th)

        dx_w = c * z_robot[0] - s * z_robot[1]
        dy_w = s * z_robot[0] + c * z_robot[1]
        mu_world_approx = np.array([p_best.pose[0] + dx_w, p_best.pose[1] + dy_w], dtype=float)

        # Along-row coordinate s using fixed row_axis
        if np.isfinite(self.row_origin_xy).all():
            s_meas = float(np.dot(mu_world_approx - self.row_origin_xy, self.row_axis))
        else:
            s_meas = float(np.dot(self.row_axis, mu_world_approx))

        j_float = (s_meas - self.row_origin_s) / self.slot_spacing
        j_idx = int(round(j_float))

        s_slot = self.row_origin_s + j_idx * self.slot_spacing
        if abs(s_meas - s_slot) > self.slot_s_gate:
            self.get_logger().debug(f"Measurement s={s_meas:.2f} too far from slot s_j={s_slot:.2f}; rejecting.")
            return None, None

        if self.use_template_map:
            j_idx = int(np.clip(j_idx, 0, self.num_slots - 1))

        return j_idx, mu_world_approx

    # ------------- Landmark EKF update -------------

    def _ekf_update_landmark(self, p: Particle, lm: LandmarkEKF, z: np.ndarray, R_meas: np.ndarray):
        """
        EKF update for a landmark given particle pose p.pose and measurement z in robot frame.

        z = Rot(-yaw) * (mu - [x,y])
        """
        x, y, th = p.pose
        c = np.cos(th)
        s = np.sin(th)

        dx = lm.mu[0] - x
        dy = lm.mu[1] - y
        z_pred = np.array([c * dx + s * dy, -s * dx + c * dy], dtype=float)

        H = np.array([[c, s],
                      [-s, c]], dtype=float)

        S = H @ lm.Sigma @ H.T + R_meas
        K = lm.Sigma @ H.T @ np.linalg.inv(S)

        innov = z - z_pred
        lm.mu = lm.mu + K @ innov
        lm.Sigma = (np.eye(2) - K @ H) @ lm.Sigma
        lm.seen_count += 1

    def _apply_downstream_spacing(self, anchor_j: int):
        """
        Enforce s-grid for downstream slots while preserving each slot's lateral offset (world Y).
        """
        t_hat, n_hat = self._row_basis_2d()

        for p in self.particles:
            for k, lm in p.landmarks.items():
                if k <= anchor_j:
                    continue

                if self.downstream_snap_mode == "unseen_only":
                    if getattr(lm, "seen_count", 0) > 0:
                        continue

                s_k = self.row_origin_s + k * self.slot_spacing

                v = lm.mu - self.row_origin_xy
                d_k = float(np.dot(v, n_hat))

                lm.mu = self.row_origin_xy + s_k * t_hat + d_k * n_hat

    # ------------- Importance weighting -------------

    def _update_weights_after_measurement(self, z: np.ndarray, slot_j: int, R_meas: np.ndarray):
        logw = np.zeros(len(self.particles), dtype=float)

        for i, p in enumerate(self.particles):
            lm = p.landmarks.get(slot_j, None)
            if lm is None:
                logw[i] = -50.0
                continue

            x, y, th = p.pose
            c = np.cos(th)
            s = np.sin(th)

            dx = lm.mu[0] - x
            dy = lm.mu[1] - y
            z_pred = np.array([c * dx + s * dy, -s * dx + c * dy], dtype=float)

            H = np.array([[c, s],
                          [-s, c]], dtype=float)
            S = H @ lm.Sigma @ H.T + R_meas

            innov = z - z_pred
            try:
                Sinv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                logw[i] = -50.0
                continue

            maha = float(innov.T @ Sinv @ innov)
            sign, logdet = np.linalg.slogdet(S)
            if sign <= 0:
                logw[i] = -50.0
                continue
            logw[i] = -0.5 * (maha + logdet)

        maxlog = float(np.max(logw))
        w = np.exp(logw - maxlog)
        for i, p in enumerate(self.particles):
            p.weight *= float(w[i])

        self._normalize_weights()

    def _normalize_weights(self):
        s = float(sum(p.weight for p in self.particles))
        if s < 1e-12:
            for p in self.particles:
                p.weight = 1.0 / max(len(self.particles), 1)
            return
        for p in self.particles:
            p.weight /= s

    # ------------- Resampling -------------

    def _effective_sample_size(self) -> float:
        w = np.array([p.weight for p in self.particles], dtype=float)
        return 1.0 / float(np.sum(w * w) + 1e-12)

    def _maybe_resample(self):
        if not self.particles:
            return
        neff = self._effective_sample_size()
        ratio = neff / float(len(self.particles))
        if ratio > self.neff_ratio_threshold:
            return

        w = np.array([p.weight for p in self.particles], dtype=float)
        cdf = np.cumsum(w)
        N = len(self.particles)
        step = 1.0 / N
        start = np.random.uniform(0.0, step)
        u = start + step * np.arange(N)

        idxs = np.searchsorted(cdf, u)

        new_particles: List[Particle] = []
        for idx in idxs:
            p = self.particles[int(idx)]
            lm_new: Dict[int, LandmarkEKF] = {}
            for j, lm in p.landmarks.items():
                lm_new[j] = LandmarkEKF(mu=lm.mu.copy(), Sigma=lm.Sigma.copy(),
                                        seen_count=lm.seen_count,
                                        width_sum=lm.width_sum, width_count=lm.width_count)
            new_particles.append(Particle(pose=p.pose.copy(), landmarks=lm_new, weight=1.0 / N))

        self.particles = new_particles
        self._normalize_weights()

    # ------------- Publishing -------------

    def _publish_pose_from_best(self):
        if not self.particles:
            return
        best_idx = int(np.argmax([p.weight for p in self.particles]))
        p_best = self.particles[best_idx]

        ps = PoseStamped()
        ps.header.frame_id = "world"
        ps.header.stamp = self.get_clock().now().to_msg()

        ps.pose.position.x = float(p_best.pose[0])
        ps.pose.position.y = float(p_best.pose[1])
        ps.pose.position.z = 0.0

        yaw = float(p_best.pose[2])
        ps.pose.orientation.z = float(np.sin(yaw / 2.0))
        ps.pose.orientation.w = float(np.cos(yaw / 2.0))

        self.pose_pub.publish(ps)

    def _publish_odom_from_best(self):
        """
        Publish best-particle pose as nav_msgs/Odometry on odom_best_topic.

        This is the “converged odom” you can feed to consumers that want
        a single corrected pose stream.
        """
        if not self.particles:
            return

        best_idx = int(np.argmax([p.weight for p in self.particles]))
        p_best = self.particles[best_idx]

        now = self.get_clock().now().to_msg()

        odom_out = Odometry()
        odom_out.header.stamp = now

        # Mirror frames from incoming odom if available, otherwise fall back.
        if self._last_odom_msg is not None:
            odom_out.header.frame_id = self._last_odom_msg.header.frame_id
            odom_out.child_frame_id = self._last_odom_msg.child_frame_id
            odom_out.twist = self._last_odom_msg.twist
        else:
            odom_out.header.frame_id = "odom"
            odom_out.child_frame_id = "amiga__base"

        odom_out.pose.pose.position.x = float(p_best.pose[0])
        odom_out.pose.pose.position.y = float(p_best.pose[1])
        odom_out.pose.pose.position.z = 0.0

        yaw = float(p_best.pose[2])
        odom_out.pose.pose.orientation.z = float(np.sin(yaw / 2.0))
        odom_out.pose.pose.orientation.w = float(np.cos(yaw / 2.0))

        self.odom_best_pub.publish(odom_out)

        # -------- TF transform odom -> base_link --------
        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = odom_out.header.frame_id
        t.child_frame_id = odom_out.child_frame_id

        t.transform.translation.x = float(p_best.pose[0])
        t.transform.translation.y = float(p_best.pose[1])
        t.transform.translation.z = 0.0

        t.transform.rotation = odom_out.pose.pose.orientation
        self.tf_broadcaster.sendTransform(t)

    def _publish_registry_from_best(self):
        """
        Publish TrunkRegistry from the best particle's map.

        IMPORTANT: trunk orientations are identity here (we are not encoding row yaw in trunks).
        """
        if not self.particles:
            return
        best_idx = int(np.argmax([p.weight for p in self.particles]))
        p_best = self.particles[best_idx]
        if not p_best.landmarks:
            return

        reg = TrunkRegistry()
        trunks: List[TrunkInfo] = []
        items = sorted(p_best.landmarks.items(), key=lambda kv: kv[0])

        if self.side_mode == "fixed":
            for j, lm in items:
                ti = TrunkInfo()
                ti.pose.position.x = float(lm.mu[0])
                ti.pose.position.y = float(lm.mu[1])
                ti.pose.position.z = 0.0
                ti.pose.orientation.w = 1.0  # identity
                ti.side = self._side_for_index(int(j))
                ti.width = lm.width_mean if np.isfinite(lm.width_mean) else float("nan")
                trunks.append(ti)
            reg.trunks = trunks
            self.registry_pub.publish(reg)
            return

        # geometry-based side (still allowed, but uses fixed lateral_dir)
        d_lats = [float(np.dot(self.lateral_dir, lm.mu)) for _, lm in items]
        d_lats_arr = np.array(d_lats, dtype=float)

        if d_lats_arr.size < 2:
            for j, lm in items:
                ti = TrunkInfo()
                ti.pose.position.x = float(lm.mu[0])
                ti.pose.position.y = float(lm.mu[1])
                ti.pose.position.z = 0.0
                ti.pose.orientation.w = 1.0
                ti.side = "near"
                ti.width = lm.width_mean if np.isfinite(lm.width_mean) else float("nan")
                trunks.append(ti)
            reg.trunks = trunks
            self.registry_pub.publish(reg)
            return

        center_lat = float(np.mean(d_lats_arr))
        robot_xy = p_best.pose[0:2]
        d_robot = float(np.dot(self.lateral_dir, robot_xy))
        robot_offset = d_robot - center_lat
        robot_side_sign = 1.0 if abs(robot_offset) < 1e-6 else np.sign(robot_offset)

        for (j, lm), d_lat in zip(items, d_lats_arr):
            offset = float(d_lat - center_lat)
            same_side = (np.sign(offset) == robot_side_sign) if abs(offset) >= 1e-6 else True

            ti = TrunkInfo()
            ti.pose.position.x = float(lm.mu[0])
            ti.pose.position.y = float(lm.mu[1])
            ti.pose.position.z = 0.0
            ti.pose.orientation.w = 1.0
            ti.side = "near" if same_side else "far"
            ti.width = lm.width_mean if np.isfinite(lm.width_mean) else float("nan")
            trunks.append(ti)

        reg.trunks = trunks
        self.registry_pub.publish(reg)


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
