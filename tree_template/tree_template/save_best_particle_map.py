#!/usr/bin/env python3

from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import math
from collections import deque

import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import Point, Pose
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix
from tree_template_interfaces.msg import TrunkRegistry
from tree_template_interfaces.srv import SaveSlamResults

from scipy.spatial.transform import Rotation as R


def _point_to_list(p: Optional[Point]) -> Optional[List[float]]:
    if p is None:
        return None
    return [float(p.x), float(p.y), float(p.z)]


def _pose_position_to_list(pose: Optional[Pose]) -> Optional[List[float]]:
    if pose is None:
        return None
    p = pose.position
    return [float(p.x), float(p.y), float(p.z)]


def _pose_yaw_rad(pose: Optional[Pose]) -> Optional[float]:
    if pose is None:
        return None
    q = pose.orientation
    quat = np.array([float(q.x), float(q.y), float(q.z), float(q.w)], dtype=np.float64)
    if not np.all(np.isfinite(quat)):
        return None
    yaw = float(R.from_quat(quat).as_euler("xyz", degrees=False)[2])
    return yaw if math.isfinite(yaw) else None


def _odom_xy_yaw_t(msg: Odometry) -> Optional[List[float]]:
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    quat = np.array([float(q.x), float(q.y), float(q.z), float(q.w)], dtype=np.float64)
    if not (math.isfinite(p.x) and math.isfinite(p.y) and np.all(np.isfinite(quat))):
        return None
    yaw = float(R.from_quat(quat).as_euler("xyz", degrees=False)[2])
    if not math.isfinite(yaw):
        return None
    t = msg.header.stamp
    t_sec = float(t.sec) + 1e-9 * float(t.nanosec)
    return [float(p.x), float(p.y), float(yaw), float(t_sec)]


def world_correct(trees_xy: np.ndarray, slam_path: np.ndarray, global_path: np.ndarray, t_valid: float,
                  window_s: float, cam_fwd_m: float):
    """
    Post-SLAM world correction with the GPS-fused global EKF (robot_localization), no extra filter.

    SLAM's map lives in its own frame and inherits the slow scale/heading drift of local odometry. For each tree:
      t_obs = time the SLAM robot pose had the tree ~cam_fwd_m ahead (camera centred on it)
      C     = G(t) * P(t)^-1 (world <- SLAM frame), G = /odometry/global, P = SLAM best pose, averaged over
              t_obs +- window_s, using only global poses after t_valid (global heading aligned)
      tree  = C * tree_slam (tree keeps its SLAM position relative to the robot; the robot is placed by GPS)
    slam_path / global_path rows: [x, y, yaw, t]. Returns (trees_world (N,2), robot_path_world (M,4), t_obs (N,)).
    """
    pt = slam_path[:, 3]
    pyaw = np.unwrap(slam_path[:, 2])
    order = np.argsort(global_path[:, 3])
    gt = global_path[order, 3]
    gx, gy = global_path[order, 0], global_path[order, 1]
    gyaw = np.unwrap(global_path[order, 2])
    t_valid = max(float(t_valid), float(gt[0]))

    def corr(ts):
        th = np.interp(ts, gt, gyaw) - np.interp(ts, pt, pyaw)
        px, py = np.interp(ts, pt, slam_path[:, 0]), np.interp(ts, pt, slam_path[:, 1])
        cx = np.interp(ts, gt, gx) - (np.cos(th) * px - np.sin(th) * py)
        cy = np.interp(ts, gt, gy) - (np.sin(th) * px + np.cos(th) * py)
        return cx, cy, th

    ok = (pt >= gt[0]) & (pt <= gt[-1])
    out = np.zeros_like(trees_xy)
    t_obs = np.zeros(len(trees_xy))
    for i, (tx, ty) in enumerate(trees_xy):
        fwd = np.cos(pyaw) * (tx - slam_path[:, 0]) + np.sin(pyaw) * (ty - slam_path[:, 1])
        k = int(np.argmin(np.where(ok, np.abs(fwd - cam_fwd_m), np.inf)))
        t0 = pt[k]
        t_obs[i] = t0
        lo = max(t0 - window_s, t_valid)
        hi = max(min(t0 + window_s, gt[-1]), lo)
        cx, cy, th = corr(np.linspace(lo, hi, 31))
        thm = math.atan2(float(np.sin(th).mean()), float(np.cos(th).mean()))
        out[i] = [cx.mean() + math.cos(thm) * tx - math.sin(thm) * ty,
                  cy.mean() + math.sin(thm) * tx + math.cos(thm) * ty]
    ts = pt[ok]
    cx, cy, th = corr(ts)
    P = slam_path[ok]
    path_w = np.c_[cx + np.cos(th) * P[:, 0] - np.sin(th) * P[:, 1],
                   cy + np.sin(th) * P[:, 0] + np.cos(th) * P[:, 1],
                   P[:, 2] + th, P[:, 3]]
    return out, path_w, t_obs


def world_correct_gps(trees_xy: np.ndarray, slam_path: np.ndarray, fixes: np.ndarray, course: np.ndarray,
                      t_valid: float, window_s: float, cam_fwd_m: float, antenna_x: float = -0.5,
                      yaw_window_s: Optional[float] = None):
    """
    Post-SLAM world correction from raw RTK GPS (no EKF dependence).

    For each tree (t_obs as in world_correct):
      heading h(t) = SLAM yaw(t) + mean(course - SLAM yaw) over GPS course headings within t_obs +- yaw_window_s
      base(t)      = antenna(t) - R(h) * (antenna_x, 0)            (antenna from the fixes, ENU from the first fix)
      tree_w       = median over t in t_obs +- window_s of  base(t) + R(h(t)) * (tree in SLAM robot frame at t)
    fixes rows [t, lat, lon]; course rows [t, yaw_enu]; slam_path rows [x, y, yaw, t].
    Returns (trees_world (N,2) in ENU metres from the first fix, robot_path_world (M,4), origin (lat, lon)).
    """
    yaw_window_s = window_s if yaw_window_s is None else yaw_window_s
    pt = slam_path[:, 3]
    pyaw = np.unwrap(slam_path[:, 2])
    fx = fixes[np.argsort(fixes[:, 0])]
    lat0, lon0 = float(fx[0, 1]), float(fx[0, 2])
    ft = fx[:, 0]
    fe = np.radians(fx[:, 2] - lon0) * 6378137.0 * math.cos(math.radians(lat0))
    fn = np.radians(fx[:, 1] - lat0) * 6378137.0
    ct, cy = course[:, 0], course[:, 1]
    t_valid = max(float(t_valid), float(ft[0]))

    def dyaw_at(t0):
        lo, hi = max(t0 - yaw_window_s, t_valid), max(t0 + yaw_window_s, t_valid)
        m = (ct >= lo) & (ct <= hi)
        idx = np.flatnonzero(m) if m.sum() >= 3 else np.argsort(np.abs(ct - t0))[:5]
        d = cy[idx] - np.interp(ct[idx], pt, pyaw)
        return math.atan2(float(np.sin(d).mean()), float(np.cos(d).mean()))

    ok = (pt >= ft[0]) & (pt <= ft[-1])
    out = np.zeros_like(trees_xy)
    for i, (tx, ty) in enumerate(trees_xy):
        fwd = np.cos(pyaw) * (tx - slam_path[:, 0]) + np.sin(pyaw) * (ty - slam_path[:, 1])
        k = int(np.argmin(np.where(ok, np.abs(fwd - cam_fwd_m), np.inf)))
        t0 = pt[k]
        dth = dyaw_at(t0)
        lo = max(t0 - window_s, t_valid)
        hi = max(min(t0 + window_s, ft[-1]), lo)
        ts = np.linspace(lo, hi, 31)
        px, py, pth = np.interp(ts, pt, slam_path[:, 0]), np.interp(ts, pt, slam_path[:, 1]), np.interp(ts, pt, pyaw)
        rx = np.cos(pth) * (tx - px) + np.sin(pth) * (ty - py)
        ry = -np.sin(pth) * (tx - px) + np.cos(pth) * (ty - py)
        h = pth + dth
        bx = np.interp(ts, ft, fe) - antenna_x * np.cos(h)
        by = np.interp(ts, ft, fn) - antenna_x * np.sin(h)
        out[i] = [float(np.median(bx + np.cos(h) * rx - np.sin(h) * ry)),
                  float(np.median(by + np.sin(h) * rx + np.cos(h) * ry))]
    P = slam_path[ok]
    h = np.array([np.unwrap(P[:, 2])[j] + dyaw_at(P[j, 3]) for j in range(len(P))]) if len(P) else np.zeros(0)
    bx = np.interp(P[:, 3], ft, fe) - antenna_x * np.cos(h)
    by = np.interp(P[:, 3], ft, fn) - antenna_x * np.sin(h)
    return out, np.c_[bx, by, h, P[:, 3]], (lat0, lon0)


class SaveBestParticleMap(Node):
    """
    Service writes a YAML file (optionally timestamped).
    Captures robot path and trunks.
    """

    def __init__(self):
        super().__init__("save_best_particle_map")

        # Core topics / output
        self.declare_parameter("trunk_topic", "/fastslam_registry")
        self.declare_parameter("odom_corr_topic", "/initial_odom_correction")
        self.declare_parameter("gps_topic", "/initial_gps_fix")

        # default output base (used if request.output_path is empty)
        self.declare_parameter("out", "/tmp/slam_trunks.yaml")

        # Service name
        self.declare_parameter("service_name", "save_slam_results")

        self.declare_parameter("require_odom_correction", False)
        self.declare_parameter("require_gps_fix", False)
        self.declare_parameter("require_trunks", False)
        self.declare_parameter("require_final_gps_fix", False)

        # Robot path capture
        self.declare_parameter("robot_odom_topic", "/odom_slam_best")
        self.declare_parameter("robot_min_dt_sec", 0.10)
        self.declare_parameter("robot_min_dist_m", 0.05)
        self.declare_parameter("robot_max_points", 5000)

        # Final GPS topics: support BOTH NavSatFix and Point (lat/lon/alt)
        self.declare_parameter("final_gps_navsat_topic", "/fix")
        self.declare_parameter("final_gps_point_topic", "/final_gps_fix")
        self.declare_parameter("final_gps_samples", 30)

        # YAML fallback
        self.declare_parameter("initials_yaml", "")
        self.declare_parameter("use_initials_yaml_fallback", False)

        # World correction with the GPS-fused global EKF (gps_fusion_mode:=gps_heading). Adds trees_world /
        # robot_path_world (global "map" frame) to the YAML and optionally writes <name>_world.yaml with them as
        # trees / robot_path. Skipped (with a warning) when no global odometry was received.
        self.declare_parameter("world_correction.enable", True)
        # source: gps = robot pose from raw RTK fixes + GPS course heading (robust to global-EKF quality/CPU load;
        # world frame = ENU metres from the first fix, gps_origin saved); global = /odometry/global pose
        self.declare_parameter("world_correction.source", "gps")
        self.declare_parameter("world_correction.gps_fix_topic", "/fix_rtk")
        self.declare_parameter("world_correction.antenna_x", -0.5)        # GPS antenna x in the robot frame
        self.declare_parameter("world_correction.global_odom_topic", "/odometry/global")
        self.declare_parameter("world_correction.valid_from_topic", "/gps/course_imu")  # first msg = heading aligned
        self.declare_parameter("world_correction.settle_s", 5.0)
        self.declare_parameter("world_correction.window_s", 10.0)
        self.declare_parameter("world_correction.cam_fwd_m", 0.73)
        self.declare_parameter("world_correction.max_points", 40000)
        self.declare_parameter("world_correction.write_separate_file", True)

        # Resolve params
        self.trunk_topic = str(self.get_parameter("trunk_topic").value)
        self.odom_corr_topic = str(self.get_parameter("odom_corr_topic").value)
        self.gps_topic = str(self.get_parameter("gps_topic").value)
        self.service_name = str(self.get_parameter("service_name").value)

        self.require_odom_correction = bool(self.get_parameter("require_odom_correction").value)
        self.require_gps_fix = bool(self.get_parameter("require_gps_fix").value)
        self.require_trunks = bool(self.get_parameter("require_trunks").value)
        self.require_final_gps_fix = bool(self.get_parameter("require_final_gps_fix").value)

        self.robot_odom_topic = str(self.get_parameter("robot_odom_topic").value)
        self.robot_min_dt_sec = float(self.get_parameter("robot_min_dt_sec").value)
        self.robot_min_dist_m = float(self.get_parameter("robot_min_dist_m").value)
        self.robot_max_points = int(self.get_parameter("robot_max_points").value)

        self.final_gps_navsat_topic = str(self.get_parameter("final_gps_navsat_topic").value)
        self.final_gps_point_topic = str(self.get_parameter("final_gps_point_topic").value)
        self.final_gps_samples = int(self.get_parameter("final_gps_samples").value)

        self.initials_yaml = Path(str(self.get_parameter("initials_yaml").value)).expanduser()
        self.use_initials_yaml_fallback = bool(self.get_parameter("use_initials_yaml_fallback").value)

        # QoS
        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.VOLATILE

        qos_latched = QoSProfile(depth=1)
        qos_latched.durability = DurabilityPolicy.TRANSIENT_LOCAL
        qos_latched.reliability = ReliabilityPolicy.RELIABLE

        qos_navsat = QoSProfile(depth=10)
        qos_navsat.reliability = ReliabilityPolicy.BEST_EFFORT
        qos_navsat.durability = DurabilityPolicy.VOLATILE

        # State from topics
        self._latest_registry: Optional[TrunkRegistry] = None
        self._latest_odom_corr_pose: Optional[Pose] = None
        self._latest_gps_fix: Optional[Point] = None
        self._latest_final_gps_list: Optional[List[float]] = None
        self._final_gps_readings: deque[List[float]] = deque(maxlen=max(self.final_gps_samples, 1))

        # Raw GPS readings (every valid NavSatFix sample, unbounded)
        self._raw_gps_readings: List[List[float]] = []

        # Robot path buffer
        self._robot_path: deque[List[float]] = deque(maxlen=max(self.robot_max_points, 1))
        self._robot_last_keep: Optional[List[float]] = None

        # World correction state
        wc = lambda n: self.get_parameter(f"world_correction.{n}").value  # noqa: E731
        self.wc_enable = bool(wc("enable"))
        self.wc_global_topic = str(wc("global_odom_topic"))
        self.wc_valid_topic = str(wc("valid_from_topic"))
        self.wc_settle = float(wc("settle_s"))
        self.wc_window = float(wc("window_s"))
        self.wc_cam_fwd = float(wc("cam_fwd_m"))
        self.wc_separate = bool(wc("write_separate_file"))
        self.wc_source = str(wc("source"))
        self.wc_fix_topic = str(wc("gps_fix_topic"))
        self.wc_antenna_x = float(wc("antenna_x"))
        self._global_path: deque[List[float]] = deque(maxlen=max(int(wc("max_points")), 1))
        self._wc_fixes: deque[List[float]] = deque(maxlen=max(int(wc("max_points")), 1))    # t, lat, lon
        self._wc_course: deque[List[float]] = deque(maxlen=max(int(wc("max_points")), 1))   # t, yaw
        self._wc_valid_from: Optional[float] = None
        if self.wc_enable:
            from sensor_msgs.msg import Imu
            if self.wc_source == "global":
                self.create_subscription(Odometry, self.wc_global_topic, self._global_odom_cb, 50)
            else:
                qos_fix = QoSProfile(depth=10)
                qos_fix.reliability = ReliabilityPolicy.BEST_EFFORT
                self.create_subscription(NavSatFix, self.wc_fix_topic, self._wc_fix_cb, qos_fix)
            self.create_subscription(Imu, self.wc_valid_topic, self._wc_valid_cb, 10)

        # Subscriptions
        self.create_subscription(TrunkRegistry, self.trunk_topic, self._registry_cb, qos)
        self.create_subscription(Pose, self.odom_corr_topic, self._odom_corr_cb, qos_latched)
        self.create_subscription(Point, self.gps_topic, self._gps_cb, qos_latched)
        self.create_subscription(Odometry, self.robot_odom_topic, self._robot_odom_cb, qos)

        self.create_subscription(NavSatFix, self.final_gps_navsat_topic, self._final_gps_navsat_cb, qos_navsat)
        self.create_subscription(Point, self.final_gps_point_topic, self._final_gps_point_cb, qos_latched)

        # Service (NEW)
        self.srv = self.create_service(SaveSlamResults, self.service_name, self._on_dump)

        self.get_logger().info(
            f"Subscribing:\n"
            f"  trunk registry:          {self.trunk_topic}\n"
            f"  initial odom correction: {self.odom_corr_topic} (Pose, latched)\n"
            f"  initial gps fix:         {self.gps_topic} (Point, latched)\n"
            f"  robot odom path:         {self.robot_odom_topic} (Odometry)\n"
            f"  final gps (navsat):      {self.final_gps_navsat_topic}\n"
            f"  final gps (point):       {self.final_gps_point_topic} (latched)\n"
            f"initials_yaml={self.initials_yaml} (fallback={self.use_initials_yaml_fallback})\n"
            f"default Output base: {self.get_parameter('out').value}\n"
            f"Service: /{self.service_name}"
        )

    def _registry_cb(self, msg: TrunkRegistry):
        self._latest_registry = msg

    def _global_odom_cb(self, msg: Odometry):
        row = _odom_xy_yaw_t(msg)
        if row is not None:
            self._global_path.append(row)

    def _wc_valid_cb(self, msg):
        t = float(msg.header.stamp.sec) + 1e-9 * float(msg.header.stamp.nanosec)
        if self._wc_valid_from is None:
            self._wc_valid_from = t
        q = msg.orientation
        self._wc_course.append([t, math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))])

    def _wc_fix_cb(self, msg: NavSatFix):
        if msg.status.status < 0 or not (math.isfinite(msg.latitude) and math.isfinite(msg.longitude)):
            return
        t = float(msg.header.stamp.sec) + 1e-9 * float(msg.header.stamp.nanosec)
        self._wc_fixes.append([t, float(msg.latitude), float(msg.longitude)])

    def _world_correction(self, trees: Dict[int, List[float]], robot_path: List[List[float]]):
        """Returns (trees_world dict, robot_path_world list, info str) or (None, None, reason)."""
        if not self.wc_enable:
            return None, None, "disabled"
        if self.wc_source != "global":
            if len(self._wc_fixes) < 10 or len(self._wc_course) < 3 or len(robot_path) < 2 or not trees:
                return None, None, (f"no data (fixes {len(self._wc_fixes)}, course headings {len(self._wc_course)}, "
                                    f"robot path {len(robot_path)})")
            ids = sorted(trees)
            T = np.array([trees[i] for i in ids], dtype=np.float64)
            Tw, Pw, origin = world_correct_gps(T, np.array(robot_path, dtype=np.float64),
                                               np.array(self._wc_fixes, dtype=np.float64),
                                               np.array(self._wc_course, dtype=np.float64),
                                               self._wc_valid_from + self.wc_settle, self.wc_window, self.wc_cam_fwd,
                                               self.wc_antenna_x)
            self._wc_origin = origin
            info = (f"source gps, window {self.wc_window}s, {len(self._wc_fixes)} fixes, "
                    f"{len(self._wc_course)} course headings, ENU origin lat/lon {origin[0]:.8f},{origin[1]:.8f}")
            return ({i: [float(x), float(y)] for i, (x, y) in zip(ids, Tw)},
                    [[float(v) for v in r] for r in Pw], info)
        if len(self._global_path) < 20 or len(robot_path) < 2 or not trees:
            return None, None, f"no data (global poses {len(self._global_path)}, robot path {len(robot_path)})"
        if self._wc_valid_from is None:
            return None, None, f"no {self.wc_valid_topic} received (global heading never aligned)"
        ids = sorted(trees)
        T = np.array([trees[i] for i in ids], dtype=np.float64)
        Tw, Pw, _ = world_correct(T, np.array(robot_path, dtype=np.float64),
                                  np.array(self._global_path, dtype=np.float64),
                                  self._wc_valid_from + self.wc_settle, self.wc_window, self.wc_cam_fwd)
        info = (f"window {self.wc_window}s, valid from {self._wc_valid_from:.2f}+{self.wc_settle}s, "
                f"{len(self._global_path)} global poses")
        return ({i: [float(x), float(y)] for i, (x, y) in zip(ids, Tw)},
                [[float(v) for v in r] for r in Pw], info)

    def _odom_corr_cb(self, msg: Pose):
        self._latest_odom_corr_pose = msg

    def _gps_cb(self, msg: Point):
        self._latest_gps_fix = msg

    def _robot_odom_cb(self, msg: Odometry):
        row = _odom_xy_yaw_t(msg)
        if row is None:
            return
        if self._robot_last_keep is None:
            self._robot_path.append(row)
            self._robot_last_keep = row
            return
        x, y, _yaw, t = row
        x0, y0, _yaw0, t0 = self._robot_last_keep
        dt = float(t - t0)
        dist = float(math.hypot(float(x - x0), float(y - y0)))
        if (dt >= self.robot_min_dt_sec) or (dist >= self.robot_min_dist_m):
            self._robot_path.append(row)
            self._robot_last_keep = row

    def _accumulate_final_gps(self, lat: float, lon: float, alt: float):
        self._final_gps_readings.append([lat, lon, alt])
        count = len(self._final_gps_readings)

        if count < self.final_gps_samples:
            self.get_logger().debug(
                f"Accumulating final GPS: {count}/{self.final_gps_samples}"
            )
            return

        arr = np.array(self._final_gps_readings, dtype=np.float64)
        mean = np.mean(arr, axis=0)
        self._latest_final_gps_list = [float(mean[0]), float(mean[1]), float(mean[2])]

        self.get_logger().debug(
            f"Updated averaged final GPS from {count} samples: "
            f"lat={self._latest_final_gps_list[0]:.8f}, "
            f"lon={self._latest_final_gps_list[1]:.8f}, "
            f"alt={self._latest_final_gps_list[2]:.3f}"
        )

    def _final_gps_navsat_cb(self, msg: NavSatFix):
        if not np.isfinite(msg.latitude) or not np.isfinite(msg.longitude):
            return
        alt = float(msg.altitude) if np.isfinite(msg.altitude) else 0.0
        self._raw_gps_readings.append([float(msg.latitude), float(msg.longitude), alt])
        self._accumulate_final_gps(float(msg.latitude), float(msg.longitude), alt)

    def _final_gps_point_cb(self, msg: Point):
        if not np.isfinite(msg.x) or not np.isfinite(msg.y):
            return
        alt = float(msg.z) if np.isfinite(msg.z) else 0.0
        self._accumulate_final_gps(float(msg.x), float(msg.y), alt)

    def _resolve_base_path(self, requested: str) -> Path:
        """
        If requested is empty -> use node param 'out'.
        If requested is a directory -> create slam_trunks.yaml inside it.
        Ensure suffix .yaml (or .yml).
        """
        s = (requested or "").strip()
        if not s:
            s = str(self.get_parameter("out").value)

        p = Path(s).expanduser()

        # directory request: ".../trial_000/" or existing dir
        if str(s).endswith("/") or (p.exists() and p.is_dir()):
            p.mkdir(parents=True, exist_ok=True)
            p = p / "slam_trunks.yaml"

        # ensure parent exists
        p.parent.mkdir(parents=True, exist_ok=True)

        # ensure yaml suffix
        if p.suffix.lower() not in (".yaml", ".yml"):
            p = p.with_suffix(".yaml")

        return p

    def _on_dump(self, req: SaveSlamResults.Request, res: SaveSlamResults.Response) -> SaveSlamResults.Response:
        if self._latest_registry is None:
            res.success = False
            res.message = "No TrunkRegistry received yet."
            res.written_path = ""
            return res

        trunks = list(self._latest_registry.trunks)
        if self.require_trunks and len(trunks) == 0:
            res.success = False
            res.message = "Latest TrunkRegistry has zero trunks."
            res.written_path = ""
            return res

        if self.require_odom_correction and self._latest_odom_corr_pose is None:
            res.success = False
            res.message = "No initial_odom_correction received (and YAML fallback missing/disabled)."
            res.written_path = ""
            return res

        if self.require_gps_fix and self._latest_gps_fix is None:
            res.success = False
            res.message = "No initial_gps_fix received (and YAML fallback missing/disabled)."
            res.written_path = ""
            return res

        if self.require_final_gps_fix and self._latest_final_gps_list is None:
            res.success = False
            res.message = "No final_gps_fix received (and YAML fallback missing/disabled)."
            res.written_path = ""
            return res

        trees: Dict[int, List[float]] = {}
        for i, t in enumerate(trunks):
            p = t.pose.position
            trees[i] = [float(p.x), float(p.y)]

        # row_fast_slam stamps the locked (or best-effort PCA) row yaw estimate
        # into every published trunk's orientation when use_global_row_yaw is
        # enabled — all trunks carry the same value, so the first is enough.
        row_yaw_rad = _pose_yaw_rad(trunks[0].pose) if trunks else None

        init_odom_list = _pose_position_to_list(self._latest_odom_corr_pose)
        init_yaw = _pose_yaw_rad(self._latest_odom_corr_pose)
        robot_path = list(self._robot_path)

        data = {
            "initial_odom_correction": init_odom_list,
            "initial_yaw_correction_rad": init_yaw,
            "row_yaw_rad": row_yaw_rad,
            "initial_gps_fix": _point_to_list(self._latest_gps_fix),
            "final_gps_fix": self._latest_final_gps_list,
            "raw_gps_coords": list(self._raw_gps_readings),
            "robot_path": robot_path,
            "trees": trees,
        }

        try:
            trees_w, path_w, wc_info = self._world_correction(trees, robot_path)
        except Exception as e:  # never lose the SLAM map because of the correction
            trees_w, path_w, wc_info = None, None, f"failed: {e}"
        if trees_w is not None:
            data["trees_world"] = trees_w
            data["robot_path_world"] = path_w
            if self.wc_source != "global" and getattr(self, "_wc_origin", None) is not None:
                data["world_gps_origin"] = [float(self._wc_origin[0]), float(self._wc_origin[1])]
        else:
            self.get_logger().warn(f"World correction skipped: {wc_info}")

        try:
            base = self._resolve_base_path(req.output_path)
            out_path = self._write_yaml(data, base_path=base, append_timestamp=bool(req.append_timestamp))
            world_path = None
            if trees_w is not None and self.wc_separate:
                dw = dict(data)
                dw["trees"], dw["robot_path"] = trees_w, path_w
                dw.pop("trees_world")
                dw.pop("robot_path_world")
                world_path = self._write_yaml(dw, base_path=out_path.with_name(f"{out_path.stem}_world{out_path.suffix}"),
                                              append_timestamp=False)
        except Exception as e:
            res.success = False
            res.message = f"Failed writing YAML: {e}"
            res.written_path = ""
            return res

        res.success = True
        res.written_path = str(out_path)
        res.message = (
            f"Wrote YAML: {out_path} ({len(trees)} trunks, {len(robot_path)} robot poses). "
            f"odom_corr={'yes' if self._latest_odom_corr_pose else 'no'}, "
            f"gps_fix={'yes' if self._latest_gps_fix else 'no'}, "
            f"final_gps={'yes' if self._latest_final_gps_list else 'no'}, "
            f"row_yaw={'yes' if row_yaw_rad is not None else 'no'}, "
            f"world_correction={'yes (' + wc_info + ')' if trees_w is not None else 'no (' + wc_info + ')'}"
            + (f", world map: {world_path}" if world_path else "") + "."
        )
        return res

    def _write_yaml(self, data: Dict, *, base_path: Path, append_timestamp: bool) -> Path:
        if append_timestamp:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = base_path.with_name(f"{base_path.stem}_{ts}{base_path.suffix}")
        else:
            out_path = base_path

        tmp = out_path.with_suffix(out_path.suffix + ".tmp")

        with open(tmp, "w") as f:
            corr = data.get("initial_odom_correction")
            yaw = data.get("initial_yaw_correction_rad")
            row_yaw = data.get("row_yaw_rad")
            gps = data.get("initial_gps_fix")
            final_gps = data.get("final_gps_fix")
            raw_gps = data.get("raw_gps_coords", [])
            robot_path = data.get("robot_path", [])
            trees = data.get("trees", {})

            f.write("initial_odom_correction: " + ("null\n" if corr is None else f"[{corr[0]:.6f}, {corr[1]:.6f}, {corr[2]:.6f}]\n"))
            f.write("initial_yaw_correction_rad: " + ("null\n" if yaw is None else f"{yaw:.8f}\n"))
            f.write("row_yaw_rad: " + ("null\n" if row_yaw is None else f"{row_yaw:.8f}\n"))
            f.write("initial_gps_fix: " + ("null\n" if gps is None else f"[{gps[0]}, {gps[1]}, {gps[2]}]\n"))
            f.write("final_gps_fix: " + ("null\n" if final_gps is None else f"[{final_gps[0]}, {final_gps[1]}, {final_gps[2]}]\n"))

            f.write("raw_gps_coords:\n")
            if not raw_gps:
                f.write("  []\n")
            else:
                for reading in raw_gps:
                    f.write(f"  - [{reading[0]:.10f}, {reading[1]:.10f}, {reading[2]:.4f}]\n")

            f.write("robot_path:\n")
            if not robot_path:
                f.write("  []\n")
            else:
                for r in robot_path:
                    f.write(f"  - [{r[0]:.6f}, {r[1]:.6f}, {r[2]:.8f}, {r[3]:.9f}]\n")

            f.write("trees:\n")
            if not trees:
                f.write("  {}\n")
            else:
                for i in sorted(trees.keys()):
                    xy = trees[i]
                    f.write(f"  {i}: [{xy[0]:.6f}, {xy[1]:.6f}]\n")

            # World frame origin (source gps: trees_world / *_world.yaml are ENU metres from this lat/lon)
            if data.get("world_gps_origin") is not None:
                o = data["world_gps_origin"]
                f.write(f"world_gps_origin: [{o[0]:.10f}, {o[1]:.10f}]\n")

            # World-corrected map, when available
            if "trees_world" in data:
                f.write("robot_path_world:\n")
                for r in data.get("robot_path_world") or []:
                    f.write(f"  - [{r[0]:.6f}, {r[1]:.6f}, {r[2]:.8f}, {r[3]:.9f}]\n")
                f.write("trees_world:\n")
                tw = data["trees_world"]
                if not tw:
                    f.write("  {}\n")
                for i in sorted(tw.keys()):
                    f.write(f"  {i}: [{tw[i][0]:.6f}, {tw[i][1]:.6f}]\n")

        tmp.replace(out_path)
        self.get_logger().info(f"Wrote YAML: {out_path}")
        return out_path


def main(args=None):
    rclpy.init(args=args)
    node = SaveBestParticleMap()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()