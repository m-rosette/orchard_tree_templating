#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import math
from collections import deque

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import Point, Pose
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix
from tree_template_interfaces.msg import TrunkRegistry
from tree_template_interfaces.srv import SaveSlamResults

from scipy.spatial.transform import Rotation as R

try:
    import yaml  # type: ignore
    _HAVE_YAML = True
except Exception:
    _HAVE_YAML = False


def _f(x: float) -> float:
    return float(x)


def _point_to_list(p: Optional[Point]) -> Optional[List[float]]:
    if p is None:
        return None
    return [_f(p.x), _f(p.y), _f(p.z)]


def _pose_position_to_list(pose: Optional[Pose]) -> Optional[List[float]]:
    if pose is None:
        return None
    p = pose.position
    return [_f(p.x), _f(p.y), _f(p.z)]


def _pose_yaw_rad(pose: Optional[Pose]) -> Optional[float]:
    if pose is None:
        return None
    q = pose.orientation
    quat = np.array([_f(q.x), _f(q.y), _f(q.z), _f(q.w)], dtype=np.float64)
    if not np.all(np.isfinite(quat)):
        return None
    yaw = float(R.from_quat(quat).as_euler("xyz", degrees=False)[2])
    return yaw if math.isfinite(yaw) else None


def _odom_xy_yaw_t(msg: Odometry) -> Optional[List[float]]:
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    quat = np.array([_f(q.x), _f(q.y), _f(q.z), _f(q.w)], dtype=np.float64)
    if not (math.isfinite(p.x) and math.isfinite(p.y) and np.all(np.isfinite(quat))):
        return None
    yaw = float(R.from_quat(quat).as_euler("xyz", degrees=False)[2])
    if not math.isfinite(yaw):
        return None
    t = msg.header.stamp
    t_sec = float(t.sec) + 1e-9 * float(t.nanosec)
    return [_f(p.x), _f(p.y), float(yaw), float(t_sec)]


def _safe_list3(v: Any) -> Optional[List[float]]:
    if not isinstance(v, (list, tuple)) or len(v) < 3:
        return None
    try:
        out = [float(v[0]), float(v[1]), float(v[2])]
        if not all(np.isfinite(out)):
            return None
        return out
    except Exception:
        return None


def _read_initials_yaml(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        if _HAVE_YAML:
            return yaml.safe_load(path.read_text()) or {}
        data: dict = {}
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            k, rhs = line.split(":", 1)
            k = k.strip()
            rhs = rhs.strip()
            if rhs == "null":
                data[k] = None
            else:
                data[k] = eval(rhs)
        return data
    except Exception:
        return None


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

        # YAML fallback
        self.declare_parameter("initials_yaml", "")
        self.declare_parameter("use_initials_yaml_fallback", False)

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

        # YAML fallback cache
        self._initials_from_yaml: Optional[dict] = None
        if str(self.initials_yaml) and self.initials_yaml.exists():
            self._initials_from_yaml = _read_initials_yaml(self.initials_yaml)
            if self._initials_from_yaml is not None:
                self.get_logger().info(f"Loaded initials YAML: {self.initials_yaml}")
            else:
                self.get_logger().warn(f"Failed to parse initials YAML: {self.initials_yaml}")

        # Robot path buffer
        self._robot_path: deque[List[float]] = deque(maxlen=max(self.robot_max_points, 1))
        self._robot_last_keep: Optional[List[float]] = None

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

    def _final_gps_navsat_cb(self, msg: NavSatFix):
        if not np.isfinite(msg.latitude) or not np.isfinite(msg.longitude):
            return
        alt = float(msg.altitude) if np.isfinite(msg.altitude) else 0.0
        self._latest_final_gps_list = [float(msg.latitude), float(msg.longitude), alt]

    def _final_gps_point_cb(self, msg: Point):
        if not np.isfinite(msg.x) or not np.isfinite(msg.y):
            return
        alt = float(msg.z) if np.isfinite(msg.z) else 0.0
        self._latest_final_gps_list = [float(msg.x), float(msg.y), alt]

    def _fill_from_yaml_if_needed(self):
        if not self.use_initials_yaml_fallback:
            return
        if not self._initials_from_yaml:
            return

        if self._latest_odom_corr_pose is None:
            corr = _safe_list3(self._initials_from_yaml.get("initial_odom_correction"))
            if corr is not None:
                p = Pose()
                p.position.x, p.position.y, p.position.z = corr[0], corr[1], corr[2]
                p.orientation.w = 1.0
                self._latest_odom_corr_pose = p

        if self._latest_gps_fix is None:
            igps = _safe_list3(self._initials_from_yaml.get("initial_gps_fix"))
            if igps is not None:
                pt = Point()
                pt.x, pt.y, pt.z = igps[0], igps[1], igps[2]
                self._latest_gps_fix = pt

        if self._latest_final_gps_list is None:
            fgps = _safe_list3(self._initials_from_yaml.get("final_gps_fix"))
            if fgps is not None:
                self._latest_final_gps_list = fgps

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
        self._fill_from_yaml_if_needed()

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
            trees[i] = [_f(p.x), _f(p.y)]

        init_odom_list = _pose_position_to_list(self._latest_odom_corr_pose)
        init_yaw = _pose_yaw_rad(self._latest_odom_corr_pose)
        robot_path = list(self._robot_path)

        data = {
            "initial_odom_correction": init_odom_list,
            "initial_yaw_correction_rad": init_yaw,
            "initial_gps_fix": _point_to_list(self._latest_gps_fix),
            "final_gps_fix": self._latest_final_gps_list,
            "robot_path": robot_path,
            "trees": trees,
        }

        try:
            base = self._resolve_base_path(req.output_path)
            out_path = self._write_yaml(data, base_path=base, append_timestamp=bool(req.append_timestamp))
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
            f"final_gps={'yes' if self._latest_final_gps_list else 'no'}."
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
            gps = data.get("initial_gps_fix")
            final_gps = data.get("final_gps_fix")
            robot_path = data.get("robot_path", [])
            trees = data.get("trees", {})

            f.write("initial_odom_correction: " + ("null\n" if corr is None else f"[{corr[0]:.6f}, {corr[1]:.6f}, {corr[2]:.6f}]\n"))
            f.write("initial_yaw_correction_rad: " + ("null\n" if yaw is None else f"{yaw:.8f}\n"))
            f.write("initial_gps_fix: " + ("null\n" if gps is None else f"[{gps[0]}, {gps[1]}, {gps[2]}]\n"))
            f.write("final_gps_fix: " + ("null\n" if final_gps is None else f"[{final_gps[0]}, {final_gps[1]}, {final_gps[2]}]\n"))

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
