#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import math
from collections import deque

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from std_srvs.srv import Trigger
from geometry_msgs.msg import Point, Pose
from nav_msgs.msg import Odometry  # NEW
from tree_template_interfaces.msg import TrunkRegistry

from scipy.spatial.transform import Rotation as R


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
    """Return yaw (rad) from Pose quaternion (ENU, yaw about +Z)."""
    if pose is None:
        return None
    q = pose.orientation
    quat = np.array([_f(q.x), _f(q.y), _f(q.z), _f(q.w)], dtype=np.float64)
    if not np.all(np.isfinite(quat)):
        return None
    yaw = float(R.from_quat(quat).as_euler("xyz", degrees=False)[2])
    return yaw if math.isfinite(yaw) else None


def _odom_xy_yaw_t(msg: Odometry) -> Optional[List[float]]:
    """Return [x, y, yaw, t_sec] from nav_msgs/Odometry."""
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


class SaveBestParticleMap(Node):
    """
    Single-shot YAML dump node. Trigger service writes one YAML file with a timestamp suffix.
    Also stores a downsampled robot path from an Odometry topic.
    """

    def __init__(self):
        super().__init__("save_best_particle_map")

        # ----------------------------
        # Parameters (ROS2)
        # ----------------------------
        self.declare_parameter("trunk_topic", "/fastslam_registry")
        self.declare_parameter("odom_corr_topic", "/initial_odom_correction")
        self.declare_parameter("gps_topic", "/initial_gps_fix")
        self.declare_parameter(
            "out",
            "/home/marcus/apple_harvest_ws/src/orchard_tree_templating/tree_template/slam_measurements/slam_trunks.yaml",
        )

        self.declare_parameter("require_odom_correction", False)
        self.declare_parameter("require_gps_fix", False)
        self.declare_parameter("require_trunks", False)

        self.declare_parameter("service_name", "dump_trunk_yaml")

        # NEW: robot path capture
        self.declare_parameter("robot_odom_topic", "/odom_slam_best")
        self.declare_parameter("robot_min_dt_sec", 0.10)     # downsample by time
        self.declare_parameter("robot_min_dist_m", 0.05)     # or by distance
        self.declare_parameter("robot_max_points", 5000)     # cap memory + yaml size

        # Resolve params
        self.trunk_topic = str(self.get_parameter("trunk_topic").value)
        self.odom_corr_topic = str(self.get_parameter("odom_corr_topic").value)
        self.gps_topic = str(self.get_parameter("gps_topic").value)
        self.out_path = Path(str(self.get_parameter("out").value))
        self.out_path.parent.mkdir(parents=True, exist_ok=True)

        self.require_odom_correction = bool(self.get_parameter("require_odom_correction").value)
        self.require_gps_fix = bool(self.get_parameter("require_gps_fix").value)
        self.require_trunks = bool(self.get_parameter("require_trunks").value)

        self.service_name = str(self.get_parameter("service_name").value)

        self.robot_odom_topic = str(self.get_parameter("robot_odom_topic").value)
        self.robot_min_dt_sec = float(self.get_parameter("robot_min_dt_sec").value)
        self.robot_min_dist_m = float(self.get_parameter("robot_min_dist_m").value)
        self.robot_max_points = int(self.get_parameter("robot_max_points").value)

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        qos_latched = QoSProfile(depth=1)
        qos_latched.durability = DurabilityPolicy.TRANSIENT_LOCAL
        qos_latched.reliability = ReliabilityPolicy.RELIABLE

        self._latest_registry: Optional[TrunkRegistry] = None
        self._latest_odom_corr_pose: Optional[Pose] = None
        self._latest_gps_fix: Optional[Point] = None

        # NEW: robot path buffer (x, y, yaw, t_sec)
        self._robot_path: deque[List[float]] = deque(maxlen=max(self.robot_max_points, 1))
        self._robot_last_keep: Optional[List[float]] = None

        self.create_subscription(TrunkRegistry, self.trunk_topic, self._registry_cb, qos)
        self.create_subscription(Pose, self.odom_corr_topic, self._odom_corr_cb, qos_latched)
        self.create_subscription(Point, self.gps_topic, self._gps_cb, qos_latched)

        # NEW
        self.create_subscription(Odometry, self.robot_odom_topic, self._robot_odom_cb, qos)

        self.srv = self.create_service(Trigger, self.service_name, self._on_trigger)

        self.get_logger().info(
            f"Subscribing:\n"
            f"  trunk registry:          {self.trunk_topic}\n"
            f"  initial odom correction: {self.odom_corr_topic} (Pose)\n"
            f"  initial gps fix:         {self.gps_topic}\n"
            f"  robot odom path:         {self.robot_odom_topic} (Odometry)\n"
            f"Output YAML base: {self.out_path}\n"
            f"Service: /{self.service_name}\n"
            f"require_odom_correction={self.require_odom_correction}, "
            f"require_gps_fix={self.require_gps_fix}, require_trunks={self.require_trunks}\n"
            f"robot_min_dt_sec={self.robot_min_dt_sec}, robot_min_dist_m={self.robot_min_dist_m}, robot_max_points={self.robot_max_points}"
        )

    def _registry_cb(self, msg: TrunkRegistry):
        self._latest_registry = msg

    def _odom_corr_cb(self, msg: Pose):
        self._latest_odom_corr_pose = msg

    def _gps_cb(self, msg: Point):
        # x=lat, y=lon, z=alt(m)
        self._latest_gps_fix = msg

    # NEW
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
        dx = float(x - x0)
        dy = float(y - y0)
        dist = float(math.hypot(dx, dy))

        if (dt >= self.robot_min_dt_sec) or (dist >= self.robot_min_dist_m):
            self._robot_path.append(row)
            self._robot_last_keep = row

    def _on_trigger(self, req: Trigger.Request, res: Trigger.Response) -> Trigger.Response:
        if self._latest_registry is None:
            res.success = False
            res.message = "No TrunkRegistry received yet."
            return res

        trunks = list(self._latest_registry.trunks)
        if self.require_trunks and len(trunks) == 0:
            res.success = False
            res.message = "Latest TrunkRegistry has zero trunks."
            return res

        if self.require_odom_correction and self._latest_odom_corr_pose is None:
            res.success = False
            res.message = "No initial_odom_correction received yet."
            return res

        if self.require_gps_fix and self._latest_gps_fix is None:
            res.success = False
            res.message = "No initial_gps_fix received yet."
            return res

        trees: Dict[int, List[float]] = {}
        for i, t in enumerate(trunks):
            p = t.pose.position
            trees[i] = [_f(p.x), _f(p.y)]

        init_odom_list = _pose_position_to_list(self._latest_odom_corr_pose)
        init_yaw = _pose_yaw_rad(self._latest_odom_corr_pose)

        robot_path = list(self._robot_path)  # [[x,y,yaw,t], ...]

        data = {
            "initial_odom_correction": init_odom_list,
            "initial_yaw_correction_rad": init_yaw,
            "initial_gps_fix": _point_to_list(self._latest_gps_fix),
            "robot_path": robot_path,  # NEW
            "trees": trees,
        }

        try:
            out_path = self._write_yaml(data)
        except Exception as e:
            res.success = False
            res.message = f"Failed writing YAML: {e}"
            return res

        res.success = True
        res.message = (
            f"Wrote YAML: {out_path} ({len(trees)} trunks, {len(robot_path)} robot poses). "
            f"odom_corr={'yes' if self._latest_odom_corr_pose else 'no'}, "
            f"yaw={'yes' if init_yaw is not None else 'no'}, "
            f"gps_fix={'yes' if self._latest_gps_fix else 'no'}."
        )
        return res

    def _write_yaml(self, data: Dict) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = self.out_path.with_name(f"{self.out_path.stem}_{ts}{self.out_path.suffix}")
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")

        with open(tmp, "w") as f:
            corr = data.get("initial_odom_correction")
            yaw = data.get("initial_yaw_correction_rad")
            gps = data.get("initial_gps_fix")
            robot_path = data.get("robot_path", [])
            trees = data.get("trees", {})

            if corr is None:
                f.write("initial_odom_correction: null\n")
            else:
                f.write(f"initial_odom_correction: [{corr[0]:.6f}, {corr[1]:.6f}, {corr[2]:.6f}]\n")

            if yaw is None:
                f.write("initial_yaw_correction_rad: null\n")
            else:
                f.write(f"initial_yaw_correction_rad: {yaw:.8f}\n")

            if gps is None:
                f.write("initial_gps_fix: null\n")
            else:
                f.write(f"initial_gps_fix: [{gps[0]}, {gps[1]}, {gps[2]}]\n")

            # NEW
            f.write("robot_path:\n")
            if not robot_path:
                f.write("  []\n")
            else:
                # each row: [x, y, yaw, t_sec]
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
