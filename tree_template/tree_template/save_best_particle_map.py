#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from std_srvs.srv import Trigger
from geometry_msgs.msg import Point
from tree_template_interfaces.msg import TrunkRegistry


def _f(x: float) -> float:
    return float(x)


def _point_to_list(p: Optional[Point]) -> Optional[List[float]]:
    if p is None:
        return None
    return [_f(p.x), _f(p.y), _f(p.z)]


class SaveBestParticleMap(Node):
    """
    Single-shot YAML dump node. Trigger service writes one YAML file with a timestamp suffix.
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

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self._latest_registry: Optional[TrunkRegistry] = None
        self._latest_odom_corr: Optional[Point] = None
        self._latest_gps_fix: Optional[Point] = None

        self.create_subscription(TrunkRegistry, self.trunk_topic, self._registry_cb, qos)
        self.create_subscription(Point, self.odom_corr_topic, self._odom_corr_cb, qos)
        self.create_subscription(Point, self.gps_topic, self._gps_cb, qos)

        self.srv = self.create_service(Trigger, self.service_name, self._on_trigger)

        self.get_logger().info(
            f"Subscribing:\n"
            f"  trunk registry:          {self.trunk_topic}\n"
            f"  initial odom correction: {self.odom_corr_topic}\n"
            f"  initial gps fix:         {self.gps_topic}\n"
            f"Output YAML base: {self.out_path}\n"
            f"Service: /{self.service_name}\n"
            f"require_odom_correction={self.require_odom_correction}, "
            f"require_gps_fix={self.require_gps_fix}, require_trunks={self.require_trunks}"
        )

    def _registry_cb(self, msg: TrunkRegistry):
        self._latest_registry = msg

    def _odom_corr_cb(self, msg: Point):
        self._latest_odom_corr = msg

    def _gps_cb(self, msg: Point):
        # x=lat, y=lon, z=alt(m)
        self._latest_gps_fix = msg

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

        if self.require_odom_correction and self._latest_odom_corr is None:
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

        data = {
            "initial_odom_correction": _point_to_list(self._latest_odom_corr),
            "initial_gps_fix": _point_to_list(self._latest_gps_fix),
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
            f"Wrote YAML: {out_path} ({len(trees)} trunks). "
            f"odom_corr={'yes' if self._latest_odom_corr else 'no'}, gps_fix={'yes' if self._latest_gps_fix else 'no'}."
        )
        return res

    def _write_yaml(self, data: Dict) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        out_path = self.out_path.with_name(
            f"{self.out_path.stem}_{ts}{self.out_path.suffix}"
        )
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")

        with open(tmp, "w") as f:
            corr = data.get("initial_odom_correction")
            gps = data.get("initial_gps_fix")
            trees = data.get("trees", {})

            if corr is None:
                f.write("initial_odom_correction: null\n")
            else:
                f.write(f"initial_odom_correction: [{corr[0]:.6f}, {corr[1]:.6f}, {corr[2]:.6f}]\n")

            if gps is None:
                f.write("initial_gps_fix: null\n")
            else:
                f.write(f"initial_gps_fix: [{gps[0]:.8f}, {gps[1]:.8f}, {gps[2]:.3f}]\n")

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