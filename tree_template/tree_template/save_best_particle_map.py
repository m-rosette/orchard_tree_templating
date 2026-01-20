#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

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


class TrunkRegistryDumpToYaml(Node):
    """
    Single-shot YAML dump node.

    Subscribes to:
      - TrunkRegistry (best particle)
      - initial_odom_correction (geometry_msgs/Point)
      - initial_gps_fix (geometry_msgs/Point) where:
          x=latitude, y=longitude, z=altitude_m

    On Trigger service call:
      - Writes ONE YAML file containing the latest values
      - Overwrites any existing file

    YAML format:

    initial_odom_correction: [x, y, z]
    initial_gps_fix: [lat, lon, alt_m]
    trees:
      0: [x, y]
      1: [x, y]
      ...
    """

    def __init__(
        self,
        trunk_topic: str,
        odom_corr_topic: str,
        gps_topic: str,
        out_path: str,
        require_odom_correction: bool,
        require_gps_fix: bool,
        require_trunks: bool,
    ):
        super().__init__("trunk_registry_dump_to_yaml")

        self.trunk_topic = trunk_topic
        self.odom_corr_topic = odom_corr_topic
        self.gps_topic = gps_topic
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)

        self.require_odom_correction = bool(require_odom_correction)
        self.require_gps_fix = bool(require_gps_fix)
        self.require_trunks = bool(require_trunks)

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

        self.srv = self.create_service(Trigger, "dump_trunk_yaml", self._on_trigger)

        self.get_logger().info(
            f"Subscribing:\n"
            f"  trunk registry:         {self.trunk_topic}\n"
            f"  initial odom correction:{self.odom_corr_topic}\n"
            f"  initial gps fix:        {self.gps_topic}\n"
            f"Output YAML: {self.out_path}\n"
            f"Service: /dump_trunk_yaml\n"
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
            self._write_yaml(data)
        except Exception as e:
            res.success = False
            res.message = f"Failed writing YAML: {e}"
            return res

        res.success = True
        res.message = (
            f"Wrote YAML with {len(trees)} trunks. "
            f"odom_corr={'yes' if self._latest_odom_corr else 'no'}, gps_fix={'yes' if self._latest_gps_fix else 'no'}."
        )
        return res

    def _write_yaml(self, data: Dict):
        tmp = self.out_path.with_suffix(self.out_path.suffix + ".tmp")
        with open(tmp, "w") as f:
            corr = data.get("initial_odom_correction")
            gps = data.get("initial_gps_fix")
            trees = data.get("trees", {})

            # initial_odom_correction
            if corr is None:
                f.write("initial_odom_correction: null\n")
            else:
                f.write(
                    f"initial_odom_correction: "
                    f"[{corr[0]:.6f}, {corr[1]:.6f}, {corr[2]:.6f}]\n"
                )

            # initial_gps_fix: [lat, lon, alt]
            if gps is None:
                f.write("initial_gps_fix: null\n")
            else:
                # keep more precision for lat/lon
                f.write(
                    f"initial_gps_fix: "
                    f"[{gps[0]:.8f}, {gps[1]:.8f}, {gps[2]:.3f}]\n"
                )

            # trees
            f.write("trees:\n")
            if not trees:
                f.write("  {}\n")
            else:
                for i in sorted(trees.keys()):
                    xy = trees[i]
                    f.write(f"  {i}: [{xy[0]:.6f}, {xy[1]:.6f}]\n")

        tmp.replace(self.out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trunk_topic", default="/fastslam_registry")
    ap.add_argument("--odom_corr_topic", default="/initial_odom_correction")
    ap.add_argument("--gps_topic", default="/initial_gps_fix")
    ap.add_argument("--out", default="/home/marcus/apple_harvest_ws/src/orchard_tree_templating/tree_template/slam_measurements/slam_trunks.yaml")

    ap.add_argument("--require_odom_correction", action="store_true")
    ap.add_argument("--require_gps_fix", action="store_true")
    ap.add_argument("--require_trunks", action="store_true")

    args = ap.parse_args()

    rclpy.init()
    node = TrunkRegistryDumpToYaml(
        trunk_topic=args.trunk_topic,
        odom_corr_topic=args.odom_corr_topic,
        gps_topic=args.gps_topic,
        out_path=args.out,
        require_odom_correction=args.require_odom_correction,
        require_gps_fix=args.require_gps_fix,
        require_trunks=args.require_trunks,
    )
    try:
        rclpy.spin(node)
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()