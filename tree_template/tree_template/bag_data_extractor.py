#!/usr/bin/env python3
from __future__ import annotations

import os
import csv
import json
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, Tuple, Type, Optional, Any, List

import rclpy
from rclpy.node import Node
from rclpy.serialization import serialize_message
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from nav_msgs.msg import Odometry
from tree_template_interfaces.msg import TrunkInfo
from geometry_msgs.msg import Pose, Point
from sensor_msgs.msg import NavSatFix

# Optional dependency: PyYAML. If not available, we'll write a minimal YAML manually.
try:
    import yaml  # type: ignore
    _HAVE_YAML = True
except Exception:
    _HAVE_YAML = False


def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + 1e-9 * float(stamp.nanosec)


def _pose_position_to_list(p: Optional[Pose]):
    if p is None:
        return None
    return [float(p.position.x), float(p.position.y), float(p.position.z)]


def _point_to_list(p: Optional[Point]):
    if p is None:
        return None
    return [float(p.x), float(p.y), float(p.z)]


def _navsat_to_list(msg: Optional[NavSatFix]):
    if msg is None:
        return None
    lat = float(msg.latitude)
    lon = float(msg.longitude)
    alt = float(msg.altitude)
    if not (lat == lat and lon == lon and alt == alt):  # NaN check
        return None
    return [lat, lon, alt]


class BagDataExtractor(Node):
    """
    Records time-series topics to:
      dataset_dir/topics/<topic_name_sanitized>/<seq>.cdr
    and writes dataset_dir/index.csv

    Additionally captures initial topics (latched when available):
      - /initial_odom_correction (Pose)
      - /initial_gps_fix (Point)

    Additionally captures final GPS from EITHER:
      - final_gps_fix_point_topic (Point)   [replayer/dataset mode]
      - final_gps_fix_navsat_topic (NavSatFix) [raw bag mode, usually /fix]

    and writes dataset_dir/initials.yaml with:
      initial_odom_correction: [x,y,z]
      initial_gps_fix: [lat,lon,alt]
      final_gps_fix: [lat,lon,alt]
    """

    def __init__(
        self,
        out_dir: Path,
        topics: Dict[str, Tuple[Type, str]],
        initial_odom_topic: str = "/initial_odom_correction",
        initial_gps_topic: str = "/initial_gps_fix",
        final_gps_fix_point_topic: str = "/final_gps_fix",
        final_gps_fix_navsat_topic: str = "/fix",
        require_initial_odom: bool = False,
        require_initial_gps: bool = False,
        require_final_gps: bool = False,
    ):
        super().__init__("bag_data_extractor")

        self._shutdown_started = False

        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "topics").mkdir(exist_ok=True)

        self.index_fp = open(self.out_dir / "index.csv", "w", newline="")
        self.index_csv = csv.writer(self.index_fp)
        self.index_csv.writerow(["topic", "t_sec", "t_ros_sec", "t_ros_nsec", "seq", "filename", "nbytes"])

        self.seq_by_topic: Dict[str, int] = {}
        self.topic_dirs: Dict[str, Path] = {}

        # QoS for recorded “inputs” (time series)
        qos_inputs = QoSProfile(depth=200)
        qos_inputs.reliability = ReliabilityPolicy.RELIABLE
        qos_inputs.durability = DurabilityPolicy.VOLATILE

        # QoS for latched topics
        self.qos_latched = QoSProfile(depth=1)
        self.qos_latched.reliability = ReliabilityPolicy.RELIABLE
        self.qos_latched.durability = DurabilityPolicy.TRANSIENT_LOCAL

        # QoS for live GPS
        self.qos_gps_live = QoSProfile(depth=10)
        self.qos_gps_live.reliability = ReliabilityPolicy.BEST_EFFORT
        self.qos_gps_live.durability = DurabilityPolicy.VOLATILE

        self.subs = []
        for topic_name, (msg_type, stamp_mode) in topics.items():
            safe_topic_dir = topic_name.strip("/").replace("/", "__")
            tdir = self.out_dir / "topics" / safe_topic_dir
            tdir.mkdir(parents=True, exist_ok=True)
            self.topic_dirs[topic_name] = tdir
            self.seq_by_topic[topic_name] = 0

            sub = self.create_subscription(
                msg_type,
                topic_name,
                lambda msg, tn=topic_name, mt=msg_type, sm=stamp_mode: self._cb(msg, tn, mt, sm),
                qos_inputs,
            )
            self.subs.append(sub)

        # ---- Initials capture ----
        self.initial_odom_topic = initial_odom_topic
        self.initial_gps_topic = initial_gps_topic
        self.final_gps_fix_point_topic = final_gps_fix_point_topic
        self.final_gps_fix_navsat_topic = final_gps_fix_navsat_topic

        self.require_initial_odom = bool(require_initial_odom)
        self.require_initial_gps = bool(require_initial_gps)
        self.require_final_gps = bool(require_final_gps)

        self._init_odom: Optional[Pose] = None
        self._init_gps: Optional[Point] = None
        self._final_gps_list: Optional[List[float]] = None  # always [lat,lon,alt]

        # Prefer latched for initial fields
        self.create_subscription(Pose, self.initial_odom_topic, self._init_odom_cb, self.qos_latched)
        self.create_subscription(Point, self.initial_gps_topic, self._init_gps_cb, self.qos_latched)

        # Final GPS: accept either Point or NavSatFix
        self.create_subscription(Point, self.final_gps_fix_point_topic, self._final_gps_point_cb, self.qos_latched)
        self.create_subscription(NavSatFix, self.final_gps_fix_navsat_topic, self._final_gps_navsat_cb, self.qos_gps_live)

        meta = {
            "format": "ros2_cdr_per_message",
            "topics": {k: {"type": v[0].__name__, "stamp_mode": v[1]} for k, v in topics.items()},
            "initials": {
                "initial_odom_topic": self.initial_odom_topic,
                "initial_gps_topic": self.initial_gps_topic,
                "final_gps_fix_point_topic": self.final_gps_fix_point_topic,
                "final_gps_fix_navsat_topic": self.final_gps_fix_navsat_topic,
                "require_initial_odom": self.require_initial_odom,
                "require_initial_gps": self.require_initial_gps,
                "require_final_gps": self.require_final_gps,
                "file": "initials.yaml",
            },
        }
        (self.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

        self.get_logger().info(f"Recording to: {str(self.out_dir)}")
        for tn in topics:
            self.get_logger().info(f"  topic: {tn}")
        self.get_logger().info("Also capturing (TRANSIENT_LOCAL if available):")
        self.get_logger().info(f"  {self.initial_odom_topic} (Pose)")
        self.get_logger().info(f"  {self.initial_gps_topic} (Point: lat,lon,alt)")
        self.get_logger().info(f"  {self.final_gps_fix_point_topic} (Point: lat,lon,alt)")
        self.get_logger().info(f"  {self.final_gps_fix_navsat_topic} (NavSatFix: lat,lon,alt)")
        self.get_logger().info("initials.yaml will be written on Ctrl+C (and again on shutdown).")

    # ------------------- time-series callback -------------------

    def _cb(self, msg, topic_name: str, msg_type: Type, stamp_mode: str):
        if stamp_mode == "header" and hasattr(msg, "header"):
            t_ros = msg.header.stamp
        elif stamp_mode == "trunkinfo_stamp" and hasattr(msg, "stamp"):
            t_ros = msg.stamp
        else:
            t_ros = self.get_clock().now().to_msg()

        t_sec = stamp_to_sec(t_ros)

        seq = self.seq_by_topic[topic_name]
        self.seq_by_topic[topic_name] = seq + 1

        filename = f"{seq:010d}.cdr"
        path = self.topic_dirs[topic_name] / filename

        b = serialize_message(msg)
        path.write_bytes(b)

        self.index_csv.writerow(
            [
                topic_name,
                f"{t_sec:.9f}",
                t_ros.sec,
                t_ros.nanosec,
                seq,
                str(path.relative_to(self.out_dir)),
                len(b),
            ]
        )

        if (seq % 50) == 0:
            self.index_fp.flush()

    # ------------------- initials callbacks -------------------

    def _init_odom_cb(self, msg: Pose):
        if self._init_odom is None:
            self._init_odom = msg
            self.get_logger().info("Captured initial odom correction (Pose).")

    def _init_gps_cb(self, msg: Point):
        if self._init_gps is None:
            self._init_gps = msg
            self.get_logger().info("Captured initial GPS fix (Point).")

    def _final_gps_point_cb(self, msg: Point):
        if not np.isfinite(msg.x) or not np.isfinite(msg.y):
            return
        alt = float(msg.z) if np.isfinite(msg.z) else 0.0
        self._final_gps_list = [float(msg.x), float(msg.y), alt]

    def _final_gps_navsat_cb(self, msg: NavSatFix):
        v = _navsat_to_list(msg)
        if v is None:
            return
        self._final_gps_list = v

    # ------------------- YAML writing -------------------

    def write_initials_yaml(self):
        initials_path = self.out_dir / "initials.yaml"
        tmp_path = initials_path.with_suffix(".yaml.tmp")

        data = {
            "initial_odom_correction": _pose_position_to_list(self._init_odom),
            "initial_gps_fix": _point_to_list(self._init_gps),
            "final_gps_fix": self._final_gps_list,
        }

        if _HAVE_YAML:
            with open(tmp_path, "w") as f:
                yaml.safe_dump(data, f, sort_keys=False)
        else:
            with open(tmp_path, "w") as f:
                corr = data["initial_odom_correction"]
                igps = data["initial_gps_fix"]
                fgps = data["final_gps_fix"]

                if corr is None:
                    f.write("initial_odom_correction: null\n")
                else:
                    f.write(f"initial_odom_correction: [{corr[0]:.6f}, {corr[1]:.6f}, {corr[2]:.6f}]\n")

                if igps is None:
                    f.write("initial_gps_fix: null\n")
                else:
                    f.write(f"initial_gps_fix: [{igps[0]:.8f}, {igps[1]:.8f}, {igps[2]:.3f}]\n")

                if fgps is None:
                    f.write("final_gps_fix: null\n")
                else:
                    f.write(f"final_gps_fix: [{fgps[0]:.8f}, {fgps[1]:.8f}, {fgps[2]:.3f}]\n")

        tmp_path.replace(initials_path)

        # Avoid logging if shutdown already started
        if not self._shutdown_started:
            self.get_logger().info(
                f"Wrote initials.yaml: have_odom={self._init_odom is not None}, "
                f"have_init_gps={self._init_gps is not None}, have_final_gps={self._final_gps_list is not None}"
            )

    def destroy_node(self):
        # Try to flush and write once more on shutdown
        try:
            self.write_initials_yaml()
        except Exception:
            pass

        try:
            self.index_fp.flush()
            self.index_fp.close()
        except Exception:
            pass

        if self.require_initial_odom and self._init_odom is None and not self._shutdown_started:
            self.get_logger().warn("require_initial_odom=True but initial odom was never captured.")
        if self.require_initial_gps and self._init_gps is None and not self._shutdown_started:
            self.get_logger().warn("require_initial_gps=True but initial gps was never captured.")
        if self.require_final_gps and self._final_gps_list is None and not self._shutdown_started:
            self.get_logger().warn("require_final_gps=True but final gps was never captured.")

        super().destroy_node()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/marcus/apple_harvest_ws/src/orchard_tree_templating/tree_template/extracted_bag_data/2023-10-17-16-32-29_267_speedup", help="Output dataset directory")

    ap.add_argument("--initial_odom_topic", default="/initial_odom_correction", help="Initial odom correction topic (Pose)")
    ap.add_argument("--initial_gps_topic", default="/initial_gps_fix", help="Initial GPS fix topic (Point)")

    # final GPS: accept either a Point topic or NavSatFix topic
    ap.add_argument("--final_gps_fix_point_topic", default="/final_gps_fix", help="Final GPS topic as Point(lat,lon,alt)")
    ap.add_argument("--final_gps_fix_navsat_topic", default="/fix", help="Final GPS topic as NavSatFix(lat,lon,alt)")

    ap.add_argument("--require_initial_odom", action="store_true")
    ap.add_argument("--require_initial_gps", action="store_true")
    ap.add_argument("--require_final_gps", action="store_true")

    args = ap.parse_args()

    out_dir = Path(args.out)

    rclpy.init()

    topics = {
        "/odometry/filtered": (Odometry, "header"),
        "trunk_measurements_raw": (TrunkInfo, "trunkinfo_stamp"),
    }

    node = BagDataExtractor(
        out_dir=out_dir,
        topics=topics,
        initial_odom_topic=str(args.initial_odom_topic),
        initial_gps_topic=str(args.initial_gps_topic),
        final_gps_fix_point_topic=str(args.final_gps_fix_point_topic),
        final_gps_fix_navsat_topic=str(args.final_gps_fix_navsat_topic),
        require_initial_odom=bool(args.require_initial_odom),
        require_initial_gps=bool(args.require_initial_gps),
        require_final_gps=bool(args.require_final_gps),
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node._shutdown_started = True
        try:
            # write BEFORE teardown; don't rely on rosout
            print("Ctrl+C received, writing initials.yaml before shutdown...")
            node.write_initials_yaml()
        except Exception as e:
            print(f"Failed writing initials.yaml on Ctrl+C: {e}")
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        # Guard against double-shutdown
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
