#!/usr/bin/env python3
from __future__ import annotations

import os
import csv
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple, Type, Optional

import rclpy
from rclpy.node import Node
from rclpy.serialization import serialize_message
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from nav_msgs.msg import Odometry
from tree_template_interfaces.msg import TrunkInfo
from geometry_msgs.msg import Pose, Point

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


class BagDataExtractor(Node):
    """
    Records time-series topics to:
      dataset_dir/topics/<topic_name_sanitized>/<seq>.cdr
    and writes dataset_dir/index.csv

    Additionally captures initial topics (unstamped, published once/latched):
      - /initial_odom_correction (Pose)
      - /initial_gps_fix (Point)

    and writes dataset_dir/initials.yaml
    """

    def __init__(
        self,
        out_dir: Path,
        topics: Dict[str, Tuple[Type, str]],
        initial_odom_topic: str = "/initial_odom_correction",
        initial_gps_topic: str = "/initial_gps_fix",
        require_initial_odom: bool = False,
        require_initial_gps: bool = False,
    ):
        super().__init__("bag_data_extractor")

        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "topics").mkdir(exist_ok=True)

        # index.csv columns: topic, t_sec, t_ros_sec, t_ros_nsec, seq, filename, nbytes
        self.index_fp = open(self.out_dir / "index.csv", "w", newline="")
        self.index_csv = csv.writer(self.index_fp)
        self.index_csv.writerow(["topic", "t_sec", "t_ros_sec", "t_ros_nsec", "seq", "filename", "nbytes"])

        self.seq_by_topic: Dict[str, int] = {}
        self.topic_dirs: Dict[str, Path] = {}

        # QoS for recorded “inputs” (time series)
        qos_inputs = QoSProfile(depth=200)
        qos_inputs.reliability = ReliabilityPolicy.RELIABLE
        qos_inputs.durability = DurabilityPolicy.VOLATILE

        # QoS for initial topics: subscribe as TRANSIENT_LOCAL to receive latched messages
        self.qos_initials = QoSProfile(depth=1)
        self.qos_initials.reliability = ReliabilityPolicy.RELIABLE
        self.qos_initials.durability = DurabilityPolicy.TRANSIENT_LOCAL

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
        self.require_initial_odom = bool(require_initial_odom)
        self.require_initial_gps = bool(require_initial_gps)

        self._init_odom: Optional[Pose] = None
        self._init_gps: Optional[Point] = None
        self._initials_written = False

        self.create_subscription(Pose, self.initial_odom_topic, self._init_odom_cb, self.qos_initials)
        self.create_subscription(Point, self.initial_gps_topic, self._init_gps_cb, self.qos_initials)

        meta = {
            "format": "ros2_cdr_per_message",
            "topics": {k: {"type": v[0].__name__, "stamp_mode": v[1]} for k, v in topics.items()},
            "initials": {
                "initial_odom_topic": self.initial_odom_topic,
                "initial_gps_topic": self.initial_gps_topic,
                "require_initial_odom": self.require_initial_odom,
                "require_initial_gps": self.require_initial_gps,
                "file": "initials.yaml",
            },
        }
        (self.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

        self.get_logger().info(f"Recording to: {str(self.out_dir)}")
        for tn in topics:
            self.get_logger().info(f"  topic: {tn}")
        self.get_logger().info("Also capturing initials:")
        self.get_logger().info(f"  {self.initial_odom_topic} (Pose, latched)")
        self.get_logger().info(f"  {self.initial_gps_topic} (Point, latched)")
        self.get_logger().info("Initials will be written to: initials.yaml")

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
            self._maybe_write_initials()

    def _init_gps_cb(self, msg: Point):
        if self._init_gps is None:
            self._init_gps = msg
            self.get_logger().info("Captured initial GPS fix (Point).")
            self._maybe_write_initials()

    def _maybe_write_initials(self):
        """
        Write initials.yaml.
        Policy:
          - Write as soon as we have at least one of them (so you always get something).
          - Overwrite once when both are present.
        """
        if self._init_odom is None and self._init_gps is None:
            return

        initials_path = self.out_dir / "initials.yaml"
        tmp_path = initials_path.with_suffix(".yaml.tmp")

        data = {
            "initial_odom_correction": _pose_position_to_list(self._init_odom),
            "initial_gps_fix": _point_to_list(self._init_gps),
        }

        if _HAVE_YAML:
            with open(tmp_path, "w") as f:
                yaml.safe_dump(data, f, sort_keys=False)
        else:
            # Minimal YAML writer (enough for your use case)
            with open(tmp_path, "w") as f:
                corr = data["initial_odom_correction"]
                gps = data["initial_gps_fix"]
                if corr is None:
                    f.write("initial_odom_correction: null\n")
                else:
                    f.write(f"initial_odom_correction: [{corr[0]:.6f}, {corr[1]:.6f}, {corr[2]:.6f}]\n")
                if gps is None:
                    f.write("initial_gps_fix: null\n")
                else:
                    f.write(f"initial_gps_fix: [{gps[0]:.8f}, {gps[1]:.8f}, {gps[2]:.3f}]\n")

        tmp_path.replace(initials_path)

        have_both = (self._init_odom is not None) and (self._init_gps is not None)
        self._initials_written = have_both

        self.get_logger().info(
            f"Wrote initials.yaml (have_odom={self._init_odom is not None}, have_gps={self._init_gps is not None})."
        )

    def destroy_node(self):
        try:
            self.index_fp.flush()
            self.index_fp.close()
        except Exception:
            pass

        # Enforce requirements if requested
        if self.require_initial_odom and self._init_odom is None:
            self.get_logger().warn("require_initial_odom=True but initial odom was never captured.")
        if self.require_initial_gps and self._init_gps is None:
            self.get_logger().warn("require_initial_gps=True but initial gps was never captured.")

        super().destroy_node()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default="/home/marcus/apple_harvest_ws/src/orchard_tree_templating/tree_template/extracted_bag_data/2023-10-17-16-32-29_267_gps_fix",
        help="Output dataset directory",
    )
    ap.add_argument("--initial_odom_topic", default="/initial_odom_correction", help="Initial odom correction topic (Pose)")
    ap.add_argument("--initial_gps_topic", default="/initial_gps_fix", help="Initial GPS fix topic (Point)")
    ap.add_argument("--require_initial_odom", action="store_true", help="Fail (warn) if initial odom not captured")
    ap.add_argument("--require_initial_gps", action="store_true", help="Fail (warn) if initial gps not captured")
    args = ap.parse_args()

    out_dir = Path(args.out)

    rclpy.init()

    topics = {
        "/odometry/filtered": (Odometry, "header"),
        "trunk_measurements_raw": (TrunkInfo, "trunkinfo_stamp"),
        # Add more time-series topics here if needed
        # "row_datum_pose": (PoseStamped, "header"),
    }

    node = BagDataExtractor(
        out_dir=out_dir,
        topics=topics,
        initial_odom_topic=str(args.initial_odom_topic),
        initial_gps_topic=str(args.initial_gps_topic),
        require_initial_odom=bool(args.require_initial_odom),
        require_initial_gps=bool(args.require_initial_gps),
    )
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
