#!/usr/bin/env python3
from __future__ import annotations

import csv
import time
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Type, Optional

import rclpy
from rclpy.node import Node
from rclpy.serialization import deserialize_message
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from nav_msgs.msg import Odometry
from tree_template_interfaces.msg import TrunkInfo
from geometry_msgs.msg import Pose, Point

# Optional dependency: PyYAML. If not available, we’ll parse a very small subset manually.
try:
    import yaml  # type: ignore
    _HAVE_YAML = True
except Exception:
    _HAVE_YAML = False


@dataclass
class Row:
    topic: str
    t_sec: float
    relpath: str


def _read_initials_yaml(path: Path) -> Optional[dict]:
    if not path.exists():
        return None

    if _HAVE_YAML:
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}

    # Minimal parser fallback: handles the lines we write.
    data: dict = {}
    txt = path.read_text().splitlines()
    for line in txt:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("initial_odom_correction:"):
            rhs = line.split(":", 1)[1].strip()
            data["initial_odom_correction"] = None if rhs == "null" else eval(rhs)  # dataset-local file
            continue

        if line.startswith("initial_gps_fix:"):
            rhs = line.split(":", 1)[1].strip()
            data["initial_gps_fix"] = None if rhs == "null" else eval(rhs)
            continue

        if line.startswith("final_gps_fix:"):
            rhs = line.split(":", 1)[1].strip()
            data["final_gps_fix"] = None if rhs == "null" else eval(rhs)
            continue

    return data


class BagDataReplayer(Node):
    def __init__(
        self,
        dataset_dir: Path,
        speed: float = 1.0,
        initial_odom_topic: str = "/initial_odom_correction",
        initial_gps_topic: str = "/initial_gps_fix",
        final_gps_topic: str = "/fix",  # NEW
        publish_initials: bool = True,
        publish_initials_delay_s: float = 0.10,
    ):
        super().__init__("bag_data_replayer")

        self.dataset_dir = dataset_dir
        self.speed = max(float(speed), 1e-6)

        self.initial_odom_topic = initial_odom_topic
        self.initial_gps_topic = initial_gps_topic
        self.final_gps_topic = final_gps_topic  # NEW
        self.publish_initials = bool(publish_initials)
        self.publish_initials_delay_s = max(float(publish_initials_delay_s), 0.0)

        self.msg_types: Dict[str, Type] = {
            "/odometry/filtered": Odometry,
            "trunk_measurements_raw": TrunkInfo,
        }

        qos_ts = QoSProfile(depth=10)
        qos_ts.reliability = ReliabilityPolicy.RELIABLE
        qos_ts.durability = DurabilityPolicy.VOLATILE

        self.pubs = {}
        for topic, mt in self.msg_types.items():
            self.pubs[topic] = self.create_publisher(mt, topic, qos_ts)

        # Latched publishers for initials (publish once, but late subscribers still get them)
        qos_latched = QoSProfile(depth=1)
        qos_latched.reliability = ReliabilityPolicy.RELIABLE
        qos_latched.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self.init_pose_pub = self.create_publisher(Pose, self.initial_odom_topic, qos_latched)
        self.init_gps_pub = self.create_publisher(Point, self.initial_gps_topic, qos_latched)
        self.final_gps_pub = self.create_publisher(Point, self.final_gps_topic, qos_latched)  # NEW

        self.initials = _read_initials_yaml(self.dataset_dir / "initials.yaml")

        self.rows: List[Row] = self._load_index()
        self.get_logger().info(f"Loaded {len(self.rows)} messages from {str(self.dataset_dir)}")
        if self.initials is not None:
            self.get_logger().info("Loaded initials.yaml")
        else:
            self.get_logger().info("No initials.yaml found (will replay time-series only).")

    def _load_index(self) -> List[Row]:
        idx = self.dataset_dir / "index.csv"
        if not idx.exists():
            raise FileNotFoundError(f"Missing index.csv: {str(idx)}")

        rows: List[Row] = []
        with open(idx, "r") as f:
            r = csv.DictReader(f)
            for rr in r:
                topic = rr["topic"]
                if topic not in self.msg_types:
                    continue
                rows.append(Row(topic=topic, t_sec=float(rr["t_sec"]), relpath=rr["filename"]))
        rows.sort(key=lambda x: x.t_sec)
        return rows

    def _publish_initials_once(self):
        if not self.publish_initials:
            return
        if not self.initials:
            return

        corr = self.initials.get("initial_odom_correction", None)
        gps_i = self.initials.get("initial_gps_fix", None)
        gps_f = self.initials.get("final_gps_fix", None)  # NEW

        # initial_odom_correction: Pose with position; keep identity quaternion
        if corr is not None:
            try:
                pose = Pose()
                pose.position.x = float(corr[0])
                pose.position.y = float(corr[1])
                pose.position.z = float(corr[2])
                pose.orientation.w = 1.0
                self.init_pose_pub.publish(pose)
            except Exception as e:
                self.get_logger().warn(f"Failed to publish initial_odom_correction: {e}")

        # initial_gps_fix: Point (lat, lon, alt)
        if gps_i is not None:
            try:
                pt = Point()
                pt.x = float(gps_i[0])
                pt.y = float(gps_i[1])
                pt.z = float(gps_i[2])
                self.init_gps_pub.publish(pt)
            except Exception as e:
                self.get_logger().warn(f"Failed to publish initial_gps_fix: {e}")

        # final_gps_fix: Point (lat, lon, alt)  (NEW)
        if gps_f is not None:
            try:
                pt = Point()
                pt.x = float(gps_f[0])
                pt.y = float(gps_f[1])
                pt.z = float(gps_f[2])
                self.final_gps_pub.publish(pt)
            except Exception as e:
                self.get_logger().warn(f"Failed to publish final_gps_fix: {e}")

        if self.publish_initials_delay_s > 0.0:
            time.sleep(self.publish_initials_delay_s)

        self.get_logger().info(
            "Published initials once (latched): "
            f"have_odom={corr is not None}, have_init_gps={gps_i is not None}, have_final_gps={gps_f is not None}"
        )

    def run(self):
        if not self.rows:
            self.get_logger().warn("No rows to replay.")
            self._publish_initials_once()
            return

        # Publish initials BEFORE replay
        self._publish_initials_once()

        t0_data = self.rows[0].t_sec
        t0_wall = time.time()

        for k, row in enumerate(self.rows):
            target_wall = t0_wall + (row.t_sec - t0_data) / self.speed
            while True:
                now = time.time()
                dt = target_wall - now
                if dt <= 0.0:
                    break
                time.sleep(min(dt, 0.005))

            msg_type = self.msg_types[row.topic]
            b = (self.dataset_dir / row.relpath).read_bytes()
            msg = deserialize_message(b, msg_type)
            self.pubs[row.topic].publish(msg)

            if (k % 500) == 0 and k > 0:
                self.get_logger().info(f"Published {k}/{len(self.rows)}")

        self.get_logger().info("Replay complete.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Dataset directory (contains index.csv)")
    ap.add_argument("--speed", default="1.0", help="Replay speed (2.0 = 2x faster)")

    ap.add_argument("--initial_odom_topic", default="/initial_odom_correction", help="Initial odom correction topic (Pose)")
    ap.add_argument("--initial_gps_topic", default="/initial_gps_fix", help="Initial GPS fix topic (Point)")

    # NEW: final gps topic (Point lat/lon/alt), published latched for downstream dump node
    ap.add_argument("--final_gps_topic", default="/final_gps_fix", help="Final GPS fix topic (Point)")

    ap.add_argument("--no_initials", action="store_true", help="Do not publish initials on startup")
    ap.add_argument("--initials_delay", default="0.10", help="Delay after publishing initials (seconds)")
    args = ap.parse_args()

    rclpy.init()
    node = BagDataReplayer(
        dataset_dir=Path(args.dataset),
        speed=float(args.speed),
        initial_odom_topic=str(args.initial_odom_topic),
        initial_gps_topic=str(args.initial_gps_topic),
        final_gps_topic=str(args.final_gps_topic),
        publish_initials=(not bool(args.no_initials)),
        publish_initials_delay_s=float(args.initials_delay),
    )
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
