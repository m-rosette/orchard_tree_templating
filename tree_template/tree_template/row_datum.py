#!/usr/bin/env python3

import os
from typing import List, Tuple, Optional

import yaml
import numpy as np

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

from std_msgs.msg import Header
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker


class TrunkRowDatumNode(Node):
    """
    Reads trunk positions from a YAML file and fits a single line
    y = m x + b through ALL trunk locations.

    Publishes the line as a green Marker on the 'trunk_row_datum' topic.

    Expects YAML like:
    - id: v_trellis_tree_0
      x: ...
      y: ...
      z: ...
    - id: v_trellis_tree_1
      x: ...
      y: ...
      z: ...
    """

    def __init__(self):
        super().__init__("trunk_row_datum_node")

        # Parameters
        self.declare_parameter("trellis_frame", "odom")
        self.declare_parameter("update_rate_hz", 20.0)
        # Name of the YAML file in tree_template/resource
        self.declare_parameter("trunk_positions_yaml", "trellis_ids.yaml")
        self.declare_parameter("line_width", 0.1)

        self.frame_id = self.get_parameter("trellis_frame").get_parameter_value().string_value
        self.update_rate_hz = self.get_parameter("update_rate_hz").get_parameter_value().double_value
        self.line_width = self.get_parameter("line_width").get_parameter_value().double_value

        yaml_name = self.get_parameter("trunk_positions_yaml").get_parameter_value().string_value

        # Default YAML path in tree_template/resource
        pkg_share = get_package_share_directory("tree_template")
        resource_dir = os.path.join(pkg_share, "resource")
        os.makedirs(resource_dir, exist_ok=True)
        self.yaml_path = os.path.join(resource_dir, yaml_name)

        # Publisher for the RViz line marker
        self.marker_pub = self.create_publisher(Marker, "trunk_row_datum", 10)

        # Timer to periodically read YAML and update the line
        timer_period = 1.0 / max(self.update_rate_hz, 1e-3)
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.get_logger().info(
            f"TrunkRowDatumNode (single-line fit) running.\n"
            f"  YAML path: {self.yaml_path}\n"
            f"  Frame: {self.frame_id}\n"
            f"  Update rate: {self.update_rate_hz} Hz"
        )

    # ---------- YAML handling ----------

    def _load_trunks_from_yaml(self) -> List[Tuple[float, float]]:
        """
        Load (x, y) trunk positions from a YAML of the form:

        - id: v_trellis_tree_0
          x: ...
          y: ...
          z: ...
        """
        if not os.path.exists(self.yaml_path):
            self.get_logger().debug(f"Trunk YAML not found at {self.yaml_path}")
            return []

        try:
            with open(self.yaml_path, "r") as f:
                data = yaml.safe_load(f)
        except Exception as e:
            self.get_logger().warn(f"Failed to read trunk YAML {self.yaml_path}: {e}")
            return []

        if data is None:
            self.get_logger().info("Trunk YAML is empty.")
            return []

        trunks: List[Tuple[float, float]] = []

        if isinstance(data, list):
            for entry in data:
                if not isinstance(entry, dict):
                    continue
                if "x" in entry and "y" in entry:
                    trunks.append((float(entry["x"]), float(entry["y"])))
                else:
                    self.get_logger().debug(f"Skipping malformed entry in YAML: {entry}")
        else:
            self.get_logger().warn(
                f"Unexpected YAML structure in {self.yaml_path}, expected a list of dicts."
            )

        if not trunks:
            self.get_logger().info("No valid trunk positions found in YAML.")

        return trunks

    # ---------- Marker publishing helpers ----------

    def _publish_delete_marker(self):
        marker = Marker()
        marker.header = Header()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "trunk_row_datum"
        marker.id = 0
        marker.action = Marker.DELETE
        self.marker_pub.publish(marker)

    def _publish_line_marker(self, m: float, b: float, xs: np.ndarray):
        if xs.size == 0:
            self._publish_delete_marker()
            return

        x_min = float(xs.min())
        x_max = float(xs.max())
        margin = 1.0
        x0 = x_min - margin
        x1 = x_max + margin
        y0 = m * x0 + b
        y1 = m * x1 + b

        marker = Marker()
        marker.header = Header()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "trunk_row_datum"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        marker.scale.x = self.line_width  # line thickness

        # Green line
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        p0 = Point()
        p0.x = x0
        p0.y = y0
        p0.z = 0.0

        p1 = Point()
        p1.x = x1
        p1.y = y1
        p1.z = 0.0

        marker.points = [p0, p1]

        self.marker_pub.publish(marker)

    # ---------- Main timer ----------

    def timer_callback(self):
        trunks = self._load_trunks_from_yaml()
        if len(trunks) < 2:
            # Not enough points to define a line
            self._publish_delete_marker()
            return

        pts = np.array(trunks, dtype=float)  # shape (N, 2)
        x = pts[:, 0]
        y = pts[:, 1]

        # Least-squares line fit: y = m x + b
        A = np.vstack([x, np.ones_like(x)]).T
        m, b = np.linalg.lstsq(A, y, rcond=None)[0]
        m = float(m)
        b = float(b)

        self._publish_line_marker(m, b, x)


def main(args=None):
    rclpy.init(args=args)
    node = TrunkRowDatumNode()
    try:
            rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down TrunkRowDatumNode...")
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
