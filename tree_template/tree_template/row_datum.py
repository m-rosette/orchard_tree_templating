#!/usr/bin/env python3

import os
from typing import List, Tuple, Optional, Dict

import yaml
import numpy as np
from sklearn.cluster import KMeans  # pip install scikit-learn

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from ament_index_python.packages import get_package_share_directory

from std_msgs.msg import Header
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker

from tf2_ros import Buffer, TransformListener, TransformException


class RowDatumFitter:
    """
    Given sets of near-row and far-row trunk points, fit separate lines
    y = m x + b for each row.
    """

    def __init__(self):
        self.near_points: Optional[np.ndarray] = None
        self.far_points: Optional[np.ndarray] = None
        self._near_line: Optional[Tuple[float, float]] = None
        self._far_line: Optional[Tuple[float, float]] = None

    @staticmethod
    def _fit_line(pts: Optional[np.ndarray]) -> Optional[Tuple[float, float]]:
        if pts is None or pts.shape[0] < 2:
            return None
        x = pts[:, 0]
        y = pts[:, 1]
        A = np.vstack([x, np.ones_like(x)]).T
        m, b = np.linalg.lstsq(A, y, rcond=None)[0]
        return float(m), float(b)

    def set_points(
        self,
        near_pts: Optional[np.ndarray],
        far_pts: Optional[np.ndarray],
    ):
        self.near_points = near_pts
        self.far_points = far_pts
        self._near_line = self._fit_line(near_pts)
        self._far_line = self._fit_line(far_pts)

    # Near row
    def has_near_line(self) -> bool:
        return self._near_line is not None

    def get_near_line(self) -> Tuple[float, float]:
        if self._near_line is None:
            raise RuntimeError("No valid near-row line.")
        return self._near_line

    # Far row
    def has_far_line(self) -> bool:
        return self._far_line is not None

    def get_far_line(self) -> Tuple[float, float]:
        if self._far_line is None:
            raise RuntimeError("No valid far-row line.")
        return self._far_line


class TrunkRowDatumNode(Node):
    """
    Reads trunk positions from a YAML file, classifies trunks into near/far rows
    with a cached label per ID, and fits separate lines for each row.

    - Near row → green line on /trunk_row_near
    - Far row  → red   line on /trunk_row_far

    Expects YAML like:
    - id: v_trellis_tree_0
      x: ...
      y: ...
      z: ...
    """

    def __init__(self):
        super().__init__("trunk_row_datum_node")

        # Parameters
        self.declare_parameter("trellis_frame", "odom")
        self.declare_parameter("robot_frame", "amiga__base")
        self.declare_parameter("min_points_for_split", 6)
        self.declare_parameter("update_rate_hz", 20.0)
        # Name of the YAML file in tree_template/resource
        self.declare_parameter("trunk_positions_yaml", "trellis_ids.yaml")
        self.declare_parameter("line_width", 0.1)

        self.frame_id = self.get_parameter("trellis_frame").get_parameter_value().string_value
        self.robot_frame = self.get_parameter("robot_frame").get_parameter_value().string_value
        self.min_points_for_split = self.get_parameter("min_points_for_split").get_parameter_value().integer_value
        self.update_rate_hz = self.get_parameter("update_rate_hz").get_parameter_value().double_value
        self.line_width = self.get_parameter("line_width").get_parameter_value().double_value

        yaml_name = self.get_parameter("trunk_positions_yaml").get_parameter_value().string_value

        # Default YAML path in tree_template/resource
        pkg_share = get_package_share_directory("tree_template")
        resource_dir = os.path.join(pkg_share, "resource")
        os.makedirs(resource_dir, exist_ok=True)
        self.yaml_path = os.path.join(resource_dir, yaml_name)

        # TF buffer + listener to get robot_y from TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Fitter: just fits lines given near/far points
        self.fitter = RowDatumFitter()

        # Cached labels: id -> "near" | "far"
        self.trunk_labels: Dict[str, str] = {}
        self.cluster_initialized: bool = False

        # Publishers: near (green) and far (red)
        self.near_marker_pub = self.create_publisher(Marker, "trunk_row_near", 10)
        self.far_marker_pub = self.create_publisher(Marker, "trunk_row_far", 10)

        # Timer to periodically read YAML and update the lines
        timer_period = 1.0 / max(self.update_rate_hz, 1e-3)
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.get_logger().info(
            f"TrunkRowDatumNode running.\n"
            f"  YAML path: {self.yaml_path}\n"
            f"  Frame: {self.frame_id}\n"
            f"  Robot frame: {self.robot_frame}\n"
            f"  Min points for split: {self.min_points_for_split}\n"
            f"  Update rate: {self.update_rate_hz} Hz"
        )

    # ---------- TF helper ----------

    def _get_robot_y(self) -> Optional[float]:
        """
        Return the robot's current y position in trellis_frame using TF.
        """
        try:
            tf = self.tf_buffer.lookup_transform(
                self.frame_id,      # target frame (e.g. "odom")
                self.robot_frame,   # source frame (e.g. "amiga__base")
                Time()              # latest available
            )
            return tf.transform.translation.y
        except TransformException as ex:
            self.get_logger().warn_throttle(
                2000,
                f"TF lookup failed for {self.frame_id} <- {self.robot_frame}: {ex}"
            )
            return None

    # ---------- YAML handling ----------

    def _load_trunks_from_yaml(self) -> List[Tuple[str, float, float]]:
        """
        Load (id, x, y) trunk positions from a YAML of the form:

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

        trunks: List[Tuple[str, float, float]] = []

        if isinstance(data, list):
            for entry in data:
                if not isinstance(entry, dict):
                    continue
                if "id" in entry and "x" in entry and "y" in entry:
                    trunks.append((str(entry["id"]), float(entry["x"]), float(entry["y"])))
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

    def _publish_delete_marker(self, pub, ns: str, marker_id: int):
        marker = Marker()
        marker.header = Header()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = ns
        marker.id = marker_id
        marker.action = Marker.DELETE
        pub.publish(marker)

    def _publish_line_marker(
        self,
        pub,
        ns: str,
        marker_id: int,
        m: float,
        b: float,
        xs: np.ndarray,
        color: Tuple[float, float, float, float],
    ):
        if xs.size == 0:
            self._publish_delete_marker(pub, ns, marker_id)
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
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        marker.scale.x = self.line_width  # line thickness

        r, g, b_, a = color
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b_
        marker.color.a = a

        p0 = Point()
        p0.x = x0
        p0.y = y0
        p0.z = 0.0

        p1 = Point()
        p1.x = x1
        p1.y = y1
        p1.z = 0.0

        marker.points = [p0, p1]

        pub.publish(marker)

    # ---------- Classification helpers ----------

    def _initialize_clusters_with_kmeans(
        self,
        trunks: List[Tuple[str, float, float]],
        robot_y: float,
    ):
        """
        Use KMeans on all current trunks to initialize near/far labels.
        """
        pts = np.array([[x, y] for (_id, x, y) in trunks], dtype=float)
        y_vals = pts[:, 1].reshape(-1, 1)

        kmeans = KMeans(
            n_clusters=2,
            n_init="auto",
            random_state=0,
        )
        labels = kmeans.fit_predict(y_vals)
        centers = kmeans.cluster_centers_.flatten()

        # Near cluster is the one whose center y is closest to robot_y
        dists_to_robot_y = np.abs(centers - robot_y)
        near_cluster_idx = int(np.argmin(dists_to_robot_y))
        far_cluster_idx = 1 - near_cluster_idx

        self.trunk_labels.clear()

        for i, (tid, _x, _y) in enumerate(trunks):
            cluster = labels[i]
            if cluster == near_cluster_idx:
                self.trunk_labels[tid] = "near"
            elif cluster == far_cluster_idx:
                self.trunk_labels[tid] = "far"

        self.cluster_initialized = True
        self.get_logger().info(
            f"Initialized near/far clusters with KMeans. "
            f"Near center y={centers[near_cluster_idx]:.3f}, "
            f"Far center y={centers[far_cluster_idx]:.3f}"
        )

    def _classify_new_trunks_incremental(
        self,
        trunks: List[Tuple[str, float, float]],
    ):
        """
        For newly seen trunk IDs, classify them as near/far using existing labeled trunks.
        Uses mean y of current near/far sets as reference.
        """
        # Build map id -> (x, y) for trunks present in YAML
        id_to_xy = {tid: (x, y) for tid, x, y in trunks}

        # Compute current near/far mean y using only IDs in current YAML
        near_ys = [
            id_to_xy[tid][1]
            for tid, label in self.trunk_labels.items()
            if label == "near" and tid in id_to_xy
        ]
        far_ys = [
            id_to_xy[tid][1]
            for tid, label in self.trunk_labels.items()
            if label == "far" and tid in id_to_xy
        ]

        near_mean = np.mean(near_ys) if near_ys else None
        far_mean = np.mean(far_ys) if far_ys else None

        new_ids = [tid for tid, _x, _y in trunks if tid not in self.trunk_labels]

        for tid in new_ids:
            x, y = id_to_xy[tid]

            if near_mean is not None and far_mean is not None:
                dist_near = abs(y - near_mean)
                dist_far = abs(y - far_mean)
                label = "near" if dist_near <= dist_far else "far"
            elif near_mean is not None:
                label = "near"
            elif far_mean is not None:
                label = "far"
            else:
                # No prior info at all (shouldn't really happen if cluster_initialized)
                label = "near"

            self.trunk_labels[tid] = label
            self.get_logger().debug(
                f"Classified new trunk {tid} at y={y:.3f} as {label} "
                f"(near_mean={near_mean}, far_mean={far_mean})"
            )

    # ---------- Main timer ----------

    def timer_callback(self):
        trunks = self._load_trunks_from_yaml()
        if not trunks:
            self._publish_delete_marker(self.near_marker_pub, "trunk_row_near", 0)
            self._publish_delete_marker(self.far_marker_pub, "trunk_row_far", 0)
            return

        # Live robot lateral position
        robot_y = self._get_robot_y()
        if robot_y is None:
            # Can't classify properly without robot reference, so just bail this cycle
            self._publish_delete_marker(self.near_marker_pub, "trunk_row_near", 0)
            self._publish_delete_marker(self.far_marker_pub, "trunk_row_far", 0)
            return

        # INITIALIZATION: use KMeans once we have enough points
        if not self.cluster_initialized:
            if len(trunks) >= self.min_points_for_split:
                self._initialize_clusters_with_kmeans(trunks, robot_y)
            else:
                # Not enough points yet to split near/far
                self._publish_delete_marker(self.near_marker_pub, "trunk_row_near", 0)
                self._publish_delete_marker(self.far_marker_pub, "trunk_row_far", 0)
                return
        else:
            # INCREMENTAL: classify only the new IDs
            self._classify_new_trunks_incremental(trunks)

        # Build near/far point sets from current YAML + cached labels
        id_to_xy = {tid: (x, y) for tid, x, y in trunks}

        near_pts_list = [
            id_to_xy[tid]
            for tid, label in self.trunk_labels.items()
            if label == "near" and tid in id_to_xy
        ]
        far_pts_list = [
            id_to_xy[tid]
            for tid, label in self.trunk_labels.items()
            if label == "far" and tid in id_to_xy
        ]

        near_pts = np.array(near_pts_list, dtype=float) if near_pts_list else None
        far_pts = np.array(far_pts_list, dtype=float) if far_pts_list else None

        # Update fitter with current classified points
        self.fitter.set_points(near_pts, far_pts)

        # Near row (green)
        if self.fitter.has_near_line() and near_pts is not None:
            m_near, b_near = self.fitter.get_near_line()
            xs_near = near_pts[:, 0]
            self._publish_line_marker(
                self.near_marker_pub,
                ns="trunk_row_near",
                marker_id=0,
                m=m_near,
                b=b_near,
                xs=xs_near,
                color=(0.0, 1.0, 0.0, 1.0),  # green
            )
        else:
            self._publish_delete_marker(self.near_marker_pub, "trunk_row_near", 0)

        # Far row (red)
        if self.fitter.has_far_line() and far_pts is not None:
            m_far, b_far = self.fitter.get_far_line()
            xs_far = far_pts[:, 0]
            self._publish_line_marker(
                self.far_marker_pub,
                ns="trunk_row_far",
                marker_id=0,
                m=m_far,
                b=b_far,
                xs=xs_far,
                color=(1.0, 0.0, 0.0, 1.0),  # red
            )
        else:
            self._publish_delete_marker(self.far_marker_pub, "trunk_row_far", 0)


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
