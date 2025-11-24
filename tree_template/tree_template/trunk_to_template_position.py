#!/usr/bin/env python3

from collections import deque
import os  # NEW

import numpy as np
from sklearn.cluster import DBSCAN 

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration

from geometry_msgs.msg import PointStamped, PoseArray, Pose
from pf_orchard_interfaces.msg import TreeImageData
from tree_template_interfaces.srv import UpdateTrellisPosition
from std_srvs.srv import Trigger

import tf2_ros
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_point

from ament_index_python.packages import get_package_share_directory
import yaml


class TrunkClusterToTemplateNode(Node):
    """
    Combined node that:
      - Subscribes to TreeImageData detections (in camera frame)
      - Clusters trunk detections with DBSCAN over a sliding window
      - Transforms cluster centroids from camera_frame -> target_frame (e.g. odom)
      - For each unique cluster centroid in target_frame, calls /update_trellis_position
      - Publishes centroids as a PoseArray in target_frame for visualization
    """

    def __init__(self):
        super().__init__("trunk_cluster_to_template_node")

        # -------- Parameters --------
        self.declare_parameter("input_topic", "tree_image_data")
        self.declare_parameter("eps", 0.6)            # DBSCAN neighborhood radius [m]
        self.declare_parameter("min_samples", 10)       # DBSCAN min_samples
        self.declare_parameter("window_seconds", 3.0)  # detection time window [s]
        self.declare_parameter("uniqueness_radius", 0.0)  # distance to consider "new" [m]
        self.declare_parameter("cluster_timer_period", 3.0)  # how often to run clustering [s]

        # New: frame parameters
        self.declare_parameter("camera_frame", "base_camera_color_optical_frame")
        self.declare_parameter("target_frame", "world")

        self.input_topic = self.get_parameter("input_topic").value
        self.eps = float(self.get_parameter("eps").value)
        self.min_samples = int(self.get_parameter("min_samples").value)
        self.window_seconds = float(self.get_parameter("window_seconds").value)
        self.uniqueness_radius = float(self.get_parameter("uniqueness_radius").value)
        self.cluster_timer_period = float(self.get_parameter("cluster_timer_period").value)

        self.camera_frame = self.get_parameter("camera_frame").value
        self.target_frame = self.get_parameter("target_frame").value

        # -------- TF2 --------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # -------- Detection buffer --------
        # Each entry: {"t": float, "point": np.array([x, y])} in CAMERA FRAME
        self.detection_buffer = deque()

        # Pending positions to avoid duplicates
        self.pending_positions = [] 

        # -------- Service client --------
        self.trellis_template_client = self.create_client(
            UpdateTrellisPosition,
            "/update_trellis_position",
        )
        while not self.trellis_template_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Trellis template service not available, waiting...")

        self.clear_trees_client = self.create_client(
            Trigger,
            "clear_trellis_trees",
        )
        while not self.clear_trees_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for clear_trellis_trees service...")

        # Clear existing trees on startup
        clear_request = Trigger.Request()
        clear_future = self.clear_trees_client.call_async(clear_request)
        clear_future.add_done_callback(
            lambda future: self.get_logger().info(
                "Cleared existing trellis trees on startup."
                if future.result() and future.result().success
                else "Failed to clear existing trellis trees on startup."
            )
        )

        # -------- Subscriber --------
        self.sub = self.create_subscription(
            TreeImageData,
            self.input_topic,
            self.tree_image_callback,
            10,
        )

        # -------- Publisher for visualization in target frame --------
        self.centroid_pub = self.create_publisher(PoseArray, "trunk_centroids_odom", 10)

        # -------- Timer for clustering --------
        self.timer = self.create_timer(self.cluster_timer_period, self.cluster_timer_callback)

        self.get_logger().info(
            "TrunkClusterToTemplateNode started.\n"
            f"  Subscribing to: {self.input_topic}\n"
            f"  DBSCAN: eps={self.eps:.3f} m, min_samples={self.min_samples}, "
            f"window={self.window_seconds:.1f} s\n"
            f"  Uniqueness radius: {self.uniqueness_radius:.3f} m\n"
            f"  Cluster period: {self.cluster_timer_period:.2f} s\n"
            f"  Frames: camera_frame='{self.camera_frame}', target_frame='{self.target_frame}'"
        )

    # ---------------- TreeImageData callback ----------------

    def tree_image_callback(self, msg: TreeImageData):
        """
        Collect detections from TreeImageData into the sliding buffer.

        xs, ys are assumed to be coordinates in the CAMERA FRAME (meters).
        """
        if not msg.object_seen:
            return

        t_msg = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        points = self.extract_points_from_msg(msg)

        if points.size == 0:
            return

        for p in points:
            self.detection_buffer.append({"t": t_msg, "point": p})

        # Trim old detections
        self.trim_buffer(t_msg)

    # ---------------- Clustering timer callback ----------------

    def cluster_timer_callback(self):
        """
        Periodically run DBSCAN on the buffered detections and, for each
        new cluster centroid (in TARGET FRAME), send a trellis placement request.
        Also publish all centroids as a PoseArray in TARGET FRAME.
        """
        if len(self.detection_buffer) == 0:
            self.get_logger().debug("No detections in buffer to cluster.")
            return

        points = np.array([entry["point"] for entry in self.detection_buffer])  # (N, 2) in camera frame
        if points.shape[0] < self.min_samples:
            self.get_logger().debug(
                f"Not enough points for clustering: {points.shape[0]} < min_samples {self.min_samples}"
            )
            return

        db = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = db.fit_predict(points)

        unique_labels = [lab for lab in set(labels) if lab != -1]  # skip noise
        if not unique_labels:
            self.get_logger().debug("No clusters found in current window.")
            return

        centroids_target = []  # in TARGET FRAME

        for lab in unique_labels:
            cluster_points = points[labels == lab]
            mean_xy = cluster_points.mean(axis=0)

            # Build a 3D point in camera frame
            pos_cam = np.array([mean_xy[0], 0.55, -mean_xy[1]], dtype=float)

            pos_target = self.transform_to_target_frame(pos_cam)
            if pos_target is None:
                self.get_logger().warn(
                    f"Skipping cluster at camera-frame {pos_cam} due to TF failure."
                )
                continue

            centroids_target.append(pos_target)

        if not centroids_target:
            self.get_logger().debug("No centroids transformed to target frame; skipping.")
            return

        # Publish all current centroids in target frame as a PoseArray
        self.publish_centroids_pose_array(centroids_target)

        new_requests = 0
        for pos in centroids_target:
            if self.is_position_new(pos):
                self.pending_positions.append(pos)

                self.get_logger().info(
                    f"New trunk cluster in {self.target_frame} at {pos}, "
                    f"sending trellis placement request."
                )

                pose = Pose()
                pose.position.x = float(pos[0])
                pose.position.y = float(pos[1])
                pose.position.z = float(pos[2])
                self.send_trellis_request(pose)
                new_requests += 1
            else:
                self.get_logger().debug(
                    f"Cluster at {pos} is within {self.uniqueness_radius:.2f} m of an "
                    f"existing or pending template; skipping."
                )

        if new_requests > 0:
            self.get_logger().info(f"Sent {new_requests} trellis placement request(s).")

    # ---------------- TF helper ----------------

    def transform_to_target_frame(self, pos_cam: np.ndarray) -> np.ndarray | None:
        """
        Transform a 3D point from camera_frame to target_frame using TF2.
        """
        ps = PointStamped()
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.header.frame_id = self.camera_frame
        ps.point.x = float(pos_cam[0])
        ps.point.y = float(pos_cam[1])
        ps.point.z = float(pos_cam[2])

        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.camera_frame,
                Time.from_msg(ps.header.stamp),
                timeout=Duration(seconds=0.1),
            )
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(
                f"Failed to lookup transform {self.camera_frame} -> {self.target_frame}: {e}"
            )
            return None

        ps_out = do_transform_point(ps, transform)
        return np.array(
            [ps_out.point.x, ps_out.point.y, ps_out.point.z],
            dtype=float
        )

    # ---------------- Service call helpers ----------------

    def send_trellis_request(self, pose: Pose):
        """
        Asynchronously call /update_trellis_position for a given position
        in TARGET FRAME coordinates.
        """
        request = UpdateTrellisPosition.Request()
        request.pose = pose

        future = self.trellis_template_client.call_async(request)

        # Attach the position to the future so we know which one it corresponds to
        position_array = np.array(
            [pose.position.x, pose.position.y, pose.position.z],
            dtype=float
        )
        future._trellis_pose = position_array
        future.add_done_callback(self.trellis_response_callback)

    def trellis_response_callback(self, future):
        """
        Handle the result of the trellis placement service.
        """
        pos = getattr(future, "_trellis_pose", None)

        if future.result() is None:
            self.get_logger().error(
                f"Trellis placement service call failed for position {pos} (no result)."
            )
            self._remove_from_pending(pos)
            return

        result = future.result()
        if result.success:
            self.get_logger().info(f"Trellis placement succeeded at {pos}.")
            self._remove_from_pending(pos)
        else:
            self.get_logger().error(f"Trellis placement FAILED for position {pos}.")
            self._remove_from_pending(pos)

    def _remove_from_pending(self, pos: np.ndarray):
        """
        Remove a position from pending_positions (using allclose for float safety).
        """
        if pos is None:
            return

        new_pending = []
        for existing in self.pending_positions:
            if not np.allclose(existing, pos):
                new_pending.append(existing)
        self.pending_positions = new_pending

    # ---------------- Helper methods ----------------

    def trim_buffer(self, latest_time: float):
        """
        Keep only detections in the last window_seconds.
        """
        cutoff = latest_time - self.window_seconds
        while self.detection_buffer and self.detection_buffer[0]["t"] < cutoff:
            self.detection_buffer.popleft()

    def extract_points_from_msg(self, msg: TreeImageData) -> np.ndarray:
        """
        Convert TreeImageData into an (N, 2) array of [x, y] points in CAMERA FRAME.
        """
        xs = np.array(msg.xs, dtype=np.float32)
        ys = np.array(msg.ys, dtype=np.float32)

        if xs.size == 0 or ys.size == 0:
            return np.empty((0, 2), dtype=np.float32)

        if xs.size != ys.size:
            self.get_logger().warn(
                f"xs and ys length mismatch: len(xs)={xs.size}, len(ys)={ys.size}"
            )
            n = min(xs.size, ys.size)
            xs = xs[:n]
            ys = ys[:n]

        if xs.size == 0:
            return np.empty((0, 2), dtype=np.float32)

        pts = np.stack([xs, ys], axis=1)  # (N, 2)
        return pts.astype(np.float32)

    def is_position_new(self, pos: np.ndarray) -> bool:
        """
        A position is "new" if it is farther than uniqueness_radius from all
        positions in the YAML (trellis_ids.yaml) and from all pending_positions.

        YAML is reloaded on every call so we always see the latest tree positions.
        """
        existing_positions = self.load_existing_positions()

        # Check against positions from YAML + pending positions
        for existing in (existing_positions + self.pending_positions):
            if np.linalg.norm(pos - existing) < self.uniqueness_radius:
                return False

        return True
    
    def load_existing_positions(self) -> list[np.ndarray]:
        """
        Load existing trellis tree positions from the YAML file.
        Supports:
          - New format (preferred):
              - id: v_trellis_tree_0
                x: ...
                y: ...
                z: ...
          - Legacy format:
              { "v_trellis_tree_0": {"x": ..., "y": ..., "z": ...}, ... }

        Returns a list of np.ndarray positions.
        """
        positions: list[np.ndarray] = []

        try:
            pkg_share = get_package_share_directory("tree_template")
            resource_dir = os.path.join(pkg_share, "resource")
            yaml_path = os.path.join(resource_dir, "trellis_ids.yaml")
        except Exception as e:
            self.get_logger().warn(f"Could not resolve trellis_ids.yaml path: {e}")
            return positions

        if not os.path.exists(yaml_path):
            # Only log at debug/info level so we don't spam when starting clean
            self.get_logger().debug("trellis_ids.yaml does not exist; no existing positions.")
            return positions

        try:
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f)
        except Exception as e:
            self.get_logger().warn(f"Failed to read trellis_ids.yaml: {e}")
            return positions

        if data is None:
            self.get_logger().debug("trellis_ids.yaml is empty.")
            return positions

        # New format: list of dicts
        if isinstance(data, list):
            for entry in data:
                if not isinstance(entry, dict):
                    continue
                try:
                    x = float(entry.get("x", 0.0))
                    y = float(entry.get("y", 0.0))
                    z = float(entry.get("z", 0.0))
                    positions.append(np.array([x, y, z], dtype=float))
                except Exception as e:
                    self.get_logger().warn(
                        f"Failed to parse list entry in trellis_ids.yaml: {e}"
                    )
            return positions

        # Legacy format: dict mapping id -> {x, y, z}
        if isinstance(data, dict):
            for obj_id, pos_dict in data.items():
                if not isinstance(pos_dict, dict):
                    continue
                try:
                    x = float(pos_dict.get("x", 0.0))
                    y = float(pos_dict.get("y", 0.0))
                    z = float(pos_dict.get("z", 0.0))
                    positions.append(np.array([x, y, z], dtype=float))
                except Exception as e:
                    self.get_logger().warn(
                        f"Failed to parse position for {obj_id} in trellis_ids.yaml: {e}"
                    )
            return positions

        self.get_logger().warn(
            f"trellis_ids.yaml has unexpected top-level type: {type(data)}"
        )
        return positions

    def publish_centroids_pose_array(self, centroids_target: list[np.ndarray]):
        """
        Publish centroids in TARGET FRAME as a PoseArray for visualization.
        """
        msg = PoseArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.target_frame

        for pos in centroids_target:
            pose = Pose()
            pose.position.x = float(pos[0])
            pose.position.y = float(pos[1])
            pose.position.z = float(pos[2])
            pose.orientation.w = 1.0  # identity orientation
            msg.poses.append(pose)

        self.centroid_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TrunkClusterToTemplateNode()
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
            node.get_logger().info("Shutting down TrunkClusterToTemplateNode...")
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
