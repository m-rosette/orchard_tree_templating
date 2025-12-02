#!/usr/bin/env python3

from typing import Dict, Optional
import os

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from std_msgs.msg import Header
from geometry_msgs.msg import PointStamped, PoseArray, Pose, Point
from visualization_msgs.msg import Marker

from pf_orchard_interfaces.msg import TreeImageData
from tree_template_interfaces.srv import UpdateTrellisPosition
from tree_template_interfaces.msg import TrunkInfo, TrunkRegistry
from std_srvs.srv import Trigger

import tf2_ros
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_point

from scipy.spatial.transform import Rotation as R


class TrunkClusterToTemplateNode(Node):
    """
    Node that:
      - Subscribes to TreeImageData detections (camera frame)
      - Transforms each detection into a fixed TARGET FRAME (e.g. world/odom)
      - Tracks detections over time into persistent tracks (one per trunk)
      - For each mature track, decides if it is a new tree and, if so:
          - Classifies side (NEAR/FAR) based on which side of the fitted row datum
            line the tree lies on
          - Sends a single /update_trellis_position request with pose + side
      - Publishes the fitted row datum line as a green Marker on 'trunk_row_datum'
      - Publishes the current set of committed trunks on 'trunk_registry'.
    """

    def __init__(self):
        super().__init__("trunk_cluster_to_template_node")

        # -------- Parameters --------
        self.declare_parameter("input_topic", "tree_image_data")
        self.declare_parameter("min_samples", 10)            # min hits per track before committing
        self.declare_parameter("cluster_timer_period", 3.0)  # how often to evaluate tracks [s]

        # Track association gates (in TARGET FRAME)
        self.declare_parameter("track_position_gate", 0.4)   # [m] max distance to associate detection to track
        self.declare_parameter("track_width_gate", 0.02)     # width tol for associating to same track

        # "New tree" logic
        self.declare_parameter("uniqueness_radius", 1.0)     # if > 0, use simple Euclidean threshold [m]

        # Row axis (along row) and anisotropic gating
        self.declare_parameter("row_axis_x", 1.0)
        self.declare_parameter("row_axis_y", 0.0)
        self.declare_parameter("row_axis_z", 0.0)
        self.declare_parameter("along_row_tolerance", 0.4)   # [m] along-row tol for "same tree"
        self.declare_parameter("lateral_tolerance", 0.4)     # [m] lateral tol for "same tree"

        # Frames
        self.declare_parameter("camera_frame", "base_camera_color_optical_frame")
        self.declare_parameter("target_frame", "world")

        # Row datum visualization params
        self.declare_parameter("row_datum_update_rate_hz", 5.0)
        self.declare_parameter("row_datum_line_width", 0.1)

        self.input_topic = self.get_parameter("input_topic").value
        self.min_samples = int(self.get_parameter("min_samples").value)
        self.cluster_timer_period = float(self.get_parameter("cluster_timer_period").value)

        self.track_position_gate = float(self.get_parameter("track_position_gate").value)
        self.track_width_gate = float(self.get_parameter("track_width_gate").value)

        self.uniqueness_radius = float(self.get_parameter("uniqueness_radius").value)

        # Row axis (for along/lateral decomposition) used for uniqueness gating
        self.row_axis = np.array(
            [
                float(self.get_parameter("row_axis_x").value),
                float(self.get_parameter("row_axis_y").value),
                float(self.get_parameter("row_axis_z").value),
            ],
            dtype=float,
        )
        norm = np.linalg.norm(self.row_axis)
        if norm > 1e-6:
            self.row_axis /= norm
        else:
            self.row_axis[:] = np.array([1.0, 0.0, 0.0], dtype=float)

        self.along_row_tolerance = float(self.get_parameter("along_row_tolerance").value)
        self.lateral_tolerance = float(self.get_parameter("lateral_tolerance").value)

        self.camera_frame = self.get_parameter("camera_frame").value
        self.target_frame = self.get_parameter("target_frame").value

        # Row datum visualization params
        self.row_datum_update_rate_hz = float(
            self.get_parameter("row_datum_update_rate_hz").value
        )
        self.row_datum_line_width = float(
            self.get_parameter("row_datum_line_width").value
        )

        # -------- TF2 --------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # -------- Tracks --------
        # track_id -> {
        #   "id": int,
        #   "sum_pos": np.array(3),
        #   "num_pos": int,
        #   "last_pos": np.array(3),
        #   "width_sum": float,
        #   "width_count": int,
        #   "width_mean": float | None,
        #   "first_seen_time": float,
        #   "last_seen_time": float,
        #   "hit_count": int,
        #   "committed": bool,
        # }
        self.tracks: Dict[int, Dict] = {}
        self.next_track_id = 0

        # Positions for which we have sent a service request but have not yet
        # received a response. Used to avoid duplicates.
        self.pending_positions: list[np.ndarray] = []

        # Registry of committed trunks (in memory only, replaces YAML)
        self.committed_trunks: list[TrunkInfo] = []

        # -------- Service clients --------
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

        # Clear existing trees on startup (collision objects) via TreeSceneNode
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

        # -------- Publisher for row datum --------
        self.row_datum_pub = self.create_publisher(Marker, "trunk_row_datum", 10)

        # -------- Publisher for trunk registry (snapshot of committed trunks) --------
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.trunk_registry_pub = self.create_publisher(
            TrunkRegistry,
            "trunk_registry",
            qos,
        )

        # -------- Timer for track evaluation / committing / updating datum --------
        self.timer = self.create_timer(
            self.cluster_timer_period, self.track_evaluation_timer_callback
        )

        self.get_logger().info(
            "TrunkClusterToTemplateNode started.\n"
            f"  Subscribing to: {self.input_topic}\n"
            f"  min_samples={self.min_samples}\n"
            f"  Track gates: position={self.track_position_gate:.3f} m, "
            f"width={self.track_width_gate:.4f}\n"
            f"  Uniqueness radius: {self.uniqueness_radius:.3f} m\n"
            f"  Row axis: {self.row_axis}, along_tol={self.along_row_tolerance:.2f}, "
            f"lat_tol={self.lateral_tolerance:.2f}\n"
            f"  Side classification: based on near/far side of fitted row datum line.\n"
            f"  Cluster period: {self.cluster_timer_period:.2f} s\n"
            f"  Row datum: line_width={self.row_datum_line_width:.3f}\n"
            f"  Frames: camera_frame='{self.camera_frame}', target_frame='{self.target_frame}'"
        )

    # ---------------- TreeImageData callback ----------------

    def tree_image_callback(self, msg: TreeImageData):
        """
        For each detection in the frame:
          - Build a 3D point in CAMERA FRAME
          - Transform to TARGET FRAME
          - Assign to an existing track or create a new track
        """
        if not msg.object_seen:
            return

        t_msg = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        points_2d, widths = self.extract_points_and_widths_from_msg(msg)

        if points_2d.size == 0:
            return

        n = points_2d.shape[0]
        for i in range(n):
            xy = points_2d[i]
            width = float(widths[i]) if widths is not None else None

            # Map 2D (x,y) to a 3D point in camera frame (same as your original mapping)
            pos_cam = np.array([xy[0], 0.55, -xy[1]], dtype=float)

            pos_target = self.transform_to_target_frame(pos_cam)
            if pos_target is None:
                continue

            self._assign_detection_to_track(t_msg, pos_target, width)

    # ---------------- Track assignment ----------------

    def _assign_detection_to_track(
        self, t: float, pos: np.ndarray, width: Optional[float]
    ):
        """
        Assign detection to the best existing track if:
          - distance in TARGET FRAME < track_position_gate,
          - (if width info available) width difference < track_width_gate.

        Otherwise, create a new track.
        """
        best_track_id = None
        best_dist = None

        for track_id, track in self.tracks.items():
            if track["committed"]:
                continue

            dist = np.linalg.norm(pos - track["last_pos"])
            if dist > self.track_position_gate:
                continue

            if width is not None and track["width_mean"] is not None:
                if abs(width - track["width_mean"]) > self.track_width_gate:
                    continue

            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_track_id = track_id

        if best_track_id is None:
            # Start new track
            track_id = self.next_track_id
            self.next_track_id += 1

            width_sum = width if width is not None else 0.0
            width_count = 1 if width is not None else 0

            track = {
                "id": track_id,
                "sum_pos": pos.copy(),
                "num_pos": 1,
                "last_pos": pos.copy(),
                "width_sum": width_sum,
                "width_count": width_count,
                "width_mean": width if width is not None else None,
                "first_seen_time": t,
                "last_seen_time": t,
                "hit_count": 1,
                "committed": False,
            }
            self.tracks[track_id] = track

            self.get_logger().debug(
                f"Started new track {track_id}: pos={pos}, width={width}"
            )
        else:
            # Update existing track
            track = self.tracks[best_track_id]
            track["sum_pos"] += pos
            track["num_pos"] += 1
            track["last_pos"] = pos
            if width is not None:
                track["width_sum"] += width
                track["width_count"] += 1
                track["width_mean"] = track["width_sum"] / track["width_count"]
            track["last_seen_time"] = t
            track["hit_count"] += 1

            self.get_logger().debug(
                f"Updated track {best_track_id}: "
                f"last_pos={track['last_pos']}, width_mean={track['width_mean']}"
            )

    # ---------------- Track evaluation / committing timer ----------------

    def track_evaluation_timer_callback(self):
        """
        Periodically:
          - For each uncommitted track with enough hits, decide if it's a new tree
            and, if so, classify its side based on the row datum and send a trellis
            placement request (once).
          - Update the row datum marker.
        """
        if not self.tracks:
            self.get_logger().debug("No tracks to evaluate.")
            return

        new_requests = 0
        for track_id, track in self.tracks.items():
            if track["committed"]:
                continue

            if track["hit_count"] < self.min_samples:
                continue

            centroid = track["sum_pos"] / track["num_pos"]

            # Classify side first so we can use it in the uniqueness check
            side = self.classify_side_for_tree(centroid)

            if self.is_position_new(centroid, side):
                track["committed"] = True
                self.pending_positions.append(centroid)

                self.get_logger().info(
                    f"New trunk track {track_id} (side={side}) in {self.target_frame} "
                    f"at {centroid}, sending trellis placement request."
                )

                pose = Pose()
                pose.position.x = float(centroid[0])
                pose.position.y = float(centroid[1])
                pose.position.z = float(centroid[2])

                # Add orientation from datum
                quat = self.compute_datum_yaw_quaternion()
                if quat is not None:
                    pose.orientation.x = float(quat[0])
                    pose.orientation.y = float(quat[1])
                    pose.orientation.z = float(quat[2])
                    pose.orientation.w = float(quat[3])
                else:
                    pose.orientation.w = 1.0  # fallback: no rotation

                width_mean = track["width_mean"]
                self.send_trellis_request(pose, side, width_mean)
                new_requests += 1
            else:
                self.get_logger().debug(
                    f"Track {track_id} centroid {centroid} is not new "
                    f"(within uniqueness criteria); skipping."
                )

        if new_requests > 0:
            self.get_logger().info(f"Sent {new_requests} trellis placement request(s).")

    # ---------------- Side classification using row datum ----------------

    def classify_side_for_tree(self, pos_target: np.ndarray) -> str:
        """
        Classify a committed tree as 'near' or 'far' based on the sign of the
        perpendicular distance from the point to the fitted row datum line
        y = m x + b in the XY plane.

        Conventions:
          - If we cannot fit a line (fewer than 2 committed trunks): 'unknown'
          - signed_dist = (y - (m x + b)) / sqrt(m^2 + 1)
              signed_dist < 0  -> 'near'
              signed_dist >= 0 -> 'far'
        """
        fit = self._fit_row_datum_line()
        if fit is None:
            self.get_logger().warn(
                "Cannot classify side: not enough committed trunks to fit datum. "
                "Defaulting to 'unknown'."
            )
            return "unknown"

        m, b = fit
        x = float(pos_target[0])
        y = float(pos_target[1])
        y_line = m * x + b

        denom = np.sqrt(m * m + 1.0)
        if denom < 1e-6:
            self.get_logger().warn(
                "Row datum line is nearly degenerate; cannot compute perpendicular distance. "
                "Defaulting to 'unknown'."
            )
            return "unknown"

        signed_dist = (y - y_line) / denom
        side = "near" if signed_dist < 0.0 else "far"

        self.get_logger().debug(
            f"Classified tree at (x={x:.2f}, y={y:.2f}) relative to datum "
            f"y={m:.3f}x+{b:.3f}: signed_dist={signed_dist:.3f} m -> side={side}"
        )

        return side

    def _get_all_positions_on_side(self, side: str) -> list[np.ndarray]:
        """
        Return all existing trunk positions (committed + pending) that lie on the
        specified side ("near" or "far") of the fitted row datum line.
        """
        if side not in ("near", "far"):
            return []

        fit = self._fit_row_datum_line()
        if fit is None:
            return []

        m, b = fit
        denom = np.sqrt(m * m + 1.0)
        if denom < 1e-6:
            return []

        positions = self.get_existing_positions() + self.pending_positions

        same_side_positions: list[np.ndarray] = []
        for pos in positions:
            if pos.shape[0] < 2:
                continue

            x = float(pos[0])
            y = float(pos[1])
            y_line = m * x + b
            signed_dist = (y - y_line) / denom
            current_side = "near" if signed_dist < 0.0 else "far"

            if current_side == side:
                same_side_positions.append(pos)

        return same_side_positions

    # ---------------- Row datum (line fit) helpers ----------------

    def _fit_row_datum_line(self) -> Optional[tuple[float, float]]:
        """
        Fit a single line y = m x + b to committed trunk positions.
        Returns (m, b) or None if not enough points.
        """
        trunks = self._trunks_for_row_datum()
        if len(trunks) < 2:
            return None

        pts = np.array(trunks, dtype=float)  # shape (N, 2)
        x = pts[:, 0]
        y = pts[:, 1]

        A = np.vstack([x, np.ones_like(x)]).T
        m, b = np.linalg.lstsq(A, y, rcond=None)[0]
        return float(m), float(b)

    def compute_datum_yaw_quaternion(self) -> Optional[np.ndarray]:
        """
        Computes the yaw quaternion for the fitted row datum line.
        Returns np.array([x, y, z, w]) or None if the line cannot be fit.
        """
        fit = self._fit_row_datum_line()
        if fit is None:
            return None

        m, _ = fit

        # Direction vector of the line in XY plane = (1, m)
        theta = np.arctan2(m, 1.0)  # yaw angle

        # Convert to quaternion
        quat = R.from_euler("z", theta).as_quat()  # [x, y, z, w]

        return quat

    def row_datum_timer_callback(self):
        """
        Update the row datum marker based on current committed trunk positions.
        """
        fit = self._fit_row_datum_line()
        if fit is None:
            # self._publish_row_datum_delete()
            return

        m, b = fit
        trunks = self._trunks_for_row_datum()
        pts = np.array(trunks, dtype=float)
        x = pts[:, 0]
        self._publish_row_datum_line(m, b, x)

    def _trunks_for_row_datum(self) -> list[tuple[float, float]]:
        """
        Extract (x, y) pairs for line fitting from committed trunk poses.
        """
        positions = self.get_existing_positions()
        trunks: list[tuple[float, float]] = []
        for p in positions:
            if p.shape[0] >= 2:
                trunks.append((float(p[0]), float(p[1])))
        return trunks

    def _publish_row_datum_delete(self):
        marker = Marker()
        marker.header = Header()
        marker.header.frame_id = self.target_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "trunk_row_datum"
        marker.id = 0
        marker.action = Marker.DELETE
        self.row_datum_pub.publish(marker)

    def _publish_row_datum_line(self, m: float, b: float, xs: np.ndarray):
        if xs.size == 0:
            self._publish_row_datum_delete()
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
        marker.header.frame_id = self.target_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "trunk_row_datum"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        marker.scale.x = self.row_datum_line_width  # line thickness

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

        self.row_datum_pub.publish(marker)

    # ---------------- TF helper ----------------

    def transform_to_target_frame(self, pos_cam: np.ndarray) -> Optional[np.ndarray]:
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
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            self.get_logger().warn(
                f"Failed to lookup transform {self.camera_frame} -> {self.target_frame}: {e}"
            )
            return None

        ps_out = do_transform_point(ps, transform)
        return np.array(
            [ps_out.point.x, ps_out.point.y, ps_out.point.z],
            dtype=float,
        )

    # ---------------- Service call helpers ----------------

    def send_trellis_request(self, pose: Pose, side: str, width: Optional[float]):
        """
        Asynchronously call /update_trellis_position for a given position
        in TARGET FRAME coordinates, including side id ("near" / "far" / "unknown").
        """
        request = UpdateTrellisPosition.Request()
        request.pose = pose
        request.side = side or ""

        self.get_logger().info(
            f"Sending trellis update: side={side}, "
            f"pose=({pose.position.x:.2f}, {pose.position.y:.2f}, {pose.position.z:.2f})"
        )

        future = self.trellis_template_client.call_async(request)

        # Attach data to the future so we know which one it corresponds to
        position_array = np.array(
            [pose.position.x, pose.position.y, pose.position.z],
            dtype=float,
        )
        future._trellis_pos = position_array
        future._trellis_pose = pose
        future._trellis_side = side
        future._trellis_width = width
        future.add_done_callback(self.trellis_response_callback)

    def trellis_response_callback(self, future):
        """
        Handle the result of the trellis placement service.
        """
        pos = getattr(future, "_trellis_pos", None)
        pose = getattr(future, "_trellis_pose", None)
        side = getattr(future, "_trellis_side", "")
        width = getattr(future, "_trellis_width", None)

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

            # Add to in-memory registry and publish updated registry
            if pose is not None:
                trunk = TrunkInfo()
                trunk.pose = pose
                trunk.side = side or ""
                trunk.width = float(width) if width is not None else float("nan")
                self.committed_trunks.append(trunk)
                self.publish_trunk_registry()

            # Recompute and republish datum based on committed trunks
            self.row_datum_timer_callback()
        else:
            self.get_logger().error(f"Trellis placement FAILED for position {pos}.")
            self._remove_from_pending(pos)

    def publish_trunk_registry(self):
        """
        Publish a snapshot of all committed trunks on trunk_registry.
        """
        msg = TrunkRegistry()
        # Directly reuse the stored TrunkInfo objects
        msg.trunks = list(self.committed_trunks)
        self.trunk_registry_pub.publish(msg)
        self.get_logger().debug(f"Published trunk registry with {len(msg.trunks)} trunks")

    def _remove_from_pending(self, pos: Optional[np.ndarray]):
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

    def extract_points_and_widths_from_msg(
        self, msg: TreeImageData
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Convert TreeImageData into:
          - pts: (N, 2) array of [x, y] points in CAMERA FRAME
          - widths: (N,) array if available, else None
        """
        xs = np.array(msg.xs, dtype=np.float32)
        ys = np.array(msg.ys, dtype=np.float32)
        widths = np.array(msg.widths, dtype=np.float32) if msg.widths else None

        if xs.size == 0 or ys.size == 0:
            return np.empty((0, 2), dtype=np.float32), None

        if widths is not None and widths.size == 0:
            widths = None

        if widths is not None:
            n = min(xs.size, ys.size, widths.size)
            if xs.size != n or ys.size != n or widths.size != n:
                self.get_logger().warn(
                    f"Length mismatch in TreeImageData: "
                    f"len(xs)={xs.size}, len(ys)={ys.size}, len(widths)={widths.size}; "
                    f"truncating to {n}"
                )
            xs = xs[:n]
            ys = ys[:n]
            widths = widths[:n]
        else:
            if xs.size != ys.size:
                self.get_logger().warn(
                    f"xs and ys length mismatch: len(xs)={xs.size}, len(ys)={ys.size}"
                )
                n = min(xs.size, ys.size)
                xs = xs[:n]
                ys = ys[:n]

        if xs.size == 0:
            return np.empty((0, 2), dtype=np.float32), None

        pts = np.stack([xs, ys], axis=1).astype(np.float32)
        return pts, widths

    def get_existing_positions(self) -> list[np.ndarray]:
        """
        Return list of positions for all committed trunks as np.array([x, y, z]).
        """
        positions: list[np.ndarray] = []
        for ti in self.committed_trunks:
            p = ti.pose.position
            positions.append(np.array([p.x, p.y, p.z], dtype=float))
        return positions

    def is_position_new(self, pos: np.ndarray, side: Optional[str] = None) -> bool:
        """
        Decide if 'pos' (in TARGET FRAME) corresponds to a new tree.

        Uses committed trunk positions (in memory) plus pending positions.
        """
        existing_positions = self.get_existing_positions()
        all_existing = existing_positions + self.pending_positions

        if not all_existing:
            return True

        # Side-aware uniqueness check
        if self.uniqueness_radius > 0.0:
            if side in ("near", "far"):
                same_side_positions = self._get_all_positions_on_side(side)

                # If nothing on this side yet, treat as new
                if not same_side_positions:
                    return True

                for existing in same_side_positions:
                    if np.linalg.norm(pos - existing) < self.uniqueness_radius:
                        return False
                return True

            # Fallback: original behavior (no valid side info)
            for existing in all_existing:
                if np.linalg.norm(pos - existing) < self.uniqueness_radius:
                    return False
            return True

        # Row-based anisotropic check (unchanged)
        for existing in all_existing:
            delta = pos - existing
            s = np.dot(delta, self.row_axis)  # along-row difference
            lateral_vec = delta - s * self.row_axis
            d_lat = np.linalg.norm(lateral_vec)
            if abs(s) < self.along_row_tolerance and d_lat < self.lateral_tolerance:
                return False

        return True


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
