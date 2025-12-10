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
from tree_template_interfaces.msg import TrunkInfo, TrunkRegistry

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
          - Classifies side (NEAR/FAR) based on the fitted row datum line
          - Publishes a TrunkInfo observation on 'trunk_observations'
      - Publishes the fitted row datum line as a green Marker on 'trunk_row_datum'
      - Optionally publishes a snapshot of committed trunks on 'trunk_registry'
        (for debugging/visualization only – the authoritative map is RowPriorMapper).
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

        # Registry of committed trunks (in memory only, for datum & debugging)
        self.committed_trunks: list[TrunkInfo] = []

        # -------- NEW: Publisher for trunk observations --------
        self.trunk_obs_pub = self.create_publisher(
            TrunkInfo,
            "trunk_observations",  # RowPriorMapper subscribes to this
            10,
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

        # -------- Optional publisher for trunk registry (debug) --------
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

        # -------- Publisher for committed trunk markers --------
        self.trunk_marker_pub = self.create_publisher(
            Marker,
            "committed_trunk_markers",
            10,
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

            # Map 2D (x,y) to a 3D point in camera frame
            pos_cam = np.array([-xy[0], 0.0, -xy[1]], dtype=float)

            pos_target = self.transform_to_target_frame(pos_cam, msg.header.stamp)
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
            and, if so, classify its side and publish a TrunkInfo observation.
          - Update the row datum marker.
        """
        if not self.tracks:
            self.get_logger().debug("No tracks to evaluate.")
            return

        new_obs = 0
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

                self.get_logger().info(
                    f"New trunk track {track_id} (side={side}) in {self.target_frame} "
                    f"at {centroid}, publishing TrunkInfo observation."
                )

                pose = Pose()
                pose.position.x = float(centroid[0])
                pose.position.y = float(centroid[1])
                pose.position.z = float(centroid[2])

                # Orientation here is optional; RowPriorMapper will overwrite yaw from its datum.
                # You can leave it as identity or use your fitted datum yaw as a better initial guess.
                quat = self.compute_datum_yaw_quaternion()
                if quat is not None:
                    pose.orientation.x = float(quat[0])
                    pose.orientation.y = float(quat[1])
                    pose.orientation.z = float(quat[2])
                    pose.orientation.w = float(quat[3])
                else:
                    pose.orientation.w = 1.0

                width_mean = track["width_mean"]

                # Build and publish TrunkInfo observation
                tinfo = TrunkInfo()
                tinfo.pose = pose
                tinfo.side = side or ""
                tinfo.width = float(width_mean) if width_mean is not None else float("nan")

                self.trunk_obs_pub.publish(tinfo)

                # Also publish a committed trunk marker
                self.publish_committed_trunk_marker(pose, track_id)

                # Also store for datum fitting & optional debug registry
                self.committed_trunks.append(tinfo)
                self.publish_trunk_registry()
                new_obs += 1
            else:
                self.get_logger().debug(
                    f"Track {track_id} centroid {centroid} is not new "
                    f"(within uniqueness criteria); skipping."
                )

        # Update / republish datum marker based on committed trunks
        self.row_datum_timer_callback()

        if new_obs > 0:
            self.get_logger().info(f"Published {new_obs} TrunkInfo observation(s).")

    # ---------------- Side classification using row datum ----------------

    def classify_side_for_tree(self, pos_target: np.ndarray) -> str:
        """
        Classify a committed tree as 'near' or 'far' based on the sign of the
        perpendicular distance from the point to the fitted row datum line
        y = m x + b in the XY plane.
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
        Return all existing trunk positions that lie on the specified side
        ("near" or "far") of the fitted row datum line.
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

        positions = self.get_existing_positions()

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
        theta = np.arctan2(m, 1.0)  # yaw angle
        quat = R.from_euler("z", theta).as_quat()  # [x, y, z, w]
        return quat

    def row_datum_timer_callback(self):
        """
        Update the row datum marker based on current committed trunk positions.
        """
        fit = self._fit_row_datum_line()
        if fit is None:
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

    def _publish_row_datum_line(self, m: float, b: float, xs: np.ndarray):
        if xs.size == 0:
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

    def transform_to_target_frame(self, pos_cam: np.ndarray, stamp) -> Optional[np.ndarray]:
        """
        Transform a 3D point from camera_frame to target_frame using TF2.
        """
        ps = PointStamped()
        ps.header.stamp = stamp
        ps.header.frame_id = self.camera_frame
        ps.point.x = float(pos_cam[0])
        ps.point.y = float(pos_cam[1])
        ps.point.z = float(pos_cam[2])

        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.camera_frame,
                Time.from_msg(stamp),
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

        Uses committed trunk positions (in memory) only.
        """
        all_existing = self.get_existing_positions()
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

    def publish_trunk_registry(self):
        """
        Publish a snapshot of all committed trunks on trunk_registry (debug).
        """
        msg = TrunkRegistry()
        msg.trunks = list(self.committed_trunks)
        self.trunk_registry_pub.publish(msg)
        self.get_logger().debug(f"Published trunk registry with {len(msg.trunks)} trunks")

    def publish_committed_trunk_marker(self, pose: Pose, marker_id: int):
        """
        Publish a red cube marker at the committed trunk pose.
        Each trunk gets a unique marker id so previous cubes stay in RViz.
        """
        marker = Marker()
        marker.header = Header()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self.target_frame

        marker.ns = "committed_trunks"
        marker.id = marker_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        # Use the trunk pose directly
        marker.pose = pose

        # Cube size in meters - adjust if you want bigger/smaller
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2

        # Red, slightly transparent
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.9

        self.trunk_marker_pub.publish(marker)


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
        # Stop the executor
        try:
            executor.shutdown()
        except Exception:
            pass

        # Remove and destroy the node
        try:
            executor.remove_node(node)
        except Exception:
            pass
        node.destroy_node()

        # Only try to shutdown the context if it is still active
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except rclpy.exceptions.RCLError:
                # Context already shut down somewhere else
                pass


if __name__ == "__main__":
    main()
