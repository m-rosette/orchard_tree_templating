#!/usr/bin/env python3

from typing import Dict, Optional

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from std_msgs.msg import Header
from geometry_msgs.msg import Pose, Point, PoseStamped
from visualization_msgs.msg import Marker

from pf_orchard_interfaces.msg import TreeImageData
from tree_template_interfaces.msg import TrunkInfo, TrunkRegistry

import tf2_ros

from scipy.spatial.transform import Rotation as R


class TrunkClusterToTemplateNode(Node):
    """
    Node that:
      - Subscribes to TreeImageData detections (camera frame)
      - Transforms each detection into a fixed TARGET FRAME (e.g. world/odom)
      - Tracks detections over time into persistent tracks (one per trunk)
      - For each mature track, decides if it is a new tree and, if so:
          - Classifies side (NEAR/FAR) based on the fitted row datum line
          - Publishes a TrunkInfo observation on 'trunk_observations' (TARGET FRAME)
          - Publishes a TrunkInfo measurement on 'trunk_measurements' (CAMERA FRAME) for FastSLAM
      - Publishes the fitted row datum line as a green Marker on 'trunk_row_datum'
      - Optionally publishes a snapshot of committed trunks on 'trunk_registry'
        (for debugging/visualization only – the authoritative map is RowPriorMapper).
    """

    def __init__(self):
        super().__init__("trunk_cluster_to_template_node")

        # -------- Parameters (declare) --------
        self.declare_parameter("input_topic", "tree_image_data")
        self.declare_parameter("min_samples", 10)
        self.declare_parameter("cluster_timer_period", 1.0)

        self.declare_parameter("track_position_gate", 0.5)
        self.declare_parameter("track_width_gate", 0.03)
        self.declare_parameter("uniqueness_radius", 0.4)
        self.declare_parameter("use_trunk_width_addition", True)

        self.declare_parameter("row_axis_x", 1.0)
        self.declare_parameter("row_axis_y", 0.0)
        self.declare_parameter("row_axis_z", 0.0)

        self.declare_parameter("side_mode", "trunk_depth")  # "datum" | "slot_alternating" | "trunk_depth"
        self.declare_parameter("slot_spacing", 0.75)
        self.declare_parameter("row_origin_s", 0.0)
        self.declare_parameter("start_side", "far")
        self.declare_parameter("along_row_tolerance", 0.25)
        self.declare_parameter("lateral_tolerance", 0.25)

        self.declare_parameter("camera_frame", "base_camera_color_optical_frame")
        self.declare_parameter("target_frame", "odom_slam")

        self.declare_parameter("row_datum_line_width", 0.1)

        # -------- Parameters (read + validate) --------
        self.input_topic = str(self.get_parameter("input_topic").value)
        self.min_samples = int(self.get_parameter("min_samples").value)
        self.cluster_timer_period = float(self.get_parameter("cluster_timer_period").value)

        self.track_position_gate = float(self.get_parameter("track_position_gate").value)
        self.track_width_gate = float(self.get_parameter("track_width_gate").value)
        self.uniqueness_radius = float(self.get_parameter("uniqueness_radius").value)
        self.use_trunk_width_addition = bool(self.get_parameter("use_trunk_width_addition").value)

        row_axis_x = float(self.get_parameter("row_axis_x").value)
        row_axis_y = float(self.get_parameter("row_axis_y").value)
        row_axis_z = float(self.get_parameter("row_axis_z").value)
        self.row_axis = np.array([row_axis_x, row_axis_y, row_axis_z], dtype=float)
        axis_norm = float(np.linalg.norm(self.row_axis))
        if axis_norm > 1e-6:
            self.row_axis = self.row_axis / axis_norm
        else:
            self.row_axis = np.array([1.0, 0.0, 0.0], dtype=float)

        self.side_mode = str(self.get_parameter("side_mode").value).strip().lower()
        if self.side_mode not in ("datum", "slot_alternating", "trunk_depth"):
            self.get_logger().warn(
                f"side_mode='{self.side_mode}' not in {{datum, slot_alternating, trunk_depth}}; defaulting to 'datum'."
            )
            self.side_mode = "datum"

        self.slot_spacing = float(self.get_parameter("slot_spacing").value)
        self.row_origin_s = float(self.get_parameter("row_origin_s").value)

        self.start_side = str(self.get_parameter("start_side").value).strip().lower()
        if self.start_side not in ("near", "far"):
            self.get_logger().warn(
                f"start_side='{self.start_side}' not in {{near, far}}; defaulting to 'near'."
            )
            self.start_side = "near"

        self.along_row_tolerance = float(self.get_parameter("along_row_tolerance").value)
        self.lateral_tolerance = float(self.get_parameter("lateral_tolerance").value)

        self.camera_frame = str(self.get_parameter("camera_frame").value)
        self.target_frame = str(self.get_parameter("target_frame").value)

        self.row_datum_line_width = float(self.get_parameter("row_datum_line_width").value)

        # -------- State --------
        self.tree_image_data_timestamp: Optional[Time] = None

        self.tracks: Dict[int, Dict] = {}
        self.next_track_id = 0
        self.committed_trunks: list[TrunkInfo] = []

        # -------- Row datum cache --------
        self._datum_fit: Optional[tuple[float, float]] = None
        self._datum_fit_cached_for_version: int = -1
        self._datum_fit_version: int = 0

        # --- Registry publish throttling ---
        self._last_registry_pub_time = 0.0
        self._registry_pub_min_period = 0.25  # seconds (2 Hz)

        # --- Datum marker publish control ---
        self._last_datum_published_version = -1

        # cache committed positions as numpy for fast uniqueness checks ---
        self._committed_pos_np = np.empty((0, 3), dtype=np.float64)  # grows on commit

        # -------- TF2 --------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # -------- Publishers --------
        self.trunk_obs_pub = self.create_publisher(TrunkInfo, "trunk_observations", 10)
        self.trunk_meas_pub = self.create_publisher(TrunkInfo, "trunk_measurements", 10)
        self.row_datum_pub = self.create_publisher(Marker, "trunk_row_datum", 10)
        self.trunk_marker_pub = self.create_publisher(Marker, "committed_trunk_markers", 10)
        self.row_datum_pose_pub = self.create_publisher(PoseStamped, "row_datum_pose", 10)

        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.trunk_registry_pub = self.create_publisher(TrunkRegistry, "trunk_registry", qos)

        # -------- Subscriber --------
        self.sub = self.create_subscription(
            TreeImageData,
            self.input_topic,
            self.tree_image_callback,
            10,
        )

        # -------- Timer --------
        self.timer = self.create_timer(
            self.cluster_timer_period,
            self.track_evaluation_timer_callback,
        )

        # -------- Startup log --------
        self.get_logger().info(
            "TrunkClusterToTemplateNode started.\n"
            f"  Subscribing to: {self.input_topic}\n"
            f"  min_samples={self.min_samples}\n"
            f"  Track gates: position={self.track_position_gate:.3f} m, width={self.track_width_gate:.4f}\n"
            f"  Uniqueness radius: {self.uniqueness_radius:.3f} m\n"
            f"  Row axis: {self.row_axis}, along_tol={self.along_row_tolerance:.2f}, lat_tol={self.lateral_tolerance:.2f}\n"
            f"  Side mode: {self.side_mode}\n"
            f"  Cluster period: {self.cluster_timer_period:.2f} s\n"
            f"  Row datum: line_width={self.row_datum_line_width:.3f}\n"
            f"  Frames: camera_frame='{self.camera_frame}', target_frame='{self.target_frame}'"
        )

    # ---------------- TreeImageData callback ----------------

    def tree_image_callback(self, msg: TreeImageData):
        """Transform detections and assign them to tracks."""
        self.tree_image_data_timestamp = msg.header.stamp
        if not msg.object_seen:
            return

        points_2d, widths, top_2d, bottom_2d = self.extract_points_and_widths_from_msg(msg)
        if points_2d.size == 0 or top_2d.size == 0 or bottom_2d.size == 0:
            return

        t_msg = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # TF once per frame
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.camera_frame,
                Time(), # latest available
                timeout=Duration(seconds=0.05),
            )
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return
        
        T = self._transformstamped_to_T(tf_msg)
        n = points_2d.shape[0]

        pos_cam = np.empty((n, 3), dtype=np.float64)
        top_cam = np.empty((n, 3), dtype=np.float64)
        bot_cam = np.empty((n, 3), dtype=np.float64)

        if self.use_trunk_width_addition and widths is not None:
            # Adjust 2D points to account for trunk radius
            radii = widths.astype(np.float64) * 0.5
            points_2d[:, 1] += radii
            top_2d[:, 1] += radii
            bottom_2d[:, 1] += radii

        pos_cam[:, 1] = 0.0
        top_cam[:, 1] = 0.0
        bot_cam[:, 1] = 0.0

        pos_cam[:, 0] = points_2d[:, 0]
        # pos_cam[:, 2] = points_2d[:, 1]
        pos_cam[:, 2] = bottom_2d[:, 1] # use bottom z for trunk base

        top_cam[:, 0] = top_2d[:, 0]
        top_cam[:, 2] = top_2d[:, 1]

        bot_cam[:, 0] = bottom_2d[:, 0]
        bot_cam[:, 2] = bottom_2d[:, 1]

        # Transform (batch)
        pos_t = self._apply_T_batch(T, pos_cam)
        top_t = self._apply_T_batch(T, top_cam)
        bot_t = self._apply_T_batch(T, bot_cam)

        if widths is None:
            for p_t, p_c, t_t, b_t in zip(pos_t, pos_cam, top_t, bot_t):
                self._assign_detection_to_track(t_msg, p_t, p_c, None, t_t, b_t)
        else:
            for p_t, p_c, w, t_t, b_t in zip(pos_t, pos_cam, widths, top_t, bot_t):
                self._assign_detection_to_track(t_msg, p_t, p_c, float(w), t_t, b_t)

    # ---------------- Track assignment ----------------

    def _assign_detection_to_track(
        self,
        t: float,
        pos: np.ndarray,
        pos_cam: np.ndarray,
        width: Optional[float],
        top_target: np.ndarray,
        bottom_target: np.ndarray,
    ):
        """
        Assign detection to the best existing (uncommitted) track if:
        - distance in TARGET FRAME < track_position_gate
        - (if width available) width difference < track_width_gate

        Otherwise create a new track.
        """
        best_track_id = None
        best_dist = float("inf")

        for track_id, track in self.tracks.items():
            if track["committed"]:
                continue

            dist = float(np.linalg.norm(pos - track["last_pos"]))
            if dist > self.track_position_gate:
                continue

            if width is not None and track["width_mean"] is not None:
                if abs(width - track["width_mean"]) > self.track_width_gate:
                    continue

            if dist < best_dist:
                best_dist = dist
                best_track_id = track_id

        # ---------------- Create ----------------
        if best_track_id is None:
            track_id = self.next_track_id
            self.next_track_id += 1

            width_sum = float(width) if width is not None else 0.0
            width_count = 1 if width is not None else 0
            width_mean = float(width) if width is not None else None

            self.tracks[track_id] = {
                "id": track_id,
                "sum_pos": pos.copy(),
                "num_pos": 1,
                "last_pos": pos.copy(),
                "sum_pos_cam": pos_cam.copy(),
                "last_pos_cam": pos_cam.copy(),
                "sum_top_target": top_target.copy(),
                "last_top_target": top_target.copy(),
                "sum_bottom_target": bottom_target.copy(),
                "last_bottom_target": bottom_target.copy(),
                "width_sum": width_sum,
                "width_count": width_count,
                "width_mean": width_mean,
                "first_seen_time": t,
                "last_seen_time": t,
                "hit_count": 1,
                "committed": False,
            }

            self.get_logger().debug(f"Started new track {track_id}: pos={pos}, width={width}")
            return

        # ---------------- Update ----------------
        track = self.tracks[best_track_id]

        track["sum_pos"] += pos
        track["num_pos"] += 1
        track["last_pos"] = pos
        track["sum_pos_cam"] += pos_cam
        track["last_pos_cam"] = pos_cam
        track["sum_top_target"] += top_target
        track["last_top_target"] = top_target
        track["sum_bottom_target"] += bottom_target
        track["last_bottom_target"] = bottom_target
        if width is not None:
            track["width_sum"] += float(width)
            track["width_count"] += 1
            track["width_mean"] = track["width_sum"] / track["width_count"]
        track["last_seen_time"] = t
        track["hit_count"] += 1

        self.get_logger().debug(
            f"Updated track {best_track_id}: last_pos={track['last_pos']}, width_mean={track['width_mean']}"
        )

    # ---------------- Track evaluation / committing timer ----------------

    def track_evaluation_timer_callback(self):
        """
        Periodically evaluate tracks and commit mature ones.
        """
        if not self.tracks:
            return

        now = self.get_clock().now().nanoseconds * 1e-9

        # Cache these once for this tick
        committed_changed = False
        new_obs = 0

        for track_id, track in self.tracks.items():
            if track["committed"] or track["hit_count"] < self.min_samples:
                continue

            inv_n = 1.0 / track["num_pos"]
            centroid = track["sum_pos"] * inv_n
            centroid_cam = track["sum_pos_cam"] * inv_n
            top_coord = track["sum_top_target"] * inv_n
            bottom_coord = track["sum_bottom_target"] * inv_n

            side = self.classify_side_for_tree(centroid, top_coord, bottom_coord)

            # Uniqueness check
            if not self.is_position_new(centroid, side):
                continue

            # Commit track
            track["committed"] = True
            committed_changed = True
            self._last_datum_published_version = self._datum_fit_version
            self._datum_fit_version += 1

            width_mean = track["width_mean"]
            width_out = float(width_mean) if width_mean is not None else float("nan")
            side_out = side or ""

            # --- TARGET FRAME observation ---
            pose = Pose()
            pose.position.x = float(centroid[0])
            pose.position.y = float(centroid[1])
            pose.position.z = float(centroid[2])
            pose.orientation.w = 1.0

            tinfo = TrunkInfo()
            tinfo.pose = pose
            tinfo.side = side_out
            tinfo.width = width_out
            self.trunk_obs_pub.publish(tinfo)

            # --- CAMERA FRAME measurement ---
            meas_pose = Pose()
            meas_pose.position.x = float(centroid_cam[0])
            meas_pose.position.y = float(centroid_cam[1])
            meas_pose.position.z = float(centroid_cam[2])
            meas_pose.orientation.w = 1.0

            tinfo_meas = TrunkInfo()
            tinfo_meas.pose = meas_pose
            tinfo_meas.side = side_out
            tinfo_meas.width = width_out
            self.trunk_meas_pub.publish(tinfo_meas)

            # --- Viz / bookkeeping ---
            self.publish_committed_trunk_marker(pose, track_id)
            self.committed_trunks.append(tinfo)

            self._committed_pos_np = np.vstack(
                (self._committed_pos_np, np.array([[centroid[0], centroid[1], centroid[2]]], dtype=np.float64))
            )

            new_obs += 1

        # Throttle registry publish (only if there was any change AND enough time elapsed)
        if committed_changed and (now - self._last_registry_pub_time) >= self._registry_pub_min_period:
            self.publish_trunk_registry()
            self._last_registry_pub_time = now

        # Publish datum marker only when committed set changed since last publish
        if self._last_datum_published_version != self._datum_fit_version:
            self.row_datum_timer_callback()
            self._last_datum_published_version = self._datum_fit_version

        if new_obs > 0:
            self.get_logger().debug(f"Published {new_obs} TrunkInfo observation(s).")

    # ---------------- Side classification using row datum ----------------
    
    def _side_for_slot(self, j: int) -> str:
        """Deterministic alternating side assignment based on slot index."""
        if self.start_side == "near":
            return "near" if (j % 2 == 0) else "far"
        else:
            return "far" if (j % 2 == 0) else "near"

    def _slot_index_for_position(self, pos_target: np.ndarray) -> int:
        """Compute the nearest slot index from a world-frame position."""
        # Use s = row_axis^T * p (absolute projection), then offset by row_origin_s
        p = np.array([float(pos_target[0]), float(pos_target[1]), 0.0], dtype=float)
        s_abs = float(np.dot(p, self.row_axis))
        j = int(np.round((s_abs - self.row_origin_s) / self.slot_spacing))
        return j

    def _get_trunk_top_and_bottom_depths(self, top_coord: np.ndarray, bottom_coord: np.ndarray) -> str:
        """Classify side based on trunk top and bottom depths in TARGET FRAME."""
        if top_coord[1] > bottom_coord[1]:
            return "far"
        elif top_coord[1] < bottom_coord[1]:
            return "near"
        else:
            return "unknown"
        
    def _get_datum_side_classification(self, pos_target: np.ndarray) -> str:
        """Classify side based on perpendicular distance to fitted row datum line."""
        fit = self._fit_row_datum_line()
        if fit is None:
            self.get_logger().warn(
                "Cannot classify side: not enough committed trunks to fit datum. Defaulting to 'unknown'."
            )
            return "unknown"

        m, b = fit
        x = float(pos_target[0])
        y = float(pos_target[1])

        denom = float(np.sqrt(m * m + 1.0))
        if denom < 1e-9:
            return "unknown"

        y_line = m * x + b
        signed_dist = (y - y_line) / denom
        side = "near" if signed_dist < 0.0 else "far"

        self.get_logger().debug(
            f"Classified tree at (x={x:.2f}, y={y:.2f}) relative to datum "
            f"y={m:.3f}x+{b:.3f}: signed_dist={signed_dist:.3f} m -> side={side}"
        )
        return side

    def classify_side_for_tree(
        self,
        pos_target: np.ndarray,
        top_coord: np.ndarray,
        bottom_coord: np.ndarray,
    ) -> str:
        """
        Classify a committed tree as 'near' or 'far'.

        Modes:
        - side_mode == "slot_alternating": side = alternating parity of the nearest slot index
        - side_mode == "trunk_depth": side = based on trunk top and bottom depths
        - side_mode == "datum": sign of perpendicular distance to fitted row datum line (legacy)
        """
        if self.side_mode == "slot_alternating":
            slot_idx = self._slot_index_for_position(pos_target)
            return self._side_for_slot(slot_idx)

        if self.side_mode == "trunk_depth":
            return self._get_trunk_top_and_bottom_depths(top_coord, bottom_coord)
        
        if self.side_mode == "datum":
            return self._get_datum_side_classification(pos_target)

    # ---------------- Row datum (line fit) helpers ----------------

    def _fit_row_datum_line(self) -> Optional[tuple[float, float]]:
        """
        Fit a single line y = m x + b to committed trunk positions.
        Returns (m, b) or None if not enough points.

        Cached: recompute only when committed trunks change.
        """
        # Use cached result if nothing changed
        if self._datum_fit_cached_for_version == self._datum_fit_version:
            return self._datum_fit

        trunks = self._trunks_for_row_datum()
        if len(trunks) < 2:
            self._datum_fit = None
            self._datum_fit_cached_for_version = self._datum_fit_version
            return None

        pts = np.array(trunks, dtype=float)  # (N, 2)
        x = pts[:, 0]
        y = pts[:, 1]

        A = np.vstack([x, np.ones_like(x)]).T
        m, b = np.linalg.lstsq(A, y, rcond=None)[0]

        self._datum_fit = (float(m), float(b))
        self._datum_fit_cached_for_version = self._datum_fit_version
        return self._datum_fit

    def row_datum_timer_callback(self):
        """
        Publish the fitted row datum line marker using cached committed positions.

        Requires:
        - self._committed_pos_np: np.ndarray of shape (M,3)
            (or it will fall back to your existing committed_trunks list)
        """
        fit = self._fit_row_datum_line()
        if fit is None:
            return

        m, b = fit

        # Prefer cached numpy positions if available
        pos_np = getattr(self, "_committed_pos_np", None)
        if isinstance(pos_np, np.ndarray) and pos_np.shape[0] > 0:
            xs = pos_np[:, 0]  # instant slice, no list comprehension
            self._publish_row_datum_line(m, b, xs)

            # Also publish yaw pose
            yaw = np.arctan2(m, 1.0)  # yaw angle
            self._publish_row_datum_pose(yaw)
            return

    def _trunks_for_row_datum(self):
        pos_np = getattr(self, "_committed_pos_np", None)
        if isinstance(pos_np, np.ndarray) and pos_np.shape[0] >= 2:
            # return list of (x,y) tuples for lstsq
            return [tuple(xy) for xy in pos_np[:, :2]]

        # fallback
        positions = self.get_existing_positions()
        return [(float(p[0]), float(p[1])) for p in positions if p.shape[0] >= 2]

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
        # marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.stamp = self.tree_image_data_timestamp if self.tree_image_data_timestamp is not None else self.get_clock().now().to_msg()
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

    def _publish_row_datum_pose(self, yaw: float):
        ps = PoseStamped()
        ps.header.frame_id = self.target_frame
        ps.header.stamp = (
            self.tree_image_data_timestamp
            if self.tree_image_data_timestamp is not None
            else self.get_clock().now().to_msg()
        )

        ps.pose.position.x = 0.0
        ps.pose.position.y = 0.0
        ps.pose.position.z = 0.0

        q = R.from_euler("z", float(yaw)).as_quat()  # [x,y,z,w]
        ps.pose.orientation.x = float(q[0])
        ps.pose.orientation.y = float(q[1])
        ps.pose.orientation.z = float(q[2])
        ps.pose.orientation.w = float(q[3])

        self.row_datum_pose_pub.publish(ps)

    # ---------------- TF helper ----------------
    
    def _transformstamped_to_T(self, tf_msg) -> np.ndarray:
        """
        Convert geometry_msgs/TransformStamped into a 4x4 homogeneous transform matrix.
        Maps points from source frame -> target frame as: p_t = T @ [p_s; 1]
        """
        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation

        # scipy Rotation expects [x, y, z, w]
        Rm = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()  # (3,3)

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = Rm
        T[:3, 3] = np.array([t.x, t.y, t.z], dtype=np.float64)
        return T

    def _apply_T_batch(self, T: np.ndarray, P: np.ndarray) -> np.ndarray:
        """
        Apply 4x4 homogeneous transform to many points.
        P: (N,3)
        returns: (N,3)
        """
        # (N,3) @ (3,3).T + (3,) -> (N,3)
        return (P @ T[:3, :3].T) + T[:3, 3]

    # ---------------- Helper methods ----------------

    def extract_points_and_widths_from_msg(
        self, msg: TreeImageData
    ) -> tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]:
        """
        Convert TreeImageData into:
        - pts: (N, 2) array of [x, y] points in CAMERA IMAGE COORDS
        - widths: (N,) array if available, else None
        - top_pts: (N, 2) array
        - bottom_pts: (N, 2) array
        """
        xs = np.array(msg.xs, dtype=np.float32)
        ys = np.array(msg.ys, dtype=np.float32)
        top_xs = np.array(msg.top_xs, dtype=np.float32)
        top_ys = np.array(msg.top_ys, dtype=np.float32)
        bottom_xs = np.array(msg.bottom_xs, dtype=np.float32)
        bottom_ys = np.array(msg.bottom_ys, dtype=np.float32)

        widths = np.array(msg.widths, dtype=np.float32) if msg.widths else None

        empty = np.empty((0, 2), dtype=np.float32)

        # ---- basic presence checks ----
        if xs.size == 0 or ys.size == 0:
            return empty, None, empty, empty

        if top_xs.size == 0 or top_ys.size == 0:
            return empty, None, empty, empty

        if bottom_xs.size == 0 or bottom_ys.size == 0:
            return empty, None, empty, empty

        if widths is not None and widths.size == 0:
            widths = None

        # ---- enforce consistent lengths ----
        if widths is not None:
            n = min(xs.size, ys.size, top_xs.size, top_ys.size, bottom_xs.size, bottom_ys.size, widths.size)
            if (
                xs.size != n or ys.size != n or widths.size != n
                or top_xs.size != n or top_ys.size != n
                or bottom_xs.size != n or bottom_ys.size != n
            ):
                self.get_logger().warn(
                    f"Length mismatch in TreeImageData; truncating to {n}"
                )

            xs = xs[:n]
            ys = ys[:n]
            top_xs = top_xs[:n]
            top_ys = top_ys[:n]
            bottom_xs = bottom_xs[:n]
            bottom_ys = bottom_ys[:n]
            widths = widths[:n]

        else:
            n = min(xs.size, ys.size, top_xs.size, top_ys.size, bottom_xs.size, bottom_ys.size)
            if (
                xs.size != n or ys.size != n
                or top_xs.size != n or top_ys.size != n
                or bottom_xs.size != n or bottom_ys.size != n
            ):
                self.get_logger().warn(
                    f"Coordinate length mismatch in TreeImageData; truncating to {n}"
                )

            xs = xs[:n]
            ys = ys[:n]
            top_xs = top_xs[:n]
            top_ys = top_ys[:n]
            bottom_xs = bottom_xs[:n]
            bottom_ys = bottom_ys[:n]

        # ---- stack ----
        pts = np.stack([xs, ys], axis=1)
        top_pts = np.stack([top_xs, top_ys], axis=1)
        bottom_pts = np.stack([bottom_xs, bottom_ys], axis=1)

        return pts, widths, top_pts, bottom_pts

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
        if getattr(self, "_committed_pos_np", None) is None or self._committed_pos_np.size == 0:
            return True

        # Radius check (fast vectorized)
        if self.uniqueness_radius > 0.0:
            d = self._committed_pos_np - pos.reshape(1, 3)
            d2 = np.einsum("ij,ij->i", d, d)  # squared distance
            return not np.any(d2 < (self.uniqueness_radius ** 2))

        # Fallback to your anisotropic check if radius disabled
        for ti in self.committed_trunks:
            p = ti.pose.position
            existing = np.array([p.x, p.y, p.z], dtype=float)
            delta = pos - existing
            s = np.dot(delta, self.row_axis)
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
        # marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.stamp = self.tree_image_data_timestamp if self.tree_image_data_timestamp is not None else self.get_clock().now().to_msg()
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
