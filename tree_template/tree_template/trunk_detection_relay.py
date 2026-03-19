#!/usr/bin/env python3
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration

from geometry_msgs.msg import Pose
from visualization_msgs.msg import Marker

from pf_orchard_interfaces.msg import TreeImageData
from tree_template_interfaces.msg import TrunkInfo

import tf2_ros
from scipy.spatial.transform import Rotation as R


class TrunkDetectionRelay(Node):
    """
    Relay node (no clustering, no committing, no map-frame logic).

    Input:  TreeImageData (camera detections)
    Output: TrunkInfo measurements, one per detection, in a robot-relative frame.

    FastSLAM should be the only node that:
      - associates detections to landmarks/slots
      - decides when a trunk is committed
      - publishes any registry/map products
    """

    def __init__(self):
        super().__init__("trunk_detection_relay")

        # -------- Parameters --------
        self.declare_parameter("input_topic", "tree_image_data")
        self.declare_parameter("camera_frame", "base_camera_color_optical_frame")
        self.declare_parameter("output_frame", "amiga__base")
        self.declare_parameter("output_topic", "trunk_measurements_raw")

        # Optional measurement tweak
        self.declare_parameter("use_trunk_width_addition", True)

        # Filtering
        self.declare_parameter("min_trunk_width", 0.03)     # meters; 0 disables
        self.declare_parameter("required_class", 0)        # only allow this class id

        # Optional debug marker publishing
        self.declare_parameter("publish_debug_markers", True)
        self.declare_parameter("debug_marker_topic", "trunk_measurements_raw_markers")
        self.declare_parameter("debug_marker_scale", 0.20)  # meters

        # Logging control
        self.declare_parameter("log_rejection_every_n_msgs", 30)  # 0 disables

        # -------- Read params --------
        self.input_topic = str(self.get_parameter("input_topic").value)
        self.camera_frame = str(self.get_parameter("camera_frame").value)
        self.output_frame = str(self.get_parameter("output_frame").value)
        self.output_topic = str(self.get_parameter("output_topic").value)

        self.use_trunk_width_addition = bool(self.get_parameter("use_trunk_width_addition").value)

        self.min_trunk_width = float(self.get_parameter("min_trunk_width").value)
        self.required_class = int(self.get_parameter("required_class").value)

        self.publish_debug_markers = bool(self.get_parameter("publish_debug_markers").value)
        self.debug_marker_topic = str(self.get_parameter("debug_marker_topic").value)
        self.debug_marker_scale = float(self.get_parameter("debug_marker_scale").value)

        self.log_rejection_every_n_msgs = int(self.get_parameter("log_rejection_every_n_msgs").value)
        self._rej_log_counter = 0

        # -------- TF2 --------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self._T_cached: Optional[np.ndarray] = None

        # -------- Pub/Sub --------
        self.meas_pub = self.create_publisher(TrunkInfo, self.output_topic, 10)

        self.marker_pub = None
        if self.publish_debug_markers:
            self.marker_pub = self.create_publisher(Marker, self.debug_marker_topic, 10)

        self.sub = self.create_subscription(
            TreeImageData,
            self.input_topic,
            self.tree_image_callback,
            10,
        )

        self.get_logger().info(
            "TrunkDetectionRelay started.\n"
            f"  Subscribing: {self.input_topic}\n"
            f"  Publishing measurements: {self.output_topic}\n"
            f"  Frames: camera_frame='{self.camera_frame}', output_frame='{self.output_frame}'\n"
            f"  width_addition={self.use_trunk_width_addition}\n"
            f"  min_trunk_width={self.min_trunk_width:.3f} m, required_class={self.required_class}\n"
            f"  debug_markers={self.publish_debug_markers}"
        )

    # ---------------- Callback ----------------

    def tree_image_callback(self, msg: TreeImageData) -> None:
        if not msg.object_seen:
            return

        pts, widths, top_pts, bottom_pts, classes = self.extract_points_and_widths_from_msg(msg)
        if pts.size == 0 or top_pts.size == 0 or bottom_pts.size == 0:
            return

        n0 = int(pts.shape[0])

        # -------------------------------------------------
        # Build a single validity mask across all criteria
        # -------------------------------------------------
        valid = np.ones((n0,), dtype=bool)

        # Class gating (if classifications present)
        if classes is not None and classes.size == n0:
            valid &= (classes == self.required_class)

        # Width gating (if widths present and threshold enabled)
        if widths is not None and widths.size == n0 and self.min_trunk_width > 0.0:
            valid &= np.isfinite(widths)
            valid &= (widths >= self.min_trunk_width)

        kept = int(np.sum(valid))
        dropped = int(n0 - kept)

        # Log rejections even if NOTHING passes (throttled)
        if dropped > 0 and self.log_rejection_every_n_msgs != 0:
            self._rej_log_counter += 1
            if (self._rej_log_counter % self.log_rejection_every_n_msgs) == 0:
                # Provide useful breakdown if available
                parts = [f"kept={kept}/{n0}"]

                if classes is not None and classes.size == n0:
                    bad_class = int(np.sum(classes != self.required_class))
                    parts.append(f"bad_class={bad_class}")

                if widths is not None and widths.size == n0 and self.min_trunk_width > 0.0:
                    bad_width = int(np.sum(~np.isfinite(widths) | (widths < self.min_trunk_width)))
                    parts.append(f"bad_width={bad_width} (<{self.min_trunk_width:.3f}m)")

                self.get_logger().debug("Detection filtering: " + ", ".join(parts))

        if kept == 0:
            return

        # Apply mask
        pts = pts[valid]
        top_pts = top_pts[valid]
        bottom_pts = bottom_pts[valid]
        if widths is not None and widths.size == n0:
            widths = widths[valid]
        else:
            widths = None  # avoid mismatched lengths downstream

        n = int(pts.shape[0])

        # --- Depth/radius tweak on DEPTH ---
        z_depth = bottom_pts[:, 1].astype(np.float64)  # meters, positive
        if self.use_trunk_width_addition and widths is not None:
            radii = widths.astype(np.float64) * 0.5
            z_depth = z_depth + radii

        # --- Lateral in your message convention ---
        x = pts[:, 0].astype(np.float64)

        # Build point in camera OPTICAL frame for TF:
        # Optical: X=RIGHT, Y=DOWN, Z=FORWARD(depth)
        pos_cam = np.empty((n, 3), dtype=np.float64)
        pos_cam[:, 0] = x         # optical X (RIGHT)
        pos_cam[:, 1] = 0.0       # optical Y (DOWN)
        pos_cam[:, 2] = z_depth   # optical Z (FORWARD / depth)

        # TF: camera optical -> base (or output frame)
        if self.output_frame != self.camera_frame:
            if self._T_cached is None:
                self._T_cached = self._lookup_T(self.output_frame, self.camera_frame)
            if self._T_cached is None:
                return
            p_base = self._apply_T_batch(self._T_cached, pos_cam)
        else:
            p_base = pos_cam

        # Mounting relationship: optical Z aligns with base Y (depth -> base Y)
        # Publish TrunkInfo in SLAM convention:
        #   pose.x = x_fwd = base_Y
        #   pose.y = y_lat = +/- base_X (left+)
        base_x_is_right = True
        lat_sign = -1.0 if base_x_is_right else 1.0

        for i in range(n):
            bx = float(p_base[i, 0])  # base X
            by = -float(p_base[i, 1])  # base Y

            ti = TrunkInfo()
            ti.pose = Pose()

            if hasattr(msg, "header"):
                ti.stamp = msg.header.stamp
            elif hasattr(msg, "stamp"):
                ti.stamp = msg.stamp
            else:
                ti.stamp = self.get_clock().now().to_msg()

            ti.pose.position.x = bx                # SLAM forward
            ti.pose.position.y = lat_sign * by     # SLAM lateral (left+)
            ti.pose.position.z = 0.0

            ti.pose.orientation.w = 1.0

            if widths is None or not np.isfinite(widths[i]):
                ti.width = float("nan")
            else:
                ti.width = float(widths[i])

            ti.side = ""
            self.meas_pub.publish(ti)

            if self.marker_pub is not None:
                self._publish_debug_marker(
                    np.array([ti.pose.position.x, ti.pose.position.y, 0.0], dtype=np.float64),
                    idx=i
                )

    # ---------------- Debug marker ----------------

    def _publish_debug_marker(self, p: np.ndarray, idx: int) -> None:
        m = Marker()
        m.header.frame_id = self.output_frame
        m.header.stamp = self.get_clock().now().to_msg()

        m.ns = "trunk_measurements_raw"
        m.id = int(idx)

        m.type = Marker.SPHERE
        m.action = Marker.ADD

        m.pose.position.x = float(p[0])
        m.pose.position.y = float(p[1])
        m.pose.position.z = float(p[2])
        m.pose.orientation.w = 1.0

        s = float(self.debug_marker_scale)
        m.scale.x = s
        m.scale.y = s
        m.scale.z = s

        m.color.a = 0.9
        m.color.r = 0.2
        m.color.g = 0.8
        m.color.b = 1.0

        m.lifetime.sec = 0
        m.lifetime.nanosec = int(0.5e9)

        self.marker_pub.publish(m)

    # ---------------- TF helpers ----------------

    def _lookup_T(self, target_frame: str, source_frame: str) -> Optional[np.ndarray]:
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                Time(),  # latest
                timeout=Duration(seconds=0.2),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            self.get_logger().warn(f"TF lookup failed ({source_frame}->{target_frame}): {e}")
            return None

        return self._transformstamped_to_T(tf_msg)

    def _transformstamped_to_T(self, tf_msg) -> np.ndarray:
        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation
        Rm = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = Rm
        T[:3, 3] = np.array([t.x, t.y, t.z], dtype=np.float64)
        return T

    def _apply_T_batch(self, T: np.ndarray, P: np.ndarray) -> np.ndarray:
        return (P @ T[:3, :3].T) + T[:3, 3]

    # ---------------- Message parsing ----------------

    def extract_points_and_widths_from_msg(
        self, msg: TreeImageData
    ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        TreeImageData -> arrays
          pts: (N,2) from xs, ys
          widths: (N,) or None
          top_pts: (N,2) from top_xs, top_ys
          bottom_pts: (N,2) from bottom_xs, bottom_ys
          classes: (N,) or None
        """
        xs = np.array(msg.xs, dtype=np.float32)
        ys = np.array(msg.ys, dtype=np.float32)
        top_xs = np.array(msg.top_xs, dtype=np.float32)
        top_ys = np.array(msg.top_ys, dtype=np.float32)
        bottom_xs = np.array(msg.bottom_xs, dtype=np.float32)
        bottom_ys = np.array(msg.bottom_ys, dtype=np.float32)

        widths = np.array(msg.widths, dtype=np.float32) if msg.widths else None
        classes = np.array(msg.classifications, dtype=np.int32) if msg.classifications else None

        empty = np.empty((0, 2), dtype=np.float32)
        if xs.size == 0 or ys.size == 0 or top_xs.size == 0 or top_ys.size == 0 or bottom_xs.size == 0 or bottom_ys.size == 0:
            return empty, None, empty, empty, None

        # Determine common length across available arrays
        lens = [xs.size, ys.size, top_xs.size, top_ys.size, bottom_xs.size, bottom_ys.size]
        if widths is not None:
            lens.append(widths.size)
        if classes is not None:
            lens.append(classes.size)

        n = int(min(lens))
        if n <= 0:
            return empty, None, empty, empty, None

        xs = xs[:n]
        ys = ys[:n]
        top_xs = top_xs[:n]
        top_ys = top_ys[:n]
        bottom_xs = bottom_xs[:n]
        bottom_ys = bottom_ys[:n]

        if widths is not None:
            widths = widths[:n]
            if widths.size == 0:
                widths = None

        if classes is not None:
            classes = classes[:n]
            if classes.size == 0:
                classes = None

        pts = np.stack([xs, ys], axis=1)
        top_pts = np.stack([top_xs, top_ys], axis=1)
        bottom_pts = np.stack([bottom_xs, bottom_ys], axis=1)
        return pts, widths, top_pts, bottom_pts, classes


def main():
    rclpy.init()
    node = TrunkDetectionRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
