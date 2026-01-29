#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped, Quaternion, Point
from tf2_ros import TransformBroadcaster
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import NavSatFix


class OdomReframer(Node):
    def __init__(self):
        super().__init__("odom_reframer")

        # self.declare_parameter("input_odom_topic", "/filter/state")
        self.declare_parameter("input_odom_topic", "/odometry/local")
        self.declare_parameter("output_odom_topic", "/odom")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "amiga__base")

        self.declare_parameter("recenter", False)

        # Split position vs orientation corrections
        self.declare_parameter("rotate_pos_deg", 0.0)          # keep 0.0 if motion direction is already correct
        self.declare_parameter("rotate_yaw_deg", 0.0)        # fix heading being 180 off
        self.declare_parameter("rotate_orientation_only", True)

        self.declare_parameter("use_now_stamp", False)
        self.declare_parameter("copy_twist", True)

        # --- GPS additions ---
        # self.declare_parameter("gps_topic", "/gps/pvt") 
        # self.declare_parameter("gps_topic", "/ublox_gps_corrected/fix")
        self.declare_parameter("gps_topic", "/fix")
        self.declare_parameter("gps_min_status", 0)         # 0=STATUS_FIX, 1=SBAS, 2=GBAS (NavSatStatus)
        self.declare_parameter("gps_require_finite_cov", False)

        input_topic = str(self.get_parameter("input_odom_topic").value)
        self.output_topic = str(self.get_parameter("output_odom_topic").value)
        self.odom_frame = str(self.get_parameter("odom_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)

        self.recenter = bool(self.get_parameter("recenter").value)

        self.rotate_pos_deg = float(self.get_parameter("rotate_pos_deg").value)
        self.rotate_yaw_deg = float(self.get_parameter("rotate_yaw_deg").value)
        self.rotate_orientation_only = bool(self.get_parameter("rotate_orientation_only").value)

        self.use_now_stamp = bool(self.get_parameter("use_now_stamp").value)
        self.copy_twist = bool(self.get_parameter("copy_twist").value)

        # --- GPS state ---
        self.gps_topic = str(self.get_parameter("gps_topic").value)
        self.gps_min_status = int(self.get_parameter("gps_min_status").value)
        self.gps_require_finite_cov = bool(self.get_parameter("gps_require_finite_cov").value)
        self.initial_gps: NavSatFix | None = None

        self.R_pos = R.from_euler("z", self.rotate_pos_deg, degrees=True) if abs(self.rotate_pos_deg) > 1e-6 else None
        self.R_yaw = R.from_euler("z", self.rotate_yaw_deg, degrees=True) if abs(self.rotate_yaw_deg) > 1e-6 else None

        self.initial_offset: np.ndarray | None = None

        self.sub = self.create_subscription(Odometry, input_topic, self.odom_callback, 10)
        self.pub = self.create_publisher(Odometry, self.output_topic, 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.initial_offset_pub = self.create_publisher(Point, "initial_odom_correction", 1)
        self.initial_gps_pub = self.create_publisher(Point, "initial_gps_fix", 1) # publish initial GPS as a Point: x=lat, y=lon, z=alt (meters)
        self.gps_sub = self.create_subscription(NavSatFix, self.gps_topic, self.gps_callback, 10)

        self.timer = self.create_timer(3.0, self.publish_initials)

        self.get_logger().info(
            "OdomReframer started.\n"
            f"  input: {input_topic} -> output: {self.output_topic}\n"
            f"  frames: {self.odom_frame} -> {self.base_frame}\n"
            f"  recenter={self.recenter}\n"
            f"  rotate_pos_deg={self.rotate_pos_deg}\n"
            f"  rotate_yaw_deg={self.rotate_yaw_deg} (orientation)\n"
            f"  rotate_orientation_only={self.rotate_orientation_only}\n"
            f"  gps_topic={self.gps_topic}\n"
            f"  use_now_stamp={self.use_now_stamp}"
        )

    def _get_stamp(self, msg: Odometry):
        return self.get_clock().now().to_msg() if self.use_now_stamp else msg.header.stamp

    def _capture_initial_offset(self, p: np.ndarray):
        if self.recenter and self.initial_offset is None:
            self.initial_offset = p.copy()
            self.get_logger().info(
                f"Captured initial offset: x={self.initial_offset[0]:.3f}, "
                f"y={self.initial_offset[1]:.3f}, z={self.initial_offset[2]:.3f}"
            )

    def _gps_is_acceptable(self, msg: NavSatFix) -> bool:
        # basic sanity: finite lat/lon
        if not np.isfinite(msg.latitude) or not np.isfinite(msg.longitude):
            return False

        # NavSatFix.status.status: -1 no fix, 0 fix, 1 SBAS, 2 GBAS
        if msg.status.status < self.gps_min_status:
            return False

        # optional: reject NaN covariance entries
        if self.gps_require_finite_cov:
            cov = np.array(msg.position_covariance, dtype=np.float64)
            if not np.all(np.isfinite(cov)):
                return False

        return True

    def gps_callback(self, msg: NavSatFix):
        if self.initial_gps is not None:
            return
        if not self._gps_is_acceptable(msg):
            return

        self.initial_gps = msg
        self.get_logger().info(
            f"Captured initial GPS fix: lat={msg.latitude:.8f}, lon={msg.longitude:.8f}, alt={msg.altitude:.3f}, "
            f"status={msg.status.status}"
        )

    def publish_initials(self):
        # publish initial odom correction (your existing behavior)
        if self.initial_offset is not None:
            pt_msg = Point()
            pt_msg.x = float(self.initial_offset[0])
            pt_msg.y = float(self.initial_offset[1])
            pt_msg.z = float(self.initial_offset[2])
            self.initial_offset_pub.publish(pt_msg)

        # publish initial gps fix (lat, lon, alt)
        if self.initial_gps is not None:
            gps_pt = Point()
            gps_pt.x = float(self.initial_gps.latitude)
            gps_pt.y = float(self.initial_gps.longitude)
            gps_pt.z = float(self.initial_gps.altitude)
            self.initial_gps_pub.publish(gps_pt)

    def _relative_position(self, p: np.ndarray) -> np.ndarray:
        if not self.recenter or self.initial_offset is None:
            return p
        return p - self.initial_offset

    @staticmethod
    def _quat_to_rot(q: Quaternion) -> R:
        return R.from_quat([q.x, q.y, q.z, q.w])

    @staticmethod
    def _rot_to_quat_msg(rot: R) -> Quaternion:
        x, y, z, w = rot.as_quat()
        return Quaternion(x=float(x), y=float(y), z=float(z), w=float(w))

    def odom_callback(self, msg: Odometry):
        stamp = self._get_stamp(msg)

        pos = np.array(
            [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z],
            dtype=np.float64,
        )
        self._capture_initial_offset(pos)
        rel = self._relative_position(pos)

        # --- Position: keep as-is unless you set rotate_pos_deg ---
        if self.R_pos is not None:
            pos_out = self.R_pos.apply(rel)
        else:
            pos_out = rel

        # --- Orientation: apply yaw correction (180 deg) ---
        q_in = msg.pose.pose.orientation
        if self.R_yaw is not None:
            R_in = self._quat_to_rot(q_in)
            # Pre-multiply is the typical frame-style correction
            R_out = self.R_yaw * R_in
            q_out = self._rot_to_quat_msg(R_out)
        else:
            q_out = q_in

        # If you want orientation-only correction while keeping position exactly as before,
        # ensure rotate_pos_deg is 0 and rotate_orientation_only True.
        if self.rotate_orientation_only:
            # explicitly keep whatever your translation is currently doing
            pos_out = rel if self.R_pos is None else pos_out

        # ---------- publish Odometry ----------
        out = Odometry()
        out.header = msg.header
        out.header.stamp = stamp
        out.header.frame_id = self.odom_frame
        out.child_frame_id = self.base_frame

        out.pose.pose.position.x = float(pos_out[0])
        out.pose.pose.position.y = float(pos_out[1])
        out.pose.pose.position.z = float(pos_out[2])
        out.pose.pose.orientation = q_out

        if self.copy_twist:
            out.twist = msg.twist

            # If we rotated the pose orientation, rotate twist vectors too
            if self.R_yaw is not None:
                # Rotate linear velocity (v) in the child/base frame
                v = out.twist.twist.linear
                v_vec = np.array([v.x, v.y, v.z], dtype=np.float64)
                v_rot = self.R_yaw.apply(v_vec)
                v.x, v.y, v.z = float(v_rot[0]), float(v_rot[1]), float(v_rot[2])

                # Rotate angular velocity (w) in the child/base frame
                w = out.twist.twist.angular
                w_vec = np.array([w.x, w.y, w.z], dtype=np.float64)
                w_rot = self.R_yaw.apply(w_vec)
                w.x, w.y, w.z = float(w_rot[0]), float(w_rot[1]), float(w_rot[2])

        self.pub.publish(out)

        # ---------- publish TF ----------
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = self.odom_frame
        t.child_frame_id = self.base_frame
        t.transform.translation.x = float(pos_out[0])
        t.transform.translation.y = float(pos_out[1])
        t.transform.translation.z = float(pos_out[2])
        t.transform.rotation = q_out
        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = OdomReframer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
