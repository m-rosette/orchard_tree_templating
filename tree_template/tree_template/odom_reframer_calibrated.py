#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped, Quaternion, Point, Pose
from tf2_ros import TransformBroadcaster
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import NavSatFix


class OdomReframer(Node):
    def __init__(self):
        super().__init__("odom_reframer")

        self.declare_parameter("input_odom_topic", "/odometry/local")
        self.declare_parameter("output_odom_topic", "/odom")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "amiga__base")
        self.declare_parameter("publish_tf", True)

        self.declare_parameter("recenter", False)

        # Split position vs orientation corrections
        self.declare_parameter("rotate_pos_deg", 0.0)          # keep 0.0 if motion direction is already correct
        self.declare_parameter("rotate_yaw_deg", 0.0)          # fix heading being 180 off
        self.declare_parameter("rotate_orientation_only", True)

        self.declare_parameter("use_now_stamp", False)
        self.declare_parameter("copy_twist", True)

        # --- GPS additions ---
        self.declare_parameter("gps_topic", "/fix")
        self.declare_parameter("gps_min_status", 0)         # 0=STATUS_FIX, 1=SBAS, 2=GBAS (NavSatStatus)
        self.declare_parameter("gps_require_finite_cov", False)

        # --- Averaging of initial readings ---
        self.declare_parameter("initial_odom_samples", 30)   # number of odom readings to average
        self.declare_parameter("initial_gps_samples", 30)    # number of gps readings to average

        input_topic = str(self.get_parameter("input_odom_topic").value)
        self.output_topic = str(self.get_parameter("output_odom_topic").value)
        self.odom_frame = str(self.get_parameter("odom_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.publish_tf = bool(self.get_parameter("publish_tf").value)

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

        # --- Averaging parameters ---
        self.initial_odom_samples = int(self.get_parameter("initial_odom_samples").value)
        self.initial_gps_samples = int(self.get_parameter("initial_gps_samples").value)

        # --- Accumulation buffers ---
        self.odom_positions: list[np.ndarray] = []  # accumulate position vectors
        self.odom_orientations: list[R] = []        # accumulate rotation objects
        self.gps_readings: list[NavSatFix] = []     # accumulate GPS readings

        self.R_pos = R.from_euler("z", self.rotate_pos_deg, degrees=True) if abs(self.rotate_pos_deg) > 1e-6 else None
        self.R_yaw = R.from_euler("z", self.rotate_yaw_deg, degrees=True) if abs(self.rotate_yaw_deg) > 1e-6 else None

        # --- Initial odom pose ---
        # Always capture/publish the first raw odom Pose (for downstream YAML dumping),
        # but only store initial_pos for recenter subtraction when recenter=True.
        self.initial_pose_msg: Pose | None = None
        self.initial_pos: np.ndarray | None = None

        # Ensure we only publish these "initials" once (they're latched anyway, but nice for clarity)
        self._published_initial_odom = False
        self._published_initial_gps = False

        self.sub = self.create_subscription(Odometry, input_topic, self.odom_callback, 10)
        self.pub = self.create_publisher(Odometry, self.output_topic, 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # Latched QoS for initials (so late subscribers still get them)
        qos_latched = QoSProfile(depth=1)
        qos_latched.durability = DurabilityPolicy.TRANSIENT_LOCAL
        qos_latched.reliability = ReliabilityPolicy.RELIABLE

        # NOTE: create_publisher signature is (msg_type, topic, qos_profile)
        self.initial_offset_pub = self.create_publisher(Pose, "initial_odom_correction", qos_latched)
        self.initial_gps_pub = self.create_publisher(Point, "initial_gps_fix", qos_latched)

        self.gps_sub = self.create_subscription(NavSatFix, self.gps_topic, self.gps_callback, 10)

        self.get_logger().info(
            "OdomReframer started.\n"
            f"  input: {input_topic} -> output: {self.output_topic}\n"
            f"  frames: {self.odom_frame} -> {self.base_frame}\n"
            f"  recenter={self.recenter}\n"
            f"  rotate_pos_deg={self.rotate_pos_deg}\n"
            f"  rotate_yaw_deg={self.rotate_yaw_deg} (orientation)\n"
            f"  rotate_orientation_only={self.rotate_orientation_only}\n"
            f"  gps_topic={self.gps_topic}\n"
            f"  use_now_stamp={self.use_now_stamp}\n"
            f"  initial_odom_samples={self.initial_odom_samples}\n"
            f"  initial_gps_samples={self.initial_gps_samples}\n"
            "  initials: publishing once with TRANSIENT_LOCAL durability"
        )

    def _get_stamp(self, msg: Odometry):
        return self.get_clock().now().to_msg() if self.use_now_stamp else msg.header.stamp

    def _maybe_capture_and_publish_initial_pose(self, msg: Odometry, pos_vec: np.ndarray) -> None:
        """
        Accumulate odom readings until we have initial_odom_samples, then compute average.

        - Accumulates position and orientation from incoming messages.
        - Once we reach the target sample count, averages them and publishes ONCE.
        - Only stores initial_pos for subtraction when recenter=True.
        """
        if self.initial_pose_msg is not None:
            return

        # Accumulate this reading
        self.odom_positions.append(pos_vec.copy())
        q_in = msg.pose.pose.orientation
        R_in = self._quat_to_rot(q_in)
        self.odom_orientations.append(R_in)

        # Check if we have enough samples
        if len(self.odom_positions) < self.initial_odom_samples:
            self.get_logger().debug(
                f"Accumulating odom: {len(self.odom_positions)}/{self.initial_odom_samples}"
            )
            return

        # Average positions
        pos_avg = np.mean(np.array(self.odom_positions), axis=0)

        # Average orientations: convert all to rotation vectors, average, convert back
        rot_vecs = np.array([rot.as_rotvec() for rot in self.odom_orientations])
        rot_vec_avg = np.mean(rot_vecs, axis=0)
        R_avg = R.from_rotvec(rot_vec_avg)

        pose = Pose()
        pose.position.x = float(pos_avg[0])
        pose.position.y = float(pos_avg[1])
        pose.position.z = float(pos_avg[2])
        pose.orientation = self._rot_to_quat_msg(R_avg)

        self.initial_pose_msg = pose

        # Only store initial_pos for recenter subtraction if enabled
        if self.recenter:
            self.initial_pos = pos_avg.copy()

        # Publish ONCE (latched topic)
        if not self._published_initial_odom:
            self.initial_offset_pub.publish(self.initial_pose_msg)
            self._published_initial_odom = True

        self.get_logger().info(
            f"Captured and averaged {self.initial_odom_samples} odom readings, published initial pose:\n"
            f"  pos=({pose.position.x:.3f}, {pose.position.y:.3f}, {pose.position.z:.3f})\n"
            f"  quat=({pose.orientation.x:.4f}, {pose.orientation.y:.4f}, "
            f"{pose.orientation.z:.4f}, {pose.orientation.w:.4f})\n"
            f"  recenter_store={'yes' if self.recenter else 'no'}"
        )

    def _gps_is_acceptable(self, msg: NavSatFix) -> bool:
        if not np.isfinite(msg.latitude) or not np.isfinite(msg.longitude):
            return False
        if msg.status.status < self.gps_min_status:
            return False
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

        # Accumulate this reading
        self.gps_readings.append(msg)

        # Check if we have enough samples
        if len(self.gps_readings) < self.initial_gps_samples:
            self.get_logger().debug(
                f"Accumulating GPS: {len(self.gps_readings)}/{self.initial_gps_samples}"
            )
            return

        # Average GPS readings
        lats = np.array([gps.latitude for gps in self.gps_readings])
        lons = np.array([gps.longitude for gps in self.gps_readings])
        alts = np.array([gps.altitude for gps in self.gps_readings])

        lat_avg = float(np.mean(lats))
        lon_avg = float(np.mean(lons))
        alt_avg = float(np.mean(alts))

        # Store the first message as reference (for status field, etc.)
        self.initial_gps = msg

        # Publish ONCE (latched topic): x=lat, y=lon, z=alt(m)
        if not self._published_initial_gps:
            gps_pt = Point()
            gps_pt.x = lat_avg
            gps_pt.y = lon_avg
            gps_pt.z = alt_avg
            self.initial_gps_pub.publish(gps_pt)
            self._published_initial_gps = True

        self.get_logger().info(
            f"Captured and averaged {self.initial_gps_samples} GPS readings, published initial fix: "
            f"lat={lat_avg:.8f}, lon={lon_avg:.8f}, alt={alt_avg:.3f}, status={self.gps_readings[0].status.status}"
        )

    def _relative_position(self, p: np.ndarray) -> np.ndarray:
        if not self.recenter or self.initial_pos is None:
            return p
        return p - self.initial_pos

    @staticmethod
    def _quat_to_rot(q: Quaternion) -> R:
        return R.from_quat([q.x, q.y, q.z, q.w])

    @staticmethod
    def _rot_to_quat_msg(rot: R) -> Quaternion:
        x, y, z, w = rot.as_quat()
        return Quaternion(x=float(x), y=float(y), z=float(z), w=float(w))

    def odom_callback(self, msg: Odometry):
        stamp = self._get_stamp(msg)

        # --- Extract full pose (position + orientation) from Odometry ---
        p_in = msg.pose.pose.position
        q_in = msg.pose.pose.orientation

        pos = np.array([p_in.x, p_in.y, p_in.z], dtype=np.float64)

        # Capture + publish initial pose once (raw from first odom)
        self._maybe_capture_and_publish_initial_pose(msg, pos)

        rel = self._relative_position(pos)

        # --- Position: keep as-is unless you set rotate_pos_deg ---
        if self.R_pos is not None:
            pos_out = self.R_pos.apply(rel)
        else:
            pos_out = rel

        # --- Orientation: apply yaw correction (e.g. 180 deg) ---
        if self.R_yaw is not None:
            R_in = self._quat_to_rot(q_in)
            R_out = self.R_yaw * R_in
            q_out = self._rot_to_quat_msg(R_out)
        else:
            q_out = q_in

        if self.rotate_orientation_only:
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

            if self.R_yaw is not None:
                v = out.twist.twist.linear
                v_vec = np.array([v.x, v.y, v.z], dtype=np.float64)
                v_rot = self.R_yaw.apply(v_vec)
                v.x, v.y, v.z = float(v_rot[0]), float(v_rot[1]), float(v_rot[2])

                w = out.twist.twist.angular
                w_vec = np.array([w.x, w.y, w.z], dtype=np.float64)
                w_rot = self.R_yaw.apply(w_vec)
                w.x, w.y, w.z = float(w_rot[0]), float(w_rot[1]), float(w_rot[2])

        self.pub.publish(out)

        # ---------- publish TF ----------
        if self.publish_tf:
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
