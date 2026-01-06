#!/usr/bin/env python3

import math
import numpy as np

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
import tf2_ros

from scipy.spatial.transform import Rotation as R

from message_filters import Subscriber, ApproximateTimeSynchronizer


def quat_to_yaw(q) -> float:
    rot = R.from_quat([q.x, q.y, q.z, q.w])
    return float(rot.as_euler("xyz", degrees=False)[2])


def yaw_to_quat(yaw: float):
    rot = R.from_euler("z", yaw, degrees=False)
    qx, qy, qz, qw = rot.as_quat()
    return float(qx), float(qy), float(qz), float(qw)


def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + 1e-9 * float(stamp.nanosec)


class SlamOdomCorrectionTf(Node):
    """
    Publishes TF: odom_slam -> odom

    - Computes correction using ApproximateTimeSynchronizer (wheel odom + slam odom).
    - Caches the latest correction.
    - Re-publishes TF continuously stamped with the latest wheel odom stamp,
      so RViz always has TF coverage for messages arriving in 'odom'.
    """

    def __init__(self):
        super().__init__("slam_odom_correction_tf")

        # -------- Parameters --------
        self.declare_parameter("wheel_odom_topic", "/odom")
        self.declare_parameter("slam_odom_topic", "/odom_slam_best")

        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("odom_slam_frame", "odom_slam")

        self.declare_parameter("sync_queue_size", 50)
        self.declare_parameter("sync_slop_sec", 0.05)
        self.declare_parameter("sync_allow_headerless", False)

        # Continuous publishing
        self.declare_parameter("publish_rate_hz", 50.0)     # timer publish rate
        self.declare_parameter("publish_on_wheel_odom", True)

        wheel_topic = self.get_parameter("wheel_odom_topic").value
        slam_topic = self.get_parameter("slam_odom_topic").value
        self.odom_frame = self.get_parameter("odom_frame").value
        self.odom_slam_frame = self.get_parameter("odom_slam_frame").value

        qsize = int(self.get_parameter("sync_queue_size").value)
        slop = float(self.get_parameter("sync_slop_sec").value)
        allow_headerless = bool(self.get_parameter("sync_allow_headerless").value)

        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.publish_on_wheel_odom = bool(self.get_parameter("publish_on_wheel_odom").value)

        # -------- TF broadcaster --------
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # -------- State (cached correction) --------
        self._have_correction = False
        self._dx = 0.0
        self._dy = 0.0
        self._dyaw = 0.0

        self._last_wheel_stamp = None  # builtin_interfaces/msg/Time

        # -------- Subscriptions --------
        # 1) message_filters sync for computing correction
        self.wheel_sub_mf = Subscriber(self, Odometry, wheel_topic)
        self.slam_sub_mf = Subscriber(self, Odometry, slam_topic)

        self.sync = ApproximateTimeSynchronizer(
            [self.wheel_sub_mf, self.slam_sub_mf],
            queue_size=qsize,
            slop=slop,
            allow_headerless=allow_headerless,
        )
        self.sync.registerCallback(self.synced_cb)

        # 2) plain rclpy wheel odom subscription for continuous stamping/publishing
        #    (lets us publish TF at wheel rate even if SLAM updates slower)
        self.wheel_sub = self.create_subscription(
            Odometry, wheel_topic, self.wheel_odom_cb, 50
        )

        # -------- Timer for steady TF publishing --------
        if self.publish_rate_hz > 0.0:
            period = 1.0 / max(self.publish_rate_hz, 1e-6)
            self.timer = self.create_timer(period, self.timer_cb)
        else:
            self.timer = None

        self.get_logger().info(
            f"Computing correction from wheel='{wheel_topic}' and slam='{slam_topic}' "
            f"(slop={slop}s, queue={qsize}).\n"
            f"Publishing TF {self.odom_slam_frame} -> {self.odom_frame} continuously "
            f"(rate={self.publish_rate_hz:.1f}Hz, publish_on_wheel_odom={self.publish_on_wheel_odom})."
        )

    # -------- Callbacks --------

    def wheel_odom_cb(self, wheel_odom: Odometry):
        # Track the latest wheel stamp for stamping outgoing TF
        self._last_wheel_stamp = wheel_odom.header.stamp

        # Optionally publish TF at wheel odom rate (helps RViz a lot)
        if self.publish_on_wheel_odom:
            self._publish_cached_tf()

    def timer_cb(self):
        # Publish at a steady rate even if wheel odom is jittery
        self._publish_cached_tf()

    def synced_cb(self, wheel_odom: Odometry, slam_odom: Odometry):
        # Sanity checks (debug only)
        if wheel_odom.header.frame_id and wheel_odom.header.frame_id != self.odom_frame:
            self.get_logger().debug(
                f"wheel_odom.frame_id='{wheel_odom.header.frame_id}' expected '{self.odom_frame}'"
            )
        if slam_odom.header.frame_id and slam_odom.header.frame_id != self.odom_slam_frame:
            self.get_logger().debug(
                f"slam_odom.frame_id='{slam_odom.header.frame_id}' expected '{self.odom_slam_frame}'"
            )

        # wheel: base pose in odom
        wx = float(wheel_odom.pose.pose.position.x)
        wy = float(wheel_odom.pose.pose.position.y)
        wyaw = quat_to_yaw(wheel_odom.pose.pose.orientation)

        # slam: base pose in odom_slam
        sx = float(slam_odom.pose.pose.position.x)
        sy = float(slam_odom.pose.pose.position.y)
        syaw = quat_to_yaw(slam_odom.pose.pose.orientation)

        # Compute correction T(odom_slam -> odom)
        dyaw = wyaw - syaw
        dyaw = (dyaw + math.pi) % (2.0 * math.pi) - math.pi

        c = math.cos(dyaw)
        s = math.sin(dyaw)
        R_c = np.array([[c, -s],
                        [s,  c]], dtype=float)

        p_w = np.array([wx, wy], dtype=float)
        p_s = np.array([sx, sy], dtype=float)
        t_c = p_w - (R_c @ p_s)

        self._dx = float(t_c[0])
        self._dy = float(t_c[1])
        self._dyaw = float(dyaw)
        self._have_correction = True

        # Use wheel stamp as the master stamp for outgoing TF
        self._last_wheel_stamp = wheel_odom.header.stamp

        # Publish immediately on update
        self._publish_cached_tf()

    # -------- Internal --------

    def _publish_cached_tf(self):
        if not self._have_correction:
            return
        if self._last_wheel_stamp is None:
            return

        t = TransformStamped()
        t.header.stamp = self._last_wheel_stamp
        t.header.frame_id = self.odom_slam_frame
        t.child_frame_id = self.odom_frame

        t.transform.translation.x = self._dx
        t.transform.translation.y = self._dy
        t.transform.translation.z = 0.0

        qx, qy, qz, qw = yaw_to_quat(self._dyaw)
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw

        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = SlamOdomCorrectionTf()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
