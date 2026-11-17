#!/usr/bin/env python3
"""
GPS course-over-ground as an absolute ENU heading, published as sensor_msgs/Imu orientation.

The robot's IMU yaw is not north-referenced, so navsat_transform and the global EKF have no true heading. With
RTK GPS (~2 cm), the direction between two fixes >= `baseline_m` apart is a good heading (< 1 deg jitter over 1.5 m)
while the robot drives straight. This node is a conversion, not a filter:

  for each fix: take the latest earlier fix >= baseline_m away (within max_dt_s); publish
  yaw = atan2(dN, dE) (+pi when reversing) only if the wheel odometry turned < max_turn_deg over that interval and
  |v| > min_speed. Stamped at the middle of the interval (consumers need smooth_lagged_data / history_length).
  yaw sigma = sqrt(2) * sigma_pos_m / baseline + floor_deg; roll/pitch covariance huge.

Consumers: navsat_transform (imu input, use_odometry_yaw false, yaw_offset 0) and ekf_global (imuN, yaw only).
"""
import math
from collections import deque
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import NavSatFix, Imu
from nav_msgs.msg import Odometry

R_EARTH = 6378137.0


def stamp_s(h) -> float:
    return h.stamp.sec + 1e-9 * h.stamp.nanosec


class GpsCourseHeading(Node):
    def __init__(self):
        super().__init__("gps_course_heading")
        self.declare_parameter("fix_topic", "/fix_rtk")
        self.declare_parameter("odom_topic", "/odometry/wheel")
        self.declare_parameter("output_topic", "/gps/course_imu")
        self.declare_parameter("frame_id", "amiga__base")
        self.declare_parameter("baseline_m", 1.5)
        self.declare_parameter("max_dt_s", 12.0)
        self.declare_parameter("sigma_pos_m", 0.025)
        self.declare_parameter("floor_deg", 0.5)
        self.declare_parameter("max_turn_deg", 3.0)
        self.declare_parameter("min_speed", 0.05)
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.frame_id = str(g("frame_id"))
        self.baseline = float(g("baseline_m"))
        self.max_dt = float(g("max_dt_s"))
        self.sigma_pos = float(g("sigma_pos_m"))
        self.floor = math.radians(float(g("floor_deg")))
        self.max_turn = math.radians(float(g("max_turn_deg")))
        self.min_speed = float(g("min_speed"))
        self.ref = None
        self.fixes = deque(maxlen=60)
        self.odom = deque(maxlen=2000)  # (t, yaw, vx)
        self.pub = self.create_publisher(Imu, str(g("output_topic")), 10)
        self.create_subscription(NavSatFix, str(g("fix_topic")), self.on_fix, qos_profile_sensor_data)
        self.create_subscription(Odometry, str(g("odom_topic")), self.on_odom, 50)
        self.n = 0
        self.get_logger().info(f"{g('fix_topic')} -> {g('output_topic')}: baseline {self.baseline} m, "
                               f"max turn {g('max_turn_deg')} deg, frame {self.frame_id}")

    def on_odom(self, m: Odometry):
        q = m.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.odom.append((stamp_s(m.header), yaw, m.twist.twist.linear.x))

    def odom_at(self, t: float):
        if not self.odom:
            return None
        r = min(self.odom, key=lambda e: abs(e[0] - t))
        return r if abs(r[0] - t) < 0.3 else None

    def on_fix(self, m: NavSatFix):
        if m.status.status < 0:
            return
        t = stamp_s(m.header)
        if self.ref is None:
            self.ref = (m.latitude, m.longitude)
        x = math.radians(m.longitude - self.ref[1]) * R_EARTH * math.cos(math.radians(self.ref[0]))
        y = math.radians(m.latitude - self.ref[0]) * R_EARTH
        self.fixes.append((t, x, y))
        for t0, x0, y0 in reversed(list(self.fixes)[:-1]):
            if t - t0 > self.max_dt:
                return
            d = math.hypot(x - x0, y - y0)
            if d < self.baseline:
                continue
            o0, o1 = self.odom_at(t0), self.odom_at(t)
            if o0 is None or o1 is None:
                return
            turn = abs(math.atan2(math.sin(o1[1] - o0[1]), math.cos(o1[1] - o0[1])))
            v = 0.5 * (o0[2] + o1[2])
            if turn > self.max_turn or abs(v) < self.min_speed:
                return
            yaw = math.atan2(y - y0, x - x0) + (math.pi if v < 0 else 0.0)
            s = math.sqrt(2.0) * self.sigma_pos / d + self.floor
            out = Imu()
            tm = 0.5 * (t + t0)
            out.header.stamp.sec = int(tm)
            out.header.stamp.nanosec = int(round((tm - int(tm)) * 1e9)) % 1_000_000_000
            out.header.frame_id = self.frame_id
            out.orientation.z, out.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
            out.orientation_covariance = [1e3, 0.0, 0.0, 0.0, 1e3, 0.0, 0.0, 0.0, float(s * s)]
            out.angular_velocity_covariance = [1e3, 0.0, 0.0, 0.0, 1e3, 0.0, 0.0, 0.0, 1e3]
            out.linear_acceleration_covariance = [-1.0] + [0.0] * 8
            self.pub.publish(out)
            self.n += 1
            if self.n % 100 == 0:
                self.get_logger().info(f"published {self.n} course headings")
            return


def main(args=None):
    rclpy.init(args=args)
    node = GpsCourseHeading()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
