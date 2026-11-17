#!/usr/bin/env python3
"""
Republish a NavSatFix with a realistic position covariance (and optional stamp shift).

The 2023 Reach RS publishes a ~1 m placeholder covariance while its real (RTK) noise is ~2 cm, so
robot_localization would barely trust it. This relay sets a diagonal covariance from `sigma_m` (vertical 2x) and
COVARIANCE_TYPE_DIAGONAL_KNOWN, drops fixes below `min_status`, and adds `stamp_shift_s` to the header stamp
(for a known constant GPS latency / clock offset; 0 when clocks are synchronised).
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import NavSatFix


class FixCovRelay(Node):
    def __init__(self):
        super().__init__("fix_cov_relay")
        self.declare_parameter("input_topic", "/fix")
        self.declare_parameter("output_topic", "/fix_rtk")
        self.declare_parameter("sigma_m", 0.05)
        self.declare_parameter("min_status", 0)
        self.declare_parameter("stamp_shift_s", 0.0)
        tin = str(self.get_parameter("input_topic").value)
        tout = str(self.get_parameter("output_topic").value)
        sigma = float(self.get_parameter("sigma_m").value)
        self.var = sigma * sigma
        self.min_status = int(self.get_parameter("min_status").value)
        self.shift_ns = int(round(float(self.get_parameter("stamp_shift_s").value) * 1e9))
        self.pub = self.create_publisher(NavSatFix, tout, 10)
        self.create_subscription(NavSatFix, tin, self.cb, qos_profile_sensor_data)
        self.get_logger().info(f"{tin} -> {tout}: sigma {sigma} m, min_status {self.min_status}, "
                               f"stamp shift {self.shift_ns * 1e-9:+.3f} s")

    def cb(self, m: NavSatFix):
        if m.status.status < self.min_status:
            return
        m.position_covariance = [self.var, 0.0, 0.0, 0.0, self.var, 0.0, 0.0, 0.0, 4.0 * self.var]
        m.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN
        if self.shift_ns:
            ns = m.header.stamp.sec * 1_000_000_000 + m.header.stamp.nanosec + self.shift_ns
            m.header.stamp.sec, m.header.stamp.nanosec = int(ns // 1_000_000_000), int(ns % 1_000_000_000)
        self.pub.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = FixCovRelay()
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
