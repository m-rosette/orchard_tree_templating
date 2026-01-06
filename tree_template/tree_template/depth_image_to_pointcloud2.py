#!/usr/bin/env python3

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
)

from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from std_msgs.msg import Header

from message_filters import ApproximateTimeSynchronizer, Subscriber


def create_point_cloud2_xyzrgb(header: Header,
                               points_xyz: np.ndarray,
                               colors_rgb: np.ndarray) -> PointCloud2:
    """
    Create a PointCloud2 message from:
      - points_xyz: (N, 3) float32 array [x, y, z]
      - colors_rgb: (N, 3) uint8 array [r, g, b]

    Packs RGB into a single float32 'rgb' field.
    """
    assert points_xyz.shape[0] == colors_rgb.shape[0]
    n_points = points_xyz.shape[0]

    # Pack r,g,b into uint32, then view as float32
    r = colors_rgb[:, 0].astype(np.uint32)
    g = colors_rgb[:, 1].astype(np.uint32)
    b = colors_rgb[:, 2].astype(np.uint32)

    rgb_uint32 = (r << 16) | (g << 8) | b
    rgb_float32 = rgb_uint32.view(np.float32)

    cloud = np.zeros((n_points, 4), dtype=np.float32)
    cloud[:, 0:3] = points_xyz
    cloud[:, 3] = rgb_float32

    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = n_points

    msg.fields = [
        PointField(name='x',   offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y',   offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z',   offset=8,  datatype=PointField.FLOAT32, count=1),
        PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
    ]

    msg.is_bigendian = False
    msg.point_step = 16  # 4 float32 = 16 bytes
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = True
    msg.data = cloud.tobytes()

    return msg


class DepthToPointCloudRGBSyncNode(Node):
    def __init__(self):
        super().__init__('depth_to_pointcloud_rgb_sync')

        # Parameters
        self.declare_parameter('depth_topic', '/camera/base_camera/depth/image_rect_raw')
        self.declare_parameter('depth_camera_info_topic', '/camera/base_camera/depth/camera_info')
        self.declare_parameter('color_topic', '/camera/base_camera/color/image_raw')
        self.declare_parameter('color_camera_info_topic', '/camera/base_camera/color/camera_info')
        self.declare_parameter('points_topic', '/camera/base_camera/depth/points_rgb_sync')
        # 0.001 for 16UC1 in millimeters, 1.0 for meters
        self.declare_parameter('depth_scale', 0.001)
        # Downsample in u,v for lighter clouds
        self.declare_parameter('stride', 1)
        # ApproximateTimeSynchronizer slop (seconds)
        self.declare_parameter('sync_slop', 0.045) # originally had it at 0.01 but the mean delta is 0.035
        # ApproximateTimeSynchronizer queue size
        self.declare_parameter('sync_queue_size', 10)
        # Maximum depth (meters) to keep in the published point cloud
        self.declare_parameter('max_depth', 3.0)

        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        depth_info_topic = self.get_parameter('depth_camera_info_topic').get_parameter_value().string_value
        color_topic = self.get_parameter('color_topic').get_parameter_value().string_value
        color_info_topic = self.get_parameter('color_camera_info_topic').get_parameter_value().string_value
        points_topic = self.get_parameter('points_topic').get_parameter_value().string_value

        self.depth_scale = self.get_parameter('depth_scale').get_parameter_value().double_value
        self.stride = self.get_parameter('stride').get_parameter_value().integer_value
        self.sync_slop = self.get_parameter('sync_slop').get_parameter_value().double_value
        self.sync_queue_size = self.get_parameter('sync_queue_size').get_parameter_value().integer_value
        self.max_depth = self.get_parameter('max_depth').get_parameter_value().double_value

        if self.stride < 1:
            self.get_logger().warn("stride < 1, clamping to 1")
            self.stride = 1

        self.get_logger().info(
            f"DepthToPointCloudRGBSyncNode:\n"
            f"  depth_topic:            {depth_topic}\n"
            f"  depth_camera_info:      {depth_info_topic}\n"
            f"  color_topic:            {color_topic}\n"
            f"  color_camera_info:      {color_info_topic}\n"
            f"  points_topic:           {points_topic}\n"
            f"  depth_scale:            {self.depth_scale}\n"
            f"  stride:                 {self.stride}\n"
            f"  sync_slop:              {self.sync_slop} s\n"
            f"  max_depth:              {self.max_depth} m"
        )

        # Match bag QoS: RELIABLE, VOLATILE
        qos_sub = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        qos_pub = qos_sub

        # State: last camera infos
        self.last_depth_info: CameraInfo | None = None
        self.last_color_info: CameraInfo | None = None

        # CameraInfo subscriptions (not part of the sync)
        self.depth_info_sub = self.create_subscription(
            CameraInfo, depth_info_topic, self.depth_info_cb, qos_sub
        )
        self.color_info_sub = self.create_subscription(
            CameraInfo, color_info_topic, self.color_info_cb, qos_sub
        )

        # Image subscribers via message_filters
        self.depth_img_sub = Subscriber(self, Image, depth_topic, qos_profile=qos_sub)
        self.color_img_sub = Subscriber(self, Image, color_topic, qos_profile=qos_sub)

        # Approximate time sync between depth + color images
        self.sync = ApproximateTimeSynchronizer(
            [self.depth_img_sub, self.color_img_sub],
            queue_size=self.sync_queue_size,
            slop=self.sync_slop,
        )
        self.sync.registerCallback(self.sync_cb)

        # Publisher
        self.points_pub = self.create_publisher(PointCloud2, points_topic, qos_pub)

        self._warned_missing_info = False

    def depth_info_cb(self, msg: CameraInfo):
        self.last_depth_info = msg

    def color_info_cb(self, msg: CameraInfo):
        self.last_color_info = msg

    def sync_cb(self, depth_img: Image, color_img: Image):
        """
        Called with approx-synced depth + color images.
        Uses latest CameraInfo for each (not synced).
        """
        if self.last_depth_info is None or self.last_color_info is None:
            if not self._warned_missing_info:
                self.get_logger().warn(
                    "Missing CameraInfo (depth or color), skipping synced frame."
                )
                self._warned_missing_info = True
            return

        self._warned_missing_info = False

        depth_info = self.last_depth_info
        color_info = self.last_color_info

        # Depth intrinsics
        Kd = depth_info.k
        fx_d, fy_d = Kd[0], Kd[4]
        cx_d, cy_d = Kd[2], Kd[5]

        depth_width = depth_img.width
        depth_height = depth_img.height
        stride = self.stride

        # Decode depth
        if depth_img.encoding == '16UC1':
            depth_raw = np.frombuffer(depth_img.data, dtype=np.uint16).reshape(depth_height, depth_width)
            depth_m = depth_raw.astype(np.float32) * self.depth_scale
        elif depth_img.encoding in ('32FC1', '32FC'):
            depth_raw = np.frombuffer(depth_img.data, dtype=np.float32).reshape(depth_height, depth_width)
            depth_m = depth_raw * self.depth_scale
        else:
            self.get_logger().error(
                f"Unsupported depth encoding: {depth_img.encoding}. "
                f"Expected '16UC1' or '32FC1'."
            )
            return

        # Downsample
        depth_m = depth_m[0:depth_height:stride, 0:depth_width:stride]
        h_ds, w_ds = depth_m.shape

        # Pixel grid in depth image coordinates
        us, vs = np.meshgrid(
            np.arange(w_ds, dtype=np.float32) * stride,
            np.arange(h_ds, dtype=np.float32) * stride
        )

        # Valid depth mask
        valid = np.isfinite(depth_m) & (depth_m > 0)
        if not np.any(valid):
            return

        z = depth_m[valid]
        u_d = us[valid]
        v_d = vs[valid]

        # Depth threshold: exclude points further than max_depth meters
        depth_mask = (z <= self.max_depth)
        if not np.any(depth_mask):
            return
        z = z[depth_mask]
        u_d = u_d[depth_mask]
        v_d = v_d[depth_mask]

        x = (u_d - cx_d) / fx_d * z
        y = (v_d - cy_d) / fy_d * z

        points_xyz = np.stack((x, y, z), axis=-1).astype(np.float32)

        # Map to color image space by simple scaling of pixel indices
        color_width = color_img.width
        color_height = color_img.height

        # Scale factors between depth & color resolutions
        su = float(color_width) / float(depth_width)
        sv = float(color_height) / float(depth_height)

        u_c = (u_d * su).astype(np.int32)
        v_c = (v_d * sv).astype(np.int32)

        # Clamp to image bounds
        u_c = np.clip(u_c, 0, color_width - 1)
        v_c = np.clip(v_c, 0, color_height - 1)

        # Decode color image to RGB
        if color_img.encoding in ('rgb8', 'rgb16'):
            step = 3
            color_arr = np.frombuffer(color_img.data, dtype=np.uint8).reshape(color_height, color_width, step)
            rgb_arr = color_arr
        elif color_img.encoding in ('bgr8', 'bgr16'):
            step = 3
            color_arr = np.frombuffer(color_img.data, dtype=np.uint8).reshape(color_height, color_width, step)
            rgb_arr = color_arr[..., ::-1]  # BGR -> RGB
        else:
            self.get_logger().error(
                f"Unsupported color encoding: {color_img.encoding}. "
                f"Expected 'rgb8' or 'bgr8'."
            )
            return

        colors_rgb = rgb_arr[v_c, u_c, :].astype(np.uint8)

        # Build and publish cloud
        header = Header()
        header.stamp = depth_img.header.stamp  # use depth time
        header.frame_id = depth_info.header.frame_id or depth_img.header.frame_id

        pc2_msg = create_point_cloud2_xyzrgb(header, points_xyz, colors_rgb)
        self.points_pub.publish(pc2_msg)


def main(args=None):
    rclpy.init(args=args)
    node = DepthToPointCloudRGBSyncNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
