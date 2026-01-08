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

# ---- Optional median filter ----
try:
    from scipy.ndimage import median_filter as _median_filter
    _HAVE_SCIPY_MEDIAN = True
except Exception:
    _HAVE_SCIPY_MEDIAN = False


# ---------- PointCloud2 builder ----------
_PC_DTYPE = np.dtype([
    ('x', np.float32),
    ('y', np.float32),
    ('z', np.float32),
    ('rgb', np.float32),
])


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

    cloud = np.empty(n_points, dtype=_PC_DTYPE)
    cloud['x'] = points_xyz[:, 0]
    cloud['y'] = points_xyz[:, 1]
    cloud['z'] = points_xyz[:, 2]

    r = colors_rgb[:, 0].astype(np.uint32, copy=False)
    g = colors_rgb[:, 1].astype(np.uint32, copy=False)
    b = colors_rgb[:, 2].astype(np.uint32, copy=False)
    rgb_uint32 = (r << 16) | (g << 8) | b
    cloud['rgb'] = rgb_uint32.view(np.float32)

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
    msg.point_step = 16
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = True
    msg.data = cloud.tobytes()
    return msg


class DepthToPointCloudRGBSyncNode(Node):
    def __init__(self):
        super().__init__('depth_to_pointcloud_rgb_sync')

        # Topics
        self.declare_parameter('depth_topic', '/camera/base_camera/depth/image_rect_raw')
        self.declare_parameter('depth_camera_info_topic', '/camera/base_camera/depth/camera_info')
        self.declare_parameter('color_topic', '/camera/base_camera/color/image_raw')
        self.declare_parameter('color_camera_info_topic', '/camera/base_camera/color/camera_info')
        self.declare_parameter('points_topic', '/camera/base_camera/depth/points_rgb_sync')

        # Depth scaling: 0.001 for 16UC1 (mm->m), 1.0 if already meters
        self.declare_parameter('depth_scale', 0.001)

        # Image-space decimation
        self.declare_parameter('stride', 1)

        # Time sync
        self.declare_parameter('sync_slop', 0.045)
        self.declare_parameter('sync_queue_size', 10)

        # Range gate (cheap early filter)
        self.declare_parameter('min_depth', 0.15)
        self.declare_parameter('max_depth', 3.0)

        # Hard cap (bounds runtime)
        self.declare_parameter('max_points', 80000)  # <=0 disables

        # 2D speckle cleanup
        self.declare_parameter('depth_median_enable', False)
        self.declare_parameter('depth_median_kernel', 3)  # odd: 3,5,...

        # 3D sparse-voxel cleanup (faster impl: sort + run-length)
        self.declare_parameter('voxel_filter_enable', True)
        self.declare_parameter('voxel_size', 0.05)          # meters
        self.declare_parameter('min_points_per_voxel', 5)   # keep voxels with >= this many points

        # Read params
        self.depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.depth_info_topic = self.get_parameter('depth_camera_info_topic').get_parameter_value().string_value
        self.color_topic = self.get_parameter('color_topic').get_parameter_value().string_value
        self.color_info_topic = self.get_parameter('color_camera_info_topic').get_parameter_value().string_value
        self.points_topic = self.get_parameter('points_topic').get_parameter_value().string_value

        self.depth_scale = float(self.get_parameter('depth_scale').get_parameter_value().double_value)
        self.stride = int(self.get_parameter('stride').get_parameter_value().integer_value)
        self.sync_slop = float(self.get_parameter('sync_slop').get_parameter_value().double_value)
        self.sync_queue_size = int(self.get_parameter('sync_queue_size').get_parameter_value().integer_value)

        self.min_depth = float(self.get_parameter('min_depth').get_parameter_value().double_value)
        self.max_depth = float(self.get_parameter('max_depth').get_parameter_value().double_value)

        self.max_points = int(self.get_parameter('max_points').get_parameter_value().integer_value)

        self.depth_median_enable = bool(self.get_parameter('depth_median_enable').get_parameter_value().bool_value)
        self.depth_median_kernel = int(self.get_parameter('depth_median_kernel').get_parameter_value().integer_value)

        self.voxel_filter_enable = bool(self.get_parameter('voxel_filter_enable').get_parameter_value().bool_value)
        self.voxel_size = float(self.get_parameter('voxel_size').get_parameter_value().double_value)
        self.min_points_per_voxel = int(self.get_parameter('min_points_per_voxel').get_parameter_value().integer_value)

        # Sanity
        if self.stride < 1:
            self.get_logger().warn("stride < 1, clamping to 1")
            self.stride = 1

        if self.min_depth < 0.0:
            self.min_depth = 0.0
        if self.max_depth <= 0.0:
            self.get_logger().warn("max_depth <= 0.0, clamping to 0.1")
            self.max_depth = 0.1
        if self.min_depth >= self.max_depth:
            self.get_logger().warn("min_depth >= max_depth, resetting min_depth to 0.0")
            self.min_depth = 0.0

        if self.depth_median_enable and not _HAVE_SCIPY_MEDIAN:
            self.get_logger().warn("depth_median_enable=True but SciPy median_filter not available; disabling.")
            self.depth_median_enable = False

        if self.depth_median_kernel < 1:
            self.depth_median_kernel = 1
        if self.depth_median_kernel % 2 == 0:
            self.get_logger().warn("depth_median_kernel must be odd; incrementing by 1.")
            self.depth_median_kernel += 1

        if self.voxel_size <= 0.0:
            self.get_logger().warn("voxel_size <= 0; disabling voxel filter.")
            self.voxel_filter_enable = False

        if self.min_points_per_voxel < 1:
            self.min_points_per_voxel = 1

        self.get_logger().info(
            "DepthToPointCloudRGBSyncNode:\n"
            f"  depth_topic:             {self.depth_topic}\n"
            f"  depth_camera_info_topic: {self.depth_info_topic}\n"
            f"  color_topic:             {self.color_topic}\n"
            f"  color_camera_info_topic: {self.color_info_topic}\n"
            f"  points_topic:            {self.points_topic}\n"
            f"  depth_scale:             {self.depth_scale}\n"
            f"  stride:                  {self.stride}\n"
            f"  sync_slop:               {self.sync_slop}\n"
            f"  sync_queue_size:         {self.sync_queue_size}\n"
            f"  min_depth:               {self.min_depth}\n"
            f"  max_depth:               {self.max_depth}\n"
            f"  max_points:              {self.max_points} (<=0 disables)\n"
            f"  depth_median_enable:     {self.depth_median_enable}\n"
            f"  depth_median_kernel:     {self.depth_median_kernel}\n"
            f"  voxel_filter_enable:     {self.voxel_filter_enable}\n"
            f"  voxel_size:              {self.voxel_size}\n"
            f"  min_points_per_voxel:    {self.min_points_per_voxel}\n"
        )

        qos_sub = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        qos_pub = qos_sub

        self.last_depth_info: CameraInfo | None = None
        self.last_color_info: CameraInfo | None = None

        self.depth_info_sub = self.create_subscription(
            CameraInfo, self.depth_info_topic, self.depth_info_cb, qos_sub
        )
        self.color_info_sub = self.create_subscription(
            CameraInfo, self.color_info_topic, self.color_info_cb, qos_sub
        )

        self.depth_img_sub = Subscriber(self, Image, self.depth_topic, qos_profile=qos_sub)
        self.color_img_sub = Subscriber(self, Image, self.color_topic, qos_profile=qos_sub)

        self.sync = ApproximateTimeSynchronizer(
            [self.depth_img_sub, self.color_img_sub],
            queue_size=self.sync_queue_size,
            slop=self.sync_slop,
        )
        self.sync.registerCallback(self.sync_cb)

        self.points_pub = self.create_publisher(PointCloud2, self.points_topic, qos_pub)
        self._warned_missing_info = False

    def depth_info_cb(self, msg: CameraInfo):
        self.last_depth_info = msg

    def color_info_cb(self, msg: CameraInfo):
        self.last_color_info = msg

    def _cap_points_deterministic(self, rr: np.ndarray, cc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Deterministic cap to bound runtime (no RNG)."""
        if self.max_points <= 0:
            return rr, cc
        n = rr.size
        if n <= self.max_points:
            return rr, cc
        idx = np.linspace(0, n - 1, self.max_points).astype(np.int64)
        return rr[idx], cc[idx]

    def _voxel_occupancy_filter_fast(self, points_xyz: np.ndarray, colors_rgb: np.ndarray):
        """
        Fast voxel occupancy filter:
          - voxelize
          - lexsort voxels
          - run-length encode counts of consecutive equal voxels
          - keep points in voxels with >= min_points_per_voxel

        Usually much faster than np.unique(axis=0, return_inverse=True, ...)
        """
        if (not self.voxel_filter_enable) or points_xyz.shape[0] == 0:
            return points_xyz, colors_rgb

        vs = float(self.voxel_size)
        min_n = int(self.min_points_per_voxel)
        if vs <= 0.0 or min_n <= 1:
            return points_xyz, colors_rgb

        vox = np.floor(points_xyz / vs).astype(np.int32, copy=False)

        # lexsort by (x, y, z)
        order = np.lexsort((vox[:, 2], vox[:, 1], vox[:, 0]))
        vox_s = vox[order]

        # boundaries between voxel runs
        diff = np.any(vox_s[1:] != vox_s[:-1], axis=1)
        run_start_flags = np.concatenate(([True], diff))
        run_starts = np.flatnonzero(run_start_flags)
        run_ends = np.concatenate((run_starts[1:], [vox_s.shape[0]]))
        run_lengths = run_ends - run_starts

        # per-point run lengths in sorted order
        run_len_per_point = np.repeat(run_lengths, run_lengths)
        keep_sorted = run_len_per_point >= min_n

        if not np.any(keep_sorted):
            return points_xyz[:0], colors_rgb[:0]

        keep = np.zeros(points_xyz.shape[0], dtype=bool)
        keep[order] = keep_sorted

        return points_xyz[keep], colors_rgb[keep]

    def sync_cb(self, depth_img: Image, color_img: Image):
        if self.last_depth_info is None or self.last_color_info is None:
            if not self._warned_missing_info:
                self.get_logger().warn("Missing CameraInfo (depth or color), skipping synced frame.")
                self._warned_missing_info = True
            return
        self._warned_missing_info = False

        depth_info = self.last_depth_info

        # Depth intrinsics
        Kd = depth_info.k
        fx_d, fy_d = float(Kd[0]), float(Kd[4])
        cx_d, cy_d = float(Kd[2]), float(Kd[5])

        depth_width = int(depth_img.width)
        depth_height = int(depth_img.height)
        stride = int(self.stride)

        # Decode depth -> meters
        if depth_img.encoding == '16UC1':
            depth_raw = np.frombuffer(depth_img.data, dtype=np.uint16).reshape(depth_height, depth_width)
            depth_m = depth_raw.astype(np.float32, copy=False) * self.depth_scale
        elif depth_img.encoding in ('32FC1', '32FC'):
            depth_raw = np.frombuffer(depth_img.data, dtype=np.float32).reshape(depth_height, depth_width)
            depth_m = depth_raw * self.depth_scale
        else:
            self.get_logger().error(
                f"Unsupported depth encoding: {depth_img.encoding}. Expected '16UC1' or '32FC1'."
            )
            return

        # Downsample in image space
        depth_ds = depth_m[0:depth_height:stride, 0:depth_width:stride]

        # Optional 2D median speckle removal
        if self.depth_median_enable and self.depth_median_kernel > 1:
            depth_ds = _median_filter(depth_ds, size=self.depth_median_kernel)

        # Range + validity gate (cheap)
        valid = (
            np.isfinite(depth_ds)
            & (depth_ds >= self.min_depth)
            & (depth_ds <= self.max_depth)
        )
        if not np.any(valid):
            return

        # Pixel indices of valid points (in downsampled image)
        rr, cc = np.nonzero(valid)

        # Cap early to bound costs downstream
        rr, cc = self._cap_points_deterministic(rr, cc)
        if rr.size == 0:
            return

        # Gather depths
        z = depth_ds[rr, cc].astype(np.float32, copy=False)

        # Full-res depth pixel coordinates
        u_d = (cc * stride).astype(np.float32, copy=False)
        v_d = (rr * stride).astype(np.float32, copy=False)

        # Reproject (depth optical frame)
        x = (u_d - cx_d) / fx_d * z
        y = (v_d - cy_d) / fy_d * z

        points_xyz = np.empty((z.shape[0], 3), dtype=np.float32)
        points_xyz[:, 0] = x
        points_xyz[:, 1] = y
        points_xyz[:, 2] = z

        # ----- Color mapping (simple resolution scaling) -----
        color_width = int(color_img.width)
        color_height = int(color_img.height)

        su = float(color_width) / float(depth_width)
        sv = float(color_height) / float(depth_height)

        u_c = np.clip((u_d * su).astype(np.int32), 0, color_width - 1)
        v_c = np.clip((v_d * sv).astype(np.int32), 0, color_height - 1)

        if color_img.encoding == 'rgb8':
            color_arr = np.frombuffer(color_img.data, dtype=np.uint8).reshape(color_height, color_width, 3)
        elif color_img.encoding == 'bgr8':
            color_arr = np.frombuffer(color_img.data, dtype=np.uint8).reshape(color_height, color_width, 3)
            color_arr = color_arr[..., ::-1]
        else:
            self.get_logger().error(
                f"Unsupported color encoding: {color_img.encoding}. Expected 'rgb8' or 'bgr8'."
            )
            return

        # Fast flat indexing
        flat = color_arr.reshape(-1, 3)
        flat_idx = v_c * color_width + u_c
        colors_rgb = flat[flat_idx]

        # ----- 3D noise removal (faster voxel occupancy filter) -----
        points_xyz, colors_rgb = self._voxel_occupancy_filter_fast(points_xyz, colors_rgb)
        if points_xyz.shape[0] == 0:
            return

        # Publish
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
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
