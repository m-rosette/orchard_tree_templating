"""
Checks for depth_registration and the extraction pose helpers (no ROS, no bag).

Run: python3 -m pytest test/test_depth_registration.py -v
"""

import numpy as np
import pytest

from tree_template.depth_registration import register_depth_to_color, rotation_from_rotvec

K_D = np.array([[427.69, 0, 422.21], [0, 427.69, 244.42], [0, 0, 1.0]])    # 848x480 depth
K_C = np.array([[614.32, 0, 322.94], [0, 614.59, 239.10], [0, 0, 1.0]])    # 640x480 color


def test_same_camera_matches_librealsense_footprint():
    # librealsense align rounds the two pixel corners (u -/+ 0.5) and fills the box inclusive, so with the same
    # camera each depth pixel fills itself and its +u / +v neighbours. The nearest value wins:
    # out[v, u] = min(depth[v-1:v+1, u-1:u+1]).
    rng = np.random.default_rng(0)
    depth = rng.integers(500, 3000, size=(48, 64)).astype(np.uint16)
    K = np.array([[50.0, 0, 32], [0, 50.0, 24], [0, 0, 1]])
    # Principal point +0.25 px: the corners land at u - 0.25 and u + 0.75, away from the .5 rounding ties
    K_c = K + np.array([[0, 0, 0.25], [0, 0, 0.25], [0, 0, 0]])
    out = register_depth_to_color(depth, K, K_c, np.eye(3), np.zeros(3), depth.shape)
    pad = np.pad(depth, ((1, 0), (1, 0)), constant_values=np.iinfo(np.uint16).max)
    expected = np.minimum.reduce([pad[1:, 1:], pad[:-1, 1:], pad[1:, :-1], pad[:-1, :-1]])
    np.testing.assert_array_equal(out, expected)


def test_fronto_parallel_plane_keeps_depth():
    depth = np.full((480, 848), 1500, dtype=np.uint16)
    out = register_depth_to_color(depth, K_D, K_C, np.eye(3), np.array([15.0, 0, 0]), (480, 640))
    valid = out > 0
    assert valid.mean() > 0.98        # the color FOV (55 deg) is inside the depth FOV (89 deg)
    assert np.all(out[valid] == 1500)


def test_baseline_shifts_a_box_by_fx_t_over_z():
    # A 1.0 m box in front of a 2.0 m wall. The color camera is t_x = 15 mm from the depth origin, so a
    # point at depth z moves by fx_c * t_x / z pixels. The box edge moves 9.2 px, the wall 4.6 px.
    z_box, z_wall, t_x = 1000.0, 2000.0, 15.0
    depth = np.full((480, 848), z_wall, dtype=np.float64)
    u = np.arange(848)
    box_cols = (u >= 422) & (u < 522)                 # depth pixels 422..521 (x >= 0 in the depth frame)
    depth[:, box_cols] = z_box
    out = register_depth_to_color(depth.astype(np.uint16), K_D, K_C, np.eye(3), np.array([t_x, 0, 0]), (480, 640))
    row = out[240].astype(float)
    left_edge = int(np.argmax(row == z_box))
    # Without the baseline, depth pixel 421.5 projects to color u = 322.94 + 614.32 * (421.5 - 422.21) / 427.69
    u_no_shift = 322.94 + 614.32 * (421.5 - 422.21) / 427.69
    expected = u_no_shift + 614.32 * t_x / z_box
    assert abs(left_edge - expected) <= 1.0


def test_nearest_surface_wins():
    depth = np.full((480, 848), 3000, dtype=np.uint16)
    depth[200:280, 400:440] = 800
    out = register_depth_to_color(depth, K_D, K_C, np.eye(3), np.array([15.0, 0, 0]), (480, 640))
    assert (out == 800).sum() > 0
    # The baseline makes an occlusion shadow (0) next to the near region, as in the driver output.
    # No far value inside the near region
    rows, cols = np.nonzero(out == 800)
    inner = out[rows.min() + 1:rows.max(), cols.min() + 1:cols.max()]
    assert np.all(inner == 800)


def test_rotvec():
    R = rotation_from_rotvec([0, 0, np.pi / 2])
    np.testing.assert_allclose(R @ [1, 0, 0], [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(rotation_from_rotvec([0, 0, 0]), np.eye(3))


rclpy = pytest.importorskip("rclpy")       # fast_slam_rgbd_extraction imports the ROS serialization


def test_last_pose_per_stamp_and_interp():
    from tree_template.fast_slam_rgbd_extraction import interp_robot_pose, last_pose_per_stamp
    path = np.array([
        [0.0, 0.0, 0.0, 10.0],
        [1.0, 0.0, 0.0, 11.0],
        [5.0, 0.0, 0.0, 11.0],      # best particle changed at t=11: this pose wins
        [6.0, 0.0, 3.1, 12.0],
        [7.0, 0.0, -3.1, 13.0],     # yaw crosses +-pi
    ])
    clean = last_pose_per_stamp(path)
    assert clean[:, 3].tolist() == [10.0, 11.0, 12.0, 13.0]
    assert clean[1, 0] == 5.0
    pose, ok = interp_robot_pose(clean, 11.5)
    assert ok and pose[0] == pytest.approx(5.5)
    pose, ok = interp_robot_pose(clean, 12.5)
    assert abs(abs(pose[2]) - np.pi) < 0.05          # through pi, not through 0
    _, ok = interp_robot_pose(clean, 9.0)
    assert not ok
