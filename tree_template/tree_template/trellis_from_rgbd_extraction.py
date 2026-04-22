#!/usr/bin/env python3
"""
trellis_from_rgbd_extraction.py

Reads ALL per-tree meta.yaml files found under extraction_dir at once,
anchors every trunk position relative to a chosen tree (placing the robot
at the origin 0,0,0), then publishes the complete set as a single
TrunkRegistry message so the downstream generate_trellis_collision_obj
node can spawn all collision objects in one shot.

Usage
-----
ros2 run <package> trellis_from_rgbd_extraction --ros-args \
    -p extraction_dir:=/path/to/rgbd_at_trees \
    -p trellis_side:=near

Parameters
----------
extraction_dir   (str)  – required. Root directory containing tree_*/meta.yaml files.
trellis_side     (str)  – fallback side when meta.yaml has no 'side' field (default "far").
z_offset         (float)– vertical offset added to all trunk z-positions (default 0.0).
registry_topic   (str)  – topic name for the TrunkRegistry message (default "fastslam_registry").

Workflow
--------
1. Launch the node — it loads all meta.yaml files immediately and logs available tree IDs.
2. Set the anchor tree at runtime to trigger publishing:

       ros2 param set /trellis_from_rgbd_extraction anchor_tree_id 8

   The node publishes the full TrunkRegistry as soon as a valid anchor ID is received.
   You can re-anchor at any time by setting anchor_tree_id again.

Directory structure expected (output of fast_slam_rgbd_extraction):
    extraction_dir/
        tree_000/meta.yaml
        tree_001/meta.yaml
        tree_008/meta.yaml
        ...

Each meta.yaml is expected to have at minimum:
    tree_id: 8
    tree_world_xy_m: [x, y]                  # trunk position in SLAM world frame
    broadside_robot_pose_xyt: [x, y, yaw]    # robot pose when tree was broadside
    side: near                               # optional, falls back to trellis_side param
"""

import os
import time
import glob
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import Pose
from rcl_interfaces.msg import SetParametersResult
from tree_template_interfaces.msg import TrunkInfo, TrunkRegistry


def _load_meta_yamls(extraction_dir: str) -> Dict[int, dict]:
    """
    Walk extraction_dir for tree_*/meta.yaml files and return a dict
    keyed by integer tree_id.
    """
    pattern = os.path.join(extraction_dir, "tree_*", "meta.yaml")
    paths = sorted(glob.glob(pattern))

    if not paths:
        raise FileNotFoundError(
            f"No tree_*/meta.yaml files found under: {extraction_dir}"
        )

    tree_data: Dict[int, dict] = {}
    for p in paths:
        with open(p, "r") as f:
            meta = yaml.safe_load(f)

        # Support tree_id stored as int or inferred from directory name
        if "tree_id" in meta:
            tid = int(meta["tree_id"])
        else:
            # Infer from directory name  tree_008 -> 8
            dir_name = Path(p).parent.name  # e.g. "tree_008"
            tid = int(dir_name.split("_")[-1])
            meta["tree_id"] = tid

        tree_data[tid] = meta

    return tree_data


def _extract_world_xy(meta: dict) -> np.ndarray:
    """Pull [x, y] world position from a meta dict (key: tree_world_xy_m)."""
    wp = meta.get("tree_world_xy_m")
    if wp is None:
        raise KeyError(f"meta.yaml for tree {meta.get('tree_id')} missing 'tree_world_xy_m'")
    return np.array([float(wp[0]), float(wp[1])], dtype=np.float64)


def _extract_world_z(meta: dict) -> float:
    # No Z stored in fast_slam_rgbd_extraction meta.yaml; always ground level.
    return 0.0


def _anchor_transform(
    tree_data: Dict[int, dict],
    anchor_id: int,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Compute the 2-D rigid transform that re-centers the row so that:

      - The robot sits at (0, 0).
      - The anchor tree is at its original observed lateral offset
        (lat_dist_to_tree_m) on the Y axis — i.e. it keeps its real
        distance from the robot, only the along-row (X) origin shifts.
      - All other trees are positioned relative to the anchor in the
        same rotated frame.

    The translation origin used is the robot's broadside position for
    the anchor tree (not the trunk itself), so the anchor trunk ends up
    at (0, lat_dist) rather than (0, 0).

    Returns
    -------
    robot_origin_xy : (2,) world-frame robot position when anchor was broadside
    row_yaw         : yaw (rad) of the row direction at that moment
    anchor_fwd      : along-row distance of the anchor tree from its broadside robot pose
    """
    if anchor_id not in tree_data:
        raise ValueError(
            f"Anchor tree id {anchor_id} not found. "
            f"Available ids: {sorted(tree_data.keys())}"
        )

    anchor_meta = tree_data[anchor_id]

    # Robot position & yaw when anchor tree was broadside
    xyt = anchor_meta.get("broadside_robot_pose_xyt", [0.0, 0.0, 0.0])
    robot_origin_xy = np.array([float(xyt[0]), float(xyt[1])], dtype=np.float64)
    row_yaw = float(xyt[2]) if len(xyt) >= 3 else 0.0

    # Anchor's own observed along-row distance — used as the X origin
    anchor_fwd = float(anchor_meta.get("fwd_dist_to_tree_m", 0.0))

    return robot_origin_xy, row_yaw, anchor_fwd


def _transform_to_local(
    world_xy: np.ndarray,
    robot_origin_xy: np.ndarray,
    row_yaw: float,
    tree_fwd: float,
    anchor_fwd: float,
) -> np.ndarray:
    """
    Compute local (x, y) for a trunk in the anchor robot frame.
    """
    c, s = np.cos(-row_yaw), np.sin(-row_yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)

    # Rotate both trunk and robot into row-aligned frame
    trunk_rot = rot @ world_xy
    robot_rot = rot @ robot_origin_xy

    # Along-row: world spacing from robot origin, re-centered on anchor_fwd
    along_row_world = trunk_rot[0] - robot_rot[0]
    x = anchor_fwd + along_row_world

    # Lateral: subtract robot's lateral world offset so robot is at Y=0
    y = trunk_rot[1] - robot_rot[1]

    return np.array([x, y], dtype=np.float64)


class TrellisFromExtractionNode(Node):
    def __init__(self):
        super().__init__("trellis_from_rgbd_extraction")

        # ── Parameters ───────────────────────────────────────────────────────
        self.declare_parameter("extraction_dir", "")
        self.declare_parameter("anchor_tree_id", -1)   # set at runtime to trigger publish
        self.declare_parameter("trellis_side", "far")
        self.declare_parameter("z_offset", 0.0)
        self.declare_parameter("registry_topic", "fastslam_registry")

        self.extraction_dir = self.get_parameter("extraction_dir").value
        self.trellis_side = self.get_parameter("trellis_side").value
        self.z_offset = self.get_parameter("z_offset").value
        self.registry_topic = str(self.get_parameter("registry_topic").value)

        if not self.extraction_dir:
            self.get_logger().fatal("Parameter 'extraction_dir' must be set.")
            raise RuntimeError("extraction_dir not set")

        registry_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.registry_pub = self.create_publisher(
            TrunkRegistry, self.registry_topic, registry_qos
        )

        # Loaded eagerly in _startup_once; empty until then
        self._tree_data: Dict[int, dict] = {}

        # ── Kick off once the node is spinning ────────────────────────────────
        self.create_timer(0.1, self._startup_once)
        self._started = False

    # ── Startup: load yamls, log available IDs, register param callback ───────

    def _startup_once(self):
        if self._started:
            return
        self._started = True

        try:
            self._tree_data = _load_meta_yamls(self.extraction_dir)
        except FileNotFoundError as e:
            self.get_logger().fatal(str(e))
            return

        self.get_logger().info(
            f"Loaded {len(self._tree_data)} trees from {self.extraction_dir}\n"
            f"  Available tree ids: {sorted(self._tree_data.keys())}\n"
            f"  Waiting for anchor — run:\n"
            f"    ros2 param set /{self.get_name()} anchor_tree_id <id>"
        )

        # Watch for anchor_tree_id being set (or changed) at runtime
        self.add_on_set_parameters_callback(self._on_parameter_change)

        # If anchor_tree_id was already provided at launch, publish right away
        anchor_id = int(self.get_parameter("anchor_tree_id").value)
        if anchor_id >= 0:
            self._try_publish(anchor_id)

    # ── Parameter change callback — fires on `ros2 param set` ────────────────

    def _on_parameter_change(self, params) -> SetParametersResult:
        for param in params:
            if param.name == "anchor_tree_id":
                anchor_id = int(param.value)
                if anchor_id >= 0:
                    if not self._tree_data:
                        self.get_logger().warn(
                            "anchor_tree_id set before yaml data was loaded — ignoring."
                        )
                    else:
                        # Use a one-shot zero-delay timer so we return from this
                        # callback immediately — blocking here deadlocks the
                        # parameter service.
                        # Store timer handle so the callback can cancel itself
                        self._pending_anchor_timer = None
                        def _fire(aid=anchor_id):
                            self._try_publish(aid)
                            if self._pending_anchor_timer is not None:
                                self._pending_anchor_timer.cancel()
                        self._pending_anchor_timer = self.create_timer(0.0, _fire)
        return SetParametersResult(successful=True)

    # ── Core publish ──────────────────────────────────────────────────────────

    def _try_publish(self, anchor_id: int):
        try:
            robot_origin_xy, row_yaw, anchor_fwd = _anchor_transform(
                self._tree_data, anchor_id
            )
        except ValueError as e:
            self.get_logger().error(str(e))
            return

        self.get_logger().info(
            f"Anchoring on tree_{anchor_id:03d}  "
            f"robot_origin=({robot_origin_xy[0]:.3f}, {robot_origin_xy[1]:.3f})  "
            f"row_yaw={np.degrees(row_yaw):.1f} deg"
        )

        self._publish_trunk_registry(self._tree_data, anchor_id, robot_origin_xy, row_yaw, anchor_fwd)

    def _publish_trunk_registry(
        self,
        tree_data: Dict[int, dict],
        anchor_id: int,
        robot_origin_xy: np.ndarray,
        row_yaw: float,
        anchor_fwd: float,
    ):
        msg = TrunkRegistry()
        skipped = 0

        for tid in sorted(tree_data.keys()):
            meta = tree_data[tid]

            try:
                world_xy = _extract_world_xy(meta)
                world_z = _extract_world_z(meta)
            except KeyError as e:
                self.get_logger().warn(f"tree_{tid:03d}: skipping — {e}")
                skipped += 1
                continue

            tree_fwd = float(meta.get("fwd_dist_to_tree_m", 0.0))
            local_xy = _transform_to_local(world_xy, robot_origin_xy, row_yaw, tree_fwd, anchor_fwd)
            side = str(meta.get("side", self.trellis_side)) or self.trellis_side

            trunk = TrunkInfo()
            trunk.stamp = self.get_clock().now().to_msg()
            trunk.pose.position.x = float(local_xy[0])
            trunk.pose.position.y = float(local_xy[1])
            trunk.pose.position.z = float(world_z) + float(self.z_offset)
            trunk.pose.orientation.x = 0.0
            trunk.pose.orientation.y = 0.0
            trunk.pose.orientation.z = 0.0
            trunk.pose.orientation.w = 1.0
            trunk.width = float(meta.get("width", 0.0))
            trunk.side = side
            msg.trunks.append(trunk)

            self.get_logger().info(
                f"tree_{tid:03d} {'[ANCHOR]' if tid == anchor_id else '        '} "
                f"local=({local_xy[0]:+.3f}, {local_xy[1]:+.3f})  side={side}"
            )

        if not self._wait_for_registry_subscriber(timeout_sec=10.0):
            self.get_logger().fatal(
                f"No subscribers to '{self.registry_topic}' after 10s. "
                "Is generate_trellis_collision_obj running with matching registry_topic?"
            )
            return

        self.registry_pub.publish(msg)
        self.get_logger().info(
            f"Published TrunkRegistry with {len(msg.trunks)} trunks, skipped {skipped}."
        )

    def _wait_for_registry_subscriber(self, timeout_sec: float = 10.0) -> bool:
        deadline = time.monotonic() + float(timeout_sec)
        while time.monotonic() < deadline:
            if self.registry_pub.get_subscription_count() > 0:
                return True
            time.sleep(0.1)
        return False


def main(args=None):
    rclpy.init(args=args)
    try:
        node = TrellisFromExtractionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()