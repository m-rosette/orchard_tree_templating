#!/usr/bin/env python3

import os
from typing import List, Dict

import rclpy
from rclpy.node import Node

from moveit_msgs.msg import CollisionObject
from geometry_msgs.msg import Pose
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Header
from std_srvs.srv import Trigger

from scipy.spatial.transform import Rotation as R
import numpy as np

from tree_template_interfaces.srv import UpdateTrellisPosition

from ament_index_python.packages import get_package_share_directory
import yaml


class TreeSceneNode(Node):
    def __init__(self):
        super().__init__('tree_scene_node')

        # Parameters for tree geometry
        self.declare_parameter("leader_branch_radii", 0.08)
        self.declare_parameter("leader_branch_len", 2.0)
        self.declare_parameter("num_side_branches", 4.0)
        self.declare_parameter("side_branch_radii", 0.04)
        self.declare_parameter("side_branch_len", 2.0)
        self.declare_parameter("trellis_angle", -18.435)  # Martin's angle
        self.declare_parameter("branch_spacing", 0.5)  # m
        self.declare_parameter('trellis_frame', 'odom')
        self.declare_parameter('trellis_prefix', 'v_trellis_tree_')

        self.leader_branch_radii = self.get_parameter("leader_branch_radii").get_parameter_value().double_value
        self.leader_branch_len = self.get_parameter("leader_branch_len").get_parameter_value().double_value
        self.num_side_branches = self.get_parameter("num_side_branches").get_parameter_value().double_value
        self.side_branch_radii = self.get_parameter("side_branch_radii").get_parameter_value().double_value
        self.side_branch_len = self.get_parameter("side_branch_len").get_parameter_value().double_value
        self.trellis_angle = np.deg2rad(self.get_parameter("trellis_angle").get_parameter_value().double_value)
        self.branch_spacing = self.get_parameter("branch_spacing").get_parameter_value().double_value
        self.trellis_frame = self.get_parameter("trellis_frame").get_parameter_value().string_value
        self.trellis_prefix = self.get_parameter("trellis_prefix").get_parameter_value().string_value

        # Locate this package's resource folder and YAML file path
        pkg_share = get_package_share_directory('tree_template')
        resource_dir = os.path.join(pkg_share, 'resource')
        os.makedirs(resource_dir, exist_ok=True)
        self.id_store_path = os.path.join(resource_dir, 'trellis_ids.yaml')

        # In-memory storage
        self.created_coords: Dict[str, Dict[str, float]] = {}

        # Load any existing IDs from YAML
        self.created_ids: List[str] = self._load_ids_from_yaml()
        self.instance_counter = self._compute_initial_counter()

        # Services
        self.update_position_service = self.create_service(
            UpdateTrellisPosition,
            'update_trellis_position',
            self.update_trellis_position_callback
        )

        self.clear_trees_service = self.create_service(
            Trigger,
            'clear_trellis_trees',
            self.clear_trellis_trees_callback
        )

        # Publisher for collision objects
        self.collision_object_publisher = self.create_publisher(
            CollisionObject, 'collision_object', 10
        )

        self.get_logger().info(
            f"Trellis template node running.\n"
            f"  YAML store: {self.id_store_path}\n"
            f"  Loaded {len(self.created_ids)} existing IDs.\n"
            f"  Next instance index: {self.instance_counter}"
        )

    # ------------ YAML helpers ------------

    def _load_ids_from_yaml(self) -> List[str]:
        """
        Load IDs (and optionally coordinates) from YAML.

        Supports:
          - legacy format: [ "v_trellis_tree_0", "v_trellis_tree_1", ... ]
          - new format:
              - id: v_trellis_tree_0
                x: ...
                y: ...
                z: ...
        """
        if not os.path.exists(self.id_store_path):
            return []

        try:
            with open(self.id_store_path, "r") as f:
                data = yaml.safe_load(f)

            self.created_coords = {}
            ids: List[str] = []

            # New preferred format: list of dicts
            if isinstance(data, list):
                for entry in data:
                    if isinstance(entry, dict) and "id" in entry:
                        obj_id = str(entry["id"])
                        ids.append(obj_id)
                        coords = {}
                        for k in ["x", "y", "z"]:
                            if k in entry:
                                coords[k] = float(entry[k])
                        if coords:
                            self.created_coords[obj_id] = coords
                    elif isinstance(entry, str):
                        # Legacy: simple list of IDs
                        ids.append(entry)
                return ids

            # Fallback: dict mapping id -> coords
            if isinstance(data, dict):
                for obj_id, coords in data.items():
                    obj_id = str(obj_id)
                    ids.append(obj_id)
                    if isinstance(coords, dict):
                        c = {}
                        for k in ["x", "y", "z"]:
                            if k in coords:
                                c[k] = float(coords[k])
                        if c:
                            self.created_coords[obj_id] = c
                return ids

            return []
        except Exception as e:
            self.get_logger().warn(f"Failed to load ID store from {self.id_store_path}: {e}")
            self.created_coords = {}
            return []

    def _save_ids_to_yaml(self):
        """
        Save IDs and their coordinates to YAML in the format:

        - id: v_trellis_tree_0
          x: ...
          y: ...
          z: ...
        """
        try:
            entries = []
            for obj_id in self.created_ids:
                entry = {"id": obj_id}
                coords = self.created_coords.get(obj_id, None)
                if coords:
                    for k in ["x", "y", "z"]:
                        if k in coords:
                            entry[k] = float(coords[k])
                entries.append(entry)

            with open(self.id_store_path, "w") as f:
                yaml.safe_dump(entries, f)
        except Exception as e:
            self.get_logger().warn(f"Failed to save ID store to {self.id_store_path}: {e}")

    def _register_id(self, obj_id: str, pose: Pose):
        """
        Store ID and its (x, y, z) position in memory and YAML.
        """
        if obj_id not in self.created_ids:
            self.created_ids.append(obj_id)

        self.created_coords[obj_id] = {
            "x": pose.position.x,
            "y": pose.position.y,
            "z": pose.position.z,
        }

        self._save_ids_to_yaml()

    def _compute_initial_counter(self) -> int:
        """
        Parse indices from IDs like 'v_trellis_tree_<n>' and start
        from max(n) + 1, so we don't reuse IDs across restarts.
        """
        max_idx = -1
        for obj_id in self.created_ids:
            if obj_id.startswith(self.trellis_prefix):
                try:
                    idx = int(obj_id.split(self.trellis_prefix)[1])
                    max_idx = max(max_idx, idx)
                except ValueError:
                    continue
        return max_idx + 1

    # ------------ Service callbacks ------------

    def update_trellis_position_callback(self, request, response):
        """
        Each call adds a new tree instance at (x, y, z).
        """
        self.add_tree_instance_at(request.pose, request.side)

        response.success = True
        self.get_logger().info(
            f"Added tree instance at: "
            f"x={request.pose.position.x:.3f}, "
            f"y={request.pose.position.y:.3f}, "
            f"z={request.pose.position.z:.3f}"
        )
        return response

    def clear_trellis_trees_callback(self, request, response):
        """
        Remove all trellis trees whose IDs are stored in the YAML file.
        Only IDs with prefix 'v_trellis_tree_' are touched.
        """
        if not self.created_ids:
            msg = "No trellis tree IDs stored. Nothing to remove."
            self.get_logger().info(msg)
            response.success = True
            response.message = msg
            return response

        removed = 0
        for obj_id in self.created_ids:
            if not obj_id.startswith(self.trellis_prefix):
                continue

            co = CollisionObject()
            co.header = Header()
            co.header.frame_id = self.trellis_frame
            co.id = obj_id
            co.operation = CollisionObject.REMOVE

            self.collision_object_publisher.publish(co)
            removed += 1
            self.get_logger().info(f"Requested removal of {obj_id}")

        # Clear local list, coords, and YAML
        self.created_ids = []
        self.created_coords = {}
        self._save_ids_to_yaml()
        self.instance_counter = 0

        response.success = True
        response.message = f"Requested removal of {removed} trellis tree objects."
        self.get_logger().info(response.message)
        return response
    
    def get_trellis_orientation(self, trellis_pose: Pose, side: str) -> np.ndarray:
        """
        Compute the trellis orientation based on side and yaw.
        Returns a quaternion [x, y, z, w].
        """
        if side == "near":
            trellis_yaw = np.pi / 2.0
        elif side == "far":
            trellis_yaw = -np.pi / 2.0
        else:
            trellis_yaw = np.pi / 2.0

        apriori_canopy_ori = R.from_euler('xyz', [0.0, self.trellis_angle, trellis_yaw]).as_quat()

        # Combine trellis orientation with apriori canopy orientation
        canopy_orientation = R.from_quat([
            apriori_canopy_ori[0],
            apriori_canopy_ori[1],
            apriori_canopy_ori[2],
            apriori_canopy_ori[3]
        ]) * R.from_quat([
            trellis_pose.orientation.x,
            trellis_pose.orientation.y,
            trellis_pose.orientation.z,
            trellis_pose.orientation.w
        ])
        return canopy_orientation.as_quat()

    # ------------ Core helper ------------

    def add_tree_instance_at(self, trellis_pose: Pose, side: str):
        """
        Publish a new CollisionObject at the provided (x, y, z) in a target frame (e.g.'odom').
        Each call gets a unique ID and that ID is persisted in YAML with coordinates.
        """
        tree_object = CollisionObject()
        tree_object.header = Header()
        tree_object.header.frame_id = self.trellis_frame

        # Unique ID per instance
        tree_object.id = self.trellis_prefix + str(self.instance_counter)
        self._register_id(tree_object.id, trellis_pose)
        self.instance_counter += 1

        # Leader branch (vertical cylinder)
        leader_branch = SolidPrimitive()
        leader_branch.type = SolidPrimitive.CYLINDER
        leader_branch.dimensions = [self.leader_branch_len, self.leader_branch_radii]

        leader_pose = Pose()
        leader_pose.position.x = 0.0
        leader_pose.position.y = 0.0
        leader_pose.position.z = self.leader_branch_len / 2.0
        leader_pose.orientation.w = 1.0

        tree_object.primitives.append(leader_branch)
        tree_object.primitive_poses.append(leader_pose)

        # Side branches (horizontal cylinders)
        for i in range(1, int(self.num_side_branches) + 1):
            side_branch = SolidPrimitive()
            side_branch.type = SolidPrimitive.CYLINDER
            side_branch.dimensions = [self.side_branch_len, self.side_branch_radii]

            branch_pose = Pose()
            branch_pose.position.x = 0.0
            branch_pose.position.y = 0.0
            branch_pose.position.z = i * self.branch_spacing

            # 90° about X
            branch_orientation = R.from_euler('xyz', [np.pi / 2, 0.0, 0.0]).as_quat()
            branch_pose.orientation.x = branch_orientation[0]
            branch_pose.orientation.y = branch_orientation[1]
            branch_pose.orientation.z = branch_orientation[2]
            branch_pose.orientation.w = branch_orientation[3]

            tree_object.primitives.append(side_branch)
            tree_object.primitive_poses.append(branch_pose)

        # Pose of the whole tree object
        tree_object.pose.position = trellis_pose.position

        canopy_orientation = self.get_trellis_orientation(trellis_pose, side)
        tree_object.pose.orientation.x = canopy_orientation[0]
        tree_object.pose.orientation.y = canopy_orientation[1]
        tree_object.pose.orientation.z = canopy_orientation[2]
        tree_object.pose.orientation.w = canopy_orientation[3]

        tree_object.operation = CollisionObject.ADD
        self.collision_object_publisher.publish(tree_object)


def main(args=None):
    rclpy.init(args=args)
    node = TreeSceneNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down TreeSceneNode...')
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
