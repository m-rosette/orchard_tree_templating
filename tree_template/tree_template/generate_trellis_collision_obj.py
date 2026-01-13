#!/usr/bin/env python3

import os
from typing import List, Dict

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from moveit_msgs.msg import CollisionObject
from geometry_msgs.msg import Pose
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Header
from std_srvs.srv import Trigger

from scipy.spatial.transform import Rotation as R
import numpy as np

from tree_template_interfaces.srv import UpdateTrellisPosition
from tree_template_interfaces.msg import TrunkRegistry


class TreeSceneNode(Node):
    def __init__(self):
        super().__init__('tree_scene_node')

        # Parameters for tree geometry
        self.declare_parameter("leader_branch_radii", 0.08)
        self.declare_parameter("leader_branch_len", 2.0)
        self.declare_parameter("num_side_branches", 4.0)
        self.declare_parameter("side_branch_radii", 0.04)
        self.declare_parameter("side_branch_len", 2.0)
        self.declare_parameter("trellis_angle", -18.435)  # Martin's angle (deg)
        self.declare_parameter("branch_spacing", 0.5)      # m
        self.declare_parameter("trellis_frame", "odom_slam")
        self.declare_parameter("trellis_prefix", "v_trellis_tree_")

        # Topic providing the row-level map of trunks.
        # For FastSLAM, use 'fastslam_registry'.
        # For the old RowPriorMapper pipeline, use 'row_prior_registry'.
        self.declare_parameter("registry_topic", "fastslam_registry")

        # Resolve parameters
        self.leader_branch_radii = self.get_parameter(
            "leader_branch_radii"
        ).get_parameter_value().double_value
        self.leader_branch_len = self.get_parameter(
            "leader_branch_len"
        ).get_parameter_value().double_value
        self.num_side_branches = self.get_parameter(
            "num_side_branches"
        ).get_parameter_value().double_value
        self.side_branch_radii = self.get_parameter(
            "side_branch_radii"
        ).get_parameter_value().double_value
        self.side_branch_len = self.get_parameter(
            "side_branch_len"
        ).get_parameter_value().double_value
        self.trellis_angle = np.deg2rad(
            self.get_parameter("trellis_angle").get_parameter_value().double_value
        )
        self.branch_spacing = self.get_parameter(
            "branch_spacing"
        ).get_parameter_value().double_value
        self.trellis_frame = self.get_parameter(
            "trellis_frame"
        ).get_parameter_value().string_value
        self.trellis_prefix = self.get_parameter(
            "trellis_prefix"
        ).get_parameter_value().string_value
        self.registry_topic = self.get_parameter(
            "registry_topic"
        ).get_parameter_value().string_value

        # In-memory storage
        self.created_coords: Dict[str, Dict[str, float]] = {}
        self.created_ids: List[str] = []
        self.instance_counter = 0

        # Services (manual add + clear)
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

        # Subscriber to trunk registry (from RowFastSLAMNode or RowPriorMapper)
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.trunk_registry_sub = self.create_subscription(
            TrunkRegistry,
            self.registry_topic,
            self.trunk_registry_callback,
            qos,
        )

        self.get_logger().info(
            "Trellis template node running.\n"
            f"  trellis_frame={self.trellis_frame}\n"
            f"  trellis_prefix={self.trellis_prefix}\n"
            f"  registry_topic={self.registry_topic}\n"
            "  Starting with empty in-memory tree registry (no YAML persistence)."
        )

    # ------------ Trunk registry subscriber ------------

    def trunk_registry_callback(self, msg: TrunkRegistry):
        """
        Update or create trellis templates based on the row-level trunk registry.

        For slot index i:
          - Collision object ID is f\"{trellis_prefix}{i}\"
          - If that ID has never been seen, we ADD a new CollisionObject with full
            geometry (leader + side branches).
          - If that ID already exists, we MOVE the existing CollisionObject to
            the new pose (no need to resend geometry).
        """
        num_trunks = len(msg.trunks)
        self.get_logger().debug(f"Received trunk registry with {num_trunks} trunks")

        for i, trunk in enumerate(msg.trunks):
            obj_id = f"{self.trellis_prefix}{i}"

            trellis_pose: Pose = trunk.pose
            side: str = trunk.side if trunk.side else "near"

            # Compute canopy orientation from row-parallel yaw and side-specific tilt
            canopy_orientation = self.get_trellis_orientation(trellis_pose, side)

            if obj_id not in self.created_ids:
                # First time for this slot: create full geometry and ADD
                tree_object = CollisionObject()
                tree_object.header = Header()
                tree_object.header.frame_id = self.trellis_frame
                tree_object.id = obj_id

                # Leader branch primitive (main trunk)
                leader_branch = SolidPrimitive()
                leader_branch.type = SolidPrimitive.CYLINDER
                leader_branch.dimensions = [
                    self.leader_branch_len,
                    self.leader_branch_radii,
                ]

                leader_pose = Pose()
                leader_pose.position.x = 0.0
                leader_pose.position.y = 0.0
                leader_pose.position.z = self.leader_branch_len / 2.0
                leader_pose.orientation.w = 1.0

                tree_object.primitives.append(leader_branch)
                tree_object.primitive_poses.append(leader_pose)

                # Side branches (V-trellis arms)
                for j in range(1, int(self.num_side_branches) + 1):
                    side_branch = SolidPrimitive()
                    side_branch.type = SolidPrimitive.CYLINDER
                    side_branch.dimensions = [
                        self.side_branch_len,
                        self.side_branch_radii,
                    ]

                    branch_pose = Pose()
                    branch_pose.position.x = 0.0
                    branch_pose.position.y = 0.0
                    branch_pose.position.z = j * self.branch_spacing

                    branch_orientation = R.from_euler(
                        "xyz", [np.pi / 2, 0.0, 0.0]
                    ).as_quat()
                    branch_pose.orientation.x = branch_orientation[0]
                    branch_pose.orientation.y = branch_orientation[1]
                    branch_pose.orientation.z = branch_orientation[2]
                    branch_pose.orientation.w = branch_orientation[3]

                    tree_object.primitives.append(side_branch)
                    tree_object.primitive_poses.append(branch_pose)

                # Pose of the whole tree object in the target frame
                tree_object.pose.position = trellis_pose.position
                tree_object.pose.orientation.x = canopy_orientation[0]
                tree_object.pose.orientation.y = canopy_orientation[1]
                tree_object.pose.orientation.z = canopy_orientation[2]
                tree_object.pose.orientation.w = canopy_orientation[3]

                tree_object.operation = CollisionObject.ADD
                self.collision_object_publisher.publish(tree_object)

                self.created_ids.append(obj_id)
                self.created_coords[obj_id] = {
                    "x": trellis_pose.position.x,
                    "y": trellis_pose.position.y,
                    "z": trellis_pose.position.z,
                }

                self.get_logger().debug(
                    f"ADD trellis '{obj_id}' at "
                    f"({trellis_pose.position.x:.2f}, "
                    f"{trellis_pose.position.y:.2f}, "
                    f"{trellis_pose.position.z:.2f}), "
                    f"side={side}"
                )
            else:
                # Existing slot: MOVE the object to the new pose
                move_obj = CollisionObject()
                move_obj.header = Header()
                move_obj.header.frame_id = self.trellis_frame
                move_obj.id = obj_id

                move_obj.pose.position = trellis_pose.position
                move_obj.pose.orientation.x = canopy_orientation[0]
                move_obj.pose.orientation.y = canopy_orientation[1]
                move_obj.pose.orientation.z = canopy_orientation[2]
                move_obj.pose.orientation.w = canopy_orientation[3]

                move_obj.operation = CollisionObject.MOVE
                self.collision_object_publisher.publish(move_obj)

                self.created_coords[obj_id] = {
                    "x": trellis_pose.position.x,
                    "y": trellis_pose.position.y,
                    "z": trellis_pose.position.z,
                }

                self.get_logger().debug(
                    f"MOVE trellis '{obj_id}' to "
                    f"({trellis_pose.position.x:.2f}, "
                    f"{trellis_pose.position.y:.2f}, "
                    f"{trellis_pose.position.z:.2f}), "
                    f"side={side}"
                )

        # Keep instance_counter in sync with number of created IDs, in case the
        # manual service path is still used elsewhere.
        self.instance_counter = len(self.created_ids)

    # ------------ Service callbacks ------------

    def update_trellis_position_callback(self, request, response):
        """
        Each call adds a new tree instance at (x, y, z) using the manual service.
        This is separate from the registry-driven map.
        """
        self.add_tree_instance_at(request.pose, request.side)

        response.success = True
        self.get_logger().info(
            "Added tree instance at: "
            f"x={request.pose.position.x:.3f}, "
            f"y={request.pose.position.y:.3f}, "
            f"z={request.pose.position.z:.3f}"
        )
        return response

    def clear_trellis_trees_callback(self, request, response):
        """
        Remove all trellis trees whose IDs are stored in memory.
        """
        if not self.created_ids:
            msg = "No trellis trees stored in memory. Nothing to remove."
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

        # Clear local list and coords, reset counter
        self.created_ids = []
        self.created_coords = {}
        self.instance_counter = 0

        response.success = True
        response.message = f"Requested removal of {removed} trellis tree objects."
        self.get_logger().info(response.message)
        return response

    # ------------ Orientation helper ------------

    def get_trellis_orientation(self, trellis_pose: Pose, side: str) -> np.ndarray:
        """
        Compute trellis orientation without using trunk pose yaw.

        Trunks are assumed to lie along the X-axis of the planning frame.
        Any global row yaw is encoded in robot odom (odom_slam_best), not in trunk poses.
        """
        # Side-dependent yaw around Z for V-trellis arms
        if side == "near":
            trellis_yaw = -np.pi / 2.0
        elif side == "far":
            trellis_yaw = np.pi / 2.0
        else:
            trellis_yaw = np.pi / 2.0

        # Global row yaw comes in via trunk.pose.orientation
        row_R = R.from_quat([
            trellis_pose.orientation.x,
            trellis_pose.orientation.y,
            trellis_pose.orientation.z,
            trellis_pose.orientation.w,
        ])

        trellis_local_R = R.from_euler("xyz", [0.0, self.trellis_angle, trellis_yaw])

        canopy_R = row_R * trellis_local_R
        return canopy_R.as_quat()

    # ------------ Core helper for manual tree creation ------------

    def add_tree_instance_at(self, trellis_pose: Pose, side: str):
        """
        Publish a new CollisionObject at the provided (x, y, z) in a target frame.
        Each call gets a unique ID and that ID is stored in memory.
        This is primarily used by the manual /update_trellis_position service.
        """
        tree_object = CollisionObject()
        tree_object.header = Header()
        tree_object.header.frame_id = self.trellis_frame

        # Unique ID per instance
        tree_object.id = self.trellis_prefix + str(self.instance_counter)
        obj_id = tree_object.id

        # Store in memory
        self.created_coords[obj_id] = {
            "x": trellis_pose.position.x,
            "y": trellis_pose.position.y,
            "z": trellis_pose.position.z,
        }
        self.created_ids.append(obj_id)
        self.instance_counter += 1

        # Leader branch primitive
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

        # Side branches
        for j in range(1, int(self.num_side_branches) + 1):
            side_branch = SolidPrimitive()
            side_branch.type = SolidPrimitive.CYLINDER
            side_branch.dimensions = [self.side_branch_len, self.side_branch_radii]

            branch_pose = Pose()
            branch_pose.position.x = 0.0
            branch_pose.position.y = 0.0
            branch_pose.position.z = j * self.branch_spacing

            branch_orientation = R.from_euler(
                "xyz", [np.pi / 2, 0.0, 0.0]
            ).as_quat()
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
        node.get_logger().info("Shutting down TreeSceneNode...")
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
