#!/usr/bin/env python3
"""
trellis_from_rgbd_extraction.py

Reads all the per-tree meta.yaml files in extraction_dir, puts each trunk position relative to the
anchor tree (the robot is at the origin 0,0,0), and publishes all of them in one TrunkRegistry message.
generate_trellis_collision_obj then makes all the collision objects at the same time.

Usage
-----
ros2 run <package> trellis_from_rgbd_extraction --ros-args \
    -p extraction_dir:=/path/to/rgbd_at_trees \
    -p trellis_side:=near

Parameters
----------
extraction_dir          (str)   – required. Root directory with the tree_*/meta.yaml files.
anchor_tree_id          (int)   – tree to anchor on; -1 = none yet (set it at runtime, see Workflow).
trellis_side            (str)   – fallback side when meta.yaml has no 'side' field (default "far").
z_offset                (float) – vertical offset added to all trunk z-positions (default 0.0).
registry_topic          (str)   – topic name for the TrunkRegistry message (default "fastslam_registry").
shift_whole_map         (bool)  – if True, correction shifts all trunks; if False, anchor only (default True).
max_correction_m        (float) – reject corrections larger than this magnitude in metres (default 1.0).
measurement_timeout_sec (float) – seconds to wait for a trunk measurement after triggering (default 5.0).
slam_results_yaml       (str)   – path to the yaml written by save_best_particle_map.py for this run.
                                  If empty, the node uses the slam_trunks_*.yaml file in extraction_dir.
                                  If it has a row_yaw_rad value, the node uses this one row yaw for
                                  the whole run to de-rotate all tree positions, so the row is along
                                  local +X. If not, the node uses the broadside yaw of the anchor tree.
                                  The trunk orientation is always identity: the robot pose defines the
                                  planning frame, so the trellis side branches are perpendicular to
                                  the row at all trees.

trellis_geometry.py has the frame definitions and the transform.

Workflow
--------
1. Launch the node.
2. Set the anchor tree at runtime to publish the initial template:
       ros2 param set /trellis_from_rgbd_extraction anchor_tree_id 8
3. Call /correct_template_anchor (std_srvs/Trigger) to refine the anchor
   trunk position using a live depth measurement.

Directory structure expected (output of fast_slam_rgbd_extraction):
    extraction_dir/
        tree_000/meta.yaml
        tree_001/meta.yaml
        tree_008/meta.yaml
        ...

Each meta.yaml must have at least:
    tree_id: 8
    tree_world_xy_m: [x, y]                  # trunk position in SLAM world frame
    broadside_robot_pose_xyt: [x, y, yaw]    # robot pose when tree was broadside
    side: near                               # optional; if missing, the node uses trellis_side
"""

import copy
import threading
from typing import Dict, List, Optional

import numpy as np

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rcl_interfaces.msg import SetParametersResult
from std_srvs.srv import Trigger
from tree_template_interfaces.msg import TrunkInfo, TrunkRegistry

from tree_template.trellis_geometry import (
    anchor_transform,
    extract_world_z,
    find_slam_results_yaml,
    load_global_row_yaw,
    load_meta_yamls,
    local_positions,
)


# =============================================================================
#  Node
# =============================================================================

class TrellisFromExtractionNode(Node):
    def __init__(self):
        super().__init__("trellis_from_rgbd_extraction")

        # ── Callback groups ───────────────────────────────────────────────────
        # The service handler and the trigger client use the same group, so the async
        # service call in the handler does not deadlock.
        self._service_cbg = MutuallyExclusiveCallbackGroup()
        # The subscriber callbacks use a different group, so they can run while the
        # service handler thread waits on the threading.Event.
        self._sub_cbg = MutuallyExclusiveCallbackGroup()

        # ── Parameters ───────────────────────────────────────────────────────
        self.declare_parameter("extraction_dir", "")
        self.declare_parameter("anchor_tree_id", -1)
        self.declare_parameter("trellis_side", "far")
        self.declare_parameter("z_offset", 0.0)
        self.declare_parameter("registry_topic", "fastslam_registry")
        self.declare_parameter("shift_whole_map", True)
        self.declare_parameter("max_correction_m", 1.0)
        self.declare_parameter("measurement_timeout_sec", 5.0)
        self.declare_parameter("slam_results_yaml", "")  # output of save_best_particle_map.py

        self.extraction_dir = self.get_parameter("extraction_dir").value
        self.trellis_side = self.get_parameter("trellis_side").value
        self.z_offset = self.get_parameter("z_offset").value
        self.registry_topic = str(self.get_parameter("registry_topic").value)
        self.shift_whole_map = bool(self.get_parameter("shift_whole_map").value)
        self.max_correction_m = float(self.get_parameter("max_correction_m").value)
        self.measurement_timeout_sec = float(self.get_parameter("measurement_timeout_sec").value)
        self.slam_results_yaml = str(self.get_parameter("slam_results_yaml").value)

        if not self.extraction_dir:
            self.get_logger().fatal("Parameter 'extraction_dir' must be set.")
            raise RuntimeError("extraction_dir not set")

        # The SLAM map and the extracted images must come from the same run (tree ids are not stable
        # between runs), so the default is the slam_trunks_*.yaml file in extraction_dir.
        if not self.slam_results_yaml:
            self.slam_results_yaml = find_slam_results_yaml(self.extraction_dir)
        self._global_row_yaw: Optional[float] = load_global_row_yaw(self.slam_results_yaml)
        if self._global_row_yaw is not None:
            self.get_logger().info(
                f"Loaded global row_yaw_rad={self._global_row_yaw:.6f} "
                f"({np.degrees(self._global_row_yaw):.2f} deg) from {self.slam_results_yaml}"
            )
        else:
            self.get_logger().warn(
                "\n" + "!" * 78 + "\n"
                f"  No usable row_yaw_rad (slam_results_yaml='{self.slam_results_yaml}',\n"
                f"  extraction_dir='{self.extraction_dir}').\n"
                "  The node uses the broadside yaw of each anchor tree (one odom sample) for the row\n"
                "  direction. The template yaw can be wrong by several degrees.\n"
                + "!" * 78
            )

        # ── State ─────────────────────────────────────────────────────────────
        self._tree_data: Dict[int, dict] = {}
        self._last_anchor_id: Optional[int] = None
        self._last_registry_msg: Optional[TrunkRegistry] = None
        # Tree id of each trunk in _last_registry_msg (same order)
        self._last_registry_ids: List[int] = []

        # Trunk measurement state: _sub_cbg writes it, the service handler reads it
        self._latest_trunk_pose = None
        self._trunk_received_event: Optional[threading.Event] = None

        # ── Publishers ───────────────────────────────────────────────────────
        self._registry_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.registry_pub = self.create_publisher(
            TrunkRegistry, self.registry_topic, self._registry_qos
        )

        # ── Subscribers ──────────────────────────────────────────────────────
        self.trunk_meas_sub = self.create_subscription(
            TrunkInfo,
            "/trunk_measurements_raw",
            self._trunk_measurement_callback,
            10,
            callback_group=self._sub_cbg,
        )

        # ── Services ─────────────────────────────────────────────────────────
        self.correction_service = self.create_service(
            Trigger,
            "/correct_template_anchor",
            self._correct_template_anchor_callback,
            callback_group=self._service_cbg,
        )

        self._do_startup()

    # =========================================================================
    #  Startup
    # =========================================================================

    def _do_startup(self):
        try:
            self._tree_data = load_meta_yamls(self.extraction_dir)
        except FileNotFoundError as e:
            self.get_logger().fatal(str(e))
            raise RuntimeError(str(e))

        self.get_logger().info(
            f"Loaded {len(self._tree_data)} trees from {self.extraction_dir}\n"
            f"  Available tree ids: {sorted(self._tree_data.keys())}\n"
            f"  Waiting for anchor — set anchor_tree_id parameter to trigger publish."
        )

        # Register the parameter change callback after the data loads and before the node
        # spins, so it gets each anchor_tree_id set at launch or with ros2 param set.
        self.add_on_set_parameters_callback(self._on_parameter_change)

        # If the launch set anchor_tree_id (for example, in launch_config.yaml), publish now.
        anchor_id = int(self.get_parameter("anchor_tree_id").value)
        if anchor_id >= 0:
            self._try_publish(anchor_id)

    # =========================================================================
    #  Parameter change — sets anchor_tree_id at runtime
    # =========================================================================

    def _on_parameter_change(self, params) -> SetParametersResult:
        for param in params:
            if param.name == "anchor_tree_id":
                anchor_id = int(param.value)
                if anchor_id >= 0:
                    if self._tree_data and anchor_id not in self._tree_data:
                        reason = (f"anchor_tree_id {anchor_id} is not in {self.extraction_dir}. "
                                  f"Available ids: {sorted(self._tree_data.keys())}")
                        self.get_logger().error(reason)
                        return SetParametersResult(successful=False, reason=reason)
                    if not self._tree_data:
                        self.get_logger().warn(
                            "anchor_tree_id set before yaml data was loaded — ignoring."
                        )
                    else:
                        # Run _try_publish in a thread, so the parameter service callback
                        # is not blocked. join() before the return, so the registry is
                        # published before the next service call from OrchardTemplating.
                        t = threading.Thread(
                            target=self._try_publish,
                            args=(anchor_id,),
                            daemon=True,
                        )
                        t.start()
                        t.join()
        return SetParametersResult(successful=True)

    # =========================================================================
    #  Core publish
    # =========================================================================

    def _try_publish(self, anchor_id: int):
        try:
            robot_origin_xy, row_yaw = anchor_transform(
                self._tree_data, anchor_id, self._global_row_yaw
            )
            ids, local_xy, skipped = local_positions(
                self._tree_data, anchor_id, self._global_row_yaw
            )
        except ValueError as e:
            self.get_logger().error(str(e))
            return

        self.get_logger().info(
            f"Anchoring on tree_{anchor_id:03d}  "
            f"robot_origin=({robot_origin_xy[0]:.3f}, {robot_origin_xy[1]:.3f})  "
            f"row_yaw={np.degrees(row_yaw):.1f} deg "
            f"({'global slam estimate' if self._global_row_yaw is not None else 'anchor sample'})"
        )
        for tid in skipped:
            self.get_logger().warn(f"tree_{tid:03d}: skipped, meta.yaml has no tree_world_xy_m")
        if anchor_id not in ids:
            self.get_logger().error(f"Anchor tree_{anchor_id:03d} has no world position. Registry not published.")
            return

        # Row direction check: a line fit through the local trunk positions must be close to local +X.
        if len(ids) >= 3:
            slope_deg = float(np.degrees(np.arctan(np.polyfit(local_xy[:, 0], local_xy[:, 1], 1)[0])))
            log = self.get_logger().warn if abs(slope_deg) > 1.0 else self.get_logger().info
            log(f"Row line fit through {len(ids)} trunks is {slope_deg:+.2f} deg from local +X "
                f"(row_yaw source: {'slam_results_yaml' if self._global_row_yaw is not None else 'anchor broadside'}).")

        msg = self._build_trunk_registry(self._tree_data, anchor_id, ids, local_xy)

        # With TRANSIENT_LOCAL durability, a subscriber that joins late gets the last
        # message, so the node does not wait for subscribers before it publishes.
        self.registry_pub.publish(msg)
        self._last_registry_msg = msg
        self._last_registry_ids = list(ids)
        self._last_anchor_id = anchor_id

        self.get_logger().info(
            f"Published TrunkRegistry with {len(msg.trunks)} trunks."
        )

    def _build_trunk_registry(
        self,
        tree_data: Dict[int, dict],
        anchor_id: int,
        ids: List[int],
        local_xy: np.ndarray,
    ) -> TrunkRegistry:
        """Make (but do not publish) a TrunkRegistry. ids and local_xy come from local_positions()."""
        msg = TrunkRegistry()
        stamp = self.get_clock().now().to_msg()

        for tid, xy in zip(ids, local_xy):
            meta = tree_data[tid]
            side = str(meta.get("side", self.trellis_side)) or self.trellis_side

            trunk = TrunkInfo()
            trunk.stamp = stamp
            trunk.pose.position.x = float(xy[0])
            trunk.pose.position.y = float(xy[1])
            trunk.pose.position.z = float(extract_world_z(meta)) + float(self.z_offset)
            # Identity orientation: the tree positions are de-rotated, so the row is
            # along local +X. generate_trellis_collision_obj sets the side-branch yaw
            # from `side`.
            trunk.pose.orientation.w = 1.0
            trunk.width = float(meta.get("width", 0.0))
            trunk.side = side
            msg.trunks.append(trunk)

            self.get_logger().info(
                f"tree_{tid:03d} {'[ANCHOR]' if tid == anchor_id else '        '} "
                f"local=({xy[0]:+.3f}, {xy[1]:+.3f})  side={side}"
            )

        return msg

    # =========================================================================
    #  Trunk measurement subscriber
    # =========================================================================

    def _trunk_measurement_callback(self, msg: TrunkInfo):
        """Runs on _sub_cbg, so it can run while the service handler runs."""
        self._latest_trunk_pose = msg.pose
        self.get_logger().debug(
            f"Trunk measurement — x={msg.pose.position.x:.3f} m, "
            f"y={msg.pose.position.y:.3f} m, width={msg.width:.3f} m"
        )
        # Release _get_live_trunk_position if it waits
        if self._trunk_received_event is not None:
            self._trunk_received_event.set()

    # =========================================================================
    #  Correction service
    # =========================================================================

    def _correct_template_anchor_callback(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        """
        Service handler for /correct_template_anchor.

        1. Checks that a registry was published after the last anchor set.
        2. Finds the anchor trunk by its tree id.
        3. Triggers a new depth estimate and waits for the result.
        4. Computes delta = live_xy - template_xy.
        5. Checks the delta magnitude.
        6. Shifts the anchor trunk only, or the full map, and publishes again.
        """
        self.get_logger().info("Correction service called — starting pipeline.")

        # --- 1. Check a registry exists --------------------------------------
        if self._last_registry_msg is None or self._last_anchor_id is None:
            response.success = False
            response.message = (
                "No registry has been published yet. "
                "Set anchor_tree_id before calling /correct_template_anchor."
            )
            self.get_logger().warn(response.message)
            return response

        # --- 2. Identify the anchor trunk ------------------------------------
        trunks = self._last_registry_msg.trunks
        if not trunks:
            response.success = False
            response.message = "Cached TrunkRegistry contains no trunks."
            self.get_logger().warn(response.message)
            return response

        if self._last_anchor_id not in self._last_registry_ids:
            response.success = False
            response.message = (
                f"Anchor tree_{self._last_anchor_id:03d} is not in the cached TrunkRegistry."
            )
            self.get_logger().warn(response.message)
            return response
        anchor_idx = self._last_registry_ids.index(self._last_anchor_id)
        template_xy = np.array([
            trunks[anchor_idx].pose.position.x,
            trunks[anchor_idx].pose.position.y,
        ])
        self.get_logger().info(
            f"Anchor trunk (idx={anchor_idx}, tree_id={self._last_anchor_id}) "
            f"template position — x={template_xy[0]:.4f} m, y={template_xy[1]:.4f} m"
        )

        # --- 3. Get live trunk position --------------------------------------
        if self._latest_trunk_pose is None:
            response.success = False
            response.message = (
                "No trunk measurement available. Call /trigger_estimation first "
                "and ensure trunk_detection_relay is publishing to /trunk_measurements_raw."
            )
            self.get_logger().warn(response.message)
            return response

        live_xy = np.array([
            self._latest_trunk_pose.position.x,
            self._latest_trunk_pose.position.y,
        ])
        self.get_logger().info(
            f"Using last trunk measurement — x={live_xy[0]:.4f} m, y={live_xy[1]:.4f} m"
        )

        # --- 4. Compute delta ------------------------------------------------
        delta = live_xy - template_xy
        delta_norm = float(np.linalg.norm(delta))
        self.get_logger().info(
            f"Live trunk position  — x={live_xy[0]:.4f} m, y={live_xy[1]:.4f} m\n"
            f"Correction delta     — dx={delta[0]:.4f} m, dy={delta[1]:.4f} m "
            f"(|Δ|={delta_norm:.4f} m)"
        )

        # --- 5. Validate delta -----------------------------------------------
        if delta_norm > self.max_correction_m:
            response.success = False
            response.message = (
                f"Correction delta {delta_norm:.3f} m exceeds max_correction_m "
                f"({self.max_correction_m} m). Verify the sensor is seeing the "
                "correct trunk."
            )
            self.get_logger().error(response.message)
            return response

        # --- 6. Shift and re-publish -----------------------------------------
        shifted = copy.deepcopy(self._last_registry_msg)

        if self.shift_whole_map:
            for trunk in shifted.trunks:
                trunk.pose.position.x += delta[0]
                trunk.pose.position.y += delta[1]
            self.get_logger().info(
                f"Shifted entire map ({len(shifted.trunks)} trunks) "
                f"by ({delta[0]:.4f}, {delta[1]:.4f}) m."
            )
        else:
            shifted.trunks[anchor_idx].pose.position.x += delta[0]
            shifted.trunks[anchor_idx].pose.position.y += delta[1]
            self.get_logger().info(
                f"Shifted anchor trunk (idx={anchor_idx}) only "
                f"by ({delta[0]:.4f}, {delta[1]:.4f}) m."
            )

        self.registry_pub.publish(shifted)
        # Keep the cache up to date with the corrected version
        self._last_registry_msg = shifted

        self.get_logger().info("Re-published corrected TrunkRegistry.")

        response.success = True
        response.message = (
            f"Correction applied — dx={delta[0]:.4f} m, dy={delta[1]:.4f} m "
            f"({'whole map' if self.shift_whole_map else 'anchor trunk only'})"
        )
        return response

    # =========================================================================
    #  Live trunk position helper
    # =========================================================================

    def _get_live_trunk_position(self) -> Optional[np.ndarray]:
        """
        Wait for a new TrunkInfo on /trunk_measurements_raw.
        Clears the old state first, so the measurement that comes back is new.
        """
        self._latest_trunk_pose = None
        self._trunk_received_event = threading.Event()

        self.get_logger().info(
            f"Waiting up to {self.measurement_timeout_sec}s for trunk measurement..."
        )

        received = self._trunk_received_event.wait(timeout=self.measurement_timeout_sec)
        self._trunk_received_event = None

        if not received:
            self.get_logger().warn(
                f"Timed out after {self.measurement_timeout_sec}s waiting for "
                "/trunk_measurements_raw. Check trunk_detection_relay is running "
                "and the camera has a clear view of the trunk."
            )
            return None

        x = self._latest_trunk_pose.position.x
        y = self._latest_trunk_pose.position.y
        self.get_logger().info(f"Live trunk position — x={x:.4f} m, y={y:.4f} m")
        return np.array([x, y])


# =============================================================================
#  Entry point
# =============================================================================

def main(args=None):
    rclpy.init(args=args)
    try:
        node = TrellisFromExtractionNode()
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()