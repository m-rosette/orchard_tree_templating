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
shift_whole_map         (bool)  – if True, correction shifts all trunks; if False, anchor only (default False).
max_correction_m        (float) – reject corrections larger than this magnitude in metres (default 1.0).
measurement_timeout_sec (float) – seconds to wait for a trunk measurement after triggering (default 5.0).
slam_results_yaml       (str)   – path to the yaml written by save_best_particle_map.py for this run.
                                  If it has a row_yaw_rad value, the node uses this one row yaw for
                                  the whole run to de-rotate all tree positions, so the row is along
                                  local +X. If not, the node uses the broadside yaw of the anchor tree.
                                  The trunk orientation is always identity: the robot pose defines the
                                  planning frame, so the trellis side branches are perpendicular to
                                  the row at all trees.

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
import glob
import os
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import yaml

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rcl_interfaces.msg import SetParametersResult
from std_srvs.srv import Trigger
from tree_template_interfaces.msg import TrunkInfo, TrunkRegistry


# =============================================================================
#  Pure helper functions
# =============================================================================

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

        if "tree_id" in meta:
            tid = int(meta["tree_id"])
        else:
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


def _load_global_row_yaw(slam_results_yaml: str) -> Optional[float]:
    """
    Return row_yaw_rad from the yaml that save_best_particle_map.py writes.
    Return None if the path is empty or missing, or the field is missing or null.
    Then the caller uses the broadside yaw of the anchor tree.
    """
    if not slam_results_yaml:
        return None
    p = Path(slam_results_yaml).expanduser()
    if not p.exists():
        return None
    try:
        with open(p, "r") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return None
    val = data.get("row_yaw_rad")
    if val is None:
        return None
    try:
        v = float(val)
        return v if np.isfinite(v) else None
    except Exception:
        return None


def _anchor_transform(
    tree_data: Dict[int, dict],
    anchor_id: int,
    global_row_yaw: Optional[float] = None,
) -> Tuple[np.ndarray, float, float]:
    """
    Compute the 2-D rigid transform that re-centers the row so that:

      - The robot sits at (0, 0).
      - The anchor tree is at its original observed lateral offset
        (lat_dist_to_tree_m) on the Y axis.
      - All other trees are positioned relative to the anchor in the
        same rotated frame.

    row_yaw is the angle used to de-rotate ALL tree positions into the
    local frame, making the row lie along local +X. When global_row_yaw
    is provided (the locked, run-wide estimate saved by
    save_best_particle_map.py) it is used instead of the anchor's own
    single broadside sample, since it's averaged over the whole run
    rather than one odom reading.

    Trunk orientation downstream is identity — the planning frame is
    defined by the robot's current pose, so "row direction" is local +X
    by construction and the trellis side branches are perpendicular to
    it everywhere along the row.

    Returns
    -------
    robot_origin_xy : (2,) world-frame robot position when anchor was broadside
    row_yaw         : yaw (rad) used to de-rotate tree positions into the local frame
    anchor_fwd      : along-row distance of the anchor tree from its broadside robot pose
    """
    if anchor_id not in tree_data:
        raise ValueError(
            f"Anchor tree id {anchor_id} not found. "
            f"Available ids: {sorted(tree_data.keys())}"
        )

    anchor_meta = tree_data[anchor_id]
    xyt = anchor_meta.get("broadside_robot_pose_xyt", [0.0, 0.0, 0.0])
    robot_origin_xy = np.array([float(xyt[0]), float(xyt[1])], dtype=np.float64)
    anchor_broadside_yaw = float(xyt[2]) if len(xyt) >= 3 else 0.0

    row_yaw = float(global_row_yaw) if global_row_yaw is not None else anchor_broadside_yaw
    anchor_fwd = float(anchor_meta.get("fwd_dist_to_tree_m", 0.0))

    return robot_origin_xy, row_yaw, anchor_fwd


def _transform_to_local(
    world_xy: np.ndarray,
    robot_origin_xy: np.ndarray,
    row_yaw: float,
    tree_fwd: float,
    anchor_fwd: float,
) -> np.ndarray:
    """Compute local (x, y) for a trunk in the anchor robot frame."""
    c, s = np.cos(-row_yaw), np.sin(-row_yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)

    trunk_rot = rot @ world_xy
    robot_rot = rot @ robot_origin_xy

    along_row_world = trunk_rot[0] - robot_rot[0]
    x = anchor_fwd + along_row_world
    y = trunk_rot[1] - robot_rot[1]

    return np.array([x, y], dtype=np.float64)


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
        self.declare_parameter("shift_whole_map", False)
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

        self._global_row_yaw: Optional[float] = _load_global_row_yaw(self.slam_results_yaml)
        if self.slam_results_yaml:
            if self._global_row_yaw is not None:
                self.get_logger().info(
                    f"Loaded global row_yaw_rad={self._global_row_yaw:.6f} "
                    f"({np.degrees(self._global_row_yaw):.2f} deg) from {self.slam_results_yaml}"
                )
            else:
                self.get_logger().warn(
                    f"slam_results_yaml set ({self.slam_results_yaml}) but row_yaw_rad "
                    "missing/null/unreadable — falling back to per-anchor broadside yaw."
                )

        if not self.extraction_dir:
            self.get_logger().fatal("Parameter 'extraction_dir' must be set.")
            raise RuntimeError("extraction_dir not set")

        # ── State ─────────────────────────────────────────────────────────────
        self._tree_data: Dict[int, dict] = {}
        self._last_anchor_id: Optional[int] = None
        self._last_registry_msg: Optional[TrunkRegistry] = None

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
            self._tree_data = _load_meta_yamls(self.extraction_dir)
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
            robot_origin_xy, row_yaw, anchor_fwd = _anchor_transform(
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

        msg = self._build_trunk_registry(
            self._tree_data, anchor_id, robot_origin_xy, row_yaw, anchor_fwd
        )
        if msg is None:
            return

        # With TRANSIENT_LOCAL durability, a subscriber that joins late gets the last
        # message, so the node does not wait for subscribers before it publishes.
        self.registry_pub.publish(msg)
        self._last_registry_msg = msg
        self._last_anchor_id = anchor_id

        self.get_logger().info(
            f"Published TrunkRegistry with {len(msg.trunks)} trunks."
        )

    def _build_trunk_registry(
        self,
        tree_data: Dict[int, dict],
        anchor_id: int,
        robot_origin_xy: np.ndarray,
        row_yaw: float,
        anchor_fwd: float,
    ) -> Optional[TrunkRegistry]:
        """Build (but do not publish) a TrunkRegistry from tree_data."""
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
            local_xy = _transform_to_local(
                world_xy, robot_origin_xy, row_yaw, tree_fwd, anchor_fwd
            )
            side = str(meta.get("side", self.trellis_side)) or self.trellis_side

            trunk = TrunkInfo()
            trunk.stamp = self.get_clock().now().to_msg()
            trunk.pose.position.x = float(local_xy[0])
            trunk.pose.position.y = float(local_xy[1])
            trunk.pose.position.z = float(world_z) + float(self.z_offset)
            # Identity orientation: the tree positions are de-rotated, so the row is
            # along local +X. generate_trellis_collision_obj sets the side-branch yaw
            # from `side`.
            trunk.pose.orientation.w = 1.0
            trunk.width = float(meta.get("width", 0.0))
            trunk.side = side
            msg.trunks.append(trunk)

            self.get_logger().info(
                f"tree_{tid:03d} {'[ANCHOR]' if tid == anchor_id else '        '} "
                f"local=({local_xy[0]:+.3f}, {local_xy[1]:+.3f})  side={side}"
            )

        if skipped:
            self.get_logger().warn(f"Skipped {skipped} trees due to missing data.")

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
        2. Finds the anchor trunk (x closest to 0 in the robot frame).
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

        anchor_idx = min(
            range(len(trunks)),
            key=lambda i: abs(trunks[i].pose.position.x)
        )
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