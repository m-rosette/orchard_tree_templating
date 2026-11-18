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
max_correction_m        (float) – reject a correction that moves the anchor trunk more than this (default 1.0 m).
slam_results_yaml       (str)   – path to the yaml written by save_best_particle_map.py for this run.
                                  If empty, the node uses the slam_trunks_*.yaml file in extraction_dir.
                                  If it has a row_yaw_rad value, the node uses this one row yaw for
                                  the whole run to de-rotate all tree positions, so the row is along
                                  local +X. If not, the node uses the broadside yaw of the anchor tree.
                                  The trunk orientation is always identity: the robot pose defines the
                                  planning frame, so the trellis side branches are perpendicular to
                                  the row at all trees.

Correction parameters (/correct_template_anchor)
------------------------------------------------
frame_offsets           (int[]) – base-camera buffer frames to use (default [-2, -1, 0, 1, 2]).
frame_select_service    (str)   – set_parameters service of the node that publishes the presaved base-camera
                                  frame (default /apple_prediction/set_parameters). Empty: no frame switch,
                                  one trigger, and the detections are taken in the frame_0 robot frame.
frame_select_param      (str)   – its frame index parameter (default presaved_images.base_cam_frame_index).
frame_switch_delay_sec  (float) – wait after a frame switch, before the trigger (default 0.3).
trigger_service         (str)   – trunk detector trigger (default /trigger_estimation).
trigger_timeout_sec     (float) – (default 30.0).
detection_collect_sec   (float) – time to collect TrunkInfo after the trigger returns (default 0.5).
measurement_topic       (str)   – TrunkInfo from trunk_detection_relay (default /trunk_measurements_raw).
base_cam_name           (str)   – camera folder name in meta.yaml (default base_cam).
cluster_eps_m           (float) – detections of one object in different frames are grouped within this (default 0.10).
association_gate_m      (float) – max distance group centroid <-> predicted trunk (default 0.35).
max_yaw_correction_deg  (float) – yaw clamp of the correction (default 5.0).

trellis_geometry.py has the frame definitions, the transform and the correction fit.

Workflow
--------
1. Launch the node.
2. Set the anchor tree at runtime to publish the initial template:
       ros2 param set /trellis_from_rgbd_extraction anchor_tree_id 8
3. Call /correct_template_anchor (std_srvs/Trigger). For each frame in frame_offsets the
   node selects the presaved base-camera frame, triggers the trunk detector and collects the TrunkInfo
   detections (robot base frame of that image). With the frame pose from meta.yaml (world_pose_xyt) each
   detection goes into the local frame. The detections of one object in different frames are grouped
   (cluster_eps_m); each group is matched one-to-one to the nearest map trunk on the same side
   (association_gate_m); the anchor trunk must be matched. Then one rigid correction moves the whole map:
   translation only if one trunk is matched, translation + yaw (clamped) if two or more are matched.

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
import json
import math
import threading
import time
from typing import Dict, List, Optional

import numpy as np
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue, SetParametersResult
from rcl_interfaces.srv import SetParameters
from std_srvs.srv import Trigger
from tree_template_interfaces.msg import TrunkInfo, TrunkRegistry

from tree_template.trellis_geometry import (
    anchor_transform,
    associate_groups,
    cluster_detections,
    extract_world_z,
    fit_map_correction,
    frame_poses_local,
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
        # The correction service handler waits on the responses of its own service clients and on the
        # TrunkInfo subscriber. The clients and the subscriber use other groups, so the executor can run their
        # callbacks while the handler waits.
        self._service_cbg = MutuallyExclusiveCallbackGroup()
        self._client_cbg = ReentrantCallbackGroup()
        self._sub_cbg = MutuallyExclusiveCallbackGroup()

        # ── Parameters ───────────────────────────────────────────────────────
        self.declare_parameter("extraction_dir", "")
        self.declare_parameter("anchor_tree_id", -1)
        self.declare_parameter("trellis_side", "far")
        self.declare_parameter("z_offset", 0.0)
        self.declare_parameter("registry_topic", "fastslam_registry")
        self.declare_parameter("max_correction_m", 1.0)
        self.declare_parameter("slam_results_yaml", "")  # output of save_best_particle_map.py
        # Correction
        self.declare_parameter("frame_offsets", [-2, -1, 0, 1, 2])
        self.declare_parameter("frame_select_service", "/apple_prediction/set_parameters")
        self.declare_parameter("frame_select_param", "presaved_images.base_cam_frame_index")
        self.declare_parameter("frame_switch_delay_sec", 0.3)
        self.declare_parameter("trigger_service", "/trigger_estimation")
        self.declare_parameter("trigger_timeout_sec", 30.0)
        self.declare_parameter("detection_collect_sec", 0.5)
        self.declare_parameter("measurement_topic", "/trunk_measurements_raw")
        self.declare_parameter("base_cam_name", "base_cam")
        self.declare_parameter("association_gate_m", 0.35)
        self.declare_parameter("cluster_eps_m", 0.10)
        self.declare_parameter("max_yaw_correction_deg", 5.0)

        self.extraction_dir = self.get_parameter("extraction_dir").value
        self.trellis_side = self.get_parameter("trellis_side").value
        self.z_offset = self.get_parameter("z_offset").value
        self.registry_topic = str(self.get_parameter("registry_topic").value)
        self.max_correction_m = float(self.get_parameter("max_correction_m").value)
        self.slam_results_yaml = str(self.get_parameter("slam_results_yaml").value)
        gp = self.get_parameter
        self.frame_offsets = [int(v) for v in gp("frame_offsets").value]
        self.frame_select_service = str(gp("frame_select_service").value)
        self.frame_select_param = str(gp("frame_select_param").value)
        self.frame_switch_delay_sec = float(gp("frame_switch_delay_sec").value)
        self.trigger_timeout_sec = float(gp("trigger_timeout_sec").value)
        self.detection_collect_sec = float(gp("detection_collect_sec").value)
        self.base_cam_name = str(gp("base_cam_name").value)
        self.association_gate_m = float(gp("association_gate_m").value)
        self.cluster_eps_m = float(gp("cluster_eps_m").value)
        self.max_yaw_correction = math.radians(float(gp("max_yaw_correction_deg").value))

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
        # Local frame of the last anchor (origin and row yaw in the world frame)
        self._robot_origin_xy: Optional[np.ndarray] = None
        self._row_yaw: float = 0.0

        # Detections collected during a trigger: _sub_cbg writes, the service handler reads
        self._collect_lock = threading.Lock()
        self._collecting = False
        self._collected: List[TrunkInfo] = []

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
            str(gp("measurement_topic").value),
            self._trunk_measurement_callback,
            10,
            callback_group=self._sub_cbg,
        )

        # ── Services ─────────────────────────────────────────────────────────
        self.trigger_client = self.create_client(
            Trigger, str(gp("trigger_service").value), callback_group=self._client_cbg)
        self.frame_select_client = (
            self.create_client(SetParameters, self.frame_select_service, callback_group=self._client_cbg)
            if self.frame_select_service else None
        )
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
        self._robot_origin_xy = robot_origin_xy
        self._row_yaw = row_yaw

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
        """Runs on _sub_cbg. Keeps the detections that arrive while a trigger is in progress."""
        with self._collect_lock:
            if self._collecting:
                self._collected.append(msg)

    # =========================================================================
    #  Service helpers (called from the correction handler thread)
    # =========================================================================

    def _call(self, client, request, timeout_sec: float):
        """Synchronous call from the service handler thread. Return the response, or None on timeout."""
        if not client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error(f"Service {client.srv_name} is not available.")
            return None
        done = threading.Event()
        future = client.call_async(request)
        future.add_done_callback(lambda _f: done.set())
        if not done.wait(timeout=timeout_sec):
            self.get_logger().error(f"Service {client.srv_name}: no response after {timeout_sec:.0f} s.")
            return None
        return future.result()

    def _select_frame(self, offset: int) -> bool:
        req = SetParameters.Request()
        req.parameters = [Parameter(
            name=self.frame_select_param,
            value=ParameterValue(type=ParameterType.PARAMETER_INTEGER, integer_value=int(offset)),
        )]
        resp = self._call(self.frame_select_client, req, 5.0)
        ok = resp is not None and all(r.successful for r in resp.results)
        if not ok:
            self.get_logger().warn(f"Could not select base-camera frame {offset:+d}.")
        return ok

    def _trigger_and_collect(self) -> List[TrunkInfo]:
        """Trigger the detector on the current image and return the TrunkInfo detections of that image."""
        with self._collect_lock:
            self._collected = []
            self._collecting = True
        resp = self._call(self.trigger_client, Trigger.Request(), self.trigger_timeout_sec)
        if resp is not None and not resp.success:
            self.get_logger().warn(f"Trunk detector trigger failed: {resp.message}")
        # The detector publishes before it returns; the relay publishes one TrunkInfo per trunk after that.
        time.sleep(self.detection_collect_sec)
        with self._collect_lock:
            self._collecting = False
            out = list(self._collected)
        return out

    # =========================================================================
    #  Correction service
    # =========================================================================

    def _fail(self, response, message: str, result: Optional[dict] = None):
        """Failure: message is a JSON object with "error" and the result fields known so far."""
        response.success = False
        response.message = json.dumps(dict(result or {}, error=message))
        self.get_logger().warn(message)
        return response

    def _correct_template_anchor_callback(self, request, response):
        """
        Service handler for /correct_template_anchor (std_srvs/Trigger): collect -> associate -> fit -> apply.

        response.message is a JSON object:
          summary (success) or error (failure)
          dx, dy, dyaw         correction applied to all trunks, local frame: p_new = R(dyaw) @ p + [dx, dy]
          anchor_dx, anchor_dy shift of the anchor trunk [m]
          n_detections         detections from all frames
          n_matched            detections in groups matched to a map trunk
          matched_tree_ids     distinct matched trunks
          rms_residual_m       RMS of (corrected prediction - detection) over the matched detections
          yaw_clamped
          frames_used          base-camera frames with at least one matched detection
        """
        result = {}
        if self._last_registry_msg is None or self._last_anchor_id is None or self._robot_origin_xy is None:
            return self._fail(response, "No registry yet. Set anchor_tree_id before calling /correct_template_anchor.")
        anchor_id = self._last_anchor_id
        if anchor_id not in self._last_registry_ids:
            return self._fail(response, f"Anchor tree_{anchor_id:03d} is not in the cached TrunkRegistry.")

        ids = list(self._last_registry_ids)
        pred_xy = np.array([[t.pose.position.x, t.pose.position.y] for t in self._last_registry_msg.trunks])

        # --- 1. Frame poses (local frame) -------------------------------------
        frame_pose = frame_poses_local(
            self._tree_data[anchor_id], self.base_cam_name, self._robot_origin_xy, self._row_yaw)
        if self.frame_select_client is None:
            offsets = [0]
        else:
            offsets = [o for o in self.frame_offsets if o in frame_pose]
            missing = sorted(set(self.frame_offsets) - set(offsets))
            if missing:
                self.get_logger().warn(f"Frames {missing} are not saved for tree_{anchor_id:03d}; skipped.")
        if not offsets:
            return self._fail(response, f"No saved {self.base_cam_name} frames for tree_{anchor_id:03d}.")
        if any(frame_pose[o][1] != "world_pose" for o in offsets):
            self.get_logger().warn(
                "Some frames have no world_pose_xyt in meta.yaml (extraction without --broadside_source odom). "
                "Their detections are taken at the broadside pose.")

        # --- 2. Collect -------------------------------------------------------
        meas, meas_frame = [], []
        for off in offsets:
            if self.frame_select_client is not None:
                if not self._select_frame(off):
                    continue
                time.sleep(self.frame_switch_delay_sec)
            dets = self._trigger_and_collect()
            to_local = frame_pose[off][0]
            for d in dets:
                meas.append(to_local([[d.pose.position.x, d.pose.position.y]])[0])
                meas_frame.append(off)
            self.get_logger().info(
                f"[frame {off:+d}] {len(dets)} detection(s)"
                + (": " + ", ".join(f"({m[0]:+.2f}, {m[1]:+.2f})" for m in meas[-len(dets):]) if dets else ""))
        if self.frame_select_client is not None and 0 in frame_pose:
            self._select_frame(0)          # leave the broadside frame on the base-camera topic
        result["n_detections"] = len(meas)
        if not meas:
            return self._fail(response, "No trunk detections in any frame.", result)
        meas = np.array(meas)

        # --- 3. Group over frames, then associate groups to map trunks --------
        groups = cluster_detections(meas, self.cluster_eps_m)
        group_matches = associate_groups(meas, groups, pred_xy, ids, self.association_gate_m)
        tree_of_group = {tuple(g): tid for g, tid, _ in group_matches}
        for g in groups:
            c = meas[g].mean(axis=0)
            tid = tree_of_group.get(tuple(g))
            self.get_logger().info(
                f"  object at ({c[0]:+.3f}, {c[1]:+.3f}) seen in frames {[meas_frame[k] for k in g]} -> "
                + (f"tree_{tid:03d}" if tid is not None else "no match"))
        matches = [(k, tid, 0.0) for g, tid, _ in group_matches for k in g]
        result["n_matched"] = len(matches)
        matched_ids = sorted({tid for (_, tid, _) in matches})
        result["matched_tree_ids"] = matched_ids
        result["frames_used"] = sorted({int(meas_frame[k]) for (k, _, _) in matches})
        if anchor_id not in matched_ids:
            return self._fail(
                response,
                f"The anchor tree_{anchor_id:03d} is not matched by any detection "
                f"(gate {self.association_gate_m} m; matched: {matched_ids}).", result)

        # --- 4. Fit -----------------------------------------------------------
        P = np.array([pred_xy[ids.index(tid)] for (_, tid, _) in matches])
        M = np.array([meas[k] for (k, _, _) in matches])
        T = [tid for (_, tid, _) in matches]
        corr, clamped = fit_map_correction(P, M, T, self.max_yaw_correction)
        dyaw, t = float(corr.rotation), corr.translation
        res = np.linalg.norm(corr(P) - M, axis=1)
        rms = float(np.sqrt(np.mean(res ** 2)))
        a_old = pred_xy[ids.index(anchor_id)]
        shift = corr([a_old])[0] - a_old
        for tid in matched_ids:
            r = res[[k for k, x in enumerate(T) if x == tid]]
            self.get_logger().info(f"  residual tree_{tid:03d}: mean {r.mean():.3f} m over {len(r)} detection(s)")
        self.get_logger().info(
            f"Correction: yaw {math.degrees(dyaw):+.2f} deg{' (clamped)' if clamped else ''}, "
            f"t ({t[0]:+.3f}, {t[1]:+.3f}) m, anchor shift ({shift[0]:+.3f}, {shift[1]:+.3f}) m, "
            f"RMS {rms:.3f} m, {len(matched_ids)} trunk(s), {len(matches)} detection(s)")
        result.update(dx=float(t[0]), dy=float(t[1]), dyaw=float(dyaw),
                      anchor_dx=float(shift[0]), anchor_dy=float(shift[1]),
                      rms_residual_m=rms, yaw_clamped=bool(clamped))

        if float(np.linalg.norm(shift)) > self.max_correction_m:
            return self._fail(
                response,
                f"Anchor shift {np.linalg.norm(shift):.3f} m is more than max_correction_m "
                f"({self.max_correction_m} m). Not applied.", result)

        # --- 5. Apply to all trunks and publish ------------------------------
        corrected = copy.deepcopy(self._last_registry_msg)
        yaw_rot = Rotation.from_euler("z", dyaw)
        for trunk, xy in zip(corrected.trunks, corr(pred_xy)):
            trunk.pose.position.x = float(xy[0])
            trunk.pose.position.y = float(xy[1])
            # Compose the yaw with the existing orientation (identity after a publish)
            o = trunk.pose.orientation
            o.x, o.y, o.z, o.w = (yaw_rot * Rotation.from_quat([o.x, o.y, o.z, o.w])).as_quat().tolist()
        self.registry_pub.publish(corrected)
        self._last_registry_msg = corrected
        self.get_logger().info(f"Published the corrected TrunkRegistry ({len(corrected.trunks)} trunks).")

        response.success = True
        result["summary"] = (
            f"anchor shift ({shift[0]:+.3f}, {shift[1]:+.3f}) m, yaw {math.degrees(dyaw):+.2f} deg, "
            f"{len(matched_ids)} trunk(s), RMS {rms:.3f} m")
        response.message = json.dumps(result)
        return response


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