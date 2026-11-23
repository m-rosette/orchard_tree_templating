#!/usr/bin/env python3
"""
trellis_from_rgbd_extraction.py

Publishes the trunks of a fast_slam_rgbd_extraction folder (tree_*/meta.yaml) as one TrunkRegistry, and corrects
the map from base-camera trunk detections.

The node keeps the map in the local frame of the anchor tree (trellis_geometry.py). It publishes the map in the
robot base frame of the displayed base-camera image (world_pose_xyt of that frame in meta.yaml). The presaved point
clouds use the same frame. The displayed image is presaved_images.base_cam_frame_index of the frame_select_service
node. The node reads it from /parameter_events and publishes the map again when it changes.

Parameters
----------
extraction_dir          (str)   required: folder with tree_*/meta.yaml
anchor_tree_id          (int)   anchor tree; -1: no registry. Setting it publishes the registry.
slam_results_yaml       (str)   save_best_particle_map yaml with row_yaw_rad (row direction). Default: the
                                slam_trunks_*.yaml in extraction_dir. Without row_yaw_rad: the broadside yaw of the anchor.
trellis_side            (str)   side when meta.yaml has no "side" (default far)
z_offset                (float) trunk z (default 0.0)
registry_topic          (str)   default fastslam_registry (latched)
max_correction_m        (float) max anchor shift of a correction (default 1.0)
correction_frames       (str)   current (default): the correction uses only the displayed base-camera image.
                                buffer: the correction uses the frames in frame_offsets (see below).
frame_offsets           (int[]) base-camera frames for correction_frames buffer (default [-2, -1, 0, 1, 2])
frame_select_service    (str)   set_parameters service that selects the presaved base-camera frame
                                (default /apple_prediction/set_parameters); empty: one trigger, frame 0
frame_select_param      (str)   default presaved_images.base_cam_frame_index
frame_switch_delay_sec  (float) wait after a frame switch (default 0.3)
trigger_service         (str)   trunk detector trigger (default /trigger_estimation)
trigger_timeout_sec     (float) default 30.0
detection_collect_sec   (float) TrunkInfo collection time after the trigger (default 0.5)
measurement_topic       (str)   TrunkInfo of trunk_detection_relay (default /trunk_measurements_raw)
base_cam_name           (str)   camera folder in meta.yaml (default base_cam)
cluster_eps_m           (float) max distance between detections of one object (default 0.10)
association_gate_m      (float) max distance object -> map trunk (default 0.35)
max_yaw_correction_deg  (float) default 5.0

/correct_template_anchor (std_srvs/Trigger): local correction of the map at the parked robot pose.
    1. Get the trunk detections:
       correction_frames current: one detector trigger on the displayed image. The detections and the map are in
       the robot base frame of that image, so the fit does not use the odometry.
       correction_frames buffer: one trigger on each frame in frame_offsets. The frame poses (world_pose_xyt) put
       the detections in the local frame, so the fit uses the odometry between the frames. Frame 0 is displayed
       at the end.
    2. Group the detections per object. Match the groups to the map trunks (one to one, same side). The anchor
       must match.
    3. Fit one rigid correction (translation; yaw also with two or more trunks) and move all trunks.
    4. Publish the corrected map in the robot base frame of the displayed image.
The response message is JSON.
"""

import copy
import json
import math
import threading
import time
from typing import List

import numpy as np
from scipy.spatial.transform import Rotation
from skimage.transform import EuclideanTransform

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rcl_interfaces.msg import Parameter, ParameterEvent, ParameterType, ParameterValue, SetParametersResult
from rcl_interfaces.srv import SetParameters
from std_srvs.srv import Trigger
from tree_template_interfaces.msg import TrunkInfo, TrunkRegistry

from tree_template.trellis_geometry import (
    anchor_transform,
    associate_groups,
    cluster_detections,
    expected_anchor_local,
    find_slam_results_yaml,
    fit_map_correction,
    frame_poses_local,
    load_global_row_yaw,
    load_meta_yamls,
    local_positions,
)


class TrellisFromExtractionNode(Node):
    def __init__(self):
        super().__init__("trellis_from_rgbd_extraction")
        for name, default in [
            ("extraction_dir", ""), ("anchor_tree_id", -1), ("slam_results_yaml", ""), ("trellis_side", "far"),
            ("z_offset", 0.0), ("registry_topic", "fastslam_registry"), ("max_correction_m", 1.0),
            ("correction_frames", "current"),
            ("frame_offsets", [-2, -1, 0, 1, 2]), ("frame_select_service", "/apple_prediction/set_parameters"),
            ("frame_select_param", "presaved_images.base_cam_frame_index"), ("frame_switch_delay_sec", 0.3),
            ("trigger_service", "/trigger_estimation"), ("trigger_timeout_sec", 30.0),
            ("detection_collect_sec", 0.5), ("measurement_topic", "/trunk_measurements_raw"),
            ("base_cam_name", "base_cam"), ("cluster_eps_m", 0.10), ("association_gate_m", 0.35),
            ("max_yaw_correction_deg", 5.0),
        ]:
            self.declare_parameter(name, default)
        gp = lambda n: self.get_parameter(n).value  # noqa: E731
        self.extraction_dir = str(gp("extraction_dir"))
        self.trellis_side = str(gp("trellis_side"))
        self.z_offset = float(gp("z_offset"))
        self.max_correction_m = float(gp("max_correction_m"))
        self.correction_frames = str(gp("correction_frames")).strip().lower()
        if self.correction_frames not in ("current", "buffer"):
            raise RuntimeError(f"correction_frames must be current or buffer, got {self.correction_frames!r}")
        self.frame_offsets = [int(v) for v in gp("frame_offsets")]
        self.frame_select_param = str(gp("frame_select_param"))
        self.frame_switch_delay_sec = float(gp("frame_switch_delay_sec"))
        self.trigger_timeout_sec = float(gp("trigger_timeout_sec"))
        self.detection_collect_sec = float(gp("detection_collect_sec"))
        self.base_cam_name = str(gp("base_cam_name"))
        self.cluster_eps_m = float(gp("cluster_eps_m"))
        self.association_gate_m = float(gp("association_gate_m"))
        self.max_yaw_correction = math.radians(float(gp("max_yaw_correction_deg")))
        if not self.extraction_dir:
            raise RuntimeError("Parameter extraction_dir is not set")

        # The map and the images must come from the same SLAM run: default yaml in extraction_dir
        slam_yaml = str(gp("slam_results_yaml")) or find_slam_results_yaml(self.extraction_dir)
        self._global_row_yaw = load_global_row_yaw(slam_yaml)
        if self._global_row_yaw is None:
            self.get_logger().warn(f"No row_yaw_rad in '{slam_yaml}': the broadside yaw of the anchor gives the row "
                                   f"direction.")
        else:
            self.get_logger().info(f"row_yaw_rad {self._global_row_yaw:.4f} from {slam_yaml}")

        self._tree_data = load_meta_yamls(self.extraction_dir)
        self._anchor_id = None
        self._registry = None              # map in the local frame of the anchor (the correction changes it)
        self._registry_ids: List[int] = []  # tree id of each trunk in _registry
        self._origin_xy = None             # local frame of the anchor
        self._row_yaw = 0.0
        self._display_index = 0            # base-camera frame of the anchor that gives the published robot frame
        self._registry_lock = threading.Lock()

        # Detections that arrive during a trigger
        self._collect_lock = threading.Lock()
        self._collected = None

        # The correction handler waits for its clients and the TrunkInfo subscription: they use other groups
        client_cbg = ReentrantCallbackGroup()
        self.registry_pub = self.create_publisher(TrunkRegistry, str(gp("registry_topic")), QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(TrunkInfo, str(gp("measurement_topic")), self._trunk_cb, 10,
                                 callback_group=MutuallyExclusiveCallbackGroup())
        self.trigger_client = self.create_client(Trigger, str(gp("trigger_service")), callback_group=client_cbg)
        self.frame_select_client = (self.create_client(SetParameters, str(gp("frame_select_service")),
                                                       callback_group=client_cbg)
                                    if gp("frame_select_service") else None)
        self.create_service(Trigger, "/correct_template_anchor", self._correct_cb,
                            callback_group=MutuallyExclusiveCallbackGroup())
        # Follow the displayed base-camera frame (parameter of the frame_select_service node)
        fs = str(gp("frame_select_service"))
        self._frame_node = fs[:-len("/set_parameters")] if fs.endswith("/set_parameters") else ""
        if self._frame_node:
            self.create_subscription(ParameterEvent, "/parameter_events", self._parameter_event_cb, 10,
                                     callback_group=MutuallyExclusiveCallbackGroup())

        self.get_logger().info(f"{len(self._tree_data)} trees in {self.extraction_dir}: {sorted(self._tree_data)}")
        self.add_on_set_parameters_callback(self._on_parameters)
        if int(gp("anchor_tree_id")) >= 0:
            self.publish_registry(int(gp("anchor_tree_id")))

    # ------------------------------------------------------------------ registry

    def _on_parameters(self, params) -> SetParametersResult:
        for p in params:
            if p.name == "anchor_tree_id" and int(p.value) >= 0:
                if int(p.value) not in self._tree_data:
                    reason = f"anchor_tree_id {p.value} is not in {self.extraction_dir}: {sorted(self._tree_data)}"
                    self.get_logger().error(reason)
                    return SetParametersResult(successful=False, reason=reason)
                self.publish_registry(int(p.value))
        return SetParametersResult(successful=True)

    def publish_registry(self, anchor_id: int):
        origin_xy, row_yaw = anchor_transform(self._tree_data, anchor_id, self._global_row_yaw)
        ids, local_xy, skipped = local_positions(self._tree_data, anchor_id, self._global_row_yaw)
        if skipped:
            self.get_logger().warn(f"Trees without tree_world_xy_m: {skipped}")
        if anchor_id not in ids:
            self.get_logger().error(f"Anchor tree_{anchor_id:03d} has no world position; no registry.")
            return

        # Checks: the row is along +X, and the anchor is at its meta.yaml fwd/lat
        if len(ids) >= 3:
            slope_deg = math.degrees(math.atan(np.polyfit(local_xy[:, 0], local_xy[:, 1], 1)[0]))
            (self.get_logger().warn if abs(slope_deg) > 1.0 else self.get_logger().info)(
                f"Line through the trunks: {slope_deg:+.2f} deg from local +X")
        meta = self._tree_data[anchor_id]
        if "fwd_dist_to_tree_m" in meta and "broadside_robot_pose_xyt" in meta:
            err = float(np.linalg.norm(local_xy[ids.index(anchor_id)] - expected_anchor_local(meta, row_yaw)))
            if err > 0.01:
                self.get_logger().error(f"Geometry check: anchor {err:.3f} m from its meta.yaml fwd/lat position")

        msg = TrunkRegistry()
        stamp = self.get_clock().now().to_msg()
        for tid, xy in zip(ids, local_xy):
            trunk = TrunkInfo(stamp=stamp, width=float(self._tree_data[tid].get("width", 0.0)),
                              side=str(self._tree_data[tid].get("side") or self.trellis_side))
            trunk.pose.position.x, trunk.pose.position.y, trunk.pose.position.z = float(xy[0]), float(xy[1]), self.z_offset
            trunk.pose.orientation.w = 1.0        # the row is along local +X
            msg.trunks.append(trunk)
        with self._registry_lock:
            self._anchor_id, self._registry, self._registry_ids = anchor_id, msg, list(ids)
            self._origin_xy, self._row_yaw = origin_xy, row_yaw
        self._publish_display()
        self.get_logger().info(f"Registry: {len(ids)} trunks, anchor tree_{anchor_id:03d} at "
                               f"({local_xy[ids.index(anchor_id)][0]:+.3f}, {local_xy[ids.index(anchor_id)][1]:+.3f})")

    def _display_pose(self):
        """(transform robot base frame of the displayed image -> local frame, frame index), or (None, None)."""
        frame_pose = frame_poses_local(self._tree_data[self._anchor_id], self.base_cam_name, self._origin_xy,
                                       self._row_yaw)
        for idx in (self._display_index, 0):
            if idx in frame_pose:
                return frame_pose[idx][0], idx
        return None, None

    def _publish_display(self):
        """Publish the registry (local frame) in the robot base frame of the displayed base-camera image."""
        with self._registry_lock:
            if self._registry is None:
                return
            T, idx = self._display_pose()
            msg = copy.deepcopy(self._registry)
            if T is None:
                self.get_logger().warn(f"tree_{self._anchor_id:03d} has no {self.base_cam_name} frame poses: "
                                       f"registry in the local frame")
            else:
                to_base = T.inverse
                yaw_rot = Rotation.from_euler("z", float(to_base.rotation))
                for trunk in msg.trunks:
                    p = trunk.pose.position
                    p.x, p.y = (float(v) for v in to_base([[p.x, p.y]])[0])
                    o = trunk.pose.orientation
                    o.x, o.y, o.z, o.w = (yaw_rot * Rotation.from_quat([o.x, o.y, o.z, o.w])).as_quat().tolist()
            self.registry_pub.publish(msg)
        if T is not None:
            self.get_logger().info(f"Registry published in the robot frame of {self.base_cam_name} frame {idx:+d} "
                                   f"of tree_{self._anchor_id:03d}")

    def _parameter_event_cb(self, event: ParameterEvent):
        if event.node != self._frame_node:
            return
        for p in list(event.new_parameters) + list(event.changed_parameters):
            if p.name == self.frame_select_param and p.value.type == ParameterType.PARAMETER_INTEGER:
                idx = int(p.value.integer_value)
                if idx != self._display_index:
                    self._display_index = idx
                    self._publish_display()

    # ------------------------------------------------------------------ detections

    def _trunk_cb(self, msg: TrunkInfo):
        with self._collect_lock:
            if self._collected is not None:
                self._collected.append(msg)

    def _call(self, client, request, timeout_sec: float):
        """Service call from the handler thread. Return the response, or None on timeout."""
        if not client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error(f"{client.srv_name} is not available")
            return None
        done = threading.Event()
        future = client.call_async(request)
        future.add_done_callback(lambda _f: done.set())
        if not done.wait(timeout=timeout_sec):
            self.get_logger().error(f"{client.srv_name}: no response after {timeout_sec:.0f} s")
            return None
        return future.result()

    def _select_frame(self, offset: int) -> bool:
        req = SetParameters.Request(parameters=[Parameter(
            name=self.frame_select_param,
            value=ParameterValue(type=ParameterType.PARAMETER_INTEGER, integer_value=int(offset)))])
        resp = self._call(self.frame_select_client, req, 5.0)
        return resp is not None and all(r.successful for r in resp.results)

    def _trigger_and_collect(self) -> List[TrunkInfo]:
        """TrunkInfo detections of one detector trigger (the relay publishes them after the trigger returns)."""
        with self._collect_lock:
            self._collected = []
        resp = self._call(self.trigger_client, Trigger.Request(), self.trigger_timeout_sec)
        if resp is not None and not resp.success:
            self.get_logger().warn(f"Trunk detector trigger failed: {resp.message}")
        time.sleep(self.detection_collect_sec)
        with self._collect_lock:
            out, self._collected = self._collected, None
        return out

    # ------------------------------------------------------------------ correction

    def _fail(self, response, error: str, result=None):
        response.success = False
        response.message = json.dumps(dict(result or {}, error=error))
        self.get_logger().warn(error)
        return response

    def _correct_cb(self, request, response):
        """
        response.message (JSON):
          summary | error
          correction_frames    current | buffer
          frame                current: the displayed base-camera frame (the robot frame of the fit)
          fit_frame            robot (current: robot base frame of the displayed image) | local (buffer)
          dx, dy, dyaw         correction in fit_frame: p_new = R(dyaw) @ p + [dx, dy]
          anchor_dx, anchor_dy shift of the anchor trunk in fit_frame
          n_detections, n_matched, matched_tree_ids, frames_used, yaw_clamped
          rms_residual_m       RMS of corrected prediction - detection, matched detections
          detections           [{frame, xy, tree_id}] in fit_frame, before the correction (tree_id null: no match)
          residual_by_tree_m   {tree id: mean residual}
          anchor_residual_m    mean residual of the anchor detections (validation, target < 0.05 m)
        """
        if self._registry is None:
            return self._fail(response, "No registry. Set anchor_tree_id first.")
        result = {"correction_frames": self.correction_frames}
        with self._registry_lock:
            anchor_id, ids = self._anchor_id, list(self._registry_ids)
            local_xy = np.array([[t.pose.position.x, t.pose.position.y] for t in self._registry.trunks])
            T_disp, disp_idx = self._display_pose()
        if self.correction_frames == "current":
            # Detections and templates in the robot base frame of the displayed image: no odometry in the fit
            to_fit = T_disp.inverse if T_disp is not None else EuclideanTransform()
            pred_xy = to_fit(local_xy)
            meas = [[d.pose.position.x, d.pose.position.y] for d in self._trigger_and_collect()]
            meas_frame = [disp_idx if disp_idx is not None else 0] * len(meas)
            result.update(frame=meas_frame[0] if meas_frame else disp_idx,
                          fit_frame="robot" if T_disp is not None else "local")
        else:
            pred_xy, to_fit = local_xy, EuclideanTransform()
            result["fit_frame"] = "local"
            frame_pose = frame_poses_local(self._tree_data[anchor_id], self.base_cam_name, self._origin_xy,
                                           self._row_yaw)
            offsets = [0] if self.frame_select_client is None else [o for o in self.frame_offsets if o in frame_pose]
            if not offsets:
                return self._fail(response, f"No saved {self.base_cam_name} frames for tree_{anchor_id:03d}.", result)
            if any(frame_pose[o][1] != "world_pose" for o in offsets):
                self.get_logger().warn("Frames without world_pose_xyt: their detections use the broadside pose.")
            meas, meas_frame = [], []
            for off in offsets:
                if self.frame_select_client is not None:
                    if not self._select_frame(off):
                        self.get_logger().warn(f"Frame {off:+d} not selected; skipped.")
                        continue
                    time.sleep(self.frame_switch_delay_sec)
                for d in self._trigger_and_collect():
                    meas.append(frame_pose[off][0]([[d.pose.position.x, d.pose.position.y]])[0])
                    meas_frame.append(off)
            if self.frame_select_client is not None:
                self._select_frame(0)              # the broadside frame stays on the base-camera topic
        result["n_detections"] = len(meas)
        if not meas:
            return self._fail(response, "No trunk detections.", result)
        meas = np.array(meas, dtype=np.float64).reshape(-1, 2)

        # Group per object, then match the groups to the templates
        matches = {k: tid for g, tid, _ in associate_groups(
            meas, cluster_detections(meas, self.cluster_eps_m), pred_xy, ids, self.association_gate_m) for k in g}
        matched_ids = sorted(set(matches.values()))
        result.update(n_matched=len(matches), matched_tree_ids=matched_ids,
                      frames_used=sorted({int(meas_frame[k]) for k in matches}),
                      detections=[{"frame": int(f), "xy": [float(m[0]), float(m[1])], "tree_id": matches.get(k)}
                                  for k, (m, f) in enumerate(zip(meas, meas_frame))])
        if anchor_id not in matched_ids:
            return self._fail(response, f"No detection matches the anchor tree_{anchor_id:03d} "
                                        f"(gate {self.association_gate_m} m; matched {matched_ids}).", result)

        # Fit in the fit frame
        k_list = sorted(matches)
        T = [matches[k] for k in k_list]
        P = np.array([pred_xy[ids.index(tid)] for tid in T])
        M = meas[k_list]
        corr, clamped = fit_map_correction(P, M, T, self.max_yaw_correction)
        res = np.linalg.norm(corr(P) - M, axis=1)
        a_old = pred_xy[ids.index(anchor_id)]
        shift = corr([a_old])[0] - a_old
        result["residual_by_tree_m"] = {str(tid): float(res[np.array(T) == tid].mean()) for tid in matched_ids}
        result["anchor_residual_m"] = result["residual_by_tree_m"][str(anchor_id)]
        result.update(dx=float(corr.translation[0]), dy=float(corr.translation[1]), dyaw=float(corr.rotation),
                      anchor_dx=float(shift[0]), anchor_dy=float(shift[1]),
                      rms_residual_m=float(np.sqrt(np.mean(res ** 2))), yaw_clamped=bool(clamped))
        result["summary"] = (f"anchor shift ({shift[0]:+.3f}, {shift[1]:+.3f}) m, yaw "
                             f"{math.degrees(corr.rotation):+.2f} deg, {len(matched_ids)} trunk(s), "
                             f"RMS {result['rms_residual_m']:.3f} m, anchor residual {result['anchor_residual_m']:.3f} m"
                             + (f" ({self.base_cam_name} frame {result['frame']:+d})" if result.get("frame") is not None
                                and self.correction_frames == "current" else ""))
        if np.linalg.norm(shift) > self.max_correction_m:
            return self._fail(response, f"Anchor shift {np.linalg.norm(shift):.3f} m > max_correction_m "
                                        f"{self.max_correction_m} m; not applied.", result)

        # Apply to all trunks of the local map: local -> fit frame -> correction -> local (same yaw change)
        new_local = to_fit.inverse(corr(to_fit(local_xy)))
        yaw_rot = Rotation.from_euler("z", corr.rotation)
        with self._registry_lock:
            corrected = copy.deepcopy(self._registry)
            for trunk, xy in zip(corrected.trunks, new_local):
                trunk.pose.position.x, trunk.pose.position.y = float(xy[0]), float(xy[1])
                o = trunk.pose.orientation
                o.x, o.y, o.z, o.w = (yaw_rot * Rotation.from_quat([o.x, o.y, o.z, o.w])).as_quat().tolist()
            self._registry = corrected
        self._publish_display()
        self.get_logger().info(f"Correction: {result['summary']}")
        response.success = True
        response.message = json.dumps(result)
        return response


def main(args=None):
    rclpy.init(args=args)
    try:
        executor = MultiThreadedExecutor()
        executor.add_node(TrellisFromExtractionNode())
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
