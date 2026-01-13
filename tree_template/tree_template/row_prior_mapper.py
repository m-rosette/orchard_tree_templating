#!/usr/bin/env python3

from typing import List, Dict, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import Pose
from std_msgs.msg import Header

from tree_template_interfaces.msg import TrunkInfo, TrunkRegistry
from scipy.spatial.transform import Rotation as R


class RowPriorMapper(Node):
    """
    Maintains a row-indexed set of tree slots, each with a small state vector
    in row coordinates (s, d) plus a shared yaw (row_yaw).

    - Initializes slots from prior (num_slots, spacing, start_side, row_origin)
    - Updates slots online with an exponential moving average from TrunkInfo observations
    - Fits a global row line from accumulated measurements to update row_yaw
    - Publishes the full row map as a TrunkRegistry on 'row_prior_registry'
    """

    def __init__(self):
        # TODO: Update the line fitting to not depend directly on a row origin. Make it such that the row origin can be updated online as well.
        super().__init__("row_prior_mapper")

        # ---------- Parameters ----------
        self.declare_parameter("num_slots", 40)
        self.declare_parameter("spacing", 0.75)

        self.declare_parameter("row_origin_x", -0.5)
        self.declare_parameter("row_origin_y", 1.8)
        self.declare_parameter("row_origin_z", 0.0)

        self.declare_parameter("initial_yaw_deg", 180.0)
        self.declare_parameter("nominal_row_yaw_deg", 180.0)

        self.declare_parameter("start_side", "near")
        self.declare_parameter("target_frame", "world")

        # Unified EMA scheduling: high alpha early, low alpha once "confident"
        # Position
        self.declare_parameter("alpha_pos_high", 0.8)
        self.declare_parameter("alpha_pos_low", 0.5)
        self.declare_parameter("n_pos_confident", 10)

        # Width
        self.declare_parameter("alpha_width_high", 0.5)
        self.declare_parameter("alpha_width_low", 0.5)
        self.declare_parameter("n_width_confident", 10)

        # Yaw (global row orientation)
        self.declare_parameter("alpha_yaw_high", 0.5)
        self.declare_parameter("alpha_yaw_low", 0.5)
        self.declare_parameter("n_yaw_confident", 10)

        self.declare_parameter("slot_s_gate", 0.5)
        self.declare_parameter("observation_topic", "trunk_observations")
        self.declare_parameter("registry_topic", "row_prior_registry")

        # ---------- Fetch parameter values ----------
        self.num_slots = int(self.get_parameter("num_slots").value)
        self.spacing = float(self.get_parameter("spacing").value)
        self.row_origin = np.array(
            [
                float(self.get_parameter("row_origin_x").value),
                float(self.get_parameter("row_origin_y").value),
                float(self.get_parameter("row_origin_z").value),
            ],
            dtype=float,
        )

        self.row_yaw = np.deg2rad(float(self.get_parameter("initial_yaw_deg").value))
        self.nominal_row_yaw = np.deg2rad(
            float(self.get_parameter("nominal_row_yaw_deg").value)
        )

        self.start_side = str(self.get_parameter("start_side").value)
        self.target_frame = str(self.get_parameter("target_frame").value)

        self.alpha_pos_high = float(self.get_parameter("alpha_pos_high").value)
        self.alpha_pos_low = float(self.get_parameter("alpha_pos_low").value)
        self.n_pos_confident = int(self.get_parameter("n_pos_confident").value)

        self.alpha_width_high = float(self.get_parameter("alpha_width_high").value)
        self.alpha_width_low = float(self.get_parameter("alpha_width_low").value)
        self.n_width_confident = int(self.get_parameter("n_width_confident").value)

        self.alpha_yaw_high = float(self.get_parameter("alpha_yaw_high").value)
        self.alpha_yaw_low = float(self.get_parameter("alpha_yaw_low").value)
        self.n_yaw_confident = int(self.get_parameter("n_yaw_confident").value)

        self.slot_s_gate = float(self.get_parameter("slot_s_gate").value)
        self.observation_topic = str(self.get_parameter("observation_topic").value)
        self.registry_topic = str(self.get_parameter("registry_topic").value)

        # ---------- Slot state ----------
        # Each slot i has:
        #   side, s, d, width, seen,
        #   pos_obs_count: how many times position updated
        #   width_obs_count: how many times width updated
        self.slots: List[Dict] = []
        for i in range(self.num_slots):
            side = self._side_for_index(i)
            s_nom = i * self.spacing
            self.slots.append(
                {
                    "side": side,
                    "s": s_nom,
                    "d": 0.0,
                    "width": float("nan"),
                    "seen": False,
                    "pos_obs_count": 0,
                    "width_obs_count": 0,
                }
            )

        # ---------- Row yaw state ----------
        self.row_fit_points: List[np.ndarray] = []
        self.yaw_obs_count = 0   # how many times we have updated yaw from a fit

        # ---------- Pub/Sub ----------
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.registry_pub = self.create_publisher(
            TrunkRegistry, self.registry_topic, qos
        )

        self.obs_sub = self.create_subscription(
            TrunkInfo,
            self.observation_topic,
            self.trunk_observation_callback,
            10,
        )

        # Publish initial prior map
        self.publish_registry()

        self.get_logger().info(
            "RowPriorMapper initialized:\n"
            f"  num_slots={self.num_slots}, spacing={self.spacing:.2f} m\n"
            f"  row_origin={tuple(self.row_origin)}\n"
            f"  initial_yaw={np.rad2deg(self.row_yaw):.1f} deg\n"
            f"  start_side={self.start_side}\n"
            f"  observation_topic='{self.observation_topic}'\n"
            f"  registry_topic='{self.registry_topic}'"
        )

    # ---------- Helpers for row coordinates ----------

    def _side_for_index(self, i: int) -> str:
        """Alternating near/far pattern from a starting side."""
        if self.start_side not in ("near", "far"):
            return "near"
        if i % 2 == 0:
            return self.start_side
        return "far" if self.start_side == "near" else "near"

    def _row_axes(self):
        """
        Return (t_hat, n_hat) where:
            t_hat: unit vector along row direction
            n_hat: unit vector lateral (left of t_hat)
        """
        cz = np.cos(self.row_yaw)
        sz = np.sin(self.row_yaw)
        t_hat = np.array([cz, sz, 0.0], dtype=float)
        n_hat = np.array([-sz, cz, 0.0], dtype=float)
        return t_hat, n_hat

    def world_to_row_coords(self, pos_world: np.ndarray):
        """
        Convert world position to row coordinates (s, d) given current yaw and origin.
        """
        t_hat, n_hat = self._row_axes()
        delta = pos_world - self.row_origin
        s = float(np.dot(delta, t_hat))  # along row
        d = float(np.dot(delta, n_hat))  # lateral
        return s, d

    def row_to_world_coords(self, s: float, d: float) -> np.ndarray:
        """
        Convert row coordinates (s, d) to world position given current yaw and origin.
        """
        t_hat, n_hat = self._row_axes()
        return self.row_origin + s * t_hat + d * n_hat

    def _angle_ema(self, angle_old: float, angle_new: float, alpha: float) -> float:
        """
        EMA for angles, handling wrap-around.
        """
        # Compute smallest difference
        diff = np.arctan2(
            np.sin(angle_new - angle_old),
            np.cos(angle_new - angle_old),
        )
        return angle_old + alpha * diff

    # ---------- Observation callback ----------

    def trunk_observation_callback(self, msg: TrunkInfo):
        p = msg.pose.position
        pos_world = np.array([p.x, p.y, p.z], dtype=float)

        # Add to yaw fit buffer
        self.row_fit_points.append(pos_world)
        if len(self.row_fit_points) > 200:
            self.row_fit_points.pop(0)

        # Convert to row coords
        s_meas, d_meas = self.world_to_row_coords(pos_world)

        side = msg.side if msg.side else ("near" if d_meas < 0.0 else "far")

        slot_idx = self._associate_to_slot(s_meas, side)
        if slot_idx is None:
            self._update_row_yaw_from_fit()
            self.publish_registry()
            return

        slot = self.slots[slot_idx]

        # ---------- Scheduled alpha for position ----------
        pos_count = slot["pos_obs_count"]
        if pos_count < self.n_pos_confident:
            alpha_pos = self.alpha_pos_high
        else:
            alpha_pos = self.alpha_pos_low

        s_old = slot["s"]
        d_old = slot["d"]
        slot["s"] = (1.0 - alpha_pos) * s_old + alpha_pos * s_meas
        slot["d"] = (1.0 - alpha_pos) * d_old + alpha_pos * d_meas
        slot["pos_obs_count"] = pos_count + 1

        # ---------- Scheduled alpha for width ----------
        if not np.isnan(msg.width):
            width_count = slot["width_obs_count"]
            if width_count < self.n_width_confident:
                alpha_w = self.alpha_width_high
            else:
                alpha_w = self.alpha_width_low

            if np.isnan(slot["width"]):
                slot["width"] = float(msg.width)
            else:
                slot["width"] = (
                    (1.0 - alpha_w) * slot["width"]
                    + alpha_w * float(msg.width)
                )
            slot["width_obs_count"] = width_count + 1

        slot["seen"] = True

        # ---------- Yaw update ----------
        self._update_row_yaw_from_fit()

        # Publish updated registry
        self.publish_registry()

    def _associate_to_slot(self, s_meas: float, side: str) -> Optional[int]:
        """
        Choose the closest slot on the right side in 's' (along-row) with a gate.
        """
        best_idx = None
        best_dist = None
        for i, slot in enumerate(self.slots):
            if slot["side"] != side:
                continue
            s_nom = i * self.spacing
            dist = abs(s_meas - s_nom)
            if dist > self.slot_s_gate:
                continue
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_idx = i
        return best_idx

    def _update_row_yaw_from_fit(self):
        """
        Fit a line y = m x + b to the accumulated measurement points
        and update row_yaw with a scheduled EMA, constrained to stay
        near the nominal row direction.
        """
        if len(self.row_fit_points) < 2:
            return

        pts = np.array(self.row_fit_points, dtype=float)
        x = pts[:, 0]
        y = pts[:, 1]

        A = np.vstack([x, np.ones_like(x)]).T
        try:
            m, b = np.linalg.lstsq(A, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            return

        yaw_meas = np.arctan2(m, 1.0)

        # Constrain near nominal direction (treat yaw and yaw+pi as same line)
        diff_nominal = np.arctan2(
            np.sin(yaw_meas - self.nominal_row_yaw),
            np.cos(yaw_meas - self.nominal_row_yaw),
        )
        if diff_nominal > np.pi / 2.0:
            yaw_meas -= np.pi
        elif diff_nominal < -np.pi / 2.0:
            yaw_meas += np.pi

        # Scheduled alpha for yaw
        if self.yaw_obs_count < self.n_yaw_confident:
            alpha_yaw = self.alpha_yaw_high
        else:
            alpha_yaw = self.alpha_yaw_low

        old_yaw = self.row_yaw
        self.row_yaw = self._angle_ema(old_yaw, yaw_meas, alpha_yaw)
        self.yaw_obs_count += 1

    # ---------- Registry publishing ----------

    def publish_registry(self):
        """
        Publish a TrunkRegistry snapshot of *all* slots in row-index order.
        """
        reg = TrunkRegistry()
        reg.trunks = []

        quat = R.from_euler("z", self.row_yaw).as_quat()

        for i, slot in enumerate(self.slots):
            pose = Pose()
            pos_world = self.row_to_world_coords(slot["s"], slot["d"])
            pose.position.x = float(pos_world[0])
            pose.position.y = float(pos_world[1])
            pose.position.z = float(pos_world[2])

            pose.orientation.x = float(quat[0])
            pose.orientation.y = float(quat[1])
            pose.orientation.z = float(quat[2])
            pose.orientation.w = float(quat[3])

            tinfo = TrunkInfo()
            tinfo.pose = pose
            tinfo.side = slot["side"]
            tinfo.width = float(slot["width"]) if not np.isnan(slot["width"]) else float(
                "nan"
            )

            reg.trunks.append(tinfo)

        self.registry_pub.publish(reg)
        self.get_logger().debug(
            f"Published row_prior_registry with {len(reg.trunks)} slots; "
            f"yaw={np.rad2deg(self.row_yaw):.2f} deg"
        )


def main(args=None):
    rclpy.init(args=args)
    node = RowPriorMapper()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down RowPriorMapper...")
    finally:
        # Clean up the node
        node.destroy_node()

        # Only try to shut down if context is still active
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except rclpy.exceptions.RCLError:
                # Context was already shut down somewhere else; ignore
                pass


if __name__ == "__main__":
    main()
