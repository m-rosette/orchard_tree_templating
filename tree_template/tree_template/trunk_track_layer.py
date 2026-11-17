#!/usr/bin/env python3
"""
Short-term trunk track layer used by RowFastSLAM data association.

Detections are placed in the odometry frame (locally accurate over a few seconds) and linked into
per-tree tracks. Each track is bound to one landmark key:
  - a template slot index j (the normal case), or
  - an "extra" landmark key (>= EXTRA_KEY_BASE) when a second, distinct tree lands in a slot that
    another tree already owns (closely spaced trees that would otherwise be merged).

Width is a secondary cue: it can break a link inside the position gate, and it can confirm that two
close tracks are different trees.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

EXTRA_KEY_BASE = 100000


def is_extra_key(key: int) -> bool:
    return int(key) >= EXTRA_KEY_BASE


@dataclass
class TrunkTrack:
    track_id: int
    xy_sum: np.ndarray
    n: int
    first_t: float
    last_t: float
    widths: List[float] = field(default_factory=list)
    landmark_key: Optional[int] = None

    @property
    def xy(self) -> np.ndarray:
        return self.xy_sum / float(max(self.n, 1))

    @property
    def width(self) -> float:
        return float(np.median(self.widths)) if self.widths else float("nan")

    def add(self, t: float, xy: np.ndarray, width: Optional[float], max_widths: int) -> None:
        self.xy_sum = self.xy_sum + xy
        self.n += 1
        self.last_t = t
        if width is not None and np.isfinite(width):
            self.widths.append(float(width))
            if len(self.widths) > max_widths:
                self.widths.pop(0)


class TrunkTrackManager:
    def __init__(
        self,
        link_gate_m: float = 0.20,
        max_gap_sec: float = 1.0,
        width_break_m: float = 0.015,
        width_min_samples: int = 3,
        min_tree_sep_m: float = 0.30,
        width_sep_m: float = 0.15,
        width_distinct_m: float = 0.010,
        split_min_hits: int = 3,
        max_widths: int = 30,
    ):
        self.link_gate_m = float(link_gate_m)
        self.max_gap_sec = float(max_gap_sec)
        self.width_break_m = float(width_break_m)
        self.width_min_samples = int(width_min_samples)
        self.min_tree_sep_m = float(min_tree_sep_m)
        self.width_sep_m = float(width_sep_m)
        self.width_distinct_m = float(width_distinct_m)
        self.split_min_hits = int(split_min_hits)
        self.max_widths = int(max_widths)

        self.tracks: Dict[int, TrunkTrack] = {}
        self._active: List[int] = []
        self._next_id = 0
        self.slot_owner: Dict[int, int] = {}      # template slot -> owning track id
        self.extra_base_slot: Dict[int, int] = {}  # extra key -> slot it was split from
        self._next_extra_key = EXTRA_KEY_BASE
        self.anchor_track_id: Optional[int] = None

    # ---------------- linking ----------------

    def update(self, t: float, xy: np.ndarray, width: Optional[float]) -> TrunkTrack:
        """Link one detection (odom-frame position) to a track, creating one if needed."""
        xy = np.asarray(xy, dtype=float).reshape(2)
        self._active = [k for k in self._active if (t - self.tracks[k].last_t) <= self.max_gap_sec]

        best_k, best_d = None, self.link_gate_m
        for k in self._active:
            tr = self.tracks[k]
            d = float(np.linalg.norm(xy - tr.xy))
            if d > best_d:
                continue
            if self._width_conflict(tr, width):
                continue
            best_k, best_d = k, d

        if best_k is None:
            tr = TrunkTrack(track_id=self._next_id, xy_sum=np.zeros(2), n=0, first_t=t, last_t=t)
            self.tracks[tr.track_id] = tr
            self._active.append(tr.track_id)
            self._next_id += 1
        else:
            tr = self.tracks[best_k]
        tr.add(t, xy, width, self.max_widths)
        return tr

    def _width_conflict(self, tr: TrunkTrack, width: Optional[float]) -> bool:
        if width is None or not np.isfinite(width) or len(tr.widths) < self.width_min_samples:
            return False
        return abs(float(width) - tr.width) > self.width_break_m

    # ---------------- landmark binding ----------------

    def same_tree(self, a: TrunkTrack, b: TrunkTrack) -> bool:
        sep = float(np.linalg.norm(a.xy - b.xy))
        if sep >= self.min_tree_sep_m:
            return False
        if sep >= self.width_sep_m and self._widths_distinct(a, b):
            return False
        return True

    def _widths_distinct(self, a: TrunkTrack, b: TrunkTrack) -> bool:
        if len(a.widths) < self.width_min_samples or len(b.widths) < self.width_min_samples:
            return False
        return abs(a.width - b.width) >= self.width_distinct_m

    def bind(self, tr: TrunkTrack, slot_j: int) -> Optional[int]:
        """
        Return the landmark key for this track, deciding it on first call:
          1. a fragment of an already-bound tree inherits that tree's key,
          2. otherwise the track owns slot_j if the slot is free,
          3. otherwise it is a second tree in an occupied slot -> new extra key, but only once the
             track has split_min_hits detections. Until then None is returned (caller holds the
             measurement back) so a single stray detection cannot create a landmark.
        """
        if tr.landmark_key is not None:
            return tr.landmark_key

        nearest = None
        nearest_d = float("inf")
        for other in self.tracks.values():
            if other.track_id == tr.track_id or other.landmark_key is None:
                continue
            d = float(np.linalg.norm(other.xy - tr.xy))
            if d < nearest_d:
                nearest, nearest_d = other, d
        if nearest is not None and self.same_tree(tr, nearest):
            tr.landmark_key = nearest.landmark_key
            return tr.landmark_key

        slot_j = int(slot_j)
        if slot_j not in self.slot_owner:
            self.slot_owner[slot_j] = tr.track_id
            tr.landmark_key = slot_j
            return slot_j

        if tr.n < self.split_min_hits:
            return None

        key = self._next_extra_key
        self._next_extra_key += 1
        self.extra_base_slot[key] = slot_j
        tr.landmark_key = key
        return key

    def claim(self, tr: TrunkTrack, slot_j: int) -> None:
        """Force-bind a track to a template slot (used for the tree the template origin was anchored on)."""
        slot_j = int(slot_j)
        self.slot_owner[slot_j] = tr.track_id
        tr.landmark_key = slot_j
        self.anchor_track_id = tr.track_id

    def is_anchor_track(self, tr: TrunkTrack) -> bool:
        return self.anchor_track_id is not None and self.anchor_track_id == tr.track_id

    def slot_owned_by_other(self, slot_j: int, tr: TrunkTrack) -> bool:
        owner = self.slot_owner.get(int(slot_j))
        if owner is None or owner == tr.track_id:
            return False
        return self.tracks[owner].landmark_key != tr.landmark_key

    def describe_split(self, tr: TrunkTrack) -> str:
        """One-line summary of an extra-landmark split decision (for logging)."""
        slot_j = self.extra_base_slot.get(tr.landmark_key)
        owner = self.tracks.get(self.slot_owner.get(slot_j, -1))
        if owner is None:
            return f"track {tr.track_id} n={tr.n}"
        return (f"track {tr.track_id} (n={tr.n}, w={tr.width*100:.1f}cm) vs slot owner track {owner.track_id} "
                f"(n={owner.n}, w={owner.width*100:.1f}cm): sep {float(np.linalg.norm(tr.xy - owner.xy)):.2f} m")

    def base_slot(self, key: int) -> int:
        return self.extra_base_slot.get(int(key), int(key))
