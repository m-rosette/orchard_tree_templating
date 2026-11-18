"""
Offline checks for trellis_geometry (no ROS).

For each dataset in TRELLIS_DATA_ROOT (default <ws>/data/post_slam_image_processing) and each anchor tree:
  - the anchor local (x, y) agrees with fwd_dist_to_tree_m / lat_dist_to_tree_m from its meta.yaml (1 cm),
  - the distance between two trunks agrees with their SLAM world distance (1 cm).

Run: python3 -m pytest test/test_trellis_geometry.py -v
"""

import itertools
import os
from pathlib import Path

import numpy as np
import pytest

from tree_template.trellis_geometry import (
    expected_anchor_local,
    extract_world_xy,
    find_slam_results_yaml,
    load_global_row_yaw,
    load_meta_yamls,
    local_positions,
)

TOL_M = 0.01


def _default_data_root() -> Path:
    # <ws>/src/orchard_tree_templating/tree_template/test/this_file.py -> <ws>/data/post_slam_image_processing
    return Path(__file__).resolve().parents[4] / "data" / "post_slam_image_processing"


DATA_ROOT = Path(os.environ.get("TRELLIS_DATA_ROOT", str(_default_data_root())))
DATASETS = sorted(p for p in DATA_ROOT.glob("*") if any(p.glob("tree_*/meta.yaml"))) if DATA_ROOT.is_dir() else []


# -----------------------------------------------------------------------------
#  Synthetic row
# -----------------------------------------------------------------------------

def _synthetic_row(yaw: float, n: int = 5, spacing: float = 0.9, lat: float = 1.7):
    """Trees on a straight row with heading yaw. The robot is broadside to each tree, with a forward offset."""
    c, s = np.cos(yaw), np.sin(yaw)
    fwd_dir = np.array([c, s])
    left_dir = np.array([-s, c])
    origin = np.array([6812.0, -7975.0])
    tree_data = {}
    for i in range(n):
        tree = origin + i * spacing * fwd_dir
        fwd = 0.1 * (i - 2)                         # the robot is not exactly broadside
        robot = tree - fwd * fwd_dir - lat * left_dir
        tree_data[i] = {
            "tree_id": i,
            "tree_world_xy_m": tree.tolist(),
            "broadside_robot_pose_xyt": [robot[0], robot[1], yaw],
            "fwd_dist_to_tree_m": fwd,
            "lat_dist_to_tree_m": lat,
        }
    return tree_data


@pytest.mark.parametrize("yaw", [0.0, -0.0396, 1.2, -2.9])
@pytest.mark.parametrize("anchor_id", [0, 2, 4])
def test_synthetic_anchor_at_fwd_lat(yaw, anchor_id):
    tree_data = _synthetic_row(yaw)
    ids, xy, skipped = local_positions(tree_data, anchor_id)
    assert not skipped
    a = xy[ids.index(anchor_id)]
    meta = tree_data[anchor_id]
    np.testing.assert_allclose(a, [meta["fwd_dist_to_tree_m"], meta["lat_dist_to_tree_m"]], atol=1e-9)
    # All trunks on the same row line, spaced 0.9 m along +X
    np.testing.assert_allclose(xy[:, 1], meta["lat_dist_to_tree_m"], atol=1e-9)
    np.testing.assert_allclose(np.diff(xy[:, 0]), 0.9, atol=1e-9)


def test_synthetic_global_row_yaw_rotates_about_robot():
    tree_data = _synthetic_row(0.3)
    ids, xy, _ = local_positions(tree_data, 2, global_row_yaw=0.3 + 0.05)
    np.testing.assert_allclose(xy[ids.index(2)], expected_anchor_local(tree_data[2], 0.35), atol=1e-9)


def test_missing_world_xy_is_skipped():
    tree_data = _synthetic_row(0.0)
    del tree_data[3]["tree_world_xy_m"]
    ids, xy, skipped = local_positions(tree_data, 0)
    assert skipped == [3] and 3 not in ids and len(xy) == 4


def test_unknown_anchor_raises():
    with pytest.raises(ValueError):
        local_positions(_synthetic_row(0.0), 99)


# -----------------------------------------------------------------------------
#  Saved datasets
# -----------------------------------------------------------------------------

def _check_dataset(ds: Path, use_global_yaw: bool):
    tree_data = load_meta_yamls(str(ds))
    row_yaw = load_global_row_yaw(find_slam_results_yaml(str(ds))) if use_global_yaw else None
    if use_global_yaw and row_yaw is None:
        pytest.skip(f"{ds.name}: no slam_trunks_*.yaml with row_yaw_rad")

    failures = []
    for anchor_id, meta in sorted(tree_data.items()):
        if "fwd_dist_to_tree_m" not in meta or "broadside_robot_pose_xyt" not in meta:
            continue
        ids, xy, _ = local_positions(tree_data, anchor_id, row_yaw)
        if anchor_id not in ids:
            continue
        yaw = row_yaw if row_yaw is not None else float(meta["broadside_robot_pose_xyt"][2])
        err = np.linalg.norm(xy[ids.index(anchor_id)] - expected_anchor_local(meta, yaw))
        if err > TOL_M:
            failures.append(f"anchor {anchor_id}: anchor error {err:.3f} m")

        for (i, a), (j, b) in itertools.combinations(zip(ids, xy), 2):
            d_world = np.linalg.norm(extract_world_xy(tree_data[i]) - extract_world_xy(tree_data[j]))
            if abs(np.linalg.norm(a - b) - d_world) > TOL_M:
                failures.append(f"anchor {anchor_id}: spacing {i}-{j} differs from world")
    assert not failures, f"{ds.name}:\n  " + "\n  ".join(failures)


@pytest.mark.skipif(not DATASETS, reason=f"no datasets under {DATA_ROOT}")
@pytest.mark.parametrize("ds", DATASETS, ids=[d.name for d in DATASETS])
@pytest.mark.parametrize("use_global_yaw", [False, True], ids=["broadside_yaw", "slam_row_yaw"])
def test_dataset_geometry(ds, use_global_yaw):
    _check_dataset(ds, use_global_yaw)


# -----------------------------------------------------------------------------
#  Map correction
# -----------------------------------------------------------------------------

from tree_template.trellis_geometry import (  # noqa: E402
    apply_map_correction,
    associate,
    base_to_local,
    fit_map_correction,
    frame_poses_local,
    rot2,
)


def _row(n=5, spacing=1.2, lat=1.8):
    return np.column_stack([np.arange(n) * -spacing, np.full(n, lat)])


def test_correction_translation_only_with_one_trunk():
    pred = _row()
    true = apply_map_correction(pred, 0.0, np.array([0.12, -0.08]))
    dets = true[[0, 0, 0]] + np.random.default_rng(1).normal(0, 0.01, (3, 2))
    m = associate(dets, pred, list(range(5)), 0.35)
    assert [tid for _, tid, _ in m] == [0, 0, 0]
    yaw, t, clamped = fit_map_correction(pred[[0, 0, 0]], dets, [0, 0, 0], np.radians(5))
    assert yaw == 0.0 and not clamped
    np.testing.assert_allclose(t, [0.12, -0.08], atol=0.02)


def test_correction_yaw_with_two_trunks_and_equal_weights():
    pred = _row()
    yaw_true, t_true = np.radians(2.0), np.array([0.05, 0.10])
    true = apply_map_correction(pred, yaw_true, t_true)
    # tree 0 seen 5 times, tree 1 once: the weights make both count the same
    dets = np.vstack([true[[0] * 5], true[[1]]])
    yaw, t, clamped = fit_map_correction(pred[[0] * 5 + [1]], dets, [0] * 5 + [1], np.radians(5))
    assert abs(np.degrees(yaw) - 2.0) < 1e-6 and not clamped
    np.testing.assert_allclose(apply_map_correction(pred, yaw, t), true, atol=1e-9)


def test_correction_yaw_clamp():
    pred = _row()
    true = apply_map_correction(pred, np.radians(10.0), np.zeros(2))
    yaw, t, clamped = fit_map_correction(pred[[0, 1, 2]], true[[0, 1, 2]], [0, 1, 2], np.radians(5))
    assert clamped and abs(np.degrees(yaw) - 5.0) < 1e-9


def test_associate_gate_and_side():
    pred = np.array([[0.0, 1.8], [1.2, 1.8], [0.0, -1.8]])
    dets = np.array([[0.1, 1.7], [0.6, 1.8], [0.05, -1.75], [0.0, 3.0]])
    m = associate(dets, pred, [10, 11, 12], 0.35)
    assert [(i, tid) for i, tid, _ in m] == [(0, 10), (2, 12)]


def test_frame_poses_and_base_to_local():
    # Robot at the broadside pose, row along world yaw 0.3; frame_1 is 0.05 m further along the robot heading.
    yaw = 0.3
    origin = np.array([100.0, 50.0])
    f1 = origin + 0.05 * np.array([np.cos(yaw), np.sin(yaw)])
    meta = {
        "broadside_robot_pose_xyt": [origin[0], origin[1], yaw],
        "cameras": {"base_cam": {"frames": {
            "frame_0": {"saved": True, "world_pose_xyt": [origin[0], origin[1], yaw]},
            "frame_1": {"saved": True, "world_pose_xyt": [f1[0], f1[1], yaw]},
            "frame_2": {"saved": False},
        }}},
    }
    fp = frame_poses_local(meta, "base_cam", origin, yaw)
    assert sorted(fp) == [0, 1]
    np.testing.assert_allclose(fp[1][0], [0.05, 0.0], atol=1e-12)
    # A trunk at base (0.0, 1.8) in frame_1 is at local (0.05, 1.8)
    np.testing.assert_allclose(base_to_local(np.array([0.0, 1.8]), *fp[1][:2]), [[0.05, 1.8]], atol=1e-12)
    # Robot yawed 10 deg relative to the row: base +y turns with it
    p = base_to_local(np.array([0.0, 1.0]), np.zeros(2), np.radians(10))[0]
    np.testing.assert_allclose(p, rot2(np.radians(10)) @ [0.0, 1.0])


def test_associate_one_detection_per_trunk_per_image():
    pred = np.array([[0.0, 1.8], [1.2, 1.8]])
    # Image 0: two detections near trunk 0 -> only the nearest matches. Image 1: one more near trunk 0 -> matches.
    dets = np.array([[0.05, 1.8], [-0.25, 1.85], [0.02, 1.79]])
    m = associate(dets, pred, [7, 8], 0.35, meas_frames=[0, 0, 1])
    assert [(i, tid) for i, tid, _ in m] == [(0, 7), (2, 7)]


def test_cluster_then_associate_keeps_objects_apart():
    from tree_template.trellis_geometry import associate_groups, cluster_detections
    # Two objects 0.24 m apart (the case of tree 1): A at -0.06 in frames -2,-1,1,2; B at -0.30 in all frames
    A = [(-2, -0.058), (-1, -0.054), (1, -0.075), (2, -0.053)]
    B = [(-2, -0.299), (-1, -0.299), (0, -0.298), (1, -0.309), (2, -0.296)]
    dets = np.array([[x, 1.96] for _, x in A] + [[x, 1.83] for _, x in B])
    frames = [f for f, _ in A] + [f for f, _ in B]
    groups = cluster_detections(dets, frames, 0.10)
    assert sorted(len(g) for g in groups) == [4, 5]
    m = associate_groups(dets, groups, np.array([[-0.022, 1.897], [-1.6, 1.9]]), [1, 2], 0.35)
    assert len(m) == 1 and m[0][1] == 1 and sorted(m[0][0]) == [0, 1, 2, 3]
