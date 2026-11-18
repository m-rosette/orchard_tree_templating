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

from skimage.transform import EuclideanTransform  # noqa: E402

from tree_template.trellis_geometry import (  # noqa: E402
    associate_groups,
    cluster_detections,
    fit_map_correction,
    frame_poses_local,
)


def _row(n=5, spacing=1.2, lat=1.8):
    return np.column_stack([np.arange(n) * -spacing, np.full(n, lat)])


def test_correction_translation_only_with_one_trunk():
    pred = _row()
    dets = pred[[0, 0, 0]] + [0.12, -0.08] + np.random.default_rng(1).normal(0, 0.01, (3, 2))
    T, clamped = fit_map_correction(pred[[0, 0, 0]], dets, [0, 0, 0], np.radians(5))
    assert T.rotation == 0.0 and not clamped
    np.testing.assert_allclose(T.translation, [0.12, -0.08], atol=0.02)


def test_correction_yaw_with_two_trunks_and_equal_weights():
    pred = _row()
    true = EuclideanTransform(rotation=np.radians(2.0), translation=(0.05, 0.10))
    # tree 0 seen 5 times, tree 1 once: each trunk counts the same
    dets = true(pred[[0] * 5 + [1]])
    T, clamped = fit_map_correction(pred[[0] * 5 + [1]], dets, [0] * 5 + [1], np.radians(5))
    assert abs(np.degrees(T.rotation) - 2.0) < 1e-6 and not clamped
    np.testing.assert_allclose(T(pred), true(pred), atol=1e-9)


def test_correction_yaw_clamp():
    pred = _row()
    true = EuclideanTransform(rotation=np.radians(10.0))
    T, clamped = fit_map_correction(pred[[0, 1, 2]], true(pred[[0, 1, 2]]), [0, 1, 2], np.radians(5))
    assert clamped and abs(np.degrees(T.rotation) - 5.0) < 1e-9


def test_associate_groups_gate_side_and_one_to_one():
    pred = np.array([[0.0, 1.8], [1.2, 1.8], [0.0, -1.8]])
    dets = np.array([[0.1, 1.7], [0.6, 1.8], [0.05, -1.75], [0.0, 3.0], [-0.05, 1.82]])
    groups = [[0], [1], [2], [3], [4]]
    m = associate_groups(dets, groups, pred, [10, 11, 12], 0.35)
    # dets 0 and 4 both near trunk 10: only the nearer one (4) gets it; det 1 is outside the gate
    assert sorted((g[0], tid) for g, tid, _ in m) == [(2, 12), (4, 10)]


def test_frame_poses_local():
    # Robot at the broadside pose, row along world yaw 0.3; frame_1 is 0.05 m further along the robot heading.
    yaw = 0.3
    origin = np.array([100.0, 50.0])
    f1 = origin + 0.05 * np.array([np.cos(yaw), np.sin(yaw)])
    meta = {
        "broadside_robot_pose_xyt": [origin[0], origin[1], yaw],
        "cameras": {"base_cam": {"frames": {
            "frame_0": {"saved": True, "world_pose_xyt": [origin[0], origin[1], yaw]},
            "frame_1": {"saved": True, "world_pose_xyt": [f1[0], f1[1], yaw + np.radians(10)]},
            "frame_2": {"saved": False},
        }}},
    }
    fp = frame_poses_local(meta, "base_cam", origin, yaw)
    assert sorted(fp) == [0, 1] and fp[1][1] == "world_pose"
    np.testing.assert_allclose(fp[0][0]([[0.0, 1.8]]), [[0.0, 1.8]], atol=1e-12)
    # frame_1: 0.05 m along the row, robot yawed 10 deg: base (0, 1) -> local (0.05 - sin10, cos10)
    a = np.radians(10)
    np.testing.assert_allclose(fp[1][0]([[0.0, 1.0]]), [[0.05 - np.sin(a), np.cos(a)]], atol=1e-12)


def test_cluster_then_associate_keeps_objects_apart():
    # Two objects 0.24 m apart (the case of tree 1): A at -0.06 in frames -2,-1,1,2; B at -0.30 in all frames
    A = [-0.058, -0.054, -0.075, -0.053]
    B = [-0.299, -0.299, -0.298, -0.309, -0.296]
    dets = np.array([[x, 1.96] for x in A] + [[x, 1.83] for x in B])
    groups = cluster_detections(dets, 0.10)
    assert sorted(len(g) for g in groups) == [4, 5]
    m = associate_groups(dets, groups, np.array([[-0.022, 1.897], [-1.6, 1.9]]), [1, 2], 0.35)
    assert len(m) == 1 and m[0][1] == 1 and sorted(m[0][0]) == [0, 1, 2, 3]
