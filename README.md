# Orchard Tree Templating — `tree_template`

A ROS 2 package for real-time trunk mapping and trellis template placement in orchard rows. The package runs a particle-filter SLAM pipeline that ingests RGB-D trunk detections, maintains a per-slot EKF landmark map, and outputs a corrected odometry stream and a trunk registry used to place collision-object trellis templates for motion planning.

---

## Architecture Overview

```
RGB-D Camera
     │
     ▼
[trunk_detection_relay]  ──►  TrunkInfo measurements (robot frame)
                                         │
[OdomReframer] ──► EKF (local+global) ──►│
                                         ▼
                               [row_fast_slam]
                               ├── particle filter (pose)
                               ├── per-landmark EKF (slots)
                               ├── TrunkRegistry  ──► [generate_trellis_collision_obj]
                               └── odom_slam_best ──► [slam_odom_correction_tf]
                                                            │
                                                            ▼
                                                    map → odom TF

[save_best_particle_map] ──► slam_trunks.yaml (offline eval)
```

---

## Nodes

### `trunk_detection_relay`

Translates raw `TreeImageData` messages (pixel-space detections from the segmentation model) into `TrunkInfo` measurements expressed in the robot base frame, ready for the SLAM filter to consume.

It does **no** clustering, commitment, or map-frame logic — those are entirely the responsibility of `row_fast_slam`. Per detection it applies:

- TF lookup to transform the depth-derived 3-D point from the camera optical frame into the configured output frame (default `amiga__base`).
- Optional width-addition offset to correct for the camera seeing the near edge of a trunk rather than its centre.
- Rejection filters: minimum trunk width, required segmentation class, depth range, and optionally skipping an entire side (`near`/`far`).

Key parameters (`launch_config.yaml` → `trunk_detection_relay`):

| Parameter | Default | Description |
|---|---|---|
| `input_topic` | `tree_image_data` | Incoming `TreeImageData` topic |
| `output_topic` | `trunk_measurements_raw` | Outgoing `TrunkInfo` topic |
| `camera_frame` | `base_camera_color_optical_frame` | Source TF frame |
| `output_frame` | `amiga__base` | Target robot-relative frame |
| `min_trunk_width` | `0.035` m | Hard rejection floor |
| `required_class` | `0` | Only pass this segmentation class |
| `min_depth` / `max_depth` | `0.3` / `3.0` m | Valid depth range |
| `use_trunk_width_addition` | `true` | Shift measurement to trunk centre |

---

### `row_fast_slam`

The core mapping node. Implements a row-specialised FastSLAM 1.0 / 2.0 hybrid:

- **Particle filter** over robot pose SE(2), with vectorised propagation and systematic resampling.
- **Per-slot EKF** landmarks — each slot index along the row has its own 2-D Gaussian landmark in each particle.
- **Template map** — landmark priors are seeded from a regular grid (slot 0, 1, 2 … N) with configurable spacing, so the filter starts with a reasonable map even before the first detection.
- **Row axis initialisation** — the row heading is estimated from the first N odometry steps (`row_yaw_init_window`) so the slot coordinate frame aligns with the actual row direction.

#### Slot association

Each incoming `TrunkInfo` is projected into the row frame (along-row `s`, lateral `d`) and matched to the nearest slot within a window of ±`max_back_assoc` / ±`max_fwd_assoc` slots. In per-particle mode (`assoc.per_particle: true`) each particle maintains its own slot hysteresis anchor, so particles that have drifted apart do not share the same association decision.

The near/far side classification can be:
- `fixed` — all detections assigned the same side.
- `geometry` — derived from the sign of the lateral measurement.
- `semantic` — uses the class label from the detector, with a majority-vote buffer (`semantic_side_min_votes`) and a fallback mode.

#### EKF landmark update

For each accepted detection the EKF for the associated slot is updated in every particle. The innovation covariance is accumulated for particle weight updates. Measurement noise is configurable in either Cartesian (`meas_std_x_fwd`, `meas_std_y_lat`) or polar (`meas_std_range`, `meas_std_bearing`) form.

A Mahalanobis gate on the median particle (`maha_gate_median`) rejects outliers that are inconsistent across the whole filter.

#### Template prior

When `template_prior.enable: true` a soft log-weight penalty is added whenever a landmark deviates from its expected slot position. The penalty decays exponentially with observation count (`template_prior.decay_k`) and is disabled entirely after `template_prior.max_seen` hits, so the prior fades once the landmark is well-observed.

#### Resampling and roughening

Resampling is triggered every `resample_interval` measurement steps, or sooner if the effective sample size ratio falls below `neff_ratio_threshold`. Three roughening tiers are applied post-resample depending on how severe the Neff collapse is:

- **Moderate** — inject scaled motion noise (`scale_moderate_xy`).
- **Severe** — larger injection (`scale_severe_xy`, `scale_severe_yaw`).
- **Catastrophic** — full re-spread over several slot spacings plus a wide yaw scatter, triggered when the median Mahalanobis distance also exceeds `maha_catastrophic`.

#### FastSLAM 2.0 proposal (optional)

When `proposal.enable: true` a measurement-informed pose proposal is used instead of the standard motion-model draw. The proposal Jacobian accounts for both the landmark EKF uncertainty and the motion prior, and a zombie guard (`proposal.zombie_maha_threshold`) prevents particles with very high innovation cost from being moved by the proposal.

#### Origin recalibration (optional)

When `origin_recal.enable: true` the along-row origin `s` is periodically re-estimated from the well-observed best-particle landmarks (those with ≥ `origin_recal.min_seen` detections). This corrects slow along-row drift without reseeding the entire filter.

#### Spacing propagation (optional)

When `spacing_propagation.enable: true` the estimated along-row position of well-observed landmarks is used to nudge neighbouring unseen slots, propagating geometric consistency forward (or in both directions) along the row.

#### Outputs

| Topic | Type | Description |
|---|---|---|
| `fastslam_registry` | `TrunkRegistry` | Best-particle landmark map (throttled at `publish.registry_min_interval`) |
| `odom_slam_best` | `Odometry` | Best-particle pose in map frame (throttled at `publish.odom_min_interval`) |

Key parameters (`launch_config.yaml` → `row_fast_slam`):

| Parameter | Default | Description |
|---|---|---|
| `num_particles` | `1500` | Particle count |
| `slot_spacing` | `1.16` m | Expected inter-trunk spacing |
| `num_slots` | `50` | Initial template slots |
| `exceed_slot_num` | `true` | Dynamically extend beyond `num_slots` |
| `use_template_map` | `true` | Seed landmarks from a regular grid |
| `template_origin_from_first_measurement` | `true` | Anchor slot 0 to the first detection |
| `template_origin_min_hits` | `6` | Minimum detections before anchoring |
| `min_seen_count_to_publish` | `6` | Suppress registry entries until observed N times |
| `resample_interval` | `15` | Steps between resampling checks |
| `neff_ratio_threshold` | `0.25` | Neff/N below which resampling fires |

---

### `odom_reframer_calibrated`

Preprocesses the wheel/EKF odometry before it reaches `row_fast_slam`. On startup it averages the first N odometry and GPS readings, then:

- Optionally recentres the odometry origin (`recenter`).
- Applies a configurable yaw rotation (`rotate_yaw_deg`) to correct heading offsets.
- Publishes the averaged initial pose on a latched topic (`/initial_odom_correction`) for downstream YAML saving and GPS alignment.
- Publishes the averaged initial GPS fix on a latched topic (`/initial_gps_fix`).

These latched topics are consumed by `save_best_particle_map` to anchor the SLAM output to real-world coordinates at save time.

---

### `slam_odom_correction_tf`

Bridges the SLAM pose estimate back into the TF tree by continuously publishing the `map → odom` transform. It time-synchronises the wheel odometry (`/odometry/global`) with the SLAM best-particle pose (`/odom_slam_best`) using an approximate-time synchroniser, computes the correction (dx, dy, dyaw), and re-broadcasts it at a configurable rate (`publish_rate_hz: 30`). This allows the rest of the navigation stack to use the SLAM-corrected frame without any modifications.

---

### `save_best_particle_map`

A service-triggered node that snapshots the current SLAM state to a YAML file. It subscribes to the trunk registry, initial odometry correction, initial GPS fix, final GPS fix, and the best-particle odometry path. When the `save_slam_results` service is called it writes:

```yaml
initial_odom_correction: [x, y, z]
initial_yaw_correction_rad: <float>
initial_gps_fix: [lat, lon, alt]
final_gps_fix:   [lat, lon, alt]
robot_path:
  - [x, y, yaw, t_sec]
  ...
trees:
  0: [x, y]
  1: [x, y]
  ...
```

The GPS fixes are used by the offline evaluation scripts to project the local SLAM frame into UTM for comparison with ground-truth surveys.

---

### `generate_trellis_collision_obj`

Listens to the `TrunkRegistry` published by `row_fast_slam` and constructs MoveIt `CollisionObject` messages for each trunk. Each tree is modelled as a central leader cylinder plus a configurable number of V-trellis side-branch cylinders. Objects are published as `ADD` (upsert) operations, so the collision scene stays consistent with the live map even after MoveIt restarts.

---

### `trellis_from_rgbd_extraction`

An offline-to-online bridge. After a prior SLAM run has extracted per-tree `meta.yaml` files (via a separate extraction pipeline), this node reads them all at startup and re-publishes the complete `TrunkRegistry` anchored to a chosen reference tree. Setting `anchor_tree_id` via `ros2 param set` at runtime triggers a fresh publish, enabling the trellis collision map to be reconstructed from saved SLAM output without re-running the filter.

---

## Offline Evaluation Scripts

### `plot_nn_tree_error.py`

Compares a saved `slam_trunks.yaml` against a ground-truth JSON survey. The SLAM output is anchored to a chosen GT tree via GPS alignment and an optional yaw correction, then each SLAM tree is nearest-neighbour matched to a GT tree. Produces:

- Per-tree error (RMSE, mean, std) in x/y and decomposed into along-row (`s`) and lateral (`d`).
- A map-view subplot showing GT trees, SLAM trees, and match lines.

```bash
python plot_nn_tree_error.py \
    --gt_json all_info_test_block.json \
    --local_yaml slam_trunks_20240501_120000.yaml \
    --anchor_object_number 12
```

### `compare_slam_methods.py`

Head-to-head comparison of two SLAM runs (e.g. `row_fast_slam` vs. a factor-graph method) against a shared ground truth. Uses Hungarian (optimal) assignment for both GT matching and inter-method comparison. Produces five plots: map view, per-tree error bars, s/d scatter, precision-recall curves, and an inter-method map.

```bash
python compare_slam_methods.py \
    --gt_json all_info_test_block.json \
    --yaml_a slam_trunks_fastslam.yaml \
    --yaml_b orchard_slam_output.yaml \
    --anchor_object_number 12
```

### `plot_slam_variance.py`

Reads the per-landmark debug CSV emitted when `debug.csv_enable: true` and plots landmark covariance ellipses in slot/s space. Ellipses are colour-coded by `sqrt(det(Sigma))` (covariance area), giving a quick visual of which slots have converged and which remain uncertain.

Enable the debug CSV in `launch_config.yaml`:
```yaml
row_fast_slam:
  ros__parameters:
    debug.csv_enable: true
    debug.csv_path: /path/to/fastslam_debug_landmarks.csv
```

### `plot_utm_lat_lon.py`

Overlays the SLAM robot path and trunk map on GPS ground-truth trees in UTM coordinates. Useful for sanity-checking the GPS alignment step and visually inspecting path quality across a full row pass.

### `width_estimation.py`

Offline batch runner for the trunk-width estimation model. Iterates over paired RGB/depth image directories, runs segmentation, extracts top/centre/bottom keypoints and depth values, and saves per-image width results. Supports optional ground-truth JSON comparison for quantitative width accuracy analysis.

---

## Data Flow for a Typical Run

1. Start GPS fusion (`gps_fusion_launch.py`) — local EKF + NavSat transform + global EKF produce `/odometry/global`.
2. Start `odom_reframer_calibrated` — averages initial pose/GPS and republishes `/odometry/wheel` for the local EKF.
3. Start `trunk_to_template_pub_launch.py` — brings up `trunk_detection_relay`, `slam_odom_correction_tf`, `odom_reframer`, and `save_best_particle_map`.
4. Optionally start `row_fast_slam` separately (commented out in the launch file by default).
5. Drive down the row. The filter initialises its row axis from the first `row_yaw_init_window` odom steps, anchors the template origin on the first `template_origin_min_hits` detections, then runs in steady state.
6. Call the `save_slam_results` service to snapshot the map.
7. Run `plot_nn_tree_error.py` or `compare_slam_methods.py` against the saved YAML for evaluation.

---

## Key Interfaces

| Message | Package | Fields used |
|---|---|---|
| `TreeImageData` | `pf_orchard_interfaces` | Per-detection pixel x/y, depth, width, class, confidence |
| `TrunkInfo` | `tree_template_interfaces` | Robot-frame pose, width, side label |
| `TrunkRegistry` | `tree_template_interfaces` | List of `TrunkInfo` (the live landmark map) |
| `SaveSlamResults` | `tree_template_interfaces` | Service: optional output path → success + message |

---

## Configuration

All runtime parameters live in `config/launch_config.yaml`. The most commonly tuned groups are:

- **`row_fast_slam.slot_spacing`** — set to match the actual inter-trunk spacing in the block.
- **`row_fast_slam.num_particles`** — higher = more accuracy, more CPU. 600–1500 is typical.
- **`row_fast_slam.motion_noise.*`** — tune to the wheel slip characteristics of the platform.
- **`row_fast_slam.meas_noise_model`** / **`meas_std_range`** / **`meas_std_bearing`** — tune to depth camera accuracy.
- **`row_fast_slam.template_prior.*`** — enable and tune when the row spacing is reliable and you want the prior to resist outlier detections.
- **`trunk_detection_relay.min_trunk_width`** / **`max_depth`** — primary levers for filtering spurious detections.