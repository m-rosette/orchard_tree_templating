# Orchard Tree Templating — `tree_template`

A ROS 2 (Humble) package that maps the trunks of an orchard row while the robot drives along it. The map is used to place trellis collision templates for motion planning.

It takes trunk detections from an RGB-D camera and the robot's odometry, and runs a row-specialised FastSLAM particle filter. It outputs:

- a trunk registry (the map),
- a corrected robot pose,
- a saved map file (`slam_trunks_*.yaml`), plus an optional GPS-corrected world copy.

This repository is meant to be used as **one sub-repo in a larger robot workspace**. It does not drive the robot, run the camera or run the trunk detector. You connect it to your robot through two config files (see [Integrating with your robot](#integrating-with-your-robot)):

- a **robot config** (`config/robot/*.yaml`): a ROS params file with the node parameters specific to one robot or machine (frames, sensor topics, output paths), plus a few launch arguments for the EKF config directory and the detector;
- a **params file** (`config/launch_config*.yaml`) with the algorithm tuning.

The repo contains two ROS packages:

| Package | Type | Contents |
|---|---|---|
| `tree_template` | `ament_python` | All nodes, launch files and configs |
| `tree_template_interfaces` | `ament_cmake` | `TrunkInfo`, `TrunkRegistry` msgs; `SaveSlamResults`, `UpdateTrellisPosition` srvs |

---

## What's new on `data_association_fix`

| Change | Where | What it does | Default |
|---|---|---|---|
| **Width-aware track layer** | `trunk_track_layer.py`, `row_fast_slam` `assoc.track_layer.*` | Links detections into short-lived per-tree tracks in the odometry frame, using position first and trunk width second. When a second, different tree lands in a slot that already has a tree, it gets its own "extra" landmark (key ≥ 100000) instead of being merged. This fixes closely spaced replants. | Code: off. `launch_config.yaml`: **on** |
| **Width likelihood** | `row_fast_slam` `assoc.width_lik.*` | Adds a penalty to particle weights when a detection's width doesn't match the landmark's width. | Off (gives about +1 tree on row 7; not adopted yet) |
| **Mask-based trunk position** | `trunk_detection_relay` `position_source` | `mask` takes the trunk position from the whole segmentation mask instead of only the bottom row of pixels, which cuts the scatter by 15–50%. | Code: `bottom`. Config: `mask` |
| **Image-time stamps** | `trunk_detection_relay` | `TrunkInfo.stamp` is now the image's `header.stamp`, not the time the detection came out. Before, SLAM looked up an odometry pose about 1.2 s too late, which shortened the row. | Always on (falls back to `now()` if the stamp is zero) |
| **Longer odometry history** | `row_fast_slam` `odom_history_sec` | Raised to 5 s so odometry poses at image time are still in the buffer when measurements arrive 1–3 s late. | 5.0 in config |
| **Velocity from pose** | `odom_reframer` `twist_from_pose`, `twist_cov_*`, `pose_cov_*` | Republishes the body velocity computed from the pose channel, with an explicit error model. This is for robots whose own odometry has zero pose covariance or whose twist disagrees with its pose. | Code: off. Config: on |
| **GPS-heading fusion mode** | `ekf_fusion.launch.py mode:=gps_heading`, `fix_cov_relay`, `gps_course_heading`, `config/nav/*_gps_heading.yaml` | `ekf_global` fuses wheel velocities, IMU yaw rate, RTK GPS position and a heading computed from the GPS course. Use it when the IMU yaw isn't referenced to north. | `gps_heading` (`legacy` turns off all of this branch's GPS work) |
| **GPS-corrected world map** | `save_best_particle_map` `world_correction.*` | At save time, places each tree in a GPS (ENU) frame using RTK fixes and GPS course near the time the tree was seen. Writes `trees_world` / `robot_path_world` and a separate `<name>_world.yaml`. | The launch turns it on only with `use_gps_fusion:=true` in `gps_heading` mode (and `use_ekf_fusion:=true`, the default) |
| **Robot config + launch restructure** | `config/robot/*.yaml`, `ekf_fusion.launch.py`, `trunk_to_template_pub.launch.py` | Robot/machine node parameters moved out of the params files into a robot config params file. `use_ekf_fusion` (local EKF) and `use_gps_fusion` (global EKF) are now separate flags. `ekf_fusion.launch.py` replaces `gps_fusion.launch.py` and `gps_fusion_2025.launch.py`. | Local EKF on, GPS (global) fusion off |
| **Registry order and anchor fixes** | `row_fast_slam` | The registry is published in along-row order (extras are no longer appended at the end). Slot 0 is anchored using the odometry pose at the time of the measurement. | — |

Results on the 2023 Prosser rows (offline replays, several seeds each):

- **Track layer:** row 7 goes from about 46 to about 49 trees matched to ground truth, and only one or two extra trees are reported.
- **Mask position + 10% edge band** (live detector rate): F1 0.973 on row 7 and 0.944 on row 6.
- **World correction:** along-row error drops from about 0.20 m to about 0.15 m and the row-length error from about 0.7% to about 0.3%. Lateral error gets worse, from about 0.10 m to 0.12–0.16 m.

---

## Architecture

```
  (external) RGB-D camera ──► (external) trunk detector ──► TreeImageData
                                                              │
                                                              ▼
                                                  [trunk_detection_relay] ──► /trunk_measurements_raw (TrunkInfo, base frame, image stamp)
                                                                                     │
  (external) robot odom ──► [odom_reframer] ──► /odometry/wheel                      │
                                   │                  │                              │
                                   │          [ekf_local] ──► /odometry/local ───────┤ odom_topic
           GPS /fix ───────────────┤                  │                              │
                                   │     [navsat_transform] + [ekf_global] ──► /odometry/global
                                   │        (gps_heading mode adds fix_cov_relay ──► /fix_rtk
                                   │                         gps_course_heading ──► /gps/course_imu)
                                   │                                                  ▼
                                   │                                          [row_fast_slam]
                                   │                                ├── particles over SE(2) pose
                                   │                                ├── per-landmark EKF (template slots + extra keys)
                                   │                                ├── track layer (data association)
                                   │                                ├── /fastslam_registry (TrunkRegistry) ──► [tree_template] MoveIt collision objects
                                   │                                └── /odom_slam_best ──► [slam_odom_correction_tf] ──► map→odom TF (optional)
                                   ▼
             /initial_odom_correction, /initial_gps_fix (latched)
                                   │
                                   ▼
                   [save_best_particle_map] ── service save_slam_results ──► slam_trunks_<stamp>.yaml (+ _world.yaml)
```

---

## Install

```bash
cd <your_ws>/src
git clone <this repo> orchard_tree_templating
# external message dependency for detections:
#   pf_orchard_interfaces (TreeImageData) must also be in the workspace
cd <your_ws>
rosdep install --from-paths src --ignore-src -y     # ROS + apt Python deps from package.xml
pip install -r src/orchard_tree_templating/requirements.txt   # utm (no rosdep key on 22.04)
colcon build --packages-select tree_template_interfaces tree_template
source install/setup.bash
```

`package.xml` declares all runtime dependencies, including:

- `robot_localization`, used by `ekf_fusion.launch.py`;
- `moveit_msgs`;
- `pf_orchard_interfaces`, which has to be built from source in the same workspace;
- the apt Python packages (`numpy`, `scipy`, `yaml`, `opencv`, `matplotlib`, `pandas`).

`utm` comes from `requirements.txt`. Only the RGB-D extraction and evaluation scripts need it.

---

## Integrating with your robot

Robot-specific settings live in two places:

- **Robot config** (`config/robot/*.yaml`): a normal ROS params file with the node parameters that tie the pipeline to one robot or machine (frames, sensor topics, output paths). The launch passes it to every node **after** `params_file`, so its values win. Copy `config/robot/prosser_2023_warthog.yaml`, edit it, and pass `robot_config:=<path>`. Because it is a plain params file, tools that start nodes directly can pass it as a second `--params-file`:

  ```bash
  ros2 run tree_template odom_reframer_calibrated --ros-args -r __node:=odom_reframer \
      --params-file launch_config.yaml --params-file my_robot.yaml
  ```

- **`robot_platform:=warthog|amiga`** (default `warthog`) picks the defaults for everything below in one go: the robot config, the params file, the EKF config directory, the navsat GPS/IMU topics, and the robot description (`publish_robot_description`). Any of them can still be set on its own.

- **Launch arguments** for the settings a params file can't hold: the `robot_localization` config directory (`nav_config`), the navsat GPS/IMU input topics (`gps_topic`, `imu_topic`), `publish_robot_description`, and the detector paths (`detector_script`, `width_estimation_package_path`, `width_estimation_package_data_path`; defaults are under `~/trunk_width_ws`).

ROS params files don't expand `~` or `$HOME`, so paths in the robot config must be absolute.

### Required inputs

| Input | Default topic / frame | Consumed by | Notes |
|---|---|---|---|
| Trunk detections | `tree_image_data` (`pf_orchard_interfaces/TreeImageData`) | `trunk_detection_relay` | `header.stamp` **must be the image time**. Per-detection pixel coordinates, depth, width, class, and (for `position_source: mask`) the mask `xs`/`ys`. |
| Robot odometry | `/odometry/filtered` (`nav_msgs/Odometry`) | `odom_reframer` | Any continuous odometry. Use the sensor timestamps (`use_now_stamp: false`). |
| GPS fix | `/fix` (`sensor_msgs/NavSatFix`) | `odom_reframer`, `navsat_transform`, `save_best_particle_map` | Optional for SLAM itself. Needed for GPS-anchored and world-corrected output. |
| IMU | `/imu/data` | `ekf_local`, `navsat` (legacy mode) | Set in the EKF config (`imu0`) and `imu_topic`. |
| TF | `<base frame>` → camera optical frame | `trunk_detection_relay` | `camera_frame` / `output_frame` params. Static TF from your URDF or a static publisher. |

### Where each robot setting goes

| Setting | Where |
|---|---|
| Detection topic, camera frame, base frame, lateral sign | robot config → `trunk_detection_relay`: `input_topic`, `camera_frame`, `output_frame`, `base_x_is_right` |
| Robot odometry topic, base frame, GPS topic | robot config → `odom_reframer`: `input_odom_topic`, `base_frame`, `gps_topic` |
| Trellis collision-object frame | robot config → `tree_scene_node.trellis_frame` |
| Map output path, GPS topic, GPS antenna x | robot config → `save_best_particle_map`: `out`, `final_gps_navsat_topic`, `world_correction.antenna_x` |
| RGB-D extraction folder | robot config → `trellis_from_rgbd_extraction.extraction_dir` |
| SLAM debug CSVs (`fastslam_debug*.csv`, rewritten each run) | `tree_template/debug_data/` in the package's source folder by default, also when running from a copy install (`row_fast_slam` finds `<ws>/src/**/package.xml` named `tree_template`). Override with `row_fast_slam.debug.csv_path` |
| EKF base frame, IMU topic, noise | EKF files in the `nav_config` directory (`ekf_local.yaml`, ...) |
| GPS relay input, course-heading base frame | `gps_heading_nodes.yaml` in the `nav_config` directory |
| navsat GPS / IMU inputs (legacy mode) | launch args `gps_topic`, `imu_topic` |
| Robot description (base → IMU / GPS / cameras) | `harvest_hardware_description` (apple-harvest repo), started with `publish_robot_description:=true`. Warthog: `launch/warthog_state_publisher.launch.py` (`warthog_2023.urdf.xacro`). Amiga: `launch/robot_state_publisher.launch.py` (`amiga_ur_gripper.urdf.xacro`; Intelligence Kit frames in `amiga.urdf.xacro`), or `arm_control` |
| Detector process | launch args `detector_script`, `width_estimation_*`, `use_detector` |

### What to change for a new robot

1. **Robot config.** Copy one from `config/robot/` and set the frames, topics and paths.
2. **EKF configs.**
   - `config/nav/` holds the 2023 Warthog setup and `config/nav_2025/` the 2025 Amiga setup.
   - For a new platform, copy one into a new directory, set the frames/topics, tune the noise to your wheel odometry and IMU, and pass `nav_config:=<dir>`.
   - `gps_heading` mode also needs `ekf_global_gps_heading.yaml`, `navsat_gps_heading.yaml` and `gps_heading_nodes.yaml` in that directory.
3. **Row geometry** (params file).
   - `row_fast_slam.slot_spacing` is the planting spacing. Use the median of the normal tree gaps, not the mean of all gaps, because short replant gaps pull the mean down.
   - Also set `row_dir_sign`, `start_side` and `side_mode`.
4. **Detector filters** (params file). `min_trunk_width`, `min_depth` / `max_depth`, `required_class` and `side_class_skipping` in `trunk_detection_relay`.

### Provided configs

| File | Platform / dataset |
|---|---|
| `config/robot/prosser_2023_warthog.yaml` | Robot config for `robot_platform:=warthog` (default): 2023 Prosser bags, Clearpath Warthog |
| `config/robot/prosser_2025_amiga.yaml` | Robot config for `robot_platform:=amiga`: 2025 Prosser, Amiga (`/filter/state`, `/gps/pvt`) |
| `config/robot/prosser_2026_amiga.yaml` | Robot config for the 2026 Prosser Amiga bags (use with `robot_platform:=amiga nav_config:=nav_2026`): `/filter/state` pose not usable, odometry from its twist only |
| `config/launch_config.yaml` | Params file (default). Same as the 2023 tuning |
| `config/launch_config_2023_with_polar_meas.yaml` | Params file: 2023 Prosser bags |
| `config/launch_config_2025_with_polar_meas.yaml` | Params file: 2025/2026 Prosser Amiga, Envy V-trellis (tree spacing about 0.60 m; `slot_spacing 0.63` in odometry units) |
| `config/nav/`, `config/nav_2025/`, `config/nav_2026/` | `robot_localization` EKF + navsat configs for each platform. `nav_2025` has no gps_heading files yet. `nav_2026` has only a velocity-only `ekf_local` |
| `config/camera_extrinsics/` | Depth -> color extrinsics for bags without an aligned depth topic (`prosser_2025_*`: fitted; `prosser_2026_mast_camera_nominal`: nominal RealSense values) |
| `config/bag_qos_override.yaml` | `/tf_static` QoS override for replaying ROS 1-converted bags |

---

## Running the full SLAM pipeline

### 1. Live, or bag replay through the full stack

```bash
CFG=$(ros2 pkg prefix tree_template)/share/tree_template/config

# Terminal 1: robot / bag
# 2023 Warthog bags: drop the bag's TF; robot_platform:=warthog republishes it from warthog_2023.urdf.xacro
ros2 bag play <bag> --clock --remap /tf:=/bag/tf /tf_static:=/bag/tf_static
# 2025 Amiga bags (no TF in the bag)
ros2 bag play <bag> --clock

# Terminal 2: SLAM pipeline (recommended setup: local EKF for SLAM, GPS-heading global EKF for the world map)
ros2 launch tree_template trunk_to_template_pub.launch.py \
    robot_platform:=warthog \
    publish_robot_description:=true \
    use_gps_fusion:=true \
    use_sim_time:=true            # false on the robot
```

Other common setups:

```bash
# Default: local fusion only (wheel odom + IMU -> /odometry/local), no GPS fusion
ros2 launch tree_template trunk_to_template_pub.launch.py

# Odometry only (no robot_localization): SLAM runs on /odometry/wheel
ros2 launch tree_template trunk_to_template_pub.launch.py use_ekf_fusion:=false

# Previous behaviour (IMU-heading global EKF, no GPS-heading nodes, no world map, twist_from_pose off)
ros2 launch tree_template trunk_to_template_pub.launch.py use_gps_fusion:=true gps_fusion_mode:=legacy

# 2025 Amiga (nav_2025 has no gps_heading files, so GPS fusion must use legacy mode)
ros2 launch tree_template trunk_to_template_pub.launch.py robot_platform:=amiga \
    use_gps_fusion:=true gps_fusion_mode:=legacy

# Your own robot config / tuning on top of a platform's defaults
ros2 launch tree_template trunk_to_template_pub.launch.py robot_platform:=warthog \
    robot_config:=$CFG/robot/prosser_2023_warthog.yaml params_file:=$CFG/launch_config.yaml
```

Launch arguments:

| Arg | Default | Meaning |
|---|---|---|
| `robot_platform` | `warthog` | `warthog` (2023 bags) or `amiga` (2025 bags). Sets the defaults of `params_file`, `robot_config`, `nav_config`, `gps_topic`, `imu_topic` and the static TFs |
| `params_file` | warthog: `config/launch_config.yaml`, amiga: `config/launch_config_2025_with_polar_meas.yaml` | Algorithm tuning |
| `robot_config` | warthog: `config/robot/prosser_2023_warthog.yaml`, amiga: `config/robot/prosser_2025_amiga.yaml` | Robot / machine params file, applied after `params_file` |
| `use_sim_time` | `false` | `true` for bag replay |
| `use_ekf_fusion` | `true` | Local fusion only: `ekf_local` (wheel odometry + IMU → `/odometry/local`) and the IMU static TF. `false` = no robot_localization, SLAM on `/odometry/wheel` |
| `use_gps_fusion` | `false` | Adds GPS (global) fusion: `navsat_transform` + `ekf_global` (→ `/odometry/global`) and the GPS static TF. GPS is never fused into `ekf_local`. Has no effect with `use_ekf_fusion:=false` |
| `gps_fusion_mode` | `gps_heading` | `gps_heading`: this branch's GPS work. `ekf_global` takes its heading from the GPS course and its position from RTK GPS, `fix_cov_relay` / `gps_course_heading` run, and `odom_reframer.twist_from_pose` and `save_best_particle_map.world_correction.enable` are set to `true`. `legacy`: IMU heading + `yaw_offset`, none of the GPS-heading nodes, and both of those set to `false`. The two parameters are also `false` whenever GPS fusion isn't running |
| `slam_odom_topic` | `/odometry/local` with the EKF, `/odometry/wheel` without | `row_fast_slam.odom_topic`. SLAM should use local, continuous odometry; `/odometry/global` is not recommended (see notes) |
| `nav_config` | warthog: `nav`, amiga: `nav_2025` | `robot_localization` config directory under `config/`, or an absolute path |
| `gps_topic` / `imu_topic` | warthog: `/fix` / `/imu/data`, amiga: `/gps/pvt` / `/oak0/imu` | navsat inputs in legacy mode |
| `publish_robot_description` | warthog `true`, amiga `false` | Start `robot_state_publisher` for `robot_platform` from `harvest_hardware_description` (apple-harvest repo; not a declared dependency). Amiga default is off because `arm_control` publishes it |
| `use_detector` | `true` | Start the external detector (`detector_script`) |
| `detector_script`, `width_estimation_package_path`, `width_estimation_package_data_path` | under `~/trunk_width_ws` | Detector script and its environment variables |
| `detector_rgb_topic` / `detector_depth_topic` | warthog: `/camera/color/image_raw` / `/camera/aligned_depth_to_color/image_raw`, amiga: `/camera/base_camera/color/image_raw` / `/camera/base_camera/aligned_depth_to_color/image_raw` | Detector inputs, passed as `RGB_IMAGE_TOPIC` / `DEPTH_IMAGE_TOPIC` (the detector's `__main__` values are only fallbacks). Depth must be aligned to the RGB image. The detector stamps `tree_image_data` with the RGB image's own `frame_id` |
| `use_slam_tf` | `false` | Run `slam_odom_correction_tf` (SLAM `map → odom`; needs `/odometry/global`) |
| `use_trellis_from_extraction` | warthog `false`, amiga `true` | Run `trellis_from_rgbd_extraction`. It publishes on `fastslam_registry` too, so turn it off when you only want the live map |

Every run starts `tree_scene_node` (collision objects), `trunk_detection_relay`, `odom_reframer`, `row_fast_slam` and `save_best_particle_map`. The other nodes depend on the flags above.

To run the EKF chain on its own:

```bash
ros2 launch tree_template ekf_fusion.launch.py robot_platform:=warthog use_gps_fusion:=true mode:=gps_heading use_sim_time:=true
```

### 2. Drive the row

When the robot drives the row:

1. `row_fast_slam` estimates the row direction from the first `row_yaw_init_window` odometry steps.
2. It anchors slot 0 on the first trunk once that trunk has `template_origin_min_hits` detections.
3. It then tracks along the row.

Trees appear in `/fastslam_registry` once they have been seen `min_seen_count_to_publish` times. Watch `trunk_measurements_raw_markers` and the registry in RViz.

### 3. Save the map

```bash
ros2 service call /save_slam_results tree_template_interfaces/srv/SaveSlamResults \
    "{output_path: /path/to/out/<name>, append_timestamp: true}"
```

This writes `<name>_<YYYYmmdd_HHMMSS>.yaml`:

```yaml
initial_odom_correction: [x, y, z]
initial_yaw_correction_rad: <float>
initial_gps_fix: [lat, lon, alt]
final_gps_fix:   [lat, lon, alt]
row_yaw_rad: <float>
robot_path:          # SLAM frame
  - [x, y, yaw, t_sec]
trees:               # SLAM frame, along-row order
  0: [x, y]
world_gps_origin: [lat, lon]   # only with world correction
trees_world: ...               # ENU metres from world_gps_origin
robot_path_world: ...
```

With `world_correction.write_separate_file: true`, it also writes `<name>_world.yaml` with the world-corrected trees under `trees`.

The default output path is `save_best_particle_map.out` from the robot config; `output_path` in the call overrides it.

If `require_*` is true for an input that never arrived, the save fails. For example, `require_gps_fix: true` with no GPS fails, so set those params to `false` on robots without GPS.

### 4. Templates from a saved map

See [Templating from a saved map](#templating-from-a-saved-map): extract the RGB-D frames at each mapped tree,
place the trellis templates around one anchor tree, and correct them locally from the trunk detections.

---

## Templating from a saved map

The SLAM map is good to about 0.2 m along a full row. That is enough to drive the mobile base, but not enough to
place collision templates for the arm. So the templating phase works at one tree at a time, with the robot parked
in front of it:

1. Place all the templates of the saved map relative to the robot at that tree (the anchor tree).
2. Detect the trunks in the base-camera image at that parked pose.
3. Shift the templates so that they agree with the detections (local correction).
4. Detect the apples and plan the arm motions with the corrected templates.

On recorded data, "parked in front of the tree" is the image where the robot was broadside to the tree. The steps
below use the RGB-D frames extracted from the bag at each mapped tree.

### Step 1: extract the RGB-D frames at each mapped tree

`fast_slam_rgbd_extraction` reads a saved map and the bag. For each tree it finds the time when the tree was
straight to the side of the robot (broadside) and saves the mast-camera frame and a buffer of base-camera frames
around it, with their robot poses, to `tree_NNN/` folders (`meta.yaml`, `camera_info.yaml`, `color.png`,
`depth.png`, `depth_aligned.png`).

Broadside source (`--broadside_source`):

| Value | Odometry is fitted to | Use it when |
|---|---|---|
| `odom` (default) | the SLAM `robot_path` | the `robot_path` is smooth |
| `trees` | the map trees, with the trunk detections (`--meas_topic`, default `/trunk_measurements_raw`) | the `robot_path` jumps, for example at the start of a row while the filter converges. Needs an `--odom_bag` with the odometry and the trunk detections of the same SLAM run |
| `slam_path` | (no fit) the `robot_path` pose itself | debugging |

`robot_path` is the pose of the best particle at each time, so it jumps when a different particle becomes the
best. With `trees`, `meta.yaml` records the fit (`n_matched`, `rms_inlier_m`, `search_score`,
`search_other_score`; an other score near the score means an ambiguous match, for example a shift of one tree
spacing).

Example (2026 Prosser bag; the bag pose of `/filter/state` is not usable, so the odometry and the trunk detections
come from a recording of `/odometry/local` and `/trunk_measurements_raw` made while the bag played through the
pipeline):

```bash
CFG=$(ros2 pkg prefix tree_template)/share/tree_template/config/camera_extrinsics
ros2 run tree_template fast_slam_rgbd_extraction \
  --yaml <map_dir>/slam_trunks.yaml \
  --bag <bag_dir> \
  --output_dir <extraction_dir> \
  --odom_bag <map_dir>/odom_rec --odom_topic /odometry/local --odom_stamp header --odom_to_base_yaw_deg 0 \
  --broadside_source trees \
  --image_time_offset_s -0.04 \
  --cam1_depth_topic /camera/mast_camera/depth/image_rect_raw \
  --cam1_depth_to_color $CFG/prosser_2026_mast_camera_nominal.yaml \
  --cam2_depth_to_color $CFG/prosser_2025_base_camera.yaml \
  --cam2_buffer_radius 2
cp <map_dir>/slam_trunks.yaml <extraction_dir>/slam_trunks_<name>.yaml   # the trial nodes read the map from here
```

- `--odom_stamp header`: a recording made during bag playback has the bag clock only in the header stamps.
- `--camN_depth_to_color`: registers raw depth to the color image when the bag has no aligned depth topic.
- `config/rgbd_extraction_cli.yaml` has the 2025 example.

### Step 2: interactive session in RViz

`templating_trial.launch.py` (package `harvest`, apple-harvest repo) starts MoveIt with RViz, `tree_scene_node`
(templates as collision objects), `trellis_from_rgbd_extraction` (map, anchor and correction),
`apple_prediction` (presaved images, point clouds, apples), the trunk detector with `trunk_detection_relay`, and
`voxelize_scan`. It does not start SLAM.

```bash
# Terminal 1
ros2 launch harvest templating_trial.launch.py trial_dir:=<extraction_dir>
```

RViz (`view_templating.rviz`): set the Fixed Frame to `amiga__base`. The displays are PlanningScene (templates),
`colorize_pt_cloud` (mast camera, `/rgbd_pointcloud`), `base_cam_pointcloud` (`/base_cam/pointcloud`),
`trunk_annotated` (trunk detector, `/annotated_segmented_image`), `apple_markers`.

```bash
# Terminal 2
TREE=10

# 1. Load the images of the tree: mast and base-camera point clouds
ros2 param set /apple_prediction presaved_images.tree_id $TREE

# 2. Place all the templates of the SLAM map, with this tree as the anchor (uncorrected SLAM placement).
#    Set it again to go back to the uncorrected placement.
ros2 param set /trellis_from_rgbd_extraction anchor_tree_id $TREE

# 3. Select the base-camera image of the parked robot (-2 ... 2; 0 = broadside)
ros2 param set /apple_prediction presaved_images.base_cam_frame_index 0

# 4. (optional) Trunk detection on that image only: see trunk_annotated
ros2 service call /trigger_estimation std_srvs/srv/Trigger {}

# 5. Local correction on that image: the templates move onto the detected trunks
ros2 service call /correct_template_anchor std_srvs/srv/Trigger {}

# 6. Apple detection on the mast image: apple_markers, /apple_annotated, poses in the response
ros2 service call /apple_prediction harvest_interfaces/srv/ApplePrediction {}
```

What you see:

- The templates and the point clouds are in the robot base frame of the selected base-camera image.
  `trellis_from_rgbd_extraction` follows `presaved_images.base_cam_frame_index` of `/apple_prediction` (through
  `/parameter_events`) and publishes the templates again in the robot frame of that image (its odometry pose,
  `world_pose_xyt` in `meta.yaml`). When you select a different image, the templates and the point clouds move
  together, because the robot moved and the trees did not. The map error stays the same in the world, so one local
  correction is enough at one parked pose.
- `/correct_template_anchor` (with `correction_frames: current`) triggers the detector once on the selected image,
  matches the detections to the templates in the robot frame of that image, fits one rigid correction
  (translation, and yaw if two or more trunks match) and moves all the templates. The odometry is not used in the
  fit. After the correction the anchor template is on its detection in that image. The corrected map then stays at
  the same place in the world when you select a different image.
- The response message is JSON: `summary`, `frame`, `dx`/`dy`/`dyaw` and `anchor_dx`/`anchor_dy` (robot frame of
  the image), `matched_tree_ids`, `rms_residual_m`, `anchor_residual_m`, `detections`, or `error` (for example
  when no detection matches the anchor within `association_gate_m`).
- The trunk detector runs only when it is triggered: `ros2 param set` publishes the images, it does not run the
  detector. `/annotated_segmented_image` is not latched, so RViz shows only the images published while it is open.

Correction modes (`trellis_from_rgbd_extraction` parameter `correction_frames`):

| Value | The correction uses |
|---|---|
| `current` (default) | Only the selected base-camera image: the parked-robot correction, no odometry in the fit |
| `buffer` | All the frames in `frame_offsets` (default `[-2, -1, 0, 1, 2]`): the node selects each frame, triggers the detector and puts the detections in the local frame with the odometry poses of the frames. At the end frame 0 is selected |

One detection can be several centimetres off (the trunk depth is read at the bottom of the mask). On a parked robot,
several images of the same pose average this without odometry.

### Step 3: automated trial

`orchard_templating.py` (package `harvest`) runs steps 1-6 for each tree, then plans the arm to each apple with the
templates and with a voxelized mast point cloud, and saves `results.yaml` and images in
`data_save_dir/<dataset>/<time>/tree_<id>/`:

```bash
ros2 run harvest orchard_templating.py --ros-args -p tree_id:=$TREE -p trial_dir:=<extraction_dir>
ros2 run harvest orchard_templating.py --ros-args -p "tree_ids:=[10, 20, 30]" -p trial_dir:=<extraction_dir>
```

Use the same `trial_dir` as the launch, and start it after the launch is ready.

---

## Nodes

### `trunk_detection_relay`

Converts `TreeImageData` (pixel-space detections) into `TrunkInfo` measurements in the robot base frame. It doesn't cluster detections or do any map logic.

For each detection it:

- takes the position from the bottom row of the mask or from the whole mask (`position_source`),
- converts depth to a 3-D point,
- transforms it to `output_frame` with TF,
- optionally adds half the trunk width so the point is at the trunk centre,
- applies rejection filters,
- stamps the result with the **image time**.

| Parameter | Default (code) | Description |
|---|---|---|
| `input_topic` / `output_topic` | `tree_image_data` / `trunk_measurements_raw` | |
| `camera_frame` / `output_frame` | `base_camera_color_optical_frame` / `amiga__base` | Robot configs set them: warthog `camera_color_optical_frame` / `warthog_base_footprint`, amiga `base_camera_color_optical_frame` / `amiga__base` |
| `position_source` | `bottom` | `bottom` or `mask` (config uses `mask`) |
| `base_x_is_right` | `true` | Lateral sign convention of the base frame |
| `use_trunk_width_addition` | `true` | Shift the measurement from the trunk's front face to its centre |
| `min_trunk_width` | `0.03` m | 0 disables |
| `required_class` | `0` | |
| `min_depth` / `max_depth` | `0.3` / `5.0` m | |
| `side_class_skipping` | `near` | `near`, `far` or `''` |
| `publish_debug_markers` | `true` | RViz markers on `trunk_measurements_raw_markers` |

### `row_fast_slam`

The core mapping node: a row-specialised FastSLAM 1.0 / 2.0 hybrid.

- **Particle filter** over the robot's SE(2) pose, with vectorised propagation and systematic resampling.
- **Per-landmark EKFs.** Landmarks are template slots spaced `slot_spacing` apart (slot 0, 1, 2 …), plus extra landmarks created by the track layer.
- **Row axis initialisation.** The row heading comes from the first `row_yaw_init_window` odometry steps.
- **Odometry history buffer.** Each measurement is applied at the odometry pose for its own timestamp, up to `odom_history_sec` back and `meas_max_lag_sec` late.

#### Data association

Each `TrunkInfo` is projected into the row frame: along-row distance `s` and lateral distance `d`. It is matched to a slot within `max_back_assoc` / `max_fwd_assoc` slots of the current one, either for each particle separately (`assoc.per_particle`) or once for the median particle. The side (near/far) comes from `side_mode` (`fixed` / `geometry` / `semantic`).

**Track layer** (`assoc.track_layer.enable: true`):

1. Detections are placed in the odometry frame, which is accurate over a few seconds.
2. Each detection is linked to a track if it is within `link_gate_m`. The link is refused if the width differs from the track's width by more than `width_break_m`.
3. A track ends after `max_gap_sec` with no detection.
4. Each track is bound to one landmark.
5. A track that falls in a slot already owned by another tree becomes a new **extra** landmark if either:
   - it is at least `min_tree_sep_m` away from the other tree, or
   - it is at least `width_sep_m` away and its width differs by at least `width_distinct_m`.

   The split only happens after `split_min_hits` detections.

The tracks are written to `fastslam_debug_tracks.csv` when `debug.csv_enable` is true.

| Parameter | Config | Description |
|---|---|---|
| `assoc.track_layer.link_gate_m` | 0.20 | Odometry-frame link gate |
| `assoc.track_layer.max_gap_sec` | 1.0 | Track expiry |
| `assoc.track_layer.width_break_m` | 0.015 | Refuse a link if the width differs by more than this |
| `assoc.track_layer.width_min_samples` | 3 | Widths needed before width is used |
| `assoc.track_layer.min_tree_sep_m` | 0.30 | Separation that always means a different tree |
| `assoc.track_layer.width_sep_m` / `width_distinct_m` | 0.15 / 0.010 | Closer separation that counts only if the widths differ |
| `assoc.track_layer.split_min_hits` | 3 | Detections needed before a new landmark is split off |
| `assoc.width_lik.enable` / `sigma_m` / `max_pen` | false / 0.004 / 4.0 | Width penalty on particle weights |

#### Landmark update, weighting and gating

- **EKF update.** Each accepted detection updates the associated landmark's EKF in every particle.
- **Measurement noise.** Either Cartesian (`meas_std_x_fwd`, `meas_std_y_lat`) or polar (`meas_std_range`, `meas_std_bearing`). The config uses polar 0.20 / 0.20. Smaller values such as 0.10 made the filter diverge in some runs.
- **Outlier gate.** A Mahalanobis-distance gate on the median particle (`maha_gate_median`) rejects detections that are inconsistent across the whole filter.
- **Template prior** (`template_prior.*`). A soft pull toward the expected slot position, which fades as a landmark gets observed more.
- **Resampling** happens every `resample_interval` steps, or when Neff falls below `neff_ratio_threshold`. After resampling, noise is added in three tiers: moderate, severe and catastrophic.
- **FastSLAM 2.0 proposal** (`proposal.*`), **origin recalibration** (`origin_recal.*`) and **spacing propagation** (`spacing_propagation.*`) are optional refinements.

#### Outputs

| Topic | Type | Description |
|---|---|---|
| `fastslam_registry` | `TrunkRegistry` | Best-particle map in along-row order (throttled by `publish.registry_min_interval`) |
| `odom_slam_best` | `Odometry` | Best-particle pose (throttled by `publish.odom_min_interval`) |

Key parameters (values from `launch_config.yaml`):

| Parameter | Value | Description |
|---|---|---|
| `odom_topic` | `/odometry/local` | Motion input |
| `num_particles` | 1500 | |
| `slot_spacing` | 1.24 m | Planting spacing |
| `num_slots` / `exceed_slot_num` | 50 / true | |
| `template_origin_min_hits` | 6 | Detections needed before slot 0 is anchored |
| `min_seen_count_to_publish` | 6 | |
| `odom_history_sec` | 5.0 | |

### `odom_reframer_calibrated` (node name `odom_reframer`)

Preprocesses the robot's odometry into `/odometry/wheel` for the EKFs.

- **Startup averages.** On startup it averages the first N odometry and GPS readings and publishes them on the latched topics `/initial_odom_correction` and `/initial_gps_fix`.
- **Reframing.** It can recentre the origin and rotate the position and yaw.
- **Velocity from pose** (`twist_from_pose`). Computes body velocity from a pose window of `twist_from_pose_window` messages. The velocity covariance is σ = `frac`·|v| + `floor`.
- **Covariance override.** `pose_cov_xy` / `pose_cov_yaw` replace a zero or invalid pose covariance.
- **Timestamps.** `use_now_stamp` should stay `false`. Using receive time bunches the stamps together and fakes a 2–4% scale error when velocities are fused.

### `fix_cov_relay` (gps_heading mode)

Republishes `/fix` as `/fix_rtk` with a realistic covariance (`sigma_m`, default 0.05 m). Some receivers publish a placeholder covariance of about 1 m even when they have an RTK fix. `stamp_shift_s` compensates for a known GPS latency.

### `gps_course_heading` (gps_heading mode)

Computes heading from the GPS course over a `baseline_m` (1.5 m) of travel and publishes it as an `Imu` yaw on `/gps/course_imu` for `navsat_transform` and `ekf_global`. It uses `/odometry/wheel` to reject turns (`max_turn_deg`) and slow motion (`min_speed`).

### `save_best_particle_map`

Saves the SLAM state to YAML when the `save_slam_results` service is called (format above).

World correction (`world_correction.*`):

| Parameter | Default | Description |
|---|---|---|
| `enable` | true | |
| `source` | `gps` | `gps`: RTK fixes (`gps_fix_topic`, default `/fix_rtk`) minus the antenna lever arm (`antenna_x`), with heading = SLAM yaw + GPS-course offset. `global`: uses `/odometry/global` |
| `window_s` / `settle_s` | 10 / 5 | Averaging window around each tree's observation time; start-up settle time |
| `cam_fwd_m` | 0.73 | Camera forward offset used to find when each tree was observed |
| `write_separate_file` | true | Also write `<name>_world.yaml` |

If the GPS inputs are missing, world correction is skipped and the log gives the reason. The plain map is still saved.

### `slam_odom_correction_tf`

Publishes `map → odom` from the difference between the SLAM best pose and the odometry, synchronised in time. It is off in the default launch.

### `tree_template` (`generate_trellis_collision_obj`)

Turns `TrunkRegistry` into MoveIt `CollisionObject`s. Each tree is a leader cylinder plus V-trellis side branches, all configurable.

### `trellis_from_rgbd_extraction`

Publishes a `TrunkRegistry` from a folder of per-tree `meta.yaml` files made by `fast_slam_rgbd_extraction`, anchored to `anchor_tree_id`. It keeps the map in the local frame of the anchor tree and publishes it in the robot base frame of the selected base-camera image (it follows `/apple_prediction` `presaved_images.base_cam_frame_index`). Its `/correct_template_anchor` service corrects the whole map from the base-camera trunk detections (`correction_frames`: `current` or `buffer`). See [Templating from a saved map](#templating-from-a-saved-map).

### Other entry points

- `fast_slam_rgbd_extraction`: extracts the RGB-D frames where the robot is broadside to each mapped tree, from a bag plus a saved map (see [Step 1](#step-1-extract-the-rgb-d-frames-at-each-mapped-tree)).

---

## Evaluation and debugging

In this package:

- `plot_fastslam_debug.py` and `plot_slam_variance.py` plot the debug CSVs (`debug.csv_enable: true`). They read `tree_template/debug_data/` by default; `--csv` sets a different file. `plot_fastslam_debug.py` writes its PNGs next to the CSV (`--out_dir` to change, `--show` for windows).

Outside this package (scripts kept with the ground-truth data, not installed):

- `plot_nn_tree_error.py`, `compare_slam_methods.py` and `plot_utm_lat_lon.py` compare a saved `slam_trunks_*.yaml` with a ground-truth survey JSON.
- `plot_utm_lat_lon.py --align reg_anchor` does a rigid fit of the whole map anchored on one tree, instead of pinning the first tree.

The ground-truth map is only for tuning. A deployed robot has no ground truth, so judge map quality by its internal geometry: F1, spacing between neighbouring trees, and row length/scale.

---

## Known issues and notes

- **Use `/odometry/local` for SLAM.** On live runs, `/odometry/global` in `legacy` mode sometimes shortens the row by 3–8%, because GPS is fused with a heading that isn't referenced to north. Bag replays don't reproduce this. SLAM should use continuous local odometry, which is what `slam_odom_topic:=auto` picks. Use `gps_heading` mode plus the world-corrected map to get global placement.
- **Spacing error comes from odometry.** The camera sees about one trunk at a time, so the spacing between trees comes from odometry. The remaining along-row error (about 0.15–0.2 m) is systematic and can't be tuned away with filter parameters.
- **Clock offsets.** Each sensor stamps with its own clock. On the 2023 bags the camera is 0.4–0.8 s off the odometry, depending on the bag. Measurements use image stamps, so the camera and odometry clocks need to be synchronised, or the offset measured, on a new robot.
- **Machine-specific paths.** The shipped robot configs have absolute paths under `/home/marcus` for the map output and extraction folder. Edit them or use your own robot config. The detector paths default to `~/trunk_width_ws`.
- **2025 platform and gps_heading.** `config/nav_2025/` has no gps_heading EKF files, so with `nav_config:=nav_2025` use `gps_fusion_mode:=legacy`. In gps_heading mode the GPS nodes fail to load their missing params files.
- **Warthog TF tree** (`harvest_hardware_description/urdf/warthog_2023.urdf.xacro` in the apple-harvest repo, published by `harvest_hardware_description/launch/warthog_state_publisher.launch.py`):
  - It is generated from the 2023 bag `2023-10-17-16-32-29_267`: all 33 `/tf_static` transforms, plus the `/tf` values at the start of the bag. `base_link → cloud` and the diff units are constant; the four wheels spin, so they are frozen at their first value. `odom → base_link` (the robot pose) is left out.
  - The bag's root `base_link` is recreated as `warthog_base_link`, because the UR arm in the harvest URDF already has a `base_link`. Without this, the bag's chassis tree ends up hanging off the arm (`amiga__tool0 → base_link → chassis_link → imu_link`).
  - `warthog_base_link` sits `base_link_height` (default 0.288 m) above `warthog_base_footprint`, the ground-level root and the Warthog's base frame for odometry, the EKFs, the relay and SLAM (it replaces `amiga__base` for this platform). This puts the wheels on the ground. In the bag the axles are only 0.012 m above `base_link`, and the wheel radius is 0.3 m (Warthog nominal; 0.299 m measured from this bag as odometry path length ÷ wheel rotation).
  - IMU: `imu_link` at (0, 0, 0.313) from `warthog_base_footprint`, no rotation (`warthog_base_link` 0.288, `→ chassis_link` z 0.025, `→ imu_link` identity in the bag). Only its rotation matters to the EKF.
  - Not in the bag: `reach_rs` (GPS antenna) at (−0.5, 0, 2.25), not measured. The base camera (`/camera/*`, the camera the detector uses) is a RealSense D435i named `camera` (harvest `sensor_d435i` macro, nominal extrinsics, D435 mesh). Its `camera_link` is 1 in ahead of the front-right wheel (x 0.783), at that wheel's y (−0.568), at 3/4 of the 0.6 m wheel diameter (z 0.45), looking to the right. Set it with the xacro args `camera_x/y/z/yaw`. `cam_2_link` is only added with `mount_cam_2:=true`.
  - Play the bag with `--remap /tf:=/bag/tf /tf_static:=/bag/tf_static`. Otherwise the bag's own tree gives `chassis_link`, `imu_link` etc. a second parent.
- **Warthog vs the harvest URDF.** The Warthog tree has no `amiga__*` frames. The harvest URDF (`arm_control`) is rooted at `amiga__base` and is **not** connected to it, so its arm and cameras float separately in RViz. Collision objects use `trellis_frame: warthog_base_footprint`; MoveIt's planning frame is `amiga__base`, so connect the two (e.g. a static TF `warthog_base_footprint → amiga__base`) if you need the trellis templates in the arm's planning scene during Warthog replays.
- **Amiga sensor frames** (harvest `amiga.urdf.xacro`, Intelligence Kit block; the 2025 bags have no TF). The Amiga bringup (`arm_control`) or `publish_robot_description:=true` must run.
  - `amiga__intelligence_kit` at `ikit_xyz` / `ikit_yaw` (macro params; default: the "GPS & Oak cameras" box of the harvest URDF (center (0.57, −0.15, 1.6)), with oak0 0.01 m inside its +y face at y −0.06, looking along +y, parallel to the side of the Amiga).
  - `oak0`: 15° down. `oak0/imu`: x = camera up, y = camera left, z = camera back. This agrees with the 2025 bag (gravity, gyro yaw rate, forward acceleration; measured tilt 12.4°).
  - `gps_antenna`: 0.09465 m behind oak0 and 0.06485 m above it (kit geometry). `world_correction.antenna_x` in the Amiga robot config is its x (0.57).
- **Warthog camera frame.** The Warthog robot config uses `camera_color_optical_frame`, the frame the bag's base-camera images are stamped with. The D435i in `warthog_2023.urdf.xacro` publishes it at (0.798, −0.568, 0.45) from `warthog_base_footprint`. `world_correction.cam_fwd_m: 0.80` in the robot config matches its x.

---

## Interfaces

| Message | Package | Fields used |
|---|---|---|
| `TreeImageData` | `pf_orchard_interfaces` (external) | Per-detection pixels, mask xs/ys, depth, width, class, confidence; `header.stamp` = image time |
| `TrunkInfo` | `tree_template_interfaces` | `stamp`, base-frame `pose`, `width`, `side`, `cam_lateral`, `cam_depth` |
| `TrunkRegistry` | `tree_template_interfaces` | List of `TrunkInfo` (the map) |
| `SaveSlamResults` | `tree_template_interfaces` | `output_path`, `append_timestamp` → `success`, `message`, `written_path` |
| `UpdateTrellisPosition` | `tree_template_interfaces` | `pose`, `side` → `success` |
