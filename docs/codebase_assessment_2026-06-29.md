# GPS-DENIED-SYSTEM Codebase Assessment

Date: 2026-06-29
Repository: `github.com/Aittrah/GPS-DENIED-SYSTEM`
Reviewed branch: `feat/claude-code-plugin` (ahead of `origin/feat/claude-code-plugin` by 5 commits)

## 1. Executive Summary

This codebase is a coherent Phase 2 GNSS-denied visual-navigation stack built around a Python package (`src/vns`), ROS 2 launch assets, Gazebo Classic simulation resources, and offline database tooling. The main runtime path is substantially implemented: configuration validation, safe reference-database loading, visual localization, GNSS-state monitoring, pose blending, MAVLink export, and ROS 2 node integration are all present and covered by automated tests.

The current baseline is healthy:

- `python3 -m pytest tests/ -q` passed locally
- Result: `78 passed, 1 warning`
- The committed simulation database is in the safe archive format and loads successfully
- The committed database contains 23 entries and an embedded BoVW vocabulary

The strongest parts of the repository are:

- clear package boundaries under `src/vns`
- solid regression coverage around previously fixed failures
- safe reference-database serialization and legacy migration support
- usable launch/test tooling for Gazebo Classic and ROS 2 Humble

The main risks are now less about missing core functionality and more about portability, configuration drift, and duplicated runtime paths.

## 2. Implemented System Architecture

### Core runtime

The packaged runtime is centered on:

- `src/vns/core/runtime.py`: stateful runtime wrapper for GPS updates, heading updates, frame processing, blending, and diagnostics
- `src/vns/core/vns_node.py`: ROS 2 node that wires camera/GPS/IMU/ground-truth topics to the runtime and forwards vision estimates over MAVLink
- `src/vns/core/gnss_monitor.py`: GNSS health state machine (`HEALTHY`, `DEGRADED`, `DENIED`)
- `src/vns/core/blender.py`: GPS/vision transition logic and failsafe selection
- `src/vns/core/diagnostics.py`: structured subsystem diagnostics snapshot

### Vision pipeline

The vision path is modular and understandable:

- `src/vns/vision/preprocessor.py`: resize, optional undistortion, grayscale, CLAHE
- `src/vns/vision/extractor.py`: ORB feature extraction
- `src/vns/vision/retrieval.py`: coarse FLANN/LSH retrieval over database descriptors
- `src/vns/vision/verification.py`: BFMatcher ratio test + homography RANSAC
- `src/vns/vision/pose_recovery.py`: pixel displacement to geodetic/NED pose
- `src/vns/vision/localizer.py`: six-stage orchestration of the full localization flow

### Database and tooling

Reference-database support is one of the most mature areas:

- `src/vns/database/reference_db.py`: safe archive format, validation, bounds checks, legacy pickle blocking, trusted migration path
- `simulation/scripts/build_reference_database.py`: offline database builder with ORB extraction and BoVW vocabulary generation
- `simulation/scripts/capture_reference_images.py`: synthetic reference-image generation and index creation
- `src/vns/cli.py`: config validation plus database build/inspect/migrate/capture commands

### Simulation and launch

The simulation stack is aligned with Gazebo Classic, not Harmonic:

- `simulation/launch/full_simulation.launch.py`: full Gazebo + PX4 SITL + VNS + GPS gate orchestration
- `simulation/launch/UAV_simulation.launch.py`: Gazebo + VNS without PX4
- `simulation/launch/localization_test.launch.py`: deterministic localization smoke path without PX4
- `simulation/scripts/gps_gate_node.py`: simple GPS relay/suppression gate for denied-mode testing
- `simulation/models/iris_downward_cam/model.sdf`: drone model with Gazebo ROS camera, IMU, GPS, and ground-truth plugins

## 3. Verification Snapshot

### Test baseline

Current collected suite:

- 78 tests collected
- 78 tests passed

Coverage is meaningful rather than superficial. It includes:

- configuration validation
- CLI behavior
- database security and migration
- FLANN retrieval
- BoVW histogram/index behavior
- localizer success/failure cases
- pose-recovery geometry
- preprocessing
- runtime diagnostics
- MAVLink interface behavior
- regression guards for ground-texture localization and altitude consistency

Notable regression tests:

- `tests/test_ground_texture_localization.py`: guards the previous `no_geometric_match` failure on the textured map
- `tests/test_db_altitude_consistency.py`: protects the earlier altitude-datum bug
- `tests/test_camera_topic.py`: validates downward-camera topic rendering/remapping behavior

### CI status in the repo

The repository currently includes:

- Python CI on 3.10 and 3.11
- a ROS Humble smoke workflow building the package with `colcon`

That means the older note that CI only covers Python 3.11 no longer matches the checked-out codebase.

## 4. Strengths

### 4.1 Clear package organization

`src/vns` is separated into config, core, database, interfaces, validation, vision, and utils. For a final-year project, this is above-average structure and makes the system easier to explain, test, and extend.

### 4.2 Security-minded database handling

`src/vns/database/reference_db.py` blocks insecure legacy pickle databases by default and forces either migration or an explicit trusted load path. That is a strong design choice and materially better than silently unpickling arbitrary data.

### 4.3 Regression discipline

The repository has moved beyond pure happy-path testing. There are explicit regression tests for previously observed localization and altitude issues, which lowers the risk of rediscovering the same failures during demo or integration work.

### 4.4 Launch-level camera topic remapping exists in this branch

The checked-out branch appears to contain a fix for the earlier downward-camera hardcoding problem:

- `src/vns/utils/camera_topic.py` renders a modified SDF camera plugin block
- `simulation/launch/full_simulation.launch.py` and `simulation/launch/UAV_simulation.launch.py` accept a `camera_topic` launch argument and rewrite the spawned SDF when overridden
- `tests/test_camera_topic.py` verifies the behavior against the real model SDF

This means the repository-level note calling B1 the current target bug does not reflect the current working tree.

## 5. Key Risks and Gaps

### 5.1 Portability is weak because simulation assets are tied to one machine path

Several launch/world/tooling files hardcode `/home/hp/GPS-DENIED-SYSTEM` and `.venv-run`, for example:

- `simulation/launch/full_simulation.launch.py`
- `simulation/launch/localization_test.launch.py`
- `simulation/worlds/uav_test_world.sdf`
- `simulation/worlds/uav_localization_test.sdf`
- `simulation/database/images/database_index.yaml`
- `simulation/scripts/build_ground_texture.py`

Impact:

- moving the repo to another machine or username will break launch behavior
- install-space portability is limited
- CI/container reuse is harder than it should be

This is the highest practical engineering risk in the current tree.

### 5.2 The primary packaged runtime does not consume the BoVW retrieval path

There is a mismatch between available capability and the default runtime path:

- `src/vns/config/models.py` exposes retrieval backend and `retrieval.bovw.*` config
- `simulation/scripts/build_reference_database.py` builds BoVW vocabulary/histograms
- `src/vns/vision/bovw_retrieval.py` implements the query-side index
- the committed database already contains BoVW data

But `src/vns/vision/localizer.py` currently constructs `RetrievalIndex()` directly and therefore uses FLANN/LSH coarse retrieval only. By contrast, `simulation/scripts/visual_navigation.py` still contains a BoVW-enabled retrieval path.

Impact:

- the main packaged runtime is not using the most scalable retrieval path that the project documents and tests
- retrieval configuration suggests flexibility that the runtime does not actually honor
- two retrieval implementations can drift semantically over time

### 5.3 There are two runtime/node paths with overlapping responsibilities

The repo has both:

- packaged runtime/node path: `src/vns/core/runtime.py` + `src/vns/core/vns_node.py`
- older script path: `simulation/scripts/visual_navigation.py`

The script path still includes logic not present in the packaged localizer, especially around BoVW usage and candidate gating. This duplication increases maintenance cost and creates ambiguity about which path is authoritative.

### 5.4 Default configuration models do not match the real simulation topic layout

`src/vns/config/models.py` defaults `ros.camera_topic` to `/vns_drone/camera`, while the actual simulation config uses `/vns_drone/downward_camera/image_raw`.

Impact:

- if the node starts with defaults or a missing config file, it will subscribe to the wrong camera topic
- the fallback behavior in `src/vns/core/vns_node.py` is therefore not safe for simulation use

### 5.5 Project status notes are stale in the current working tree

The local instructions claim:

- 68/68 tests passing
- B1 is still the current bug
- CI only tests Python 3.11

Observed on 2026-06-29:

- 78 tests pass
- camera-topic remapping appears implemented and tested in this branch
- CI covers Python 3.10 and 3.11, plus ROS Humble smoke

This is not a code bug, but it is a documentation/process gap that can mislead future contributors.

## 6. Recommended Next Steps

1. Remove hardcoded absolute repo paths from launch files, world assets, generated indexes, and texture tooling. This gives the biggest practical reliability gain.
2. Decide on one authoritative runtime path. Either retire `simulation/scripts/visual_navigation.py` or fold its BoVW behavior into the packaged `VisualLocalizer`.
3. Make `src/vns/vision/localizer.py` honor the configured retrieval backend so `retrieval.backend` and `retrieval.bovw.*` are real runtime controls, not mostly offline metadata.
4. Align default config model values with actual simulation topics so a missing config file degrades safely.
5. Refresh project-status notes to match the branch that is actually being used for development and reporting.

## 7. Overall Assessment

This is a solid final-year-project codebase with real engineering effort behind it, not just a prototype script collection. The core localization/runtime/database stack is implemented, test-backed, and understandable. The remaining work is mostly integration hardening: portability cleanup, runtime-path consolidation, and removing drift between documented capabilities and the behavior of the main packaged runtime.

## 8. Project Status vs Documentation Matrix

| Item | Documentation / notes state | Observed codebase state | Status |
|---|---|---|---|
| Test baseline | `AGENTS.md` says `68/68` tests passing | Local run passed with `78 passed, 1 warning` | Stale documentation |
| Camera-topic remap bug (B1) | `AGENTS.md` says downward camera topic cannot be remapped | Launch layer rewrites the spawned SDF via `src/vns/utils/camera_topic.py`; behavior is covered by `tests/test_camera_topic.py` | Appears implemented |
| CI Python coverage (B2) | `AGENTS.md` says CI only tests Python 3.11 | `.github/workflows/ci.yml` runs Python 3.10 and 3.11, plus a ROS Humble smoke job | Stale documentation |
| GNSS denial scenarios (B4) | `AGENTS.md` says README mentions unimplemented scenarios | README now explicitly says only the simple on/off gate is implemented and the named profiles in `gps_control.yaml` are future work | Appears resolved |
| Synthetic capture tooling (B5) | `AGENTS.md` says docs imply real imagery capture | `simulation/scripts/capture_reference_images.py` still generates synthetic images; `docs/database_creation.md` now labels simulation capture as synthetic | Mostly aligned, but hardware capture remains unimplemented |
| Database format warning | README warns older `.vnsdb` files must be migrated before default loading | The currently committed `simulation/database/qau_campus.vnsdb` already loads as a safe archive; warning still applies to legacy files in general | Documentation is broadly correct |
| Cleanup task P2 | `AGENTS.md` lists `SESSION_NOTES_orb-localization-fix.md` and `simulation/database/qau_campus.vnsdb.legacy.bak` for cleanup | Those files are not present in the current working tree | Stale task list |
| “Hardware ready” messaging | README presents the system as hardware-ready | Core architecture is hardware-oriented, but full PX4 SITL in-flight validation and real imagery replacement are still pending | Overstated maturity |

### Interpretation

- The implementation is ahead of the status notes in `AGENTS.md`.
- Public-facing documentation is in better shape than the local status note, especially for GNSS denial behavior.
- The remaining gap is less about missing core modules and more about validation maturity, portability, and documentation hygiene.

## 9. Task Register

| Task | Category | Current state | Priority | Recommended next action |
|---|---|---|---|---|
| Open PR to `main` | Project management | Still listed as pending in `AGENTS.md`; no evidence in the local tree that it has been done | Medium | Prepare branch summary and open the PR once status notes are refreshed |
| Full PX4 SITL in-flight validation | Validation | Still listed as pending; not verified in this assessment | High | Run an end-to-end flight scenario with PX4 SITL, Gazebo Classic, GPS denial, and VNS pose injection |
| Replace synthetic DB imagery with real imagery | Data quality | Still listed as pending; current capture script is synthetic-first | High | Define the source of real imagery, update the capture/build workflow, and rebuild `qau_campus.vnsdb` |
| Remove hardcoded absolute repo paths | Portability | Hardcoded `/home/hp/GPS-DENIED-SYSTEM` and `.venv-run` appear in launch files, world assets, and tooling | High | Refactor paths to derive from package/share locations or launch arguments |
| Unify the runtime path | Architecture | Packaged runtime and legacy script runtime overlap; BoVW-specific logic still lives in `simulation/scripts/visual_navigation.py` | High | Choose one authoritative runtime path and retire or fold the duplicate path |
| Honor retrieval backend config in packaged runtime | Runtime behavior | `src/vns/vision/localizer.py` always constructs `RetrievalIndex()` and does not use the richer retrieval config surface | Medium | Wire `retrieval.backend` and BoVW-aware retrieval into the packaged localizer |
| Align default ROS camera topic values | Configuration | Default model value is `/vns_drone/camera`, while the actual simulation config uses `/vns_drone/downward_camera/image_raw` | Medium | Change the default config model to match the real simulation topic layout |
| Refresh project-status notes and docs | Documentation | `AGENTS.md` is stale on tests, CI, B1, and cleanup tasks | High | Update the status block, bug list, and pending task list to match the current branch |
| Verify hardware-deployment guide commands and paths | Documentation | `docs/hardware_deployment.md` still uses `pip` and outdated config-path references | Medium | Update commands to `python3`/`pip3` and correct the repo-specific config paths |

### Done vs Pending Summary

| State | Items |
|---|---|
| Done / largely implemented | Core VNS pipeline, safe reference database format, CLI, launch files, Gazebo Classic integration, ROS 2 node, MAVLink interface, camera-topic remapping, regression tests |
| Pending validation | Full PX4 SITL in-flight test, stronger hardware-readiness proof |
| Pending productization | Real imagery workflow, portability cleanup, runtime-path consolidation |
| Stale notes to fix | `AGENTS.md` test count, B1, B2, cleanup task P2 |
