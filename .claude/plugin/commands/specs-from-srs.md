---
description: Bulk-generate spec stub files for all 17 VNS functional requirements (FR-1 through FR-17). Creates specs/<FR-ID>.md for each requirement. Skips files that already exist.
---

# VNS: Generate All Spec Stubs (FR-1 to FR-17)

Create stub spec files for every functional requirement of the Visual Navigation System SRS.

## Instructions

1. Create directory `/home/hp/GPS-DENIED-SYSTEM/specs/` if it does not exist.

2. For each FR in the table below, create `/home/hp/GPS-DENIED-SYSTEM/specs/<FR-ID>.md`.
   **Skip (do not overwrite) any file that already exists.**

3. Use the spec template at the end of this document for every file.

4. After creating all files, print a summary table showing FR-ID, Title, and Created/Skipped.

---

## FR Definitions

| FR-ID | Title | Requirement (shall statement) | Linked Source Files |
|-------|-------|-------------------------------|---------------------|
| FR-1 | Load Satellite Images | The system shall load and store geo-referenced satellite/aerial images of the QAU Campus operational area. | `simulation/scripts/capture_reference_images.py` |
| FR-2 | Preprocess Geographic Images | The system shall preprocess geographic reference images (resize, normalise, convert colour space) to prepare them for feature extraction. | `simulation/scripts/capture_reference_images.py` |
| FR-3 | GeoTIFF Reference Creation | The system shall create a structured reference database that maps pixel coordinates to geographic coordinates, backed by YAML metadata. | `simulation/database/qau_reference_metadata.yaml` |
| FR-4 | UAV Sensor Data Acquisition | The system shall acquire real-time sensor data from the UAV: camera imagery, IMU measurements, and GPS fix (when available), via the MAVLink interface and Gazebo Classic SDF plugins. | `simulation/models/uav_model/model.sdf`, `src/vns/interfaces/mavlink_interface.py` |
| FR-5 | UAV Image Preprocessing | The system shall preprocess UAV camera frames (undistort using calibration parameters, normalise, resize to match database resolution) before feature extraction. | `src/vns/vision/` |
| FR-6 | Image Tiling and Patch Generation | The system shall tile or crop UAV images into overlapping patches for hierarchical matching against the reference database. | `src/vns/vision/` |
| FR-7 | Sensor Calibration | The system shall apply camera intrinsic calibration (fx=554.25, fy=554.25, cx=320.0, cy=240.0, no distortion in simulation) and support IMU bias corrections. | `src/vns/interfaces/` |
| FR-8 | Feature Extraction | The system shall extract ORB keypoints and descriptors (up to 500 features per image) from both reference and query images. | `src/vns/vision/`, `simulation/scripts/build_reference_database.py` |
| FR-9 | Image Matching | The system shall match query image ORB descriptors against the reference database using BFMatcher with Lowe ratio test, returning the best-matching database entry and confidence score. | `src/vns/vision/`, `src/vns/database/` |
| FR-10 | Dead Reckoning Navigation | The system shall maintain a dead-reckoning position estimate by integrating IMU measurements when neither GPS nor visual matches are available. | `src/vns/core/` |
| FR-11 | Visual-Based Localisation | The system shall compute a geographic position estimate from matched image pairs using homography or PnP solving, with uncertainty quantification. | `src/vns/vision/`, `src/vns/core/` |
| FR-12 | Sensor Fusion | The system shall fuse GPS (when available), visual localisation, and dead-reckoning estimates using a weighted blend or Kalman filter with configurable fusion duration. | `src/vns/core/` |
| FR-13 | Dataset Preparation | The system shall provide tooling to build and validate the ORB feature reference database from captured or synthetic imagery. | `simulation/scripts/build_reference_database.py` |
| FR-14 | Model Training | The system shall support training or fine-tuning learned components (e.g. confidence estimators) on the QAU campus dataset using python3. | `src/vns/core/` |
| FR-15 | Model Evaluation | The system shall evaluate navigation accuracy against ground truth and generate accuracy reports including position error statistics and match confidence histograms. | `simulation/scripts/generate_accuracy_report.py`, `src/vns/validation/` |
| FR-16 | Visualisation | The system shall visualise the navigation state, matched reference images, feature correspondences, and position error in real-time or post-hoc. | `simulation/scripts/visual_navigation.py`, `src/vns/utils/` |
| FR-17 | Error Handling | The system shall detect and gracefully handle errors including lost visual lock, sensor dropouts, database misses, and navigation divergence, with configurable failsafe behaviours. | `src/vns/core/`, `src/vns/interfaces/` |

---

## Spec Template (apply to every FR above)

Substitute `<FR-ID>`, `<Title>`, `<requirement>`, `<linked files>` from the table.  Fill in Acceptance Criteria, Dependencies, and Test Plan from the requirement semantics.

```markdown
# <FR-ID>: <Title>

## Requirement
> <Requirement from table>

## Rationale
<Derive from the requirement context — what problem does this solve?>

## Acceptance Criteria
1. <Testable criterion derived from requirement>
2. <Second testable criterion>
3. <Third testable criterion>

## Dependencies
<Other FR-IDs or "None">

## Linked Source Files
<From the table — one line per file with role description>

## Test Plan

### Unit Tests
- <Unit test approach for this FR>

### Integration Tests
- <Integration test approach — e.g. run against qau_campus.vnsdb>

### Simulation Tests
- <Gazebo Classic + ROS 2 simulation test approach>

## Status
- [ ] Not Started
- [ ] In Progress
- [ ] Done
```

---

## Dependency Reference (use when filling Dependencies section)

| FR | Depends on |
|----|-----------|
| FR-2 | FR-1 |
| FR-3 | FR-1, FR-2 |
| FR-5 | FR-4 |
| FR-6 | FR-5 |
| FR-8 | FR-2, FR-5 |
| FR-9 | FR-3, FR-8 |
| FR-10 | FR-4, FR-7 |
| FR-11 | FR-6, FR-9 |
| FR-12 | FR-10, FR-11 |
| FR-13 | FR-1, FR-2, FR-8 |
| FR-14 | FR-13 |
| FR-15 | FR-12, FR-14 |
| FR-16 | FR-12 |
| FR-17 | FR-4, FR-12 |
