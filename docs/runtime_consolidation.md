# VNS Runtime Consolidation

**Date:** 2026-06-30
**Resolves:** B-BOVW (BoVW retrieval built but unused on the default packaged path),
B-DUAL (two parallel runtime implementations that could drift)

## Background

The repository shipped two parallel VNS runtimes:

- **Path A — packaged/canonical:** `src/vns/core/vns_node.py` → `VnsRuntime`
  (`src/vns/core/runtime.py`) → `VisualLocalizer` (`src/vns/vision/localizer.py`).
  Used by all four launch files, the only console_script (`vns_node`), CI, the README,
  and the MAVLink/PX4 `VISION_POSITION_ESTIMATE` output. Also has preprocessing,
  dead-reckoning, IMU fusion, typed Pydantic config, diagnostics, and richer evaluation logging.
- **Path B — legacy script:** `simulation/scripts/visual_navigation.py` (self-contained
  `VnsNode` + `VnsRosNode`). Nothing launched it; it had no MAVLink, no preprocessing, no
  dead-reckoning. Only `simulation/scripts/test_run.py` imported it.

## Canonical path: A

Path A is canonical. It is the only path wired into launch files, packaging, CI, and the
MAVLink output consumed downstream by PX4 EKF2. Consolidating onto it loses no launcher or
output-format compatibility.

## What was already present (no porting needed)

Exploration found that two capabilities the original consolidation brief assumed lived only in
Path B were already implemented in Path A:

- **BoVW as a config-selectable retrieval backend.** `src/vns/vision/retrieval.py` already
  defined a `RetrievalBackend` ABC with `FlannLshBackend` and `BoVWRetrievalBackend`, selected by
  `RetrievalIndex.from_config` via `retrieval.mode: flann|bovw`. Config
  (`RetrievalConfig`/`RetrievalBoVWConfig`) and tests (`tests/test_retrieval.py`,
  `tests/test_localizer.py`, `tests/test_bovw.py`) already existed. BoVW and FLANN/LSH remain
  **selectable**, not one silently replacing the other.
- **Ground-truth logging.** `src/vns/validation/jsonl_logger.py` (`JsonlEvaluationLogger` +
  `build_evaluation_record`) already produced a **superset** of the script's JSONL fields, with
  the same `ground_truth_<epoch>.jsonl` filename scheme.

## What was ported in

The only genuine capability unique to Path B was **BoVW geo-gating**: intersecting the BoVW
appearance top-k with a geographic radius around a trustworthy position prior, with a
`query_region` fallback when appearance retrieval yields nothing usable. This was ported into the
ROS-free retrieval layer, additively and backward-compatibly:

- `RetrievalIndex` (`src/vns/vision/retrieval.py`) now stores the database reference and the
  `retrieval.bovw.geo_gate` / `geo_gate_radius_deg` config, and `query()` takes an optional
  `prior=(lat, lon)`. For the **BoVW backend only**, it post-filters candidates by radius (never
  returning empty — a stale prior falls back to the unfiltered top-k with a warning) and falls back
  to `ReferenceDatabase.query_region` when descriptors are empty. The FLANN path and all existing
  callers/tests are unaffected (default `prior=None`).
- `VisualLocalizer.localize()` (`src/vns/vision/localizer.py`) accepts an optional `prior` and
  forwards it to retrieval.
- `VnsRuntime` (`src/vns/core/runtime.py`) computes the prior via `_current_position_prior()`
  (a live non-denied GPS fix, else the most recent fused estimate, else `None`) and passes it into
  `localize()`.

All three edited files are ROS-free; no ROS2 imports were added, so Windows/MPVI importability is
preserved.

## Dead-reckoning bridge hardening

The canonical runtime now treats dead reckoning as a **bias-aware short bridge**, not an
open-ended inertial navigator:

- `DeadReckoning` (`src/vns/core/dead_reckoning.py`) accepts configurable gyro and
  accelerometer bias vectors, subtracts them before propagation, and tracks the timestamp of the
  last trusted correction together with a linearly decaying confidence.
- `VnsRuntime` (`src/vns/core/runtime.py`) instantiates that bias-aware DR state from the typed
  `dead_reckoning` config section and re-anchors it on every healthy GPS fix.
- `PositionBlender` (`src/vns/core/blender.py`) re-anchors DR on each accepted visual fix and only
  uses DR fallback while the bridge remains valid; after the configured limit, GNSS-denied +
  no-visual transitions to `FAILSAFE` instead of trusting stale inertial drift.

The packaged simulation defaults are intentionally conservative:

- `dead_reckoning.max_bridge_duration_seconds: 10.0`
- `dead_reckoning.confidence_decay_per_second: 0.1`
- zero configured bias vectors unless a calibration or test scenario provides better values

## BoVW now actually used (B-BOVW)

The **code default** stays `retrieval.mode: flann` (FLANN/LSH needs no prebuilt vocabulary and
works on any database). The **simulation config** (`simulation/config/simulation.yaml`) now sets
`retrieval.mode: bovw` with `retrieval.bovw.fallback_to_flann: true`, so the canonical runtime
runs BoVW + geo-gate where the legacy script did — degrading gracefully to FLANN/LSH if a database
lacks a vocabulary. This resolves the "built but unused" half of **B-BOVW**.

## Path B removed (B-DUAL)

- `simulation/scripts/visual_navigation.py` was **deleted**.
- `simulation/scripts/test_run.py` was **rewritten** to drive the canonical `VnsRuntime` directly
  (no ROS required) over the same 30-step GNSS-healthy → GNSS-denied trajectory, writing the
  packaged `JsonlEvaluationLogger` output. The offline flight-eval capability is preserved on the
  canonical path.
- Live pointers were updated (`.claude/plugin/skills/vns-environment/SKILL.md`,
  `.claude/plugin/commands/specs-from-srs.md`, `specs/FR-9.md`, `specs/FR-11.md`). Historical
  reports/assessments were left as-is.

## Note: evaluation logging is a superset, not byte-identical

The canonical `JsonlEvaluationLogger` record contains every field the deleted script logged
(`estimated_*`, `true_*`, `*_heading`, `horizontal_error`, `vertical_error`, `heading_error`,
`mode`, `timestamp`) **plus** additional fields (`ros_time`, `visual_*`, `position_error_m`,
`vision_confidence`, `localization_success`, `gnss_status`, `navigation_mode`, `failure_reason`,
`matched_ref_id`, `inlier_count`). This superset is intentional, not a regression;
`tests/test_runtime.py::test_canonical_jsonl_is_superset_of_legacy_gt_fields` enforces field coverage.

## Tests

- `tests/test_retrieval.py::TestRetrievalGeoGate` — geo-gate filtering, gate-off, no-prior,
  never-empty, region fallback, empty-without-prior, FLANN-ignores-prior.
- `tests/test_runtime.py` — `_current_position_prior` selection (GPS healthy / denied-fallback /
  none), FLANN vs BoVW+geo-gate parity, and the JSONL superset assertion.
- Full suite: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/ -q`.
