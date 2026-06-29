# FR-9: Image Matching

## Requirement
> The system shall match query image ORB descriptors against the reference
> database using BFMatcher with Lowe ratio test, returning the best-matching
> database entry and confidence score.

## Rationale
Visual localisation depends on a reliable image-matching stage that can narrow a
live camera frame down to plausible reference tiles before geometric
verification and pose recovery run. The packaged runtime must support the
existing FLANN/LSH coarse retrieval path and the newer BoVW coarse retrieval
path while preserving the same downstream verification and localisation
behavior.

## Acceptance Criteria
1. The packaged localizer can perform coarse retrieval over the reference
   database and pass ordered candidate entries into geometric verification.
2. Coarse retrieval can be selected from config using `retrieval.mode` with
   support for both `flann` and `bovw`, while the legacy `retrieval.backend`
   alias remains accepted.
3. FLANN/LSH retrieval remains supported for existing databases and
   configurations.
4. BoVW retrieval works from the packaged `src/vns` path when the loaded
   `.vnsdb` contains a BoVW vocabulary and per-entry histograms.
5. If BoVW retrieval is selected but the database lacks its BoVW index, the
   packaged localizer fails with a clear error unless
   `retrieval.bovw.fallback_to_flann` is enabled.
6. Unit tests cover both retrieval modes and prove the packaged runtime path
   uses the configured retrieval backend without importing
   `simulation/scripts/visual_navigation.py`.

## Dependencies
FR-3, FR-8

## Linked Source Files
- `src/vns/vision/retrieval.py` — packaged coarse-retrieval backend selection and adapters.
- `src/vns/vision/bovw_retrieval.py` — reusable BoVW histogram retrieval implementation.
- `src/vns/vision/localizer.py` — packaged localizer orchestration that consumes retrieval candidates.
- `src/vns/database/reference_db.py` — persisted ORB descriptors, BoVW vocabulary, and histograms.
- `simulation/config/simulation.yaml` — packaged runtime retrieval mode configuration.

## Test Plan

### Unit Tests
- `tests/test_retrieval.py` — covers FLANN/LSH, BoVW, fallback, and BoVW
  missing-index failure behavior.
- `tests/test_localizer.py` — proves the packaged `VisualLocalizer` uses the
  selected coarse retrieval backend.

### Integration Tests
- `tests/test_runtime.py` — proves `VnsRuntime` activates the configured
  packaged retrieval mode.
- `tests/test_bovw.py` — verifies the BoVW index can be reconstructed from the
  persisted `.vnsdb` contents.

### Simulation Tests
- The packaged ROS runtime (`src/vns/core/vns_node.py` -> `VnsRuntime` ->
  `VisualLocalizer`) can be launched against a BoVW-enabled database without
  relying on the legacy `simulation/scripts/visual_navigation.py` path.

## Status
- [ ] Not Started
- [ ] In Progress
- [x] Done

**Implementation evidence:** The packaged retrieval path now supports both
FLANN/LSH and BoVW selection in `src/vns`, with regression coverage in
`tests/test_retrieval.py`, `tests/test_localizer.py`, `tests/test_runtime.py`,
and `tests/test_bovw.py`.
