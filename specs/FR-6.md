# FR-6: Image Tiling and Patch Generation

## Requirement
> The system shall tile or crop UAV images into overlapping patches for
> hierarchical matching against the reference database.

## Rationale
Patch-based matching is the runtime path that satisfies the SRS intent behind
FR-6. The repository already contained a tested `PatchGenerator`, but it was
not wired into the packaged `VisualLocalizer` pipeline. This requirement closes
that gap by making patching an optional runtime mode that runs after
preprocessing, uses one shared configured patch size for both UAV and reference
imagery, and preserves the existing full-image path when disabled.

## Acceptance Criteria
1. **[Met]** Patching can be enabled or disabled through config, and the default
   remains full-image mode (`patching.enabled: false`).
2. **[Met]** Patch size is configurable through config rather than hardcoded in
   the live localization path.
3. **[Met]** When patching is enabled, `PatchGenerator` runs after
   preprocessing and before feature extraction, and the same configured patch
   size is applied to both UAV and satellite/reference images.
4. **[Met]** `VisualLocalizer` still produces valid match output when patching
   is enabled, including `success`, `confidence`, `matched_ref_id`, and
   success/failure `reason` fields.
5. **[Met]** Existing FLANN/LSH and BoVW retrieval behavior remains intact when
   patching is disabled, and the current full-image path remains behaviorally
   unchanged.

## Dependencies
FR-2, FR-5

## Linked Source Files
- `src/vns/config/models.py` - validated `patching` config model and defaults.
- `simulation/config/simulation.yaml` - runtime config surface for enabling patch mode and setting `patch_size`.
- `src/vns/preprocessing/patch_generator.py` - configurable patch generator used at runtime.
- `src/vns/core/models.py` - `ImagePatch` metadata used to map patch-local keypoints back to parent-image coordinates.
- `src/vns/vision/localizer.py` - live orchestration path that inserts patching after preprocessing and before extraction, then reuses the existing retrieval and verification interfaces.
- `tests/test_localizer.py`, `tests/conftest.py`, `tests/unit/test_preprocessing.py`, `tests/test_config_manager.py` - regression coverage for config-gated patch mode and patch-size propagation.

## Test Plan

### Unit Tests
- Verify config defaults and overrides for `patching.enabled` and `patching.patch_size`.
- Verify `PatchGenerator` honors configurable patch size, keeps half-patch overlap by default, and records parent-image offsets for coordinate translation.
- Verify patch mode is not invoked when disabled and is invoked when enabled.

### Integration Tests
- Run `VisualLocalizer` in default full-image mode and confirm the existing match output shape remains valid.
- Run `VisualLocalizer` in patch mode with source-backed reference images and confirm localization still succeeds with valid confidence and reason fields.
- Run `VisualLocalizer` in BoVW mode with patching enabled and confirm patch-derived descriptors still drive coarse retrieval successfully.

### Simulation Tests
- Run the packaged runtime or simulation launch path with `patching.enabled: true`
  and confirm localized output is still produced against the committed reference
  database while keeping the disabled path unchanged by default.

## Status
- [ ] Not Started
- [ ] In Progress
- [x] Done

**Implementation status:** Implemented.

**Implementation evidence:** `PatchGenerator` is now wired into
`src/vns/vision/localizer.py` as an optional runtime branch. When
`patching.enabled` is `true`, the localizer preprocesses the UAV frame first,
tiles it into overlapping patches, extracts ORB features per patch, and
aggregates those features back into the existing retrieval interface. Retrieved
reference candidates are resolved back to their source images, preprocessed,
patched with the same configured `patch_size`, converted into cached
patch-derived verification entries, and verified through the existing geometric
matcher. When `patching.enabled` is `false`, the prior full-image behavior is
preserved.

**Test evidence (2026-06-30):**
- `python3 -m pytest tests/test_config_manager.py tests/unit/test_preprocessing.py tests/test_localizer.py -q` passed (`40 passed`).
- `python3 -m pytest tests/test_runtime.py tests/test_retrieval.py tests/test_bovw.py tests/test_preprocessor.py tests/test_localizer.py tests/test_config_manager.py tests/unit/test_preprocessing.py -q` passed (`72 passed`).
- `python3 -m pytest tests/ -q` passed (`143 passed`).
