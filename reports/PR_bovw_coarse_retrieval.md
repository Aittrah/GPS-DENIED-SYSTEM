# Add BoVW coarse-retrieval index with hybrid geo gating

> Base: `main` · Head: `feat/claude-code-plugin`
> Compare/open: https://github.com/Aittrah/GPS-DENIED-SYSTEM/compare/main...feat/claude-code-plugin

## Summary

Adds a **Bag of Visual Words (BoVW)** coarse-retrieval layer to the GNSS-denied Visual Navigation System so reference lookup is O(1)-ish top-k instead of O(N) brute-force descriptor matching. ORB stays the extractor; this is an index built **on top of** the descriptors already computed.

> **Scope note:** this branch is 7 commits ahead of `main`. The headline change here is the BoVW index (latest commit `183ad85`); earlier commits on the branch cover the PX4/Gazebo airframe, GNSS-denial toggle, the ROS 2 node + MAVLink VISION_POSITION_ESTIMATE, and the FLANN/LSH localization pipeline.

## BoVW feature (`183ad85`)

**Offline — `build_reference_database.py`**
- Trains a `MiniBatchKMeans` vocabulary (K=1000, configurable, **clamped** for tiny DBs, **seeded** for reproducibility). ORB descriptors are binary/Hamming; Euclidean k-means is the documented, accepted approximation.
- Builds per-image L2-normalized histograms; persists vocabulary + histograms in the `.vnsdb` pickle (**version 1.1.0**). Legacy DBs without a BoVW block still load.
- New flags: `--build-vocab` / `--no-vocab`, `--vocab-size`, `--metric`, `--random-state`, `--stats` (prints sub-ms query timing).

**Online — `src/vns/vision/bovw_retrieval.py` (new)**
- `compute_bovw_histogram(...)` — the **single shared** assign+normalize routine used by both offline and online paths, so they can't diverge.
- `BoVWIndex` — `load()` rebuilds the index from the `.vnsdb`; `query()` uses a vectorized numpy top-k (**~0.65 ms**), with a `sklearn` `NearestNeighbors` index fit per spec (`query_sklearn()` for parity).

**Wiring — `visual_navigation.py`**
- BoVW is the primary coarse-retrieval step, with graceful fallback to the geographic radius search for legacy DBs / empty descriptors.
- **Hybrid geo gate:** intersects the appearance top-k with a radius around the position prior — but **only when a trustworthy prior exists** (live GPS fix or recent visual estimate, never the default origin), and falls back to the unfiltered top-k if it would discard everything.

**Config — `simulation.yaml`**
- `retrieval.bovw`: `enabled`, `vocab_size`, `metric`, `random_state`, `batch_size`, `geo_gate`, `geo_gate_radius_deg`. Config authority: K/metric are read from the persisted DB at query time; yaml supplies build-time defaults + `top_k`.

## Tests
`tests/test_bovw.py` — histogram normalization/determinism, K-clamping, self-query → #1 candidate, vectorized↔sklearn parity, legacy-load error. **Full suite: 42 passed.**

🤖 Generated with [Claude Code](https://claude.com/claude-code)
