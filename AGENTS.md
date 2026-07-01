# GPS-DENIED-SYSTEM Codex Instructions

## Project identity

This repository is `github.com/Aittrah/GPS-DENIED-SYSTEM`, a Final Year Project implementing a GNSS-Free Visual Navigation System for UAVs using PX4 SITL, Gazebo Classic, ROS 2 Humble, MAVSDK, and a VNS vision pipeline.

Team ownership:

* Muhammad Ahsan: Ubuntu, ROS 2, PX4 SITL, Gazebo simulation, VNS runtime.
* Aittrah Sardar: Windows MPVI desktop app side.

Current project status:

* Tests: 68/68 passing.
* Phase 2 is active.
* VNS vision pipeline, reference database, runtime orchestration, MAVLink interface, config system, CLI, simulation model/worlds, launch files, CI, and Claude Code plugin are already implemented.
* The canonical runtime is the packaged path `src/vns/core/vns_node.py` → `VnsRuntime` → `VisualLocalizer`; the legacy `simulation/scripts/visual_navigation.py` was removed (B-BOVW, B-DUAL resolved). See `docs/runtime_consolidation.md`.
* Bug B3 altitude mismatch has already been solved.
* Current target bug is B1: `iris_downward_cam` camera topic is hardcoded and cannot be remapped via launch args.

## Important environment rules

Use the existing repository environment. Do not switch the project to Gazebo Harmonic. This project uses Gazebo Classic.

Use `python3` and `pip3`, not bare `python` or `pip`.

Do not run commands from the wrong directory. Work from the repository root unless a task explicitly says otherwise.

Do not modify `.claude/plugin/` unless the task is explicitly about migrating or updating the Claude plugin.

Do not make large unrelated refactors. One task, one focused diff.

## Development workflow

Before editing files:

1. Inspect the relevant files.
2. Check current branch and git status.
3. Identify exact target files.
4. Explain the plan briefly.
5. Then edit only the files needed for the task.

For complex or multi-file work, produce a plan first and do not edit until the plan is approved.

For broad codebase research, explicitly spawn subagents only when requested in the task prompt.

After editing:

1. Run relevant unit tests.
2. Run the full pytest suite when the change affects shared code, launch behavior, config, runtime, simulation, or database code.
3. Show the final diff summary.
4. Report any commands that could not be run.

## Testing expectations

Default Python test command:

```bash
python3 -m pytest tests/ -q
```

For simulation changes, prefer lightweight verification first. Only run full PX4 SITL when explicitly requested.

## Current known pending tasks

P1: Open PR to main.
P2: Clean up untracked files:

* `SESSION_NOTES_orb-localization-fix.md`
* `simulation/database/qau_campus.vnsdb.legacy.bak`

P3: Full PX4 SITL in-flight test.
P4: Replace synthetic DB images with real satellite imagery.

## Current known bugs

B1: `simulation/models/iris_downward_cam/model.sdf` hardcodes camera topic behavior. The launch layer cannot remap it.

B2: CI only tests Python 3.11; runtime uses Python 3.10 with ROS Humble.

B4: README GPS denial scenario table mentions scenarios not actually implemented in `gps_gate_node.py`.

B5: Skill docs imply `capture_reference_images.py` produces real imagery, but it only generates synthetic data.

B6: Noisy no-PX4 warning in `localization_test.launch.py`.

## Commit style

Use conventional commits:

```text
fix(simulation): make downward camera topic remappable via launch arg
```

Include FR-ID only if the repository docs/specs clearly identify the correct FR. Do not invent an FR mapping.
