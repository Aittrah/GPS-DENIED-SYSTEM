# Open House Proof Bundle

WHAT WORKS:
- Desktop mission planner
- Test suite
- Satellite texture generation
- Real reference tile generation
- VNS database generation
- Dashboard generation
- Offline VNS database self-check, if passing

WHAT IS BLOCKED:
- Live ROS/Gazebo/PX4 simulation if ROS Humble is missing

WHY LIVE SIM MAY NOT RUN:
- ROS Humble setup not found at /opt/ros/humble/setup.bash, or other missing runtime dependency

WHAT TO OPEN DURING OPEN HOUSE:
- reports/openhouse_proof/test_output.txt
- reports/openhouse_proof/build_db_output.txt
- reports/openhouse_proof/offline_vns_check.txt
- reports/openhouse_proof/localization_diagnosis.txt
- simulation/models/qau_ground_plane/materials/textures/qau_satellite.png
- simulation/database/images_real/
- simulation/database/qau_campus.vnsdb
- reports/dashboard/index.html

HONEST DASHBOARD NOTE:
- Dashboard overlays are generated only from verified localization records.
- If overlays are 0, it means no verified localization_success records exist in the current logs.

Current shell note:
- ROS 2 base setup is missing from /opt/ros/humble/setup.bash.
- No ros2 executable was found anywhere on disk.
- The repo's venv packages live under python3.10 site-packages, but this shell only exposes Python 3.12.
- Stdlib-only diagnostics were used where possible; rebuild/test steps may fall back to artifact verification.
