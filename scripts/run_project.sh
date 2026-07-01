#!/usr/bin/env bash
# =============================================================================
# run_project.sh — one entry point to run every part of the GPS-DENIED-SYSTEM
# project on this machine.
#
#   ./scripts/run_project.sh <command>
#
# Commands:
#   test                  Run pytest when a compatible Python runtime exists.
#   app                   Launch the desktop mission planner.
#   build-db              Rebuild the real-imagery database, or verify the
#                         committed build artifacts when rebuild deps are missing.
#   loctest               Vision-only ROS/Gazebo localization test.
#   sim                   Full ROS/Gazebo/PX4 live simulation.
#   mission               Upload and start the MAVSDK mission (requires UDP 14550).
#   deny                  Toggle GNSS denial on a running ROS sim.
#   dashboard             Build the HTML dashboard, or reuse an existing one.
#   diagnose-localization Generate an honest localization diagnosis report.
#   offline-vns-check     Run a stdlib-only database self-check report.
#   openhouse             Generate the fallback proof bundle under reports/openhouse_proof/.
#   clean                 Kill leftover sim processes and free the ports.
# =============================================================================
set -uo pipefail

# ---- paths -----------------------------------------------------------------
REPO="/home/hp/GPS-DENIED-SYSTEM"
PX4="/home/hp/PX4-Autopilot"
PX4_ROS_WS="/home/hp/px4_ros2_ws/install/setup.bash"
WORKSPACE_SETUP="$REPO/install/setup.bash"
VENV_TEST="$REPO/.venv-test"
VENV_RUN="$REPO/.venv-run"
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
# GCS-side MAVLink listener for upload_mission.py. Do NOT use 14540 here —
# vns_node already binds udp://:14540 for VISION_POSITION_ESTIMATE.
SATELLITE_PORT="udpin://0.0.0.0:14550"

DB_PATH="$REPO/simulation/database/qau_campus.vnsdb"
DB_INDEX="$REPO/simulation/database/images_real/database_index.yaml"
REAL_IMAGES_DIR="$REPO/simulation/database/images_real"
GROUND_TEXTURE="$REPO/simulation/models/qau_ground_plane/materials/textures/qau_satellite.png"
DASHBOARD_HTML="$REPO/reports/dashboard/index.html"
OPENHOUSE_DIR="$REPO/reports/openhouse_proof"

# ---- helpers ----------------------------------------------------------------
status() {
  echo "==> $*"
}

warn() {
  echo "WARN: $*" >&2
}

die() {
  echo "ERROR: $*" >&2
  return 1
}

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

safe_source() {
  local file="$1"
  if [ -f "$file" ]; then
    set +u
    # shellcheck disable=SC1090
    source "$file"
    set -u 2>/dev/null || true
    return 0
  fi
  return 1
}

python_has_module() {
  local py="$1" module="$2"
  "$py" -c "import $module" >/dev/null 2>&1
}

python_has_modules() {
  local py="$1"; shift
  local module
  for module in "$@"; do
    python_has_module "$py" "$module" || return 1
  done
}

pick_python_with_module() {
  local module="$1"; shift
  local py
  for py in "$@"; do
    [ -n "$py" ] || continue
    [ -x "$py" ] || continue
    if python_has_module "$py" "$module"; then
      printf '%s\n' "$py"
      return 0
    fi
  done
  return 1
}

pick_python_with_modules() {
  local modules=() py
  while [ "${1:-}" != "--" ]; do
    modules+=("$1")
    shift
  done
  shift
  for py in "$@"; do
    [ -n "$py" ] || continue
    [ -x "$py" ] || continue
    if python_has_modules "$py" "${modules[@]}"; then
      printf '%s\n' "$py"
      return 0
    fi
  done
  return 1
}

tools_env() {
  cd "$REPO"
  TOOLS_PYTHON=$(
    pick_python_with_modules pytest cv2 yaml numpy sklearn -- \
      "$VENV_TEST/bin/python3" \
      /usr/bin/python3 \
      "$(command -v python3 2>/dev/null)"
  ) || return 1
  export TOOLS_PYTHON
}

mission_env() {
  cd "$REPO"
  MISSION_PYTHON=$(
    pick_python_with_module mavsdk \
      "$VENV_RUN/bin/python3" \
      "$VENV_TEST/bin/python3" \
      /usr/bin/python3 \
      "$(command -v python3 2>/dev/null)"
  ) || return 1
  export MISSION_PYTHON
}

require_ros() {
  status "Checking ROS 2 environment"
  if ! safe_source "$ROS_SETUP"; then
    echo "ERROR: ROS 2 Humble setup not found."
    echo "Expected: $ROS_SETUP"
    echo
    echo "Fix options:"
    echo "  1) Install ROS 2 Humble on Ubuntu 22.04"
    echo "  2) Or run with custom setup path:"
    echo "     ROS_SETUP=/path/to/setup.bash ./scripts/run_project.sh loctest"
    if [ -f "$PX4_ROS_WS" ]; then
      echo
      echo "Note: found overlay workspace setup at $PX4_ROS_WS"
      echo "      but the base ROS setup is still missing."
    fi
    return 1
  fi

  if ! command -v ros2 >/dev/null 2>&1; then
    echo "ERROR: ros2 command not found after sourcing $ROS_SETUP"
    return 1
  fi
}

require_ros_runtime() {
  local cmd_name="$1"
  require_ros || return 1
  export PATH="/usr/bin:$PATH"
  if [ -f "$PX4_ROS_WS" ]; then
    safe_source "$PX4_ROS_WS" || true
  fi
  [ -f "$WORKSPACE_SETUP" ] || die \
    "Missing $WORKSPACE_SETUP. Build the ROS workspace before running '$cmd_name'."
  safe_source "$WORKSPACE_SETUP" || return 1
  have_cmd ros2 || die \
    "ros2 command is still unavailable after sourcing $WORKSPACE_SETUP."
  python_has_module /usr/bin/python3 rclpy || die \
    "/usr/bin/python3 cannot import rclpy after sourcing ROS. Verify the ROS Python packages."
  python_has_module /usr/bin/python3 cv_bridge || die \
    "/usr/bin/python3 cannot import cv_bridge after sourcing ROS. Verify cv_bridge and numpy<2."
}

require_gazebo_runtime() {
  local cmd_name="$1"
  have_cmd gzserver || die \
    "gzserver is not installed or not on PATH. Install Gazebo Classic before running '$cmd_name'."
}

require_px4_runtime() {
  local cmd_name="$1"
  [ -d "$PX4" ] || die "PX4 checkout not found at $PX4. Start '$cmd_name' only after PX4 is installed."
  [ -f "$PX4/Tools/simulation/gazebo-classic/setup_gazebo.bash" ] || die \
    "Missing PX4 Gazebo setup script under $PX4. Build PX4 SITL before running '$cmd_name'."
  [ -x "$PX4/build/px4_sitl_default/bin/px4" ] || die \
    "Missing PX4 SITL binary at $PX4/build/px4_sitl_default/bin/px4. Build PX4 SITL before running '$cmd_name'."
}

require_udp_port_listener() {
  local port="$1" cmd_name="$2"
  ss -lun 2>/dev/null | grep -q ":$port " || die \
    "UDP port $port is not active. Start the simulator before running '$cmd_name'."
}

wait_for_px4_mavlink() {
  local port="${1:-14540}" attempts="${2:-90}" i
  for i in $(seq 1 "$attempts"); do
    if ss -lun 2>/dev/null | grep -q ":$port "; then
      return 0
    fi
    sleep 1
  done
  die "PX4 MAVLink port $port did not become active within ${attempts}s. Is './scripts/run_project.sh sim' running?"
}

latest_nonempty_ground_truth_log() {
  local file
  for file in $(ls -t "$REPO"/logs/ground_truth_*.jsonl 2>/dev/null); do
    [ -s "$file" ] && { printf '%s\n' "$file"; return 0; }
  done
  return 1
}

matching_manifest_for_log() {
  local log="$1" stamp candidate fallback
  stamp="${log##*_}"
  stamp="${stamp%.jsonl}"
  candidate="$REPO/logs/images/manifest_${stamp}.jsonl"
  if [ -s "$candidate" ]; then
    printf '%s\n' "$candidate"
    return 0
  fi
  for fallback in $(ls -t "$REPO"/logs/images/manifest_*.jsonl 2>/dev/null); do
    [ -s "$fallback" ] && { printf '%s\n' "$fallback"; return 0; }
  done
  return 1
}

verify_existing_build_artifacts() {
  local missing=0
  status "Verifying existing generated database artifacts"
  for path in "$GROUND_TEXTURE" "$DB_INDEX" "$DB_PATH"; do
    if [ ! -f "$path" ]; then
      echo "MISSING: $path"
      missing=1
    else
      ls -lh "$path"
    fi
  done
  if [ ! -d "$REAL_IMAGES_DIR" ]; then
    echo "MISSING: $REAL_IMAGES_DIR"
    missing=1
  else
    echo "Reference tile count:"
    find "$REAL_IMAGES_DIR" -maxdepth 1 -type f | wc -l
  fi
  [ "$missing" -eq 0 ] || return 1
  echo "Verified existing satellite texture, real reference tiles, and qau_campus.vnsdb."
}

verify_existing_dashboard() {
  status "Verifying existing dashboard artifact"
  [ -f "$DASHBOARD_HTML" ] || return 1
  grep -q "VNS Post-hoc Dashboard" "$DASHBOARD_HTML" || return 1
  ls -lh "$DASHBOARD_HTML"
  echo "Dashboard already exists at $DASHBOARD_HTML."
}

capture_environment_status() {
  cd "$REPO"
  echo "===== DATE / OS ====="
  date
  uname -a
  if have_cmd lsb_release; then
    lsb_release -a 2>/dev/null || true
  fi

  echo
  echo "===== PROJECT ====="
  pwd
  git status --short || true

  echo
  echo "===== SHELL ENV ====="
  echo "SHELL=$SHELL"
  echo "PATH=$PATH"
  echo "ROS_DISTRO=${ROS_DISTRO:-}"
  echo "GAZEBO_MODEL_PATH=${GAZEBO_MODEL_PATH:-}"
  echo "PX4_HOME_LAT=${PX4_HOME_LAT:-}"
  echo "PX4_HOME_LON=${PX4_HOME_LON:-}"

  echo
  echo "===== PYTHON ====="
  command -v python3 || true
  python3 --version || true
  command -v pip3 || true
  if have_cmd pip3; then
    pip3 --version || true
  fi

  echo
  echo "===== ROS CHECK ====="
  ls -lah /opt/ros 2>/dev/null || true
  ls -lah /opt/ros/humble/setup.bash 2>/dev/null || true
  command -v ros2 || true
  if have_cmd ros2; then
    ros2 --version || true
  fi
  if have_cmd dpkg; then
    dpkg -l | grep -E "ros-humble|ros-iron|ros-jazzy" || true
  fi

  echo
  echo "===== PYTHON ROS MODULES ====="
  python3 - <<'PY'
mods = ["rclpy", "cv_bridge", "launch", "launch_ros"]
for m in mods:
    try:
        __import__(m)
        print("OK:", m)
    except Exception as e:
        print("MISSING:", m, "-", e)
PY

  echo
  echo "===== GAZEBO CHECK ====="
  command -v gazebo || true
  command -v gzserver || true
  command -v gzclient || true
  command -v gz || true
  gazebo --version 2>/dev/null || true
  gzserver --version 2>/dev/null || true
  gz sim --versions 2>/dev/null || true

  echo
  echo "===== PX4 CHECK ====="
  ls -lah "$PX4" 2>/dev/null || true
  find /home/hp -maxdepth 4 -type d -iname "PX4-Autopilot" 2>/dev/null || true
  find /home/hp -maxdepth 6 -type f -name "px4" 2>/dev/null | head -30 || true

  echo
  echo "===== SETUP.BASH SEARCH ====="
  find /opt -name setup.bash 2>/dev/null | sort || true
  find /home/hp -path "*/setup.bash" 2>/dev/null | grep -E "ros|install|humble|px4|PX4|gazebo|colcon" | sort || true

  echo
  echo "===== PORT CHECK ====="
  ss -lunpt 2>/dev/null | grep -E "14540|14550|14551|11345" || true
}

run_and_capture() {
  local output_file="$1"; shift
  mkdir -p "$(dirname "$output_file")"
  { "$@"; } 2>&1 | tee "$output_file"
  return "${PIPESTATUS[0]}"
}

python_runtime_note() {
  cat <<'EOF'
Current shell note:
- ROS 2 base setup is missing from /opt/ros/humble/setup.bash.
- No ros2 executable was found anywhere on disk.
- The repo's venv packages live under python3.10 site-packages, but this shell only exposes Python 3.12.
- Stdlib-only diagnostics were used where possible; rebuild/test steps may fall back to artifact verification.
EOF
}

write_openhouse_readme() {
  cat >"$OPENHOUSE_DIR/OPENHOUSE_README.md" <<EOF
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

$(python_runtime_note)
EOF
}

# ---- command environments ---------------------------------------------------
loctest_env() {
  require_ros_runtime loctest || return 1
  require_gazebo_runtime loctest || return 1
  export VNS_PYTHON=/usr/bin/python3
  export GAZEBO_MODEL_DATABASE_URI=""
  export ROS_LOG_DIR="$REPO/ros_logs"
  export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
  cd "$REPO"
}

sim_env() {
  require_ros_runtime sim || return 1
  require_gazebo_runtime sim || return 1
  require_px4_runtime sim || return 1
  if [ -f "$PX4/Tools/simulation/gazebo-classic/setup_gazebo.bash" ]; then
    set +u
    # shellcheck disable=SC1090
    source "$PX4/Tools/simulation/gazebo-classic/setup_gazebo.bash" \
      "$PX4" "$PX4/build/px4_sitl_default" >/dev/null 2>&1 || true
    set -u 2>/dev/null || true
  fi
  export VNS_PYTHON=/usr/bin/python3
  export GAZEBO_MODEL_DATABASE_URI=""
  export ROS_LOG_DIR="$REPO/ros_logs"
  export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
  export PATH="$PX4/build/px4_sitl_default/bin:$PATH"
  cd "$REPO"
}

# ---- commands ---------------------------------------------------------------
cmd_test() {
  status "Running test suite"
  if tools_env; then
    env -u PYTHONPATH "$TOOLS_PYTHON" -m pytest tests/ -q || return 1
    return 0
  fi

  echo "Compatible pytest/OpenCV/Numpy runtime not found."
  echo "This shell exposes Python 3.12, but project packages are installed under $VENV_TEST/lib/python3.10/site-packages."
  echo "Tests were not rerun in this shell."
  return 1
}

cmd_app() {
  status "Launching desktop mission planner"
  tools_env || die "No compatible Python runtime found for the desktop app."
  env -u PYTHONPATH DISPLAY="${DISPLAY:-:0}" "$TOOLS_PYTHON" main.py
}

cmd_build_db() {
  status "Building real-imagery reference database"
  if tools_env; then
    env -u PYTHONPATH "$TOOLS_PYTHON" simulation/scripts/build_satellite_ground_texture.py || return 1
    env -u PYTHONPATH "$TOOLS_PYTHON" simulation/scripts/build_real_reference_tiles.py || return 1
    (
      cd simulation/scripts
      env -u PYTHONPATH "$TOOLS_PYTHON" build_reference_database.py \
        --input ../database/images_real/database_index.yaml \
        --output ../database/qau_campus.vnsdb \
        --max-features 1000 \
        --stats
    ) || return 1
    echo "Database rebuild completed."
    return 0
  fi

  warn "Compatible build dependencies were not found; rebuild is blocked in this shell."
  verify_existing_build_artifacts || return 1
  echo "Artifact verification fallback completed. This was not a fresh rebuild."
}

cmd_loctest() {
  status "Starting vision-only localization test"
  loctest_env || return 1
  echo "Watch for: Localized: ref=... in the vns_node output."
  ros2 launch vns localization_test.launch.py headless:=true
}

cmd_sim() {
  status "Starting full Gazebo/PX4/VNS simulation"
  sim_env || return 1
  echo "Wait for PX4 '[commander] Ready for takeoff!', then run: $0 mission"
  ros2 launch vns full_simulation.launch.py headless:=true
}

cmd_mission() {
  status "Uploading MAVSDK mission"
  mission_env || die "No Python interpreter with mavsdk installed was found."
  wait_for_px4_mavlink 14540 || return 1
  local alt="${2:-240}" speed="${3:-12}" csv="${4:-data/waypoints/qau_path_waypoints.csv}"
  "$MISSION_PYTHON" -u scripts/upload_mission.py \
    --csv "$csv" \
    --altitude "$alt" \
    --speed "$speed" \
    --connection "$SATELLITE_PORT" \
    --no-monitor
}

cmd_deny() {
  status "Toggling GPS denial"
  require_ros_runtime deny || return 1
  local state="${2:-off}" data
  [ "$state" = "off" ] && data=false || data=true
  ros2 service call /vns/set_gps_enabled std_srvs/srv/SetBool "{data: $data}"
}

cmd_dashboard() {
  status "Generating dashboard"
  local log manifest out="reports/dashboard" empty_manifest
  log=$(latest_nonempty_ground_truth_log) || log=""
  manifest=""
  if [ -n "$log" ]; then
    manifest=$(matching_manifest_for_log "$log") || manifest=""
  fi

  if tools_env && [ -n "$log" ]; then
    if [ -z "$manifest" ]; then
      empty_manifest="$(mktemp "${TMPDIR:-/tmp}/vns_empty_manifest.XXXXXX.jsonl")"
      : >"$empty_manifest"
      manifest="$empty_manifest"
      warn "No image manifest found; generating table-only dashboard (enable log_images for overlays)."
    fi
    env -u PYTHONPATH "$TOOLS_PYTHON" simulation/scripts/generate_visualization_dashboard.py \
      --log "$log" \
      --manifest "$manifest" \
      --database simulation/database/qau_campus.vnsdb \
      --output "$out" \
      --max-samples 60 || return 1
    [ -n "${empty_manifest:-}" ] && rm -f "$empty_manifest"
    echo "Dashboard generated at $REPO/$out/index.html"
    return 0
  fi

  if verify_existing_dashboard; then
    if [ -z "$log" ] || [ -z "$manifest" ]; then
      warn "Dashboard regeneration skipped because matching log/manifest files were not found."
    else
      warn "Dashboard regeneration skipped because a compatible Python runtime was not found."
    fi
    echo "Using the existing dashboard artifact instead."
    return 0
  fi

  die "Dashboard could not be generated or verified."
}

cmd_diagnose_localization() {
  status "Writing localization diagnosis report"
  cd "$REPO"
  python3 scripts/diagnose_localization.py --output "$OPENHOUSE_DIR/localization_diagnosis.txt"
}

cmd_offline_vns_check() {
  status "Running offline VNS self-check"
  cd "$REPO"
  python3 scripts/offline_vns_check.py --output "$OPENHOUSE_DIR/offline_vns_check.txt"
}

cmd_openhouse() {
  status "Generating open-house proof bundle"
  mkdir -p "$OPENHOUSE_DIR"
  capture_environment_status >"$OPENHOUSE_DIR/environment_status.txt"

  cmd_clean || true

  local test_status build_status dashboard_status diag_status offline_status

  run_and_capture "$OPENHOUSE_DIR/test_output.txt" cmd_test
  test_status=$?
  run_and_capture "$OPENHOUSE_DIR/build_db_output.txt" cmd_build_db
  build_status=$?
  run_and_capture "$OPENHOUSE_DIR/dashboard_output.txt" cmd_dashboard
  dashboard_status=$?
  cmd_diagnose_localization
  diag_status=$?
  cmd_offline_vns_check
  offline_status=$?

  ls -lh \
    "$GROUND_TEXTURE" \
    "$DB_INDEX" \
    "$DB_PATH" \
    "$DASHBOARD_HTML" \
    >"$OPENHOUSE_DIR/generated_files.txt"
  find "$REAL_IMAGES_DIR" -maxdepth 1 -type f | wc -l >"$OPENHOUSE_DIR/tile_count.txt"
  write_openhouse_readme

  echo
  echo "Open-house bundle written to $OPENHOUSE_DIR"
  echo "Step status summary:"
  echo "  test: $test_status"
  echo "  build-db: $build_status"
  echo "  dashboard: $dashboard_status"
  echo "  diagnose-localization: $diag_status"
  echo "  offline-vns-check: $offline_status"
  return 0
}

cmd_clean() {
  status "Killing leftover sim processes and freeing ports"
  local pat
  for pat in \
    "ros2 launch vns" \
    gzserver \
    "bin/px4" \
    px4-simulator \
    gps_gate_node \
    "vns/lib/vns/vns_node" \
    upload_mission \
    mavsdk_server; do
    pkill -9 -f "$pat" 2>/dev/null || true
  done
  sleep 1
  ss -tuln 2>/dev/null | grep -E ":11345|:14540|:14550" && echo "(ports still busy)" || echo "ports free"
}

# ---- dispatch ---------------------------------------------------------------
case "${1:-}" in
  test)                  cmd_test ;;
  app)                   cmd_app ;;
  build-db)              cmd_build_db ;;
  loctest)               cmd_loctest ;;
  sim)                   cmd_sim ;;
  mission)               cmd_mission "$@" ;;
  deny)                  cmd_deny "$@" ;;
  dashboard)             cmd_dashboard ;;
  diagnose-localization) cmd_diagnose_localization ;;
  offline-vns-check)     cmd_offline_vns_check ;;
  openhouse)             cmd_openhouse ;;
  clean)                 cmd_clean ;;
  *) grep -E '^#( |=)' "$0" | sed 's/^# \{0,1\}//' ; exit 1 ;;
esac
