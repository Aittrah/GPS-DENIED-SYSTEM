#!/usr/bin/env bash
# =============================================================================
# run_demo.sh — one-shot verification of the GPS-DENIED-SYSTEM demo stack.
#
# Runs staged checks and prints PASS/FAIL for each. Exits non-zero on first
# critical failure unless VERIFY_ONLY=1 (skips live sim stages).
#
# Usage:
#   ./scripts/run_demo.sh
#   VERIFY_ONLY=1 ./scripts/run_demo.sh   # skip live Gazebo/PX4 stages
# =============================================================================
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="$REPO/scripts/run_project.sh"
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
VERIFY_ONLY="${VERIFY_ONLY:-0}"

pass() { echo "PASS: $*"; }
fail() { echo "FAIL: $*" >&2; FAILED=1; }
step() { echo; echo "==> $*"; }

FAILED=0
CRITICAL=0

require_file() {
  local path="$1" label="$2"
  if [ -f "$path" ]; then
    pass "$label ($path)"
  else
    fail "$label missing: $path"
    CRITICAL=1
  fi
}

cd "$REPO"

step "Stage 0 — Python environment"
python3 --version || { fail "python3 not found"; CRITICAL=1; }
if "$RUNNER" test >/tmp/vns_demo_test.log 2>&1; then
  tail -1 /tmp/vns_demo_test.log | grep -q "passed" && pass "pytest ($(tail -1 /tmp/vns_demo_test.log))" || pass "pytest completed"
else
  fail "pytest — see /tmp/vns_demo_test.log"
  CRITICAL=1
fi

step "Stage 1 — ROS 2 Humble"
if [ -f "$ROS_SETUP" ]; then
  set +u
  # shellcheck disable=SC1090
  source "$ROS_SETUP"
  set -u
  if command -v ros2 >/dev/null 2>&1; then
    pass "ROS 2 Humble ($ROS_SETUP)"
  else
    fail "ros2 not found after sourcing $ROS_SETUP"
    CRITICAL=1
  fi
else
  fail "ROS 2 Humble setup missing: $ROS_SETUP"
  CRITICAL=1
fi

step "Stage 2 — Gazebo Classic"
if command -v gzserver >/dev/null 2>&1; then
  pass "Gazebo Classic (gzserver on PATH)"
else
  fail "gzserver not installed"
  CRITICAL=1
fi

step "Stage 3 — Build artifacts"
require_file "$REPO/simulation/models/qau_ground_plane/materials/textures/qau_satellite.png" "Satellite texture"
require_file "$REPO/simulation/database/images_real/database_index.yaml" "Reference tile index"
require_file "$REPO/simulation/database/qau_campus.vnsdb" "VNS database"
require_file "$REPO/install/setup.bash" "ROS workspace install"

step "Stage 4 — PX4 SITL binary"
PX4="${PX4:-$HOME/PX4-Autopilot}"
if [ -x "$PX4/build/px4_sitl_default/bin/px4" ]; then
  pass "PX4 SITL binary"
else
  fail "PX4 SITL binary missing at $PX4/build/px4_sitl_default/bin/px4"
fi
if [ -f "$PX4/build/px4_sitl_default/build_gazebo-classic/libgazebo_mavlink_interface.so" ]; then
  pass "PX4 Gazebo MAVLink plugin"
else
  fail "libgazebo_mavlink_interface.so missing — run: cd $PX4 && make px4_sitl_default"
fi

if [ "$CRITICAL" -ne 0 ]; then
  echo
  echo "CRITICAL checks failed — skipping live simulation stages."
  exit 1
fi

if [ "$VERIFY_ONLY" = "1" ]; then
  step "Skipping live sim (VERIFY_ONLY=1)"
  pass "Offline verification complete"
  exit 0
fi

step "Stage 5 — Vision localization smoke test (45 s)"
"$RUNNER" clean >/dev/null 2>&1 || true
if timeout 45 "$RUNNER" loctest >/tmp/vns_demo_loctest.log 2>&1; then
  :
fi
if grep -q "Localized: ref=" /tmp/vns_demo_loctest.log; then
  pass "Localization ($(grep -m1 'Localized: ref=' /tmp/vns_demo_loctest.log | sed 's/.*Localized: /Localized: /'))"
else
  fail "Localization — no 'Localized: ref=' in /tmp/vns_demo_loctest.log"
fi
"$RUNNER" clean >/dev/null 2>&1 || true

step "Stage 6 — Dashboard from latest log"
if "$RUNNER" dashboard >/tmp/vns_demo_dashboard.log 2>&1; then
  if [ -f "$REPO/reports/dashboard/index.html" ]; then
    pass "Dashboard ($REPO/reports/dashboard/index.html)"
  else
    fail "Dashboard HTML not found after generation"
  fi
else
  fail "Dashboard generation — see /tmp/vns_demo_dashboard.log"
fi

step "Stage 7 — Full sim MAVLink handshake (120 s, headless)"
"$RUNNER" clean >/dev/null 2>&1 || true
timeout 120 "$RUNNER" sim >/tmp/vns_demo_sim.log 2>&1 || true
if grep -q "libgazebo_mavlink_interface.so: cannot open" /tmp/vns_demo_sim.log; then
  fail "PX4 Gazebo plugin not loaded — check GAZEBO_PLUGIN_PATH"
elif grep -q "MAVLink connected on" /tmp/vns_demo_sim.log; then
  pass "MAVLink connected ($(grep -m1 'MAVLink connected' /tmp/vns_demo_sim.log | sed 's/.*MAVLink/MAVLink/'))"
elif grep -q "Ready for takeoff" /tmp/vns_demo_sim.log; then
  pass "PX4 ready for takeoff"
else
  fail "Full sim — no MAVLink connect or Ready for takeoff in /tmp/vns_demo_sim.log"
fi
if grep -q "Successfully spawned entity" /tmp/vns_demo_sim.log; then
  pass "UAV spawn"
else
  fail "UAV spawn not confirmed"
fi
"$RUNNER" clean >/dev/null 2>&1 || true

echo
if [ "$FAILED" -eq 0 ]; then
  echo "========================================"
  echo "  DEMO VERIFICATION: ALL STAGES PASSED"
  echo "========================================"
  echo
  echo "Run the live demo:"
  echo "  Terminal A: $RUNNER sim"
  echo "  Terminal B: $RUNNER mission"
  echo "  Terminal B: $RUNNER deny off"
  exit 0
fi

echo "========================================"
echo "  DEMO VERIFICATION: SOME STAGES FAILED"
echo "========================================"
echo "Review logs under /tmp/vns_demo_*.log"
exit 1
