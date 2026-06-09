#!/bin/bash
# HOOK 4: Block PX4 gz_* make targets — they require Gazebo Harmonic.
# This machine has Gazebo Classic 11.  Correct SITL target: none_iris.
set -uo pipefail

TOOL_INPUT=$(cat)

MATCH=$(python3 -c "
import json, sys, re

try:
    data = json.loads(sys.argv[1])
except Exception:
    sys.exit(0)

command = data.get('tool_input', {}).get('command', '')

# Match make commands that include a gz_ prefixed target
if re.search(r'\bmake\b.*\bgz_[a-zA-Z0-9_]+', command):
    print('BLOCK')
" "$TOOL_INPUT" 2>/dev/null) || MATCH=""

if [ "$MATCH" = "BLOCK" ]; then
    cat >&2 <<'ERRMSG'
[VNS HOOK] BLOCKED: PX4 `gz_*` make targets (gz_x500, gz_iris, gz_standard_vtol …)
require Gazebo Harmonic, which is NOT installed on this machine.

This machine has Gazebo Classic 11.  Use the Classic-compatible SITL target:

  cd /home/hp/PX4-Autopilot && HEADLESS=1 make px4_sitl_default none_iris

PX4 binary (already built):
  /home/hp/PX4-Autopilot/build/px4_sitl_default/bin/px4

MAVLink ports after launch:
  UDP 14540 — VNS / offboard
  UDP 14550 — QGroundControl
ERRMSG
    exit 2
fi

exit 0
