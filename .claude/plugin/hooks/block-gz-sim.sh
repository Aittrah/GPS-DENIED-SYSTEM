#!/bin/bash
# HOOK 1: Block 'gz sim' — Gazebo Harmonic command, wrong for this machine.
# This project uses Gazebo Classic 11.  Correct binaries: gzserver + gzclient.
set -uo pipefail

TOOL_INPUT=$(cat)

MATCH=$(python3 -c "
import json, sys, re

try:
    data = json.loads(sys.argv[1])
except Exception:
    sys.exit(0)

command = data.get('tool_input', {}).get('command', '')

# Match 'gz sim' with word boundaries / surrounding whitespace or EOL
if re.search(r'(^|[\s;|&])gz\s+sim(\s|$|;|&&|\|)', command):
    print('BLOCK')
" "$TOOL_INPUT" 2>/dev/null) || MATCH=""

if [ "$MATCH" = "BLOCK" ]; then
    cat >&2 <<'ERRMSG'
[VNS HOOK] BLOCKED: `gz sim` is a Gazebo Harmonic command.
This machine runs Gazebo Classic 11.  Use the Classic binaries instead:

  TERMINAL 1 — physics server:
    gzserver /home/hp/GPS-DENIED-SYSTEM/simulation/worlds/vns_test_world.sdf \
             -s libgazebo_ros_init.so \
             -s libgazebo_ros_factory.so \
             --verbose

  TERMINAL 2 — GUI client:
    gzclient --verbose

Binaries:
  /usr/bin/gzserver  (physics + sensors)
  /usr/bin/gzclient  (GUI)
ERRMSG
    exit 2
fi

exit 0
