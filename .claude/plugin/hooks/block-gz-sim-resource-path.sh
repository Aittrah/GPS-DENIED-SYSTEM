#!/bin/bash
# HOOK 3: Block GZ_SIM_RESOURCE_PATH — Gazebo Harmonic env var.
# The correct variable for Gazebo Classic 11 is GAZEBO_MODEL_PATH.
set -uo pipefail

TOOL_INPUT=$(cat)

MATCH=$(python3 -c "
import json, sys

try:
    data = json.loads(sys.argv[1])
except Exception:
    sys.exit(0)

name = data.get('tool_name', '')
inp  = data.get('tool_input', {})

if name == 'Bash':
    content = inp.get('command', '')
elif name == 'Edit':
    content = inp.get('new_string', '')
elif name == 'Write':
    content = inp.get('content', '')
else:
    sys.exit(0)

if 'GZ_SIM_RESOURCE_PATH' in content:
    print('BLOCK')
" "$TOOL_INPUT" 2>/dev/null) || MATCH=""

if [ "$MATCH" = "BLOCK" ]; then
    cat >&2 <<'ERRMSG'
[VNS HOOK] BLOCKED: `GZ_SIM_RESOURCE_PATH` is a Gazebo Harmonic environment variable.
This machine runs Gazebo Classic 11, which uses GAZEBO_MODEL_PATH.

GAZEBO_MODEL_PATH is currently UNSET on this machine.
Set it before launching gzserver or any gazebo_ros tool:

  export GAZEBO_MODEL_PATH=/home/hp/GPS-DENIED-SYSTEM/simulation/models

Add this line to your launch scripts or terminal setup.
ERRMSG
    exit 2
fi

exit 0
