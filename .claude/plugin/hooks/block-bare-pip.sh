#!/bin/bash
# HOOK 5: Block bare `pip install` — use pip3.
# bare pip on this machine (/home/hp/.local/bin/pip) may target the wrong Python.
# pip3 (/home/hp/.local/bin/pip3) targets /usr/bin/python3 (3.10.12).
set -uo pipefail

TOOL_INPUT=$(cat)

MATCH=$(python3 -c "
import json, sys, re

try:
    data = json.loads(sys.argv[1])
except Exception:
    sys.exit(0)

command = data.get('tool_input', {}).get('command', '')

# Match 'pip install' (bare pip) but NOT 'pip3 install'
# 'pip3 install' has '3' after 'pip', so 'pip\s+install' does NOT match it.
if (re.search(r'(^|\s)pip\s+install', command)
        and not re.search(r'(^|\s)pip3\s+install', command)
        and not re.search(r'-m\s+pip\s+install', command)):
    print('BLOCK')
" "$TOOL_INPUT" 2>/dev/null) || MATCH=""

if [ "$MATCH" = "BLOCK" ]; then
    cat >&2 <<'ERRMSG'
[VNS HOOK] BLOCKED: bare `pip install` detected.
On this machine, bare `pip` (/home/hp/.local/bin/pip) may install to the wrong
Python environment.

Use pip3 instead:

  pip3 install <package>

pip3: /home/hp/.local/bin/pip3
Targets: /usr/bin/python3 (Python 3.10.12)

Examples:
  pip3 install opencv-python
  pip3 install pyyaml
  pip3 install numpy
ERRMSG
    exit 2
fi

exit 0
