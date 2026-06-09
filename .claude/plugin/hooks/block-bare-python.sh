#!/bin/bash
# HOOK 6: Block bare `python ` — the python binary does NOT exist on this machine.
# All scripts must use python3 (/usr/bin/python3, 3.10.12).
set -uo pipefail

TOOL_INPUT=$(cat)

MATCH=$(python3 -c "
import json, sys, re

try:
    data = json.loads(sys.argv[1])
except Exception:
    sys.exit(0)

command = data.get('tool_input', {}).get('command', '')

# Match 'python ' at start or ' python ' anywhere — but NOT 'python3'
# 'python3 foo' has '3' after 'python', so it does NOT match 'python\s'
if (re.search(r'^python\s', command) or re.search(r'\s+python\s', command)):
    # Exclude any that actually use python3
    if not re.search(r'python3', command):
        print('BLOCK')
" "$TOOL_INPUT" 2>/dev/null) || MATCH=""

if [ "$MATCH" = "BLOCK" ]; then
    cat >&2 <<'ERRMSG'
[VNS HOOK] BLOCKED: bare `python` command.
The `python` binary does NOT exist on this machine (/usr/bin/python is not installed).

Use python3 instead:

  python3 <script.py>

Examples:
  python3 simulation/scripts/capture_reference_images.py --synthetic ...
  python3 simulation/scripts/build_reference_database.py --input ...
  python3 src/vns/cli.py ...

python3: /usr/bin/python3 (Python 3.10.12)
ERRMSG
    exit 2
fi

exit 0
