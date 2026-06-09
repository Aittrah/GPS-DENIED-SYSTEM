#!/bin/bash
# HOOK 7: Block wrong absolute path patterns.
#   /home/ubuntu/  — wrong user (correct: /home/hp/)
#   /opt/px4/      — wrong PX4 location (correct: /home/hp/PX4-Autopilot)
#   ~/             — tilde in Bash commands (unreliable in scripts)
set -uo pipefail

TOOL_INPUT=$(cat)

RESULT=$(python3 -c "
import json, sys, re

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

errors = []

if '/home/ubuntu/' in content:
    errors.append('  /home/ubuntu/  ->  wrong user; correct base: /home/hp/')

if '/opt/px4/' in content:
    errors.append('  /opt/px4/      ->  wrong PX4 path; correct path: /home/hp/PX4-Autopilot')

# Only flag tilde in Bash commands (not in file content where it may be docs/Python)
if name == 'Bash' and re.search(r'(^|\s)~/', content):
    errors.append('  ~/             ->  tilde expansion unreliable in scripts; use /home/hp/ explicitly')

if errors:
    print('BLOCK')
    for e in errors:
        print(e)
" "$TOOL_INPUT" 2>/dev/null) || RESULT=""

if printf '%s' "$RESULT" | grep -q '^BLOCK'; then
    DETAILS=$(printf '%s' "$RESULT" | tail -n +2)
    printf '[VNS HOOK] BLOCKED: Wrong path pattern(s) detected:\n\n%s\n\nAll absolute paths must use /home/hp/ as the base.\nPX4-Autopilot is at /home/hp/PX4-Autopilot (already compiled).\n' "$DETAILS" >&2
    exit 2
fi

exit 0
