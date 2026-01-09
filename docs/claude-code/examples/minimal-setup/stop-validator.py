#!/usr/bin/env python3
"""
Minimal Stop Hook Validator

A simplified version that shows a basic compliance checklist.
For the full version with git diff change detection, see config/hooks/stop-validator.py

Exit codes:
  0 - Allow stop
  2 - Block stop (stderr shown to Claude)
"""
import json
import sys


def main():
    try:
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError:
        sys.exit(0)

    # Break the loop - if we already blocked once, allow stop
    if input_data.get("stop_hook_active", False):
        sys.exit(0)

    # First stop - block and give instructions
    instructions = """Before stopping, consider these checks:

1. If you wrote code:
   - Does it follow project conventions?
   - Are there any obvious issues?

2. If you changed behavior:
   - Did you update relevant documentation?
   - Are tests passing?

3. Ready to commit?
   - Stage changes: git add -A
   - Commit with clear message
   - Push to remote

After reviewing, you may stop."""

    print(instructions, file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    main()
