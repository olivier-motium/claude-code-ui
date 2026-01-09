#!/usr/bin/env python3
"""
Stop hook - instructs Claude to update status before stopping.

This hook fires when Claude attempts to stop. It blocks (exit 2) on first
attempt, providing instructions to update the status file. Uses the
stop_hook_active flag to prevent infinite loops.

Exit codes:
  0 - Allow stop (when stop_hook_active=true, loop prevention)
  2 - Block stop (stderr shown to Claude as feedback)
"""
import json
import sys
from datetime import datetime, timezone


def main():
    try:
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError:
        # Can't parse input, allow stop to prevent blocking
        sys.exit(0)

    stop_hook_active = input_data.get("stop_hook_active", False)

    # Loop prevention: if we already blocked once, allow stop
    if stop_hook_active:
        sys.exit(0)

    # Generate deterministic values
    status_file = ".claude/status.md"
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    instructions = f"""Before stopping, update your session status:

1. UPDATE STATUS FILE at: {status_file}

```markdown
---
status: [completed|error|blocked|waiting_for_input]
updated: {timestamp}
task: [What was the task - 5-10 words]
---

## Summary
[What was accomplished or what went wrong - 1-2 sentences]

## Blockers
[If blocked/error: what's preventing progress. Otherwise: None]

## Next Steps
[What needs to happen next, if anything. If completed: None]
```

2. VALID STATUS VALUES:
   - completed: Task finished successfully
   - error: Encountered an error that needs attention
   - blocked: Cannot proceed without external action
   - waiting_for_input: Need more information from user

3. THEN complete standard checks:
   - CLAUDE.md compliance (if code written)
   - Documentation updates (if code written)
   - Update .claude/MEMORIES.md (if significant learnings)
   - Commit and push (if requested)

After updating status file, you may stop."""

    print(instructions, file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    main()
