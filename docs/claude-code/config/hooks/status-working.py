#!/usr/bin/env python3
"""
UserPromptSubmit hook - instructs Claude to update status to 'working'.

This hook fires when a user sends a prompt. It provides a template
for Claude to update the project's status file with the current task.

Exit codes:
  0 - Success (context added to Claude)
"""
import json
import sys
from datetime import datetime, timezone


def main():
    try:
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError:
        sys.exit(0)

    # Generate deterministic values
    status_file = ".claude/status.md"
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Template for model to fill - minimal decisions required
    template = f"""Update project status file at: {status_file}

```markdown
---
status: working
updated: {timestamp}
task: [Brief description of this task - 5-10 words]
---

## Summary
[What you're about to do - 1-2 sentences]

## Next Steps
[What needs to happen to complete this task]
```

Create this file (mkdir -p .claude if needed) before starting work."""

    print(template)
    sys.exit(0)


if __name__ == "__main__":
    main()
