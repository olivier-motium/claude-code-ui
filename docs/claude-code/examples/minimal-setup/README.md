# Minimal Setup

This is a stripped-down version of the toolkit with just the essentials:
- **settings.json** — Basic settings + SessionStart and Stop hooks
- **stop-validator.py** — Compliance checklist before stopping

This gives you:
- Claude reads project docs at session start
- Compliance reminder before stopping (no commit/push enforcement)

## Installation

```bash
# Copy these files to ~/.claude/
mkdir -p ~/.claude/hooks
cp settings.json ~/.claude/
cp stop-validator.py ~/.claude/hooks/
chmod +x ~/.claude/hooks/stop-validator.py
```

## What's Different from Full Setup

| Feature | Minimal | Full |
|---------|---------|------|
| SessionStart hook | ✓ | ✓ |
| Stop hook | ✓ (basic) | ✓ (change-aware) |
| Commands | ✗ | 7 commands |
| Skills | ✗ | 6 skills |
| Skill reminder hook | ✗ | ✓ |

## When to Use Minimal

- You want basic session hygiene without all the commands
- You're building your own commands/skills from scratch
- You want to understand the core patterns before adding more

## Customizing

### Add a Command

Create `~/.claude/commands/my-command.md`:
```markdown
---
description: My custom command
---

Your prompt here.
```

### Add a Skill

Create `~/.claude/skills/my-skill/SKILL.md`:
```markdown
---
name: my-skill
description: Triggers on [keywords]
---

Your skill content here.
```

See the full toolkit's [customization guide](../../docs/guides/customization.md) for details.
