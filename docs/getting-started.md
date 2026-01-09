# Getting Started

Get up and running with Claude Code Session Tracker in under 5 minutes.

## Prerequisites

- **Node.js** 22.13.1 or later
- **pnpm** 10.26.0 or later
- **Claude Code CLI** installed and used at least once
- **Anthropic API key** for AI-powered summaries

## Quick Start

### 1. Clone and Install

```bash
git clone <repo-url>
cd claude-code-ui
pnpm install
```

### 2. Configure Environment

Create a `.env` file in the root directory:

```bash
echo "ANTHROPIC_API_KEY=sk-ant-api03-..." > .env
```

> Get an API key from [console.anthropic.com](https://console.anthropic.com)

### 3. Start the Application

```bash
pnpm start
```

This starts both:
- **Daemon** on port 4450 (watches for session changes)
- **UI** on port 5173 (Vite dev server)

### 4. Open the Dashboard

Navigate to [http://localhost:5173](http://localhost:5173)

You should see your active Claude Code sessions appear as cards organized by repository.

## What You'll See

The dashboard shows sessions in a Kanban-style board:

| Column | Meaning |
|--------|---------|
| **Working** | Claude is actively processing |
| **Needs Approval** | Tool use waiting for your approval |
| **Waiting** | Claude finished, waiting for your input |
| **Idle** | No activity for 5+ minutes |

Each session card shows:
- AI-generated goal and summary
- Current git branch

## Verify It's Working

1. Open a terminal and start a Claude Code session:
   ```bash
   cd /path/to/any/project
   claude
   ```

2. Ask Claude to do something (e.g., "What files are in this directory?")

3. The session should appear in the dashboard within seconds

## Optional: Kitty Terminal Integration

If you use [kitty terminal](https://sw.kovidgoyal.net/kitty/), terminal control is **automatically configured** on first daemon start:

- Creates `~/.config/kitty/claude-code.conf` with remote control settings
- Adds include directive to your `kitty.conf`
- Reloads kitty config (or restart kitty if needed)

After setup, you can:
- Click "Open in kitty" to launch terminals for sessions
- Link existing terminals to sessions
- Send text directly to linked terminals

For manual setup: `pnpm --filter @claude-code-ui/daemon setup:kitty`

## Running Separately

If you prefer to run daemon and UI in separate terminals:

```bash
# Terminal 1: Daemon
pnpm serve

# Terminal 2: UI
pnpm dev
```

## CLI Flags

Filter sessions when starting:

```bash
pnpm watch --recent   # Only sessions from last hour
pnpm watch --active   # Only non-idle sessions
```

## Next Steps

- [CLI Reference](cli-reference.md) - All command options
- [UI Components](ui-components.md) - Understanding the interface
- [Configuration](operations/configuration.md) - All environment variables
- [Deployment](operations/deployment.md) - Production setup

## Troubleshooting

**Sessions not appearing?**
- Verify Claude Code is running: `ls ~/.claude/projects/`
- Check file permissions on `~/.claude/projects/`

**AI summaries showing "Processing..."?**
- Verify your API key: `echo $ANTHROPIC_API_KEY | head -c 10`
- Check daemon logs for rate limiting errors

**Port already in use?**
- Check what's using port 4450: `lsof -i :4450`
- Use a different port: `PORT=4451 pnpm serve`

See [Deployment Guide](operations/deployment.md) for more troubleshooting tips.
