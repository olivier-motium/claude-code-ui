# Deployment Guide

How to run Claude Code Session Tracker in production.

## Prerequisites

- **Node.js** 22.13.1 or later
- **pnpm** 10.26.0 or later
- **Anthropic API key** for AI summaries

---

## Environment Variables

### Daemon (packages/daemon)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | Yes | - | Claude API key for AI summaries |
| `STREAM_HOST` | No | `127.0.0.1` | Host to bind stream server |
| `PORT` | No | `4450` | Port for stream server |
| `MAX_AGE_HOURS` | No | `24` | Filter sessions older than this |
| `GITHUB_TOKEN` | No | `gh` CLI auth | GitHub API for PR/CI status |

### UI (packages/ui)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `VITE_STREAM_URL` | No | `http://127.0.0.1:4450/sessions` | URL to daemon stream endpoint |

See [Configuration Reference](configuration.md) for all internal constants.

---

## Development

```bash
# Install dependencies
pnpm install

# Start both daemon and UI
pnpm start

# Or run separately:
pnpm serve  # Daemon on port 4450
pnpm dev    # UI on port 5173
```

---

## Production Deployment

### Option 1: PM2 (Recommended)

```bash
# Install PM2
npm install -g pm2

# Start daemon
pm2 start pnpm --name "claude-code-daemon" -- serve

# Start UI (if serving built assets)
cd packages/ui && pnpm build
pm2 start pnpm --name "claude-code-ui" -- preview

# Save configuration
pm2 save
pm2 startup
```

### Option 2: macOS launchd

Create `~/Library/LaunchAgents/com.claude-code-ui.daemon.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.claude-code-ui.daemon</string>
  <key>ProgramArguments</key>
  <array>
    <string>/opt/homebrew/bin/pnpm</string>
    <string>serve</string>
  </array>
  <key>WorkingDirectory</key>
  <string>/path/to/claude-code-ui</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>ANTHROPIC_API_KEY</key>
    <string>sk-ant-...</string>
  </dict>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>/tmp/claude-code-ui.log</string>
  <key>StandardErrorPath</key>
  <string>/tmp/claude-code-ui.error.log</string>
</dict>
</plist>
```

Load with:
```bash
launchctl load ~/Library/LaunchAgents/com.claude-code-ui.daemon.plist
```

### Option 3: Linux systemd

Create `/etc/systemd/user/claude-code-ui.service`:

```ini
[Unit]
Description=Claude Code Session Tracker Daemon
After=network.target

[Service]
Type=simple
WorkingDirectory=/path/to/claude-code-ui
ExecStart=/usr/bin/pnpm serve
Environment=ANTHROPIC_API_KEY=sk-ant-...
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
```

Enable with:
```bash
systemctl --user enable claude-code-ui
systemctl --user start claude-code-ui
```

---

## Data Locations

| Path | Purpose | Managed By |
|------|---------|------------|
| `~/.claude/projects/` | Session log files | Claude Code (read-only) |
| `~/.claude-code-ui/streams/` | Durable Streams persistence | Daemon |

### Log Retention

Claude Code retains logs for 30 days by default. Extend via `~/.claude/settings.json`:

```json
{
  "logRetentionDays": 99999
}
```

---

## Ports

| Port | Service | Configurable |
|------|---------|--------------|
| 4450 | Durable Streams server | Yes (`PORT` env var) |
| 5173 | UI dev server (Vite) | Yes (vite.config.ts) |

---

## Health Checks

### Daemon Health

Check if daemon is responding:

```bash
curl http://127.0.0.1:4450/sessions
# Should return SSE stream
```

### UI Health

Check if UI is serving:

```bash
curl http://localhost:5173
# Should return HTML
```

---

## Troubleshooting

### Daemon won't start

1. Check Node.js version: `node --version` (need 22.13.1+)
2. Check environment: `echo $ANTHROPIC_API_KEY`
3. Check port: `lsof -i :4450`

### Sessions not appearing

1. Verify Claude Code is running: `ls ~/.claude/projects/`
2. Check file permissions on `~/.claude/projects/`
3. Run `pnpm watch` to debug file detection

### AI summaries failing

1. Verify API key: `echo $ANTHROPIC_API_KEY | head -c 10`
2. Check Anthropic API status
3. Review daemon logs for rate limiting errors

### PR status not updating

1. Check GitHub auth: `gh auth status`
2. Or set `GITHUB_TOKEN` environment variable
3. Verify branch has a PR open

---

## Building for Production

```bash
# Build UI
cd packages/ui
pnpm build

# Output in packages/ui/dist/
# Serve with any static file server
```

### Serving Built UI

```bash
# Using Vite preview
pnpm preview

# Or with any static server
npx serve packages/ui/dist
```

---

## Related Documentation

- [Configuration Reference](configuration.md) - All environment variables and tunables
- [CLI Reference](../cli-reference.md) - Command line flags
