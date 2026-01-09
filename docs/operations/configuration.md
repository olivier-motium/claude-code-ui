# Configuration Reference

Complete reference for all environment variables and tunable constants.

## Environment Variables

### Required

| Variable | Purpose | Example |
|----------|---------|---------|
| `ANTHROPIC_API_KEY` | Claude API key for AI summaries | `sk-ant-api03-...` |

### Optional - Daemon

| Variable | Default | Purpose |
|----------|---------|---------|
| `STREAM_HOST` | `127.0.0.1` | HTTP server bind address |
| `PORT` | `4450` | Stream server port |
| `API_PORT` | `4451` | API server port for terminal control |
| `PTY_WS_HOST` | `127.0.0.1` | PTY WebSocket server bind address |
| `PTY_WS_PORT` | `4452` | PTY WebSocket server port |
| `MAX_AGE_HOURS` | `24` | Filter sessions older than this |
| `KITTY_SOCKET` | `unix:/tmp/claude-cc-kitty` | Kitty remote control socket |
| `KITTY_RC_PASSWORD` | (none) | Password for kitty remote control |
| `DB_PATH` | `~/.claude-code-ui/data.db` | SQLite database path |

### Optional - UI

| Variable | Default | Purpose |
|----------|---------|---------|
| `VITE_STREAM_URL` | `http://127.0.0.1:4450/sessions` | Daemon stream endpoint |
| `VITE_API_URL` | `http://127.0.0.1:4451/api` | Terminal control API endpoint |

---

## Internal Constants

These constants are defined in `packages/daemon/src/config.ts` and affect daemon behavior.

### Timeout Configuration

| Constant | Value | Purpose |
|----------|-------|---------|
| `IDLE_TIMEOUT_MS` | 5 minutes | Session marked idle after inactivity |
| `APPROVAL_TIMEOUT_MS` | 5 seconds | Time before stale tool_use detection |
| `STALE_TIMEOUT_MS` | 60 seconds | Fallback for older Claude Code versions |
| `RECENT_THRESHOLD_MS` | 1 hour | `--recent` flag filter threshold |

### Session Scoring (UI)

| Constant | Value | Purpose |
|----------|-------|---------|
| `STATUS_WEIGHTS` | working:100, waiting:50, idle:1 | Base score by status |
| `PENDING_TOOL_BONUS` | 30 | Bonus for pending tool approval |
| `DECAY_HALF_LIFE_MINUTES` | 30 | Activity score decay rate |

### Summary Cache

| Constant | Value | Purpose |
|----------|-------|---------|
| `SUMMARY_CACHE_MAX_SIZE` | 500 | Maximum cached summary entries |
| `SUMMARY_CACHE_TTL_MS` | 30 minutes | Summary cache entry TTL |
| `GOAL_CACHE_MAX_SIZE` | 500 | Maximum cached goal entries |
| `GOAL_CACHE_TTL_MS` | 30 minutes | Goal cache entry TTL |

### External Call Timeouts

| Constant | Value | Purpose |
|----------|-------|---------|
| `EXTERNAL_CALL_TIMEOUT_MS` | 30 seconds | Timeout for Anthropic API calls |
| `GH_CLI_TIMEOUT_MS` | 15 seconds | Timeout for gh CLI operations |

### Kitty Terminal Control

| Constant | Value | Purpose |
|----------|-------|---------|
| `KITTY_SOCKET` | `unix:/tmp/claude-cc-kitty` | Kitty remote control socket path |
| `KITTY_PASSWORD_ENV` | `KITTY_RC_PASSWORD` | Env var name for kitty password |
| `KITTY_COMMAND_TIMEOUT_MS` | 5 seconds | Timeout for kitty commands |
| `API_PORT` | `4451` | Port for terminal control API |
| `API_PREFIX` | `/api` | URL prefix for API endpoints |

### Embedded PTY Server

| Constant | Value | Purpose |
|----------|-------|---------|
| `PTY_WS_HOST` | `127.0.0.1` | WebSocket server bind address |
| `PTY_WS_PORT` | `4452` | WebSocket server port |
| `PTY_IDLE_TIMEOUT_MS` | 30 minutes | Cleanup inactive PTYs |
| `PTY_DEFAULT_COLS` | `120` | Default terminal width |
| `PTY_DEFAULT_ROWS` | `40` | Default terminal height |

---

## Configuration Examples

### Development (default)

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-api03-...
```

### Production (custom host/port)

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-api03-...
STREAM_HOST=0.0.0.0
PORT=8080
MAX_AGE_HOURS=48
```

### UI connecting to remote daemon

```bash
# packages/ui/.env
VITE_STREAM_URL=http://192.168.1.100:4450/sessions
```

---

## Related Documentation

- [Deployment Guide](deployment.md) - Production setup with PM2/systemd
- [CLI Reference](../cli-reference.md) - Command line flags
