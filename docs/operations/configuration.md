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
| `MAX_AGE_HOURS` | `24` | Filter sessions older than this |
| `GITHUB_TOKEN` | gh CLI auth | GitHub API for PR/CI status |

### Optional - UI

| Variable | Default | Purpose |
|----------|---------|---------|
| `VITE_STREAM_URL` | `http://127.0.0.1:4450/sessions` | Daemon stream endpoint |

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

### GitHub Polling

| Constant | Value | Purpose |
|----------|-------|---------|
| `PR_CACHE_TTL` | 1 minute | PR info cache duration |
| `CI_POLL_INTERVAL_ACTIVE` | 30 seconds | Poll frequency for working sessions |
| `CI_POLL_INTERVAL_IDLE` | 5 minutes | Poll frequency for idle sessions |
| `PR_CACHE_MAX_SIZE` | 1000 | Maximum cached PR entries |
| `PR_CACHE_ENTRY_TTL` | 30 minutes | Individual entry expiration |

### Session Scoring (UI)

| Constant | Value | Purpose |
|----------|-------|---------|
| `STATUS_WEIGHTS` | working:100, waiting:50, idle:1 | Base score by status |
| `PENDING_TOOL_BONUS` | 30 | Bonus for pending tool approval |
| `DECAY_HALF_LIFE_MINUTES` | 30 | Activity score decay rate |

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
GITHUB_TOKEN=ghp_...
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
