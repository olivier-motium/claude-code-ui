# Session Memories - claude-code-ui

## Documentation Structure

Documentation lives in `docs/` folder with `docs/index.md` as the entry point.

| Doc | Purpose |
|-----|---------|
| `docs/index.md` | Documentation hub - start here |
| `docs/getting-started.md` | Onboarding quickstart |
| `docs/cli-reference.md` | CLI commands and flags |
| `docs/ui-components.md` | React component hierarchy |
| `docs/summarizer.md` | AI summarization service |
| `docs/operations/deployment.md` | Production deployment |
| `docs/operations/configuration.md` | Env vars and internal constants |
| `docs/api/daemon-api.md` | github.ts, git.ts, summarizer.ts APIs |
| `docs/guides/testing.md` | Manual testing strategies |

## CLI Flags

- `pnpm watch --recent` - Sessions from last 1 hour only (RECENT_THRESHOLD_MS = 3600000)
- `pnpm watch --active` - Non-idle sessions only

## Tech Stack

- Node.js 22.13.1, pnpm 10.26.0
- chokidar 4.0.3
- Radix UI Themes (not Tailwind)
- Vite 7.2.4
- XState 5.25.0 for status machine

## Architectural Decisions

### XState for Status Detection
Status is derived via state machine (not imperative if/else) to handle edge cases:
- Stale message detection (pending tool_use older than 5s)
- Timeout fallbacks for older Claude Code versions
- Clean event-driven transitions

### Durable Streams for Real-time Sync
Port 4450 serves SSE stream that UI subscribes to via `@durable-streams/state`.
This allows multiple UI clients to stay in sync without polling.

### Centralized Configuration (config.ts)
All daemon constants live in `packages/daemon/src/config.ts`:
- Stream server config (STREAM_HOST, STREAM_PORT, STREAM_PATH)
- Timeout constants (IDLE_TIMEOUT_MS, APPROVAL_TIMEOUT_MS, STALE_TIMEOUT_MS)
- GitHub polling intervals (PR_CACHE_TTL, CI_POLL_INTERVAL_ACTIVE/IDLE)
- Cache limits (PR_CACHE_MAX_SIZE, PR_CACHE_ENTRY_TTL)

This prevents magic numbers scattered across files and enables env var overrides.

### Utility Modules Pattern
Shared utilities live in `packages/daemon/src/utils/`:
- `timeout.ts` - withTimeout() wrapper for async operations
- `colors.ts` - ANSI codes for CLI output
- `errors.ts` - standardized error message extraction

### Cache Eviction Pattern
Caches use LRU-style eviction with TTL (see summarizer.ts):
```typescript
interface CacheEntry { value: T; timestamp: number; }
// Evict before lookup: expired entries + oldest if over maxSize
```

### Security: Command Injection Prevention
Use `execFile` with array args instead of `exec` with template strings:
```typescript
// Bad: execAsync(`gh pr list --head "${branch}"`)
// Good: execFileAsync("gh", ["pr", "list", "--head", branch])
```

### Kitty Terminal Control (Dual Port Architecture)
Daemon runs two HTTP servers on different ports:
- Port 4450: Durable Streams SSE (existing, one-way daemon→UI)
- Port 4451: Hono API for terminal control (new, request/response)

This separation was chosen because DurableStreamTestServer binds its own HTTP server, making port sharing impractical. The UI API client at `packages/ui/src/lib/api.ts` points to port 4451.

SQLite (better-sqlite3) chosen over in-memory storage for:
- Concurrent access from multiple UI clients
- Data persistence across daemon restarts
- Future extensibility (command history, preferences)

Terminal links sync to UI via Durable Streams by including `terminalLink` field in Session schema updates.

### Kitty Auto-Setup (Non-Invasive Config)
`kitty-setup.ts` automatically configures kitty remote control on daemon startup:
- Uses `socket-only` mode (secure, no passwords needed, filesystem permissions)
- Creates separate `claude-code.conf` via `include` directive (non-invasive)
- Sends SIGUSR1 to reload config without restart
- Creates `macos-launch-services-cmdline` for GUI launches on macOS

This approach was chosen over:
- Password mode: Requires manual password setup/env vars
- Direct kitty.conf modification: Risky, could break user config
- --override flags: Only work for terminal-launched kitty

## Known Issues

### Pre-existing Test Failures
`pnpm test` in daemon has 7 unique failures (14 total, running twice from dist/src):
- Status derivation tests have mismatched expectations
- SessionWatcher test reads real data instead of fixtures
- Parser test missing test directory setup
These are unrelated to cache/timeout refactoring.
