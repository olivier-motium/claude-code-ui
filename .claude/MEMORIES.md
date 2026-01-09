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
- `type-guards.ts` - type narrowing guards (isUserEntry, isError, getErrorMessage)

### Summarizer Module Structure
Summarizer is split into `packages/daemon/src/summarizer/`:
- `index.ts` - barrel exports
- `summarizer.ts` - main generateAISummary, generateGoal
- `context-extraction.ts` - extractContext, extractEarlyContext
- `summaries.ts` - getWorkingSummary, getFallbackSummary
- `cache.ts` - evictStaleEntries, generateContentHash
- `text-utils.ts` - cleanGoalText

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

### Session Resume via Kitty
"Open in kitty" runs `claude --resume <sessionId> --dangerously-skip-permissions`:
- `--resume` continues the exact Claude Code session from its log file
- `--dangerously-skip-permissions` skips the folder trust prompt (user controls which sessions to open)
- Session IDs from our JSONL filenames match Claude Code's internal format

### Terminal Link Recovery (Cascading Fallback)
Kitty window IDs are ephemeral - they change on kitty restart or when other windows are created/destroyed. Solution: cascading recovery before each terminal operation:
1. Check if stored window ID exists (fast path)
2. Search by `user_vars.cc_session_id` (set via `--var` when launching)
3. Search by `--resume <sessionId>` in cmdline (fallback)
4. If not found, create new tab or return error

**Important limitation:** Cmdline search only works for windows launched via our "Open in Kitty" button. Windows started as normal shells show `/bin/zsh` in cmdline, not claude. User_vars is the reliable recovery method.

### Entry Limit to Prevent Memory Leaks
Sessions can have thousands of log entries over time. Without trimming, memory grows unbounded causing OOM kills (exit 137). Solution: `MAX_ENTRIES_PER_SESSION = 500` in config.ts, trimmed in watcher.ts. This is sufficient for status detection and summarization while preventing memory exhaustion.

### StreamDB Corruption Recovery
If durable-streams client shows `Symbol(liveQueryInternal)` errors, the stream data may be corrupted. Fix: backup and clear `~/.claude-code-ui/streams/`, restart daemon. The stream will rebuild from session files.

### File-Based Status System (Jan 2026)
Alternative to AI summaries for session status. Claude Code writes status to `.claude/status.md` via hooks, daemon watches and streams to UI.

**Why file-based over AI summaries:**
- Deterministic: Hooks provide exact paths/templates, model fills slots
- Scalable: Single file per project, always overwritten (no growth)
- Reliable: YAML frontmatter parsed deterministically vs LLM variability
- Cheaper: No API calls for status derivation

**Architecture:**
- `status-working.py` (UserPromptSubmit hook): Instructs model to write "working" status
- `status-stop.py` (Stop hook): Instructs model to write completion status
- `status-watcher.ts`: Watches `.claude/status.md` across projects
- `status-parser.ts`: Parses YAML frontmatter from status files
- `fileStatus` field in Session schema takes precedence when present and fresh

**Staleness:** Status files valid for 5 minutes (STATUS_FILE_TTL_MS), then falls back to XState derivation.

**Status taxonomy:** working, waiting_for_approval, waiting_for_input, completed, error, blocked, idle

## Documentation Audit (Jan 2026)

12 documentation fixes applied:
- Fixed broken link in `getting-started.md` (api/ui-components.md → ui-components.md)
- Updated SessionCard to module path in `ui-components.md`
- Updated summarizer to module path in `summarizer.md` and `CLAUDE.md`
- Added "Needs Approval" column to component hierarchy
- Documented debug endpoint `/api/debug/sessions` in `daemon-api.md`
- Documented session scoring algorithm in `ui-components.md`
- Documented parser API (tailJSONL, extractMetadata) in `daemon-api.md`
- Documented SessionWatcher API in `daemon-api.md`
- Added core type definitions (SessionState, StatusResult, LogEntry) to `daemon-api.md`
- Added database commands (db:generate, db:migrate, db:studio) to `cli-reference.md`
- Fixed TODO.md reference in CLAUDE.md → .claude/MEMORIES.md
- Deleted orphaned `packages/ui/README.md` (Vite template)

## Known Issues

### Pre-existing Test Failures
`pnpm test` in daemon has 7 unique failures (14 total, running twice from dist/src):
- Status derivation tests have mismatched expectations
- SessionWatcher test reads real data instead of fixtures
- Parser test missing test directory setup
These are unrelated to cache/timeout refactoring.
