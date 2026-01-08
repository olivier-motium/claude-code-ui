# Daemon Internal APIs

Reference documentation for the daemon's internal service modules.

## GitHub PR Tracking (`github.ts`)

Provides PR status and CI badge information for sessions.

### Public Functions

#### `queuePRCheck(cwd: string, branch: string, sessionId: string): void`

Queues a PR check for a session. Requests are deduplicated and processed through a FIFO queue with concurrency of 2.

```typescript
import { queuePRCheck } from "./github.js";

// Queue a PR check when session is detected
queuePRCheck("/path/to/repo", "feature-branch", "session-123");
```

#### `getCachedPR(cwd: string, branch: string): PRInfo | null`

Returns cached PR info for a repository/branch combination without making API calls.

```typescript
import { getCachedPR } from "./github.js";

const pr = getCachedPR("/path/to/repo", "main");
if (pr) {
  console.log(`PR #${pr.number}: ${pr.ciStatus}`);
}
```

#### `setOnPRUpdate(callback: (sessionId, pr) => void): void`

Register callback for PR status changes. Used by the watcher to update session state.

```typescript
import { setOnPRUpdate } from "./github.js";

setOnPRUpdate((sessionId, pr) => {
  if (pr) {
    console.log(`Session ${sessionId}: PR #${pr.number} is ${pr.ciStatus}`);
  }
});
```

#### `stopAllPolling(): void`

Cleanup function to stop all active CI polling intervals.

### PRInfo Schema

```typescript
interface PRInfo {
  number: number;           // PR number
  url: string;              // GitHub PR URL
  title: string;            // PR title
  ciStatus: CIStatus;       // Overall CI status
  ciChecks: CICheck[];      // Individual check results
  lastChecked: string;      // ISO timestamp
}

interface CICheck {
  name: string;             // Check name (e.g., "build")
  status: CIStatus;         // Check status
  url: string | null;       // Link to check details
}

type CIStatus = "success" | "failure" | "running" | "pending" | "cancelled" | "unknown";
```

### CI Status Mapping

GitHub states are mapped to simplified statuses:

| GitHub State | Mapped Status |
|--------------|---------------|
| SUCCESS, COMPLETED, NEUTRAL, SKIPPED | `success` |
| FAILURE, ERROR, TIMED_OUT, ACTION_REQUIRED | `failure` |
| IN_PROGRESS, QUEUED, REQUESTED, WAITING | `running` |
| PENDING | `pending` |
| CANCELLED | `cancelled` |

### Caching Strategy

| Constant | Value | Purpose |
|----------|-------|---------|
| `PR_CACHE_TTL` | 1 minute | Cache duration before re-checking |
| `PR_CACHE_MAX_SIZE` | 1000 | Maximum cached entries |
| `PR_CACHE_ENTRY_TTL` | 30 minutes | Individual entry expiration |

### Polling Behavior

| Constant | Value | When Used |
|----------|-------|-----------|
| `CI_POLL_INTERVAL_ACTIVE` | 30 seconds | CI is running/pending |
| `CI_POLL_INTERVAL_IDLE` | 5 minutes | CI is complete |

Polling automatically adjusts based on CI status:
- Active polling starts when CI is `pending` or `running`
- Switches to idle polling when CI completes
- Stops entirely when session ends

### Requirements

- Requires `gh` CLI installed and authenticated
- Falls back gracefully if `gh` is unavailable

### Timeout Handling

All `gh` CLI calls are wrapped with a 15-second timeout (`GH_CLI_TIMEOUT_MS`). If a call times out, it logs a warning and returns a null/unknown result rather than blocking indefinitely.

### Security

Uses `execFile` instead of `exec` with array arguments to prevent command injection vulnerabilities. Branch names and PR numbers are passed as separate arguments, not interpolated into shell strings.

---

## Git Operations (`git.ts`)

Detects git repository info for sessions without shell commands.

### Public Functions

#### `getGitInfo(cwd: string): Promise<GitInfo>`

Reads git repository information directly from `.git` directory.

```typescript
import { getGitInfo } from "./git.js";

const info = await getGitInfo("/path/to/project");
if (info.isGitRepo) {
  console.log(`Repo: ${info.repoId}, Branch: ${info.branch}`);
}
```

#### `getGitInfoCached(cwd: string): Promise<GitInfo>`

Returns cached git info for a directory. Avoids repeated filesystem lookups.

```typescript
import { getGitInfoCached } from "./git.js";

// First call reads from filesystem
const info1 = await getGitInfoCached("/path/to/project");

// Subsequent calls return cached result
const info2 = await getGitInfoCached("/path/to/project");
```

### GitInfo Schema

```typescript
interface GitInfo {
  repoUrl: string | null;   // Full URL: https://github.com/owner/repo
  repoId: string | null;    // Normalized: owner/repo
  branch: string | null;    // Current branch name (null if detached HEAD)
  isGitRepo: boolean;       // Whether directory is in a git repo
}
```

### URL Parsing

Handles both HTTPS and SSH remote URL formats:

| Format | Example |
|--------|---------|
| HTTPS | `https://github.com/owner/repo.git` |
| SSH | `git@github.com:owner/repo.git` |

Both are normalized to `owner/repo` for `repoId`.

### Implementation Details

- Walks up directory tree to find `.git` folder
- Reads `.git/HEAD` for current branch
- Parses `.git/config` for origin remote URL
- No shell commands - pure filesystem operations
- Cache is per-directory, persists for daemon lifetime

---

## AI Summarization (`summarizer.ts`)

Generates session goals and summaries using Claude API.

### Public Functions

#### `generateGoal(session: SessionState): Promise<string>`

Generates a 5-10 word high-level goal from session context.

```typescript
import { generateGoal } from "./summarizer.js";

const goal = await generateGoal(session);
// Returns: "Build UI for monitoring sessions"
```

**Caching behavior:**
- Results cached per session
- Regenerates if session grows 5x since last generation
- For sessions < 5 entries, returns cleaned original prompt

#### `generateAISummary(session: SessionState): Promise<string>`

Generates a short summary of current session state.

```typescript
import { generateAISummary } from "./summarizer.js";

const summary = await generateAISummary(session);
// Returns: "Editing config.ts" or "Waiting for approval"
```

**Quick summaries (no API call):**
- Sessions < 3 entries: "Just started"
- Working sessions: Tool-based summary (e.g., "Editing file.ts")
- Cached if content hash matches

#### `clearSummaryCache(sessionId: string): void`

Clear the summary cache for a specific session.

### Model Configuration

| Setting | Value |
|---------|-------|
| Model | `claude-sonnet-4-20250514` |
| Max tokens (goal) | 30 |
| Max tokens (summary) | 100 |

### Rate Limiting

- API calls queued through `fastq` with concurrency of 1
- Prevents rate limit errors from Claude API
- Requests processed in FIFO order
- API calls timeout after 30 seconds (`EXTERNAL_CALL_TIMEOUT_MS`)

### Cache Eviction

Both summary and goal caches implement LRU-style eviction to prevent memory leaks:

| Cache | Max Size | TTL |
|-------|----------|-----|
| Summary | 500 entries | 30 minutes |
| Goal | 500 entries | 30 minutes |

Eviction is triggered before each cache lookup. Entries are removed based on:
1. TTL expiration (entries older than 30 minutes)
2. Size limit (oldest entries removed when over 500)

### Quick Summary Logic

For working sessions, summaries are generated locally based on the last tool use:

| Tool | Summary Format |
|------|----------------|
| `Edit`, `Write` | "Editing {filename}" |
| `Read` | "Reading {filename}" |
| `Bash` | "Running {command}" |
| `Grep`, `Glob` | "Searching codebase" |
| `Task` | "Running agent task" |
| Other | "Using {tool}" |

### Context Extraction

For AI-generated summaries, context is built from:
- Original prompt
- Current status
- Message count
- Pending tool use flag
- Last 10 entries (truncated)

---

## Utility Modules (`utils/`)

Shared utilities used across the daemon.

### `timeout.ts`

Provides timeout wrapping for async operations.

```typescript
import { withTimeout, TimeoutError } from "./utils/timeout.js";

// Wrap a promise with a timeout
const result = await withTimeout(
  someAsyncOperation(),
  15000, // 15 seconds
  "Operation description"
);

// Handle timeout errors
try {
  await withTimeout(promise, 5000);
} catch (error) {
  if (error instanceof TimeoutError) {
    console.warn(`Timed out after ${error.timeoutMs}ms`);
  }
}
```

### `colors.ts`

ANSI color codes for terminal output.

```typescript
import { colors } from "./utils/colors.js";

console.log(`${colors.green}Success${colors.reset}`);
console.log(`${colors.bold}${colors.cyan}Title${colors.reset}`);
```

Available colors: `reset`, `dim`, `bold`, `green`, `yellow`, `blue`, `cyan`, `magenta`, `red`, `gray`.

### `errors.ts`

Standardized error handling utilities.

```typescript
import { getErrorMessage, logError, logWarn } from "./utils/errors.js";

// Extract message from unknown error
const msg = getErrorMessage(error);

// Log with consistent formatting
logError("github", error, "checking PR");
logWarn("parser", "Skipped malformed entry");
```

---

## Kitty Terminal Control (`kitty-rc.ts`)

Remote control wrapper for kitty terminal using `kitten @` commands.

### Public Functions

#### `health(): Promise<boolean>`

Check if kitty is reachable via the socket.

```typescript
import { KittyRc } from "./kitty-rc.js";

const kitty = new KittyRc();
const available = await kitty.health();
```

#### `focusWindow(windowId: number): Promise<boolean>`

Focus a kitty window by ID. Returns true if successful.

#### `launchTab(opts: LaunchOptions): Promise<number>`

Launch a new kitty tab with options. Returns the window ID.

```typescript
const windowId = await kitty.launchTab({
  cwd: "/path/to/project",
  tabTitle: "Session Tab",
  vars: { cc_session_id: "abc123" },
});
```

#### `sendText(windowId: number, text: string, submit: boolean): Promise<void>`

Send text to a window. If submit is true, appends carriage return.

#### `selectWindow(): Promise<number | null>`

Open interactive window picker. Returns null if cancelled.

### Automatic Configuration

Kitty remote control is **automatically configured** when the daemon starts. No manual setup required.

The daemon:
1. Checks if kitty is installed
2. Creates `~/.config/kitty/claude-code.conf` with remote control settings
3. Adds `include claude-code.conf` to your `kitty.conf`
4. Creates `macos-launch-services-cmdline` for GUI launches (macOS only)
5. Sends SIGUSR1 to reload running kitty instances

For manual setup: `pnpm --filter @claude-code-ui/daemon setup:kitty`

### Generated Config Files

**~/.config/kitty/claude-code.conf:**
```conf
allow_remote_control socket-only
listen_on unix:/tmp/claude-cc-kitty
```

Uses `socket-only` mode for security (no passwords needed).

---

## Kitty Auto-Setup (`kitty-setup.ts`)

Automatic kitty terminal configuration module.

### Public Functions

#### `setupKitty(): Promise<KittySetupResult>`

Run full setup process. Creates config files and reloads kitty.

```typescript
import { setupKitty } from "./kitty-setup.js";

const result = await setupKitty();
// { success: true, status: "ready", message: "...", actions: [...] }
```

#### `getKittyStatus(): Promise<KittyStatusDetails>`

Get detailed status for diagnostics.

```typescript
import { getKittyStatus } from "./kitty-setup.js";

const status = await getKittyStatus();
// { installed, running, socketExists, socketReachable, configExists }
```

### KittyStatus Types

```typescript
type KittyStatus =
  | "not_installed"      // kitty not found
  | "not_running"        // kitty installed but not running
  | "not_configured"     // needs setup
  | "config_needs_reload" // config created, restart kitty
  | "ready"              // fully working
  | "setup_failed";      // setup error
```

---

## Terminal Link API (`api/router.ts`)

HTTP API for terminal control (Port 4451).

### Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/kitty/health` | Check kitty availability with detailed status |
| POST | `/api/kitty/setup` | Trigger manual kitty setup |
| POST | `/api/sessions/:id/focus` | Focus linked terminal |
| POST | `/api/sessions/:id/open` | Open/create terminal |
| POST | `/api/sessions/:id/link-terminal` | Link existing terminal |
| DELETE | `/api/sessions/:id/link-terminal` | Unlink terminal |
| POST | `/api/sessions/:id/send-text` | Send text to terminal |

### Request/Response Examples

**Check kitty health:**
```bash
curl http://127.0.0.1:4451/api/kitty/health
# {
#   "available": true,
#   "details": {
#     "installed": true,
#     "running": true,
#     "socketExists": true,
#     "socketReachable": true,
#     "configExists": true
#   }
# }
```

**Trigger kitty setup:**
```bash
curl -X POST http://127.0.0.1:4451/api/kitty/setup
# {
#   "success": true,
#   "status": "ready",
#   "message": "Kitty remote control configured successfully",
#   "actions": ["Created ~/.config/kitty/claude-code.conf", ...]
# }
```

**Open session terminal:**
```bash
curl -X POST http://127.0.0.1:4451/api/sessions/abc123/open
# { "success": true, "windowId": 42, "created": true }
```

**Send text:**
```bash
curl -X POST http://127.0.0.1:4451/api/sessions/abc123/send-text \
  -H "Content-Type: application/json" \
  -d '{"text": "npm test", "submit": true}'
```

---

## Database (`db/`)

SQLite persistence for terminal links using Drizzle ORM.

### Tables

| Table | Purpose |
|-------|---------|
| `terminal_links` | Session to kitty window mappings |
| `command_history` | Commands sent to terminals |
| `session_preferences` | Per-session UI preferences |

### TerminalLinkRepo

Repository for terminal link CRUD operations.

```typescript
import { TerminalLinkRepo } from "./db/terminal-link-repo.js";

const repo = new TerminalLinkRepo();
const link = repo.get(sessionId);
repo.upsert({ sessionId, kittyWindowId, linkedAt, stale: false });
repo.markStale(sessionId);
```

---

## Related Documentation

- [Configuration Reference](../operations/configuration.md) - All daemon constants
- [Deployment Guide](../operations/deployment.md) - Running the daemon
