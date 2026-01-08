# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Claude Code Session Tracker - A real-time dashboard for monitoring Claude Code sessions across multiple projects. Watches `~/.claude/projects/` for session log changes, derives status using XState, generates AI summaries, and streams updates to a React UI via Durable Streams.

## Development Commands

```bash
# Install dependencies
pnpm install

# Start both daemon and UI (recommended)
pnpm start

# Run separately:
pnpm serve           # Start daemon on port 4450
pnpm dev             # Start UI dev server

# Daemon watch modes:
pnpm watch           # CLI watcher (no streaming server)
pnpm watch:recent    # Only sessions from last 24h
pnpm watch:active    # Only non-idle sessions

# Testing:
cd packages/daemon && pnpm test        # Run tests
cd packages/daemon && pnpm test:watch  # Watch mode

# Build:
cd packages/ui && pnpm build           # Build UI for production
cd packages/daemon && pnpm build       # TypeScript compile
```

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Claude Code    │     │     Daemon      │     │       UI        │
│   Sessions      │────▶│   (Watcher)     │────▶│   (React)       │
│  ~/.claude/     │     │                 │     │                 │
│   projects/     │     │  Durable Stream │     │  TanStack DB    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Daemon (`packages/daemon`)

- **`watcher.ts`** - Chokidar file watcher for `~/.claude/projects/**/*.jsonl`. Emits session events (created/updated/deleted). Tracks byte positions for incremental reads.
- **`parser.ts`** - JSONL parser with incremental reading (`tailJSONL`). Extracts session metadata and handles partial lines.
- **`status-machine.ts`** - XState state machine for session status detection:
  - States: `idle`, `working`, `waiting_for_approval`, `waiting_for_input`
  - Events: `USER_PROMPT`, `TOOL_RESULT`, `ASSISTANT_STREAMING`, `ASSISTANT_TOOL_USE`, `TURN_END`, timeout events
  - Maps 4 internal states to 3 UI states (working/waiting/idle)
- **`server.ts`** - Durable Streams server wrapper. Publishes `Session` state to stream, handles PR update callbacks.
- **`summarizer.ts`** - AI-powered goal/summary generation via Claude Sonnet API
- **`github.ts`** - PR/CI status polling and caching
- **`git.ts`** - Git repo info extraction (branch, remote URL)
- **`schema.ts`** - Zod schemas for session state, exported via `@claude-code-ui/daemon/schema`

### UI (`packages/ui`)

- **TanStack Router** for routing with file-based routes in `src/routes/`
- **Radix UI Themes** for all components (never use plain HTML with custom styles)
- **TanStack DB** for reactive local state from Durable Streams
- **`src/data/sessionsDb.ts`** - Singleton StreamDB connection to daemon
- **`src/hooks/useSessions.ts`** - React hook for session data
- **`src/components/`** - KanbanColumn, RepoSection, SessionCard

## Key Data Types

### Session Status State Machine

```
                    ┌─────────────────┐
                    │      idle       │
                    └────────┬────────┘
                             │ USER_PROMPT
                             ▼
┌─────────────────┐  TOOL_RESULT  ┌─────────────────┐
│ waiting_for_    │◄──────────────│     working     │
│   approval      │               └────────┬────────┘
└────────┬────────┘                        │
         │                    TURN_END / STALE_TIMEOUT
         │                                 │
         └────────────────────────────────▶│
                  IDLE_TIMEOUT     ┌─────────────────┐
                                   │ waiting_for_    │
                                   │     input       │
                                   └─────────────────┘
```

### Log Entry Types

The JSONL files contain discriminated union entries (`type` field):
- `user` - User prompts (string content) and tool results (array with `tool_result`)
- `assistant` - Claude responses with `TextBlock`, `ToolUseBlock`, `ThinkingBlock`
- `system` - Hook summaries, turn duration markers (`subtype: "turn_duration"` or `"stop_hook_summary"`)
- `queue-operation` - Queued prompts
- `file-history-snapshot` - File state tracking

### Timeouts (in status-machine.ts)

- **5 seconds** - If tool_use pending → `waiting_for_approval`
- **60 seconds** - Stale working state → `waiting_for_input`
- **5 minutes** - No activity → `idle`

## UI Guidelines

- Always use Radix UI components - never use plain HTML elements with custom styles
- Let Radix and capsize handle typography sizing - don't set fontSize or lineHeight manually
- Use Radix's style props (size, color, variant, etc.) instead of inline styles
- For code/monospace content, use the `Code` component

## Environment Variables

```bash
ANTHROPIC_API_KEY=sk-ant-...  # Required for AI summaries
GITHUB_TOKEN=ghp_...          # Optional: for PR/CI status (uses gh CLI auth if not set)
```

## Important Patterns

### Incremental JSONL Reading

The daemon tracks byte positions per file to avoid re-reading entire logs. On file change, only new bytes are read and parsed. Partial lines (incomplete writes) are detected and skipped until complete.

### Durable Streams Integration

- Daemon runs server on port 4450 at `/sessions` endpoint
- UI connects via `@durable-streams/client` with SSE for live updates
- Schema exported from `@claude-code-ui/daemon/schema` for type safety

### Status Detection

Uses XState for deterministic state transitions. The machine processes all log entries in order, then applies timeout checks based on `lastActivityAt`. This ensures consistent status even when replaying from log files.

## File Locations

- Claude session logs: `~/.claude/projects/<encoded-dir>/<session-id>.jsonl`
- Daemon stream data: `~/.claude-code-ui/streams/`
- Encoded directory format: `/Users/foo/bar` → `-Users-foo-bar`

## Tech Stack

| Component | Technology |
|-----------|------------|
| Runtime | Node.js 22.13.1 |
| Package Manager | pnpm 10.26.0 |
| File Watching | chokidar v5 (ESM-only) |
| State Machine | XState v5 |
| Streaming | @durable-streams/* |
| UI Framework | React 19 |
| Routing | TanStack Router |
| UI Components | Radix UI Themes |
| Validation | Zod |
| Build | Vite |
