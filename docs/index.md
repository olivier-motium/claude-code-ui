# Claude Code Session Tracker Documentation

Real-time monitoring dashboard for Claude Code sessions.

## Quick Links

| I want to... | Read this |
|--------------|-----------|
| Get started quickly | [Getting Started](getting-started.md) |
| Run in production | [Deployment Guide](operations/deployment.md) |
| Configure the app | [Configuration Reference](operations/configuration.md) |
| Understand the CLI | [CLI Reference](cli-reference.md) |
| Learn the architecture | [README](../README.md) |

---

## Documentation Map

### For New Contributors

| Document | Purpose |
|----------|---------|
| [Getting Started](getting-started.md) | Setup and first run |
| [README](../README.md) | Project overview and architecture |
| [CLAUDE.md](../CLAUDE.md) | Coding guidelines and conventions |
| [Testing Guide](guides/testing.md) | Manual testing strategies |

### For Operators

| Document | Purpose |
|----------|---------|
| [Deployment Guide](operations/deployment.md) | Running as a service |
| [Configuration Reference](operations/configuration.md) | Environment variables and tunables |
| [CLI Reference](cli-reference.md) | Watcher flags and options |

### For Developers

| Document | Purpose |
|----------|---------|
| [UI Components](ui-components.md) | React component hierarchy |
| [Daemon APIs](api/daemon-api.md) | Internal service documentation |
| [spec.md](../spec.md) | Log format specification and types |

### Reference Documentation

External knowledge bases for related technologies:

| Guide | Description |
|-------|-------------|
| [PydanticAI Reference](PydanticAI/INDEX.md) | AI agent framework documentation |
| [Logfire Guide](Logfire_cleaned/Index.md) | Observability platform guide |

---

## Key Concepts

| Concept | Description | Location |
|---------|-------------|----------|
| **Session Watcher** | Monitors `~/.claude/projects/` for JSONL changes | [README](../README.md#daemon) |
| **Status Machine** | XState state machine for session status | [README](../README.md#session-status-state-machine) |
| **File-Based Status** | Claude Code hooks write status to `.claude/status.md` for goal/summary | [Daemon API](api/daemon-api.md#file-based-status-system-status-watcherts-status-parserts) |
| **Durable Streams** | Real-time state sync between daemon and UI | [CLAUDE.md](../CLAUDE.md#durable-streams-integration) |

---

## Project Structure

```
claude-code-ui/
├── packages/
│   ├── daemon/     # Node.js file watcher + stream server
│   └── ui/         # React dashboard
├── docs/           # Extended documentation (you are here)
│   ├── api/        # Internal API documentation
│   ├── guides/     # How-to guides
│   ├── operations/ # Deployment and configuration
│   ├── Logfire_cleaned/  # Logfire knowledge base
│   └── PydanticAI/ # PydanticAI knowledge base
└── *.md            # Root-level docs
```

---

## Getting Help

- Check [Testing Guide](guides/testing.md) for debugging strategies
- Review the [CLI Reference](cli-reference.md) for watcher options
- See [Configuration](operations/configuration.md) for all tunables
- See [Deployment](operations/deployment.md) for production setup
