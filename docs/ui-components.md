# UI Components

The React UI is built with TanStack Router and Radix UI Themes.

## Component Hierarchy

```
__root.tsx (Radix Theme provider)
└── index.tsx (Main board view)
    └── RepoSection (per GitHub repo)
        └── KanbanColumn (per status: Working/Waiting/Idle)
            └── SessionCard (individual session)
```

---

## Core Components

### `__root.tsx`

Root layout that wraps the app in Radix UI Theme provider.

**Location:** `packages/ui/src/routes/__root.tsx`

**Responsibilities:**
- Radix Theme configuration
- Global styles
- Router outlet

### `index.tsx`

Main board view that displays all sessions grouped by repository.

**Location:** `packages/ui/src/routes/index.tsx`

**Responsibilities:**
- Fetches sessions via `useSessions()` hook
- Groups sessions by `gitRepoId`
- Renders `RepoSection` for each repository

---

## Display Components

### `RepoSection`

Groups sessions belonging to the same GitHub repository.

**Location:** `packages/ui/src/components/RepoSection.tsx`

**Props:**
- `repoId: string` - Repository identifier (e.g., "owner/repo")
- `repoUrl: string | null` - GitHub URL for linking
- `sessions: Session[]` - Sessions in this repo

**Features:**
- Repository name with GitHub link
- Activity score (weighted sum of non-idle sessions)
- Collapsible session list

### `KanbanColumn`

Displays sessions filtered by status.

**Location:** `packages/ui/src/components/KanbanColumn.tsx`

**Props:**
- `status: SessionStatus` - "working" | "waiting" | "idle"
- `sessions: Session[]` - Sessions to display

**Status Colors:**
| Status | Color |
|--------|-------|
| working | Green |
| waiting | Yellow/Orange |
| idle | Gray |

### `SessionCard`

Individual session card with details.

**Location:** `packages/ui/src/components/SessionCard.tsx`

**Displays:**
- **Goal**: AI-generated high-level objective
- **Summary**: Current activity description
- **Status indicator**: Color-coded badge
- **PR badge**: If branch has open PR, shows number and CI status
- **Tool icons**: Shows pending tool with icon

**Tool Icons:**
| Tool | Icon |
|------|------|
| Edit | Pencil |
| Write | Document |
| Read | Book |
| Bash | Terminal |
| Grep | Magnifying glass |
| Glob | Folder |
| Task | Robot |

---

## Data Layer

The UI uses **Durable Streams** (`@durable-streams/state`) for real-time session synchronization:

- `createStreamDB()` creates the reactive state container
- Subscribes to SSE endpoint at `VITE_STREAM_URL` (default: `http://127.0.0.1:4450/sessions`)
- Provides `useSessions()` hook for React components

### `useSessions` Hook

React hook for subscribing to session updates.

**Location:** `packages/ui/src/hooks/useSessions.ts`

**Usage:**
```tsx
import { useSessions } from '../hooks/useSessions';

function MyComponent() {
  const sessions = useSessions();
  // sessions: Map<sessionId, Session>

  return (
    <div>
      {Array.from(sessions.values()).map(session => (
        <SessionCard key={session.sessionId} session={session} />
      ))}
    </div>
  );
}
```

**Behavior:**
- Connects to Durable Streams via SSE
- Automatically reconnects on disconnect
- Updates trigger re-renders via React's reactive state

### `sessionsDb`

Singleton StreamDB connection.

**Location:** `packages/ui/src/data/sessionsDb.ts`

**Functions:**
- `getSessionsDb()` - Async getter, initializes connection
- `getSessionsDbSync()` - Sync getter (throws if not initialized)
- `closeSessionsDb()` - Closes connection

**Configuration:**
- `VITE_STREAM_URL` - Override the daemon endpoint (default: `http://127.0.0.1:4450/sessions`)

---

## Styling Guidelines

From `packages/ui/CLAUDE.md`:

1. **Always use Radix UI components** - Never use plain HTML elements with custom styles
2. **Let Radix handle typography** - Don't set `fontSize` or `lineHeight` manually
3. **Use Radix style props** - `size`, `color`, `variant` instead of inline styles
4. **Code content** - Use the `Code` component for monospace text
