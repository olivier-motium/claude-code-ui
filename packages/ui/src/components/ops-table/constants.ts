/**
 * Constants for OpsTable components
 */

export const TOOL_ICONS: Record<string, string> = {
  Edit: "✏️",
  Write: "📝",
  Bash: "▶️",
  Read: "📖",
  Grep: "🔍",
  Glob: "📁",
  MultiEdit: "✏️",
  Task: "🤖",
  TodoWrite: "📋",
  WebSearch: "🌐",
  WebFetch: "🔗",
};

export const STATUS_COLORS: Record<string, string> = {
  working: "green",
  waiting: "orange",
  idle: "gray",
};

export const STATUS_ICONS: Record<string, string> = {
  working: "●",
  waiting: "○",
  idle: "◐",
};

export const STATUS_LABELS: Record<string, string> = {
  working: "Working - Claude is actively processing",
  waiting: "Waiting - Needs your input or approval",
  idle: "Idle - Session is inactive",
};

/** Stale threshold: 10 minutes without activity while working */
export const STALE_THRESHOLD_MS = 10 * 60 * 1000;
