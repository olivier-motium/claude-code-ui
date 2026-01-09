/**
 * Tool registry - centralized configuration for Claude Code tools.
 *
 * Provides icons, formatters, and target extractors for all known tools.
 * This eliminates hardcoded tool names scattered throughout the codebase.
 */

// Import directly from content.ts to avoid pulling in Node-only config (stream.ts uses process.env)
import { COMMAND_TRUNCATE_LENGTH, SHORT_CONTENT_LENGTH } from "../config/content.js";

// =============================================================================
// Types
// =============================================================================

/** Input for tool operations - generic record from Claude tool_use blocks */
export type ToolInput = Record<string, unknown>;

/** Configuration for a single tool */
export interface ToolConfig {
  /** Emoji icon for display */
  icon: string;
  /** Format tool use for display (e.g., "Reading path/to/file") */
  format: (input: ToolInput) => string;
  /** Extract the target value from input (e.g., file_path, command) */
  extractTarget: (input: ToolInput) => string;
}

// =============================================================================
// Helpers
// =============================================================================

/**
 * Shorten file path for display.
 */
function shortenPath(filepath: string | undefined): string {
  if (!filepath) return "file";
  const parts = filepath.split("/");
  return parts.length > 2 ? `.../${parts.slice(-2).join("/")}` : filepath;
}

/**
 * Truncate string to a maximum length.
 */
function truncate(value: string | undefined, maxLen: number): string {
  if (!value) return "";
  return value.length > maxLen ? `${value.slice(0, maxLen)}...` : value;
}

// =============================================================================
// Tool Registry
// =============================================================================

/**
 * Registry of all known Claude Code tools with their display configurations.
 */
export const TOOL_REGISTRY: Record<string, ToolConfig> = {
  Read: {
    icon: "📖",
    format: (input) => `📖 Reading ${shortenPath(input.file_path as string)}`,
    extractTarget: (input) => (input.file_path as string) ?? "",
  },
  Edit: {
    icon: "✏️",
    format: (input) => `✏️ Editing ${shortenPath(input.file_path as string)}`,
    extractTarget: (input) => (input.file_path as string) ?? "",
  },
  Write: {
    icon: "📝",
    format: (input) => `📝 Writing ${shortenPath(input.file_path as string)}`,
    extractTarget: (input) => (input.file_path as string) ?? "",
  },
  Bash: {
    icon: "▶️",
    format: (input) => `▶️ Running: ${truncate(input.command as string, COMMAND_TRUNCATE_LENGTH)}`,
    extractTarget: (input) => (input.command as string) ?? "",
  },
  Grep: {
    icon: "🔍",
    format: (input) => `🔍 Searching for "${input.pattern}"`,
    extractTarget: (input) => (input.pattern as string) ?? "",
  },
  Glob: {
    icon: "📁",
    format: (input) => `📁 Finding files: ${input.pattern}`,
    extractTarget: (input) => (input.pattern as string) ?? "",
  },
  Task: {
    icon: "🤖",
    format: (input) => `🤖 Spawning agent: ${(input.description as string) || "task"}`,
    extractTarget: (input) => (input.description as string) ?? "",
  },
  TodoWrite: {
    icon: "📋",
    format: () => `📋 Updating todo list`,
    extractTarget: () => "",
  },
  WebSearch: {
    icon: "🌐",
    format: (input) => `🌐 Searching: "${truncate(input.query as string, 40)}"`,
    extractTarget: (input) => (input.query as string) ?? "",
  },
  WebFetch: {
    icon: "🔗",
    format: (input) => `🔗 Fetching: ${truncate(input.url as string, 50)}`,
    extractTarget: (input) => (input.url as string) ?? "",
  },
} as const;

/** Known tool names */
export type ToolName = keyof typeof TOOL_REGISTRY;

/** List of all known tool names */
export const TOOL_NAMES = Object.keys(TOOL_REGISTRY) as ToolName[];

// =============================================================================
// Public API
// =============================================================================

/**
 * Format a tool use for display.
 * Falls back to a generic format for unknown tools.
 */
export function formatToolUse(tool: string, input: ToolInput): string {
  const config = TOOL_REGISTRY[tool];
  if (config) {
    return config.format(input);
  }
  // Fallback for unknown tools
  return `🔧 ${tool}`;
}

/**
 * Get the icon for a tool.
 */
export function getToolIcon(tool: string): string {
  return TOOL_REGISTRY[tool]?.icon ?? "🔧";
}

/**
 * Extract the target value from tool input.
 * Falls back to JSON stringification for unknown tools.
 */
export function extractToolTarget(tool: string, input: ToolInput): string {
  const config = TOOL_REGISTRY[tool];
  if (config) {
    return config.extractTarget(input);
  }
  // Fallback: JSON stringify with truncation
  return JSON.stringify(input).slice(0, SHORT_CONTENT_LENGTH);
}
