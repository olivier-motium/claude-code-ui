/**
 * Context extraction functions for summarization.
 */

import type { SessionState } from "../watcher.js";
import type { LogEntry } from "../types.js";
import {
  CONTENT_TRUNCATE_LENGTH,
  USER_PROMPT_TRUNCATE_LENGTH,
  CONTEXT_TEXT_LENGTH,
  SHORT_CONTEXT_LENGTH,
  SHORT_USER_CONTENT_LENGTH,
  EARLY_ENTRIES_COUNT,
  RECENT_ENTRIES_COUNT,
} from "../config.js";

/**
 * Extract context for summarization
 */
export function extractContext(session: SessionState): string {
  const { entries, status, originalPrompt } = session;

  // Get recent meaningful entries
  const recentEntries = entries.slice(-RECENT_ENTRIES_COUNT);
  const context: string[] = [];

  context.push(`Original task: ${originalPrompt}`);
  context.push(`Current status: ${status.status}`);
  context.push(`Messages: ${status.messageCount}`);

  if (status.hasPendingToolUse) {
    context.push("Has pending tool use awaiting approval");
  }

  context.push("\nRecent activity:");

  for (const entry of recentEntries) {
    if (entry.type === "assistant") {
      for (const block of entry.message.content) {
        if (block.type === "text") {
          context.push(`Claude: ${block.text.slice(0, CONTENT_TRUNCATE_LENGTH)}`);
        } else if (block.type === "tool_use") {
          context.push(`Tool: ${block.name}`);
        }
      }
    } else if (entry.type === "user" && typeof entry.message.content === "string") {
      context.push(`User: ${entry.message.content.slice(0, USER_PROMPT_TRUNCATE_LENGTH)}`);
    }
  }

  return context.join("\n");
}

/**
 * Extract context from early session entries for goal generation.
 */
export function extractEarlyContext(entries: LogEntry[]): string[] {
  const context: string[] = [];
  const earlyEntries = entries.slice(0, EARLY_ENTRIES_COUNT);
  for (const entry of earlyEntries) {
    if (entry.type === "assistant") {
      const textBlock = entry.message.content.find((b) => b.type === "text");
      if (textBlock && textBlock.type === "text") {
        context.push(`Claude: ${textBlock.text.slice(0, CONTEXT_TEXT_LENGTH)}`);
      }
    }
  }
  return context;
}

/**
 * Extract context from recent session entries for goal generation.
 */
export function extractRecentGoalContext(entries: LogEntry[]): string[] {
  const context: string[] = [];
  const recentEntries = entries.slice(-RECENT_ENTRIES_COUNT);
  for (const entry of recentEntries) {
    if (entry.type === "assistant") {
      const tools = entry.message.content.filter((b) => b.type === "tool_use");
      if (tools.length > 0) {
        const toolNames = tools.map((t) => t.name).join(", ");
        context.push(`Tools used: ${toolNames}`);
      }
      const textBlock = entry.message.content.find(
        (b): b is { type: "text"; text: string } => b.type === "text"
      );
      if (textBlock) {
        context.push(`Claude: ${textBlock.text.slice(0, SHORT_CONTEXT_LENGTH)}`);
      }
    } else if (entry.type === "user" && typeof entry.message.content === "string") {
      context.push(`User: ${entry.message.content.slice(0, SHORT_USER_CONTENT_LENGTH)}`);
    }
  }
  return context;
}
