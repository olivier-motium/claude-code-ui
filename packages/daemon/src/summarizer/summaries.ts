/**
 * Quick summary functions that don't require API calls.
 */

import type { SessionState } from "../watcher.js";

/**
 * Get a quick summary for working sessions (no API call needed)
 */
export function getWorkingSummary(session: SessionState): string {
  const { entries } = session;
  const lastAssistant = entries.findLast((e) => e.type === "assistant");

  if (lastAssistant) {
    const toolBlocks = lastAssistant.message.content.filter(
      (b): b is { type: "tool_use"; id: string; name: string; input: Record<string, unknown> } =>
        b.type === "tool_use"
    );

    if (toolBlocks.length > 0) {
      const tool = toolBlocks[0].name;
      const input = toolBlocks[0].input;

      if (tool === "Edit" || tool === "Write") {
        const file = (input?.file_path as string)?.split("/").pop() || "file";
        return `Editing ${file}`;
      }
      if (tool === "Read") {
        const file = (input?.file_path as string)?.split("/").pop() || "file";
        return `Reading ${file}`;
      }
      if (tool === "Bash") {
        const cmd = ((input?.command as string) || "").split(" ")[0];
        return `Running ${cmd}`;
      }
      if (tool === "Grep" || tool === "Glob") {
        return "Searching codebase";
      }
      if (tool === "Task") {
        return "Running agent task";
      }
      return `Using ${tool}`;
    }
  }

  return "Processing...";
}

/**
 * Fallback summary when AI is unavailable
 */
export function getFallbackSummary(session: SessionState): string {
  const { status, originalPrompt } = session;

  if (status.hasPendingToolUse) {
    return "Waiting for approval";
  }

  if (status.status === "waiting") {
    return "Waiting for input";
  }

  // Extract first few words of original prompt
  const words = originalPrompt.split(" ").slice(0, 4).join(" ");
  return words.length < originalPrompt.length ? `${words}...` : words;
}
