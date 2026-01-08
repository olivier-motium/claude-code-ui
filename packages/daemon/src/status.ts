import type {
  LogEntry,
  StatusResult,
  SessionStatus,
  UserEntry,
  AssistantEntry,
  ToolUseBlock,
} from "./types.js";

const DEFAULT_IDLE_THRESHOLD_MS = 5 * 60 * 1000; // 5 minutes
const DEFAULT_WORKING_TIMEOUT_MS = 30 * 1000; // 30 seconds - if no response after this, not "working"

/**
 * Derive session status from log entries.
 *
 * Status logic:
 * - "working": Last message was from user AND it was recent (within workingTimeoutMs)
 * - "waiting": Last message was from assistant, or user message is stale
 *   - hasPendingToolUse: true if last assistant message has unresolved tool_use
 * - "idle": No activity for idleThresholdMs
 */
export function deriveStatus(
  entries: LogEntry[],
  idleThresholdMs: number = DEFAULT_IDLE_THRESHOLD_MS,
  workingTimeoutMs: number = DEFAULT_WORKING_TIMEOUT_MS
): StatusResult {
  // Filter to message entries only
  const messageEntries = entries.filter(
    (e): e is UserEntry | AssistantEntry =>
      e.type === "user" || e.type === "assistant"
  );

  if (messageEntries.length === 0) {
    return {
      status: "waiting",
      lastRole: "user",
      hasPendingToolUse: false,
      lastActivityAt: "",
      messageCount: 0,
    };
  }

  const lastEntry = messageEntries[messageEntries.length - 1];
  const lastActivityAt = lastEntry.timestamp;

  // Check for idle
  const lastActivityTime = new Date(lastActivityAt).getTime();
  const now = Date.now();
  const isIdle = now - lastActivityTime > idleThresholdMs;

  if (isIdle) {
    return {
      status: "idle",
      lastRole: lastEntry.type === "user" ? "user" : "assistant",
      hasPendingToolUse: false,
      lastActivityAt,
      messageCount: messageEntries.length,
    };
  }

  // Check for pending tool use in last assistant message
  let hasPendingToolUse = false;

  if (lastEntry.type === "assistant") {
    // Get tool_use IDs from the last assistant message
    const toolUseIds = new Set<string>();
    for (const block of lastEntry.message.content) {
      if (block.type === "tool_use") {
        toolUseIds.add(block.id);
      }
    }

    // Since this is the last entry, any tool_use blocks are pending
    // (no subsequent tool_result could exist yet)
    hasPendingToolUse = toolUseIds.size > 0;
  }

  // Determine status based on last role and pending tools
  let status: SessionStatus;

  if (lastEntry.type === "user") {
    // Check if this is a tool_result (array) vs human prompt (string)
    const isToolResult = Array.isArray(lastEntry.message.content);
    const timeSinceUserMessage = now - lastActivityTime;

    console.log(`[Status] User entry: isToolResult=${isToolResult}, timeSince=${Math.round(timeSinceUserMessage/1000)}s`);

    if (isToolResult) {
      // Tool result means Claude is processing - always "working"
      // (tools can take a long time, especially Task agents)
      status = "working";
    } else {
      // Human prompt - check timeout
      if (timeSinceUserMessage > workingTimeoutMs) {
        // User message is stale - Claude probably isn't working
        // (session was interrupted, or Claude crashed, etc.)
        status = "waiting";
      } else {
        status = "working";
      }
    }
  } else {
    // Last entry is assistant message
    console.log(`[Status] Assistant entry: hasPendingToolUse=${hasPendingToolUse}`);
    if (hasPendingToolUse) {
      // Tool is executing - this is "working" not "waiting"
      status = "working";
    } else {
      // Claude finished, waiting for user input
      status = "waiting";
    }
  }

  return {
    status,
    lastRole: lastEntry.type === "user" ? "user" : "assistant",
    hasPendingToolUse,
    lastActivityAt,
    messageCount: messageEntries.length,
  };
}

/**
 * Compare two status results to detect meaningful changes.
 */
export function statusChanged(
  prev: StatusResult | null,
  next: StatusResult
): boolean {
  if (!prev) return true;

  return (
    prev.status !== next.status ||
    prev.lastRole !== next.lastRole ||
    prev.hasPendingToolUse !== next.hasPendingToolUse
  );
}

/**
 * Format status for display.
 */
export function formatStatus(result: StatusResult): string {
  const icons: Record<SessionStatus, string> = {
    working: "🟢",
    waiting: result.hasPendingToolUse ? "🟠" : "🟡",
    idle: "⚪",
  };

  const labels: Record<SessionStatus, string> = {
    working: "Working",
    waiting: result.hasPendingToolUse ? "Tool pending" : "Waiting for input",
    idle: "Idle",
  };

  return `${icons[result.status]} ${labels[result.status]}`;
}

/**
 * Get a short status string for logging.
 */
export function getStatusKey(result: StatusResult): string {
  if (result.status === "waiting" && result.hasPendingToolUse) {
    return "waiting:tool";
  }
  return result.status;
}
