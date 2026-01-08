import type {
  LogEntry,
  StatusResult,
  SessionStatus,
} from "./types.js";
import {
  deriveStatusFromMachine,
  machineStatusToResult,
} from "./status-machine.js";

const DEFAULT_IDLE_THRESHOLD_MS = 5 * 60 * 1000; // 5 minutes

/**
 * Derive session status from log entries using XState state machine.
 *
 * Status logic:
 * - "working": Claude is actively processing (streaming or executing tools)
 * - "waiting": Claude finished, waiting for user input or approval
 *   - hasPendingToolUse: true if waiting for tool approval
 * - "idle": No activity for idleThresholdMs
 */
export function deriveStatus(
  entries: LogEntry[],
  _idleThresholdMs: number = DEFAULT_IDLE_THRESHOLD_MS,
): StatusResult {
  // Use the state machine for status derivation
  const { status: machineStatus, context } = deriveStatusFromMachine(entries);
  const result = machineStatusToResult(machineStatus, context);

  console.log(`[Status] Machine: state=${machineStatus}, hasPendingToolUse=${context.hasPendingToolUse}, messageCount=${context.messageCount}`);

  return result;
}

/**
 * Compare two status results to detect meaningful changes.
 */
export function statusChanged(
  prev: StatusResult | null | undefined,
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
