/**
 * XState state machine for session status detection.
 *
 * This replaces the ad-hoc if-statements with a proper state machine
 * that makes transitions explicit and testable.
 */

import { setup, createActor } from "xstate";
import type { LogEntry, AssistantEntry, UserEntry, SystemEntry } from "./types.js";

// Context holds computed state from log entries
export interface StatusContext {
  lastActivityAt: string;
  messageCount: number;
  hasPendingToolUse: boolean;
  pendingToolIds: string[];
}

// Events derived from log entries
export type StatusEvent =
  | { type: "USER_PROMPT"; timestamp: string }
  | { type: "TOOL_RESULT"; timestamp: string; toolUseIds: string[] }
  | { type: "ASSISTANT_STREAMING"; timestamp: string }
  | { type: "ASSISTANT_TOOL_USE"; timestamp: string; toolUseIds: string[] }
  | { type: "TURN_END"; timestamp: string }
  | { type: "IDLE_TIMEOUT" }
  | { type: "APPROVAL_TIMEOUT" }
  | { type: "STALE_TIMEOUT" };

// The four possible status states
export type StatusState = "idle" | "working" | "waiting_for_approval" | "waiting_for_input";

/**
 * State machine for session status.
 *
 * States:
 * - idle: No activity for 5+ minutes
 * - working: Claude is actively processing
 * - waiting_for_approval: Tool use needs user approval
 * - waiting_for_input: Claude finished, waiting for user
 */
export const statusMachine = setup({
  types: {
    context: {} as StatusContext,
    events: {} as StatusEvent,
  },
}).createMachine({
  id: "sessionStatus",
  initial: "waiting_for_input",
  context: {
    lastActivityAt: "",
    messageCount: 0,
    hasPendingToolUse: false,
    pendingToolIds: [],
  },
  states: {
    idle: {
      on: {
        USER_PROMPT: {
          target: "working",
          actions: ({ context, event }) => {
            context.lastActivityAt = event.timestamp;
            context.messageCount += 1;
          },
        },
      },
    },
    working: {
      on: {
        ASSISTANT_STREAMING: {
          actions: ({ context, event }) => {
            context.lastActivityAt = event.timestamp;
          },
        },
        ASSISTANT_TOOL_USE: {
          target: "waiting_for_approval",
          actions: ({ context, event }) => {
            context.lastActivityAt = event.timestamp;
            context.messageCount += 1;
            context.hasPendingToolUse = true;
            context.pendingToolIds = event.toolUseIds;
          },
        },
        TURN_END: {
          target: "waiting_for_input",
          actions: ({ context, event }) => {
            context.lastActivityAt = event.timestamp;
            context.hasPendingToolUse = false;
            context.pendingToolIds = [];
          },
        },
        STALE_TIMEOUT: {
          target: "waiting_for_input",
          actions: ({ context }) => {
            context.hasPendingToolUse = false;
          },
        },
        IDLE_TIMEOUT: {
          target: "idle",
        },
      },
    },
    waiting_for_approval: {
      on: {
        TOOL_RESULT: {
          target: "working",
          actions: ({ context, event }) => {
            context.lastActivityAt = event.timestamp;
            context.messageCount += 1;
            // Remove approved tools from pending
            const remaining = context.pendingToolIds.filter(
              (id) => !event.toolUseIds.includes(id)
            );
            context.pendingToolIds = remaining;
            context.hasPendingToolUse = remaining.length > 0;
          },
        },
        IDLE_TIMEOUT: {
          target: "idle",
        },
      },
    },
    waiting_for_input: {
      on: {
        USER_PROMPT: {
          target: "working",
          actions: ({ context, event }) => {
            context.lastActivityAt = event.timestamp;
            context.messageCount += 1;
          },
        },
        IDLE_TIMEOUT: {
          target: "idle",
        },
      },
    },
  },
});

/**
 * Convert a log entry to a status event.
 */
export function logEntryToEvent(entry: LogEntry): StatusEvent | null {
  if (entry.type === "user") {
    const userEntry = entry as UserEntry;
    const content = userEntry.message.content;

    if (typeof content === "string") {
      // Human prompt
      return { type: "USER_PROMPT", timestamp: userEntry.timestamp };
    } else if (Array.isArray(content)) {
      // Tool result(s)
      const toolUseIds = content
        .filter((b) => b.type === "tool_result")
        .map((b) => b.tool_use_id);
      if (toolUseIds.length > 0) {
        return { type: "TOOL_RESULT", timestamp: userEntry.timestamp, toolUseIds };
      }
    }
  }

  if (entry.type === "assistant") {
    const assistantEntry = entry as AssistantEntry;
    const toolUseBlocks = assistantEntry.message.content.filter((b) => b.type === "tool_use");

    if (toolUseBlocks.length > 0) {
      const toolUseIds = toolUseBlocks.map((b) => b.type === "tool_use" ? b.id : "");
      return { type: "ASSISTANT_TOOL_USE", timestamp: assistantEntry.timestamp, toolUseIds };
    }

    // Streaming assistant message (no tool_use)
    return { type: "ASSISTANT_STREAMING", timestamp: assistantEntry.timestamp };
  }

  if (entry.type === "system") {
    const systemEntry = entry as SystemEntry;
    if (systemEntry.subtype === "turn_duration" || systemEntry.subtype === "stop_hook_summary") {
      return { type: "TURN_END", timestamp: systemEntry.timestamp };
    }
  }

  return null;
}

/**
 * Derive status by running all log entries through the state machine.
 */
export function deriveStatusFromMachine(entries: LogEntry[]): {
  status: StatusState;
  context: StatusContext;
} {
  // Create a fresh actor and start it
  const actor = createActor(statusMachine);
  actor.start();

  // Process each entry
  for (const entry of entries) {
    const event = logEntryToEvent(entry);
    if (event) {
      actor.send(event);
    }
  }

  // Get current state
  const snapshot = actor.getSnapshot();
  let context = snapshot.context;
  let stateValue = snapshot.value as StatusState;

  // Check for timeouts based on last activity
  const now = Date.now();
  const lastActivityTime = context.lastActivityAt ? new Date(context.lastActivityAt).getTime() : 0;
  const timeSinceActivity = now - lastActivityTime;

  const IDLE_TIMEOUT_MS = 5 * 60 * 1000; // 5 minutes
  const APPROVAL_TIMEOUT_MS = 5 * 1000; // 5 seconds
  const STALE_TIMEOUT_MS = 60 * 1000; // 60 seconds

  // Apply timeout transitions
  if (timeSinceActivity > IDLE_TIMEOUT_MS) {
    actor.send({ type: "IDLE_TIMEOUT" });
  } else if (stateValue === "working" && context.hasPendingToolUse && timeSinceActivity > APPROVAL_TIMEOUT_MS) {
    // Tool use pending for too long - should be in waiting_for_approval
    actor.send({ type: "APPROVAL_TIMEOUT" });
  } else if (stateValue === "working" && !context.hasPendingToolUse && timeSinceActivity > STALE_TIMEOUT_MS) {
    // Stale without tool use - probably turn ended without marker
    actor.send({ type: "STALE_TIMEOUT" });
  }

  // Get final state
  const finalSnapshot = actor.getSnapshot();
  actor.stop();

  return {
    status: finalSnapshot.value as StatusState,
    context: finalSnapshot.context,
  };
}

/**
 * Map machine status to the existing StatusResult format for compatibility.
 */
export function machineStatusToResult(
  machineStatus: StatusState,
  context: StatusContext
): {
  status: "working" | "waiting" | "idle";
  lastRole: "user" | "assistant";
  hasPendingToolUse: boolean;
  lastActivityAt: string;
  messageCount: number;
} {
  // Map the 4 machine states to the 3 UI states
  let status: "working" | "waiting" | "idle";
  switch (machineStatus) {
    case "working":
      status = "working";
      break;
    case "waiting_for_approval":
    case "waiting_for_input":
      status = "waiting";
      break;
    case "idle":
      status = "idle";
      break;
  }

  return {
    status,
    lastRole: "assistant", // Could track this in context if needed
    hasPendingToolUse: context.hasPendingToolUse,
    lastActivityAt: context.lastActivityAt,
    messageCount: context.messageCount,
  };
}
