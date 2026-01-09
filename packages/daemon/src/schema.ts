import { z } from "zod";
import { createStateSchema } from "@durable-streams/state";

// Session status enum
export const SessionStatusSchema = z.enum(["working", "waiting", "idle"]);
export type SessionStatus = z.infer<typeof SessionStatusSchema>;

// Pending tool info
export const PendingToolSchema = z.object({
  tool: z.string(),
  target: z.string(),
});
export type PendingTool = z.infer<typeof PendingToolSchema>;

// Recent output entry for live view
export const RecentOutputSchema = z.object({
  role: z.enum(["user", "assistant", "tool"]),
  content: z.string(),
});
export type RecentOutput = z.infer<typeof RecentOutputSchema>;

// Terminal link info (kitty)
export const TerminalLinkSchema = z.object({
  kittyWindowId: z.number(),
  linkedAt: z.string(), // ISO timestamp
  stale: z.boolean(),
});
export type TerminalLink = z.infer<typeof TerminalLinkSchema>;

// Embedded PTY info
export const EmbeddedPtySchema = z.object({
  ptyId: z.string(),
  wsUrl: z.string(),
  connectedAt: z.string(), // ISO timestamp
  active: z.boolean(),
});
export type EmbeddedPty = z.infer<typeof EmbeddedPtySchema>;

// File-based status (from .claude/status.md)
export const FileStatusValueSchema = z.enum([
  "working",
  "waiting_for_approval",
  "waiting_for_input",
  "completed",
  "error",
  "blocked",
  "idle",
]);
export type FileStatusValue = z.infer<typeof FileStatusValueSchema>;

export const FileStatusSchema = z.object({
  status: FileStatusValueSchema,
  updated: z.string(), // ISO timestamp
  task: z.string().optional(),
  summary: z.string().optional(),
  blockers: z.string().optional(),
  nextSteps: z.string().optional(),
});
export type FileStatus = z.infer<typeof FileStatusSchema>;

// Main session state schema
export const SessionSchema = z.object({
  sessionId: z.string(),
  cwd: z.string(),
  gitBranch: z.string().nullable(),
  gitRepoUrl: z.string().nullable(),
  gitRepoId: z.string().nullable(),
  originalPrompt: z.string(),
  status: SessionStatusSchema,
  lastActivityAt: z.string(), // ISO timestamp
  messageCount: z.number(),
  hasPendingToolUse: z.boolean(),
  pendingTool: PendingToolSchema.nullable(),
  goal: z.string(), // High-level goal of the session
  summary: z.string(), // Current activity summary
  recentOutput: z.array(RecentOutputSchema), // Last few messages for live view
  terminalLink: TerminalLinkSchema.nullable(), // Linked kitty terminal window
  embeddedPty: EmbeddedPtySchema.nullable(), // Embedded PTY terminal
  fileStatus: FileStatusSchema.nullable(), // File-based status from .claude/status.md
});
export type Session = z.infer<typeof SessionSchema>;

// Create the state schema for durable streams
export const sessionsStateSchema = createStateSchema({
  sessions: {
    schema: SessionSchema,
    type: "session",
    primaryKey: "sessionId",
  },
});

// Re-export tool registry for UI consumption
export {
  TOOL_REGISTRY,
  TOOL_NAMES,
  type ToolName,
  type ToolConfig,
  formatToolUse,
  getToolIcon,
} from "./tools/index.js";
