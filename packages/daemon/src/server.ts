#!/usr/bin/env node
/**
 * Durable Streams server for session state.
 */

import { DurableStreamTestServer } from "@durable-streams/server";
import { DurableStream } from "@durable-streams/client";
import {
  sessionsStateSchema,
  type Session,
  type RecentOutput,
  type PRInfo,
  type TerminalLink,
} from "./schema.js";
import type { SessionState } from "./watcher.js";
import type { LogEntry } from "./types.js";
import { generateAISummary, generateGoal } from "./summarizer/index.js";
import { queuePRCheck, getCachedPR, setOnPRUpdate, stopAllPolling } from "./github.js";
import {
  STREAM_HOST,
  STREAM_PORT,
  STREAM_PATH,
  getStreamUrl,
  MESSAGE_LOOKBACK_COUNT,
  RECENT_OUTPUT_MAX_ITEMS,
  CONTENT_PREVIEW_LENGTH,
  CONTENT_TRUNCATE_LENGTH,
  COMMAND_TRUNCATE_LENGTH,
  SHORT_CONTENT_LENGTH,
} from "./config.js";
import { TerminalLinkRepo } from "./db/terminal-link-repo.js";
import path from "node:path";
import os from "node:os";

export interface StreamServerOptions {
  port?: number;
  dataDir?: string;
}

export class StreamServer {
  private server: DurableStreamTestServer;
  private stream: DurableStream | null = null;
  private port: number;
  private streamUrl: string;
  // Track sessions for PR update callbacks
  private sessionCache = new Map<string, SessionState>();
  // Terminal link repository for lookups
  private linkRepo: TerminalLinkRepo;
  // Flag to prevent race conditions during shutdown
  private stopping = false;

  constructor(options: StreamServerOptions = {}) {
    this.linkRepo = new TerminalLinkRepo();
    this.port = options.port ?? STREAM_PORT;
    const dataDir = options.dataDir ?? path.join(os.homedir(), ".claude-code-ui", "streams");

    this.server = new DurableStreamTestServer({
      port: this.port,
      host: STREAM_HOST,
      dataDir,
    });

    this.streamUrl = getStreamUrl(STREAM_HOST, this.port);
  }

  async start(): Promise<void> {
    await this.server.start();
    console.log(`Durable Streams server running on http://${STREAM_HOST}:${this.port}`);

    // Create or connect to the sessions stream
    try {
      this.stream = await DurableStream.create({
        url: this.streamUrl,
        contentType: "application/json",
      });
    } catch (error: unknown) {
      // Stream might already exist, try to connect
      if ((error as { code?: string }).code === "CONFLICT_EXISTS") {
        this.stream = await DurableStream.connect({ url: this.streamUrl });
      } else {
        throw error;
      }
    }

    // Set up PR update callback
    setOnPRUpdate(async (sessionId, pr) => {
      console.log(`[PR] Received PR update for session ${sessionId.slice(0, 8)}: ${pr ? `PR #${pr.number}` : "no PR"}`);
      const sessionState = this.sessionCache.get(sessionId);
      if (sessionState) {
        await this.publishSessionWithPR(sessionState, pr);
      } else {
        console.log(`[PR] No cached session state for ${sessionId.slice(0, 8)}`);
      }
    });
  }

  async stop(): Promise<void> {
    this.stopping = true;  // Prevent new publishes during shutdown
    stopAllPolling();
    await this.server.stop();
    this.stream = null;
  }

  getStreamUrl(): string {
    return this.streamUrl;
  }

  /**
   * Convert SessionState to Session schema and publish to stream
   */
  async publishSession(sessionState: SessionState, operation: "insert" | "update" | "delete"): Promise<void> {
    // Skip publishing during shutdown or if stream is not ready
    if (this.stopping || !this.stream) {
      return;
    }

    // Cache session state for PR update callbacks
    this.sessionCache.set(sessionState.sessionId, sessionState);

    // Generate AI goal and summary (goals are cached, summaries update more frequently)
    const [goal, summary] = await Promise.all([
      generateGoal(sessionState),
      generateAISummary(sessionState),
    ]);

    // Get cached PR info if available
    const pr = sessionState.gitBranch
      ? getCachedPR(sessionState.cwd, sessionState.gitBranch)
      : null;

    // Queue PR check if we have a branch (will update via callback)
    if (sessionState.gitBranch) {
      console.log(`[PR] Session ${sessionState.sessionId.slice(0, 8)} has branch: ${sessionState.gitBranch}`);
      queuePRCheck(sessionState.cwd, sessionState.gitBranch, sessionState.sessionId);
    } else {
      console.log(`[PR] Session ${sessionState.sessionId.slice(0, 8)} has no branch`);
    }

    // Get terminal link if it exists
    const link = this.linkRepo.get(sessionState.sessionId);
    const terminalLink = link
      ? {
          kittyWindowId: link.kittyWindowId,
          linkedAt: link.linkedAt,
          stale: link.stale,
        }
      : null;

    const session: Session = {
      sessionId: sessionState.sessionId,
      cwd: sessionState.cwd,
      gitBranch: sessionState.gitBranch,
      gitRepoUrl: sessionState.gitRepoUrl,
      gitRepoId: sessionState.gitRepoId,
      originalPrompt: sessionState.originalPrompt,
      status: sessionState.status.status,
      lastActivityAt: sessionState.status.lastActivityAt,
      messageCount: sessionState.status.messageCount,
      hasPendingToolUse: sessionState.status.hasPendingToolUse,
      pendingTool: extractPendingTool(sessionState),
      goal,
      summary,
      recentOutput: extractRecentOutput(sessionState.entries),
      pr,
      terminalLink,
    };

    // Create the event using the schema helpers
    let event;
    if (operation === "insert") {
      event = sessionsStateSchema.sessions.insert({ value: session });
    } else if (operation === "update") {
      event = sessionsStateSchema.sessions.update({ value: session });
    } else {
      event = sessionsStateSchema.sessions.delete({
        key: session.sessionId,
        oldValue: session,
      });
    }

    await this.stream.append(event);
  }

  /**
   * Publish session with updated PR info (called from PR update callback)
   */
  async publishSessionWithPR(sessionState: SessionState, pr: PRInfo | null): Promise<void> {
    // Skip publishing during shutdown or if stream is not ready
    if (this.stopping || !this.stream) {
      return;
    }

    // Generate AI goal and summary
    const [goal, summary] = await Promise.all([
      generateGoal(sessionState),
      generateAISummary(sessionState),
    ]);

    // Get terminal link if it exists
    const link = this.linkRepo.get(sessionState.sessionId);
    const terminalLink = link
      ? {
          kittyWindowId: link.kittyWindowId,
          linkedAt: link.linkedAt,
          stale: link.stale,
        }
      : null;

    const session: Session = {
      sessionId: sessionState.sessionId,
      cwd: sessionState.cwd,
      gitBranch: sessionState.gitBranch,
      gitRepoUrl: sessionState.gitRepoUrl,
      gitRepoId: sessionState.gitRepoId,
      originalPrompt: sessionState.originalPrompt,
      status: sessionState.status.status,
      lastActivityAt: sessionState.status.lastActivityAt,
      messageCount: sessionState.status.messageCount,
      hasPendingToolUse: sessionState.status.hasPendingToolUse,
      pendingTool: extractPendingTool(sessionState),
      goal,
      summary,
      recentOutput: extractRecentOutput(sessionState.entries),
      pr,
      terminalLink,
    };

    const event = sessionsStateSchema.sessions.update({ value: session });
    await this.stream.append(event);
  }

  /**
   * Publish terminal link update for a session
   */
  async publishTerminalLinkUpdate(
    sessionId: string,
    terminalLink: TerminalLink | null
  ): Promise<void> {
    // Skip publishing during shutdown or if stream is not ready
    if (this.stopping || !this.stream) {
      return;
    }

    const sessionState = this.sessionCache.get(sessionId);
    if (!sessionState) {
      console.log(`[LINK] No cached session state for ${sessionId.slice(0, 8)}`);
      return;
    }

    // Generate AI goal and summary
    const [goal, summary] = await Promise.all([
      generateGoal(sessionState),
      generateAISummary(sessionState),
    ]);

    // Get cached PR info if available
    const pr = sessionState.gitBranch
      ? getCachedPR(sessionState.cwd, sessionState.gitBranch)
      : null;

    const session: Session = {
      sessionId: sessionState.sessionId,
      cwd: sessionState.cwd,
      gitBranch: sessionState.gitBranch,
      gitRepoUrl: sessionState.gitRepoUrl,
      gitRepoId: sessionState.gitRepoId,
      originalPrompt: sessionState.originalPrompt,
      status: sessionState.status.status,
      lastActivityAt: sessionState.status.lastActivityAt,
      messageCount: sessionState.status.messageCount,
      hasPendingToolUse: sessionState.status.hasPendingToolUse,
      pendingTool: extractPendingTool(sessionState),
      goal,
      summary,
      recentOutput: extractRecentOutput(sessionState.entries),
      pr,
      terminalLink,
    };

    const event = sessionsStateSchema.sessions.update({ value: session });
    await this.stream.append(event);
  }
}

/**
 * Extract recent output from entries for live view
 * Returns the last few meaningful messages in chronological order
 */
function extractRecentOutput(entries: LogEntry[], maxItems = RECENT_OUTPUT_MAX_ITEMS): RecentOutput[] {
  const output: RecentOutput[] = [];

  // Get the last N entries that are messages (user or assistant)
  const messageEntries = entries
    .filter((e) => e.type === "user" || e.type === "assistant")
    .slice(-MESSAGE_LOOKBACK_COUNT); // Look at last messages to find good content

  for (const entry of messageEntries) {
    if (entry.type === "assistant") {
      // Get first text block if any
      const textBlock = entry.message.content.find(
        (b): b is { type: "text"; text: string } => b.type === "text" && b.text.trim() !== ""
      );
      if (textBlock) {
        output.push({
          role: "assistant",
          content: textBlock.text.slice(0, CONTENT_PREVIEW_LENGTH),
        });
      }

      // Get tool uses
      const toolUses = entry.message.content.filter(
        (b): b is { type: "tool_use"; id: string; name: string; input: Record<string, unknown> } => b.type === "tool_use"
      );
      for (const tool of toolUses.slice(0, 2)) {
        output.push({
          role: "tool",
          content: formatToolUse(tool.name, tool.input),
        });
      }
    } else if (entry.type === "user") {
      // User prompts (string content, not tool results)
      if (typeof entry.message.content === "string" && entry.message.content.trim()) {
        output.push({
          role: "user",
          content: entry.message.content.slice(0, CONTENT_TRUNCATE_LENGTH),
        });
      }
    }
  }

  // Return only the last maxItems
  return output.slice(-maxItems);
}

/**
 * Format tool use for display
 */
function formatToolUse(tool: string, input: Record<string, unknown>): string {
  switch (tool) {
    case "Read":
      return `📖 Reading ${shortenPath(input.file_path as string)}`;
    case "Edit":
      return `✏️ Editing ${shortenPath(input.file_path as string)}`;
    case "Write":
      return `📝 Writing ${shortenPath(input.file_path as string)}`;
    case "Bash":
      return `▶️ Running: ${(input.command as string)?.slice(0, COMMAND_TRUNCATE_LENGTH)}`;
    case "Grep":
      return `🔍 Searching for "${input.pattern}"`;
    case "Glob":
      return `📁 Finding files: ${input.pattern}`;
    case "Task":
      return `🤖 Spawning agent: ${(input.description as string) || "task"}`;
    default:
      return `🔧 ${tool}`;
  }
}

/**
 * Shorten file path for display
 */
function shortenPath(filepath: string | undefined): string {
  if (!filepath) return "file";
  const parts = filepath.split("/");
  return parts.length > 2 ? `.../${parts.slice(-2).join("/")}` : filepath;
}

/**
 * Extract pending tool info from session state
 */
function extractPendingTool(session: SessionState): Session["pendingTool"] {
  if (!session.status.hasPendingToolUse) {
    return null;
  }

  // Find the last assistant message with tool_use
  const entries = session.entries;
  for (let i = entries.length - 1; i >= 0; i--) {
    const entry = entries[i];
    if (entry.type === "assistant") {
      for (const block of entry.message.content) {
        if (block.type === "tool_use") {
          const tool = block.name;
          // Extract target based on tool type
          let target = "";
          const input = block.input as Record<string, unknown>;

          if (tool === "Edit" || tool === "Read" || tool === "Write") {
            target = (input.file_path as string) ?? "";
          } else if (tool === "Bash") {
            target = (input.command as string) ?? "";
          } else if (tool === "Grep" || tool === "Glob") {
            target = (input.pattern as string) ?? "";
          } else {
            target = JSON.stringify(input).slice(0, SHORT_CONTENT_LENGTH);
          }

          return { tool, target };
        }
      }
    }
  }

  return null;
}

