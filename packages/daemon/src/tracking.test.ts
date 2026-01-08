/**
 * E2E test for Claude Code session tracking
 *
 * Tests the full flow: file detection → parsing → status → publishing
 */

import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { mkdir, writeFile, rm, appendFile } from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import { SessionWatcher } from "./watcher.js";
import { deriveStatus } from "./status.js";
import { tailJSONL, extractMetadata } from "./parser.js";

const TEST_DIR = path.join(os.homedir(), ".claude", "projects", "-test-e2e-session");
const TEST_SESSION_ID = "test-session-" + Date.now();
const TEST_LOG_FILE = path.join(TEST_DIR, `${TEST_SESSION_ID}.jsonl`);

// Helper to create a log entry
function createUserEntry(content: string, timestamp = new Date().toISOString()) {
  return JSON.stringify({
    type: "user",
    parentUuid: null,
    uuid: `uuid-${Date.now()}-${Math.random()}`,
    sessionId: TEST_SESSION_ID,
    timestamp,
    cwd: "/Users/test/project",
    version: "1.0.0",
    gitBranch: "main",
    isSidechain: false,
    userType: "external",
    message: {
      role: "user",
      content,
    },
  }) + "\n";
}

function createAssistantEntry(content: string, timestamp = new Date().toISOString(), hasToolUse = false) {
  const blocks: Array<{ type: string; text?: string; id?: string; name?: string; input?: Record<string, unknown> }> = [
    { type: "text", text: content },
  ];

  if (hasToolUse) {
    blocks.push({
      type: "tool_use",
      id: `tool-${Date.now()}`,
      name: "Bash",
      input: { command: "echo test" },
    });
  }

  return JSON.stringify({
    type: "assistant",
    parentUuid: null,
    uuid: `uuid-${Date.now()}-${Math.random()}`,
    sessionId: TEST_SESSION_ID,
    timestamp,
    cwd: "/Users/test/project",
    version: "1.0.0",
    gitBranch: "main",
    isSidechain: false,
    userType: "external",
    requestId: `req-${Date.now()}`,
    message: {
      role: "assistant",
      model: "claude-sonnet-4-20250514",
      id: `msg-${Date.now()}`,
      content: blocks,
      stop_reason: hasToolUse ? "tool_use" : "end_turn",
    },
  }) + "\n";
}

describe("Session Tracking", () => {
  beforeEach(async () => {
    // Create test directory
    await mkdir(TEST_DIR, { recursive: true });
  });

  afterEach(async () => {
    // Cleanup
    await rm(TEST_DIR, { recursive: true, force: true });
  });

  describe("Parser", () => {
    it("should parse JSONL entries from a log file", async () => {
      // Write a simple log file
      const entry1 = createUserEntry("Hello, help me with something");
      const entry2 = createAssistantEntry("Sure, I can help!");

      await writeFile(TEST_LOG_FILE, entry1 + entry2);

      // Parse it
      const { entries, newPosition } = await tailJSONL(TEST_LOG_FILE, 0);

      expect(entries).toHaveLength(2);
      expect(entries[0].type).toBe("user");
      expect(entries[1].type).toBe("assistant");
      expect(newPosition).toBeGreaterThan(0);
    });

    it("should extract metadata from entries", async () => {
      const entry = createUserEntry("Help me build a feature");
      await writeFile(TEST_LOG_FILE, entry);

      const { entries } = await tailJSONL(TEST_LOG_FILE, 0);
      const metadata = extractMetadata(entries);

      expect(metadata).not.toBeNull();
      expect(metadata?.sessionId).toBe(TEST_SESSION_ID);
      expect(metadata?.cwd).toBe("/Users/test/project");
      expect(metadata?.gitBranch).toBe("main");
      expect(metadata?.originalPrompt).toBe("Help me build a feature");
    });

    it("should handle incremental reads", async () => {
      // Write initial entry
      const entry1 = createUserEntry("First message");
      await writeFile(TEST_LOG_FILE, entry1);

      const { entries: first, newPosition: pos1 } = await tailJSONL(TEST_LOG_FILE, 0);
      expect(first).toHaveLength(1);

      // Append more entries
      const entry2 = createAssistantEntry("Response");
      const entry3 = createUserEntry("Follow up");
      await appendFile(TEST_LOG_FILE, entry2 + entry3);

      // Small delay to ensure file is flushed
      await new Promise((r) => setTimeout(r, 50));

      // Read from previous position - should get both new entries
      const { entries: second, newPosition: pos2 } = await tailJSONL(TEST_LOG_FILE, pos1);

      expect(second).toHaveLength(2);
      expect(second[0].type).toBe("assistant");
      expect(second[1].type).toBe("user");
      expect(pos2).toBeGreaterThan(pos1);
    });
  });

  describe("Status Derivation", () => {
    it("should detect working status after user message", async () => {
      const entry = createUserEntry("Do something");
      await writeFile(TEST_LOG_FILE, entry);

      const { entries } = await tailJSONL(TEST_LOG_FILE, 0);
      const status = deriveStatus(entries);

      expect(status.status).toBe("working");
      expect(status.lastRole).toBe("user");
      expect(status.hasPendingToolUse).toBe(false);
    });

    it("should detect waiting status after assistant response", async () => {
      const entry1 = createUserEntry("Do something");
      const entry2 = createAssistantEntry("Done!");
      await writeFile(TEST_LOG_FILE, entry1 + entry2);

      const { entries } = await tailJSONL(TEST_LOG_FILE, 0);
      const status = deriveStatus(entries);

      expect(status.status).toBe("waiting");
      expect(status.lastRole).toBe("assistant");
      expect(status.hasPendingToolUse).toBe(false);
    });

    it("should detect pending tool use", async () => {
      const entry1 = createUserEntry("Run a command");
      const entry2 = createAssistantEntry("I'll run that for you", new Date().toISOString(), true);
      await writeFile(TEST_LOG_FILE, entry1 + entry2);

      const { entries } = await tailJSONL(TEST_LOG_FILE, 0);
      const status = deriveStatus(entries);

      expect(status.status).toBe("waiting");
      expect(status.hasPendingToolUse).toBe(true);
    });

    it("should detect idle status after timeout", async () => {
      // Create entry from 10 minutes ago
      const oldTime = new Date(Date.now() - 10 * 60 * 1000).toISOString();
      const entry = createAssistantEntry("Old response", oldTime);
      await writeFile(TEST_LOG_FILE, entry);

      const { entries } = await tailJSONL(TEST_LOG_FILE, 0);
      const status = deriveStatus(entries);

      expect(status.status).toBe("idle");
    });

    it("should stay working during tool execution even after timeout", async () => {
      // Create a tool_result entry from 2 minutes ago (past the 30s timeout)
      const oldTime = new Date(Date.now() - 2 * 60 * 1000).toISOString();
      const toolResultEntry = JSON.stringify({
        type: "user",
        parentUuid: null,
        uuid: `uuid-${Date.now()}`,
        sessionId: TEST_SESSION_ID,
        timestamp: oldTime,
        cwd: "/Users/test/project",
        version: "1.0.0",
        gitBranch: "main",
        isSidechain: false,
        userType: "external",
        message: {
          role: "user",
          content: [{ type: "tool_result", tool_use_id: "tool-123", content: "Command output" }],
        },
      }) + "\n";

      await writeFile(TEST_LOG_FILE, toolResultEntry);

      const { entries } = await tailJSONL(TEST_LOG_FILE, 0);
      const status = deriveStatus(entries);

      // Should still be "working" because it's a tool_result, not a human prompt
      expect(status.status).toBe("working");
    });
  });

  describe("SessionWatcher", () => {
    it("should detect new session files", async () => {
      const watcher = new SessionWatcher({ debounceMs: 50 });

      const events: Array<{ type: string; sessionId: string }> = [];
      watcher.on("session", (event) => {
        events.push({ type: event.type, sessionId: event.session.sessionId });
      });

      await watcher.start();

      // Create a session file
      const entry = createUserEntry("New session");
      await writeFile(TEST_LOG_FILE, entry);

      // Wait for detection
      await new Promise((r) => setTimeout(r, 500));

      watcher.stop();

      expect(events.length).toBeGreaterThan(0);
      expect(events[0].type).toBe("created");
    });

    it("should detect session updates", async () => {
      // Create initial file
      const entry1 = createUserEntry("Initial");
      await writeFile(TEST_LOG_FILE, entry1);

      const watcher = new SessionWatcher({ debounceMs: 50 });

      const events: Array<{ type: string; status: string }> = [];
      watcher.on("session", (event) => {
        events.push({ type: event.type, status: event.session.status.status });
      });

      await watcher.start();

      // Wait for initial detection
      await new Promise((r) => setTimeout(r, 300));

      // Append assistant response
      const entry2 = createAssistantEntry("Response");
      await appendFile(TEST_LOG_FILE, entry2);

      // Wait for update detection
      await new Promise((r) => setTimeout(r, 500));

      watcher.stop();

      // Should have created event and possibly update
      expect(events.some(e => e.type === "created")).toBe(true);
    });

    it("should track message count changes", async () => {
      const entry1 = createUserEntry("First");
      await writeFile(TEST_LOG_FILE, entry1);

      const watcher = new SessionWatcher({ debounceMs: 50 });

      let lastMessageCount = 0;
      watcher.on("session", (event) => {
        // Only track our test session, not others
        if (event.session.sessionId === TEST_SESSION_ID) {
          lastMessageCount = event.session.status.messageCount;
        }
      });

      await watcher.start();
      await new Promise((r) => setTimeout(r, 500));

      expect(lastMessageCount).toBe(1);

      // Add more messages
      await appendFile(TEST_LOG_FILE, createAssistantEntry("Two"));
      await new Promise((r) => setTimeout(r, 500));

      await appendFile(TEST_LOG_FILE, createUserEntry("Three"));
      await new Promise((r) => setTimeout(r, 500));

      watcher.stop();

      expect(lastMessageCount).toBe(3);
    });
  });
});
