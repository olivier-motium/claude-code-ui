#!/usr/bin/env node
/**
 * Starts the session watcher and durable streams server.
 * Sessions are published to the stream for the UI to consume.
 */

import dotenv from "dotenv";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { existsSync } from "node:fs";

// Load .env from project root (handles both src and dist execution)
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const envPaths = [
  path.resolve(__dirname, "../../../.env"),  // from src/
  path.resolve(__dirname, "../../.env"),     // from dist/
  path.resolve(process.cwd(), ".env"),       // from cwd
];
for (const envPath of envPaths) {
  if (existsSync(envPath)) {
    dotenv.config({ path: envPath });
    break;
  }
}
import { serve } from "@hono/node-server";
import { Hono } from "hono";
import { SessionWatcher, type SessionEvent, type SessionState } from "./watcher.js";
import { StreamServer } from "./server.js";
import { formatStatus } from "./status.js";
import { STREAM_PORT, STREAM_HOST, API_PORT, API_PREFIX, MAX_AGE_HOURS, MAX_AGE_MS } from "./config.js";
import { colors } from "./utils/colors.js";
import { createApiRouter } from "./api/router.js";
import { KittyRc } from "./kitty-rc.js";
import { TerminalLinkRepo } from "./db/terminal-link-repo.js";
import { closeDb } from "./db/index.js";
import { setupKitty, getKittyStatus } from "./kitty-setup.js";

// Validate required environment variables at startup
const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY;
if (!ANTHROPIC_API_KEY) {
  console.error("Missing required env var: ANTHROPIC_API_KEY");
  console.error("Set ANTHROPIC_API_KEY=sk-ant-... to enable AI summaries");
  process.exit(1);
}

/**
 * Check if a session is recent enough to include
 */
function isRecentSession(session: SessionState): boolean {
  const lastActivity = new Date(session.status.lastActivityAt).getTime();
  return Date.now() - lastActivity < MAX_AGE_MS;
}

async function main(): Promise<void> {
  // Global error handlers to prevent silent crashes
  process.on('unhandledRejection', (reason) => {
    console.error(`${colors.red}[FATAL]${colors.reset} Unhandled Rejection:`, reason);
    // Don't exit - log and continue to keep daemon running
  });

  process.on('uncaughtException', (error) => {
    console.error(`${colors.red}[FATAL]${colors.reset} Uncaught Exception:`, error);
    process.exit(1);  // Exit cleanly on uncaught exceptions
  });

  console.log(`${colors.bold}Claude Code Session Daemon${colors.reset}`);
  console.log(`${colors.dim}Showing sessions from last ${MAX_AGE_HOURS} hours${colors.reset}`);
  console.log();

  // Start the durable streams server
  const streamServer = new StreamServer({ port: STREAM_PORT });
  await streamServer.start();

  console.log(`Stream URL: ${colors.cyan}${streamServer.getStreamUrl()}${colors.reset}`);

  // Initialize kitty remote control and link repository
  const kittyRc = new KittyRc();
  const linkRepo = new TerminalLinkRepo();

  // Auto-setup kitty remote control if needed
  const kittyStatus = await getKittyStatus();

  if (!kittyStatus.installed) {
    console.log(`${colors.dim}[KITTY]${colors.reset} Not installed - terminal control disabled`);
  } else if (!kittyStatus.socketReachable) {
    console.log(`${colors.yellow}[KITTY]${colors.reset} Running automatic setup...`);

    const result = await setupKitty();

    for (const action of result.actions) {
      console.log(`${colors.green}[KITTY]${colors.reset} ${action}`);
    }

    if (result.success) {
      console.log(`${colors.green}[KITTY]${colors.reset} ${result.message}`);
    } else {
      console.log(`${colors.yellow}[KITTY]${colors.reset} ${result.message}`);
    }
  } else {
    console.log(`${colors.green}[KITTY]${colors.reset} Remote control ready`);
  }

  // Start the session watcher
  const watcher = new SessionWatcher({ debounceMs: 300 });

  // Create API server with Hono
  const app = new Hono();
  app.route(
    API_PREFIX,
    createApiRouter({
      kittyRc,
      linkRepo,
      streamServer,
      getSession: (id) => watcher.getSessions().get(id),
      getAllSessions: () => watcher.getSessions(),
    })
  );

  // Start API server on separate port
  const apiServer = serve({
    fetch: app.fetch,
    port: API_PORT,
    hostname: STREAM_HOST,
  });
  console.log(`API server: ${colors.cyan}http://${STREAM_HOST}:${API_PORT}${API_PREFIX}${colors.reset}`);

  // Validate existing terminal links on startup
  const staleSessions = await linkRepo.validateAll(kittyRc);
  if (staleSessions.length > 0) {
    console.log(
      `${colors.yellow}[LINKS]${colors.reset} Marked ${staleSessions.length} terminal links as stale`
    );
  }

  console.log();

  watcher.on("session", async (event: SessionEvent) => {
    const { type, session } = event;

    // Only publish recent sessions
    if (!isRecentSession(session) && type !== "deleted") {
      return;
    }

    const timestamp = new Date().toLocaleTimeString();

    // Log to console - show directory name for easier identification
    const statusStr = formatStatus(session.status);
    const dirName = session.cwd.split("/").pop() || session.cwd;
    console.log(
      `${colors.gray}${timestamp}${colors.reset} ` +
      `${type === "created" ? colors.green : type === "deleted" ? colors.blue : colors.yellow}[${type.toUpperCase().slice(0, 3)}]${colors.reset} ` +
      `${colors.cyan}${session.sessionId.slice(0, 8)}${colors.reset} ` +
      `${colors.dim}${dirName}${colors.reset} ` +
      `${statusStr}`
    );

    // Publish to stream
    try {
      const operation = type === "created" ? "insert" : type === "deleted" ? "delete" : "update";
      await streamServer.publishSession(session, operation);
    } catch (error) {
      console.error(`${colors.yellow}[ERROR]${colors.reset} Failed to publish:`, error);
    }
  });

  watcher.on("error", (error: Error) => {
    console.error(`${colors.yellow}[ERROR]${colors.reset}`, error.message);
  });

  // Handle shutdown with timeout to prevent hangs
  process.on("SIGINT", async () => {
    console.log();
    console.log(`${colors.dim}Shutting down...${colors.reset}`);

    // Set a shutdown timeout to force exit if cleanup hangs
    const shutdownTimeout = setTimeout(() => {
      console.error(`${colors.yellow}[WARN]${colors.reset} Shutdown timed out, forcing exit`);
      process.exit(1);
    }, 5000);

    try {
      watcher.stop();
      apiServer.close();
      closeDb();
      await streamServer.stop();
    } finally {
      clearTimeout(shutdownTimeout);
      process.exit(0);
    }
  });

  // Start watching
  await watcher.start();

  // Publish initial sessions (filtered to recent only)
  const allSessions = watcher.getSessions();
  const recentSessions = Array.from(allSessions.values()).filter(isRecentSession);

  console.log(`${colors.dim}Found ${recentSessions.length} recent sessions (of ${allSessions.size} total), publishing...${colors.reset}`);

  for (const session of recentSessions) {
    try {
      await streamServer.publishSession(session, "insert");
    } catch (error) {
      console.error(`${colors.yellow}[ERROR]${colors.reset} Failed to publish initial session:`, error);
    }
  }

  console.log();
  console.log(`${colors.green}✓${colors.reset} Ready - watching for changes`);
  console.log(`${colors.dim}Press Ctrl+C to exit${colors.reset}`);
}

main().catch((error) => {
  console.error("Fatal error:", error);
  process.exit(1);
});
