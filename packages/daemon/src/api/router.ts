/**
 * Hono API router for terminal control endpoints.
 */

import { Hono } from "hono";
import { cors } from "hono/cors";
import type { KittyRc } from "../kitty-rc.js";
import {
  TerminalLinkRepo,
  type TerminalLink,
} from "../db/terminal-link-repo.js";
import { getDb, schema } from "../db/index.js";
import type { StreamServer } from "../server.js";
import type { SessionState } from "../watcher.js";
import { getKittyStatus, setupKitty } from "../kitty-setup.js";

interface RouterDependencies {
  kittyRc: KittyRc;
  linkRepo: TerminalLinkRepo;
  streamServer: StreamServer;
  getSession: (id: string) => SessionState | undefined;
}

/**
 * Create the API router with all terminal control endpoints.
 */
export function createApiRouter(deps: RouterDependencies) {
  const { kittyRc, linkRepo, streamServer, getSession } = deps;
  const api = new Hono();

  // Enable CORS for UI
  api.use(
    "*",
    cors({
      origin: ["http://localhost:5173", "http://127.0.0.1:5173"],
      credentials: true,
    })
  );

  // Health check with detailed status
  api.get("/kitty/health", async (c) => {
    const details = await getKittyStatus();
    return c.json({
      available: details.socketReachable,
      details,
    });
  });

  // Manual setup trigger
  api.post("/kitty/setup", async (c) => {
    const result = await setupKitty();
    return c.json(result);
  });

  // Focus existing linked terminal
  api.post("/sessions/:id/focus", async (c) => {
    const sessionId = c.req.param("id");
    const link = linkRepo.get(sessionId);

    if (!link) {
      return c.json({ error: "No terminal linked" }, 404);
    }

    const success = await kittyRc.focusWindow(link.kittyWindowId);

    if (!success) {
      linkRepo.markStale(sessionId);
      await publishLinkUpdate(streamServer, sessionId, linkRepo.get(sessionId));
      return c.json({ error: "Terminal window not found", stale: true }, 410);
    }

    return c.json({ success: true });
  });

  // Focus or create terminal
  api.post("/sessions/:id/open", async (c) => {
    const sessionId = c.req.param("id");
    const session = getSession(sessionId);

    if (!session) {
      return c.json({ error: "Session not found" }, 404);
    }

    // Try existing link first
    const existingLink = linkRepo.get(sessionId);
    if (existingLink && !existingLink.stale) {
      const success = await kittyRc.focusWindow(existingLink.kittyWindowId);
      if (success) {
        return c.json({ success: true, windowId: existingLink.kittyWindowId });
      }
      // Mark stale and continue to create new
      linkRepo.markStale(sessionId);
    }

    // Create new tab
    const dirName = session.cwd.split("/").pop() || session.cwd;
    const windowId = await kittyRc.launchTab({
      cwd: session.cwd,
      tabTitle: `${dirName} • ${sessionId.slice(0, 8)}`,
      vars: { cc_session_id: sessionId },
    });

    const link: TerminalLink = {
      sessionId,
      kittyWindowId: windowId,
      linkedAt: new Date().toISOString(),
      stale: false,
      repoPath: session.cwd,
      createdVia: "auto_open",
    };
    linkRepo.upsert(link);
    await publishLinkUpdate(streamServer, sessionId, link);

    // Focus the new window
    await kittyRc.focusWindow(windowId);

    return c.json({ success: true, windowId, created: true });
  });

  // Link existing terminal via interactive select
  api.post("/sessions/:id/link-terminal", async (c) => {
    const sessionId = c.req.param("id");
    const session = getSession(sessionId);

    if (!session) {
      return c.json({ error: "Session not found" }, 404);
    }

    const windowId = await kittyRc.selectWindow();

    if (windowId === null) {
      return c.json({ error: "Selection cancelled" }, 400);
    }

    const link: TerminalLink = {
      sessionId,
      kittyWindowId: windowId,
      linkedAt: new Date().toISOString(),
      stale: false,
      repoPath: session.cwd,
      createdVia: "manual_link",
    };
    linkRepo.upsert(link);
    await publishLinkUpdate(streamServer, sessionId, link);

    return c.json({ success: true, windowId });
  });

  // Unlink terminal
  api.delete("/sessions/:id/link-terminal", async (c) => {
    const sessionId = c.req.param("id");
    linkRepo.delete(sessionId);
    await publishLinkUpdate(streamServer, sessionId, null);
    return c.json({ success: true });
  });

  // Send text to terminal
  api.post("/sessions/:id/send-text", async (c) => {
    const sessionId = c.req.param("id");
    const body = await c.req.json<{ text: string; submit: boolean }>();

    const link = linkRepo.get(sessionId);
    if (!link) {
      return c.json({ error: "No terminal linked" }, 404);
    }
    if (link.stale) {
      return c.json({ error: "Terminal link is stale" }, 410);
    }

    // Verify window exists before sending
    const osWindows = await kittyRc.ls();
    if (!kittyRc.windowExists(osWindows, link.kittyWindowId)) {
      linkRepo.markStale(sessionId);
      await publishLinkUpdate(streamServer, sessionId, linkRepo.get(sessionId));
      return c.json({ error: "Terminal window not found", stale: true }, 410);
    }

    await kittyRc.sendText(link.kittyWindowId, body.text, body.submit);

    // Log to command history (fire and forget)
    logCommand(sessionId, link.kittyWindowId, body.text, body.submit);

    return c.json({ success: true });
  });

  return api;
}

/**
 * Publish terminal link update to the stream.
 */
async function publishLinkUpdate(
  server: StreamServer,
  sessionId: string,
  link: TerminalLink | null | undefined
): Promise<void> {
  // Convert to schema format and publish
  const terminalLink = link
    ? {
        kittyWindowId: link.kittyWindowId,
        linkedAt: link.linkedAt,
        stale: link.stale,
      }
    : null;

  await server.publishTerminalLinkUpdate(sessionId, terminalLink);
}

/**
 * Log command to history table (fire and forget).
 */
function logCommand(
  sessionId: string,
  windowId: number,
  text: string,
  submitted: boolean
): void {
  try {
    const db = getDb();
    db.insert(schema.commandHistory)
      .values({
        sessionId,
        kittyWindowId: windowId,
        command: text,
        sentAt: new Date().toISOString(),
        submitted,
      })
      .run();
  } catch (err) {
    // Fire and forget - don't fail the request
    console.error("[API] Failed to log command:", err);
  }
}
