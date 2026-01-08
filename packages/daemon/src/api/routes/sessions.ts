/**
 * Session terminal control endpoints
 */

import { Hono } from "hono";
import type { TerminalLink } from "../../db/terminal-link-repo.js";
import type { RouterDependencies } from "../types.js";
import { tryRecoverLink } from "../helpers/link-recovery.js";
import { publishLinkUpdate } from "../helpers/link-publisher.js";
import { logCommand } from "../helpers/command-logger.js";
import { getErrorMessage } from "../../utils/type-guards.js";

/**
 * Create session-related routes
 */
export function createSessionRoutes(deps: RouterDependencies): Hono {
  const { kittyRc, linkRepo, streamServer, getSession } = deps;
  const router = new Hono();

  // Focus existing linked terminal (with recovery)
  router.post("/sessions/:id/focus", async (c) => {
    const sessionId = c.req.param("id");
    const link = linkRepo.get(sessionId);

    if (!link) {
      return c.json({ error: "No terminal linked" }, 404);
    }

    // Try recovery if window ID changed
    const result = await tryRecoverLink(sessionId, link, kittyRc, linkRepo);

    if (!result) {
      linkRepo.markStale(sessionId);
      await publishLinkUpdate(streamServer, sessionId, linkRepo.get(sessionId));
      return c.json({ error: "Terminal window not found", stale: true }, 410);
    }

    // If recovered, publish the updated link
    if (result.recovered) {
      await publishLinkUpdate(streamServer, sessionId, linkRepo.get(sessionId));
    }

    const success = await kittyRc.focusWindow(result.windowId);

    if (!success) {
      linkRepo.markStale(sessionId);
      await publishLinkUpdate(streamServer, sessionId, linkRepo.get(sessionId));
      return c.json({ error: "Focus failed" }, 500);
    }

    return c.json({ success: true, recovered: result.recovered });
  });

  // Focus or create terminal (with recovery)
  router.post("/sessions/:id/open", async (c) => {
    const sessionId = c.req.param("id");
    const allSessions = deps.getAllSessions?.();
    console.log(`[API] /open called for session ${sessionId.slice(0, 8)}, watcher has ${allSessions?.size ?? 0} sessions`);

    const session = getSession(sessionId);

    if (!session) {
      // Log available sessions to help diagnose why this session wasn't found
      const available = [...(allSessions?.keys() ?? [])].slice(0, 10).map(k => k.slice(0, 8)).join(', ');
      console.log(`[API] Session ${sessionId.slice(0, 8)} NOT FOUND. Available (first 10): ${available || 'none'}`);
      return c.json({ error: "Session not found" }, 404);
    }

    console.log(`[API] Session found: status=${session.status.status}, cwd=${session.cwd}`);

    // Try existing link with recovery
    const existingLink = linkRepo.get(sessionId);
    if (existingLink) {
      console.log(`[API] Trying existing link: windowId=${existingLink.kittyWindowId}, stale=${existingLink.stale}`);
      const result = await tryRecoverLink(sessionId, existingLink, kittyRc, linkRepo);

      if (result) {
        // Found or recovered - focus it
        if (result.recovered) {
          await publishLinkUpdate(streamServer, sessionId, linkRepo.get(sessionId));
        }
        const success = await kittyRc.focusWindow(result.windowId);
        if (success) {
          return c.json({
            success: true,
            windowId: result.windowId,
            recovered: result.recovered,
          });
        }
      }

      // Recovery failed or focus failed - will create new tab
      console.log(`[API] Recovery failed or window not found, will create new tab`);
    }

    // Create new tab with claude --resume to continue the session
    try {
      const dirName = session.cwd.split("/").pop() || session.cwd;
      console.log(`[API] Launching new tab for ${dirName}`);
      const windowId = await kittyRc.launchTab({
        cwd: session.cwd,
        tabTitle: `${dirName} • ${sessionId.slice(0, 8)}`,
        vars: { cc_session_id: sessionId },
        command: ["claude", "--resume", sessionId, "--dangerously-skip-permissions"],
      });

      console.log(`[API] Tab launched: windowId=${windowId}`);

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
    } catch (error) {
      console.error(`[API] Failed to launch tab:`, getErrorMessage(error));
      return c.json({ error: getErrorMessage(error) }, 500);
    }
  });

  // Link existing terminal via interactive select
  router.post("/sessions/:id/link-terminal", async (c) => {
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
  router.delete("/sessions/:id/link-terminal", async (c) => {
    const sessionId = c.req.param("id");
    linkRepo.delete(sessionId);
    await publishLinkUpdate(streamServer, sessionId, null);
    return c.json({ success: true });
  });

  // Send text to terminal (with recovery)
  router.post("/sessions/:id/send-text", async (c) => {
    const sessionId = c.req.param("id");
    const body = await c.req.json<{ text: string; submit: boolean }>();

    const link = linkRepo.get(sessionId);
    if (!link) {
      return c.json({ error: "No terminal linked" }, 404);
    }

    // Try recovery (handles stale links too)
    const result = await tryRecoverLink(sessionId, link, kittyRc, linkRepo);

    if (!result) {
      linkRepo.markStale(sessionId);
      await publishLinkUpdate(streamServer, sessionId, linkRepo.get(sessionId));
      return c.json({ error: "Terminal window not found", stale: true }, 410);
    }

    // If recovered, publish the updated link
    if (result.recovered) {
      await publishLinkUpdate(streamServer, sessionId, linkRepo.get(sessionId));
    }

    await kittyRc.sendText(result.windowId, body.text, body.submit);

    // Log to command history (fire and forget)
    logCommand(sessionId, result.windowId, body.text, body.submit);

    return c.json({ success: true, recovered: result.recovered });
  });

  return router;
}
