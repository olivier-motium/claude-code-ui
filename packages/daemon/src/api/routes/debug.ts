/**
 * Debug endpoints for development
 */

import { Hono } from "hono";
import type { RouterDependencies } from "../types.js";

/**
 * Create debug routes
 */
export function createDebugRoutes(deps: RouterDependencies): Hono {
  const router = new Hono();

  // List all sessions known to the watcher
  router.get("/debug/sessions", (c) => {
    if (!deps.getAllSessions) {
      return c.json({
        message: "getAllSessions not available",
        hint: "Check daemon console logs for '[API] Session X not found'",
      });
    }

    const allSessions = deps.getAllSessions();
    const sessions = Array.from(allSessions.values()).map((s) => ({
      id: s.sessionId,
      status: s.status.status,
      cwd: s.cwd,
      lastActivityAt: s.status.lastActivityAt,
    }));

    return c.json({
      total: sessions.length,
      sessions: sessions.slice(0, 50), // Limit to 50 for readability
    });
  });

  return router;
}
