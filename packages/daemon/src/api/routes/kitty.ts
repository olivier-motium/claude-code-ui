/**
 * Kitty terminal control endpoints
 */

import { Hono } from "hono";
import { getKittyStatus, setupKitty } from "../../kitty-setup.js";
import type { RouterDependencies } from "../types.js";

/**
 * Create kitty-related routes
 */
export function createKittyRoutes(_deps: RouterDependencies): Hono {
  const router = new Hono();

  // Health check with detailed status
  router.get("/kitty/health", async (c) => {
    const details = await getKittyStatus();
    return c.json({
      available: details.socketReachable,
      details,
    });
  });

  // Manual setup trigger
  router.post("/kitty/setup", async (c) => {
    const result = await setupKitty();
    return c.json(result);
  });

  return router;
}
