/**
 * Hono API router for terminal control endpoints.
 *
 * Composes sub-routers for:
 * - /kitty/* - Terminal health and setup
 * - /debug/* - Debug endpoints
 * - /sessions/* - Session terminal control
 */

import { Hono } from "hono";
import { cors } from "hono/cors";
import type { RouterDependencies } from "./types.js";
import { createKittyRoutes } from "./routes/kitty.js";
import { createDebugRoutes } from "./routes/debug.js";
import { createSessionRoutes } from "./routes/sessions.js";

// Re-export types for consumers
export type { RouterDependencies } from "./types.js";

/**
 * Create the API router with all terminal control endpoints.
 */
export function createApiRouter(deps: RouterDependencies): Hono {
  const api = new Hono();

  // Enable CORS for UI (allow any localhost port)
  api.use(
    "*",
    cors({
      origin: (origin) => {
        if (!origin) return origin;
        try {
          const url = new URL(origin);
          if (url.hostname === "localhost" || url.hostname === "127.0.0.1") {
            return origin;  // Return the origin string to allow it
          }
          return null;  // Reject non-localhost origins
        } catch {
          return null;
        }
      },
      credentials: true,
    })
  );

  // Mount sub-routers
  api.route("/", createKittyRoutes(deps));
  api.route("/", createDebugRoutes(deps));
  api.route("/", createSessionRoutes(deps));

  return api;
}
