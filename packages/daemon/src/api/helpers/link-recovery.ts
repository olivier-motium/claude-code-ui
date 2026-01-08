/**
 * Terminal link recovery logic
 */

import type { KittyRc } from "../../kitty-rc.js";
import type { TerminalLinkRepo, TerminalLink } from "../../db/terminal-link-repo.js";

/**
 * Try to recover a terminal link when the stored window ID is invalid.
 * Searches by user_vars, then cmdline. Returns the valid window ID or null.
 */
export async function tryRecoverLink(
  sessionId: string,
  link: TerminalLink,
  kittyRc: KittyRc,
  linkRepo: TerminalLinkRepo
): Promise<{ windowId: number; recovered: boolean } | null> {
  // First check if existing ID still works (fast path)
  const osWindows = await kittyRc.ls();
  if (kittyRc.windowExists(osWindows, link.kittyWindowId)) {
    return { windowId: link.kittyWindowId, recovered: false };
  }

  // Try to find by user_vars or cmdline
  const found = await kittyRc.findWindowByAny(sessionId);
  if (found) {
    linkRepo.updateWindowId(sessionId, found.windowId);
    console.log(
      `[API] Recovered link for ${sessionId.slice(0, 8)} via ${found.method}: ` +
      `${link.kittyWindowId} → ${found.windowId}`
    );
    return { windowId: found.windowId, recovered: true };
  }

  return null; // Window not found, needs new tab
}
