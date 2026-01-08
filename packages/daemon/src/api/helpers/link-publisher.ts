/**
 * Terminal link stream publishing
 */

import type { StreamServer } from "../../server.js";
import type { TerminalLink } from "../../db/terminal-link-repo.js";

/**
 * Publish terminal link update to the stream.
 */
export async function publishLinkUpdate(
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
