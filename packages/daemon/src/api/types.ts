/**
 * Type definitions for API router
 */

import type { KittyRc } from "../kitty-rc.js";
import type { TerminalLinkRepo } from "../db/terminal-link-repo.js";
import type { StreamServer } from "../server.js";
import type { SessionState } from "../watcher.js";

export interface RouterDependencies {
  kittyRc: KittyRc;
  linkRepo: TerminalLinkRepo;
  streamServer: StreamServer;
  getSession: (id: string) => SessionState | undefined;
  getAllSessions?: () => Map<string, SessionState>;
}
