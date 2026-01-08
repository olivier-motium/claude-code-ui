/**
 * Centralized configuration for the daemon.
 * All tunable constants and environment variables are defined here.
 */

// =============================================================================
// Stream Server Configuration
// =============================================================================

/** Host for the durable streams server */
export const STREAM_HOST = process.env.STREAM_HOST ?? "127.0.0.1";

/** Port for the durable streams server */
export const STREAM_PORT = parseInt(process.env.PORT ?? "4450", 10);

/** Path for the sessions stream endpoint */
export const STREAM_PATH = "/sessions";

/** Construct the full stream URL */
export function getStreamUrl(host = STREAM_HOST, port = STREAM_PORT): string {
  return `http://${host}:${port}${STREAM_PATH}`;
}

// =============================================================================
// Timeout Constants (milliseconds)
// =============================================================================

/** Time before a session is considered idle (5 minutes) */
export const IDLE_TIMEOUT_MS = 5 * 60 * 1000;

/** Time to wait before detecting pending tool approval (5 seconds) */
export const APPROVAL_TIMEOUT_MS = 5 * 1000;

/** Time before a working session without tool use is considered stale (60 seconds) */
export const STALE_TIMEOUT_MS = 60 * 1000;

/** Threshold for "recent" sessions in CLI (1 hour) */
export const RECENT_THRESHOLD_MS = 60 * 60 * 1000;

// =============================================================================
// GitHub PR Polling Configuration
// =============================================================================

/** Cache TTL for PR info (1 minute) */
export const PR_CACHE_TTL = 60_000;

/** CI polling interval while checks are running (30 seconds) */
export const CI_POLL_INTERVAL_ACTIVE = 30_000;

/** CI polling interval after checks complete (5 minutes) */
export const CI_POLL_INTERVAL_IDLE = 5 * 60_000;

/** Maximum entries in PR cache before pruning */
export const PR_CACHE_MAX_SIZE = 1000;

/** TTL for individual cache entries (30 minutes) */
export const PR_CACHE_ENTRY_TTL = 30 * 60 * 1000;

// =============================================================================
// Session Scoring Weights (for UI sorting)
// =============================================================================

/** Status weights for activity scoring */
export const STATUS_WEIGHTS: Record<string, number> = {
  working: 100,
  waiting: 50,
  idle: 1,
};

/** Bonus score for sessions with pending tool use */
export const PENDING_TOOL_BONUS = 30;

/** Half-life for score decay (minutes) */
export const DECAY_HALF_LIFE_MINUTES = 30;

// =============================================================================
// Session Age Configuration
// =============================================================================

/** Maximum session age to display (24 hours by default) */
export const MAX_AGE_HOURS = parseInt(process.env.MAX_AGE_HOURS ?? "24", 10);

/** Maximum session age in milliseconds */
export const MAX_AGE_MS = MAX_AGE_HOURS * 60 * 60 * 1000;
