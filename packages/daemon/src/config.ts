/**
 * Centralized configuration for the daemon.
 * All tunable constants and environment variables are defined here.
 */

import path from "node:path";
import os from "node:os";

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
// Session Entry Limits
// =============================================================================

/** Maximum entries to keep per session (prevents memory leaks) */
export const MAX_ENTRIES_PER_SESSION = 500;

// =============================================================================
// Session Age Configuration
// =============================================================================

/** Maximum session age to display (24 hours by default) */
export const MAX_AGE_HOURS = parseInt(process.env.MAX_AGE_HOURS ?? "24", 10);

/** Maximum session age in milliseconds */
export const MAX_AGE_MS = MAX_AGE_HOURS * 60 * 60 * 1000;

// =============================================================================
// Summary Cache Configuration
// =============================================================================

/** Maximum entries in summary cache before eviction */
export const SUMMARY_CACHE_MAX_SIZE = 500;

/** TTL for summary cache entries (30 minutes) */
export const SUMMARY_CACHE_TTL_MS = 30 * 60 * 1000;

/** Maximum entries in goal cache before eviction */
export const GOAL_CACHE_MAX_SIZE = 500;

/** TTL for goal cache entries (30 minutes) */
export const GOAL_CACHE_TTL_MS = 30 * 60 * 1000;

// =============================================================================
// External Call Timeouts
// =============================================================================

/** Timeout for external API calls (30 seconds) */
export const EXTERNAL_CALL_TIMEOUT_MS = 30_000;

/** Timeout for gh CLI calls (15 seconds) */
export const GH_CLI_TIMEOUT_MS = 15_000;

// =============================================================================
// Kitty Remote Control Configuration
// =============================================================================

/** Socket path for kitty remote control */
export const KITTY_SOCKET = process.env.KITTY_SOCKET ?? "unix:/tmp/claude-cc-kitty";

/** Environment variable name for kitty password */
export const KITTY_PASSWORD_ENV = "KITTY_RC_PASSWORD";

/** Timeout for kitty commands (5 seconds) */
export const KITTY_COMMAND_TIMEOUT_MS = 5_000;

/** Port for the API server */
export const API_PORT = parseInt(process.env.API_PORT ?? "4451", 10);

/** API endpoint prefix */
export const API_PREFIX = "/api";

// =============================================================================
// Database Configuration
// =============================================================================

/** Path to SQLite database */
export const DB_PATH = process.env.DB_PATH ??
  path.join(os.homedir(), ".claude-code-ui", "data.db");

// =============================================================================
// Content Length Limits
// =============================================================================

/** Standard truncation length for text content */
export const CONTENT_TRUNCATE_LENGTH = 300;

/** Preview length for longer content in output display */
export const CONTENT_PREVIEW_LENGTH = 500;

/** Truncation length for user prompt content */
export const USER_PROMPT_TRUNCATE_LENGTH = 200;

/** Short content length for JSON/tool input display */
export const SHORT_CONTENT_LENGTH = 50;

/** Truncation length for context text blocks */
export const CONTEXT_TEXT_LENGTH = 150;

/** Short truncation for text in context */
export const SHORT_CONTEXT_LENGTH = 100;

/** Short user content length in context */
export const SHORT_USER_CONTENT_LENGTH = 80;

/** Command truncation length for display */
export const COMMAND_TRUNCATE_LENGTH = 60;

/** Goal text truncation length */
export const GOAL_TRUNCATE_LENGTH = 50;

// =============================================================================
// Entry/Message Counts
// =============================================================================

/** Number of messages to look back for extracting output */
export const MESSAGE_LOOKBACK_COUNT = 20;

/** Number of characters to show for session ID display */
export const SESSION_ID_DISPLAY_LENGTH = 8;

/** Number of early entries to examine for context */
export const EARLY_ENTRIES_COUNT = 5;

/** Number of recent entries to examine for context */
export const RECENT_ENTRIES_COUNT = 10;

/** Maximum items to show in recent output */
export const RECENT_OUTPUT_MAX_ITEMS = 8;

// =============================================================================
// AI Generation Limits
// =============================================================================

/** Maximum tokens for summary generation */
export const SUMMARY_MAX_TOKENS = 100;

/** Maximum tokens for goal generation */
export const GOAL_MAX_TOKENS = 30;
