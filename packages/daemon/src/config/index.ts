/**
 * Centralized configuration for the daemon.
 * All tunable constants and environment variables are defined here.
 *
 * This module re-exports from domain-specific config files:
 * - stream.ts: Stream server configuration
 * - timeouts.ts: Timeout constants
 * - scoring.ts: Session scoring weights
 * - ai.ts: AI generation and cache configuration
 * - content.ts: Content length limits
 * - pty.ts: PTY (Embedded Terminal) configuration
 * - paths.ts: File paths and API configuration
 */

// Helpers (also exported for use by other modules)
export { parsePositiveInt } from "./helpers.js";

// Stream server
export {
  STREAM_HOST,
  STREAM_PORT,
  STREAM_PATH,
  getStreamUrl,
} from "./stream.js";

// Timeouts
export {
  IDLE_TIMEOUT_MS,
  APPROVAL_TIMEOUT_MS,
  STALE_TIMEOUT_MS,
  RECENT_THRESHOLD_MS,
} from "./timeouts.js";

// Scoring
export {
  STATUS_WEIGHTS,
  PENDING_TOOL_BONUS,
  DECAY_HALF_LIFE_MINUTES,
} from "./scoring.js";

// AI
export {
  MAX_ENTRIES_PER_SESSION,
  MAX_AGE_HOURS,
  MAX_AGE_MS,
  SUMMARY_CACHE_MAX_SIZE,
  SUMMARY_CACHE_TTL_MS,
  GOAL_CACHE_MAX_SIZE,
  GOAL_CACHE_TTL_MS,
  EXTERNAL_CALL_TIMEOUT_MS,
  GH_CLI_TIMEOUT_MS,
  SUMMARY_MAX_TOKENS,
  GOAL_MAX_TOKENS,
} from "./ai.js";

// Content length limits
export {
  CONTENT_TRUNCATE_LENGTH,
  CONTENT_PREVIEW_LENGTH,
  USER_PROMPT_TRUNCATE_LENGTH,
  SHORT_CONTENT_LENGTH,
  CONTEXT_TEXT_LENGTH,
  SHORT_CONTEXT_LENGTH,
  SHORT_USER_CONTENT_LENGTH,
  COMMAND_TRUNCATE_LENGTH,
  GOAL_TRUNCATE_LENGTH,
  MESSAGE_LOOKBACK_COUNT,
  SESSION_ID_DISPLAY_LENGTH,
  EARLY_ENTRIES_COUNT,
  RECENT_ENTRIES_COUNT,
  RECENT_OUTPUT_MAX_ITEMS,
} from "./content.js";

// PTY
export {
  PTY_WS_HOST,
  PTY_WS_PORT,
  PTY_IDLE_TIMEOUT_MS,
  PTY_DEFAULT_COLS,
  PTY_DEFAULT_ROWS,
  PTY_IDLE_CHECK_INTERVAL_MS,
  getPtyWsUrl,
} from "./pty.js";

// Paths
export {
  DB_PATH,
  STATUS_FILE_TTL_MS,
  STATUS_FILENAME,
  STATUS_DIR,
  KITTY_SOCKET,
  KITTY_PASSWORD_ENV,
  KITTY_COMMAND_TIMEOUT_MS,
  API_PORT,
  API_PREFIX,
} from "./paths.js";
