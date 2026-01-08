/**
 * AI-powered session summarization using Claude Sonnet
 */

import Anthropic from "@anthropic-ai/sdk";
import type { MessageCreateParamsNonStreaming } from "@anthropic-ai/sdk/resources/messages";
import fastq from "fastq";
import type { queueAsPromised } from "fastq";
import type { SessionState } from "../watcher.js";
import {
  SUMMARY_CACHE_MAX_SIZE,
  SUMMARY_CACHE_TTL_MS,
  GOAL_CACHE_MAX_SIZE,
  GOAL_CACHE_TTL_MS,
  EXTERNAL_CALL_TIMEOUT_MS,
  CONTENT_TRUNCATE_LENGTH,
  GOAL_TRUNCATE_LENGTH,
  EARLY_ENTRIES_COUNT,
  SUMMARY_MAX_TOKENS,
  GOAL_MAX_TOKENS,
} from "../config.js";
import { withTimeout, TimeoutError } from "../utils/timeout.js";
import { isError } from "../utils/type-guards.js";
import {
  evictStaleEntries,
  generateContentHash,
  type SummaryCacheEntry,
  type GoalCacheEntry,
} from "./cache.js";
import { extractContext, extractEarlyContext, extractRecentGoalContext } from "./context-extraction.js";
import { getWorkingSummary, getFallbackSummary } from "./summaries.js";
import { cleanGoalText } from "./text-utils.js";

// Lazy-load client to ensure env vars are loaded first
let client: Anthropic | null = null;
function getClient(): Anthropic {
  if (!client) {
    client = new Anthropic();
  }
  return client;
}

// Queue for Anthropic API calls to avoid rate limit errors
interface APITask {
  params: MessageCreateParamsNonStreaming;
  resolve: (result: string) => void;
  reject: (error: Error) => void;
}

async function processAPITask(task: APITask): Promise<void> {
  try {
    const response = await getClient().messages.create(task.params);
    const text = response.content[0].type === "text"
      ? response.content[0].text.trim()
      : "";
    task.resolve(text);
  } catch (error) {
    task.reject(isError(error) ? error : new Error(String(error)));
  }
}

// Concurrency of 1 to be safe with rate limits
const apiQueue: queueAsPromised<APITask> = fastq.promise(processAPITask, 1);

/**
 * Queue an API call and return the result
 */
function queueAPICall(params: MessageCreateParamsNonStreaming): Promise<string> {
  return new Promise((resolve, reject) => {
    apiQueue.push({ params, resolve, reject });
  });
}

// Cache summaries to avoid redundant API calls
const summaryCache = new Map<string, SummaryCacheEntry>();

// Cache goals with entry count - regenerate if session has grown significantly
const goalCache = new Map<string, GoalCacheEntry>();

/**
 * Generate an AI summary of the session's current state
 */
export async function generateAISummary(session: SessionState): Promise<string> {
  const { sessionId, entries, status } = session;

  // Quick heuristic summaries for simple cases
  if (entries.length < 3) {
    return "Just started";
  }

  if (status.status === "working") {
    return getWorkingSummary(session);
  }

  // Evict stale entries before checking cache
  evictStaleEntries(summaryCache, SUMMARY_CACHE_TTL_MS, SUMMARY_CACHE_MAX_SIZE);

  // Check cache
  const contentHash = generateContentHash(entries);
  const cached = summaryCache.get(sessionId);
  if (cached && cached.hash === contentHash) {
    // Update timestamp on access (LRU behavior)
    cached.timestamp = Date.now();
    return cached.summary;
  }

  // Generate AI summary for idle/waiting sessions
  try {
    const context = extractContext(session);

    const summary = await withTimeout(
      queueAPICall({
        model: "claude-sonnet-4-20250514",
        max_tokens: SUMMARY_MAX_TOKENS,
        messages: [
          {
            role: "user",
            content: `Summarize this Claude Code session's current state in 5-10 words. Be specific about what was accomplished or what's being worked on. Don't use generic phrases like "working on code" - mention specific files, features, or tasks.

${context}

Summary:`,
          },
        ],
      }),
      EXTERNAL_CALL_TIMEOUT_MS,
      "AI summary generation timed out"
    );

    const result = summary || "Session active";

    // Cache the result with timestamp
    summaryCache.set(sessionId, {
      summary: result,
      hash: contentHash,
      timestamp: Date.now(),
    });

    return result;
  } catch (error) {
    if (error instanceof TimeoutError) {
      console.warn(`[summarizer] Summary timed out for ${sessionId}`);
    } else {
      console.error("Failed to generate AI summary:", error);
    }
    return getFallbackSummary(session);
  }
}

/**
 * Generate the high-level goal of the session
 * Cached but regenerated if session grows significantly
 */
export async function generateGoal(session: SessionState): Promise<string> {
  const { sessionId, originalPrompt, entries } = session;

  // Evict stale entries
  evictStaleEntries(goalCache, GOAL_CACHE_TTL_MS, GOAL_CACHE_MAX_SIZE);

  // Check cache - but regenerate if session has grown 5x since last generation
  const cached = goalCache.get(sessionId);
  if (cached && entries.length < cached.entryCount * 5) {
    // Update timestamp on access (LRU behavior)
    cached.timestamp = Date.now();
    return cached.goal;
  }

  // For new sessions, use the original prompt
  if (entries.length < EARLY_ENTRIES_COUNT) {
    return cleanGoalText(originalPrompt);
  }

  // Generate AI goal
  try {
    // Build context from early and recent entries
    const context = [
      `Original task: ${originalPrompt.slice(0, CONTENT_TRUNCATE_LENGTH)}`,
      "\nEarly activity:",
      ...extractEarlyContext(entries),
      "\nRecent activity:",
      ...extractRecentGoalContext(entries),
    ];

    const goalResponse = await withTimeout(
      queueAPICall({
        model: "claude-sonnet-4-20250514",
        max_tokens: GOAL_MAX_TOKENS,
        messages: [
          {
            role: "user",
            content: `What is the HIGH-LEVEL GOAL of this coding session based on what's actually being built/done? Focus on the ACTUAL WORK. Respond with ONLY a short phrase (5-10 words max). No punctuation. No quotes.

Examples:
- Build UI for monitoring sessions
- Fix authentication bug in login
- Add dark mode support

${context.join("\n")}

Goal:`,
          },
        ],
      }),
      EXTERNAL_CALL_TIMEOUT_MS,
      "Goal generation timed out"
    );

    const goal = cleanGoalText(goalResponse || originalPrompt.slice(0, GOAL_TRUNCATE_LENGTH));

    // Cache with current entry count and timestamp
    goalCache.set(sessionId, {
      goal,
      entryCount: entries.length,
      timestamp: Date.now(),
    });

    return goal;
  } catch (error) {
    if (error instanceof TimeoutError) {
      console.warn(`[summarizer] Goal generation timed out for ${sessionId}`);
    } else {
      console.error("Failed to generate goal:", error);
    }
    return cleanGoalText(originalPrompt);
  }
}

/**
 * Clear the summary cache for a session
 */
export function clearSummaryCache(sessionId: string): void {
  summaryCache.delete(sessionId);
}
