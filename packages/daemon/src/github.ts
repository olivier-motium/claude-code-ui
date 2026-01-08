/**
 * GitHub PR tracking and CI status polling
 */

import { execFile } from "node:child_process";
import { promisify } from "node:util";
import fastq from "fastq";
import type { queueAsPromised } from "fastq";
import type { PRInfo, CIStatus } from "./schema.js";
import {
  PR_CACHE_TTL,
  CI_POLL_INTERVAL_ACTIVE,
  CI_POLL_INTERVAL_IDLE,
  PR_CACHE_MAX_SIZE,
  PR_CACHE_ENTRY_TTL,
  GH_CLI_TIMEOUT_MS,
} from "./config.js";
import { withTimeout, TimeoutError } from "./utils/timeout.js";
import { getErrorMessage } from "./utils/type-guards.js";

const defaultExecFileAsync = promisify(execFile);

// Allow injection of execFile function for testing
type ExecFileFn = (
  cmd: string,
  args: string[],
  opts: { cwd: string }
) => Promise<{ stdout: string; stderr: string }>;
let execFileAsync: ExecFileFn = defaultExecFileAsync;

// Types for queue tasks
interface PRCheckTask {
  type: "check_pr";
  cwd: string;
  branch: string;
  sessionId: string;
}

interface CICheckTask {
  type: "check_ci";
  cwd: string;
  prNumber: number;
  sessionId: string;
}

type QueueTask = PRCheckTask | CICheckTask;

// Callbacks for when PR/CI info is updated
type PRUpdateCallback = (sessionId: string, pr: PRInfo | null) => void;

// Cache PR info to avoid redundant API calls
const prCache = new Map<string, { pr: PRInfo | null; lastChecked: number }>();

// Track which sessions need CI polling
const activeCIPolling = new Map<string, NodeJS.Timeout>();

let onPRUpdate: PRUpdateCallback | null = null;

/**
 * Prune stale entries from the PR cache to prevent memory leaks.
 */
function prunePRCache(): void {
  const now = Date.now();

  // Remove entries older than TTL
  for (const [key, entry] of prCache) {
    if (now - entry.lastChecked > PR_CACHE_ENTRY_TTL) {
      prCache.delete(key);
    }
  }

  // Hard limit - remove oldest entries if over max size
  if (prCache.size > PR_CACHE_MAX_SIZE) {
    const entries = Array.from(prCache.entries())
      .sort((a, b) => a[1].lastChecked - b[1].lastChecked);
    const toRemove = entries.length - PR_CACHE_MAX_SIZE;
    for (let i = 0; i < toRemove; i++) {
      prCache.delete(entries[i][0]);
    }
  }
}

/**
 * Set the callback for PR updates
 */
export function setOnPRUpdate(callback: PRUpdateCallback): void {
  onPRUpdate = callback;
}

/**
 * Process queue tasks
 */
async function processTask(task: QueueTask): Promise<void> {
  if (task.type === "check_pr") {
    await checkPRForBranch(task.cwd, task.branch, task.sessionId);
  } else if (task.type === "check_ci") {
    await checkCIStatus(task.cwd, task.prNumber, task.sessionId);
  }
}

// Create the queue with concurrency of 2
const queue: queueAsPromised<QueueTask> = fastq.promise(processTask, 2);

/**
 * Check if a branch has an associated PR
 */
async function checkPRForBranch(cwd: string, branch: string, sessionId: string): Promise<void> {
  // Prune stale cache entries on each check
  prunePRCache();

  const cacheKey = `${cwd}:${branch}`;

  // Check cache
  const cached = prCache.get(cacheKey);
  if (cached && Date.now() - cached.lastChecked < PR_CACHE_TTL) {
    if (cached.pr && onPRUpdate) {
      onPRUpdate(sessionId, cached.pr);
    }
    return;
  }

  try {
    // Use gh CLI to find PR for this branch (with timeout)
    // Using execFile with array args to prevent command injection
    const { stdout } = await withTimeout(
      execFileAsync(
        "gh",
        ["pr", "list", "--head", branch, "--json", "number,url,title,headRefName", "--limit", "1"],
        { cwd }
      ),
      GH_CLI_TIMEOUT_MS,
      `gh pr list timed out for branch ${branch}`
    );

    const prs = JSON.parse(stdout);
    if (prs.length === 0) {
      console.log(`[PR] No PR found for branch: ${branch}`);
      prCache.set(cacheKey, { pr: null, lastChecked: Date.now() });
      if (onPRUpdate) {
        onPRUpdate(sessionId, null);
      }
      return;
    }

    const pr = prs[0];
    console.log(`[PR] Found PR #${pr.number} for branch: ${branch}`);

    // Get CI status for this PR
    const ciInfo = await getCIStatus(cwd, pr.number);

    const prInfo: PRInfo = {
      number: pr.number,
      url: pr.url,
      title: pr.title,
      ciStatus: ciInfo.overallStatus,
      ciChecks: ciInfo.checks,
      lastChecked: new Date().toISOString(),
    };

    prCache.set(cacheKey, { pr: prInfo, lastChecked: Date.now() });

    if (onPRUpdate) {
      onPRUpdate(sessionId, prInfo);
    }

    // Start CI polling if CI is not complete
    if (ciInfo.overallStatus === "pending" || ciInfo.overallStatus === "running") {
      startCIPolling(cwd, pr.number, sessionId);
    }
  } catch (error) {
    if (error instanceof TimeoutError) {
      console.warn(`[PR] ${error.message}`);
    } else {
      // gh CLI not available or not in a git repo
      console.error(`Failed to check PR for ${branch}:`, getErrorMessage(error));
    }
    prCache.set(cacheKey, { pr: null, lastChecked: Date.now() });
  }
}

/**
 * Get CI status for a PR
 */
async function getCIStatus(cwd: string, prNumber: number): Promise<{
  overallStatus: CIStatus;
  checks: PRInfo["ciChecks"];
}> {
  try {
    // Using execFile with array args to prevent command injection
    const { stdout } = await withTimeout(
      execFileAsync(
        "gh",
        ["pr", "checks", String(prNumber), "--json", "name,state,link"],
        { cwd }
      ),
      GH_CLI_TIMEOUT_MS,
      `gh pr checks timed out for PR #${prNumber}`
    );

    const checks = JSON.parse(stdout);

    const mappedChecks: PRInfo["ciChecks"] = checks.map((check: { name: string; state: string; link?: string }) => ({
      name: check.name,
      status: mapGHState(check.state),
      url: check.link || null,
    }));

    // Determine overall status
    let overallStatus: CIStatus = "success";
    for (const check of mappedChecks) {
      if (check.status === "failure" || check.status === "cancelled") {
        overallStatus = "failure";
        break;
      }
      if (check.status === "running") {
        overallStatus = "running";
      } else if (check.status === "pending" && overallStatus !== "running") {
        overallStatus = "pending";
      }
    }

    if (mappedChecks.length === 0) {
      overallStatus = "unknown";
    }

    console.log(`[PR] CI status for PR #${prNumber}: ${overallStatus} (${mappedChecks.length} checks)`);
    return { overallStatus, checks: mappedChecks };
  } catch (error) {
    if (error instanceof TimeoutError) {
      console.warn(`[PR] ${error.message}`);
    } else {
      console.error(`Failed to get CI status for PR #${prNumber}:`, getErrorMessage(error));
    }
    return { overallStatus: "unknown", checks: [] };
  }
}

/**
 * GitHub state to CIStatus mapping
 */
const GH_STATE_MAP: Record<string, CIStatus> = {
  SUCCESS: "success",
  COMPLETED: "success",
  NEUTRAL: "success",
  SKIPPED: "success",
  FAILURE: "failure",
  ERROR: "failure",
  TIMED_OUT: "failure",
  ACTION_REQUIRED: "failure",
  CANCELLED: "cancelled",
  IN_PROGRESS: "running",
  QUEUED: "running",
  REQUESTED: "running",
  WAITING: "running",
  PENDING: "pending",
};

/**
 * Map GitHub state to our CIStatus
 */
function mapGHState(state: string): CIStatus {
  return GH_STATE_MAP[state.toUpperCase()] ?? "unknown";
}

/**
 * Check CI status for a PR and update
 */
async function checkCIStatus(cwd: string, prNumber: number, sessionId: string): Promise<void> {
  try {
    const ciInfo = await getCIStatus(cwd, prNumber);

    // Get existing PR info from cache to update
    const cacheKey = Array.from(prCache.entries()).find(([_, v]) => v.pr?.number === prNumber)?.[0];
    if (!cacheKey) return;

    const cached = prCache.get(cacheKey);
    if (!cached?.pr) return;

    const updatedPR: PRInfo = {
      ...cached.pr,
      ciStatus: ciInfo.overallStatus,
      ciChecks: ciInfo.checks,
      lastChecked: new Date().toISOString(),
    };

    prCache.set(cacheKey, { pr: updatedPR, lastChecked: Date.now() });

    if (onPRUpdate) {
      onPRUpdate(sessionId, updatedPR);
    }

    // Adjust polling interval based on status
    if (ciInfo.overallStatus === "success" || ciInfo.overallStatus === "failure" || ciInfo.overallStatus === "cancelled") {
      // CI complete - switch to idle polling
      stopCIPolling(sessionId);
      startIdleCIPolling(cwd, prNumber, sessionId);
    }
  } catch (error) {
    console.error(`Failed to check CI for PR #${prNumber}:`, getErrorMessage(error));
  }
}

/**
 * Start active CI polling for a PR
 */
function startCIPolling(cwd: string, prNumber: number, sessionId: string): void {
  // Don't start if already polling
  if (activeCIPolling.has(sessionId)) return;

  const interval = setInterval(() => {
    queue.push({ type: "check_ci", cwd, prNumber, sessionId });
  }, CI_POLL_INTERVAL_ACTIVE);

  activeCIPolling.set(sessionId, interval);
}

/**
 * Start idle CI polling (less frequent, for detecting new CI runs)
 */
function startIdleCIPolling(cwd: string, prNumber: number, sessionId: string): void {
  stopCIPolling(sessionId);

  const interval = setInterval(() => {
    queue.push({ type: "check_ci", cwd, prNumber, sessionId });
  }, CI_POLL_INTERVAL_IDLE);

  activeCIPolling.set(sessionId, interval);
}

/**
 * Stop CI polling for a session
 */
function stopCIPolling(sessionId: string): void {
  const interval = activeCIPolling.get(sessionId);
  if (interval) {
    clearInterval(interval);
    activeCIPolling.delete(sessionId);
  }
}

/**
 * Queue a PR check for a session
 */
export function queuePRCheck(cwd: string, branch: string, sessionId: string): void {
  if (!branch) return;
  console.log(`[PR] Queueing PR check for branch: ${branch}`);
  queue.push({ type: "check_pr", cwd, branch, sessionId });
}

/**
 * Stop all polling (cleanup)
 */
export function stopAllPolling(): void {
  for (const [sessionId] of activeCIPolling) {
    stopCIPolling(sessionId);
  }
}

/**
 * Get cached PR info for a session (for initial publish)
 */
export function getCachedPR(cwd: string, branch: string): PRInfo | null {
  const cacheKey = `${cwd}:${branch}`;
  return prCache.get(cacheKey)?.pr ?? null;
}

// Test helpers
export const __test__ = {
  setExecFileAsync(fn: ExecFileFn) {
    execFileAsync = fn;
  },
  resetExecFileAsync() {
    execFileAsync = defaultExecFileAsync;
  },
  clearCache() {
    prCache.clear();
  },
  getQueue() {
    return queue;
  },
};
