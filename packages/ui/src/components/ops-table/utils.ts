/**
 * Utility functions for OpsTable components
 */

import type { Session, SessionStatus } from "../../types/schema";
import type { StatusFilter, StatusCounts } from "./types";
import { getEffectiveStatus } from "../../lib/sessionStatus";
import { STALE_THRESHOLD_MS } from "./constants";

/**
 * Format a timestamp as relative time (e.g., "2m", "1h", "3d")
 */
export function formatTimeAgo(isoString: string): string {
  const now = Date.now();
  const then = new Date(isoString).getTime();
  const diff = now - then;

  const seconds = Math.floor(diff / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);

  if (days > 0) return `${days}d`;
  if (hours > 0) return `${hours}h`;
  if (minutes > 0) return `${minutes}m`;
  return `${seconds}s`;
}

/**
 * Format a tool target (file path or command) for display
 */
export function formatTarget(target: string): string {
  // Shorten file paths to just filename
  if (target.includes("/")) {
    const parts = target.split("/");
    return parts[parts.length - 1];
  }
  // Truncate long commands
  if (target.length > 30) {
    return target.slice(0, 27) + "...";
  }
  return target;
}

/**
 * Get the display text for a goal or prompt
 */
export function formatGoal(session: Session, maxLength = 60): string {
  const text = session.goal || session.originalPrompt;
  if (text.length > maxLength) {
    return text.slice(0, maxLength - 3) + "...";
  }
  return text;
}

/**
 * Check if a session is stale (working but no activity for STALE_THRESHOLD_MS)
 */
export function isSessionStale(session: Session): boolean {
  const { status } = getEffectiveStatus(session);
  if (status !== "working") return false;

  const lastActivity = new Date(session.lastActivityAt).getTime();
  return Date.now() - lastActivity > STALE_THRESHOLD_MS;
}

/**
 * Get row class based on status
 */
export function getRowClass(session: Session, isSelected: boolean): string {
  const classes = ["ops-table-row"];
  const { status, fileStatusValue } = getEffectiveStatus(session);

  if (isSelected) {
    classes.push("selected");
  }

  if (status === "working") {
    classes.push("status-working");
  }
  if (status === "waiting" && session.hasPendingToolUse) {
    classes.push("status-needs-approval");
  }
  if (isSessionStale(session)) {
    classes.push("status-stale");
  }
  if (fileStatusValue === "error") {
    classes.push("status-error");
  }
  if (fileStatusValue === "blocked") {
    classes.push("status-blocked");
  }

  return classes.join(" ");
}

/**
 * Filter sessions by status filter
 */
export function filterSessions(sessions: Session[], filter: StatusFilter): Session[] {
  if (filter === "all") return sessions;

  return sessions.filter((session) => {
    const { status, fileStatusValue } = getEffectiveStatus(session);

    switch (filter) {
      case "working":
        return status === "working";
      case "waiting":
        return status === "waiting";
      case "idle":
        return status === "idle";
      case "stale":
        return isSessionStale(session);
      case "error":
        return fileStatusValue === "error";
      case "blocked":
        return fileStatusValue === "blocked";
      default:
        return true;
    }
  });
}

/**
 * Count sessions by status
 */
export function countSessionsByStatus(sessions: Session[]): StatusCounts {
  const counts: StatusCounts = {
    all: sessions.length,
    working: 0,
    waiting: 0,
    idle: 0,
    stale: 0,
    error: 0,
  };

  for (const session of sessions) {
    const { status, fileStatusValue } = getEffectiveStatus(session);

    if (status === "working") counts.working++;
    else if (status === "waiting") counts.waiting++;
    else counts.idle++;

    if (isSessionStale(session)) counts.stale++;
    if (fileStatusValue === "error") counts.error++;
  }

  return counts;
}

/**
 * Sort sessions by activity (most recent first) and status priority
 */
export function sortSessions(sessions: Session[]): Session[] {
  return [...sessions].sort((a, b) => {
    const { status: statusA } = getEffectiveStatus(a);
    const { status: statusB } = getEffectiveStatus(b);

    // Priority: working > waiting > idle
    const priorityOrder: Record<SessionStatus, number> = {
      working: 0,
      waiting: 1,
      idle: 2,
    };

    const priorityDiff = priorityOrder[statusA] - priorityOrder[statusB];
    if (priorityDiff !== 0) return priorityDiff;

    // Within same status, sort by most recent activity
    const timeA = new Date(a.lastActivityAt).getTime();
    const timeB = new Date(b.lastActivityAt).getTime();
    return timeB - timeA;
  });
}
