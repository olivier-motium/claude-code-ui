/**
 * Utility functions for SessionCard components
 */

import type { Session } from "../../types/schema";
import { getEffectiveStatus } from "../../lib/sessionStatus";

export function getCardClass(session: Session): string {
  const classes = ["session-card"];
  const { status, fileStatusValue } = getEffectiveStatus(session);

  // Base status classes
  if (status === "working") {
    classes.push("status-working");
  }
  if (status === "waiting" && session.hasPendingToolUse) {
    classes.push("status-needs-approval");
  }

  // File-status-specific classes for visual indicators
  if (fileStatusValue === "completed") {
    classes.push("file-status-completed");
  }
  if (fileStatusValue === "error") {
    classes.push("file-status-error");
  }
  if (fileStatusValue === "blocked") {
    classes.push("file-status-blocked");
  }

  return classes.join(" ");
}

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

export function formatTarget(target: string): string {
  // Shorten file paths
  if (target.includes("/")) {
    const parts = target.split("/");
    return parts[parts.length - 1];
  }
  // Truncate long commands
  if (target.length > 30) {
    return target.slice(0, 27) + "…";
  }
  return target;
}

export function getRoleColor(role: "user" | "assistant" | "tool"): string {
  switch (role) {
    case "user":
      return "var(--blue-11)";
    case "assistant":
      return "var(--gray-12)";
    case "tool":
      return "var(--violet-11)";
  }
}

export function getRolePrefix(role: "user" | "assistant" | "tool"): string {
  switch (role) {
    case "user":
      return "You: ";
    case "assistant":
      return "";
    case "tool":
      return "";
  }
}
