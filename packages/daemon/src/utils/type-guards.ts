/**
 * Type guards for safer type narrowing.
 *
 * These replace type assertions (as X) with runtime checks,
 * providing better type safety and error handling.
 */

import type { LogEntry, UserEntry, AssistantEntry, SystemEntry } from "../types.js";

/**
 * Check if a log entry is a user entry
 */
export function isUserEntry(entry: LogEntry): entry is UserEntry {
  return entry.type === "user";
}

/**
 * Check if a log entry is an assistant entry
 */
export function isAssistantEntry(entry: LogEntry): entry is AssistantEntry {
  return entry.type === "assistant";
}

/**
 * Check if a log entry is a system entry
 */
export function isSystemEntry(entry: LogEntry): entry is SystemEntry {
  return entry.type === "system";
}

/**
 * Check if a value is an Error instance
 */
export function isError(value: unknown): value is Error {
  return value instanceof Error;
}

/**
 * Check if a value is a plain object (Record<string, unknown>)
 */
export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Get error message from an unknown caught value
 */
export function getErrorMessage(error: unknown): string {
  if (isError(error)) {
    return error.message;
  }
  if (typeof error === "string") {
    return error;
  }
  return String(error);
}
