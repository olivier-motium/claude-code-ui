/**
 * Cache utilities for summarization.
 */

import type { LogEntry } from "../types.js";

// Cache entry with timestamp for TTL-based eviction
export interface SummaryCacheEntry {
  summary: string;
  hash: string;
  timestamp: number;
}

export interface GoalCacheEntry {
  goal: string;
  entryCount: number;
  timestamp: number;
}

/**
 * Evict stale entries from a cache based on TTL and max size.
 * Uses LRU-style eviction when size limit is exceeded.
 */
export function evictStaleEntries<K, V extends { timestamp: number }>(
  cache: Map<K, V>,
  ttlMs: number,
  maxSize: number
): void {
  const now = Date.now();

  // Remove expired entries
  for (const [key, entry] of cache) {
    if (now - entry.timestamp > ttlMs) {
      cache.delete(key);
    }
  }

  // Enforce max size - remove oldest entries
  if (cache.size > maxSize) {
    const entries = Array.from(cache.entries())
      .sort((a, b) => a[1].timestamp - b[1].timestamp);
    const toRemove = cache.size - maxSize;
    for (let i = 0; i < toRemove; i++) {
      cache.delete(entries[i][0]);
    }
  }
}

/**
 * Generate a content hash for cache invalidation
 */
export function generateContentHash(entries: LogEntry[]): string {
  // Use last few entries to determine if content changed significantly
  const recent = entries.slice(-5);
  return recent.map((e) => {
    if ("timestamp" in e) {
      return `${e.type}:${e.timestamp}`;
    }
    return e.type;
  }).join("|");
}
