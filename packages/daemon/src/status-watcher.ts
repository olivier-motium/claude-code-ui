/**
 * Watches .claude/status.md files across project directories.
 * Parses YAML frontmatter and emits status updates.
 */

import { watch, type FSWatcher } from "chokidar";
import { EventEmitter } from "node:events";
import { readFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";
import { parseStatusFile, isStatusStale, type ParsedStatus } from "./status-parser.js";
import { STATUS_FILE_TTL_MS, STATUS_DIR, STATUS_FILENAME } from "./config.js";
import type { FileStatus } from "./schema.js";

// =============================================================================
// Types
// =============================================================================

export interface StatusUpdateEvent {
  cwd: string;
  status: FileStatus | null; // null if file was deleted or is stale
}

// =============================================================================
// StatusWatcher Class
// =============================================================================

export class StatusWatcher extends EventEmitter {
  private watchers = new Map<string, FSWatcher>();
  private statusCache = new Map<string, FileStatus>();
  private debounceTimers = new Map<string, NodeJS.Timeout>();
  private debounceMs: number;

  constructor(options: { debounceMs?: number } = {}) {
    super();
    this.debounceMs = options.debounceMs ?? 100;
  }

  /**
   * Start watching a project directory for status file changes.
   * Call this when a new session is discovered.
   */
  watchProject(cwd: string): void {
    // Already watching this directory
    if (this.watchers.has(cwd)) {
      return;
    }

    const statusFilePath = path.join(cwd, STATUS_DIR, STATUS_FILENAME);
    const statusDir = path.join(cwd, STATUS_DIR);

    // Watch the .claude directory (not just the file) to catch file creation
    const watcher = watch(statusDir, {
      persistent: true,
      ignoreInitial: false,
      depth: 0,
    });

    watcher
      .on("add", (filepath) => {
        if (path.basename(filepath) === STATUS_FILENAME) {
          this.handleStatusFile(cwd, filepath);
        }
      })
      .on("change", (filepath) => {
        if (path.basename(filepath) === STATUS_FILENAME) {
          this.debouncedHandleFile(cwd, filepath);
        }
      })
      .on("unlink", (filepath) => {
        if (path.basename(filepath) === STATUS_FILENAME) {
          this.handleDelete(cwd);
        }
      })
      .on("error", (error) => this.emit("error", error));

    this.watchers.set(cwd, watcher);

    // Check if status file already exists
    if (existsSync(statusFilePath)) {
      this.handleStatusFile(cwd, statusFilePath);
    }
  }

  /**
   * Stop watching a project directory.
   */
  unwatchProject(cwd: string): void {
    const watcher = this.watchers.get(cwd);
    if (watcher) {
      watcher.close();
      this.watchers.delete(cwd);
    }

    // Clear debounce timer
    const timer = this.debounceTimers.get(cwd);
    if (timer) {
      clearTimeout(timer);
      this.debounceTimers.delete(cwd);
    }

    // Remove from cache
    this.statusCache.delete(cwd);
  }

  /**
   * Stop all watchers.
   */
  stop(): void {
    for (const cwd of this.watchers.keys()) {
      this.unwatchProject(cwd);
    }
  }

  /**
   * Get current status for a project.
   * Returns null if no status file or if stale.
   */
  getStatus(cwd: string): FileStatus | null {
    const cached = this.statusCache.get(cwd);
    if (!cached) {
      return null;
    }

    // Check if stale
    if (isStatusStale(cached.updated, STATUS_FILE_TTL_MS)) {
      this.statusCache.delete(cwd);
      return null;
    }

    return cached;
  }

  /**
   * Refresh status for a project from disk.
   * Useful when you need the latest status synchronously.
   */
  async refreshStatus(cwd: string): Promise<FileStatus | null> {
    const statusFilePath = path.join(cwd, STATUS_DIR, STATUS_FILENAME);

    if (!existsSync(statusFilePath)) {
      this.statusCache.delete(cwd);
      return null;
    }

    try {
      const content = await readFile(statusFilePath, "utf-8");
      const parsed = parseStatusFile(content);

      if (!parsed) {
        return null;
      }

      // Check staleness
      if (isStatusStale(parsed.frontmatter.updated, STATUS_FILE_TTL_MS)) {
        return null;
      }

      const fileStatus = this.parsedToFileStatus(parsed);
      this.statusCache.set(cwd, fileStatus);
      return fileStatus;
    } catch {
      return null;
    }
  }

  // ===========================================================================
  // Private Methods
  // ===========================================================================

  private debouncedHandleFile(cwd: string, filepath: string): void {
    const existing = this.debounceTimers.get(cwd);
    if (existing) {
      clearTimeout(existing);
    }

    const timer = setTimeout(() => {
      this.debounceTimers.delete(cwd);
      this.handleStatusFile(cwd, filepath);
    }, this.debounceMs);

    this.debounceTimers.set(cwd, timer);
  }

  private async handleStatusFile(cwd: string, filepath: string): Promise<void> {
    try {
      const content = await readFile(filepath, "utf-8");
      const parsed = parseStatusFile(content);

      if (!parsed) {
        // Invalid file format - emit null status
        this.statusCache.delete(cwd);
        this.emit("status", { cwd, status: null } satisfies StatusUpdateEvent);
        return;
      }

      // Check staleness
      if (isStatusStale(parsed.frontmatter.updated, STATUS_FILE_TTL_MS)) {
        this.statusCache.delete(cwd);
        this.emit("status", { cwd, status: null } satisfies StatusUpdateEvent);
        return;
      }

      const fileStatus = this.parsedToFileStatus(parsed);
      this.statusCache.set(cwd, fileStatus);
      this.emit("status", { cwd, status: fileStatus } satisfies StatusUpdateEvent);
    } catch (error) {
      this.emit("error", error);
    }
  }

  private handleDelete(cwd: string): void {
    this.statusCache.delete(cwd);
    this.emit("status", { cwd, status: null } satisfies StatusUpdateEvent);
  }

  private parsedToFileStatus(parsed: ParsedStatus): FileStatus {
    return {
      status: parsed.frontmatter.status,
      updated: parsed.frontmatter.updated,
      task: parsed.frontmatter.task,
      summary: parsed.summary,
      blockers: parsed.blockers,
      nextSteps: parsed.nextSteps,
    };
  }
}
