/**
 * Kitty terminal remote control wrapper.
 * Uses kitten @ commands via subprocess.
 */

import { execFile } from "node:child_process";
import { promisify } from "node:util";
import {
  KITTY_SOCKET,
  KITTY_PASSWORD_ENV,
  KITTY_COMMAND_TIMEOUT_MS,
} from "./config.js";
import { withTimeout } from "./utils/timeout.js";

const execFileAsync = promisify(execFile);

export interface KittyWindow {
  id: number;
  title: string;
  cwd: string;
  cmdline: string[];
  env: Record<string, string>;
  user_vars: Record<string, string>;
}

export interface KittyTab {
  id: number;
  title: string;
  windows: KittyWindow[];
}

export interface KittyOsWindow {
  id: number;
  tabs: KittyTab[];
}

export interface LaunchOptions {
  cwd?: string;
  tabTitle?: string;
  windowTitle?: string;
  vars?: Record<string, string>;
  /** Command to execute in the new tab (e.g., ["claude", "--resume", "abc123"]) */
  command?: string[];
}

/**
 * Kitty remote control client.
 * Wraps kitten @ commands for controlling kitty terminal.
 */
export class KittyRc {
  private socket: string;
  private passwordEnv?: string;

  constructor(socket = KITTY_SOCKET, passwordEnv = KITTY_PASSWORD_ENV) {
    this.socket = socket;
    // Only use password env if the variable is actually set
    this.passwordEnv = process.env[passwordEnv] ? passwordEnv : undefined;
  }

  /**
   * Build base arguments for kitten @ commands.
   */
  private baseArgs(): string[] {
    const args = ["@", "--to", this.socket];
    if (this.passwordEnv) {
      args.push("--password-env", this.passwordEnv);
    }
    return args;
  }

  /**
   * Check if kitty is reachable via the socket.
   */
  async health(): Promise<boolean> {
    try {
      await this.ls();
      return true;
    } catch {
      return false;
    }
  }

  /**
   * List all kitty OS windows, tabs, and windows.
   */
  async ls(): Promise<KittyOsWindow[]> {
    const { stdout } = await withTimeout(
      execFileAsync("kitten", [...this.baseArgs(), "ls"]),
      KITTY_COMMAND_TIMEOUT_MS
    );
    return JSON.parse(stdout);
  }

  /**
   * Focus a window by ID.
   * Returns true if successful, false if window not found.
   */
  async focusWindow(windowId: number): Promise<boolean> {
    try {
      await withTimeout(
        execFileAsync("kitten", [
          ...this.baseArgs(),
          "focus-window",
          "--match",
          `id:${windowId}`,
        ]),
        KITTY_COMMAND_TIMEOUT_MS
      );
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Launch a new tab with the given options.
   * Returns the window ID of the newly created window.
   */
  async launchTab(opts: LaunchOptions): Promise<number> {
    const args = [...this.baseArgs(), "launch", "--type=tab"];

    if (opts.cwd) args.push("--cwd", opts.cwd);
    if (opts.tabTitle) args.push("--tab-title", opts.tabTitle);
    if (opts.windowTitle) args.push("--title", opts.windowTitle);
    if (opts.vars) {
      for (const [key, value] of Object.entries(opts.vars)) {
        args.push("--var", `${key}=${value}`);
      }
    }

    // Add command to run in the new tab
    if (opts.command && opts.command.length > 0) {
      args.push(...opts.command);
    }

    const { stdout } = await withTimeout(
      execFileAsync("kitten", args),
      KITTY_COMMAND_TIMEOUT_MS
    );
    return parseInt(stdout.trim(), 10);
  }

  /**
   * Send text to a window.
   * If submit is true, appends a carriage return to press Enter.
   */
  async sendText(
    windowId: number,
    text: string,
    submit: boolean
  ): Promise<void> {
    const args = [
      ...this.baseArgs(),
      "send-text",
      "--match",
      `id:${windowId}`,
      "--stdin",
    ];

    const payload = submit ? text + "\n" : text;

    await withTimeout(
      new Promise<void>((resolve, reject) => {
        const child = execFile("kitten", args, (err) => {
          if (err) reject(err);
          else resolve();
        });
        child.stdin?.write(payload);
        child.stdin?.end();
      }),
      KITTY_COMMAND_TIMEOUT_MS
    );
  }

  /**
   * Open interactive window selector and return selected window ID.
   * Returns null if user cancels selection.
   */
  async selectWindow(): Promise<number | null> {
    try {
      const { stdout } = await execFileAsync("kitten", [
        ...this.baseArgs(),
        "select-window",
      ]);
      const id = parseInt(stdout.trim(), 10);
      return isNaN(id) ? null : id;
    } catch {
      return null; // User cancelled selection
    }
  }

  /**
   * Set the tab title for a window.
   */
  async setTabTitle(windowId: number, title: string): Promise<void> {
    await withTimeout(
      execFileAsync("kitten", [
        ...this.baseArgs(),
        "set-tab-title",
        "--match",
        `id:${windowId}`,
        title,
      ]),
      KITTY_COMMAND_TIMEOUT_MS
    );
  }

  /**
   * Find window ID for a session by checking user_vars.
   */
  findWindowBySessionId(
    osWindows: KittyOsWindow[],
    sessionId: string
  ): number | null {
    for (const osWin of osWindows) {
      for (const tab of osWin.tabs) {
        for (const win of tab.windows) {
          if (win.user_vars?.cc_session_id === sessionId) {
            return win.id;
          }
        }
      }
    }
    return null;
  }

  /**
   * Check if a window ID exists in the current window list.
   */
  windowExists(osWindows: KittyOsWindow[], windowId: number): boolean {
    for (const osWin of osWindows) {
      for (const tab of osWin.tabs) {
        for (const win of tab.windows) {
          if (win.id === windowId) return true;
        }
      }
    }
    return false;
  }
}
