/**
 * Automatic kitty terminal remote control configuration.
 * Creates non-invasive config via include directives.
 */
import fs from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { KITTY_SOCKET } from "./config.js";

const execFileAsync = promisify(execFile);

/** Our dedicated config file (separate from user's main config) */
const CLAUDE_CODE_CONF = "claude-code.conf";

/** Comment marker for include directive */
const INCLUDE_COMMENT = "# Claude Code Session Tracker - kitty integration";

/** Result of kitty setup operation */
export interface KittySetupResult {
  success: boolean;
  status: KittyStatus;
  message: string;
  actions: string[];
}

/** Possible kitty configuration states */
export type KittyStatus =
  | "not_installed"
  | "not_running"
  | "not_configured"
  | "config_needs_reload"
  | "ready"
  | "setup_failed";

/** Detailed kitty status for diagnostics */
export interface KittyStatusDetails {
  installed: boolean;
  running: boolean;
  socketExists: boolean;
  socketReachable: boolean;
  configExists: boolean;
}

/**
 * Get the kitty config directory path.
 * Follows XDG base directory spec.
 */
function getKittyConfigDir(): string {
  if (process.env.KITTY_CONFIG_DIRECTORY) {
    return process.env.KITTY_CONFIG_DIRECTORY;
  }
  if (process.env.XDG_CONFIG_HOME) {
    return path.join(process.env.XDG_CONFIG_HOME, "kitty");
  }
  return path.join(os.homedir(), ".config", "kitty");
}

/**
 * Check if kitty is installed by looking for the kitten command.
 */
export async function isKittyInstalled(): Promise<boolean> {
  try {
    await execFileAsync("which", ["kitten"]);
    return true;
  } catch {
    return false;
  }
}

/**
 * Check if kitty is currently running.
 */
async function isKittyRunning(): Promise<boolean> {
  try {
    await execFileAsync("pgrep", ["-x", "kitty"]);
    return true;
  } catch {
    return false;
  }
}

/**
 * Check if kitty socket is reachable and responding.
 */
export async function isKittyReachable(): Promise<boolean> {
  try {
    const socketPath = KITTY_SOCKET.replace("unix:", "");
    await fs.access(socketPath);
    await execFileAsync("kitten", ["@", "--to", KITTY_SOCKET, "ls"], {
      timeout: 5000,
    });
    return true;
  } catch {
    return false;
  }
}

/**
 * Check if our claude-code.conf file exists.
 */
export async function hasClaudeCodeConfig(): Promise<boolean> {
  const configDir = getKittyConfigDir();
  const configPath = path.join(configDir, CLAUDE_CODE_CONF);
  try {
    await fs.access(configPath);
    return true;
  } catch {
    return false;
  }
}

/**
 * Create the claude-code.conf file with required settings.
 */
async function createClaudeCodeConfig(): Promise<void> {
  const configDir = getKittyConfigDir();
  const configPath = path.join(configDir, CLAUDE_CODE_CONF);

  // Ensure config directory exists
  await fs.mkdir(configDir, { recursive: true });

  const socketPath = KITTY_SOCKET.replace("unix:", "");
  const config = `# Claude Code Session Tracker - Kitty Integration
# This file is auto-generated. Safe to delete if you want to disable integration.
# See: https://github.com/olivier-motium/claude-code-ui

# Enable remote control via socket only (secure, no passwords)
allow_remote_control socket-only

# Listen on a fixed socket path for daemon connection
listen_on unix:${socketPath}
`;

  await fs.writeFile(configPath, config, "utf-8");
}

/**
 * Add include directive to main kitty.conf.
 * Returns true if directive was added, false if already present.
 */
async function addIncludeDirective(): Promise<boolean> {
  const configDir = getKittyConfigDir();
  const mainConfigPath = path.join(configDir, "kitty.conf");
  const includeStatement = `include ${CLAUDE_CODE_CONF}`;

  // Ensure config directory exists
  await fs.mkdir(configDir, { recursive: true });

  let existingConfig = "";
  try {
    existingConfig = await fs.readFile(mainConfigPath, "utf-8");
  } catch {
    // Config doesn't exist, create it
  }

  // Check if already included
  if (existingConfig.includes(includeStatement)) {
    return false; // Already configured
  }

  // Append include directive
  const newConfig =
    existingConfig + `\n\n${INCLUDE_COMMENT}\n${includeStatement}\n`;

  // Backup existing config if it exists
  if (existingConfig) {
    const backupPath = `${mainConfigPath}.backup.${Date.now()}`;
    await fs.writeFile(backupPath, existingConfig, "utf-8");
  }

  await fs.writeFile(mainConfigPath, newConfig, "utf-8");
  return true;
}

/**
 * Create macOS launch services command line file for GUI launches.
 * This ensures kitty launched from Dock/Spotlight has remote control enabled.
 */
async function createMacOSLaunchConfig(): Promise<void> {
  if (process.platform !== "darwin") return;

  const configDir = getKittyConfigDir();
  const launchConfigPath = path.join(
    configDir,
    "macos-launch-services-cmdline"
  );

  const socketPath = KITTY_SOCKET.replace("unix:", "");
  // Each argument on its own line
  const config = `-o
allow_remote_control=socket-only
--listen-on
unix:${socketPath}
`;

  await fs.writeFile(launchConfigPath, config, "utf-8");
}

/**
 * Send SIGUSR1 to all kitty processes to reload config.
 * Returns true if signal was sent to at least one process.
 */
async function reloadKittyConfig(): Promise<boolean> {
  try {
    const { stdout } = await execFileAsync("pgrep", ["-x", "kitty"]);
    const pids = stdout.trim().split("\n").filter(Boolean);

    for (const pid of pids) {
      try {
        process.kill(parseInt(pid, 10), "SIGUSR1");
      } catch {
        // Process may have exited between pgrep and kill
      }
    }

    return pids.length > 0;
  } catch {
    return false;
  }
}

/**
 * Run full kitty setup process.
 * Detects current state and applies necessary configuration.
 */
export async function setupKitty(): Promise<KittySetupResult> {
  const actions: string[] = [];

  // Step 1: Check if kitty is installed
  if (!(await isKittyInstalled())) {
    return {
      success: false,
      status: "not_installed",
      message:
        "Kitty terminal is not installed. Install from https://sw.kovidgoyal.net/kitty/",
      actions,
    };
  }

  // Step 2: Check if already working
  if (await isKittyReachable()) {
    return {
      success: true,
      status: "ready",
      message: "Kitty remote control is already configured and working",
      actions,
    };
  }

  try {
    // Step 3: Create our config file
    const hadConfig = await hasClaudeCodeConfig();
    if (!hadConfig) {
      await createClaudeCodeConfig();
      actions.push("Created ~/.config/kitty/claude-code.conf");
    }

    // Step 4: Add include directive to main config
    const addedInclude = await addIncludeDirective();
    if (addedInclude) {
      actions.push("Added include directive to kitty.conf");
    }

    // Step 5: Create macOS launch config (for GUI launches)
    if (process.platform === "darwin") {
      await createMacOSLaunchConfig();
      actions.push("Created macos-launch-services-cmdline for GUI launches");
    }

    // Step 6: Try to reload kitty config
    const reloaded = await reloadKittyConfig();
    if (reloaded) {
      actions.push("Sent SIGUSR1 to reload kitty config");

      // Wait a moment for reload and socket creation
      await new Promise((resolve) => setTimeout(resolve, 500));

      // Check if it worked
      if (await isKittyReachable()) {
        return {
          success: true,
          status: "ready",
          message: "Kitty remote control configured successfully",
          actions,
        };
      }
    }

    // If kitty is not running or reload didn't work, config will apply on next launch
    return {
      success: true,
      status: "config_needs_reload",
      message: "Config created. Restart kitty to enable remote control.",
      actions,
    };
  } catch (error) {
    return {
      success: false,
      status: "setup_failed",
      message: `Setup failed: ${error instanceof Error ? error.message : String(error)}`,
      actions,
    };
  }
}

/**
 * Get detailed kitty status for diagnostics.
 */
export async function getKittyStatus(): Promise<KittyStatusDetails> {
  const socketPath = KITTY_SOCKET.replace("unix:", "");

  let socketExists = false;
  try {
    await fs.access(socketPath);
    socketExists = true;
  } catch {
    // Socket doesn't exist
  }

  return {
    installed: await isKittyInstalled(),
    running: await isKittyRunning(),
    socketExists,
    socketReachable: await isKittyReachable(),
    configExists: await hasClaudeCodeConfig(),
  };
}
