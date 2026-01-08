#!/usr/bin/env node
/**
 * CLI command for manually setting up kitty terminal remote control.
 * Run with: pnpm setup:kitty
 */

import { setupKitty, getKittyStatus } from "../kitty-setup.js";
import { colors } from "../utils/colors.js";

async function main(): Promise<void> {
  console.log(`${colors.bold}Claude Code - Kitty Setup${colors.reset}\n`);

  // Show current status
  const status = await getKittyStatus();
  console.log("Current status:");
  console.log(`  Kitty installed: ${status.installed ? colors.green + "yes" : colors.yellow + "no"}${colors.reset}`);
  console.log(`  Kitty running: ${status.running ? colors.green + "yes" : colors.dim + "no"}${colors.reset}`);
  console.log(`  Socket exists: ${status.socketExists ? colors.green + "yes" : colors.dim + "no"}${colors.reset}`);
  console.log(`  Socket reachable: ${status.socketReachable ? colors.green + "yes" : colors.yellow + "no"}${colors.reset}`);
  console.log(`  Config exists: ${status.configExists ? colors.green + "yes" : colors.dim + "no"}${colors.reset}`);
  console.log();

  if (status.socketReachable) {
    console.log(`${colors.green}Kitty remote control is already working.${colors.reset}`);
    return;
  }

  // Run setup
  console.log("Running setup...\n");
  const result = await setupKitty();

  for (const action of result.actions) {
    console.log(`${colors.green}✓${colors.reset} ${action}`);
  }

  console.log();

  if (result.success) {
    console.log(`${colors.green}${result.message}${colors.reset}`);
  } else {
    console.log(`${colors.yellow}${result.message}${colors.reset}`);
  }

  if (result.status === "config_needs_reload") {
    console.log(`\n${colors.yellow}Note:${colors.reset} Restart kitty to apply changes.`);
  }
}

main().catch((error) => {
  console.error(`${colors.red}Error:${colors.reset}`, error.message);
  process.exit(1);
});
