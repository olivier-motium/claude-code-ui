/**
 * Text utility functions for summarization.
 */

import { GOAL_TRUNCATE_LENGTH } from "../config.js";

/**
 * Clean and truncate goal text
 */
export function cleanGoalText(text: string): string {
  // Remove markdown, quotes, extra whitespace
  let clean = text
    .replace(/^["']|["']$/g, "") // Remove surrounding quotes
    .replace(/\*\*/g, "") // Remove bold markdown
    .replace(/#{1,6}\s*/g, "") // Remove headers
    .replace(/\n.*/g, "") // Only keep first line
    .replace(/\s+/g, " ") // Normalize whitespace
    .trim();

  // Truncate to reasonable length
  if (clean.length > GOAL_TRUNCATE_LENGTH) {
    clean = clean.slice(0, GOAL_TRUNCATE_LENGTH - 3) + "...";
  }

  return clean;
}
