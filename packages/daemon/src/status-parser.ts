/**
 * Parser for .claude/status.md files with YAML frontmatter.
 * Extracts structured status data from markdown files.
 */

import { z } from "zod";

// =============================================================================
// Schema Definitions
// =============================================================================

/** Valid status values from file-based status */
export const FileStatusValueSchema = z.enum([
  "working",
  "waiting_for_approval",
  "waiting_for_input",
  "completed",
  "error",
  "blocked",
  "idle",
]);
export type FileStatusValue = z.infer<typeof FileStatusValueSchema>;

/** Schema for parsed frontmatter */
export const StatusFrontmatterSchema = z.object({
  status: FileStatusValueSchema,
  updated: z.string(), // ISO timestamp
  task: z.string().optional(),
});

/** Full parsed status file */
export const ParsedStatusSchema = z.object({
  frontmatter: StatusFrontmatterSchema,
  summary: z.string().optional(),
  blockers: z.string().optional(),
  nextSteps: z.string().optional(),
});
export type ParsedStatus = z.infer<typeof ParsedStatusSchema>;

// =============================================================================
// Frontmatter Parsing
// =============================================================================

/**
 * Extract YAML frontmatter from markdown content.
 * Frontmatter must be delimited by --- on separate lines.
 */
function extractFrontmatter(content: string): { yaml: string; body: string } | null {
  const lines = content.split("\n");

  // Must start with ---
  if (lines[0]?.trim() !== "---") {
    return null;
  }

  // Find closing ---
  let closingIndex = -1;
  for (let i = 1; i < lines.length; i++) {
    if (lines[i]?.trim() === "---") {
      closingIndex = i;
      break;
    }
  }

  if (closingIndex === -1) {
    return null;
  }

  const yaml = lines.slice(1, closingIndex).join("\n");
  const body = lines.slice(closingIndex + 1).join("\n").trim();

  return { yaml, body };
}

/**
 * Parse simple YAML key: value pairs.
 * Only handles top-level string values (no nested objects/arrays).
 */
function parseSimpleYaml(yaml: string): Record<string, string> {
  const result: Record<string, string> = {};
  const lines = yaml.split("\n");

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;

    const colonIndex = trimmed.indexOf(":");
    if (colonIndex === -1) continue;

    const key = trimmed.slice(0, colonIndex).trim();
    const value = trimmed.slice(colonIndex + 1).trim();

    if (key && value) {
      result[key] = value;
    }
  }

  return result;
}

// =============================================================================
// Section Extraction
// =============================================================================

/**
 * Extract content under a markdown heading.
 * Returns content until the next heading of same or higher level.
 */
function extractSection(body: string, heading: string): string | undefined {
  const lines = body.split("\n");
  const headingPattern = new RegExp(`^##\\s+${heading}\\s*$`, "i");

  let inSection = false;
  const sectionLines: string[] = [];

  for (const line of lines) {
    if (inSection) {
      // Stop at next ## heading
      if (line.match(/^##\s+/)) {
        break;
      }
      sectionLines.push(line);
    } else if (line.match(headingPattern)) {
      inSection = true;
    }
  }

  const content = sectionLines.join("\n").trim();
  return content || undefined;
}

// =============================================================================
// Main Parser
// =============================================================================

/**
 * Parse a status.md file content into structured data.
 * Returns null if parsing fails.
 */
export function parseStatusFile(content: string): ParsedStatus | null {
  // Extract frontmatter
  const extracted = extractFrontmatter(content);
  if (!extracted) {
    return null;
  }

  // Parse YAML
  const yamlData = parseSimpleYaml(extracted.yaml);

  // Validate frontmatter
  const frontmatterResult = StatusFrontmatterSchema.safeParse(yamlData);
  if (!frontmatterResult.success) {
    return null;
  }

  // Extract sections from body
  const summary = extractSection(extracted.body, "Summary");
  const blockers = extractSection(extracted.body, "Blockers");
  const nextSteps = extractSection(extracted.body, "Next Steps");

  return {
    frontmatter: frontmatterResult.data,
    summary,
    blockers,
    nextSteps,
  };
}

/**
 * Map file status to UI status.
 * File has more granular statuses; UI uses 3 categories.
 */
export function mapToUiStatus(fileStatus: FileStatusValue): "working" | "waiting" | "idle" {
  switch (fileStatus) {
    case "working":
      return "working";
    case "waiting_for_approval":
    case "waiting_for_input":
      return "waiting";
    case "completed":
    case "error":
    case "blocked":
    case "idle":
      return "idle";
  }
}

/**
 * Check if a status file timestamp is stale.
 */
export function isStatusStale(updated: string, ttlMs: number): boolean {
  const updatedTime = new Date(updated).getTime();
  const now = Date.now();
  return now - updatedTime > ttlMs;
}
