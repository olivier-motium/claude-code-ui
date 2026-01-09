/**
 * Single row in the OpsTable
 * Displays session status, goal, branch, pending tool, activity time, and repo
 */

import { useState } from "react";
import { Flex, Text, Code, Badge, Box, Tooltip } from "@radix-ui/themes";
import { SendTextDialog } from "../SendTextDialog";
import { SessionActions } from "../session-card/SessionActions";
import { getEffectiveStatus } from "../../lib/sessionStatus";
import { formatTimeAgo, formatTarget, formatGoal, getRowClass } from "./utils";
import { TOOL_ICONS, STATUS_ICONS, STATUS_COLORS, STATUS_LABELS } from "./constants";
import type { OpsTableRowProps } from "./types";

export function OpsTableRow({ session, isSelected, onSelect }: OpsTableRowProps) {
  const [sendTextOpen, setSendTextOpen] = useState(false);
  const { status, fileStatusValue } = getEffectiveStatus(session);

  return (
    <>
      <Box
        className={getRowClass(session, isSelected)}
        onClick={onSelect}
        tabIndex={0}
        role="button"
        aria-pressed={isSelected}
        style={{
          padding: "var(--space-2) var(--space-3)",
          cursor: "pointer",
          borderBottom: "1px solid var(--gray-a4)",
        }}
      >
        <Flex align="center" gap="3">
          {/* Status indicator with tooltip */}
          <Box style={{ width: "24px", textAlign: "center" }}>
            <Tooltip content={STATUS_LABELS[status]} side="right">
              <Text
                size="2"
                color={STATUS_COLORS[status] as "green" | "orange" | "gray"}
                aria-label={STATUS_LABELS[status]}
              >
                {STATUS_ICONS[status]}
              </Text>
            </Tooltip>
          </Box>

          {/* Goal / prompt - flex grow */}
          <Box style={{ flex: 1, minWidth: 0 }}>
            <Flex align="center" gap="2">
              <Text size="2" weight={isSelected ? "medium" : "regular"} truncate>
                {formatGoal(session)}
              </Text>
              {fileStatusValue === "completed" && (
                <Badge color="blue" variant="soft" size="1">Done</Badge>
              )}
              {fileStatusValue === "error" && (
                <Badge color="red" variant="soft" size="1">Error</Badge>
              )}
              {fileStatusValue === "blocked" && (
                <Badge color="orange" variant="soft" size="1">Blocked</Badge>
              )}
            </Flex>
          </Box>

          {/* Branch */}
          <Box style={{ width: "100px" }}>
            {session.gitBranch ? (
              <Code size="1" variant="soft" color="gray" truncate>
                {session.gitBranch.length > 15
                  ? session.gitBranch.slice(0, 12) + "..."
                  : session.gitBranch}
              </Code>
            ) : (
              <Text size="1" color="gray">-</Text>
            )}
          </Box>

          {/* Pending tool */}
          <Box style={{ width: "80px" }}>
            {session.pendingTool ? (
              <Flex align="center" gap="1">
                <Text size="1">{TOOL_ICONS[session.pendingTool.tool] || "🔧"}</Text>
                <Code size="1" color="orange" variant="soft">
                  {formatTarget(session.pendingTool.target)}
                </Code>
              </Flex>
            ) : (
              <Text size="1" color="gray">-</Text>
            )}
          </Box>

          {/* Activity time */}
          <Box style={{ width: "50px", textAlign: "right" }}>
            <Text size="1" color="gray">
              {formatTimeAgo(session.lastActivityAt)}
            </Text>
          </Box>

          {/* Repo */}
          <Box style={{ width: "80px" }}>
            <Text size="1" color="gray" truncate>
              {session.gitRepoId?.split("/")[1] || "-"}
            </Text>
          </Box>

          {/* Actions */}
          <Box style={{ width: "32px" }}>
            <SessionActions
              session={session}
              onSendText={() => setSendTextOpen(true)}
            />
          </Box>
        </Flex>
      </Box>

      <SendTextDialog
        sessionId={session.sessionId}
        open={sendTextOpen}
        onOpenChange={setSendTextOpen}
      />
    </>
  );
}
