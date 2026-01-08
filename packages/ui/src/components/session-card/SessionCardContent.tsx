/**
 * Main card body content
 */

import { Card, Flex, Text, Code, Badge } from "@radix-ui/themes";
import { SessionActions } from "./SessionActions";
import { toolIcons } from "./constants";
import { getCardClass, formatTimeAgo, formatTarget, getCIStatusIcon, getCIStatusColor } from "./utils";
import type { SessionCardContentProps } from "./types";

export function SessionCardContent({ session, onSendText }: SessionCardContentProps) {
  const { pendingTool } = session;
  const dirPath = session.cwd.replace(/^\/Users\/[^/]+/, "~");

  return (
    <Card size="2" className={getCardClass(session)} style={{ cursor: "pointer" }}>
      <Flex direction="column" gap="2">
        {/* Header: directory, terminal status, time, and actions */}
        <Flex justify="between" align="center">
          <Flex align="center" gap="2">
            <Text size="1" color="gray" style={{ fontFamily: "var(--code-font-family)" }}>
              {dirPath}
            </Text>
            {session.terminalLink && (
              <Badge
                size="1"
                color={session.terminalLink.stale ? "orange" : "green"}
                variant="soft"
              >
                {session.terminalLink.stale ? "stale" : "linked"}
              </Badge>
            )}
          </Flex>
          <Flex align="center" gap="2">
            <Text size="1" color="gray">
              {formatTimeAgo(session.lastActivityAt)}
            </Text>
            <SessionActions session={session} onSendText={onSendText} />
          </Flex>
        </Flex>

        {/* Main content: goal as primary text */}
        <Text size="2" weight="medium" highContrast mb="1">
          {session.goal || session.originalPrompt.slice(0, 50)}
        </Text>

        {/* Secondary: current activity (pending tool or summary) */}
        {pendingTool ? (
          <Flex align="center" gap="2">
            <Text size="1" color="gray">
              {toolIcons[pendingTool.tool]}
            </Text>
            <Code size="1" color="orange" variant="soft">
              {pendingTool.tool}: {formatTarget(pendingTool.target)}
            </Code>
          </Flex>
        ) : (
          <Text size="1" color="gray">
            {session.summary}
          </Text>
        )}

        {/* Footer: branch/PR info and message count */}
        <Flex align="center" justify="between" gap="2">
          <Flex align="center" gap="2">
            {session.pr ? (
              <a
                href={session.pr.url}
                target="_blank"
                rel="noopener noreferrer"
                onClick={(e) => e.stopPropagation()}
                style={{ textDecoration: "none" }}
              >
                <Badge color={getCIStatusColor(session.pr.ciStatus)} variant="soft" size="1">
                  {getCIStatusIcon(session.pr.ciStatus)} #{session.pr.number}
                </Badge>
              </a>
            ) : session.gitBranch ? (
              <Code size="1" variant="soft" color="gray">
                {session.gitBranch.length > 20
                  ? session.gitBranch.slice(0, 17) + "..."
                  : session.gitBranch}
              </Code>
            ) : null}
          </Flex>
          <Text size="1" color="gray">
            {session.messageCount} msgs
          </Text>
        </Flex>
      </Flex>
    </Card>
  );
}
