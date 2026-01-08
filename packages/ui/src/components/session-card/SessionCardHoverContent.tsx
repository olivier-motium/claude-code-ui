/**
 * Hover panel content with detailed session info
 */

import { Flex, Text, Box, Badge } from "@radix-ui/themes";
import { getRoleColor, getRolePrefix, getCIStatusIcon, getCIStatusColor } from "./utils";
import type { SessionCardProps } from "./types";
import type { RecentOutput, CIStatus } from "@claude-code-ui/daemon/schema";

interface CICheck {
  name: string;
  status: CIStatus;
  url: string | null;
}

export function SessionCardHoverContent({ session }: SessionCardProps) {
  return (
    <Flex direction="column" gap="3" style={{ height: "100%" }}>
      {/* Header: goal */}
      <Text size="2" weight="bold" highContrast>
        {session.goal || session.originalPrompt.slice(0, 60)}
      </Text>

      {/* Recent output */}
      <Box
        p="3"
        flexGrow="1"
        style={{
          backgroundColor: "var(--gray-2)",
          borderRadius: "var(--radius-3)",
          overflow: "auto",
        }}
      >
        {session.recentOutput.length > 0 ? (
          session.recentOutput.map((output: RecentOutput, i: number) => (
            <Text
              key={i}
              as="p"
              size="1"
              mb="2"
              style={{
                color: getRoleColor(output.role),
                whiteSpace: "pre-wrap",
                margin: 0,
                marginBottom: i < session.recentOutput.length - 1 ? "8px" : 0,
              }}
            >
              {getRolePrefix(output.role)}
              {output.content}
            </Text>
          ))
        ) : (
          <Text size="1" color="gray">
            No recent output
          </Text>
        )}
        {session.status === "working" && (
          <Text color="grass" size="1">█</Text>
        )}
      </Box>

      {/* PR Info if available */}
      {session.pr && (
        <Box>
          <Flex align="center" gap="2" mb="2">
            <a
              href={session.pr.url}
              target="_blank"
              rel="noopener noreferrer"
              style={{ fontSize: "var(--font-size-1)", fontWeight: 500 }}
            >
              PR #{session.pr.number}: {session.pr.title}
            </a>
          </Flex>
          {session.pr.ciChecks.length > 0 && (
            <Flex gap="2" wrap="wrap">
              {session.pr.ciChecks.map((check: CICheck) => (
                <Badge
                  key={check.name}
                  color={getCIStatusColor(check.status)}
                  variant="soft"
                  size="1"
                >
                  {getCIStatusIcon(check.status)} {check.name.slice(0, 20)}
                </Badge>
              ))}
            </Flex>
          )}
        </Box>
      )}

      {/* Footer */}
      <Flex justify="between">
        <Text size="1" color="gray">
          {session.cwd.replace(/^\/Users\/\w+\//, "~/")}
        </Text>
        <Text size="1" color="gray">
          {session.sessionId.slice(0, 8)}
        </Text>
      </Flex>
    </Flex>
  );
}
