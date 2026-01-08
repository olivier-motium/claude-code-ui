import { Box, Flex, Heading, Text, ScrollArea } from "@radix-ui/themes";
import { SessionCard } from "./session-card";
import type { Session } from "@claude-code-ui/daemon/schema";

interface KanbanColumnProps {
  title: string;
  sessions: Session[];
  color: "green" | "orange" | "yellow" | "gray";
}

const headerClassMap = {
  green: "column-header-working",
  orange: "column-header-approval",
  yellow: "column-header-waiting",
  gray: "column-header-idle",
};

export function KanbanColumn({ title, sessions, color }: KanbanColumnProps) {
  const colorMap = {
    green: "var(--grass-3)",
    orange: "var(--orange-3)",
    yellow: "var(--amber-3)",
    gray: "var(--slate-3)",
  };

  const countColorMap = {
    green: "grass",
    orange: "orange",
    yellow: "amber",
    gray: "gray",
  } as const;

  const borderColorMap = {
    green: "grass",
    orange: "orange",
    yellow: "amber",
    gray: "slate",
  } as const;

  return (
    <Box
      style={{
        flex: 1,
        minWidth: 320,
        maxWidth: 500,
        backgroundColor: colorMap[color],
        borderRadius: "var(--radius-4)",
        border: `1px solid var(--${borderColorMap[color]}-6)`,
      }}
      p="3"
    >
      <Flex direction="column" gap="3" style={{ height: "100%" }}>
        <Flex justify="between" align="center">
          <Heading
            size="3"
            weight="bold"
            className={`column-header ${headerClassMap[color]}`}
          >
            {title}
          </Heading>
          <Text size="2" weight="bold" color={countColorMap[color]}>
            {sessions.length}
          </Text>
        </Flex>

        <ScrollArea style={{ flex: 1 }}>
          <Flex direction="column" gap="2" pr="2">
            {sessions.map((session) => (
              <SessionCard key={session.sessionId} session={session} />
            ))}
            {sessions.length === 0 && (
              <Text
                size="2"
                color="gray"
                align="center"
                style={{ padding: "32px 20px", opacity: 0.6 }}
              >
                No sessions
              </Text>
            )}
          </Flex>
        </ScrollArea>
      </Flex>
    </Box>
  );
}
