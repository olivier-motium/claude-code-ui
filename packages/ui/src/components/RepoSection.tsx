import { Box, Flex, Heading, Link, Text, Separator } from "@radix-ui/themes";
import { KanbanColumn } from "./KanbanColumn";
import type { Session } from "@claude-code-ui/daemon/schema";

interface RepoSectionProps {
  repoId: string;
  repoUrl: string | null;
  sessions: Session[];
  activityScore: number;
}

export function RepoSection({ repoId, repoUrl, sessions, activityScore }: RepoSectionProps) {
  const working = sessions.filter((s) => s.status === "working");
  const needsApproval = sessions.filter(
    (s) => s.status === "waiting" && s.hasPendingToolUse
  );
  const waiting = sessions.filter(
    (s) => s.status === "waiting" && !s.hasPendingToolUse
  );
  const idle = sessions.filter((s) => s.status === "idle");

  const isHot = activityScore > 50;

  return (
    <Box mb="7">
      <Flex align="center" gap="3" mb="4">
        <Heading size="6" weight="bold">
          {repoId === "Other" ? (
            <Text color="gray">Other</Text>
          ) : repoUrl ? (
            <Link href={repoUrl} target="_blank" color="violet" highContrast>
              {repoId}
            </Link>
          ) : (
            repoId
          )}
        </Heading>
        {isHot && (
          <Text size="2" color="orange">
            🔥
          </Text>
        )}
        <Text size="2" color="gray">
          {sessions.length} session{sessions.length !== 1 ? "s" : ""}
        </Text>
      </Flex>

      <Flex gap="3" style={{ minHeight: 240 }}>
        <KanbanColumn
          title="Working"
          sessions={working}
          color="green"
        />
        <KanbanColumn
          title="Needs Approval"
          sessions={needsApproval}
          color="orange"
        />
        <KanbanColumn
          title="Waiting"
          sessions={waiting}
          color="yellow"
        />
        <KanbanColumn
          title="Idle"
          sessions={idle}
          color="gray"
        />
      </Flex>

      <Separator size="4" mt="6" />
    </Box>
  );
}
