import { createRootRoute, Outlet } from "@tanstack/react-router";
import { TanStackRouterDevtools } from "@tanstack/router-devtools";
import { Theme, Box, Heading, Flex, Text } from "@radix-ui/themes";
import { getSessionsDb } from "../data/sessionsDb";

export const Route = createRootRoute({
  loader: async () => {
    // Initialize db and preload data before any route renders
    await getSessionsDb();
    return {};
  },
  component: RootLayout,
});

function RootLayout() {
  return (
    <Theme
      accentColor="violet"
      grayColor="slate"
      radius="large"
      scaling="100%"
      appearance="dark"
    >
      <Box px="5" py="5" style={{ maxWidth: "1800px", margin: "0 auto" }}>
        <Flex direction="column" gap="5">
          <Flex align="center" gap="3">
            <Heading size="8" weight="bold">
              Sessions
            </Heading>
            <Text size="2" color="gray" style={{ marginTop: "8px" }}>
              Claude Code
            </Text>
          </Flex>
          <Outlet />
        </Flex>
      </Box>
      <TanStackRouterDevtools />
    </Theme>
  );
}
