import { createRootRoute, Outlet } from "@tanstack/react-router";
import { TanStackRouterDevtools } from "@tanstack/router-devtools";
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
    <div className="dark min-h-screen bg-background text-foreground">
      <Outlet />
      <TanStackRouterDevtools />
    </div>
  );
}
