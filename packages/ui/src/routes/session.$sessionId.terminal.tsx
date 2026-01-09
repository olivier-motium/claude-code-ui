/**
 * Fullscreen terminal route for embedded Claude Code sessions.
 *
 * Route: /session/:sessionId/terminal
 *
 * Renders a fullscreen terminal overlay with header showing
 * session info and close button.
 */

import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useEffect, useCallback } from "react";
import { ArrowLeft, X, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { Terminal } from "../components/terminal";
import { usePtySession } from "../hooks/usePtySession";
import { useSessions } from "../hooks/useSessions";

export const Route = createFileRoute("/session/$sessionId/terminal")({
  component: TerminalPage,
});

const STATUS_STYLES = {
  working: "bg-status-working/10 text-status-working border-status-working/20",
  waiting: "bg-status-waiting/10 text-status-waiting border-status-waiting/20",
  idle: "bg-status-idle/10 text-status-idle border-status-idle/20",
} as const;

function TerminalPage() {
  const { sessionId } = Route.useParams();
  const navigate = useNavigate();
  const { sessions } = useSessions();
  const session = sessions.find((s) => s.sessionId === sessionId);

  const { ptyInfo, isLoading, error, createPtySession } =
    usePtySession(sessionId);

  // Auto-create PTY if none exists
  useEffect(() => {
    if (!ptyInfo && !isLoading && !error) {
      createPtySession();
    }
  }, [ptyInfo, isLoading, error, createPtySession]);

  const handleClose = useCallback(() => {
    navigate({ to: "/" });
  }, [navigate]);

  // Handle escape key to close
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        handleClose();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleClose]);

  const status = session?.status as keyof typeof STATUS_STYLES | undefined;

  // Get directory name for display
  const dirName = session?.cwd.split("/").pop() || "Terminal";

  return (
    <div className="fixed inset-0 z-[1000] bg-background flex flex-col">
      {/* Header */}
      <div className="flex justify-between items-center px-4 py-3 border-b border-border bg-card shrink-0">
        <div className="flex items-center gap-3">
          <Button
            variant="ghost"
            size="icon"
            onClick={handleClose}
            title="Back to dashboard"
          >
            <ArrowLeft className="h-5 w-5" />
          </Button>

          <div className="flex flex-col gap-1">
            <div className="flex items-center gap-2">
              <span className="text-base font-medium">
                {session?.goal || dirName}
              </span>
              {session && status && (
                <span
                  className={cn(
                    "inline-flex items-center px-2 py-0.5 text-xs font-medium rounded-full border",
                    STATUS_STYLES[status]
                  )}
                >
                  {session.status}
                </span>
              )}
            </div>
            <span className="text-xs text-muted-foreground">
              {session?.cwd.replace(/^\/Users\/[^/]+/, "~") || sessionId}
            </span>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground">
            Press Esc to close
          </span>
          <Button variant="ghost" size="icon" onClick={handleClose}>
            <X className="h-5 w-5" />
          </Button>
        </div>
      </div>

      {/* Terminal area */}
      <div className="flex-1 overflow-hidden">
        {isLoading && (
          <div className="flex items-center justify-center h-full">
            <div className="flex flex-col items-center gap-3">
              <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
              <span className="text-muted-foreground">
                Connecting to terminal...
              </span>
            </div>
          </div>
        )}

        {error && (
          <div className="flex items-center justify-center h-full">
            <div className="flex flex-col items-center gap-3">
              <span className="text-destructive text-lg">{error}</span>
              <span className="text-muted-foreground text-sm">
                Make sure the daemon is running and try again.
              </span>
            </div>
          </div>
        )}

        {ptyInfo && (
          <Terminal
            wsUrl={ptyInfo.wsUrl}
            wsToken={ptyInfo.wsToken}
            onDisconnect={handleClose}
            onError={(err) => console.error("[Terminal] Error:", err)}
          />
        )}
      </div>
    </div>
  );
}
