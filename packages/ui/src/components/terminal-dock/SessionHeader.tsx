/**
 * SessionHeader - Header bar for the TerminalDock
 *
 * Shows:
 * - Session goal/status
 * - Branch info
 * - Terminal status
 * - Close button
 */

import { X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { getEffectiveStatus } from "../../lib/sessionStatus";
import type { Session } from "../../types/schema";

interface SessionHeaderProps {
  session: Session;
  isConnected: boolean;
  isLoading: boolean;
  onClose: () => void;
}

const STATUS_STYLES = {
  working: "bg-status-working/10 text-status-working border-status-working/20",
  waiting: "bg-status-waiting/10 text-status-waiting border-status-waiting/20",
  idle: "bg-status-idle/10 text-status-idle border-status-idle/20",
} as const;

const STATUS_LABELS = {
  working: "Working",
  waiting: "Waiting",
  idle: "Idle",
} as const;

export function SessionHeader({
  session,
  isConnected,
  isLoading,
  onClose,
}: SessionHeaderProps) {
  const { status } = getEffectiveStatus(session);
  const goalText = session.goal || session.originalPrompt.slice(0, 50);

  return (
    <div className="terminal-dock-header flex items-center justify-between gap-3">
      <div className="flex items-center gap-3 flex-1 min-w-0">
        {/* Status badge */}
        <span
          className={cn(
            "inline-flex items-center px-2 py-0.5 text-xs font-medium rounded-full border",
            STATUS_STYLES[status]
          )}
        >
          {STATUS_LABELS[status]}
        </span>

        {/* Goal text */}
        <span className="text-sm font-medium truncate flex-1">
          {goalText}
        </span>

        {/* Branch */}
        {session.gitBranch && (
          <code className="px-1.5 py-0.5 text-xs bg-muted text-muted-foreground rounded font-mono">
            {session.gitBranch.length > 20
              ? session.gitBranch.slice(0, 17) + "..."
              : session.gitBranch}
          </code>
        )}

        {/* Connection status */}
        <span
          className={cn(
            "inline-flex items-center px-2 py-0.5 text-xs font-medium rounded-full border",
            isLoading
              ? "bg-status-waiting/10 text-status-waiting border-status-waiting/20"
              : isConnected
              ? "bg-status-working/10 text-status-working border-status-working/20"
              : "bg-status-error/10 text-status-error border-status-error/20"
          )}
        >
          {isLoading ? "Connecting..." : isConnected ? "Connected" : "Disconnected"}
        </span>
      </div>

      {/* Close button */}
      <Button
        variant="ghost"
        size="icon-sm"
        onClick={onClose}
        title="Close terminal"
        className="h-7 w-7"
      >
        <X className="h-4 w-4" />
      </Button>
    </div>
  );
}
