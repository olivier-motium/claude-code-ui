/**
 * Session action menu for terminal control
 *
 * Provides unified access to:
 * - Embedded terminal (primary)
 * - Kitty terminal (secondary submenu)
 * - Send message (works for both)
 */

import { useState } from "react";
import { useNavigate } from "@tanstack/react-router";
import { MoreVertical } from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Button } from "@/components/ui/button";
import * as api from "../../lib/api";
import type { SessionActionsProps } from "./types";

export function SessionActions({ session, onSendText }: SessionActionsProps) {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);

  /**
   * Factory for async action handlers.
   * Wraps an async action with stopPropagation, loading state, and error logging.
   */
  const createAsyncHandler = (
    action: () => Promise<void>,
    actionName: string
  ) => async (e: React.MouseEvent) => {
    e.stopPropagation();
    setLoading(true);
    try {
      await action();
    } catch (error) {
      console.error(`${actionName} failed:`, error);
    } finally {
      setLoading(false);
    }
  };

  // Open embedded terminal (primary action)
  const handleOpenTerminal = (e: React.MouseEvent) => {
    e.stopPropagation();
    navigate({
      to: "/session/$sessionId/terminal",
      params: { sessionId: session.sessionId },
    });
  };

  // Kitty actions
  const handleKittyFocus = createAsyncHandler(
    () => api.focusSession(session.sessionId),
    "Focus"
  );
  const handleKittyOpen = createAsyncHandler(
    () => api.openSession(session.sessionId),
    "Open"
  );
  const handleKittyLink = createAsyncHandler(
    () => api.linkTerminal(session.sessionId),
    "Link"
  );
  const handleKittyUnlink = createAsyncHandler(
    () => api.unlinkTerminal(session.sessionId),
    "Unlink"
  );

  // Send text (works for both embedded and kitty)
  const handleSendText = (e: React.MouseEvent) => {
    e.stopPropagation();
    onSendText?.();
  };

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="ghost"
          size="icon-sm"
          onClick={(e) => e.stopPropagation()}
          disabled={loading}
          className="h-7 w-7"
        >
          <MoreVertical className="h-4 w-4" />
          <span className="sr-only">Actions</span>
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" onClick={(e) => e.stopPropagation()}>
        {/* Primary: Open embedded terminal */}
        <DropdownMenuItem onClick={handleOpenTerminal}>
          Open terminal
        </DropdownMenuItem>

        {/* Kitty submenu */}
        <DropdownMenuSub>
          <DropdownMenuSubTrigger>Kitty...</DropdownMenuSubTrigger>
          <DropdownMenuSubContent>
            {session.terminalLink ? (
              <>
                <DropdownMenuItem onClick={handleKittyFocus}>
                  Focus kitty window
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem variant="destructive" onClick={handleKittyUnlink}>
                  Unlink kitty
                </DropdownMenuItem>
              </>
            ) : (
              <>
                <DropdownMenuItem onClick={handleKittyOpen}>
                  Open in kitty
                </DropdownMenuItem>
                <DropdownMenuItem onClick={handleKittyLink}>
                  Link existing kitty window...
                </DropdownMenuItem>
              </>
            )}
          </DropdownMenuSubContent>
        </DropdownMenuSub>

        {/* Send text - works for both */}
        <DropdownMenuSeparator />
        <DropdownMenuItem onClick={handleSendText}>
          Send message...
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
