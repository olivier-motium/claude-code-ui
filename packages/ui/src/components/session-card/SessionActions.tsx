/**
 * Session action menu for terminal control
 */

import { useState } from "react";
import { DropdownMenu, IconButton } from "@radix-ui/themes";
import { DotsVerticalIcon } from "@radix-ui/react-icons";
import * as api from "../../lib/api";
import type { SessionActionsProps } from "./types";

export function SessionActions({ session, onSendText }: SessionActionsProps) {
  const [loading, setLoading] = useState(false);

  const handleFocus = async (e: React.MouseEvent) => {
    e.stopPropagation();
    setLoading(true);
    try {
      await api.focusSession(session.sessionId);
    } catch (error) {
      console.error("Focus failed:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleOpen = async (e: React.MouseEvent) => {
    e.stopPropagation();
    setLoading(true);
    try {
      await api.openSession(session.sessionId);
    } catch (error) {
      console.error("Open failed:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleLink = async (e: React.MouseEvent) => {
    e.stopPropagation();
    setLoading(true);
    try {
      await api.linkTerminal(session.sessionId);
    } catch (error) {
      console.error("Link failed:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleUnlink = async (e: React.MouseEvent) => {
    e.stopPropagation();
    setLoading(true);
    try {
      await api.unlinkTerminal(session.sessionId);
    } catch (error) {
      console.error("Unlink failed:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleSendText = (e: React.MouseEvent) => {
    e.stopPropagation();
    onSendText?.();
  };

  return (
    <DropdownMenu.Root>
      <DropdownMenu.Trigger>
        <IconButton
          variant="ghost"
          size="1"
          onClick={(e) => e.stopPropagation()}
          disabled={loading}
        >
          <DotsVerticalIcon />
        </IconButton>
      </DropdownMenu.Trigger>
      <DropdownMenu.Content>
        {session.terminalLink ? (
          <>
            <DropdownMenu.Item onClick={handleFocus}>
              Focus terminal
            </DropdownMenu.Item>
            <DropdownMenu.Item onClick={handleSendText}>
              Send message...
            </DropdownMenu.Item>
            <DropdownMenu.Separator />
            <DropdownMenu.Item color="red" onClick={handleUnlink}>
              Unlink terminal
            </DropdownMenu.Item>
          </>
        ) : (
          <>
            <DropdownMenu.Item onClick={handleOpen}>
              Open in kitty
            </DropdownMenu.Item>
            <DropdownMenu.Item onClick={handleLink}>
              Link existing terminal...
            </DropdownMenu.Item>
          </>
        )}
      </DropdownMenu.Content>
    </DropdownMenu.Root>
  );
}
