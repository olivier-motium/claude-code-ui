/**
 * TerminalDock - Persistent terminal panel for the Command Center
 *
 * Features:
 * - Shows terminal for selected session
 * - Creates PTY on first attach
 * - Preserves terminal state when switching (via CSS show/hide)
 * - Header with session info and controls
 */

import { useState, useEffect, useCallback, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Terminal } from "../terminal/Terminal";
import { SessionHeader } from "./SessionHeader";
import { createPty, getPty, type PtyInfo } from "../../lib/api";
import type { Session } from "../../types/schema";

interface TerminalDockProps {
  session: Session | null;
  onClose: () => void;
}

interface TerminalState {
  ptyInfo: PtyInfo | null;
  isLoading: boolean;
  error: string | null;
  isConnected: boolean;
}

/**
 * TerminalDock component - persistent terminal panel
 */
export function TerminalDock({ session, onClose }: TerminalDockProps) {
  const [state, setState] = useState<TerminalState>({
    ptyInfo: null,
    isLoading: false,
    error: null,
    isConnected: false,
  });

  // Track which session we've already initialized PTY for
  const initializedSessionRef = useRef<string | null>(null);

  // Initialize PTY when session changes
  const initializePty = useCallback(async (sessionId: string) => {
    // Skip if already initializing or initialized for this session
    if (initializedSessionRef.current === sessionId) {
      return;
    }

    setState((prev) => ({ ...prev, isLoading: true, error: null }));
    initializedSessionRef.current = sessionId;

    try {
      // Check if PTY already exists
      let ptyInfo = await getPty(sessionId);

      // Create if doesn't exist
      if (!ptyInfo) {
        ptyInfo = await createPty(sessionId);
      }

      setState({
        ptyInfo,
        isLoading: false,
        error: null,
        isConnected: false,
      });
    } catch (err) {
      console.error("[TerminalDock] Failed to initialize PTY:", err);
      setState({
        ptyInfo: null,
        isLoading: false,
        error: err instanceof Error ? err.message : "Failed to create terminal",
        isConnected: false,
      });
      initializedSessionRef.current = null;
    }
  }, []);

  // When session changes, initialize PTY
  useEffect(() => {
    if (session) {
      initializePty(session.sessionId);
    } else {
      // Reset state when no session selected
      setState({
        ptyInfo: null,
        isLoading: false,
        error: null,
        isConnected: false,
      });
      initializedSessionRef.current = null;
    }
  }, [session?.sessionId, initializePty]);

  // Handle WebSocket connection status
  const handleConnect = useCallback(() => {
    setState((prev) => ({ ...prev, isConnected: true }));
  }, []);

  const handleDisconnect = useCallback(() => {
    setState((prev) => ({ ...prev, isConnected: false }));
  }, []);

  const handleError = useCallback((error: string) => {
    setState((prev) => ({ ...prev, error }));
  }, []);

  // Retry connection
  const handleRetry = useCallback(() => {
    if (session) {
      initializedSessionRef.current = null;
      initializePty(session.sessionId);
    }
  }, [session, initializePty]);

  // Don't render if no session
  if (!session) {
    return null;
  }

  const { ptyInfo, isLoading, error, isConnected } = state;

  return (
    <div className="terminal-dock">
      {/* Header */}
      <SessionHeader
        session={session}
        isConnected={isConnected}
        isLoading={isLoading}
        onClose={onClose}
      />

      {/* Terminal body */}
      <div className="terminal-dock-body h-[280px]">
        {isLoading ? (
          <div className="flex items-center justify-center h-full">
            <span className="text-sm text-muted-foreground">
              Connecting to terminal...
            </span>
          </div>
        ) : error ? (
          <div className="flex flex-col items-center justify-center gap-3 h-full">
            <span className="text-sm text-destructive">{error}</span>
            <Button size="sm" variant="outline" onClick={handleRetry}>
              Retry
            </Button>
          </div>
        ) : ptyInfo ? (
          <Terminal
            wsUrl={ptyInfo.wsUrl}
            wsToken={ptyInfo.wsToken}
            onConnect={handleConnect}
            onDisconnect={handleDisconnect}
            onError={handleError}
          />
        ) : (
          <div className="flex items-center justify-center h-full">
            <span className="text-sm text-muted-foreground">
              No terminal available
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
