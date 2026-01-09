/**
 * FleetCommand - Ops-First Bridge Layout
 *
 * Hybrid layout with mode switching:
 * - Ops Mode (default): OpsTable center, TerminalDock at bottom
 * - Focus Mode: Terminal center, mini roster
 *
 * Zones:
 * - Header: CommandBar + StatusStrip
 * - Left: Roster (full in ops, mini in focus)
 * - Center: OpsTable (ops) or Terminal (focus)
 * - Right: Tactical Intel
 * - Bottom: TerminalDock (ops only)
 * - Footer: Event Ticker
 */

import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { CommandBar } from "./CommandBar";
import { Roster } from "./Roster";
import { Viewport } from "./Viewport";
import { TacticalIntel } from "./TacticalIntel";
import { EventTicker } from "./EventTicker";
import { DataTable, columns } from "../data-table";
import { TerminalDock } from "../terminal-dock/TerminalDock";
import { StatusStrip } from "../StatusStrip";
import { getEffectiveStatus } from "../../lib/sessionStatus";
import { getAgentName } from "./constants";
import { countSessionsByStatus } from "../ops-table/utils";
import type { FleetCommandProps, AgentEvent, ViewMode } from "./types";
import type { StatusFilter } from "../ops-table/types";
import type { Session } from "../../types/schema";

/** Generate event from status change */
function generateStatusEvent(
  session: Session,
  prevStatus: string | null
): AgentEvent | null {
  const { status, fileStatusValue } = getEffectiveStatus(session);

  if (prevStatus === status) return null;

  let message = "";
  let type: AgentEvent["type"] = "status_change";

  if (fileStatusValue === "error") {
    type = "error";
    message = "encountered an error";
  } else if (status === "waiting") {
    type = "waiting";
    message = "requires input";
  } else if (status === "working" && prevStatus === "idle") {
    type = "started";
    message = "started working";
  } else if (status === "idle" && prevStatus === "working") {
    type = "completed";
    message = "finished task";
  } else {
    return null;
  }

  return {
    id: `${session.sessionId}-${Date.now()}`,
    timestamp: new Date(),
    sessionId: session.sessionId,
    sessionName: getAgentName(session),
    type,
    message,
  };
}

export function FleetCommand({ sessions }: FleetCommandProps) {
  // View mode: ops (table center) or focus (terminal center)
  const [viewMode, setViewMode] = useState<ViewMode>("ops");
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [filter, setFilter] = useState<StatusFilter>("all");
  const [events, setEvents] = useState<AgentEvent[]>([]);

  // Track previous session statuses for event detection
  const prevStatusesRef = useRef<Map<string, string>>(new Map());

  // Detect status changes and generate events
  useEffect(() => {
    const newEvents: AgentEvent[] = [];

    sessions.forEach((session) => {
      const { status } = getEffectiveStatus(session);
      const prevStatus = prevStatusesRef.current.get(session.sessionId) ?? null;

      const event = generateStatusEvent(session, prevStatus);
      if (event) {
        newEvents.push(event);
      }

      prevStatusesRef.current.set(session.sessionId, status);
    });

    if (newEvents.length > 0) {
      setEvents((prev) => [...newEvents, ...prev].slice(0, 50));
    }
  }, [sessions]);

  // Count sessions by status for StatusStrip
  const statusCounts = useMemo(
    () => countSessionsByStatus(sessions),
    [sessions]
  );

  // Count active sessions for CommandBar
  const workingCount = statusCounts.working;

  // Get selected session object
  const selectedSession = useMemo(
    () => sessions.find((s) => s.sessionId === selectedSessionId) ?? null,
    [sessions, selectedSessionId]
  );

  // Handle session selection (single click in ops mode)
  const handleSelectSession = useCallback((sessionId: string) => {
    setSelectedSessionId(sessionId);
  }, []);

  // Handle double-click to enter focus mode
  const handleDoubleClickSession = useCallback((sessionId: string) => {
    setSelectedSessionId(sessionId);
    setViewMode("focus");
  }, []);

  // Handle terminal dock close (in ops mode)
  const handleCloseDock = useCallback(() => {
    setSelectedSessionId(null);
  }, []);

  // Handle back to ops mode
  const handleBackToOps = useCallback(() => {
    setViewMode("ops");
  }, []);

  // Handle command sent
  const handleSendCommand = useCallback((text: string) => {
    console.log("[FleetCommand] Command sent:", text);
  }, []);

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      if (
        target.tagName === "INPUT" ||
        target.tagName === "TEXTAREA" ||
        target.isContentEditable
      ) {
        return;
      }

      const currentIndex = sessions.findIndex(
        (s) => s.sessionId === selectedSessionId
      );

      switch (e.key) {
        case "ArrowUp": {
          e.preventDefault();
          if (sessions.length === 0) return;
          if (currentIndex <= 0) {
            setSelectedSessionId(sessions[sessions.length - 1].sessionId);
          } else {
            setSelectedSessionId(sessions[currentIndex - 1].sessionId);
          }
          break;
        }
        case "ArrowDown": {
          e.preventDefault();
          if (sessions.length === 0) return;
          if (currentIndex < 0 || currentIndex >= sessions.length - 1) {
            setSelectedSessionId(sessions[0].sessionId);
          } else {
            setSelectedSessionId(sessions[currentIndex + 1].sessionId);
          }
          break;
        }
        case "Enter": {
          if (selectedSessionId && viewMode === "ops") {
            e.preventDefault();
            setViewMode("focus");
          }
          break;
        }
        case "Escape": {
          e.preventDefault();
          if (viewMode === "focus") {
            setViewMode("ops");
          } else {
            setSelectedSessionId(null);
          }
          break;
        }
        // Filter shortcuts
        case "a":
        case "A":
          if (!e.metaKey && !e.ctrlKey) {
            e.preventDefault();
            setFilter("all");
          }
          break;
        case "w":
        case "W":
          if (!e.metaKey && !e.ctrlKey) {
            e.preventDefault();
            setFilter("working");
          }
          break;
        case "i":
        case "I":
          if (!e.metaKey && !e.ctrlKey) {
            e.preventDefault();
            setFilter("waiting");
          }
          break;
        case "e":
        case "E":
          if (!e.metaKey && !e.ctrlKey) {
            e.preventDefault();
            setFilter("error");
          }
          break;
        case "s":
        case "S":
          if (!e.metaKey && !e.ctrlKey) {
            e.preventDefault();
            setFilter("stale");
          }
          break;
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [sessions, selectedSessionId, viewMode]);

  const layoutClass = viewMode === "ops" ? "fleet-command--ops" : "fleet-command--focus";

  return (
    <div className={`fleet-command ${layoutClass}`}>
      {/* Header */}
      <CommandBar
        sessionCount={sessions.length}
        workingCount={workingCount}
        viewMode={viewMode}
        onViewModeChange={setViewMode}
        selectedSession={viewMode === "focus" ? selectedSession : null}
        onBackToOps={handleBackToOps}
      />

      {/* StatusStrip (ops mode only) */}
      {viewMode === "ops" && (
        <div className="fleet-filters">
          <StatusStrip
            counts={statusCounts}
            activeFilter={filter}
            onFilterChange={setFilter}
          />
        </div>
      )}

      {/* Left: Roster */}
      <Roster
        sessions={sessions}
        selectedSessionId={selectedSessionId}
        onSelectSession={handleSelectSession}
        searchQuery={searchQuery}
        onSearchChange={setSearchQuery}
        compact={viewMode === "focus"}
      />

      {/* Center: DataTable (ops) or Viewport (focus) */}
      {viewMode === "ops" ? (
        <div className="fleet-table">
          <DataTable
            columns={columns}
            data={sessions}
            selectedId={selectedSessionId}
            onSelect={handleSelectSession}
            filter={filter}
          />
        </div>
      ) : (
        <Viewport session={selectedSession} onSendCommand={handleSendCommand} />
      )}

      {/* Right: Tactical Intel */}
      <TacticalIntel session={selectedSession} />

      {/* Bottom: TerminalDock (ops mode only) */}
      {viewMode === "ops" && (
        <div className="fleet-dock">
          {selectedSession ? (
            <TerminalDock session={selectedSession} onClose={handleCloseDock} />
          ) : (
            <div className="fleet-dock__empty">
              <span>Select a session to attach terminal</span>
            </div>
          )}
        </div>
      )}

      {/* Footer: Event Ticker */}
      <EventTicker events={events} />
    </div>
  );
}
