/**
 * Roster - Left sidebar showing all agents
 *
 * Zone A of the Fleet Command layout
 * Supports compact mode for focus view
 */

import { Search } from "lucide-react";
import { RosterItem } from "./RosterItem";
import type { RosterProps } from "./types";

export function Roster({
  sessions,
  selectedSessionId,
  onSelectSession,
  searchQuery,
  onSearchChange,
  compact = false,
}: RosterProps) {
  // Filter sessions by search query
  const filteredSessions = sessions.filter((session) => {
    if (!searchQuery) return true;
    const query = searchQuery.toLowerCase();
    return (
      session.gitBranch?.toLowerCase().includes(query) ||
      session.goal?.toLowerCase().includes(query) ||
      session.originalPrompt?.toLowerCase().includes(query) ||
      session.sessionId.toLowerCase().includes(query)
    );
  });

  const rosterClass = compact ? "fleet-roster fleet-roster--compact" : "fleet-roster";

  return (
    <aside className={rosterClass}>
      {/* Hide search in compact mode */}
      {!compact && (
        <div className="fleet-roster__search">
          <div className="fleet-roster__search-wrapper">
            <Search className="fleet-roster__search-icon" />
            <input
              type="text"
              className="fleet-roster__search-input"
              placeholder="Filter Units..."
              value={searchQuery}
              onChange={(e) => onSearchChange(e.target.value)}
            />
          </div>
        </div>
      )}

      <div className="fleet-roster__list">
        {filteredSessions.length === 0 ? (
          <div style={{ padding: compact ? 12 : 24, textAlign: "center", color: "var(--nb-text-muted)" }}>
            {searchQuery ? "No matching agents" : "No active agents"}
          </div>
        ) : (
          filteredSessions.map((session) => (
            <RosterItem
              key={session.sessionId}
              session={session}
              isSelected={session.sessionId === selectedSessionId}
              onSelect={() => onSelectSession(session.sessionId)}
            />
          ))
        )}
      </div>
    </aside>
  );
}
