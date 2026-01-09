/**
 * CommandBar - Top header for Fleet Command
 *
 * In Ops mode: Logo, mode toggle, agent counts
 * In Focus mode: Back button, session info
 */

import { Cpu, Layers, ArrowLeft, GitBranch } from "lucide-react";
import { getEffectiveStatus } from "../../lib/sessionStatus";
import type { CommandBarProps } from "./types";

export function CommandBar({
  sessionCount,
  workingCount,
  viewMode,
  onViewModeChange,
  selectedSession,
  onBackToOps,
}: CommandBarProps) {
  // Focus mode: show session context with back button
  if (viewMode === "focus" && selectedSession) {
    const { status } = getEffectiveStatus(selectedSession);
    const statusLabel = status === "working" ? "Working" : status === "waiting" ? "Waiting" : "Idle";
    const statusClass = `fleet-command-bar__status-badge--${status}`;

    return (
      <header className="fleet-command-bar fleet-command-bar--focus">
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <button
            className="fleet-command-bar__back"
            onClick={onBackToOps}
            title="Back to Ops (Esc)"
          >
            <ArrowLeft size={16} />
            <span>Back to Ops</span>
          </button>

          <div className="fleet-command-bar__divider" />

          <div className="fleet-command-bar__session-info">
            <span className="fleet-command-bar__session-name">
              {selectedSession.gitBranch || selectedSession.sessionId.slice(-8)}
            </span>
            {selectedSession.gitBranch && (
              <span className="fleet-command-bar__branch">
                <GitBranch size={12} />
                {selectedSession.gitBranch}
              </span>
            )}
            <span className={`fleet-command-bar__status-badge ${statusClass}`}>
              {statusLabel}
            </span>
          </div>
        </div>

        <div className="fleet-command-bar__meta">
          <div className="fleet-command-bar__version">v2.0.0</div>
        </div>
      </header>
    );
  }

  // Ops mode: show logo, mode toggle, agent counts
  return (
    <header className="fleet-command-bar">
      <div style={{ display: "flex", alignItems: "center" }}>
        <div className="fleet-command-bar__logo">
          <Cpu size={18} />
          <span>NANO // BANANA // PRO</span>
        </div>

        <div className="fleet-command-bar__divider" />

        {/* Mode toggle */}
        <div className="fleet-command-bar__mode-toggle">
          <button
            className={`fleet-command-bar__mode-btn ${viewMode === "ops" ? "fleet-command-bar__mode-btn--active" : ""}`}
            onClick={() => onViewModeChange("ops")}
          >
            Ops
          </button>
          <button
            className={`fleet-command-bar__mode-btn ${viewMode === "focus" ? "fleet-command-bar__mode-btn--active" : ""}`}
            onClick={() => onViewModeChange("focus")}
          >
            Focus
          </button>
        </div>

        <div className="fleet-command-bar__divider" />

        <div className="fleet-command-bar__status">
          <span className="fleet-command-bar__online">
            <div className="fleet-command-bar__online-dot" />
            ONLINE
          </span>
          <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <Layers size={12} />
            AGENTS: {workingCount}/{sessionCount} Active
          </span>
        </div>
      </div>

      <div className="fleet-command-bar__meta">
        <div className="fleet-command-bar__version">v2.0.0</div>
      </div>
    </header>
  );
}
