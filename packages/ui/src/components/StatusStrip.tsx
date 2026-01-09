/**
 * StatusStrip - Compact status overview with clickable filters
 *
 * Shows counts of sessions by status (Working, Waiting, Idle, etc.)
 * Clicking a badge filters the DataTable to that status.
 */

import { cn } from "@/lib/utils";
import type { StatusCounts, StatusFilter } from "./ops-table/types";

interface StatusStripProps {
  counts: StatusCounts;
  activeFilter: StatusFilter;
  onFilterChange: (filter: StatusFilter) => void;
}

const STATUS_STYLES: Record<string, { bg: string; bgActive: string; text: string; border: string }> = {
  all: {
    bg: "bg-blue-500/10",
    bgActive: "bg-blue-500",
    text: "text-blue-400",
    border: "border-blue-500/20",
  },
  working: {
    bg: "bg-status-working/10",
    bgActive: "bg-status-working",
    text: "text-status-working",
    border: "border-status-working/20",
  },
  waiting: {
    bg: "bg-status-waiting/10",
    bgActive: "bg-status-waiting",
    text: "text-status-waiting",
    border: "border-status-waiting/20",
  },
  idle: {
    bg: "bg-status-idle/10",
    bgActive: "bg-status-idle",
    text: "text-status-idle",
    border: "border-status-idle/20",
  },
  error: {
    bg: "bg-status-error/10",
    bgActive: "bg-status-error",
    text: "text-status-error",
    border: "border-status-error/20",
  },
  stale: {
    bg: "bg-orange-500/10",
    bgActive: "bg-orange-500",
    text: "text-orange-400",
    border: "border-orange-500/20",
  },
};

export function StatusStrip({ counts, activeFilter, onFilterChange }: StatusStripProps) {
  const badges: Array<{
    filter: StatusFilter;
    label: string;
    count: number;
  }> = [
    { filter: "all", label: "All", count: counts.all },
    { filter: "working", label: "Working", count: counts.working },
    { filter: "waiting", label: "Needs Input", count: counts.waiting },
    { filter: "idle", label: "Idle", count: counts.idle },
  ];

  // Only show error/stale badges if there are any
  if (counts.error > 0) {
    badges.push({ filter: "error", label: "Errors", count: counts.error });
  }
  if (counts.stale > 0) {
    badges.push({ filter: "stale", label: "Stale", count: counts.stale });
  }

  return (
    <div className="flex items-center gap-2 flex-wrap">
      {badges.map(({ filter, label, count }) => {
        const isActive = activeFilter === filter;
        const needsAttention = filter === "waiting" && count > 0 && !isActive;
        const styles = STATUS_STYLES[filter];

        return (
          <button
            key={filter}
            type="button"
            onClick={() => onFilterChange(filter)}
            className={cn(
              "inline-flex items-center gap-1 rounded-full px-2.5 py-1 text-xs font-medium border transition-all cursor-pointer",
              isActive
                ? [styles.bgActive, "text-white border-transparent"]
                : [styles.bg, styles.text, styles.border, "hover:opacity-80"],
              needsAttention && "animate-pulse"
            )}
          >
            {label}: {count}
          </button>
        );
      })}
    </div>
  );
}
