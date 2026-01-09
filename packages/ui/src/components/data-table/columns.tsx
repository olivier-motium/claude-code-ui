/**
 * TanStack Table column definitions for sessions DataTable
 */

import { ColumnDef } from "@tanstack/react-table"
import { ArrowUpDown } from "lucide-react"
import type { Session, SessionStatus } from "@/types/schema"
import { getEffectiveStatus } from "@/lib/sessionStatus"
import { StatusCell } from "./cells/StatusCell"
import { GoalCell } from "./cells/GoalCell"
import { BranchCell } from "./cells/BranchCell"
import { ToolCell } from "./cells/ToolCell"
import { AgeCell } from "./cells/AgeCell"
import { RepoCell } from "./cells/RepoCell"
import { ActionsCell } from "./cells/ActionsCell"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

// Sortable header component
function SortableHeader({
  column,
  children,
}: {
  column: { getIsSorted: () => false | "asc" | "desc"; toggleSorting: (desc?: boolean) => void }
  children: React.ReactNode
}) {
  return (
    <Button
      variant="ghost"
      size="sm"
      className="-ml-3 h-8 data-[state=open]:bg-accent"
      onClick={() => column.toggleSorting(column.getIsSorted() === "asc")}
    >
      {children}
      <ArrowUpDown
        className={cn(
          "ml-2 h-4 w-4",
          column.getIsSorted() && "text-foreground"
        )}
      />
    </Button>
  )
}

// Status priority for sorting
const STATUS_PRIORITY: Record<SessionStatus, number> = {
  working: 0,
  waiting: 1,
  idle: 2,
}

export const columns: ColumnDef<Session>[] = [
  // Status indicator - 40px
  {
    id: "status",
    accessorFn: (row) => getEffectiveStatus(row).status,
    header: () => <span className="text-muted-foreground">●</span>,
    cell: ({ row }) => <StatusCell session={row.original} />,
    size: 40,
    enableSorting: true,
    sortingFn: (rowA, rowB) => {
      const statusA = getEffectiveStatus(rowA.original).status
      const statusB = getEffectiveStatus(rowB.original).status
      return STATUS_PRIORITY[statusA] - STATUS_PRIORITY[statusB]
    },
  },

  // Goal/prompt - flex grow
  {
    accessorKey: "goal",
    header: ({ column }) => <SortableHeader column={column}>Goal</SortableHeader>,
    cell: ({ row }) => <GoalCell session={row.original} />,
    enableSorting: true,
  },

  // Git branch - 120px
  {
    accessorKey: "gitBranch",
    header: ({ column }) => <SortableHeader column={column}>Branch</SortableHeader>,
    cell: ({ row }) => <BranchCell branch={row.getValue("gitBranch")} />,
    size: 120,
    enableSorting: true,
  },

  // Pending tool - 80px
  {
    accessorKey: "pendingTool",
    header: "Tool",
    cell: ({ row }) => <ToolCell pendingTool={row.getValue("pendingTool")} />,
    size: 80,
    enableSorting: false,
  },

  // Activity age - 60px
  {
    accessorKey: "lastActivityAt",
    header: ({ column }) => <SortableHeader column={column}>Age</SortableHeader>,
    cell: ({ row }) => <AgeCell timestamp={row.getValue("lastActivityAt")} />,
    size: 60,
    enableSorting: true,
    sortingFn: (rowA, rowB) => {
      const timeA = new Date(rowA.original.lastActivityAt).getTime()
      const timeB = new Date(rowB.original.lastActivityAt).getTime()
      return timeB - timeA // Most recent first
    },
  },

  // Repository - 100px
  {
    accessorKey: "gitRepoId",
    header: "Repo",
    cell: ({ row }) => <RepoCell repoId={row.getValue("gitRepoId")} />,
    size: 100,
    enableSorting: true,
  },

  // Actions menu - 40px
  {
    id: "actions",
    cell: ({ row }) => <ActionsCell session={row.original} />,
    size: 40,
    enableSorting: false,
  },
]
