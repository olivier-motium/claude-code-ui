/**
 * DataTable - TanStack Table implementation for sessions
 *
 * Features:
 * - Column sorting (click headers)
 * - Row selection (single click)
 * - Status-based row highlighting
 * - Integrates with StatusStrip filter
 */

import { useMemo, useState } from "react"
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  flexRender,
  type ColumnDef,
  type SortingState,
} from "@tanstack/react-table"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { ScrollArea } from "@/components/ui/scroll-area"
import type { Session } from "@/types/schema"
import type { StatusFilter } from "@/components/ops-table/types"
import { filterSessions } from "@/components/ops-table/utils"
import { getEffectiveStatus } from "@/lib/sessionStatus"
import { cn } from "@/lib/utils"

interface DataTableProps {
  columns: ColumnDef<Session>[]
  data: Session[]
  selectedId: string | null
  onSelect: (sessionId: string) => void
  filter: StatusFilter
}

// Get row styling based on session status
function getStatusRowClass(session: Session): string {
  const { status, fileStatusValue } = getEffectiveStatus(session)
  const classes: string[] = []

  if (status === "working") {
    classes.push("border-l-4 border-l-status-working bg-status-working/5")
  }
  if (status === "waiting" && session.hasPendingToolUse) {
    classes.push("border-l-4 border-l-status-waiting bg-status-waiting/5")
  }
  if (fileStatusValue === "error") {
    classes.push("border-l-4 border-l-status-error bg-status-error/5")
  }
  if (fileStatusValue === "blocked") {
    classes.push("border-l-4 border-l-status-waiting bg-status-waiting/5")
  }

  return classes.join(" ")
}

export function DataTable({
  columns,
  data,
  selectedId,
  onSelect,
  filter,
}: DataTableProps) {
  // Apply status filter
  const filteredData = useMemo(
    () => filterSessions(data, filter),
    [data, filter]
  )

  // Default sorting: status priority, then most recent
  const [sorting, setSorting] = useState<SortingState>([
    { id: "status", desc: false },
    { id: "lastActivityAt", desc: false },
  ])

  const table = useReactTable({
    data: filteredData,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getRowId: (row) => row.sessionId,
  })

  if (filteredData.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
        <p className="text-sm">
          {filter === "all" ? "No sessions found" : `No ${filter} sessions`}
        </p>
      </div>
    )
  }

  return (
    <div className="rounded-lg border border-border bg-card/50 backdrop-blur-sm">
      <Table>
        <TableHeader className="bg-muted/50">
          {table.getHeaderGroups().map((headerGroup) => (
            <TableRow key={headerGroup.id} className="border-b border-border hover:bg-transparent">
              {headerGroup.headers.map((header) => (
                <TableHead
                  key={header.id}
                  style={{ width: header.getSize() !== 150 ? header.getSize() : undefined }}
                  className="text-xs font-medium text-muted-foreground"
                >
                  {header.isPlaceholder
                    ? null
                    : flexRender(header.column.columnDef.header, header.getContext())}
                </TableHead>
              ))}
            </TableRow>
          ))}
        </TableHeader>
        <TableBody>
          <ScrollArea className="h-[calc(100vh-400px)]">
            {table.getRowModel().rows.map((row) => {
              const isSelected = selectedId === row.id
              return (
                <TableRow
                  key={row.id}
                  onClick={() => onSelect(row.original.sessionId)}
                  className={cn(
                    "cursor-pointer transition-colors border-b border-border/50",
                    "hover:bg-accent/50",
                    isSelected && "bg-accent border-l-4 border-l-primary",
                    getStatusRowClass(row.original)
                  )}
                  data-state={isSelected ? "selected" : undefined}
                >
                  {row.getVisibleCells().map((cell) => (
                    <TableCell
                      key={cell.id}
                      style={{ width: cell.column.getSize() !== 150 ? cell.column.getSize() : undefined }}
                      className="py-2"
                    >
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </TableCell>
                  ))}
                </TableRow>
              )
            })}
          </ScrollArea>
        </TableBody>
      </Table>
    </div>
  )
}
