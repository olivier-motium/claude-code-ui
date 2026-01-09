/**
 * ToolCell - Pending tool display with tooltip
 */

import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import type { PendingTool } from "@claude-code-ui/daemon/schema"

interface ToolCellProps {
  pendingTool: PendingTool | null
}

export function ToolCell({ pendingTool }: ToolCellProps) {
  if (!pendingTool) {
    return <span className="text-sm text-muted-foreground">-</span>
  }

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <code className="text-xs bg-orange-500/10 text-orange-400 px-1.5 py-0.5 rounded truncate block max-w-full">
            {pendingTool.tool}
          </code>
        </TooltipTrigger>
        <TooltipContent side="top">
          <p className="font-mono text-xs">{pendingTool.target}</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  )
}
