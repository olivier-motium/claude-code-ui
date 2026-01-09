/**
 * GoalCell - Goal/prompt with status badges
 */

import { Badge } from "@/components/ui/badge"
import { getEffectiveStatus } from "@/lib/sessionStatus"
import type { Session } from "@/types/schema"

interface GoalCellProps {
  session: Session
}

export function GoalCell({ session }: GoalCellProps) {
  const { fileStatusValue } = getEffectiveStatus(session)
  const text = session.goal || session.originalPrompt
  const displayText = text.length > 60 ? text.slice(0, 57) + "..." : text

  return (
    <div className="flex items-center gap-2 min-w-0">
      <span className="truncate text-sm">{displayText}</span>
      {fileStatusValue === "completed" && (
        <Badge variant="outline" className="bg-blue-500/10 text-blue-400 border-blue-500/20 shrink-0">
          Done
        </Badge>
      )}
      {fileStatusValue === "error" && (
        <Badge variant="outline" className="bg-red-500/10 text-red-400 border-red-500/20 shrink-0">
          Error
        </Badge>
      )}
      {fileStatusValue === "blocked" && (
        <Badge variant="outline" className="bg-orange-500/10 text-orange-400 border-orange-500/20 shrink-0">
          Blocked
        </Badge>
      )}
    </div>
  )
}
