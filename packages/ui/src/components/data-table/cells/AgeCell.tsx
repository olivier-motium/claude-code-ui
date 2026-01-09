/**
 * AgeCell - Activity time ago display
 */

interface AgeCellProps {
  timestamp: string
}

function formatTimeAgo(isoString: string): string {
  const now = Date.now()
  const then = new Date(isoString).getTime()
  const diff = now - then

  const seconds = Math.floor(diff / 1000)
  const minutes = Math.floor(seconds / 60)
  const hours = Math.floor(minutes / 60)
  const days = Math.floor(hours / 24)

  if (days > 0) return `${days}d`
  if (hours > 0) return `${hours}h`
  if (minutes > 0) return `${minutes}m`
  return `${seconds}s`
}

export function AgeCell({ timestamp }: AgeCellProps) {
  return (
    <span className="text-xs text-muted-foreground tabular-nums">
      {formatTimeAgo(timestamp)}
    </span>
  )
}
