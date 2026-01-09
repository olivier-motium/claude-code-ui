/**
 * RepoCell - Repository name display
 */

interface RepoCellProps {
  repoId: string | null
}

export function RepoCell({ repoId }: RepoCellProps) {
  if (!repoId) {
    return <span className="text-sm text-muted-foreground">-</span>
  }

  // Extract repo name from "owner/repo" format
  const repoName = repoId.split("/")[1] || repoId
  const displayName = repoName.length > 12 ? repoName.slice(0, 10) + "..." : repoName

  return (
    <span className="text-xs text-muted-foreground truncate">
      {displayName}
    </span>
  )
}
