/**
 * BranchCell - Git branch display
 */

interface BranchCellProps {
  branch: string | null
}

export function BranchCell({ branch }: BranchCellProps) {
  if (!branch) {
    return <span className="text-sm text-muted-foreground">-</span>
  }

  const displayBranch = branch.length > 15 ? branch.slice(0, 12) + "..." : branch

  return (
    <code className="text-xs bg-muted px-1.5 py-0.5 rounded text-muted-foreground">
      {displayBranch}
    </code>
  )
}
