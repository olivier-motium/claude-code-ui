/**
 * ActionsCell - Session actions dropdown
 *
 * Re-uses the existing SessionActions component
 */

import { useState } from "react"
import { SessionActions } from "@/components/session-card/SessionActions"
import { SendTextDialog } from "@/components/SendTextDialog"
import type { Session } from "@/types/schema"

interface ActionsCellProps {
  session: Session
}

export function ActionsCell({ session }: ActionsCellProps) {
  const [sendTextOpen, setSendTextOpen] = useState(false)

  return (
    <>
      <SessionActions
        session={session}
        onSendText={() => setSendTextOpen(true)}
      />
      <SendTextDialog
        sessionId={session.sessionId}
        open={sendTextOpen}
        onOpenChange={setSendTextOpen}
      />
    </>
  )
}
