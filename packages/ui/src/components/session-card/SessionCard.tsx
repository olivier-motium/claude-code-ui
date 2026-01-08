/**
 * Main SessionCard component - orchestrates card, hover, and dialog
 */

import { useState } from "react";
import { HoverCard } from "@radix-ui/themes";
import { SendTextDialog } from "../SendTextDialog";
import { SessionCardContent } from "./SessionCardContent";
import { SessionCardHoverContent } from "./SessionCardHoverContent";
import type { SessionCardProps } from "./types";

export function SessionCard({ session }: SessionCardProps) {
  const [sendTextOpen, setSendTextOpen] = useState(false);

  return (
    <>
      <HoverCard.Root openDelay={300}>
        <HoverCard.Trigger>
          <SessionCardContent
            session={session}
            onSendText={() => setSendTextOpen(true)}
          />
        </HoverCard.Trigger>
        <HoverCard.Content size="3" style={{ minWidth: "600px", minHeight: "400px" }}>
          <SessionCardHoverContent session={session} />
        </HoverCard.Content>
      </HoverCard.Root>
      <SendTextDialog
        sessionId={session.sessionId}
        open={sendTextOpen}
        onOpenChange={setSendTextOpen}
      />
    </>
  );
}
