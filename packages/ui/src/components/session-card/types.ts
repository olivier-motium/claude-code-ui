/**
 * Type definitions for SessionCard components
 */

import type { Session } from "@claude-code-ui/daemon/schema";

export interface SessionCardProps {
  session: Session;
}

export interface SessionActionsProps {
  session: Session;
  onSendText?: () => void;
}

export interface SessionCardContentProps {
  session: Session;
  onSendText?: () => void;
}
