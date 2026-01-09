/**
 * Dialog for sending text to a linked terminal.
 */

import { useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Checkbox } from "@/components/ui/checkbox";
import * as api from "../lib/api";

interface SendTextDialogProps {
  sessionId: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function SendTextDialog({ sessionId, open, onOpenChange }: SendTextDialogProps) {
  const [text, setText] = useState("");
  const [submit, setSubmit] = useState(false);
  const [loading, setLoading] = useState(false);

  const handleSend = async () => {
    if (!text.trim()) return;

    setLoading(true);
    try {
      await api.sendText(sessionId, text, submit);
      setText("");
      onOpenChange(false);
    } catch (e) {
      console.error("Send failed:", e);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && e.metaKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[450px]">
        <DialogHeader>
          <DialogTitle>Send to Terminal</DialogTitle>
          <DialogDescription>
            Text will be sent to the linked kitty terminal.
          </DialogDescription>
        </DialogHeader>

        <div className="flex flex-col gap-3">
          <Textarea
            placeholder="Enter text to send..."
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={4}
            className="resize-none"
          />

          <div className="flex items-center gap-2">
            <Checkbox
              id="submit-checkbox"
              checked={submit}
              onCheckedChange={(checked) => setSubmit(Boolean(checked))}
            />
            <label
              htmlFor="submit-checkbox"
              className="text-sm text-muted-foreground cursor-pointer"
            >
              Press Enter after sending
            </label>
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={handleSend} disabled={loading || !text.trim()}>
            {loading ? "Sending..." : "Send"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
