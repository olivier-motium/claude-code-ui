/**
 * Dialog for sending text to a linked terminal.
 */

import { useState } from "react";
import { Dialog, Button, TextArea, Checkbox, Flex, Text } from "@radix-ui/themes";
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
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Content maxWidth="450px">
        <Dialog.Title>Send to Terminal</Dialog.Title>
        <Dialog.Description size="2" mb="4">
          Text will be sent to the linked kitty terminal.
        </Dialog.Description>

        <Flex direction="column" gap="3">
          <TextArea
            placeholder="Enter text to send..."
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={4}
          />

          <Flex align="center" gap="2">
            <Checkbox
              checked={submit}
              onCheckedChange={(checked) => setSubmit(Boolean(checked))}
            />
            <Text size="2">Press Enter after sending</Text>
          </Flex>
        </Flex>

        <Flex gap="3" mt="4" justify="end">
          <Dialog.Close>
            <Button variant="soft" color="gray">
              Cancel
            </Button>
          </Dialog.Close>
          <Button onClick={handleSend} disabled={loading || !text.trim()}>
            {loading ? "Sending..." : "Send"}
          </Button>
        </Flex>
      </Dialog.Content>
    </Dialog.Root>
  );
}
