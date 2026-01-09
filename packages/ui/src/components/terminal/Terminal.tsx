/**
 * Terminal component - xterm.js wrapper with WebSocket connection.
 *
 * Connects to PTY via WebSocket for real-time terminal I/O.
 */

import { useEffect, useRef, useCallback } from "react";
import { Terminal as XTerminal } from "@xterm/xterm";
import { FitAddon } from "@xterm/addon-fit";
import { WebLinksAddon } from "@xterm/addon-web-links";
import { radixDarkTheme, terminalStyles } from "./theme";

export interface TerminalProps {
  /** WebSocket URL for PTY connection */
  wsUrl: string;
  /** Authentication token for WebSocket */
  wsToken: string;
  /** Called when WebSocket connects successfully */
  onConnect?: () => void;
  /** Called when WebSocket disconnects */
  onDisconnect?: () => void;
  /** Called when terminal is resized */
  onResize?: (cols: number, rows: number) => void;
  /** Called when connection error occurs */
  onError?: (error: string) => void;
}

interface WsMessage {
  type: "data" | "input" | "resize" | "ping" | "pong";
  payload?: string;
  cols?: number;
  rows?: number;
}

/**
 * Embedded terminal component using xterm.js.
 */
export function Terminal({
  wsUrl,
  wsToken,
  onConnect,
  onDisconnect,
  onResize,
  onError,
}: TerminalProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const terminalRef = useRef<XTerminal | null>(null);
  const fitAddonRef = useRef<FitAddon | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  // Initialize xterm.js
  useEffect(() => {
    if (!containerRef.current) return;

    const terminal = new XTerminal({
      theme: radixDarkTheme,
      fontFamily: terminalStyles.fontFamily,
      fontSize: terminalStyles.fontSize,
      lineHeight: terminalStyles.lineHeight,
      cursorBlink: true,
      cursorStyle: "block",
      scrollback: 10000,
      allowProposedApi: true,
    });

    const fitAddon = new FitAddon();
    const webLinksAddon = new WebLinksAddon();

    terminal.loadAddon(fitAddon);
    terminal.loadAddon(webLinksAddon);

    terminal.open(containerRef.current);
    fitAddon.fit();

    terminalRef.current = terminal;
    fitAddonRef.current = fitAddon;

    return () => {
      terminal.dispose();
      terminalRef.current = null;
      fitAddonRef.current = null;
    };
  }, []);

  // Connect WebSocket
  useEffect(() => {
    const terminal = terminalRef.current;
    if (!terminal) return;

    const wsUrlWithToken = `${wsUrl}?token=${wsToken}`;
    const ws = new WebSocket(wsUrlWithToken);

    ws.onopen = () => {
      console.log("[Terminal] WebSocket connected");
      onConnect?.();

      // Send initial resize
      if (fitAddonRef.current) {
        const { cols, rows } = terminal;
        const msg: WsMessage = { type: "resize", cols, rows };
        ws.send(JSON.stringify(msg));
      }
    };

    ws.onmessage = (event) => {
      try {
        const msg: WsMessage = JSON.parse(event.data);
        if (msg.type === "data" && msg.payload) {
          terminal.write(msg.payload);
        }
      } catch {
        // Binary data or invalid JSON - write directly
        terminal.write(event.data);
      }
    };

    ws.onclose = (event) => {
      console.log("[Terminal] WebSocket closed:", event.code, event.reason);
      onDisconnect?.();
    };

    ws.onerror = () => {
      console.error("[Terminal] WebSocket error");
      onError?.("WebSocket connection failed");
    };

    wsRef.current = ws;

    // Send input to PTY
    const inputDisposable = terminal.onData((data) => {
      if (ws.readyState === WebSocket.OPEN) {
        const msg: WsMessage = { type: "input", payload: data };
        ws.send(JSON.stringify(msg));
      }
    });

    return () => {
      inputDisposable.dispose();
      if (ws.readyState === WebSocket.OPEN) {
        ws.close(1000, "Component unmounting");
      }
      wsRef.current = null;
    };
  }, [wsUrl, wsToken, onConnect, onDisconnect, onError]);

  // Handle resize
  const handleResize = useCallback(() => {
    const terminal = terminalRef.current;
    const fitAddon = fitAddonRef.current;
    const ws = wsRef.current;

    if (!terminal || !fitAddon) return;

    fitAddon.fit();
    const { cols, rows } = terminal;

    // Notify server
    if (ws?.readyState === WebSocket.OPEN) {
      const msg: WsMessage = { type: "resize", cols, rows };
      ws.send(JSON.stringify(msg));
    }

    onResize?.(cols, rows);
  }, [onResize]);

  // Resize on window resize
  useEffect(() => {
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [handleResize]);

  // Resize when container size changes (for layout changes)
  useEffect(() => {
    if (!containerRef.current) return;

    const resizeObserver = new ResizeObserver(() => {
      handleResize();
    });

    resizeObserver.observe(containerRef.current);
    return () => resizeObserver.disconnect();
  }, [handleResize]);

  return (
    <div
      ref={containerRef}
      className="w-full h-full p-2"
      style={{ backgroundColor: radixDarkTheme.background }}
    />
  );
}
