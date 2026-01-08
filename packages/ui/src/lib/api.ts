/**
 * API client for kitty terminal remote control endpoints.
 */

const API_BASE = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:4451/api";

/** Detailed kitty status for diagnostics */
interface KittyStatusDetails {
  installed: boolean;
  running: boolean;
  socketExists: boolean;
  socketReachable: boolean;
  configExists: boolean;
}

/** Health check response with detailed status */
export interface KittyHealthResponse {
  available: boolean;
  details: KittyStatusDetails;
}

/** Result of kitty setup operation */
export interface KittySetupResponse {
  success: boolean;
  status: string;
  message: string;
  actions: string[];
}

interface ApiResponse {
  success?: boolean;
  error?: string;
  windowId?: number;
  created?: boolean;
  stale?: boolean;
}

/**
 * Make an API call with proper error handling.
 */
async function apiCall<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ error: "Unknown error" }));
    throw new Error(error.error || `HTTP ${response.status}`);
  }

  return response.json();
}

/**
 * Check if kitty terminal is available with detailed status.
 */
export async function getKittyHealth(): Promise<KittyHealthResponse> {
  return apiCall("/kitty/health");
}

/**
 * Trigger kitty remote control setup.
 */
export async function runKittySetup(): Promise<KittySetupResponse> {
  return apiCall("/kitty/setup", { method: "POST" });
}

/**
 * Focus an existing linked terminal for a session.
 */
export async function focusSession(sessionId: string): Promise<ApiResponse> {
  return apiCall(`/sessions/${sessionId}/focus`, { method: "POST" });
}

/**
 * Open or focus a terminal for a session.
 * Creates a new tab if no terminal is linked.
 */
export async function openSession(sessionId: string): Promise<ApiResponse> {
  return apiCall(`/sessions/${sessionId}/open`, { method: "POST" });
}

/**
 * Link an existing terminal to a session via interactive selection.
 */
export async function linkTerminal(sessionId: string): Promise<ApiResponse> {
  return apiCall(`/sessions/${sessionId}/link-terminal`, { method: "POST" });
}

/**
 * Unlink a terminal from a session.
 */
export async function unlinkTerminal(sessionId: string): Promise<ApiResponse> {
  return apiCall(`/sessions/${sessionId}/link-terminal`, { method: "DELETE" });
}

/**
 * Send text to a session's linked terminal.
 */
export async function sendText(
  sessionId: string,
  text: string,
  submit: boolean
): Promise<ApiResponse> {
  return apiCall(`/sessions/${sessionId}/send-text`, {
    method: "POST",
    body: JSON.stringify({ text, submit }),
  });
}
