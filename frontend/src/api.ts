import type { ChatRequest, ChatResponse, ChatStreamEvent } from "./types";

async function parseJsonResponse<T>(response: Response): Promise<T> {
  const text = await response.text();
  const data = text ? JSON.parse(text) : {};
  if (!response.ok) {
    const message =
      typeof data.detail === "string"
        ? data.detail
        : `Request failed with status ${response.status}`;
    throw new Error(message);
  }
  return data as T;
}

export async function checkHealth(): Promise<void> {
  const response = await fetch("/api/health");
  await parseJsonResponse<{ status: string }>(response);
}

export async function sendChat(payload: ChatRequest): Promise<ChatResponse> {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return parseJsonResponse<ChatResponse>(response);
}

export async function streamChat(
  payload: ChatRequest,
  onEvent: (event: ChatStreamEvent) => void,
): Promise<void> {
  const response = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok || !response.body) {
    await parseJsonResponse(response);
    throw new Error(`Stream request failed with status ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop() ?? "";
    for (const eventText of events) {
      const event = parseSseEvent(eventText);
      if (event) {
        onEvent(event);
      }
    }
  }

  buffer += decoder.decode();
  const finalEvent = parseSseEvent(buffer);
  if (finalEvent) {
    onEvent(finalEvent);
  }
}

export async function resetDemoData(): Promise<void> {
  const response = await fetch("/api/demo/reset", {
    method: "POST",
  });
  await parseJsonResponse<{ status: string }>(response);
}

function parseSseEvent(text: string): ChatStreamEvent | null {
  const lines = text.split("\n");
  const eventLine = lines.find((line) => line.startsWith("event:"));
  const dataLines = lines
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice("data:".length).trimStart());

  if (!eventLine || dataLines.length === 0) {
    return null;
  }

  const event = eventLine.slice("event:".length).trim();
  const data = JSON.parse(dataLines.join("\n")) as unknown;

  if (event === "started" || event === "node" || event === "final" || event === "error") {
    return { event, data } as ChatStreamEvent;
  }

  return null;
}
