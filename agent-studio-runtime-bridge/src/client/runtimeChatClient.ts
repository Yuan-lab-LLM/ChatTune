export type RuntimeChatInput = {
  userId: string;
  sessionId: string;
  message: string;
};

export type RuntimeSseEvent = {
  sequence_number?: number;
  object?: string;
  status?: string;
  text?: string;
  message?: string;
  error?: unknown;
  body?: string;
  detail?: unknown;
  metadata?: {
    protocol?: unknown;
  };
  [key: string]: unknown;
};

export type RuntimeChatCallbacks = {
  onEvent?: (event: RuntimeSseEvent) => void;
  onDelta?: (text: string, event: RuntimeSseEvent) => void;
  onCompleted?: (finalText: string, protocol: unknown) => void;
  onError?: (message: string, event: RuntimeSseEvent) => void;
};

function parseSseEvent(rawEvent: string): RuntimeSseEvent[] {
  return rawEvent
    .split('\n')
    .filter((line) => line.startsWith('data:'))
    .map((line) => line.slice(5).trim())
    .filter(Boolean)
    .map((payload) => JSON.parse(payload) as RuntimeSseEvent);
}

function extractRuntimeErrorMessage(event: RuntimeSseEvent) {
  const error =
    event.error && typeof event.error === 'object'
      ? (event.error as { message?: unknown; detail?: unknown })
      : null;
  const candidates = [
    event.message,
    typeof event.error === 'string' ? event.error : undefined,
    error?.message,
    error?.detail,
    event.body,
    typeof event.detail === 'string' ? event.detail : undefined,
  ];
  return (
    candidates.find(
      (candidate): candidate is string =>
        typeof candidate === 'string' && candidate.trim().length > 0,
    ) || 'Agent Runtime returned an error event'
  );
}

export async function sendRuntimeChatMessage(
  input: RuntimeChatInput,
  callbacks: RuntimeChatCallbacks = {},
  endpoint = '/api/agent/process',
) {
  const response = await fetch(endpoint, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
    },
    body: JSON.stringify(input),
  });

  if (!response.ok || !response.body) {
    throw new Error(`Agent request failed: ${response.status} ${await response.text()}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let finalText = '';
  let latestProtocol: unknown = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    while (buffer.includes('\n\n')) {
      const splitIndex = buffer.indexOf('\n\n');
      const rawEvent = buffer.slice(0, splitIndex);
      buffer = buffer.slice(splitIndex + 2);

      for (const event of parseSseEvent(rawEvent)) {
        callbacks.onEvent?.(event);

        if (event.object === 'content' && typeof event.text === 'string') {
          finalText += event.text;
          callbacks.onDelta?.(event.text, event);
        }

        if (event.object === 'message' && event.status === 'completed') {
          latestProtocol = event.metadata?.protocol || latestProtocol;
        }

        if (
          event.object === 'error' ||
          (event.object === 'response' &&
            ['failed', 'canceled', 'cancelled'].includes(String(event.status || '')))
        ) {
          const errorMessage = extractRuntimeErrorMessage(event);
          callbacks.onError?.(errorMessage, event);
          throw new Error(errorMessage);
        }
      }
    }
  }

  callbacks.onCompleted?.(finalText, latestProtocol);
  return {
    text: finalText,
    protocol: latestProtocol,
  };
}
