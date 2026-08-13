import cors from "cors";
import express, { Request, Response } from "express";

type RuntimeContentBlock = {
  type: string;
  text?: string;
  [key: string]: unknown;
};

type RuntimeMessage = {
  role: "user" | "assistant" | "system";
  type?: "message";
  content: RuntimeContentBlock[];
};

type BridgeChatRequest = {
  userId?: string;
  user_id?: string;
  sessionId?: string;
  session_id?: string;
  message?: string;
  training_container?: string;
  evaluation_container?: string;
  grpo_container?: string;
  resource_group_id?: string;
  training_pool_id?: string;
  user_role?: string;
  ownerUserId?: string;
  owner_user_id?: string;
  ownerAliases?: string[];
  owner_aliases?: string[];
  contextUsername?: string;
  context_username?: string;
  input?: RuntimeMessage[];
};

const RUNTIME_CONTEXT_MARKER = "__medflow_runtime_context__";

type RuntimeEvent = {
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
};

type SelectedBackend = {
  baseUrl: string;
  routeKey: string;
  index: number;
};

const app = express();
const port = Number(process.env.PORT || 3100);
const agentApiToken = (process.env.MEDFLOW_AGENT_API_TOKEN || "").trim();
const agentApiProcessPath = `/${(
  process.env.AGENT_API_PROCESS_PATH || "runtime-process"
)
  .trim()
  .replace(/^\/+/, "")}`;
const agentApiBackends = (
  process.env.AGENT_API_BACKENDS ||
  process.env.AGENT_API_BASE_URL ||
  "http://localhost:8099"
)
  .split(",")
  .map((url) => url.trim().replace(/\/+$/, ""))
  .filter(Boolean);

app.use(cors());
app.use(express.json({ limit: "25mb" }));

function stableHash(value: string) {
  let hash = 2166136261;
  for (let i = 0; i < value.length; i += 1) {
    hash ^= value.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function getRequestIdentity(body: BridgeChatRequest) {
  const userId = body.user_id || body.userId || "anonymous";
  const sessionId = body.session_id || body.sessionId || `${userId}-default`;
  return { userId, sessionId };
}

function selectBackend(
  body: BridgeChatRequest,
  forcedBackendUrl?: string,
): SelectedBackend {
  if (forcedBackendUrl) {
    return {
      baseUrl: forcedBackendUrl,
      routeKey: "forced",
      index: -1,
    };
  }

  if (agentApiBackends.length === 0) {
    throw new Error("No Agent Runtime backends configured");
  }

  const { userId, sessionId } = getRequestIdentity(body);
  const routeKey = `${userId}:${sessionId}`;
  const index = stableHash(routeKey) % agentApiBackends.length;
  return {
    baseUrl: agentApiBackends[index],
    routeKey,
    index,
  };
}

function backendAt(index: number, routeKey: string): SelectedBackend {
  return {
    baseUrl: agentApiBackends[index],
    routeKey,
    index,
  };
}

function toRuntimeRequest(body: BridgeChatRequest) {
  const { userId, sessionId } = getRequestIdentity(body);
  const ownerUserId = String(body.owner_user_id || body.ownerUserId || "").trim();
  const ownerAliases = Array.isArray(body.owner_aliases)
    ? body.owner_aliases
    : Array.isArray(body.ownerAliases)
      ? body.ownerAliases
      : [];
  const ownerAliasesText = ownerAliases
    .map((value) => String(value || "").trim())
    .filter(Boolean)
    .join(",");
  const contextUsername = String(
    body.context_username || body.contextUsername || "",
  ).trim();
  const input =
    body.input && body.input.length > 0
      ? body.input
      : [
          {
            role: "user" as const,
            type: "message" as const,
            content: [
              {
                type: "text",
                text: body.message || "",
              },
            ],
          },
        ];
  const runtimeContextMessage: RuntimeMessage | null =
    body.training_container || body.evaluation_container || body.grpo_container || body.resource_group_id || body.training_pool_id || body.user_role || ownerUserId || ownerAliasesText || contextUsername
      ? {
          role: "system",
          type: "message",
          content: [
            {
              type: "text",
              text: `${RUNTIME_CONTEXT_MARKER} training_container=${body.training_container || ""} evaluation_container=${body.evaluation_container || ""} grpo_container=${body.grpo_container || ""} resource_group_id=${body.resource_group_id || ""} training_pool_id=${body.training_pool_id || ""} user_role=${body.user_role || ""} owner_user_id=${ownerUserId} owner_aliases=${ownerAliasesText} context_username=${contextUsername}`,
            },
          ],
        }
      : null;
  const requestInput = runtimeContextMessage
    ? [runtimeContextMessage, ...input]
    : input;

  const runtimeRequest = {
    user_id: userId,
    session_id: sessionId,
    stream: true,
    training_container: body.training_container,
    evaluation_container: body.evaluation_container,
    grpo_container: body.grpo_container,
    resource_group_id: body.resource_group_id,
    training_pool_id: body.training_pool_id,
    user_role: body.user_role,
    owner_user_id: ownerUserId,
    owner_aliases: ownerAliases,
    context_username: contextUsername,
    input: requestInput,
  };
  return runtimeRequest;
}

async function fetchRuntimeBackend(
  backend: SelectedBackend,
  runtimeRequest: ReturnType<typeof toRuntimeRequest>,
  signal?: AbortSignal,
) {
  if (!agentApiToken) {
    throw new Error("MEDFLOW_AGENT_API_TOKEN is not configured");
  }
  return fetch(`${backend.baseUrl}${agentApiProcessPath}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
      "X-Agent-Route-Key": backend.routeKey,
      "X-Agent-Backend-Index": String(backend.index),
      Authorization: `Bearer ${agentApiToken}`,
    },
    body: JSON.stringify(runtimeRequest),
    signal,
  });
}

async function callRuntime(
  body: BridgeChatRequest,
  forcedBackendUrl?: string,
  signal?: AbortSignal,
) {
  const firstBackend = selectBackend(body, forcedBackendUrl);
  const runtimeRequest = toRuntimeRequest(body);
  const errors: string[] = [];

  // 如果强制指定了 backend，只尝试该 backend
  if (forcedBackendUrl) {
    try {
      const response = await fetchRuntimeBackend(
        firstBackend,
        runtimeRequest,
        signal,
      );
      return { response, backend: firstBackend, fallback: false, errors };
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      errors.push(`${firstBackend.baseUrl}: ${message}`);
      console.warn(
        `Forced Agent Runtime backend failed: ${firstBackend.baseUrl}`,
        error,
      );
      throw new Error(
        `Forced Agent Runtime backend failed: ${errors.join("; ")}`,
      );
    }
  }

  for (let offset = 0; offset < agentApiBackends.length; offset += 1) {
    const index = (firstBackend.index + offset) % agentApiBackends.length;
    const backend = backendAt(index, firstBackend.routeKey);

    try {
      const response = await fetchRuntimeBackend(
        backend,
        runtimeRequest,
        signal,
      );
      return { response, backend, fallback: offset > 0, errors };
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      errors.push(`${backend.baseUrl}: ${message}`);
      console.warn(`Agent Runtime backend failed: ${backend.baseUrl}`, error);
    }
  }

  throw new Error(`All Agent Runtime backends failed: ${errors.join("; ")}`);
}

function parseSsePayloads(raw: string): RuntimeEvent[] {
  return raw
    .split(/\r?\n\r?\n/)
    .flatMap((event) =>
      event
        .split(/\r?\n/)
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trim())
        .filter(Boolean),
    )
    .map((payload) => JSON.parse(payload) as RuntimeEvent);
}

function shouldCloseStream(event: RuntimeEvent) {
  return (
    event.object === "response" &&
    ["completed", "failed", "canceled", "cancelled"].includes(
      String(event.status || ""),
    )
  );
}

function extractRuntimeErrorMessage(event: RuntimeEvent) {
  const error =
    event.error && typeof event.error === "object"
      ? (event.error as { message?: unknown; detail?: unknown })
      : null;
  const candidates = [
    event.message,
    typeof event.error === "string" ? event.error : undefined,
    error?.message,
    error?.detail,
    event.body,
    typeof event.detail === "string" ? event.detail : undefined,
  ];
  return (
    candidates.find(
      (candidate): candidate is string =>
        typeof candidate === "string" && candidate.trim().length > 0,
    ) || ""
  );
}

function writeSseEvent(res: Response, rawEvent: string) {
  if (!rawEvent.trim()) return;
  res.write(`${rawEvent}\n\n`);
}

function takeNextSseEvent(
  buffer: string,
): { event: string; rest: string } | null {
  const match = /\r?\n\r?\n/.exec(buffer);
  if (!match || match.index === undefined) return null;
  return {
    event: buffer.slice(0, match.index),
    rest: buffer.slice(match.index + match[0].length),
  };
}

app.get("/health", (_req, res) => {
  res.json({
    ok: true,
    mode:
      agentApiBackends.length > 1 ? "sticky-multi-backend" : "single-backend",
    agentApiBackends,
    agentApiProcessPath,
  });
});

app.post("/api/agent/process", async (req: Request, res: Response) => {
  const controller = new AbortController();
  req.on("aborted", () => {
    controller.abort();
  });
  res.on("close", () => {
    if (!res.writableEnded && !controller.signal.aborted) {
      controller.abort();
    }
  });

  try {
    const forcedBackendUrl =
      typeof req.headers["x-agent-target-backend-url"] === "string"
        ? req.headers["x-agent-target-backend-url"]
        : undefined;

    const {
      response: runtimeResponse,
      backend,
      fallback,
      errors,
    } = await callRuntime(
      req.body as BridgeChatRequest,
      forcedBackendUrl,
      controller.signal,
    );

    if (!runtimeResponse.ok || !runtimeResponse.body) {
      const text = await runtimeResponse.text();
      res.status(502).json({
        error: "Agent Runtime request failed",
        backend,
        fallback,
        backendErrors: errors,
        status: runtimeResponse.status,
        body: text,
      });
      return;
    }

    res.status(200);
    res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
    res.setHeader("Cache-Control", "no-cache, no-transform");
    res.setHeader("Connection", "keep-alive");
    res.setHeader("X-Agent-Backend", backend.baseUrl);
    res.setHeader("X-Agent-Backend-Index", String(backend.index));
    res.setHeader("X-Agent-Backend-Fallback", String(fallback));
    res.flushHeaders?.();

    const reader = runtimeResponse.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let shouldEnd = false;

    while (!shouldEnd) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      while (true) {
        const nextEvent = takeNextSseEvent(buffer);
        if (!nextEvent) break;
        const rawEvent = nextEvent.event;
        buffer = nextEvent.rest;

        writeSseEvent(res, rawEvent);

        try {
          const events = parseSsePayloads(`${rawEvent}\n\n`);
          if (events.some(shouldCloseStream)) {
            shouldEnd = true;
            break;
          }
        } catch (parseError) {
          console.warn("Failed to parse SSE event from runtime:", parseError);
        }
      }
    }

    if (buffer.trim()) {
      writeSseEvent(res, buffer);
    }

    await reader.cancel().catch(() => undefined);
    res.end();
  } catch (error) {
    if (controller.signal.aborted && res.headersSent) {
      res.end();
      return;
    }

    if (!res.headersSent) {
      res.status(500).json({
        error: error instanceof Error ? error.message : String(error),
      });
    } else {
      res.write(
        `data: ${JSON.stringify({
          object: "error",
          message: error instanceof Error ? error.message : String(error),
        })}\n\n`,
      );
      res.end();
    }
  } finally {
    controller.abort();
  }
});

app.post("/api/agent/message", async (req: Request, res: Response) => {
  try {
    const forcedBackendUrl =
      typeof req.headers["x-agent-target-backend-url"] === "string"
        ? req.headers["x-agent-target-backend-url"]
        : undefined;

    const { response: runtimeResponse, backend } = await callRuntime(
      req.body as BridgeChatRequest,
      forcedBackendUrl,
    );
    const raw = await runtimeResponse.text();

    if (!runtimeResponse.ok) {
      res.status(502).json({
        error: "Agent Runtime request failed",
        backend,
        status: runtimeResponse.status,
        body: raw,
      });
      return;
    }

    const events = parseSsePayloads(raw);
    const errorEvent = [...events].reverse().find(
      (event) =>
        event.object === "error" ||
        (event.object === "response" &&
          ["failed", "canceled", "cancelled"].includes(
            String(event.status || ""),
          )),
    );
    if (errorEvent) {
      res.status(502).json({
        error:
          extractRuntimeErrorMessage(errorEvent) ||
          "Agent Runtime returned an error event",
        backend,
        events,
      });
      return;
    }

    const completedMessage = [...events]
      .reverse()
      .find(
        (event) => event.object === "message" && event.status === "completed",
      );
    const contentText = events
      .filter(
        (event) => event.object === "content" && typeof event.text === "string",
      )
      .map((event) => event.text)
      .join("");

    res.json({
      message:
        completedMessage?.metadata?.protocol &&
        typeof completedMessage.metadata.protocol === "object" &&
        "message" in completedMessage.metadata.protocol
          ? (completedMessage.metadata.protocol as { message?: string })
              .message || contentText
          : contentText,
      protocol: completedMessage?.metadata?.protocol || null,
      backend,
      events,
    });
  } catch (error) {
    res.status(500).json({
      error: error instanceof Error ? error.message : String(error),
    });
  }
});

app.post("/api/agent/reset", async (req: Request, res: Response) => {
  try {
    const {
      userId,
      user_id,
      sessionId,
      session_id,
      contextUsername,
      username,
      cancelWorkflows,
    } = req.body;
    const uid = user_id || userId || "anonymous";
    const sid = session_id || sessionId || `${uid}-default`;

    const forcedBackendUrl =
      typeof req.headers["x-agent-target-backend-url"] === "string"
        ? req.headers["x-agent-target-backend-url"]
        : undefined;

    const body = {
      user_id: uid,
      session_id: sid,
      contextUsername,
      username,
      cancelWorkflows: Boolean(cancelWorkflows),
    };
    const { baseUrl, routeKey, index } = selectBackend(body, forcedBackendUrl);

    if (!agentApiToken) {
      throw new Error("MEDFLOW_AGENT_API_TOKEN is not configured");
    }
    const resetRes = await fetch(`${baseUrl}/reset`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${agentApiToken}`,
      },
      body: JSON.stringify(body),
    });

    if (!resetRes.ok) {
      const text = await resetRes.text();
      res.status(502).json({
        error: "Agent Runtime reset failed",
        backend: { baseUrl, routeKey, index },
        status: resetRes.status,
        body: text,
      });
      return;
    }

    const result = await resetRes.json();
    res.json({
      success: true,
      backend: { baseUrl, routeKey, index },
      result,
    });
  } catch (error) {
    res.status(500).json({
      error: error instanceof Error ? error.message : String(error),
    });
  }
});

app.listen(port, () => {
  console.log(
    `Agent Studio Runtime Bridge listening on http://localhost:${port}`,
  );
  console.log(
    `Proxying Agent Runtime backends: ${agentApiBackends.join(", ")}`,
  );
  console.log("Routing mode: sticky hash by userId/sessionId");
});

