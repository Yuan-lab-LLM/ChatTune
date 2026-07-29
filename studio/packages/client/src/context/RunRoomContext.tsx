import {
  createContext,
  ReactNode,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  ResponseBody,
  InputRequestData,
  ModelInvocationData,
  Reply,
  RunData,
  SocketEvents,
} from "../../../shared/src/types/trpc";
import { useSocket } from "./SocketContext";

import { useParams } from "react-router-dom";
import {
  ContentBlocks,
  ContentType,
} from "../../../shared/src/types/messageForm";
import {
  SpanData,
  TraceData,
  TraceStatus,
} from "../../../shared/src/types/trace";
import { getTimeDifferenceNano } from "../../../shared/src/utils/timeUtils";
import { ProjectNotFoundPage } from "../pages/DefaultPage";
import { useMessageApi } from "./MessageApiContext.tsx";
import { useWandb } from "@/context/WandbContext.tsx";
import { useAuth } from "@/context/AuthContext.tsx";
import { trpc } from "@/api/trpc.ts";

/**
 * Extract text from content (similar to TrainingMetricsPanel)
 * Handles both string and array (ContentBlocks) content
 */
function extractTextFromContent(content: ContentType): string {
  if (typeof content === "string") {
    return content;
  }
  if (Array.isArray(content)) {
    return content
      .map((block) => {
        if (block.type === "text" && "text" in block) {
          return (block as { text: string }).text || "";
        }
        // Handle tool_result type
        if (block.type === "tool_result" && "output" in block) {
          const output = (block as { output: unknown }).output;
          if (typeof output === "string") {
            return output;
          }
          if (Array.isArray(output)) {
            return output
              .map((o: { type?: string; text?: string }) =>
                o?.type === "text" ? o.text || "" : "",
              )
              .join("");
          }
        }
        return "";
      })
      .join("");
  }
  return "";
}

const getMetadataString = (
  metadata: object | undefined,
  key: string,
): string | undefined => {
  const value = (metadata as Record<string, unknown> | undefined)?.[key];
  return typeof value === "string" ? value : undefined;
};

const nameMatchesContextUsername = (
  name: string | undefined,
  username: string | null,
): boolean => {
  if (!name || !username) {
    return false;
  }

  // 优先：完整 username 匹配（含 #sessionId）
  if (name.includes(`[${username}]`)) {
    return true;
  }

  if (username.includes("#")) {
    // 新增：如果完整 username 不匹配，尝试基础用户名匹配
    const baseUsername = username.split("#")[0];
    if (name.includes(`[${baseUsername}]`)) {
      return true;
    }
    // 原有逻辑：包含完整 username（如 Orchestrator_uuid#runId 匹配 username）
    return name.includes(username);
  }

  return name === username || name.endsWith(`_${username}`);
};

const messageMatchesContextUsername = (
  contentText: string,
  name: string | undefined,
  metadata: object | undefined,
  username: string | null,
): boolean => {
  if (!username) {
    return false;
  }

  // 1. 优先：metadata 中的完整 contextUsername（最精确，含 #sessionId）
  const metadataContextUsername = getMetadataString(
    metadata,
    "__medflowContextUsername",
  );
  if (metadataContextUsername === username) {
    return true;
  }

  // 2. metadata 中的基础 username（兼容不带 # 的旧逻辑）
  const metadataUsername = getMetadataString(metadata, "__medflowUsername");
  if (
    metadataUsername === username ||
    metadataUsername === username.split("#")[0]
  ) {
    return true;
  }

  // 3. contentText 前缀匹配（用户消息通常以 [username] 开头）
  if (contentText.startsWith(`[${username}]`)) {
    return true;
  }

  // 4. name 匹配（支持完整 username 和基础 username fallback）
  return nameMatchesContextUsername(name, username);
};

type WandbExtraction = { url: string | null; pending: boolean };

const normalizeWandbUrl = (url: string): string =>
  url.trim().replace(/%60(?=($|[?#]))/gi, "").replace(/[`),.;，。；]+(?=($|[?#]))/g, "");
const asRecord = (value: unknown): Record<string, unknown> | null =>
  value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;

const wandbFromMetrics = (metrics: unknown): WandbExtraction | null => {
  const data = asRecord(metrics);
  if (!data) return null;

  const rawUrl = data.wandb_url ?? data.wandbUrl;
  if (typeof rawUrl === "string" && rawUrl.trim()) {
    return { url: normalizeWandbUrl(rawUrl), pending: false };
  }

  const pending = data.wandb_url_pending ?? data.wandbUrlPending;
  if (pending === true) {
    return { url: null, pending: true };
  }

  return null;
};

const extractWandbFromProtocol = (payload: unknown): WandbExtraction | null => {
  const data = asRecord(payload);
  if (!data) return null;

  const direct = wandbFromMetrics(data.metrics) ?? wandbFromMetrics(data);
  if (direct?.url) return direct;

  const stages = asRecord(data.stages);
  const currentStage = typeof data.currentStage === "string" ? data.currentStage : null;
  const orderedStageNames = [
    currentStage,
    "train",
    ...Object.keys(stages || {}),
  ].filter((name, index, names): name is string => Boolean(name) && names.indexOf(name) === index);

  let pendingFallback: WandbExtraction | null = direct?.pending ? direct : null;
  for (const stageName of orderedStageNames) {
    const stage = asRecord(stages?.[stageName]);
    if (!stage) continue;

    const fromStageMetrics = wandbFromMetrics(stage.metrics);
    if (fromStageMetrics?.url) return fromStageMetrics;
    if (!pendingFallback && fromStageMetrics?.pending) {
      pendingFallback = fromStageMetrics;
    }

    const fromStage = wandbFromMetrics(stage);
    if (fromStage?.url) return fromStage;
    if (!pendingFallback && fromStage?.pending) {
      pendingFallback = fromStage;
    }
  }

  return pendingFallback;
};
/**
 * Extract wandb info from text content
 * Supports both JSON format and plain text format
 * @param text - The text content to parse
 * @returns Object with wandb_url and wandb_url_pending or null
 */
function extractWandbFromText(
  text: string,
): { url: string | null; pending: boolean } | null {
  if (!text || text.trim() === "") return null;

  // Try to parse as JSON first
  try {
    const jsonData = JSON.parse(text);

    const protocolWandb = extractWandbFromProtocol(jsonData);
    if (protocolWandb) {
      return protocolWandb;
    }
  } catch (e) {
    // Not valid JSON, continue with text extraction
  }

  // Fallback to regex extraction for plain text
  // Match wandb_url_pending: true/false
  const pendingMatch = text.match(
    /["']wandb_url_pending["']\s*:\s*(true|false)/i,
  );
  const pending = pendingMatch
    ? pendingMatch[1].toLowerCase() === "true"
    : false;

  // Match wandb_url: "url" or wandb_url: null
  const urlMatch = text.match(
    /["']wandb_url["']\s*:\s*(?:["'](https?:\/\/[^"']+)["']|null)/i,
  );

  if (urlMatch && urlMatch[1]) {
    // Found URL
    return { url: normalizeWandbUrl(urlMatch[1]), pending: false };
  } else if (pendingMatch) {
    // Found pending flag (url might be null or not present)
    return { url: null, pending };
  }

  const plainUrlMatch = text.match(
    /https?:\/\/wandb\.ai\/[^\s\])"'<>`，。]+/i,
  );
  if (plainUrlMatch?.[0]) {
    return { url: normalizeWandbUrl(plainUrlMatch[0]), pending: false };
  }

  return null;
}

function isUserRole(role: string | undefined | null): boolean {
  const normalizedRole = role?.toLowerCase() || "";
  return normalizedRole === "user" || normalizedRole.endsWith("_user");
}

interface RunRoomContextType {
  trace: TraceData | null;
  spans: SpanData[];
  inputRequests: InputRequestData[];
  runData: RunData | null;
  runId: string;
  modelInvocationData: ModelInvocationData | null;
  sendUserInputToServer: (
    requestId: string,
    blocksInput: ContentBlocks,
    structuredInput: Record<string, unknown> | null,
  ) => void;
  resetAgentContext: (
    contextUsername?: string,
    options?: { silent?: boolean; cancelWorkflows?: boolean },
  ) => Promise<ResponseBody | undefined>;
  cancelRuntimeResponse: () => Promise<ResponseBody | undefined>;
}

const RunRoomContext = createContext<RunRoomContextType | null>(null);

interface Props {
  children: ReactNode;
}

const calculateTraceData = (spans: SpanData[]) => {
  if (!spans.length) return null;

  // Find earliest start time and latest end time by comparing nanosecond timestamps directly
  const startTimes = spans.map((span) => parseInt(span.startTimeUnixNano));
  const endTimes = spans.map((span) => parseInt(span.endTimeUnixNano));

  const earliestStartNano = Math.min(...startTimes);
  const latestEndNano = Math.max(...endTimes);

  // Convert to Date objects for display
  const earliestStart = new Date(earliestStartNano / 1000000).toISOString();
  const latestEnd = new Date(latestEndNano / 1000000).toISOString();

  const status = spans.some((span) => span.status.code === 2) // ERROR status code
    ? TraceStatus.ERROR
    : TraceStatus.OK;

  // Calculate duration directly from nanosecond timestamps
  const durationNano = getTimeDifferenceNano(earliestStartNano, latestEndNano);

  const data = {
    startTime: earliestStart,
    endTime: latestEnd,
    duration: durationNano,
    status: status,
  };
  return data;
};

export function RunRoomContextProvider({ children }: Props) {
  const { runId } = useParams<{ runId: string }>();
  const { messageApi } = useMessageApi();
  const { setUserWandbUrlInfo } = useWandb();
  const { user } = useAuth();

  const socket = useSocket();
  const roomName = `run-${runId}`;
  const [replies, setReplies] = useState<Reply[]>([]);

  const [spans, setSpans] = useState<SpanData[]>([]);
  const [trace, setTrace] = useState<TraceData | null>(null);

  const [inputRequests, setInputRequests] = useState<InputRequestData[]>([]);
  const [runData, setRunData] = useState<RunData | null>(null);
  const [modelInvocationData, setModelInvocationData] =
    useState<ModelInvocationData | null>(null);

  const chatSessionQuery = trpc.getChatSession.useQuery(
    { runId: runId || "" },
    { enabled: Boolean(runId), retry: false },
  );
  const currentUsername = useMemo(() => {
    const username = user?.username || "";
    const sessionId = chatSessionQuery.data?.data?.sessionId;
    return username && sessionId ? `${username}#${sessionId}` : username;
  }, [chatSessionQuery.data?.data?.sessionId, user?.username]);
  const currentUsernameRef = useRef(currentUsername);
  useEffect(() => {
    currentUsernameRef.current = currentUsername;
  }, [currentUsername]);

  // 使用服务端返回的时间戳作为 joinTime（在下面的 useEffect 中设置）
  const [joinTime, setJoinTime] = useState<string>("");

  useEffect(() => {
    if (spans.length > 0) {
      const traceData = calculateTraceData(spans);

      if (traceData) {
        setTrace({
          startTime: traceData.startTime,
          endTime: traceData.endTime,
          latencyNs: traceData.duration,
          status: traceData.status,
          runId: runId,
        } as TraceData);
      }
    }
  }, [spans]);

  useEffect(() => {
    if (!socket) {
      console.error("Socket is null. Cannot join run room.");
      return;
    }


    // Clear the data first
    setInputRequests([]);
    setReplies([]);
    setSpans([]);
    setRunData(null);
    setModelInvocationData(null);
    // 重置 joinTime
    setJoinTime("");


    // 标记是否已收到响应
    let hasReceivedResponse = false;

    socket.emit(
      SocketEvents.client.joinRunRoom,
      runId,
      (response: ResponseBody & { serverTimestamp?: string }) => {
        hasReceivedResponse = true;

        if (!response.success) {
          console.error("Failed to join run room:", response.message);
          messageApi.error(response.message);
        } else if (response.serverTimestamp) {
          // 使用服务端返回的时间戳作为 joinTime
          setJoinTime(response.serverTimestamp);
        } else {
          // 备用：使用浏览器时间
          const browserTime = new Date().toISOString();
          console.warn(
            "No server timestamp in joinRunRoom response, using browser time:",
            browserTime,
          );
          setJoinTime(browserTime);
        }
      },
    );

    // 5秒超时保护：如果 joinRunRoom 没有响应，使用浏览器时间
    const timeout = setTimeout(() => {
      if (!hasReceivedResponse) {
        console.warn(
          "joinRunRoom timeout (5s), using browser time",
        );
        const browserTime = new Date().toISOString();
        setJoinTime((current) => {
          if (!current) {
            return browserTime;
          }
          return current;
        });
      }
    }, 5000);

    // 监听 socket 断开和重连
    const handleDisconnect = () => {
    };

    const handleConnect = () => {
      // 重新加入房间
      socket.emit(
        SocketEvents.client.joinRunRoom,
        runId,
        (response: ResponseBody & { serverTimestamp?: string }) => {
          if (response.success && response.serverTimestamp) {
            setJoinTime(response.serverTimestamp);
          }
        },
      );
    };

    socket.on("disconnect", handleDisconnect);
    socket.on("connect", handleConnect);

    // New messages
    socket.on(SocketEvents.server.pushMessages, (newReplies: Reply[]) => {

      const myUsername = currentUsernameRef.current;

      // Filter messages: only process messages belonging to current user
      const myReplies = newReplies.filter((reply) => {
        if (!reply.messages || reply.messages.length === 0) return false;

        const firstMsg = reply.messages[0];
        const msgRole = firstMsg.role;

        // 1. System messages: keep them for side panels such as
        // inference service parsing, but hide them later in chat UI.
        if (msgRole === "system") {
          return true;
        }

        // 2. User messages: check if content starts with [User-XXXX]
        if (isUserRole(msgRole)) {
          const contentText = extractTextFromContent(firstMsg.content);
          const isMyMessage = messageMatchesContextUsername(
            contentText,
            firstMsg.name,
            firstMsg.metadata,
            myUsername,
          );
          return isMyMessage;
        }

        // 3. Agent replies: check if msg.name or metadata belongs to the current context user
        if (msgRole === "assistant") {
          const contentText = extractTextFromContent(firstMsg.content);
          const isMyReply = messageMatchesContextUsername(
            contentText,
            firstMsg.name,
            firstMsg.metadata,
            myUsername,
          );
          return isMyReply;
        }

        return false;
      });


      // Extract wandb info for current user from replies
      let latestWandbUrl: string | null = null;
      let latestWandbPending = false;
      let foundWandbInfo = false;

      myReplies.forEach((reply, replyIndex) => {

        // Extract text from all messages in the reply
        if (reply.messages && Array.isArray(reply.messages)) {
          reply.messages.forEach((message, msgIndex) => {
            const contentText = extractTextFromContent(message.content);

            // Extract wandb info from text content
            const wandbInfo = extractWandbFromText(contentText);

            if (wandbInfo) {
              latestWandbUrl = wandbInfo.url;
              latestWandbPending = wandbInfo.pending;
              foundWandbInfo = true;
            }
          });
        }
      });

      // Update wandb URL info for current user (new links will replace old ones)
      if (foundWandbInfo) {
        const latestUsername =
          currentUsernameRef.current.split("#")[0] || currentUsernameRef.current;
        setUserWandbUrlInfo(latestUsername, latestWandbUrl, latestWandbPending);
      } else {
      }

      // Update replies state (only show my messages)
      if (myReplies.length === 0) {
        return;
      }

      setReplies((prev) => {
        const updatedReplies: Reply[] = [...prev];
        myReplies.forEach((newReply) => {
          const index = updatedReplies.findIndex(
            (reply) => reply.replyId === newReply.replyId,
          );

          if (index === -1) {
            // New reply, add it
            updatedReplies.push(newReply);
          } else {
            // Existing reply, update messages
            updatedReplies[index] = newReply;
          }
        });
        return updatedReplies;
      });
    });

    socket.on(SocketEvents.server.pushSpans, (newSpans: SpanData[]) => {
      setSpans((prevSpans) => {
        const updatedSpans = [...prevSpans];
        newSpans.forEach((newSpan) => {
          const index = updatedSpans.findIndex(
            (span) => span.spanId === newSpan.spanId,
          );
          if (index === -1) {
            updatedSpans.push(newSpan);
          } else {
            updatedSpans[index] = newSpan;
          }
        });

        return updatedSpans.sort((a, b) => {
          return parseInt(a.startTimeUnixNano) - parseInt(b.startTimeUnixNano);
        });
      });
    });

    socket.on(
      SocketEvents.server.pushModelInvocationData,
      (newModelInvocationData: ModelInvocationData) => {
        setModelInvocationData(newModelInvocationData);
      },
    );

    // New user input requests
    socket.on(
      SocketEvents.server.pushInputRequests,
      (newInputRequests: InputRequestData[]) => {
        setInputRequests((prevRequests) => {
          const requestsById = new Map<string, InputRequestData>();
          [...prevRequests, ...newInputRequests].forEach((request) => {
            requestsById.set(request.requestId, request);
          });
          return Array.from(requestsById.values());
        });
      },
    );

    socket.on(SocketEvents.server.removeInputRequest, (requestId: string) => {
      setInputRequests((prevRequests) =>
        prevRequests.filter((request) => request.requestId !== requestId),
      );
    });

    // Run data updates
    socket.on(SocketEvents.server.pushRunData, (newRunData: RunData) => {
      setRunData(newRunData);
    });

    // Clear input requests
    socket.on(SocketEvents.server.clearInputRequests, () => {
      setInputRequests([]);
    });

    return () => {

      // Clear timeout
      clearTimeout(timeout);

      if (socket) {
        // Clear the listeners and leave the room
        socket.off(SocketEvents.server.pushMessages);
        socket.off(SocketEvents.server.pushSpans);
        socket.off(SocketEvents.server.pushInputRequests);
        socket.off(SocketEvents.server.removeInputRequest);
        socket.off(SocketEvents.server.pushRunData);
        socket.off(SocketEvents.server.clearInputRequests);
        socket.off(SocketEvents.server.pushModelInvocationData);
        // Remove socket connection listeners
        socket.off("disconnect", handleDisconnect);
        socket.off("connect", handleConnect);
        socket.emit(SocketEvents.client.leaveRoom, roomName);
      }
    };
  }, [socket, runId, roomName]);

  if (!runId) {
    return <ProjectNotFoundPage />;
  }

  /**
   * Send the user input to the server
   *
   * @param requestId
   * @param blocksInput
   * @param structuredInput
   */
  const sendUserInputToServer = (
    requestId: string,
    blocksInput: ContentBlocks,
    structuredInput: Record<string, unknown> | null,
  ) => {
    if (!socket) {
      messageApi.error("Server is not connected, please refresh the page.");
    } else {
      socket.emit(
        SocketEvents.client.sendUserInputToServer,
        requestId,
        blocksInput,
        structuredInput,
        (response?: ResponseBody) => {
          if (response && !response.success) {
            messageApi.error(response.message);
            return;
          }

          // Update the request queue only after the server accepts it.
          setInputRequests((prevRequests) =>
            prevRequests.filter((request) => request.requestId !== requestId),
          );
        },
      );
    }
  };

  const resetAgentContext = (
    contextUsername?: string,
    options?: { silent?: boolean; cancelWorkflows?: boolean },
  ) =>
    new Promise<ResponseBody | undefined>((resolve) => {
      if (!socket) {
        const response = {
          success: false,
          message: "Server is not connected, please refresh the page.",
        };
        if (!options?.silent) {
          messageApi.error(response.message);
        }
        resolve(response);
        return;
      }

      socket.emit(
        SocketEvents.client.resetAgentContext,
        runId,
        { contextUsername, cancelWorkflows: Boolean(options?.cancelWorkflows) },
        (response?: ResponseBody) => {
          if (response && !response.success && !options?.silent) {
            messageApi.error(response.message);
          }
          resolve(response);
        },
      );
    });

  const cancelRuntimeResponse = () =>
    new Promise<ResponseBody | undefined>((resolve) => {
      if (!socket) {
        const response = {
          success: false,
          message: "Server is not connected, please refresh the page.",
        };
        messageApi.error(response.message);
        resolve(response);
        return;
      }

      socket.emit(
        SocketEvents.client.cancelRuntimeResponse,
        runId,
        (response?: ResponseBody) => {
          if (response && !response.success) {
            messageApi.error(response.message);
          }
          resolve(response);
        },
      );
    });

  return (
    <RunRoomContext.Provider
      value={{
        runId,
        replies,
        trace,
        spans,
        inputRequests,
        runData,
        sendUserInputToServer,
        resetAgentContext,
        cancelRuntimeResponse,
        modelInvocationData,
      }}
    >
      {children}
    </RunRoomContext.Provider>
  );
}

export function useRunRoom() {
  const context = useContext(RunRoomContext);
  if (!context) {
    throw new Error("useRunRoom must be used within a RunRoomProvider");
  }
  return context;
}



