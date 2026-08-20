import { spawn, exec } from "child_process";
import { Server as HttpServer } from "http";
import { Server } from "socket.io";
import { ContentBlocks, Status } from "../../../shared/src/types/messageForm";
import {
  ResponseBody,
  InputRequestData,
  OverviewData,
  Reply,
  RunData,
  SocketEvents,
  SocketRoomName,
  GPUInfo,
} from "../../../shared/src/types/trpc";
import { RunDao } from "../dao/Run";
import { ReplyDao } from "../dao/Reply";
import { MessageDao } from "../dao/Message";
import { MessageForm, RegisterReplyParams } from "../../../shared/src";

import { ConfigManager } from "../../../shared/src/config/server";
import { SpanData } from "../../../shared/src/types/trace";
import { InputRequestDao } from "../dao/InputRequest";
import { SpanDao } from "../dao/Trace";
import { AuthDao, SafeAuthUser } from "../dao/Auth";
import { ChatSessionDao } from "../dao/ChatSession";
import { UserRole } from "../models/Auth";
import { ResourceAccessService } from "../services/resourceAccessService";
import {
  remoteResourceClient,
  resourceNodeRegistry,
} from "../services/resourceNodeService";

import { promisify } from "util";

const execAsync = promisify(exec);
const getConfiguredRuntimeToken = () =>
  process.env.MEDFLOW_STUDIO_RUNTIME_TOKEN?.trim() ||
  process.env.AGENTSCOPE_STUDIO_RUNTIME_TOKEN?.trim() ||
  "";
const runtimeRequestPrefix = "runtime:";
const AUTH_COOKIE_NAME = "medflow_auth_token";

const parseCookies = (header?: string | string[]) =>
  Object.fromEntries(
    (Array.isArray(header) ? header.join(";") : header || "")
      .split(";")
      .map((part) => part.trim())
      .filter(Boolean)
      .map((part) => {
        const [rawKey, ...rawValue] = part.split("=");
        return [decodeURIComponent(rawKey), decodeURIComponent(rawValue.join("="))];
      }),
  );

const configuredSocketCorsOrigins = () => {
  const raw = process.env.MEDFLOW_SOCKET_CORS_ORIGINS?.trim();
  if (raw) {
    return raw.split(",").map((origin) => origin.trim()).filter(Boolean);
  }
  const publicUrl = ConfigManager.getInstance().getConfig().publicUrl;
  return publicUrl ? [publicUrl] : true;
};

function createRuntimeId(prefix: string): string {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function contentBlocksToText(blocksInput: ContentBlocks): string {
  return blocksInput
    .map((block) => {
      if (block.type === "text") {
        return block.text;
      }
      if (block.type === "thinking") {
        return block.thinking;
      }
      return "";
    })
    .join("");
}

function textToContentBlocks(text: string): ContentBlocks {
  return [
    {
      type: "text",
      text,
    } as ContentBlocks[number],
  ];
}

function extractRuntimeEventText(payload: Record<string, any>): string {
  if (payload.object === "content" && typeof payload.text === "string") {
    return payload.text;
  }

  if (payload.object === "message" && Array.isArray(payload.content)) {
    return payload.content
      .map((item: Record<string, any>) =>
        typeof item.text === "string" ? item.text : "",
      )
      .join("");
  }

  const protocolMessage = payload.metadata?.protocol?.message;
  return typeof protocolMessage === "string" ? protocolMessage : "";
}

function extractRuntimeProtocolMessage(
  metadata: Record<string, any> | null | undefined,
): string {
  const message = metadata?.protocol?.message;
  return typeof message === "string" ? message : "";
}

function isRuntimeWorkflowProtocol(
  metadata: Record<string, any> | null | undefined,
): boolean {
  const protocolType = metadata?.protocol?.type;
  return (
    typeof protocolType === "string" && protocolType.startsWith("workflow_")
  );
}

function isRuntimeWorkflowStatusMessage(
  metadata: Record<string, any> | null | undefined,
): boolean {
  return (
    metadata?.protocol?.type === "workflow_status" &&
    typeof metadata.protocol.message === "string" &&
    metadata.protocol.message.trim().length > 0
  );
}

function isWorkflowControlMessage(message: string): boolean {
  const normalized = message.trim().toLowerCase();
  if (!normalized) {
    return false;
  }
  return (
    /^(resume|continue|retry|stop|cancel)\b/.test(
      normalized,
    ) ||
    /^(继续|续跑|恢复|停止|结束|取消)/.test(normalized)
  );
}

function extractRuntimeErrorMessage(payload: Record<string, any>): string {
  const candidates = [
    payload.message,
    payload.error,
    payload.error?.message,
    payload.error?.detail,
    payload.body,
    payload.detail,
    payload.metadata?.protocol?.message,
  ];

  for (const candidate of candidates) {
    if (typeof candidate === "string" && candidate.trim()) {
      return candidate.trim();
    }
  }

  return "";
}

function formatRuntimeErrorMessage(rawMessage: string): string {
  const lowerMessage = rawMessage.toLowerCase();
  const isConnectionError =
    lowerMessage.includes("apiconnectionerror") ||
    lowerMessage.includes("connection error") ||
    lowerMessage.includes("timeout") ||
    lowerMessage.includes("timed out") ||
    lowerMessage.includes("fetch failed") ||
    lowerMessage.includes("econnrefused") ||
    lowerMessage.includes("etimedout");

  const friendlyMessage = isConnectionError
    ? "后台模型服务连接超时或暂时不可用，请稍后重试；如果持续失败，请检查模型服务、网络或 API 配置。"
    : "后台模型服务返回异常，请稍后重试；如果持续失败，请联系管理员查看服务日志。";

  const detail = rawMessage.trim();
  return detail ? `${friendlyMessage}\n\n错误详情：${detail}` : friendlyMessage;
}

function isAbortError(error: unknown): boolean {
  return (
    error instanceof Error &&
    (error.name === "AbortError" || error.message.includes("aborted"))
  );
}

export class SocketManager {
  private static io: Server;
  private static connectedClients = new Map<
    string,
    { joinedAt: Date; userId: string }
  >();
  private static serverStartTime = new Date();
  private static gpuInfoCache: { data: GPUInfo[]; expiresAt: number } | null = null;
  private static gpuInfoRefreshPromise: Promise<GPUInfo[]> | null = null;
  private static gpuInfoByNodeCache = new Map<
    string,
    { data: GPUInfo[]; expiresAt: number }
  >();
  private static gpuInfoByNodeRefreshPromises = new Map<string, Promise<GPUInfo[]>>();
  private static runtimeBridgeControllers = new Map<string, AbortController>();
  private static readonly gpuInfoCacheTtlMs = Number(
    process.env.MEDFLOW_GPU_CACHE_TTL_MS || 5000,
  );
  private static readonly gpuOverviewNodeTimeoutMs = Math.max(
    500,
    Number(process.env.MEDFLOW_GPU_OVERVIEW_NODE_TIMEOUT_MS) || 3000,
  );

  private static systemAdminUser(): SafeAuthUser {
    return {
      id: "__system__",
      username: "system",
      role: UserRole.ADMIN,
      disabled: false,
      mustChangePassword: false,
      createdAt: "",
    };
  }

  private static isRuntimeAuthorized(token: unknown) {
    const expectedToken = getConfiguredRuntimeToken();
    if (!expectedToken) {
      console.error(
        "MEDFLOW_STUDIO_RUNTIME_TOKEN is not configured; Python socket runtime access is disabled.",
      );
      return false;
    }

    return typeof token === "string" && token === expectedToken;
  }

  private static projectListRoom(user?: SafeAuthUser | null) {
    return user?.role === UserRole.ADMIN
      ? `${SocketRoomName.ProjectListRoom}-admin`
      : `${SocketRoomName.ProjectListRoom}-user-${user?.id}`;
  }

  private static projectRoom(project: string, user?: SafeAuthUser | null) {
    return user?.role === UserRole.ADMIN
      ? `project-${project}-admin`
      : `project-${project}-user-${user?.id}`;
  }

  private static overviewRoom(user?: SafeAuthUser | null) {
    return user?.role === UserRole.ADMIN
      ? `${SocketRoomName.OverviewRoom}-admin`
      : `${SocketRoomName.OverviewRoom}-user-${user?.id}`;
  }

  private static getSocketUser(socket: { data: { user?: SafeAuthUser } }) {
    return socket.data.user;
  }

  private static runtimeBridgeRequestKey(
    runId: string,
    user: SafeAuthUser | null | undefined,
  ) {
    return `${runId}:${user?.id || user?.username || "anonymous"}`;
  }

  private static connectedNormalUsers() {
    const users = new Map<string, SafeAuthUser>();
    this.io.of("/client").sockets.forEach((socket) => {
      const user = this.getSocketUser(socket);
      if (user?.role === UserRole.USER) users.set(user.id, user);
    });
    return [...users.values()];
  }

  private static async withCurrentResourceAccess(user: SafeAuthUser) {
    if (user.role === UserRole.ADMIN) return user;
    const group = await ResourceAccessService.getGroupForUser(user.id);
    return {
      ...user,
      assignedNodeId: await ResourceAccessService.getAssignedNodeId(user.id),
      group: group
        ? {
            id: group.id,
            name: group.name,
            defaultContainerName: group.defaultContainerName,
            defaultEvaluateContainerName: group.defaultEvaluateContainerName,
            defaultGrpoContainerName: group.defaultGrpoContainerName,
            defaultMultinodeContainerName: group.defaultMultinodeContainerName,
          }
        : null,
    };
  }

  private static async auditAdminAction(
    actor: SafeAuthUser,
    eventType: string,
    details?: Record<string, unknown>,
  ) {
    await ResourceAccessService.recordAuditEvent(eventType, null, actor.id, {
      actorUsername: actor.username,
      ...(details || {}),
    });
  }

  private static async resolveRuntimeBridgeResourceUser(
    runId: string,
    currentUser: SafeAuthUser | null | undefined,
    structuredInput?: Record<string, unknown> | null,
  ) {
    if (!currentUser || currentUser.role !== UserRole.ADMIN) {
      return currentUser;
    }
    const explicitGroupId =
      typeof structuredInput?.__medflowResourceGroupId === "string"
        ? structuredInput.__medflowResourceGroupId
        : typeof structuredInput?.resource_group_id === "string"
          ? structuredInput.resource_group_id
          : "";
    const explicitGroup = await ResourceAccessService.getGroupById(explicitGroupId);
    if (explicitGroup) {
      return {
        ...currentUser,
        group: {
          id: explicitGroup.id,
          name: explicitGroup.name,
          defaultContainerName: explicitGroup.defaultContainerName,
          defaultEvaluateContainerName: explicitGroup.defaultEvaluateContainerName,
          defaultGrpoContainerName: explicitGroup.defaultGrpoContainerName,
          defaultMultinodeContainerName: explicitGroup.defaultMultinodeContainerName,
        },
      };
    }
    const runOwnerUserId = await RunDao.getRunOwnerUserId(runId);
    if (runOwnerUserId) {
      const runOwnerGroup = await ResourceAccessService.getGroupForUser(
        runOwnerUserId,
      );
      if (runOwnerGroup) {
        return {
          ...currentUser,
          group: {
            id: runOwnerGroup.id,
            name: runOwnerGroup.name,
            defaultContainerName: runOwnerGroup.defaultContainerName,
            defaultEvaluateContainerName: runOwnerGroup.defaultEvaluateContainerName,
            defaultGrpoContainerName: runOwnerGroup.defaultGrpoContainerName,
            defaultMultinodeContainerName: runOwnerGroup.defaultMultinodeContainerName,
          },
        };
      }
      console.warn("[RuntimeBridge] Run owner has no resource group", {
        runId,
        ownerUserId: runOwnerUserId,
        adminUserId: currentUser.id,
      });
      return currentUser;
    }
    console.warn("[RuntimeBridge] Admin input has no explicit resource group", {
      runId,
      adminUserId: currentUser.id,
    });
    return currentUser;
  }

  private static runtimeBridgeContainers(
    resourceUser: SafeAuthUser | null | undefined,
  ) {
    return {
      trainingContainer:
        resourceUser?.group?.defaultContainerName ||
        ConfigManager.getInstance().getDefaultContainerName(),
      evaluationContainer:
        resourceUser?.group?.defaultEvaluateContainerName ||
        ConfigManager.getInstance().getDefaultEvaluateContainerName(),
      grpoContainer:
        resourceUser?.group?.defaultGrpoContainerName ||
        ConfigManager.getInstance().getDefaultGrpoContainerName(),
      multinodeTrainingContainer:
        resourceUser?.group?.defaultMultinodeContainerName ||
        ConfigManager.getInstance().getDefaultMultinodeContainerName(),
      resourceGroupId: resourceUser?.group?.id,
    };
  }

  private static async canUseRun(
    runId: string,
    user?: SafeAuthUser | null,
  ): Promise<boolean> {
    if (!user) return false;
    if (user.role === UserRole.ADMIN) return RunDao.doesRunExist(runId, user);
    const assignedNodeId =
      user.assignedNodeId || (await ResourceAccessService.getAssignedNodeId(user.id));
    if (!assignedNodeId) return false;
    return RunDao.isRunnableRunForNode(runId, assignedNodeId);
  }

  static async getGPUInfo(): Promise<GPUInfo[]> {
    const now = Date.now();
    if (this.gpuInfoCache && this.gpuInfoCache.expiresAt > now) {
      return this.gpuInfoCache.data;
    }
    if (this.gpuInfoRefreshPromise) {
      return this.gpuInfoRefreshPromise;
    }
    this.gpuInfoRefreshPromise = this.refreshGPUInfo();
    try {
      return await this.gpuInfoRefreshPromise;
    } finally {
      this.gpuInfoRefreshPromise = null;
    }
  }

  private static getCachedGPUInfo(): GPUInfo[] {
    if (this.gpuInfoCache?.data.length) {
      return this.gpuInfoCache.data;
    }
    return [...this.gpuInfoByNodeCache.values()].flatMap((cache) => cache.data);
  }

  private static getCachedGPUInfoForNode(nodeId?: string | null): GPUInfo[] {
    if (!nodeId) return [];
    return this.gpuInfoByNodeCache.get(nodeId)?.data || [];
  }

  private static async getCachedGPUInfoForUser(
    user?: SafeAuthUser | null,
  ): Promise<GPUInfo[]> {
    if (!user) return [];
    if (user.role === UserRole.ADMIN) return this.getCachedGPUInfo();
    const assignedNodeId =
      user.assignedNodeId || (await ResourceAccessService.getAssignedNodeId(user.id));
    return this.getCachedGPUInfoForNode(assignedNodeId);
  }

  private static async refreshGPUInfo(): Promise<GPUInfo[]> {
    const now = Date.now();
    const results = await Promise.all(
      resourceNodeRegistry
        .list()
        .map((node) => this.getGPUInfoForNode(node.id, true)),
    );
    const data = results.flat();
    this.gpuInfoCache = {
      data,
      expiresAt: now + this.gpuInfoCacheTtlMs,
    };
    return data;
  }

  private static async getGPUInfoForNode(
    nodeId: string,
    forceRefresh = false,
  ): Promise<GPUInfo[]> {
    const now = Date.now();
    const cached = this.gpuInfoByNodeCache.get(nodeId);
    if (!forceRefresh && cached && cached.expiresAt > now) {
      return cached.data;
    }

    const existingRefresh = this.gpuInfoByNodeRefreshPromises.get(nodeId);
    if (existingRefresh) return existingRefresh;

    const refreshPromise = this.refreshGPUInfoForNode(nodeId);
    this.gpuInfoByNodeRefreshPromises.set(nodeId, refreshPromise);
    try {
      return await refreshPromise;
    } finally {
      this.gpuInfoByNodeRefreshPromises.delete(nodeId);
    }
  }

  private static async refreshGPUInfoForNode(nodeId: string): Promise<GPUInfo[]> {
    const node = resourceNodeRegistry.get(nodeId);
    try {
      const response = await remoteResourceClient.request<{
        collectedAt?: string;
        data: GPUInfo[];
      }>(
        node.id,
        "gpus",
        undefined,
        undefined,
        this.gpuOverviewNodeTimeoutMs,
      );
      const data = response.data.map((gpu) => ({
        ...gpu,
        nodeId: node.id,
        nodeName: node.name,
        collectedAt: gpu.collectedAt || response.collectedAt,
      }));
      this.gpuInfoByNodeCache.set(node.id, {
        data,
        expiresAt: Date.now() + this.gpuInfoCacheTtlMs,
      });
      return data;
    } catch (error) {
      console.error(`Failed to get GPU info from ${node.name}:`, error);
      const cached = this.gpuInfoByNodeCache.get(node.id)?.data || [];
      if (cached.length > 0) {
        const message = error instanceof Error ? error.message : String(error);
        return cached.map((gpu) => ({
          ...gpu,
          stale: true,
          error: gpu.error || message,
        }));
      }
      return [];
    }
  }

  private static async emitGPUInfoForUser(
    socket: { emit: (event: string, data: GPUInfo[]) => void },
    user?: SafeAuthUser | null,
  ) {
    if (!user) {
      socket.emit(SocketEvents.server.pushGPUInfo, []);
      return;
    }
    const assignedNodeId =
      user.role === UserRole.ADMIN
        ? null
        : user.assignedNodeId || (await ResourceAccessService.getAssignedNodeId(user.id));
    if (user.role === UserRole.ADMIN) {
      socket.emit(SocketEvents.server.pushGPUInfo, this.getCachedGPUInfo());
      socket.emit(SocketEvents.server.pushGPUInfo, await this.getGPUInfo());
      return;
    }

    if (!assignedNodeId) {
      socket.emit(SocketEvents.server.pushGPUInfo, []);
      return;
    }

    const cachedGpuInfo = this.getCachedGPUInfoForNode(assignedNodeId);
    if (cachedGpuInfo.length > 0) {
      socket.emit(SocketEvents.server.pushGPUInfo, cachedGpuInfo);
    }
    const gpuInfo = await this.getGPUInfoForNode(assignedNodeId);
    socket.emit(SocketEvents.server.pushGPUInfo, gpuInfo);
  }

  static close() {
    if (this.io) {
      this.io.close();
    }
  }

  static getOnlineUsersCount(user?: SafeAuthUser | null): number {
    if (!user) {
      return 0;
    }

    if (user.role === UserRole.ADMIN) {
      return new Set(
        Array.from(this.connectedClients.values()).map(
          (client) => client.userId,
        ),
      ).size;
    }

    return Array.from(this.connectedClients.values()).some(
      (client) => client.userId === user.id,
    )
      ? 1
      : 0;
  }

  static getServerUptime(): string {
    const now = new Date();
    const diff = now.getTime() - this.serverStartTime.getTime();
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));
    const hours = Math.floor((diff % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
    const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
    return `${days}天${hours}小时${minutes}分钟`;
  }

  static init(httpServer: HttpServer) {
    this.io = new Server(httpServer, {
      cors: {
        origin: configuredSocketCorsOrigins(),
        credentials: true,
      },
      maxHttpBufferSize: Infinity,
    });

    // Python client connection
    const pythonNamespace = this.io.of("/python");
    pythonNamespace.use((socket, next) => {
      if (!this.isRuntimeAuthorized(socket.handshake.auth?.runtimeToken)) {
        next(new Error("Runtime token is required"));
        return;
      }
      next();
    });
    pythonNamespace.on("connection", (socket) => {
      const runId = socket.handshake.auth.run_id;
      console.debug(`${socket.id}: Python client connected`);
      if (typeof runId === "string" && runId) {
        socket.join(`python-run-${runId}`);
        console.debug(
          `${socket.id}: Python client joined room: python-run-${runId}`,
        );
      }

      socket.on("disconnect", async () => {
        // Delete all input requests for this run to prevent state corruption
        const deletedCount =
          await InputRequestDao.deleteInputRequestsByRunId(runId);
        if (deletedCount > 0) {
          console.log(
            `[Socket] Cleaned up ${deletedCount} input requests for run ${runId}`,
          );
        }

        this.changeRunStatusAndTriggerEvents(runId, Status.DONE).catch(
          (error) => {
            console.error(error);
            throw error;
          },
        );
      });
    });

    const clientNamespace = this.io.of("/client");
    clientNamespace.use(async (socket, next) => {
      try {
        const cookies = parseCookies(socket.handshake.headers.cookie);
        const token = socket.handshake.auth?.token || cookies[AUTH_COOKIE_NAME];
        const user = await AuthDao.getUserByToken(
          typeof token === "string" ? token : undefined,
        );

        if (!user) {
          next(new Error("Unauthorized"));
          return;
        }

        socket.data.user = user;
        next();
      } catch (error) {
        next(error instanceof Error ? error : new Error("Unauthorized"));
      }
    });

    clientNamespace.on("connection", (socket) => {
      const user = this.getSocketUser(socket);
      console.debug("Client connected");

      // 记录在线用户
      this.connectedClients.set(socket.id, {
        joinedAt: new Date(),
        userId: user?.id ?? "",
      });
      // 广播在线用户数更新
      this.broadcastOnlineUsersCount();

      socket.on("disconnect", () => {
        // 移除在线用户记录
        this.connectedClients.delete(socket.id);
        // 广播在线用户数更新
        this.broadcastOnlineUsersCount();
      });

      socket.on(SocketEvents.client.joinProjectListRoom, () => {
        if (!user) return;
        socket.join(this.projectListRoom(user));
        console.debug(
          `${socket.id}: joined room: ${this.projectListRoom(user)}`,
        );

        RunDao.getAllProjects(user)
          .then((projects) => {
            // Push projects to the client
            socket.emit(SocketEvents.server.pushProjects, projects);
          })
          .catch((error) => {
            console.error(error);
            throw error;
          });
      });

      socket.on(
        SocketEvents.client.joinProjectRoom,
        async (project: string, callback) => {
          const projectExist = await RunDao.doesProjectExist(
            project,
            SocketManager.systemAdminUser(),
          );
          const projectVisible = await RunDao.doesProjectExist(project, user);
          if (!projectExist || !projectVisible) {
            callback({
              success: false,
              message: `Project ${project} not found`,
            });
          } else {
            const roomName = this.projectRoom(project, user);
            socket.join(roomName);
            console.debug(`${socket.id}: joined room: ${roomName}`);

            // Return runs to this socket/client
            RunDao.getAllProjectRuns(project, user)
              .then((runs) => {
                // Push runs to the client
                socket.emit(SocketEvents.server.pushRunsData, runs);
              })
              .catch((error) => {
                console.error(`[StudioInput] Failed to join project room ${project}:`, error);
              });
          }
        },
      );

      socket.on(
        SocketEvents.client.joinRunRoom,
        async (runId: string, callback) => {
          const runExist = await this.canUseRun(runId, user);
          // 获取服务端当前时间戳
          const serverTimestamp = new Date().toISOString();

          if (!runExist) {
            if (typeof callback === "function") {
              callback({
                success: false,
                message: `Run ${runId} not found`,
              });
            }
          } else {
            const roomName = `run-${runId}`;
            socket.join(roomName);
            console.debug(`${socket.id}: joined room: ${roomName}`);

            // Return run data, input requests and messages to this socket/client
            RunDao.getRunData(runId, this.systemAdminUser())
              .then(async (data) => {
                if (user?.role !== UserRole.ADMIN) {
                  const session = await ChatSessionDao.getOrCreate(user!.id, runId);
                  const contextUsername = `${user!.username}#${session.sessionId}`;
                  data.replies = data.replies.filter((reply) =>
                    reply.messages.some((message) => {
                      const metadata = message.metadata as Record<string, unknown> | undefined;
                      return metadata?.__medflowContextUsername === contextUsername;
                    }),
                  );
                  data.inputRequests = [];
                  data.spans = [];
                  data.runData = {
                    ...data.runData,
                    project: "",
                    name: "MedFlow 智能服务",
                    timestamp: "",
                    run_dir: "",
                    pid: 0,
                    nodeId: null,
                  };
                }
                console.debug(
                  `[StudioInput] joinRunRoom run=${runId} status=${data.runData.status} inputRequests=${data.inputRequests.length} user=${user?.username ?? "unknown"}`,
                );
                socket.emit(SocketEvents.server.pushRunData, data.runData);
                socket.emit(
                  SocketEvents.server.pushInputRequests,
                  data.inputRequests,
                );
                // 对data.replies.messages按时间排序
                data.replies.forEach((reply) => {
                  reply.messages.sort((a, b) => {
                    return a.timestamp.localeCompare(b.timestamp);
                  });
                });

                socket.emit(SocketEvents.server.pushMessages, data.replies);
                socket.emit(SocketEvents.server.pushSpans, data.spans);
              })
              .catch((error) => {
                console.error(`[StudioInput] Failed to join run room ${runId}:`, error);
              });

            // Return model invocation data
            if (user?.role === UserRole.ADMIN) SpanDao.getModelInvocationData(runId).then((data) => {
              socket.emit(SocketEvents.server.pushModelInvocationData, data);
            });

            // 返回服务端时间戳给客户端
            if (typeof callback === "function") {
              callback({
                success: true,
                serverTimestamp: serverTimestamp,
              });
            }
          }
        },
      );

      socket.on(
        SocketEvents.client.sendUserInputToServer,
        async (
          requestId: string,
          blocksInput: ContentBlocks,
          structuredInput: Record<string, unknown> | null,
          callback,
        ) => {
          console.debug(
            `[StudioInput] sendUserInputToServer received request=${requestId} user=${user?.username ?? "unknown"}`,
          );
          if (requestId.startsWith(runtimeRequestPrefix)) {
            const runId = requestId.slice(runtimeRequestPrefix.length);
            try {
              const canAccessRun = await this.canUseRun(runId, user);
              if (!canAccessRun) {
                callback?.({
                  success: false,
                  message: "Permission denied",
                });
                return;
              }
              await this.forwardRuntimeBridgeUserInput(
                runId,
                user,
                blocksInput,
                structuredInput,
              );
              callback?.({
                success: true,
                message: "User input forwarded to Runtime bridge",
              });
            } catch (error) {
              if (isAbortError(error)) {
                callback?.({
                  success: true,
                  message: "Model response stopped",
                });
                return;
              }
              console.error(
                "[RuntimeBridge] Failed to send user input:",
                error,
              );
              this.broadcastRuntimeErrorToRunRoom(
                runId,
                user,
                error instanceof Error ? error.message : String(error),
              );
              callback?.({
                success: false,
                message: error instanceof Error ? error.message : String(error),
              });
            }
            return;
          }

          const inputRequest =
            await InputRequestDao.getInputRequestByRequestId(requestId);

          if (!inputRequest) {
            if (typeof callback === "function") {
              callback({
                success: false,
                message: `Input request ${requestId} not found`,
              });
            }
          } else {
            const runId = inputRequest.runId;
            const canAccessRun = await this.canUseRun(runId, user);
            if (!canAccessRun) {
              console.warn(
                `[StudioInput] permission denied request=${requestId} run=${runId} user=${user?.username ?? "unknown"} userId=${user?.id ?? "unknown"}`,
              );
              if (typeof callback === "function") {
                callback({
                  success: false,
                  message: "Permission denied",
                });
              }
              return;
            }
            // Only update status if run is not already DONE
            // This prevents resurrecting finished runs
            const res = await RunDao.getRunData(runId, user);
            if (
              res.runData.status !== Status.DONE &&
              res.inputRequests.length === 0
            ) {
              this.changeRunStatusAndTriggerEvents(runId, Status.RUNNING).catch(
                (error) => {
                  console.error(error);
                  throw error;
                },
              );
            }

            // Emit the input to the python client
            const pythonNamespace = this.io.of("/python");
            const pythonRunRoom = `python-run-${runId}`;
            const targetRoom = pythonNamespace.adapter.rooms.get(pythonRunRoom);

            if (!targetRoom?.size) {
              console.warn(
                `[StudioInput] no python socket found for request=${requestId} run=${runId}; forwarding to Runtime bridge`,
              );
              try {
                await InputRequestDao.deleteInputRequest(requestId);
                this.removeInputRequestToRunRoom(runId, requestId);
                await this.forwardRuntimeBridgeUserInput(
                  runId,
                  user,
                  blocksInput,
                  structuredInput,
                );
                callback?.({
                  success: true,
                  message: "User input forwarded to Runtime bridge",
                });
              } catch (error) {
                if (isAbortError(error)) {
                  callback?.({
                    success: true,
                    message: "Model response stopped",
                  });
                  return;
                }
                console.error("[RuntimeBridge] fallback failed:", error);
                this.broadcastRuntimeErrorToRunRoom(
                  runId,
                  user,
                  error instanceof Error ? error.message : String(error),
                );
                callback?.({
                  success: false,
                  message:
                    error instanceof Error ? error.message : String(error),
                });
              }
              return;
            }

            await InputRequestDao.deleteInputRequest(requestId);
            this.removeInputRequestToRunRoom(runId, requestId);

            const target = targetRoom?.size
              ? pythonNamespace.to(pythonRunRoom)
              : pythonNamespace;
            const currentUser = user
              ? await this.withCurrentResourceAccess(user)
              : user;
            const resourceUser = await this.resolveRuntimeBridgeResourceUser(
              runId,
              currentUser,
              structuredInput,
            );
            if (currentUser?.role === UserRole.ADMIN && !resourceUser?.group?.id) {
              const message = "管理员发送 Runtime 输入前必须选择目标用户组";
              this.broadcastRuntimeErrorToRunRoom(runId, user, message);
              callback?.({
                success: false,
                message,
              });
              return;
            }
            const medflowOwnerUserId = currentUser?.id

              ? `auth:${currentUser.id}`

              : currentUser?.username;

            const medflowOwnerAliases = Array.from(

              new Set([currentUser?.id, currentUser?.username].filter(Boolean) as string[]),

            );

            const trustedStructuredInput = {

              ...(structuredInput || {}),

              __medflowUsername: currentUser?.username,

              __medflowUserId: currentUser?.id,

              __medflowOwnerUserId: medflowOwnerUserId,

              __medflowOwnerAliases: medflowOwnerAliases,

              __medflowResourceGroupId: resourceUser?.group?.id,

            };
            console.debug(
              `[StudioInput] forwarding request=${requestId} run=${runId} target=${pythonRunRoom} sockets=${targetRoom.size}`,
            );

            target.emit(
              SocketEvents.server.forwardUserInput,
              requestId,
              blocksInput,
              trustedStructuredInput,
            );
            if (typeof callback === "function") {
              callback({
                success: true,
                message: "User input forwarded",
              });
            }
          }
        },
      );

      socket.on(
        SocketEvents.client.cancelRuntimeResponse,
        async (runId: string, callback) => {
          try {
            const canAccessRun = await this.canUseRun(runId, user);
            if (!canAccessRun) {
              callback?.({
                success: false,
                message: "Permission denied",
              });
              return;
            }

            const requestKey = this.runtimeBridgeRequestKey(runId, user);
            const controller = this.runtimeBridgeControllers.get(requestKey);
            if (!controller) {
              callback?.({
                success: false,
                message: "No running model response to stop",
              });
              return;
            }

            controller.abort();
            this.runtimeBridgeControllers.delete(requestKey);
            await this.broadcastRuntimeStoppedToRunRoom(runId, user);
            callback?.({
              success: true,
              message: "Model response stopped",
            });
          } catch (error) {
            console.error("[RuntimeBridge] cancelRuntimeResponse failed:", error);
            callback?.({
              success: false,
              message: error instanceof Error ? error.message : String(error),
            });
          }
        },
      );

      socket.on(
        SocketEvents.client.resetAgentContext,
        async (
          runId: string,
          payloadOrCallback?:
            | { contextUsername?: string; cancelWorkflows?: boolean }
            | ((response: ResponseBody) => void),
          maybeCallback?: (response: ResponseBody) => void,
        ) => {
          const payload =
            typeof payloadOrCallback === "function"
              ? undefined
              : payloadOrCallback;
          const callback =
            typeof payloadOrCallback === "function"
              ? payloadOrCallback
              : maybeCallback;
          try {
            if (!runId) {
              callback?.({
                success: false,
                message: "Run id is required",
              });
              return;
            }

            const canAccessRun = await this.canUseRun(runId, user);
            if (!canAccessRun) {
              callback?.({
                success: false,
                message: "Permission denied",
              });
              return;
            }

            // 通过 Bridge 的一致性哈希路由重置请求（替代 Socket.IO 广播）
            const bridgeBaseUrl = (
              process.env.AGENT_BRIDGE_BASE_URL || "http://localhost:3100"
            ).replace(/\/+$/, "");

            try {
              // 从 contextUsername（格式：username#chatSessionId）解析旧的 chatSessionId
              const contextUsername = payload?.contextUsername || "";
              const chatSessionId = contextUsername.includes("#")
                ? contextUsername.split("#").slice(1).join("#")
                : "";
              const sessionId = chatSessionId
                ? `${runId}#${chatSessionId}`
                : runId;

              const response = await fetch(`${bridgeBaseUrl}/api/agent/reset`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                  user_id: user?.id,
                  session_id: sessionId,
                  contextUsername: payload?.contextUsername,
                  username: user?.username,
                  cancelWorkflows: Boolean(payload?.cancelWorkflows),
                }),
              });

              if (!response.ok) {
                const errorText = await response.text().catch(() => "");
                console.error(
                  `[StudioInput] resetAgentContext Bridge error: ${response.status} ${errorText}`,
                );
                callback?.({
                  success: false,
                  message: `Agent reset failed: ${response.status} ${errorText}`,
                });
                return;
              }

              const result = (await response.json()) as {
                success?: boolean;
                error?: string;
              };

              if (result.error) {
                callback?.({
                  success: false,
                  message: result.error,
                });
                return;
              }

              callback?.({
                success: true,
                message: "Agent context reset requested",
              });
            } catch (fetchError) {
              console.error(
                "[StudioInput] resetAgentContext fetch failed:",
                fetchError,
              );
              callback?.({
                success: false,
                message:
                  fetchError instanceof Error
                    ? fetchError.message
                    : "Failed to reset agent context",
              });
            }
          } catch (error) {
            console.error("[StudioInput] resetAgentContext failed:", error);
            callback?.({
              success: false,
              message:
                error instanceof Error
                  ? error.message
                  : "Failed to reset agent context",
            });
          }
        },
      );

      socket.on(SocketEvents.client.joinOverviewRoom, async () => {
        if (!user) return;
        const roomName = this.overviewRoom(user);
        socket.join(roomName);
        console.debug(`${socket.id}: joined room: ${roomName}`);

        // Return current overview data
        const res = await this._getOverViewData(user);
        socket.emit(SocketEvents.server.pushOverviewData, res);

        // 推送系统概览数据（不包含 GPU 信息）
        if (user?.role === UserRole.ADMIN) {
          const systemOverviewData = await RunDao.getSystemOverviewStats(user);
          socket.emit(SocketEvents.server.pushSystemOverviewData, {
            ...systemOverviewData,
            onlineUsers: this.getOnlineUsersCount(user),
            serverUptime: this.getServerUptime(),
            gpuInfo: this.getCachedGPUInfo(),
          });
          void this.emitGPUInfoForUser(socket, user);
        } else if (user) {
          socket.emit(SocketEvents.server.pushSystemOverviewData, {
            onlineUsers: 1,
            serverUptime: "",
            messageStats: { today: 0, thisWeek: 0, thisMonth: 0 },
            gpuInfo: await this.getCachedGPUInfoForUser(user),
          });
          void this.emitGPUInfoForUser(socket, user);
        }
      });

      // 处理请求 GPU 信息
      socket.on(SocketEvents.client.requestGPUInfo, async () => {
        await this.emitGPUInfoForUser(socket, user);
      });

      // 处理请求系统概览数据
      socket.on(SocketEvents.client.requestSystemOverviewData, async () => {
        if (user?.role !== UserRole.ADMIN) {
          socket.emit(SocketEvents.server.pushSystemOverviewData, {
            onlineUsers: 1,
            serverUptime: "",
            messageStats: { today: 0, thisWeek: 0, thisMonth: 0 },
            gpuInfo: user
              ? await this.getCachedGPUInfoForUser(user)
              : [],
          });
          return;
        }
        const systemOverviewData = await RunDao.getSystemOverviewStats(user);
        socket.emit(SocketEvents.server.pushSystemOverviewData, {
          ...systemOverviewData,
          onlineUsers: this.getOnlineUsersCount(user),
          serverUptime: this.getServerUptime(),
          gpuInfo: this.getCachedGPUInfo(),
        });
      });

      socket.on(SocketEvents.client.leaveRoom, (room: string) => {
        socket.leave(room);
        console.debug(`${socket.id}: left room: ${room}`);
      });

      socket.on(
        SocketEvents.client.deleteProjects,
        async (projects: string[], callback) => {
          try {
            if (user?.role !== UserRole.ADMIN) {
              callback({
                success: false,
                message: "Administrator permission required",
              });
              return;
            }
            await RunDao.deleteProjects(projects, user);
            callback({
              success: true,
              message: `Success: ${projects.length} project deleted`,
            });
            // Update projectListRoom, overviewRoom,
            this.broadcastOverviewDataToDashboardRoom();
            this.broadcastRunToProjectListRoom();
          } catch (error) {
            callback({
              success: false,
              message: `Error: ${error}`,
            });
          }
        },
      );

      socket.on(
        SocketEvents.client.deleteRuns,
        async (runIds: string[], callback) => {
          try {
            if (user?.role !== UserRole.ADMIN) {
              callback({
                success: false,
                message: "Administrator permission required",
              });
              return;
            }
            const nDelete = await RunDao.deleteRuns(runIds, user);
            callback({
              success: nDelete === runIds.length,
              message: `Deleted ${nDelete} runs`,
            });
            // Update data to overviewRoom, projectRoom
            this.broadcastOverviewDataToDashboardRoom();
            this.broadcastRunToProjectListRoom();
          } catch (error) {
            callback({
              success: false,
              message: `Failed to delete runs: ${error}`,
            });
          }
        },
      );
      socket.on("disconnect", () => {
        console.debug("Client disconnected");
      });
    });
  }

  /*
   * Emit events to the project list room.
   */
  static broadcastRunToProjectListRoom(ownerUserId?: string | null) {
    RunDao.getAllProjects({
      id: "",
      username: "",
      role: UserRole.ADMIN,
      disabled: false,
      mustChangePassword: false,
      createdAt: "",
    })
      .then((projects) => {
        // Push projects to the client
        this.io
          .of("/client")
          .to(`${SocketRoomName.ProjectListRoom}-admin`)
          .emit(SocketEvents.server.pushProjects, projects);
      })
      .catch((error) => {
        console.error(error);
        throw error;
      });

    this.connectedNormalUsers().forEach((user) => {
      this.withCurrentResourceAccess(user)
        .then((currentUser) => RunDao.getAllProjects(currentUser))
        .then((projects) => {
          this.io
            .of("/client")
            .to(`${SocketRoomName.ProjectListRoom}-user-${user.id}`)
            .emit(SocketEvents.server.pushProjects, projects);
        })
        .catch((error) => {
          console.error(error);
          throw error;
        });
    });
  }

  static broadcastRunToProjectRoom(
    project: string,
    ownerUserId?: string | null,
  ) {
    RunDao.getAllProjectRuns(project, {
      id: "",
      username: "",
      role: UserRole.ADMIN,
      disabled: false,
      mustChangePassword: false,
      createdAt: "",
    })
      .then((runs) => {
        // Push runs to the client
        this.io
          .of("/client")
          .to(`project-${project}-admin`)
          .emit(SocketEvents.server.pushRunsData, runs);
      })
      .catch((error) => {
        console.error(error);
        throw error;
      });

    this.connectedNormalUsers().forEach((user) => {
      this.withCurrentResourceAccess(user)
        .then((currentUser) => RunDao.getAllProjectRuns(project, currentUser))
        .then((runs) => {
          this.io
            .of("/client")
            .to(`project-${project}-user-${user.id}`)
            .emit(SocketEvents.server.pushRunsData, runs);
        })
        .catch((error) => {
          console.error(error);
          throw error;
        });
    });
  }

  /*
   * Emit events to the run room.
   */
  static async forwardRuntimeBridgeUserInput(
    runId: string,
    user: SafeAuthUser | null | undefined,
    blocksInput: ContentBlocks,
    structuredInput?: Record<string, unknown> | null,
  ) {
    const currentUser = user
      ? await SocketManager.withCurrentResourceAccess(user)
      : user;
    const resourceUser = await SocketManager.resolveRuntimeBridgeResourceUser(
      runId,
      currentUser,
      structuredInput,
    );
    const runtimeContext = SocketManager.runtimeBridgeContainers(resourceUser);
    const trainingPoolId =
      typeof structuredInput?.__medflowTrainingPoolId === "string"
        ? structuredInput.__medflowTrainingPoolId.trim()
        : typeof structuredInput?.training_pool_id === "string"
          ? structuredInput.training_pool_id.trim()
          : "";
    const rawText = contentBlocksToText(blocksInput).trim();
    const username = currentUser?.username || "anonymous";
    const userId = currentUser?.id || username;
    const message =
      rawText.replace(/^\[[^\]]+\]\s*/, "").trim() || rawText || "你好";
    if (currentUser?.role === UserRole.ADMIN && !runtimeContext.resourceGroupId) {
      throw new Error("管理员发送 Runtime 输入前必须选择目标用户组");
    }
    const contextUsername =
      (structuredInput?.__medflowContextUsername as string) || username;
    const medflowUsername =

      (structuredInput?.__medflowUsername as string) || username;

    const ownerUserId =

      typeof structuredInput?.__medflowOwnerUserId === "string" &&

      structuredInput.__medflowOwnerUserId.trim()

        ? structuredInput.__medflowOwnerUserId.trim()

        : userId

          ? `auth:${userId}`

          : medflowUsername;

    const structuredOwnerAliases = Array.isArray(

      structuredInput?.__medflowOwnerAliases,

    )

      ? (structuredInput.__medflowOwnerAliases as unknown[])

          .map((value) => String(value || "").trim())

          .filter(Boolean)

      : [];

    const ownerAliases = Array.from(

      new Set([userId, medflowUsername, ...structuredOwnerAliases].filter(Boolean)),

    );

    const now = new Date().toISOString();
    const assistantReplyId = createRuntimeId("runtime-assistant");
    const assistantMessageId = createRuntimeId("runtime-message");

    const userReply: Reply = {
      replyId: createRuntimeId("runtime-user"),
      replyName: username,
      replyRole: "user",
      createdAt: now,
      finishedAt: now,
      messages: [
        {
          id: createRuntimeId("runtime-user-message"),
          name: username,
          role: "user",
          content: blocksInput,
          timestamp: now,
          metadata: {
            __medflowContextUsername: contextUsername,
            __medflowUsername: medflowUsername,
            __medflowUserId: userId,
            __medflowOwnerUserId: ownerUserId,
            __medflowOwnerAliases: ownerAliases,
            __medflowResourceGroupId: runtimeContext.resourceGroupId,
            __medflowTrainingPoolId: trainingPoolId || undefined,
          },
        },
      ],
    };

    const buildAssistantReply = (
      text: string,
      finished = false,
      metadata: object = {},
    ): Reply => ({
      replyId: assistantReplyId,
      replyName: "Assistant",
      replyRole: "assistant",
      createdAt: now,
      finishedAt: finished ? new Date().toISOString() : undefined,
      messages: [
        {
          id: assistantMessageId,
          name: `Assistant [${username}]`,
          role: "assistant",
          content: textToContentBlocks(text),
          timestamp: new Date().toISOString(),
          metadata: {
            ...metadata,
            __medflowContextUsername: contextUsername,
            __medflowUsername: medflowUsername,
            __medflowUserId: userId,
            __medflowOwnerUserId: ownerUserId,
            __medflowOwnerAliases: ownerAliases,
            __medflowResourceGroupId: runtimeContext.resourceGroupId,
            __medflowTrainingPoolId: trainingPoolId || undefined,
          },
        },
      ],
    });

    await SocketManager.persistAndBroadcastReply(runId, userReply, currentUser?.id);

    const bridgeBaseUrl = (
      process.env.AGENT_BRIDGE_BASE_URL || "http://localhost:3100"
    ).replace(/\/+$/, "");

    // 让 session_id 包含前端 chatSessionId，实现真正的会话隔离
    const chatSessionId = (structuredInput?.__medflowSessionId as string) || "";
    const sessionId = chatSessionId ? `${runId}#${chatSessionId}` : runId;

    // 查询 run 注册的节点，实现精准路由
    let targetBackendUrl: string | null = null;
    try {
      const runNodeId = await RunDao.getRunNodeId(runId);
      if (runNodeId && runNodeId !== "unknown") {
        const node = resourceNodeRegistry.get(runNodeId);
        targetBackendUrl = node.baseUrl;
      }
    } catch (error) {
      console.warn(
        `[RuntimeBridge] Failed to resolve backend for run ${runId}, fallback to hash routing:`,
        error,
      );
    }

    const fetchHeaders: Record<string, string> = {
      "Content-Type": "application/json",
    };
    if (targetBackendUrl) {
      fetchHeaders["X-Agent-Target-Backend-Url"] = targetBackendUrl;
    }
    const requestKey = SocketManager.runtimeBridgeRequestKey(runId, currentUser);
    SocketManager.runtimeBridgeControllers.get(requestKey)?.abort();
    const controller = new AbortController();
    SocketManager.runtimeBridgeControllers.set(requestKey, controller);

    console.info("[RuntimeBridge] Forward user input with containers", {
      userId,
      userRole: currentUser?.role,
      runId,
      resourceUserId: resourceUser?.id,
      resourceGroupId: runtimeContext.resourceGroupId,
      trainingPoolId,
      trainingContainer: runtimeContext.trainingContainer,
      evaluationContainer: runtimeContext.evaluationContainer,
      grpoContainer: runtimeContext.grpoContainer,
      multinodeTrainingContainer: runtimeContext.multinodeTrainingContainer,
    });
    if (currentUser?.role === UserRole.ADMIN) {
      await SocketManager.auditAdminAction(
        currentUser,
        "admin_runtime_input_forwarded",
        {
          runId,
          resourceGroupId: runtimeContext.resourceGroupId,
          trainingPoolId: trainingPoolId || null,
          multinodeTrainingContainer: runtimeContext.multinodeTrainingContainer,
          targetBackendUrl,
        },
      );
    }

    let response: globalThis.Response;
    try {
      response = await fetch(`${bridgeBaseUrl}/api/agent/process`, {
        method: "POST",
        headers: fetchHeaders,
        signal: controller.signal,
        body: JSON.stringify({
          user_id: userId,

          owner_user_id: ownerUserId,

          owner_aliases: ownerAliases,

          context_username: contextUsername,

          user_role: currentUser?.role,
          session_id: sessionId,
          training_container: runtimeContext.trainingContainer,
          evaluation_container: runtimeContext.evaluationContainer,
          grpo_container: runtimeContext.grpoContainer,
          multinode_training_container: runtimeContext.multinodeTrainingContainer,
          resource_group_id: runtimeContext.resourceGroupId,
          training_pool_id: trainingPoolId || undefined,
          message,
          input: [
            {
              role: "user",
              type: "message",
              content: [
                {
                  type: "text",
                  text: message,
                },
              ],
            },
          ],
        }),
      });
    } catch (error) {
      if (SocketManager.runtimeBridgeControllers.get(requestKey) === controller) {
        SocketManager.runtimeBridgeControllers.delete(requestKey);
      }
      throw error;
    }

    if (!response.ok || !response.body) {
      const errorText = await response.text().catch(() => "");
      if (SocketManager.runtimeBridgeControllers.get(requestKey) === controller) {
        SocketManager.runtimeBridgeControllers.delete(requestKey);
      }
      throw new Error(
        `Runtime bridge request failed: ${response.status} ${errorText}`,
      );
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let assistantContentText = "";
    let assistantMetadata: object = {};
    let emittedFinishedReply = false;
    let runtimeErrorText = "";
    let sawWorkflowProgress = false;
    const isWorkflowControlRequest = isWorkflowControlMessage(message);

    const emitAssistant = async (finished = false) => {
      const protocolMessage = extractRuntimeProtocolMessage(
        assistantMetadata as Record<string, any>,
      );
      const displayText =
        runtimeErrorText || protocolMessage || assistantContentText;

      await SocketManager.persistAndBroadcastReply(
        runId,
        buildAssistantReply(displayText, finished, assistantMetadata),
        user?.id,
      );
      emittedFinishedReply = emittedFinishedReply || finished;
    };

    const processSseEvent = async (event: string) => {
      const dataLines = event
        .split(/\r?\n/)
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trim())
        .filter((line) => line && line !== "[DONE]");

      for (const dataLine of dataLines) {
        let payload: Record<string, any>;
        try {
          payload = JSON.parse(dataLine);
        } catch {
          console.warn(
            "[RuntimeBridge] Failed to parse SSE payload:",
            dataLine,
          );
          continue;
        }

        if (payload.object === "error") {
          runtimeErrorText = formatRuntimeErrorMessage(
            extractRuntimeErrorMessage(payload),
          );
          assistantMetadata = {
            ...(assistantMetadata as Record<string, any>),
            runtimeBridgeError: true,
            runtimeErrorPayload: payload,
          };
          await emitAssistant(true);
          continue;
        }

        if (
          payload.object === "response" &&
          ["failed", "canceled", "cancelled"].includes(
            String(payload.status || ""),
          )
        ) {
          runtimeErrorText = formatRuntimeErrorMessage(
            extractRuntimeErrorMessage(payload),
          );
          assistantMetadata = {
            ...(assistantMetadata as Record<string, any>),
            runtimeBridgeError: true,
            runtimeErrorPayload: payload,
          };
          await emitAssistant(true);
          continue;
        }

        const eventText = extractRuntimeEventText(payload);
        if (payload.object === "content" && typeof payload.text === "string") {
          assistantContentText += payload.text;
          if (assistantContentText.includes("一键工作流")) {
            sawWorkflowProgress = true;
          }
          await emitAssistant(false);
          continue;
        }

        if (payload.object === "message") {
          assistantMetadata = payload.metadata || {};
          const hasWorkflowStatusMessage = isRuntimeWorkflowStatusMessage(
            assistantMetadata as Record<string, any>,
          );
          if (isRuntimeWorkflowProtocol(assistantMetadata as Record<string, any>)) {
            sawWorkflowProgress = true;
          }
          if (
            eventText &&
            !extractRuntimeProtocolMessage(
              assistantMetadata as Record<string, any>,
            )
          ) {
            assistantContentText = eventText;
          }
          if (eventText.includes("一键工作流")) {
            sawWorkflowProgress = true;
          }
          await emitAssistant(
            payload.status === "completed" || hasWorkflowStatusMessage,
          );
        }
      }
    };

    try {
      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) {
            break;
          }

          buffer += decoder.decode(value, { stream: true });
          const events = buffer.split(/\r?\n\r?\n/);
          buffer = events.pop() || "";

          for (const event of events) {
            await processSseEvent(event);
          }
        }
        if (buffer.trim()) {
          await processSseEvent(buffer);
        }
      } catch (error) {
        const errorMessage =
          error instanceof Error ? error.message : String(error);
        const isTerminatedStream = errorMessage.toLowerCase().includes("terminated");
        const hasAssistantProgress =
          assistantContentText.trim().length > 0 ||
          isRuntimeWorkflowProtocol(assistantMetadata as Record<string, any>) ||
          sawWorkflowProgress;
        if (isWorkflowControlRequest && isTerminatedStream && hasAssistantProgress) {
          console.warn(
            "[RuntimeBridge] Suppressed terminated stream after workflow progress",
            { runId, errorMessage },
          );
          if (!emittedFinishedReply) {
            await emitAssistant(true);
          }
          return;
        }
        runtimeErrorText = formatRuntimeErrorMessage(
          `Runtime stream interrupted: ${errorMessage}`,
        );
        assistantMetadata = {
          ...(assistantMetadata as Record<string, any>),
          runtimeBridgeError: true,
          runtimeErrorPayload: {
            object: "error",
            message: errorMessage,
          },
        };
        await emitAssistant(true);
      }
    } finally {
      reader.releaseLock();
      if (SocketManager.runtimeBridgeControllers.get(requestKey) === controller) {
        SocketManager.runtimeBridgeControllers.delete(requestKey);
      }
    }

    if (!emittedFinishedReply) {
      if (!assistantContentText.trim() && !extractRuntimeProtocolMessage(
        assistantMetadata as Record<string, any>,
      )) {
        runtimeErrorText = formatRuntimeErrorMessage("");
        assistantMetadata = {
          ...(assistantMetadata as Record<string, any>),
          runtimeBridgeError: true,
          runtimeErrorPayload: {
            object: "error",
            message: "Runtime stream ended without assistant content.",
          },
        };
      }
      await emitAssistant(true);
    }
  }

  static broadcastRuntimeErrorToRunRoom(
    runId: string,
    user: SafeAuthUser | null | undefined,
    errorMessage: string,
  ) {
    const now = new Date().toISOString();
    const username = user?.username || "anonymous";
    this.broadcastMessageToRunRoom(runId, {
      replyId: createRuntimeId("runtime-error"),
      replyName: "Assistant",
      replyRole: "assistant",
      createdAt: now,
      finishedAt: now,
      messages: [
        {
          id: createRuntimeId("runtime-error-message"),
          name: `Assistant [${username}]`,
          role: "assistant",
          content: textToContentBlocks(`对话连接异常：${errorMessage}`),
          timestamp: now,
          metadata: {
            runtimeBridgeError: true,
          },
        },
      ],
    });
  }

  static async broadcastRuntimeStoppedToRunRoom(
    runId: string,
    user: SafeAuthUser | null | undefined,
  ) {
    const now = new Date().toISOString();
    const username = user?.username || "anonymous";
    await this.persistAndBroadcastReply(
      runId,
      {
        replyId: createRuntimeId("runtime-stopped"),
        replyName: "Assistant",
        replyRole: "assistant",
        createdAt: now,
        finishedAt: now,
        messages: [
          {
            id: createRuntimeId("runtime-stopped-message"),
            name: `Assistant [${username}]`,
            role: "assistant",
            content: textToContentBlocks("已停止生成。"),
            timestamp: now,
            metadata: {
              runtimeBridgeStopped: true,
              __medflowUsername: username,
            },
          },
        ],
      },
      user?.id,
    );
  }

  static async persistAndBroadcastReply(
    runId: string,
    reply: Reply,
    targetUserId?: string,
  ) {
    try {
      // 1. Ensure reply exists
      if (!(await ReplyDao.doesReplyExist(reply.replyId))) {
        await ReplyDao.saveReply({
          runId,
          replyId: reply.replyId,
          replyRole: reply.replyRole,
          replyName: reply.replyName,
          createdAt: reply.createdAt,
        } as RegisterReplyParams);
      }

      // 2. Persist each message
      if (reply.messages && reply.messages.length > 0) {
        for (const msg of reply.messages) {
          const msgFormData = {
            id: msg.id,
            runId,
            replyId: reply.replyId,
            msg: {
              name: msg.name,
              role: msg.role,
              content: msg.content,
              metadata: msg.metadata,
              timestamp: msg.timestamp,
            },
          } as MessageForm;
          await MessageDao.saveMessage(msgFormData);
        }
      }

      // 3. Broadcast to room
      const resolvedTargetUserId =
        targetUserId ?? (await this.resolveReplyTargetUserId(reply));
      await this.broadcastMessageToRunRoom(runId, reply, resolvedTargetUserId);
    } catch (error) {
      console.error(
        `[SocketManager] Failed to persistAndBroadcastReply for run ${runId}:`,
        error,
      );
    }
  }

  private static async resolveReplyTargetUserId(reply: Reply) {
    for (const message of reply.messages ?? []) {
      const metadata = message.metadata as Record<string, unknown> | undefined;
      const rawUsername =
        (typeof metadata?.__medflowUsername === "string"
          ? metadata.__medflowUsername
          : undefined) ||
        (typeof metadata?.__medflowContextUsername === "string"
          ? metadata.__medflowContextUsername
          : undefined);
      const username = rawUsername?.split("#", 1)[0]?.trim();
      if (username) {
        return (await AuthDao.getUserByUsername(username))?.id;
      }
      const bracketUsername =
        message.name?.match(/\[([^\]#]+)(?:#[^\]]+)?\]/)?.[1]?.trim() ||
        reply.replyName?.match(/\[([^\]#]+)(?:#[^\]]+)?\]/)?.[1]?.trim();
      if (bracketUsername) {
        return (await AuthDao.getUserByUsername(bracketUsername))?.id;
      }
    }
    return undefined;
  }

  static async broadcastMessageToRunRoom(
    runId: string,
    reply: Reply,
    targetUserId?: string,
  ) {
    const sockets = await this.io
      .of("/client")
      .in(`run-${runId}`)
      .fetchSockets();
    for (const socket of sockets) {
      const socketUser = this.getSocketUser(socket);
      if (
        socketUser?.role === UserRole.ADMIN ||
        (targetUserId && socketUser?.id === targetUserId)
      ) {
        socket.emit(SocketEvents.server.pushMessages, [reply] as Reply[]);
      }
    }
  }

  static broadcastSpanDataToRunRoom(spanDataArray: SpanData[]) {
    // Group spans by runId
    const groupedSpans: Record<string, SpanData[]> = {};
    spanDataArray.forEach((spanData) => {
      if (!groupedSpans[spanData.conversationId]) {
        groupedSpans[spanData.conversationId] = [];
      }
      groupedSpans[spanData.conversationId].push(spanData);
    });

    // Send grouped spans to each run room
    for (const runId in groupedSpans) {
      void this.emitToRunAdmins(
        runId,
        SocketEvents.server.pushSpans,
        groupedSpans[runId],
      );
      void this.broadcastModelInvocationDataToRunRoom(runId);
    }
  }

  static broadcastInputRequestToRunRoom(
    runId: string,
    inputRequest: InputRequestData,
  ) {
    void this.emitToRunAdmins(
      runId,
      SocketEvents.server.pushInputRequests,
      [inputRequest],
    );
  }

  static removeInputRequestToRunRoom(runId: string, requestId: string) {
    this.io
      .of("/client")
      .to(`run-${runId}`)
      .emit(SocketEvents.server.removeInputRequest, requestId);
  }

  static broadcastRunDataToRunRoom(runId: string, runData: RunData) {
    void this.emitToRunMembers(runId, SocketEvents.server.pushRunData, runData, {
      ...runData,
      project: "",
      name: "MedFlow 智能服务",
      timestamp: "",
      run_dir: "",
      pid: 0,
      nodeId: null,
    });
  }

  private static async emitToRunAdmins(
    runId: string,
    event: string,
    payload: unknown,
  ) {
    const sockets = await this.io.of("/client").in(`run-${runId}`).fetchSockets();
    for (const socket of sockets) {
      if (this.getSocketUser(socket)?.role === UserRole.ADMIN) {
        socket.emit(event, payload);
      }
    }
  }

  private static async emitToRunMembers(
    runId: string,
    event: string,
    adminPayload: unknown,
    userPayload: unknown,
  ) {
    const sockets = await this.io.of("/client").in(`run-${runId}`).fetchSockets();
    for (const socket of sockets) {
      socket.emit(
        event,
        this.getSocketUser(socket)?.role === UserRole.ADMIN
          ? adminPayload
          : userPayload,
      );
    }
  }

  static clearInputRequestsToRunRoom(runId: string) {
    this.io
      .of("/client")
      .to(`run-${runId}`)
      .emit(SocketEvents.server.clearInputRequests);
  }

  static async changeRunStatusAndTriggerEvents(
    runId: string,
    newStatus: Status,
  ) {
    const runExist = await RunDao.doesRunExist(runId, this.systemAdminUser());

    if (runExist) {
      // Update the run status to "finished"
      await RunDao.changeRunStatus(runId, newStatus);

      // Find the project by runId
      const res = await RunDao.getRunData(runId, this.systemAdminUser());
      const project = res.runData.project;
      const ownerUserId = res.runData.ownerUserId;

      // Broadcast projects to all clients in the ProjectList room
      this.broadcastRunToProjectListRoom(ownerUserId);
      // Broadcast runs to all clients in the project room
      this.broadcastRunToProjectRoom(project, ownerUserId);
      // Broadcast run data to all clients in the run room
      this.broadcastRunDataToRunRoom(runId, res.runData);
      this.broadcastOverviewDataToDashboardRoom(ownerUserId);

      if (newStatus === Status.DONE) {
        // Clear the input requests for all clients in the run room
        this.clearInputRequestsToRunRoom(runId);
      }
    }
  }

  static broadcastOverviewDataToDashboardRoom(ownerUserId?: string | null) {
    this._getOverViewData({
      id: "",
      username: "",
      role: UserRole.ADMIN,
      disabled: false,
      mustChangePassword: false,
      createdAt: "",
    })
      .then((res) => {
        this.io
          .of("/client")
          .to(`${SocketRoomName.OverviewRoom}-admin`)
          .emit(SocketEvents.server.pushOverviewData, res);
      })
      .catch((error) => {
        console.error(error);
        throw error;
      });

    this.connectedNormalUsers().forEach((user) => {
      this.withCurrentResourceAccess(user)
        .then((currentUser) => this._getOverViewData(currentUser))
        .then((res) => {
          this.io
            .of("/client")
            .to(`${SocketRoomName.OverviewRoom}-user-${user.id}`)
            .emit(SocketEvents.server.pushOverviewData, res);
        })
        .catch((error) => {
          console.error(error);
          throw error;
        });
    });
  }

  static broadcastOnlineUsersCount() {
    this.io.of("/client").sockets.forEach((socket) => {
      const user = this.getSocketUser(socket);
      socket.emit(SocketEvents.server.pushOnlineUsersCount, {
        count: this.getOnlineUsersCount(user),
      });
    });
  }

  static forceLogoutUser(userId: string) {
    this.io.of("/client").sockets.forEach((socket) => {
      const user = this.getSocketUser(socket);
      if (user?.id === userId) {
        socket.emit(SocketEvents.server.forceLogout, {
          reason: "session_revoked",
        });
        setTimeout(() => socket.disconnect(true), 250);
      }
    });
    this.broadcastOnlineUsersCount();
  }

  static async _getOverViewData(user?: SafeAuthUser | null) {
    const res1 = await RunDao.getRunViewData(user);
    const res2 =
      user?.role === UserRole.ADMIN
        ? await SpanDao.getModelInvocationViewData()
        : {};

    return {
      ...res1,
      ...res2,
    } as OverviewData;
  }

  static broadcastModelInvocationDataToRunRoom(runId: string) {
    SpanDao.getModelInvocationData(runId).then((data) => {
      void this.emitToRunAdmins(
        runId,
        SocketEvents.server.pushModelInvocationData,
        data,
      );
    });
  }
}

interface PythonResult {
  success: boolean;
  data?: string;
  error?: string;
}

export async function runPythonScript(
  pythonEnv: string,
  commands: string[],
): Promise<PythonResult> {
  return new Promise((resolve) => {
    console.debug("The execute command:", pythonEnv, commands);
    const pythonProcess = spawn(pythonEnv, commands, {
      env: {
        ...process.env,
        FORCE_COLOR: "0",
      },
    });

    let output = "";
    let errorOutput = "";

    // 收集标准输出
    pythonProcess.stdout.on("data", (data) => {
      output += data.toString();
    });

    // 收集错误输出
    pythonProcess.stderr.on("data", (data) => {
      errorOutput += data.toString();
    });

    // 进程结束时处理结果
    pythonProcess.on("close", (code) => {
      if (code === 0) {
        resolve({
          success: true,
          data: output.trim(),
        });
      } else {
        resolve({
          success: false,
          error: errorOutput.trim(),
        });
      }
    });
  });
}


