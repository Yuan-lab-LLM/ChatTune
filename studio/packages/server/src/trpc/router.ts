import { initTRPC, TRPCError } from "@trpc/server";
import type { CreateExpressContextOptions } from "@trpc/server/adapters/express";
import { z } from "zod";

import {
  GetTraceListParamsSchema,
  GetTraceParamsSchema,
  GetTraceStatisticParamsSchema,
  InputRequestData,
  // RunData,
  TableData,
  ProjectData,
  TableRequestParamsSchema,
  ResponseBody,
  RegisterReplyParams,
  RegisterReplyParamsSchema,
  RunData,
  Reply,
  Message,
  BlockType,
  ContentBlocks,
  Status,
  EnvironmentCheckResult,
} from "../../../shared/src";
import { RunDao } from "../dao/Run";
import { InputRequestDao } from "../dao/InputRequest";
import { SocketManager } from "./socket";
import { ReplyDao } from "../dao/Reply";
import { SpanDao } from "../dao/Trace";
import { APP_INFO } from "../../../shared/src";
import { ConfigManager } from "../../../shared/src/config/server";

import { GrpoResourceInfo } from "../services/dockerManager";
import { managementCacheService } from "../services/managementCacheService";
import { managementRefreshScheduler } from "../services/managementRefreshScheduler";
import {
  remoteResourceClient,
  resourceNodeRegistry,
} from "../services/resourceNodeService";
import { AuthDao, SafeAuthUser } from "../dao/Auth";
import { UserRole } from "../models/Auth";
import { ChatSessionDao } from "../dao/ChatSession";
import { ResourceAccessService } from "../services/resourceAccessService";
import { TrainingResourceService } from "../services/trainingResourceService";

const textBlock = z.object({
  text: z.string(),
  type: z.literal(BlockType.TEXT),
});

const thinkingBlock = z.object({
  thinking: z.string(),
  type: z.literal(BlockType.THINKING),
});

const base64Source = z.object({
  type: z.literal("base64"),
  media_type: z.string(),
  data: z.string(),
});

const urlSource = z.object({
  type: z.literal("url"),
  url: z.string(),
});

const imageBlock = z.object({
  type: z.literal(BlockType.IMAGE),
  source: z.union([base64Source, urlSource]),
});

const audioBlock = z.object({
  type: z.literal(BlockType.AUDIO),
  source: z.union([base64Source, urlSource]),
});

const videoBlock = z.object({
  type: z.literal(BlockType.VIDEO),
  source: z.union([base64Source, urlSource]),
});

const toolUseBlock = z.object({
  type: z.literal(BlockType.TOOL_USE),
  id: z.string(),
  name: z.string(),
  input: z.record(z.unknown()),
});

const toolResultBlock = z.object({
  type: z.literal(BlockType.TOOL_RESULT),
  id: z.string(),
  name: z.string(),
  output: z.union([
    z.string(),
    z.array(z.union([textBlock, imageBlock, audioBlock])),
  ]),
});

// Define ContentBlock as a union of all possible block types
const contentBlock = z.union([
  textBlock,
  thinkingBlock,
  imageBlock,
  audioBlock,
  videoBlock,
  toolUseBlock,
  toolResultBlock,
]);

// Define ContentBlocks as an array of ContentBlock
const contentBlocks = z.array(contentBlock);

// Define ContentType as a string or ContentBlocks
const contentType = z.union([z.string(), contentBlocks]);

export interface TrpcContext {
  authToken?: string;
  runtimeToken?: string;
  setAuthCookie?: (token: string) => void;
  clearAuthCookie?: () => void;
}

const AUTH_COOKIE_NAME = "medflow_auth_token";

const parseCookies = (header?: string) =>
  Object.fromEntries(
    (header || "")
      .split(";")
      .map((part) => part.trim())
      .filter(Boolean)
      .map((part) => {
        const [rawKey, ...rawValue] = part.split("=");
        return [decodeURIComponent(rawKey), decodeURIComponent(rawValue.join("="))];
      }),
  );

const serializeAuthCookie = (token: string, maxAgeSeconds: number) => {
  const secure = process.env.MEDFLOW_AUTH_COOKIE_SECURE === "true";
  return [
    `${AUTH_COOKIE_NAME}=${encodeURIComponent(token)}`,
    "Path=/",
    "HttpOnly",
    "SameSite=Lax",
    `Max-Age=${maxAgeSeconds}`,
    ...(secure ? ["Secure"] : []),
  ].join("; ");
};

export const createContext = ({
  req,
  res,
}: CreateExpressContextOptions): TrpcContext => {
  const authorization = req.headers.authorization;
  const authHeader = Array.isArray(authorization)
    ? authorization[0]
    : authorization;
  const cookies = parseCookies(req.headers.cookie);
  const authToken = authHeader?.startsWith("Bearer ")
    ? authHeader.slice("Bearer ".length).trim()
    : cookies[AUTH_COOKIE_NAME];
  const runtimeTokenHeader = req.headers["x-medflow-runtime-token"];
  const runtimeToken = Array.isArray(runtimeTokenHeader)
    ? runtimeTokenHeader[0]
    : runtimeTokenHeader;

  return {
    authToken,
    runtimeToken,
    setAuthCookie: (token: string) => {
      res.setHeader("Set-Cookie", serializeAuthCookie(token, 7 * 24 * 60 * 60));
    },
    clearAuthCookie: () => {
      res.setHeader("Set-Cookie", serializeAuthCookie("", 0));
    },
  };
};

const t = initTRPC.context<TrpcContext>().create();
const forbiddenError = (message: string) =>
  new TRPCError({
    code: "FORBIDDEN",
    message,
  });
const forbiddenMessages = new Set([
  "auth.error.resourceForbidden",
  "auth.error.adminRequired",
  "Permission denied",
  "Administrator permission required",
]);
const throwIfForbidden = (error: unknown) => {
  if (error instanceof TRPCError) {
    throw error;
  }
  if (!(error instanceof Error)) return;
  if (forbiddenMessages.has(error.message)) {
    throw forbiddenError(error.message);
  }
  const authErrorKey = error.message.match(/auth\.error\.[A-Za-z]+/)?.[0];
  if (authErrorKey && forbiddenMessages.has(authErrorKey)) {
    throw forbiddenError(authErrorKey);
  }
};
const getConfiguredRuntimeToken = () =>
  process.env.MEDFLOW_STUDIO_RUNTIME_TOKEN?.trim() ||
  process.env.AGENTSCOPE_STUDIO_RUNTIME_TOKEN?.trim() ||
  "";
const getConfiguredRuntimeNodeTokens = () => {
  const raw = process.env.MEDFLOW_STUDIO_RUNTIME_NODE_TOKENS?.trim();
  if (!raw) return {} as Record<string, string>;
  try {
    const parsed = JSON.parse(raw) as Record<string, unknown>;
    const tokens = Object.fromEntries(
      Object.entries(parsed)
        .map(([nodeId, token]) => [nodeId.trim(), String(token || "").trim()])
        .filter(([nodeId, token]) => nodeId && token),
    ) as Record<string, string>;
    const values = Object.values(tokens);
    if (new Set(values).size !== values.length || values.includes(getConfiguredRuntimeToken())) {
      console.error(
        "Runtime node tokens must be unique and different from MEDFLOW_STUDIO_RUNTIME_TOKEN.",
      );
      return {} as Record<string, string>;
    }
    return tokens;
  } catch {
    console.error("MEDFLOW_STUDIO_RUNTIME_NODE_TOKENS must be a JSON object.");
    return {} as Record<string, string>;
  }
};

const isRuntimeAuthorized = (ctx: TrpcContext) => {
  const expectedToken = getConfiguredRuntimeToken();
  if (!expectedToken) {
    console.error(
      "MEDFLOW_STUDIO_RUNTIME_TOKEN is not configured; runtime write endpoints are disabled.",
    );
    return false;
  }

  return Boolean(ctx.runtimeToken && ctx.runtimeToken === expectedToken);
};
const isRuntimeNodeAuthorized = (ctx: TrpcContext) => {
  const nodeTokens = Object.values(getConfiguredRuntimeNodeTokens());
  return Boolean(ctx.runtimeToken && nodeTokens.includes(ctx.runtimeToken));
};
const assertRuntimeNodeAuthorized = (ctx: TrpcContext, runtimeNodeId: string) => {
  const nodeTokens = getConfiguredRuntimeNodeTokens();
  const expected = nodeTokens[runtimeNodeId];
  if (!expected) {
    throw new TRPCError({
      code: "FORBIDDEN",
      message: `Runtime node token is not configured for ${runtimeNodeId}`,
    });
  }
  if (ctx.runtimeToken !== expected) {
    throw new TRPCError({
      code: "FORBIDDEN",
      message: "Runtime token does not match runtimeNodeId",
    });
  }
};

const passwordChangeAllowedProcedures = new Set([
  "getCurrentUser",
  "changePassword",
]);

const protectedProcedure = t.procedure.use(async ({ ctx, path, next }) => {
  const user = await AuthDao.getUserByToken(ctx.authToken);

  if (!user) {
    throw new TRPCError({
      code: "UNAUTHORIZED",
      message: "auth.error.loginRequired",
    });
  }

  if (user.mustChangePassword && !passwordChangeAllowedProcedures.has(path)) {
    throw new TRPCError({
      code: "FORBIDDEN",
      message: "auth.error.passwordChangeRequired",
    });
  }

  return next({
    ctx: {
      ...ctx,
      user,
    },
  });
});

const getDefaultResourceNodeId = () =>
  resourceNodeRegistry.list()[0]?.id || "local";
const resourceNodeIdSchema = z
  .string()
  .optional()
  .transform((value) => value?.trim() || getDefaultResourceNodeId());

const resolveResourceNodeIdForUser = async (
  user: SafeAuthUser,
  requestedNodeId?: string | null,
) => {
  if (user.role === UserRole.ADMIN) {
    return requestedNodeId?.trim() || getDefaultResourceNodeId();
  }
  const nodeId =
    user.assignedNodeId || (await ResourceAccessService.getAssignedNodeId(user.id));
  if (!nodeId) {
    throw forbiddenError("auth.error.resourceForbidden");
  }
  const requested = requestedNodeId?.trim();
  if (requested && requested !== "all" && requested !== nodeId) {
    throw forbiddenError("auth.error.resourceForbidden");
  }
  return nodeId;
};

const resolveSingleResourceNodeIdForUser = async (
  user: SafeAuthUser,
  requestedNodeId?: string | null,
) => {
  const nodeId = await resolveResourceNodeIdForUser(user, requestedNodeId);
  if (nodeId === "all") {
    throw new Error("请选择具体节点后再执行该操作");
  }
  return nodeId;
};

const resolveMonitoringRunNodeIdForUser = async (
  user: SafeAuthUser,
  runId?: string | null,
  requestedNodeId?: string | null,
) => {
  if (user.role === UserRole.ADMIN) {
    const storedRunNodeId = runId ? await RunDao.getRunNodeId(runId) : null;
    const runNodeId =
      storedRunNodeId && storedRunNodeId !== "unknown"
        ? storedRunNodeId
        : null;
    return resolveSingleResourceNodeIdForUser(
      user,
      requestedNodeId || runNodeId,
    );
  }

  if (!runId?.trim()) {
    throw forbiddenError("auth.error.resourceForbidden");
  }
  const scopedUser = {
    ...user,
    assignedNodeId:
      user.assignedNodeId || (await ResourceAccessService.getAssignedNodeId(user.id)),
  };
  const visibleRunNodeId = await RunDao.getVisibleRunNodeId(
    runId.trim(),
    scopedUser,
  );
  if (!visibleRunNodeId || visibleRunNodeId === "unknown") {
    throw forbiddenError("auth.error.resourceForbidden");
  }
  const requested = requestedNodeId?.trim();
  if (requested && requested !== visibleRunNodeId) {
    throw forbiddenError("auth.error.resourceForbidden");
  }
  return resolveSingleResourceNodeIdForUser(scopedUser, visibleRunNodeId);
};

const getRecordValue = (value: unknown, key: string): unknown =>
  value && typeof value === "object"
    ? (value as Record<string, unknown>)[key]
    : undefined;

const getWorkflowTrainMetricsHint = (
  workflowStatus: Record<string, unknown>,
) => {
  const stages = getRecordValue(workflowStatus, "stages");
  const trainStage = getRecordValue(stages, "train");
  const trainMetrics = getRecordValue(trainStage, "metrics");
  const currentStage = getRecordValue(workflowStatus, "currentStage");
  const currentStageData =
    typeof currentStage === "string"
      ? getRecordValue(stages, currentStage)
      : undefined;
  const currentMetrics = getRecordValue(currentStageData, "metrics");
  const firstString = (...values: unknown[]) =>
    values
      .map((value) =>
        value === undefined || value === null ? "" : String(value).trim(),
      )
      .find(Boolean);

  return {
    container: firstString(
      getRecordValue(currentStageData, "container"),
      getRecordValue(currentStageData, "container_name"),
      getRecordValue(currentMetrics, "container_name"),
      getRecordValue(currentMetrics, "container"),
      getRecordValue(trainStage, "container"),
      getRecordValue(trainStage, "container_name"),
      getRecordValue(trainMetrics, "container_name"),
      getRecordValue(trainMetrics, "container"),
      getRecordValue(workflowStatus, "container"),
      getRecordValue(workflowStatus, "container_name"),
    ),
    pid: firstString(
      getRecordValue(currentStageData, "pid"),
      getRecordValue(currentMetrics, "pid"),
      getRecordValue(trainStage, "pid"),
      getRecordValue(trainMetrics, "pid"),
      getRecordValue(workflowStatus, "pid"),
    ),
    trainType: firstString(
      getRecordValue(currentStageData, "trainType"),
      getRecordValue(currentStageData, "train_type"),
      getRecordValue(currentMetrics, "train_type"),
      getRecordValue(currentMetrics, "trainType"),
      getRecordValue(trainStage, "trainType"),
      getRecordValue(trainStage, "train_type"),
      getRecordValue(trainMetrics, "train_type"),
      getRecordValue(trainMetrics, "trainType"),
      getRecordValue(workflowStatus, "trainType"),
      getRecordValue(workflowStatus, "train_type"),
      getRecordValue(workflowStatus, "trainTypeText"),
    ),
  };
};

const probeResourceNode = async (nodeId: string) => {
  await remoteResourceClient.request(nodeId, "health");
};

const aggregateSnapshots = async <T>(
  bizType: "dataset" | "model" | "medicalTest" | "evaluationResult",
  container: string,
  refresh: boolean,
  user?: SafeAuthUser,
  requestedNodeId: string = "all",
  includeConfiguredGroupTargets: boolean = true,
) => {
  const visibleNodeIds = user
    ? await ResourceAccessService.getVisibleNodeIds(user)
    : null;
  const nodes = resourceNodeRegistry
    .list()
    .filter(
      (node) =>
        (!visibleNodeIds || visibleNodeIds.has(node.id)) &&
        (requestedNodeId === "all" || node.id === requestedNodeId),
    );
  const isTrainingResource = bizType === "dataset" || bizType === "model";
  const configuredGroupTargets =
    user?.role === UserRole.ADMIN && includeConfiguredGroupTargets
      ? (await ResourceAccessService.listGroups())
          .map((group) => ({
            node: nodes.find((node) => node.id === group.nodeId),
            containerName: isTrainingResource
              ? group.defaultContainerName
              : group.defaultEvaluateContainerName,
          }))
          .filter(
            (
              target,
            ): target is {
              node: (typeof nodes)[number];
              containerName: string;
            } => Boolean(target.node && target.containerName),
          )
      : [];
  const targets =
    user?.role === UserRole.ADMIN
      ? [
          ...nodes.map((node) => ({ node, containerName: container })),
          ...configuredGroupTargets,
        ]
          .filter(
            (target, index, allTargets) =>
              allTargets.findIndex(
                (candidate) =>
                  candidate.node.id === target.node.id &&
                  candidate.containerName === target.containerName,
              ) === index,
          )
      : nodes.map((node) => ({
          node,
          containerName: container || node.defaultContainer,
        }));
  const results = await Promise.all(
    targets.map(async ({ node, containerName }) => {
      try {
        const snapshot = refresh
          ? await managementCacheService
              .refreshCache(
                bizType,
                node.id,
                containerName,
                "manual",
              )
              .then(() =>
                managementCacheService.getSnapshot(
                  bizType,
                  node.id,
                  containerName,
                ),
              )
          : await managementCacheService.getSnapshotOrBootstrap(
              bizType,
              node.id,
              containerName,
            );
        return {
          ...snapshot,
          items: snapshot.items.map((item) => ({
            ...item,
            nodeId: node.id,
            nodeName: node.name,
            containerName,
          })),
          nodeId: node.id,
          nodeName: node.name,
          containerName,
          status: "online" as const,
        };
      } catch (error) {
        throwIfForbidden(error);
        return {
          items: [] as T[],
          meta: null,
          nodeId: node.id,
          nodeName: node.name,
          containerName,
          status: "offline" as const,
          error: error instanceof Error ? error.message : String(error),
        };
      }
    }),
  );
  const items = results.flatMap((result) => result.items as unknown[]) as T[];
  const visibleItems = user
    ? await ResourceAccessService.filterResourceItems(
        user,
        bizType,
        container,
        items as Array<{ nodeId?: string; name?: string; filename?: string; jobId?: string }>,
      )
    : items;
  return {
    items: visibleItems as T[],
    meta: {
      bizType,
      nodeId: requestedNodeId,
      containerName: container,
      containerNames: [...new Set(targets.map((target) => target.containerName))],
      itemCount: results.reduce(
        (count, result) => count + result.items.length,
        0,
      ),
      updatedAt: new Date().toISOString(),
      lastRefreshStartedAt: null,
      lastRefreshFinishedAt: new Date().toISOString(),
      lastRefreshStatus: results.some((result) => result.status === "offline")
        ? ("failed" as const)
        : ("success" as const),
      isRefreshing: false,
      lastErrorMessage:
        results
          .filter((result) => result.error)
          .map((result) => result.error)
          .join("; ") || null,
    },
    nodes: results,
  };
};

const resolveAdminGroupResourceTarget = async (
  groupId: string | undefined,
  bizType: "dataset" | "model" | "medicalTest",
) => {
  const normalizedGroupId = groupId?.trim();
  if (!normalizedGroupId) return null;

  const group = (await ResourceAccessService.listGroups()).find(
    (item) => item.id === normalizedGroupId,
  );
  if (!group) {
    throw new Error("auth.error.groupNotFound");
  }
  if (!group.nodeId) {
    throw new Error("auth.error.resourceForbidden");
  }

  const container =
    bizType === "medicalTest"
      ? group.defaultEvaluateContainerName?.trim()
      : group.defaultContainerName?.trim();
  if (!container) {
    throw new Error("auth.error.containerNotFound");
  }

  return {
    nodeId: group.nodeId,
    container,
  };
};

const sanitizeResourceItemsForUser = <T extends object>(
  user: SafeAuthUser,
  bizType: "dataset" | "model" | "medicalTest" | "evaluationResult",
  items: T[],
): T[] => {
  if (user.role === UserRole.ADMIN) return items;
  return items.map((source) => {
    const item = source as Record<string, unknown>;
    if (bizType === "dataset") {
      return {
        name: item.name,
        type: item.type ?? "",
        path: item.path,
        description: item.description,
        files: item.files,
        filePreviews: item.filePreviews,
        sampleContent: item.sampleContent,
        fileName: item.fileName,
        size: item.size,
        createdAt: item.createdAt,
        nodeId: item.nodeId,
        nodeName: item.nodeName,
        containerName: item.containerName,
        canDelete: item.canDelete === true,
        available: true,
      };
    }
    if (bizType === "model") {
      return {
        name: item.name,
        type: item.type ?? "",
        path: item.path,
        description: item.description,
        size: item.size,
        createdAt: item.createdAt,
        checkpoints: item.checkpoints,
        merged: item.merged,
        nodeId: item.nodeId,
        nodeName: item.nodeName,
        containerName: item.containerName,
        canDelete: item.canDelete === true,
        available: true,
      };
    }
    if (bizType === "medicalTest") {
      return {
        filename: item.filename,
        type: item.type ?? "",
        size: item.size ?? "",
        description: item.description ?? "",
        category: item.category,
        nodeId: item.nodeId,
        nodeName: item.nodeName,
        containerName: item.containerName,
        canDelete: item.canDelete === true,
        available: true,
      };
    }
    return {
      jobId: item.jobId,
      model: item.model,
      dataset: item.dataset,
      status: item.status,
      accuracy: item.accuracy ?? 0,
      avgF1: item.avgF1 ?? 0,
      totalScore: item.totalScore,
      startTime: item.startTime ?? "",
      endTime: item.endTime,
      folderPath: item.folderPath ?? "",
      nodeId: item.nodeId,
      nodeName: item.nodeName,
      containerName: item.containerName,
      canDelete: item.canDelete === true,
      available: true,
    };
  }) as unknown as T[];
};
const runtimeProcedure = t.procedure.use(async ({ ctx, next }) => {
  if (!isRuntimeAuthorized(ctx)) {
    throw new TRPCError({
      code: "UNAUTHORIZED",
      message: "Runtime token is required",
    });
  }

  const user = await AuthDao.getUserByToken(ctx.authToken);

  return next({
    ctx: {
      ...ctx,
      user: user ?? null,
    },
  });
});
const runtimeNodeProcedure = t.procedure.use(async ({ ctx, next }) => {
  if (!isRuntimeNodeAuthorized(ctx)) {
    throw new TRPCError({
      code: "UNAUTHORIZED",
      message: "Runtime node token is required",
    });
  }
  return next({ ctx });
});
const adminProcedure = protectedProcedure.use(async ({ ctx, next }) => {
  const user = ctx.user as SafeAuthUser;

  if (user.role !== UserRole.ADMIN) {
    throw new TRPCError({
      code: "FORBIDDEN",
      message: "auth.error.adminRequired",
    });
  }

  return next({ ctx });
});
const assertResourceNodeAccess = async (user: SafeAuthUser, nodeId: string) => {
  if (nodeId === "all") {
    if (user.role === UserRole.ADMIN) return;
    throw new TRPCError({
      code: "FORBIDDEN",
      message: "auth.error.resourceForbidden",
    });
  }
  try {
    await ResourceAccessService.assertNodeAccess(user, nodeId);
  } catch {
    throw new TRPCError({
      code: "FORBIDDEN",
      message: "auth.error.resourceForbidden",
    });
  }
};
const getDefaultContainerName = () =>
  ConfigManager.getInstance().getDefaultContainerName();
const getDefaultEvaluateContainerName = () =>
  ConfigManager.getInstance().getDefaultEvaluateContainerName();
const getDefaultGrpoContainerName = () =>
  ConfigManager.getInstance().getDefaultGrpoContainerName();
const systemAdminUser = (): SafeAuthUser => ({
  id: "__system__",
  username: "system",
  role: UserRole.ADMIN,
  disabled: false,
  mustChangePassword: false,
  createdAt: "",
});

const auditAdminAction = async (
  actor: SafeAuthUser,
  eventType: string,
  details?: Record<string, unknown>,
) => {
  await ResourceAccessService.recordAuditEvent(
    eventType,
    null,
    actor.id,
    {
      actorUsername: actor.username,
      ...(details || {}),
    },
  );
};

export const appRouter = t.router({
  queryResourceNodes: protectedProcedure.query(async ({ ctx }) => {
    const user = ctx.user as SafeAuthUser;
    const visibleNodeIds = await ResourceAccessService.getVisibleNodeIds(user);
    const nodes = await Promise.all(
      resourceNodeRegistry
        .list()
        .filter((node) => !visibleNodeIds || visibleNodeIds.has(node.id))
        .map(async (node) => {
        try {
          await remoteResourceClient.request(node.id, "health");
          return { id: node.id, name: node.name, status: "online" as const };
        } catch (error) {
          throwIfForbidden(error);
          return {
            id: node.id,
            name: node.name,
            status: "offline" as const,
            error: error instanceof Error ? error.message : String(error),
          };
        }
      }),
    );
    return {
      success: true,
      message: "Resource nodes retrieved",
      data:
        user.role === UserRole.ADMIN
          ? nodes
          : nodes.map((node) => ({
              id: node.id,
              name: "可用服务",
              status: node.status,
            })),
    };
  }),
  login: t.procedure
    .input(
      z.object({
        username: z.string().trim().min(1, "auth.error.usernameRequired"),
        password: z.string().min(1, "auth.error.passwordRequired"),
      }),
    )
    .mutation(async ({ ctx, input }) => {
      let result;
      try {
        result = await AuthDao.login(input.username, input.password);
      } catch (error) {
        throwIfForbidden(error);
        throw new TRPCError({
          code: "UNAUTHORIZED",
          message:
            error instanceof Error
              ? error.message
              : "auth.error.invalidCredentials",
        });
      }

      if (!result) {
        throw new TRPCError({
          code: "UNAUTHORIZED",
          message: "auth.error.invalidCredentials",
        });
      }

      ctx.setAuthCookie?.(result.token);

      return {
        success: true,
        message: "登录成功",
        data: { user: result.user },
      };
    }),

  logout: t.procedure.mutation(async ({ ctx }) => {
    await AuthDao.logout(ctx.authToken);
    ctx.clearAuthCookie?.();

    return {
      success: true,
      message: "已退出登录",
    };
  }),

  getCurrentUser: protectedProcedure.query(({ ctx }) => {
    return ResourceAccessService.getGroupForUser((ctx.user as SafeAuthUser).id).then((group) => ({
      success: true,
      message: "Current user retrieved successfully",
      data: { ...ctx.user, group },
    }));
  }),

  getChatSession: protectedProcedure
    .input(z.object({ runId: z.string().trim().min(1) }))
    .query(async ({ ctx, input }) => {
      const user = ctx.user as SafeAuthUser;
      if (
        user.role !== UserRole.ADMIN &&
        !(await RunDao.isSharedRunnableRun(input.runId))
      ) {
        throw new TRPCError({ code: "FORBIDDEN", message: "Service unavailable" });
      }
      const session = await ChatSessionDao.getOrCreate(
        user.id,
        input.runId,
      );
      return {
        success: true,
        message: "Chat session retrieved",
        data: { sessionId: session.sessionId, clearedAt: session.clearedAt },
      };
    }),

  resetChatSession: protectedProcedure
    .input(z.object({ runId: z.string().trim().min(1) }))
    .mutation(async ({ ctx, input }) => {
      const user = ctx.user as SafeAuthUser;
      if (
        user.role !== UserRole.ADMIN &&
        !(await RunDao.isSharedRunnableRun(input.runId))
      ) {
        throw new TRPCError({ code: "FORBIDDEN", message: "Service unavailable" });
      }
      const session = await ChatSessionDao.reset(
        user.id,
        input.runId,
      );
      return {
        success: true,
        message: "Chat session reset",
        data: { sessionId: session.sessionId, clearedAt: session.clearedAt },
      };
    }),

  changePassword: protectedProcedure
    .input(
      z.object({
        currentPassword: z
          .string()
          .min(1, "auth.error.currentPasswordRequired"),
        newPassword: z.string().min(6, "auth.error.newPasswordTooShort"),
      }),
    )
    .mutation(async ({ ctx, input }) => {
      try {
        await AuthDao.changePassword(
          (ctx.user as SafeAuthUser).id,
          input.currentPassword,
          input.newPassword,
        );

        return {
          success: true,
          message: "密码已修改，请重新登录",
        };
      } catch (error) {
        throwIfForbidden(error);
        throw new TRPCError({
          code: "BAD_REQUEST",
          message:
            error instanceof Error
              ? error.message
              : "Failed to change password",
        });
      }
    }),

  listUsers: adminProcedure.query(async () => {
    const users = await AuthDao.listUsers();

    return {
      success: true,
      message: "Users retrieved successfully",
      data: users,
    };
  }),

  createUser: adminProcedure
    .input(
      z.object({
        username: z
          .string()
          .trim()
          .min(3, "auth.error.usernameTooShort")
          .max(32, "auth.error.usernameTooLong")
          .regex(/^[a-zA-Z0-9_-]+$/, "auth.error.usernameInvalid"),
        password: z.string().min(6, "auth.error.passwordTooShort"),
        role: z.nativeEnum(UserRole).default(UserRole.USER),
        groupId: z.string().optional(),
      }),
    )
    .mutation(async ({ ctx, input }) => {
      try {
        if (input.role === UserRole.USER) {
          if (!input.groupId) throw new Error("auth.error.groupRequired");
          await ResourceAccessService.assertGroupExists(input.groupId);
        }
        const user = await AuthDao.createUser(
          input.username,
          input.password,
          input.role,
          (ctx.user as SafeAuthUser).id,
        );
        if (input.role === UserRole.USER) {
          await ResourceAccessService.moveUser(user.id, input.groupId!);
        }
        await auditAdminAction(ctx.user as SafeAuthUser, "admin_user_created", {
          targetUserId: user.id,
          targetUsername: user.username,
          role: input.role,
          groupId: input.groupId || null,
        });

        return {
          success: true,
          message: "用户创建成功",
          data: user,
        };
      } catch (error) {
        throwIfForbidden(error);
        throw new TRPCError({
          code: "BAD_REQUEST",
          message:
            error instanceof Error ? error.message : "Failed to create user",
        });
      }
    }),

  updateUserRole: adminProcedure
    .input(
      z.object({
        userId: z.string().min(1),
        role: z.nativeEnum(UserRole),
      }),
    )
    .mutation(async ({ ctx, input }) => {
      try {
        const user = await AuthDao.updateUserRole(
          input.userId,
          input.role,
          (ctx.user as SafeAuthUser).id,
        );
        if (input.role !== UserRole.ADMIN) {
          SocketManager.forceLogoutUser(input.userId);
        }
        await auditAdminAction(ctx.user as SafeAuthUser, "admin_user_role_updated", {
          targetUserId: input.userId,
          role: input.role,
        });

        return {
          success: true,
          message: "用户角色已更新",
          data: user,
        };
      } catch (error) {
        throwIfForbidden(error);
        throw new TRPCError({
          code: "BAD_REQUEST",
          message:
            error instanceof Error
              ? error.message
              : "Failed to update user role",
        });
      }
    }),

  resetUserPassword: adminProcedure
    .input(
      z.object({
        userId: z.string().min(1),
        newPassword: z.string().min(6, "auth.error.passwordTooShort"),
      }),
    )
    .mutation(async ({ ctx, input }) => {
      try {
        await AuthDao.resetUserPassword(input.userId, input.newPassword);
        SocketManager.forceLogoutUser(input.userId);
        await auditAdminAction(ctx.user as SafeAuthUser, "admin_user_password_reset", {
          targetUserId: input.userId,
        });

        return {
          success: true,
          message: "用户密码已重置",
        };
      } catch (error) {
        throwIfForbidden(error);
        throw new TRPCError({
          code: "BAD_REQUEST",
          message:
            error instanceof Error
              ? error.message
              : "Failed to reset user password",
        });
      }
    }),

  setUserDisabled: adminProcedure
    .input(
      z.object({
        userId: z.string().min(1),
        disabled: z.boolean(),
      }),
    )
    .mutation(async ({ ctx, input }) => {
      try {
        const user = await AuthDao.setUserDisabled(
          input.userId,
          input.disabled,
          (ctx.user as SafeAuthUser).id,
        );
        if (input.disabled) {
          SocketManager.forceLogoutUser(input.userId);
        }
        await auditAdminAction(ctx.user as SafeAuthUser, "admin_user_disabled_set", {
          targetUserId: input.userId,
          disabled: input.disabled,
        });

        return {
          success: true,
          message: input.disabled ? "用户已禁用" : "用户已启用",
          data: user,
        };
      } catch (error) {
        throwIfForbidden(error);
        throw new TRPCError({
          code: "BAD_REQUEST",
          message:
            error instanceof Error
              ? error.message
              : "Failed to update user status",
        });
      }
    }),

  revokeUserSessions: adminProcedure
    .input(
      z.object({
        userId: z.string().min(1),
      }),
    )
    .mutation(async ({ ctx, input }) => {
      try {
        await AuthDao.revokeUserSessions(
          input.userId,
          (ctx.user as SafeAuthUser).id,
        );
        SocketManager.forceLogoutUser(input.userId);
        await auditAdminAction(ctx.user as SafeAuthUser, "admin_user_sessions_revoked", {
          targetUserId: input.userId,
        });

        return {
          success: true,
          message: "用户会话已撤销",
        };
      } catch (error) {
        throwIfForbidden(error);
        throw new TRPCError({
          code: "BAD_REQUEST",
          message:
            error instanceof Error
              ? error.message
              : "Failed to revoke user sessions",
        });
      }
    }),

  deleteUser: adminProcedure
    .input(
      z.object({
        userId: z.string().min(1),
      }),
    )
    .mutation(async ({ ctx, input }) => {
      try {
        await AuthDao.deleteUser(input.userId, (ctx.user as SafeAuthUser).id);
        SocketManager.forceLogoutUser(input.userId);
        await auditAdminAction(ctx.user as SafeAuthUser, "admin_user_deleted", {
          targetUserId: input.userId,
        });

        return {
          success: true,
          message: "用户已删除",
        };
      } catch (error) {
        throwIfForbidden(error);
        throw new TRPCError({
          code: "BAD_REQUEST",
          message:
            error instanceof Error ? error.message : "Failed to delete user",
        });
      }
    }),

  listResourceGroups: adminProcedure.query(async () => ({
    success: true,
    message: "Resource groups retrieved",
    data: await ResourceAccessService.listGroups(),
  })),

  listTrainingResourcePools: adminProcedure.query(async () => ({
    success: true,
    message: "Training resource pools retrieved",
    data: await TrainingResourceService.listPoolsWithRuntimeSummary(),
  })),

  listRunnableTrainingResourcePools: protectedProcedure
    .input(z.object({ groupId: z.string().min(1).optional() }).optional())
    .query(async ({ ctx, input }) => ({
      success: true,
      message: "Runnable training resource pools retrieved",
      data: await TrainingResourceService.listRunnablePoolsForUser(
        ctx.user as SafeAuthUser,
        input?.groupId,
      ),
    })),

  listTrainingReservations: adminProcedure.query(async () => ({
    success: true,
    message: "Training reservations retrieved",
    data: await TrainingResourceService.listReservations(),
  })),

  upsertTrainingResourcePool: adminProcedure
    .input(z.object({ id: z.string().optional(), name: z.string().trim().min(1), description: z.string().optional() }))
    .mutation(async ({ ctx, input }) => {
      const data = await TrainingResourceService.upsertPool(input);
      await auditAdminAction(ctx.user as SafeAuthUser, "admin_training_pool_upserted", {
        poolId: data.id,
        name: input.name,
      });
      return {
        success: true,
        message: "训练资源池已保存",
        data,
      };
    }),

  setTrainingResourcePoolEnabled: adminProcedure
    .input(z.object({ poolId: z.string().min(1), enabled: z.boolean() }))
    .mutation(async ({ ctx, input }) => {
      try {
        const data = await TrainingResourceService.setPoolEnabled(input.poolId, input.enabled);
        await auditAdminAction(ctx.user as SafeAuthUser, "admin_training_pool_status_changed", {
          poolId: input.poolId,
          enabled: input.enabled,
        });
        return {
          success: true,
          message: input.enabled ? "训练资源池已启用" : "训练资源池已禁用",
          data,
        };
      } catch (error) {
        throwIfForbidden(error);
        throw new TRPCError({
          code: "BAD_REQUEST",
          message: error instanceof Error ? error.message : "Failed to change training resource pool status",
        });
      }
    }),

  setTrainingResourcePoolNodes: adminProcedure
    .input(z.object({
      poolId: z.string().min(1),
      nodes: z.array(z.object({
        nodeId: z.string().min(1),
        sshAlias: z.string().min(1),
        trainAddress: z.string().min(1),
        ncclSocketIfname: z.string().optional(),
        allowedGpuIndexes: z.array(z.number().int().min(0)).optional(),
        enabled: z.boolean().optional(),
      })).min(1),
    }))
    .mutation(async ({ ctx, input }) => {
      try {
        const data = await TrainingResourceService.setPoolNodes(input.poolId, input.nodes);
        await auditAdminAction(ctx.user as SafeAuthUser, "admin_training_pool_nodes_set", {
          poolId: input.poolId,
          nodeIds: input.nodes.map((node) => node.nodeId),
        });
        return {
          success: true,
          message: "训练资源池节点已保存",
          data,
        };
      } catch (error) {
        throwIfForbidden(error);
        throw new TRPCError({
          code: "BAD_REQUEST",
          message: error instanceof Error ? error.message : "Failed to save training resource pool nodes",
        });
      }
    }),

  setTrainingGroupQuota: adminProcedure
    .input(z.object({
      groupId: z.string().min(1),
      poolId: z.string().min(1),
      homeNodeId: z.string().min(1),
      guaranteedGpuCount: z.number().int().min(0),
      maxGpuCount: z.number().int().min(1),
      maxConcurrentJobs: z.number().int().min(1),
      maxNodesPerJob: z.number().int().min(1),
    }))
    .mutation(async ({ ctx, input }) => {
      try {
        const data = await TrainingResourceService.setGroupQuota(input);
        await auditAdminAction(ctx.user as SafeAuthUser, "admin_training_group_quota_set", {
          groupId: input.groupId,
          poolId: input.poolId,
          maxGpuCount: input.maxGpuCount,
          maxConcurrentJobs: input.maxConcurrentJobs,
        });
        return {
          success: true,
          message: "用户组训练配额已保存",
          data,
        };
      } catch (error) {
        throwIfForbidden(error);
        throw new TRPCError({
          code: "BAD_REQUEST",
          message: error instanceof Error ? error.message : "Failed to save training resource quota",
        });
      }
    }),

  deleteTrainingGroupQuota: adminProcedure
    .input(z.object({ groupId: z.string().min(1), poolId: z.string().min(1) }))
    .mutation(async ({ ctx, input }) => {
      try {
        const data = await TrainingResourceService.deleteGroupQuota(input.groupId, input.poolId);
        await auditAdminAction(ctx.user as SafeAuthUser, "admin_training_group_quota_deleted", {
          groupId: input.groupId,
          poolId: input.poolId,
        });
        return {
          success: true,
          message: "用户组训练配额已删除",
          data,
        };
      } catch (error) {
        throwIfForbidden(error);
        throw new TRPCError({
          code: "BAD_REQUEST",
          message: error instanceof Error ? error.message : "Failed to delete training resource quota",
        });
      }
    }),

  reserveTrainingResources: adminProcedure
    .input(z.object({
      groupId: z.string().optional(),
      poolId: z.string().min(1),
      nodeCount: z.number().int().min(1),
      gpusPerNode: z.number().int().min(1),
      taskCategory: z.enum(["training", "assessment", "evaluation"]).optional(),
      taskType: z.string().trim().min(1).optional(),
      taskTypeText: z.string().trim().min(1).optional(),
    }))
    .mutation(async ({ ctx, input }) => {
      const data = await TrainingResourceService.reserve(ctx.user as SafeAuthUser, input);
      await auditAdminAction(ctx.user as SafeAuthUser, "admin_training_resources_reserved", {
        reservationId: data.reservationId,
        groupId: input.groupId || null,
        poolId: input.poolId,
        nodeCount: input.nodeCount,
        gpusPerNode: input.gpusPerNode,
      });
      return {
        success: true,
        message: "训练资源已预约",
        data,
      };
    }),

  releaseTrainingResources: adminProcedure
    .input(z.object({ reservationId: z.string().min(1), force: z.boolean().optional() }))
    .mutation(async ({ ctx, input }) => {
      const data = await TrainingResourceService.release(ctx.user as SafeAuthUser, input.reservationId, {
        force: input.force === true,
        reason: input.force ? "管理员强制释放" : null,
      });
      await auditAdminAction(ctx.user as SafeAuthUser, "admin_training_resources_released", {
        reservationId: input.reservationId,
        status: data.status,
        force: input.force === true,
        mode: input.force === true ? "force_release_reservation" : "release_reservation",
      });
      return {
        success: data.status === "released",
        message: data.status === "released"
          ? (input.force === true ? "trainingResource.forceReleaseSuccess" : "trainingResource.releaseSuccess")
          : data.errorMessage || "trainingResource.releaseFailed",
        data,
      };
    }),

  stopAndReleaseTrainingReservation: adminProcedure
    .input(z.object({ reservationId: z.string().min(1) }))
    .mutation(async ({ ctx, input }) => {
      try {
        const result = await TrainingResourceService.stopProcessAndRelease(
          ctx.user as SafeAuthUser,
          input.reservationId,
        );
        await auditAdminAction(ctx.user as SafeAuthUser, "admin_training_resources_stop_and_released", {
          reservationId: input.reservationId,
          status: result.reservation.status,
          mode: "stop_process_and_release_reservation",
          container: result.stopResult?.container || null,
          pid: result.stopResult?.pid || null,
          alreadyExited: result.stopResult?.alreadyExited === true,
          stopped: result.stopResult?.stopped === true,
        });
        return {
          success: result.reservation.status === "released",
          message: result.reservation.status === "released"
            ? "trainingResource.stopAndReleaseSuccess"
            : result.reservation.errorMessage || "trainingResource.releaseFailed",
          data: {
            reservation: result.reservation,
            stopResult: result.stopResult,
          },
        };
      } catch (error) {
        throw new TRPCError({
          code: "BAD_REQUEST",
          message: error instanceof Error ? error.message : String(error),
        });
      }
    }),

  preflightTrainingResourcesForRuntime: runtimeNodeProcedure
    .input(z.object({
      groupId: z.string().min(1),
      poolId: z.string().min(1).optional(),
      runtimeNodeId: z.string().min(1),
      nodeCount: z.number().int().min(1),
      gpusPerNode: z.number().int().min(1),
      runtimeGpuSnapshot: z.object({
        gpus: z.array(z.object({
          index: z.number().int().min(0),
          available: z.boolean().optional(),
          memoryUsed: z.number().optional(),
        })).optional(),
        collectedAt: z.string().optional(),
        ageSeconds: z.number().optional(),
        maxAgeSeconds: z.number().optional(),
      }).optional(),
      taskCategory: z.enum(["training", "assessment", "evaluation"]).optional(),
      taskType: z.string().trim().min(1).optional(),
      taskTypeText: z.string().trim().min(1).optional(),
    }))
    .mutation(async ({ ctx, input }) => {
      assertRuntimeNodeAuthorized(ctx, input.runtimeNodeId);
      return {
        success: true,
        message: "Runtime 训练资源预检通过",
        data: await TrainingResourceService.preflightForRuntime(input),
      };
    }),
  reserveTrainingResourcesForRuntime: runtimeNodeProcedure
    .input(z.object({
      groupId: z.string().min(1),
      poolId: z.string().min(1).optional(),
      runtimeNodeId: z.string().min(1),
      nodeCount: z.number().int().min(1),
      gpusPerNode: z.number().int().min(1),
      runtimeGpuSnapshot: z.object({
        gpus: z.array(z.object({
          index: z.number().int().min(0),
          available: z.boolean().optional(),
          memoryUsed: z.number().optional(),
        })).optional(),
        collectedAt: z.string().optional(),
        ageSeconds: z.number().optional(),
        maxAgeSeconds: z.number().optional(),
      }).optional(),
      taskCategory: z.enum(["training", "assessment", "evaluation"]).optional(),
      taskType: z.string().trim().min(1).optional(),
      taskTypeText: z.string().trim().min(1).optional(),
    }))
    .mutation(async ({ ctx, input }) => {
      assertRuntimeNodeAuthorized(ctx, input.runtimeNodeId);
      return {
      success: true,
      message: "Runtime 训练资源已预约",
      data: await TrainingResourceService.reserveForRuntime(input),
      };
    }),

  releaseTrainingResourcesForRuntime: runtimeNodeProcedure
    .input(z.object({
      reservationId: z.string().min(1),
      runtimeNodeId: z.string().min(1),
    }))
    .mutation(async ({ ctx, input }) => {
      assertRuntimeNodeAuthorized(ctx, input.runtimeNodeId);
      const data = await TrainingResourceService.releaseForRuntime(input.reservationId, input.runtimeNodeId);
      return {
      success: data.status === "released",
      message: data.status === "released" ? "Runtime 训练资源已释放" : data.errorMessage || "Runtime 训练资源释放失败",
      data,
      };
    }),

  renewTrainingResourcesForRuntime: runtimeNodeProcedure
    .input(z.object({
      reservationId: z.string().min(1),
      runtimeNodeId: z.string().min(1),
    }))
    .mutation(async ({ ctx, input }) => {
      assertRuntimeNodeAuthorized(ctx, input.runtimeNodeId);
      return {
      success: true,
      message: "Runtime 训练资源已续期",
      data: await TrainingResourceService.renewForRuntime(input.reservationId, input.runtimeNodeId),
      };
    }),

  listRuntimeGroupsForRuntime: runtimeNodeProcedure
    .input(z.object({
      runtimeNodeId: z.string().min(1),
    }))
    .mutation(async ({ ctx, input }) => {
      assertRuntimeNodeAuthorized(ctx, input.runtimeNodeId);
      return {
        success: true,
        message: "Runtime groups retrieved",
        data: await TrainingResourceService.listRuntimeGroupsForNode(input.runtimeNodeId),
      };
    }),

  createResourceGroup: adminProcedure
    .input(z.object({
      name: z.string().trim().min(1).max(64),
      containerName: z.string().trim().min(1).max(128),
      evaluateContainerName: z.string().trim().min(1).max(128),
      grpoContainerName: z.string().trim().min(1).max(128),
      description: z.string().max(256).optional(),
    }))
    .mutation(async ({ ctx, input }) => {
      const data = await ResourceAccessService.createGroup(
        input.name,
        input.containerName,
        input.evaluateContainerName,
        input.grpoContainerName,
        input.description,
      );
      await auditAdminAction(ctx.user as SafeAuthUser, "admin_resource_group_created", {
        groupId: data.id,
        name: input.name,
      });
      return {
        success: true,
        message: "用户组已创建",
        data,
      };
    }),

  listAllGroupDatasets: adminProcedure.query(async () => {
    const groups = await ResourceAccessService.listGroups();
    const data = await Promise.all(groups.map(async (group) => {
      if (!group.nodeId) {
        return { groupId: group.id, groupName: group.name, containerName: group.defaultContainerName, nodeId: null, items: [], error: null };
      }
      try {
        const snapshot = await managementCacheService.getDatasets(group.nodeId, group.defaultContainerName);
        return {
          groupId: group.id,
          groupName: group.name,
          containerName: group.defaultContainerName,
          nodeId: group.nodeId,
          items: snapshot.items,
          error: null,
        };
      } catch (error) {
        throwIfForbidden(error);
        return {
          groupId: group.id,
          groupName: group.name,
          containerName: group.defaultContainerName,
          nodeId: group.nodeId,
          items: [],
          error: error instanceof Error ? error.message : String(error),
        };
      }
    }));
    return { success: true, message: "All group datasets retrieved", data };
  }),

  deleteResourceGroup: adminProcedure
    .input(z.object({ groupId: z.string().min(1) }))
    .mutation(async ({ ctx, input }) => {
      try {
        await ResourceAccessService.deleteGroup(input.groupId);
        await auditAdminAction(ctx.user as SafeAuthUser, "admin_resource_group_deleted", {
          groupId: input.groupId,
        });
        return { success: true, message: "用户组已删除" };
      } catch (error) {
        throwIfForbidden(error);
        throw new TRPCError({
          code: "BAD_REQUEST",
          message: error instanceof Error ? error.message : "Failed to delete resource group",
        });
      }
    }),

  setResourceGroupContainer: adminProcedure
    .input(z.object({ groupId: z.string().min(1), containerName: z.string().trim().min(1).max(128) }))
    .mutation(async ({ ctx, input }) => {
      const data = await ResourceAccessService.setGroupContainer(input.groupId, input.containerName);
      await auditAdminAction(ctx.user as SafeAuthUser, "admin_resource_group_container_set", {
        groupId: input.groupId,
        containerName: input.containerName,
      });
      return {
        success: true,
        message: "用户组 Docker 已更新",
        data,
      };
    }),

  setResourceGroupEvaluateContainer: adminProcedure
    .input(z.object({ groupId: z.string().min(1), containerName: z.string().trim().min(1).max(128) }))
    .mutation(async ({ ctx, input }) => {
      const data = await ResourceAccessService.setGroupEvaluateContainer(input.groupId, input.containerName);
      await auditAdminAction(ctx.user as SafeAuthUser, "admin_resource_group_evaluate_container_set", {
        groupId: input.groupId,
        containerName: input.containerName,
      });
      return {
        success: true,
        message: "用户组评测 Docker 已更新",
        data,
      };
    }),


  setResourceGroupGrpoContainer: adminProcedure
    .input(z.object({ groupId: z.string().min(1), containerName: z.string().trim().min(1).max(128) }))
    .mutation(async ({ ctx, input }) => {
      const data = await ResourceAccessService.setGroupGrpoContainer(input.groupId, input.containerName);
      await auditAdminAction(ctx.user as SafeAuthUser, "admin_resource_group_grpo_container_set", {
        groupId: input.groupId,
        containerName: input.containerName,
      });
      return {
        success: true,
        message: "用户组 GRPO Docker 已更新",
        data,
      };
    }),
  moveUserToResourceGroup: adminProcedure
    .input(z.object({ userId: z.string().min(1), groupId: z.string().min(1) }))
    .mutation(async ({ ctx, input }) => {
      const data = await ResourceAccessService.moveUser(input.userId, input.groupId);
      await auditAdminAction(ctx.user as SafeAuthUser, "admin_user_resource_group_moved", {
        targetUserId: input.userId,
        groupId: input.groupId,
      });
      return {
        success: true,
        message: "用户组已更新",
        data,
      };
    }),

  listNodeAssignments: adminProcedure.query(async () => ({
    success: true,
    message: "Node assignments retrieved",
    data: await ResourceAccessService.listNodeAssignments(),
  })),

  setResourceGroupNode: adminProcedure
    .input(z.object({ groupId: z.string().min(1), nodeId: z.string().nullable() }))
    .mutation(async ({ ctx, input }) => {
      const data = await ResourceAccessService.setGroupNode(input.groupId, input.nodeId);
      await auditAdminAction(ctx.user as SafeAuthUser, "admin_resource_group_node_set", {
        groupId: input.groupId,
        nodeId: input.nodeId,
      });
      return {
        success: true,
        message: "用户组节点已更新",
        data,
      };
    }),

  validateResourceGroupContainers: adminProcedure
    .input(z.object({ groupId: z.string().min(1) }))
    .mutation(async ({ ctx, input }) => {
      const data = await ResourceAccessService.validateGroupContainers(input.groupId);
      await auditAdminAction(ctx.user as SafeAuthUser, "admin_resource_group_containers_validated", {
        groupId: input.groupId,
      });
      return {
        success: true,
        message: "用户组 Docker 已重新校验",
        data,
      };
    }),

  listResourceCatalog: protectedProcedure.query(async ({ ctx }) => ({
    success: true,
    message: "Resource catalog retrieved",
    data: await ResourceAccessService.listCatalog(ctx.user as SafeAuthUser),
  })),

  requestResourcePublication: protectedProcedure
    .input(z.object({ resourceId: z.string().min(1) }))
    .mutation(async ({ ctx, input }) => ({
      success: true,
      message: "发布申请已提交",
      data: await ResourceAccessService.requestPublication(input.resourceId, (ctx.user as SafeAuthUser).id),
    })),

  publishResource: adminProcedure
    .input(z.object({ resourceId: z.string().min(1) }))
    .mutation(async ({ ctx, input }) => ({
      success: true,
      message: "资源已发布",
      data: await ResourceAccessService.publishResource(input.resourceId, (ctx.user as SafeAuthUser).id),
    })),

  reviewResourcePublication: adminProcedure
    .input(z.object({
      requestId: z.string().min(1),
      approved: z.boolean(),
      note: z.string().max(256).optional(),
    }))
    .mutation(async ({ ctx, input }) => ({
      success: true,
      message: input.approved ? "资源已发布" : "发布申请已拒绝",
      data: await ResourceAccessService.reviewPublication(
        input.requestId,
        input.approved,
        (ctx.user as SafeAuthUser).id,
        input.note,
      ),
    })),

  listResourceAuditEvents: adminProcedure.query(async () => ({
    success: true,
    message: "资源审计记录已获取",
    data: await ResourceAccessService.listAuditEvents(),
  })),

  getEnvironmentConfig: protectedProcedure.query(async ({ ctx }) => {
    const user = ctx.user as SafeAuthUser;
    const defaultContainerName =
      await ResourceAccessService.getDefaultContainerForUser(user) ||
      getDefaultContainerName();
    const defaultEvaluateContainerName = getDefaultEvaluateContainerName();
    const resolvedEvaluateContainerName =
      await ResourceAccessService.getDefaultEvaluateContainerForUser(user) ||
      defaultEvaluateContainerName;
    const defaultGrpoContainerName = getDefaultGrpoContainerName();
    const resolvedGrpoContainerName =
      await ResourceAccessService.getDefaultGrpoContainerForUser(user) ||
      defaultGrpoContainerName;
    return {
      success: true,
      message: "Environment config retrieved successfully",
      data: {
        defaultContainerName,
        defaultEvaluateContainerName: resolvedEvaluateContainerName,
        defaultGrpoContainerName: resolvedGrpoContainerName,
      },
    } as ResponseBody<{
      defaultContainerName: string;
      defaultEvaluateContainerName: string;
      defaultGrpoContainerName: string;
    }>;
  }),

  registerRun: runtimeProcedure
    .input(
      z.object({
        id: z.string(),
        project: z.string(),
        name: z.string(),
        timestamp: z.string(),
        pid: z.number(),
        status: z.enum(Object.values(Status) as [string, ...string[]]),
        nodeId: z.string().optional().nullable(),
        // Deprecated
        run_dir: z.string().optional().nullable(),
      }),
    )
    .mutation(async ({ ctx, input }) => {
      // 处理项目名称：去掉 "Unnamed" 前缀
      const processedProject = input.project.replace(/^Unnamed/, "");

      const authenticatedUser = ctx.user as SafeAuthUser | null;
      const runData = {
        id: input.id,
        project: processedProject,
        name: input.name,
        timestamp: input.timestamp,
        run_dir: input.run_dir || "", // Deprecated
        pid: input.pid,
        status: input.status,
        nodeId: input.nodeId || "unknown",
        ownerUserId: authenticatedUser?.id,
      } as RunData;

      await RunDao.addRun(runData);
      if (
        runData.nodeId &&
        runData.nodeId !== "unknown" &&
        [Status.RUNNING, Status.PENDING].includes(runData.status)
      ) {
        await RunDao.keepOnlyRunnableRunForNode(runData.nodeId, runData.id);
      }

      // Notify the subscribers of the specific project
      SocketManager.broadcastRunToProjectRoom(
        processedProject,
        runData.ownerUserId,
      );

      // Notify the clients of the project list
      SocketManager.broadcastRunToProjectListRoom(runData.ownerUserId);

      // Notify the clients of the overview room
      SocketManager.broadcastOverviewDataToDashboardRoom(runData.ownerUserId);
    }),

  updateRunStatus: runtimeProcedure
    .input(
      z.object({
        runId: z.string(),
        status: z.enum(Object.values(Status) as [string, ...string[]]),
      }),
    )
    .mutation(async ({ input }) => {
      await SocketManager.changeRunStatusAndTriggerEvents(
        input.runId,
        input.status as Status,
      );
      return {
        success: true,
        message: "Run status updated successfully",
      } as ResponseBody<null>;
    }),

  requestUserInput: runtimeProcedure
    .input(
      z.object({
        requestId: z.string(),
        runId: z.string(),
        agentId: z.string(),
        agentName: z.string(),
        structuredInput: z.record(z.unknown()).nullable(),
      }),
    )
    .mutation(async ({ input }) => {
      console.debug(
        `[StudioInput] requestUserInput received run=${input.runId} request=${input.requestId} agent=${input.agentName}`,
      );
      const runExist = await RunDao.doesRunExist(
        input.runId,
        systemAdminUser(),
      );

      if (!runExist) {
        throw new TRPCError({
          code: "BAD_REQUEST",
          message: `Run with id ${input.runId} does not exist`,
        });
      }

      try {
        // Save the input request to the database
        await InputRequestDao.saveInputRequest({
          requestId: input.requestId,
          runId: input.runId,
          agentId: input.agentId,
          agentName: input.agentName,
          structuredInput: input.structuredInput,
        });
        console.debug(
          `${input.runId}: input request saved with id ${input.requestId}`,
        );

        // Broadcast the input request to the run room
        SocketManager.broadcastInputRequestToRunRoom(input.runId, {
          requestId: input.requestId,
          agentId: input.agentId,
          agentName: input.agentName,
          structuredInput: input.structuredInput,
        } as InputRequestData);
      } catch (error) {
        throwIfForbidden(error);
        console.error(error);
        throw new TRPCError({
          code: "BAD_REQUEST",
          message:
            "Failed to save input request, look at the server logs for more information",
        });
      }
    }),

  registerReply: runtimeProcedure
    .input(RegisterReplyParamsSchema)
    .mutation(async ({ input }) => {
      try {
        await ReplyDao.saveReply(input);
      } catch (error) {
        throwIfForbidden(error);
        console.error(error);
        throw new TRPCError({
          code: "BAD_REQUEST",
          message: `Failed to register reply for error: ${error}`,
        });
      }
    }),

  pushMessage: runtimeProcedure
    .input(
      z.object({
        runId: z.string(),
        replyId: z.string().optional().nullable(),
        replyName: z.string().optional().nullable(),
        replyRole: z.string().optional().nullable(),
        msg: z.object({
          id: z.string(),
          name: z.string(),
          role: z.string(),
          content: contentType,
          metadata: z.unknown(),
          timestamp: z.string(),
        }),
        // The name and role here are deprecated, use replyName and replyRole instead
        name: z.string().optional().nullable(),
        role: z.string().optional().nullable(),
      }),
    )
    .mutation(async ({ input }) => {
      const runExist = await RunDao.doesRunExist(
        input.runId,
        systemAdminUser(),
      );
      console.log("Received pushMessage:", input);
      if (!runExist) {
        throw new TRPCError({
          code: "BAD_REQUEST",
          message: `Run with id ${input.runId} does not exist`,
        });
      }

      const replyId = input.replyId ?? input.msg.id;
      const replyRole = input.replyRole ?? input.role ?? input.msg.role;
      const replyName = input.replyName ?? input.name ?? input.msg.name;

      const reply: Reply = {
        replyId,
        replyName,
        replyRole,
        createdAt: input.msg.timestamp,
        messages: [input.msg as Message],
      };

      await SocketManager.persistAndBroadcastReply(input.runId, reply);
    }),

  /**
   * Get paginated projects with optional sorting and filtering
   *
   * @param pagination - Pagination parameters (page number and page size)
   * @param sort - Optional sorting configuration (field name and order)
   * @param filters - Optional filters for project search (e.g., project name)
   * @returns ResponseBody containing TableData with project list and metadata
   *
   * @example
   * Input: {
   *   pagination: { page: 1, pageSize: 10 },
   *   sort: { field: 'createdAt', order: 'desc' },
   *   filters: { project: 'my-project' }
   * }
   *
   * Output: {
   *   success: true,
   *   message: 'Projects fetched successfully',
   *   data: {
   *     list: [...],
   *     total: 100,
   *     page: 1,
   *     pageSize: 10
   *   }
   * }
   */
  getProjects: adminProcedure
    .input(TableRequestParamsSchema)
    .query(async ({ ctx, input }) => {
      try {
        const result = await RunDao.getProjects(
          input.pagination,
          input.sort,
          input.filters,
          ctx.user as SafeAuthUser,
        );

        return {
          success: true,
          message: "Projects fetched successfully",
          data: result,
        } as ResponseBody<TableData<ProjectData>>;
      } catch (error) {
        throwIfForbidden(error);
        console.error("Error fetching projects:", error);
        return {
          success: false,
          message: error instanceof Error ? error.message : "Unknown error",
        } as ResponseBody<TableData<ProjectData>>;
      }
    }),

  getLatestRun: adminProcedure.query(async ({ ctx }) => {
    try {
      const latestRun = await RunDao.getLatestRunSummary(
        ctx.user as SafeAuthUser,
      );
      return {
        success: true,
        message: latestRun
          ? "Latest run fetched successfully"
          : "No runs found",
        data: latestRun,
      } as ResponseBody<{
        project: string;
        runId: string;
        runName: string;
        timestamp: string;
        status: Status;
      } | null>;
    } catch (error) {
      throwIfForbidden(error);
      console.error("Error fetching latest run:", error);
      throw new TRPCError({
        code: "INTERNAL_SERVER_ERROR",
        message:
          error instanceof Error ? error.message : "Failed to fetch latest run",
      });
    }
  }),

  getLatestRunnableRun: protectedProcedure.query(async ({ ctx }) => {
    try {
      const user = ctx.user as SafeAuthUser;
      const assignedNodeId =
        user.role === UserRole.ADMIN
          ? null
          : user.assignedNodeId || (await ResourceAccessService.getAssignedNodeId(user.id));
      const latestRun =
        user.role === UserRole.ADMIN
          ? await RunDao.getLatestRunnableRunSummary(user)
          : assignedNodeId
            ? await RunDao.getRunnableRunSummaryForNode(assignedNodeId)
            : null;
      return {
        success: true,
        message: latestRun
          ? "Latest runnable run fetched successfully"
          : "No runnable runs found",
        data: latestRun,
      } as ResponseBody<{
        project: string;
        runId: string;
        runName: string;
        timestamp: string;
        status: Status;
      } | null>;
    } catch (error) {
      throwIfForbidden(error);
      console.error("Error fetching latest runnable run:", error);
      throw new TRPCError({
        code: "INTERNAL_SERVER_ERROR",
        message:
          error instanceof Error
            ? error.message
            : "Failed to fetch latest runnable run",
      });
    }
  }),

  getSharedServiceAvailability: protectedProcedure.query(async ({ ctx }) => {
    const user = ctx.user as SafeAuthUser;
    const nodeId =
      user.role === UserRole.ADMIN
        ? getDefaultResourceNodeId()
        : user.assignedNodeId || (await ResourceAccessService.getAssignedNodeId(user.id));
    if (!nodeId) {
      return {
        success: true,
        message: "Service unavailable",
        data: {
          available: false,
          name: "MedFlow 智能服务",
          description: "服务暂不可用",
          nodeId: null,
        },
      };
    }
    const sharedRun =
      user.role === UserRole.ADMIN
        ? await RunDao.getLatestSharedRunnableRunSummary()
        : await RunDao.getRunnableRunSummaryForNode(nodeId);
    let nodeAvailable = false;
    let nodeError: string | null = null;
    if (sharedRun) {
      try {
        await probeResourceNode(nodeId);
        nodeAvailable = true;
      } catch (error) {
        throwIfForbidden(error);
        nodeError = error instanceof Error ? error.message : String(error);
      }
    }
    return {
      success: true,
      message: sharedRun && nodeAvailable ? "Service available" : "Service unavailable",
      data: {
        available: Boolean(sharedRun && nodeAvailable),
        name: "MedFlow 智能服务",
        description:
          sharedRun && nodeAvailable
            ? "服务已就绪，可以开始使用"
            : nodeError || "服务暂不可用",
        nodeId,
      },
    };
  }),

  getTraceList: adminProcedure
    .input(GetTraceListParamsSchema)
    .query(async ({ ctx, input }) => {
      try {
        console.debug("[TRPC] getTraceList called with input:", input);
        const user = ctx.user as SafeAuthUser;
        const result = await SpanDao.getTraceList({
          ...input,
          ownerUserId: user.role === UserRole.ADMIN ? undefined : user.id,
        });
        console.debug("[TRPC] getTraceList result:", {
          total: result.total,
          tracesCount: result.traces.length,
        });
        return result;
      } catch (error) {
        throwIfForbidden(error);
        console.error("Error in getTraceList:", error);
        throw new TRPCError({
          code: "INTERNAL_SERVER_ERROR",
          message:
            error instanceof Error ? error.message : "Failed to get trace list",
        });
      }
    }),

  getTrace: adminProcedure
    .input(GetTraceParamsSchema)
    .query(async ({ ctx, input }) => {
      try {
        const user = ctx.user as SafeAuthUser;
        return await SpanDao.getTrace(
          input.traceId,
          user.role === UserRole.ADMIN ? undefined : user.id,
        );
      } catch (error) {
        throwIfForbidden(error);
        console.error("Error in getTrace:", error);
        throw new TRPCError({
          code: "INTERNAL_SERVER_ERROR",
          message:
            error instanceof Error ? error.message : "Failed to get trace",
        });
      }
    }),

  getTraceStatistic: adminProcedure
    .input(GetTraceStatisticParamsSchema)
    .query(async ({ ctx, input }) => {
      try {
        const user = ctx.user as SafeAuthUser;
        return await SpanDao.getTraceStatistic({
          ...input,
          ownerUserId: user.role === UserRole.ADMIN ? undefined : user.id,
        });
      } catch (error) {
        throwIfForbidden(error);
        console.error("Error in getTraceStatistic:", error);
        throw new TRPCError({
          code: "INTERNAL_SERVER_ERROR",
          message:
            error instanceof Error
              ? error.message
              : "Failed to get trace statistics",
        });
      }
    }),

  getCurrentVersion: protectedProcedure.query(async () => {
    try {
      const version = APP_INFO.version;
      return {
        success: true,
        message: "Version retrieved successfully",
        data: {
          version: version,
        },
      } as ResponseBody<{ version: string }>;
    } catch (error) {
      throwIfForbidden(error);
      console.error("Error get current version:", error);
      throw new TRPCError({
        code: "INTERNAL_SERVER_ERROR",
        message:
          error instanceof Error
            ? error.message
            : "Failed to get current version",
      });
    }
  }),

  getDataInfo: adminProcedure.query(async () => {
    try {
      const configManager = ConfigManager.getInstance();
      const dbStats = configManager.getDataStats();
      return {
        success: true,
        message: "Database info retrieved successfully",
        data: {
          path: dbStats.path,
          size: dbStats.size,
          formattedSize: dbStats.formattedSize,
        },
      } as ResponseBody<{
        path: string;
        size: number;
        formattedSize: string;
      }>;
    } catch (error) {
      throwIfForbidden(error);
      console.error("Error get database info:", error);
      throw new TRPCError({
        code: "INTERNAL_SERVER_ERROR",
        message:
          error instanceof Error
            ? error.message
            : "Failed to get database info",
      });
    }
  }),

  /**
   * 一键环境体检：检查 Docker 容器、GPU、数据集、模型、评测文件和评测结果。
   */
  checkEnvironment: protectedProcedure
    .input(
      z.object({
        nodeId: z.string().trim().optional(),
        container: z.string().trim().optional(),
      }),
    )
    .mutation(async ({ input, ctx }) => {
      try {
        const user = ctx.user as SafeAuthUser;
        const container = await ResourceAccessService.resolveContainerForUser(
          user,
          input.container || "",
          getDefaultContainerName(),
        );
        const targetNodeId = await resolveSingleResourceNodeIdForUser(
          user,
          input.nodeId,
        );
        const remote = await remoteResourceClient.request<{
          data: EnvironmentCheckResult;
        }>(
          targetNodeId,
          "environment-check",
          undefined,
          { container },
          120_000,
        );
        return {
          success: true,
          message: "Environment check completed successfully",
          data: remote.data,
        } as ResponseBody<EnvironmentCheckResult>;
      } catch (error) {
        throwIfForbidden(error);
        const errorMessage =
          error instanceof Error
            ? error.message
            : "Failed to check environment";
        console.error("Error checking environment:", errorMessage);

        return {
          success: false,
          message: errorMessage,
        } as ResponseBody<null>;
      }
    }),

  queryTrainingMetrics: protectedProcedure
    .input(
      z.object({
        runId: z.string().trim().optional(),
        workflowId: z.string().trim().optional(),
        nodeId: z.string().trim().optional(),
        container: z.string().trim().optional(),
        pid: z.string().optional(),
        trainType: z.string().optional(),
        historyLimit: z.number().int().positive().max(1000).optional(),
        timeWindowMinutes: z.number().int().positive().max(1440).optional(),
      }),
    )
    .mutation(async ({ input, ctx }) => {
      try {
        const user = ctx.user as SafeAuthUser;
        const targetNodeId = await resolveMonitoringRunNodeIdForUser(
          user,
          input.runId,
          input.nodeId,
        );
        const workflowHint = input.workflowId
          ? getWorkflowTrainMetricsHint(
              (
                await remoteResourceClient.request<{
                  data: Record<string, unknown>;
                }>(targetNodeId, "workflow-status", undefined, {
                  workflowId: input.workflowId,
                })
              ).data,
            )
          : null;
        const requestedContainer =
          workflowHint?.container ||
          (user.role === UserRole.ADMIN ? input.container : undefined) ||
          "";
        const container = await ResourceAccessService.resolveContainerForUser(
          user,
          requestedContainer,
          getDefaultContainerName(),
        );
        const response = await remoteResourceClient.request<{
          data: Record<string, unknown>;
        }>(
          targetNodeId,
          "training-metrics",
          {
            method: "POST",
            body: JSON.stringify({
              container,
              pid: workflowHint?.pid || input.pid,
              trainType: workflowHint?.trainType || input.trainType,
              historyLimit: input.historyLimit,
              timeWindowMinutes: input.timeWindowMinutes,
            }),
          },
          undefined,
          100_000,
        );
        const data = response.data;

        return {
          success: true,
          message: "Training metrics retrieved successfully",
          data,
        } as ResponseBody<typeof data>;
      } catch (error) {
        throwIfForbidden(error);
        const errorMessage =
          error instanceof Error
            ? error.message
            : "Failed to query training metrics";
        console.error("Error querying training metrics:", errorMessage);

        return {
          success: false,
          message: errorMessage,
        } as ResponseBody<null>;
      }
    }),

  queryWorkflowStatus: protectedProcedure
    .input(
      z.object({
        workflowId: z.string().trim().min(1),
        runId: z.string().trim().optional(),
        nodeId: z.string().trim().optional(),
      }),
    )
    .mutation(async ({ input, ctx }) => {
      try {
        const targetNodeId = await resolveMonitoringRunNodeIdForUser(
          ctx.user,
          input.runId,
          input.nodeId,
        );
        const response = await remoteResourceClient.request<{
          data: Record<string, unknown>;
        }>(targetNodeId, "workflow-status", undefined, {
          workflowId: input.workflowId,
        });
        const data = response.data;

        return {
          success: true,
          message: "Workflow status retrieved successfully",
          data,
        } as ResponseBody<typeof data>;
      } catch (error) {
        throwIfForbidden(error);
        const errorMessage =
          error instanceof Error
            ? error.message
            : "Failed to query workflow status";
        console.error("Error querying workflow status:", errorMessage);

        return {
          success: false,
          message: errorMessage,
        } as ResponseBody<null>;
      }
    }),

  queryDataFilterStatus: protectedProcedure
    .input(
      z.object({
        container: z.string().trim().min(1),
        outputFolder: z.string().trim().optional(),
        inputFolder: z.string().trim().optional(),
        threshold: z.number().optional(),
        nodeId: z.string().trim().optional(),
        runId: z.string().trim().optional(),
      }),
    )
    .mutation(async ({ input, ctx }) => {
      try {
        const targetNodeId = await resolveMonitoringRunNodeIdForUser(
          ctx.user,
          input.runId,
          input.nodeId,
        );
        const resolvedContainer = await ResourceAccessService.resolveContainerForUser(
          ctx.user,
          input.container,
          getDefaultContainerName(),
        );
        const response = await remoteResourceClient.request<{
          data: Record<string, unknown>;
        }>(
          targetNodeId,
          "datasets/filter-status",
          {
            method: "POST",
            body: JSON.stringify({
              container: resolvedContainer,
              outputFolder: input.outputFolder,
              inputFolder: input.inputFolder,
              threshold: input.threshold,
            }),
          },
          undefined,
          30_000,
        );
        const data = response.data;

        return {
          success: true,
          message: "Data filter status retrieved successfully",
          data,
        } as ResponseBody<typeof data>;
      } catch (error) {
        throwIfForbidden(error);
        const errorMessage =
          error instanceof Error
            ? error.message
            : "Failed to query data filter status";
        console.error("Error querying data filter status:", errorMessage);

        return {
          success: false,
          message: errorMessage,
        } as ResponseBody<null>;
      }
    }),

  /**
   * 从 Docker 容器查询数据集
   */
  queryDatasets: protectedProcedure
    .input(
      z.object({
        nodeId: resourceNodeIdSchema,
        groupId: z.string().trim().optional(),
        container: z.string().optional().default(""),
      }),
    )
    .mutation(async ({ ctx, input }) => {
      const user = ctx.user as SafeAuthUser;
      const groupTarget =
        user.role === UserRole.ADMIN
          ? await resolveAdminGroupResourceTarget(input.groupId, "dataset")
          : null;
      const container =
        groupTarget?.container ??
        (await ResourceAccessService.resolveContainerForUser(
          user,
          input.container,
          getDefaultContainerName(),
        ));
      const targetNodeId =
        groupTarget?.nodeId ??
        (await resolveResourceNodeIdForUser(user, input.nodeId));
      await assertResourceNodeAccess(user, targetNodeId);
      try {
        if (user.role === UserRole.ADMIN) {
          return {
            success: true,
            message: groupTarget
              ? "Group datasets retrieved"
              : "Datasets aggregated",
            data: await aggregateSnapshots(
              "dataset",
              container,
              false,
              user,
              targetNodeId,
              !groupTarget,
            ),
          };
        }
        managementRefreshScheduler.registerContainer(
          "dataset",
          container,
          targetNodeId,
        );
        const datasets = await managementCacheService.getDatasets(
          targetNodeId,
          container,
        );
        datasets.items = sanitizeResourceItemsForUser(
          user,
          "dataset",
          await ResourceAccessService.filterResourceItems(user, "dataset", container, datasets.items),
        ) as typeof datasets.items;

        return {
          success: true,
          message: "Datasets snapshot retrieved successfully",
          data: datasets,
        } as ResponseBody<typeof datasets>;
      } catch (error) {
        throwIfForbidden(error);
        const errorMessage =
          error instanceof Error ? error.message : "Failed to query datasets";
        console.error("Error querying datasets:", errorMessage);

        // 返回错误响应而不是抛出异常，让前端能获取错误信息
        return {
          success: false,
          message: errorMessage,
          data: undefined,
        } as ResponseBody<undefined>;
      }
    }),

  refreshDatasets: protectedProcedure
    .input(
      z.object({
        nodeId: resourceNodeIdSchema,
        groupId: z.string().trim().optional(),
        container: z
          .string()
          .optional()
          .transform(
            (container) => container?.trim() || getDefaultContainerName(),
          ),
      }),
    )
    .mutation(async ({ ctx, input }) => {
      const user = ctx.user as SafeAuthUser;
      const groupTarget =
        user.role === UserRole.ADMIN
          ? await resolveAdminGroupResourceTarget(input.groupId, "dataset")
          : null;
      const container =
        groupTarget?.container ??
        (await ResourceAccessService.resolveContainerForUser(
          user,
          input.container,
          getDefaultContainerName(),
        ));
      const targetNodeId =
        groupTarget?.nodeId ??
        (await resolveResourceNodeIdForUser(user, input.nodeId));
      await assertResourceNodeAccess(user, targetNodeId);
      try {
        if (user.role === UserRole.ADMIN) {
          return {
            success: true,
            message: groupTarget
              ? "Group datasets refreshed"
              : "Datasets refreshed",
            data: await aggregateSnapshots(
              "dataset",
              container,
              true,
              user,
              targetNodeId,
              !groupTarget,
            ),
          };
        }
        managementRefreshScheduler.registerContainer(
          "dataset",
          container,
          targetNodeId,
        );
        const datasets = await managementCacheService.refreshDatasets(
          targetNodeId,
          container,
        );
        datasets.items = sanitizeResourceItemsForUser(
          user,
          "dataset",
          await ResourceAccessService.filterResourceItems(user, "dataset", container, datasets.items),
        ) as typeof datasets.items;

        return {
          success: true,
          message: "Datasets refreshed successfully",
          data: datasets,
        } as ResponseBody<typeof datasets>;
      } catch (error) {
        throwIfForbidden(error);
        const errorMessage =
          error instanceof Error ? error.message : "Failed to refresh datasets";
        console.error("Error refreshing datasets:", errorMessage);

        return {
          success: false,
          message: errorMessage,
        } as ResponseBody<null>;
      }
    }),

  getDatasetFilePreviews: protectedProcedure
    .input(
      z.object({
        container: z
          .string()
          .optional()
          .transform(
            (container) => container?.trim() || getDefaultContainerName(),
          ),
        datasetType: z.enum(["raw", "sft", "dpo"]),
        datasetName: z.string(),
        nodeId: resourceNodeIdSchema,
      }),
    )
    .mutation(async ({ ctx, input }) => {
      const user = ctx.user as SafeAuthUser;
      const container = await ResourceAccessService.resolveContainerForUser(
        user,
        input.container,
        getDefaultContainerName(),
      );
      const targetNodeId = await resolveSingleResourceNodeIdForUser(
        user,
        input.nodeId,
      );
      await ResourceAccessService.assertResourceAccess(
        user,
        "dataset",
        container,
        `${input.datasetType}:${input.datasetName}`,
      );
      try {
        const response = await remoteResourceClient.request<{
          data: { filename: string; preview: string }[];
        }>(
          targetNodeId,
          "datasets/previews",
          {
            method: "POST",
            body: JSON.stringify({
              container,
              datasetType: input.datasetType,
              datasetName: input.datasetName,
            }),
          },
        );
        const previews = response.data || [];

        return {
          success: true,
          message: "Dataset file previews retrieved successfully",
          data: previews,
        } as ResponseBody<typeof previews>;
      } catch (error) {
        throwIfForbidden(error);
        const errorMessage =
          error instanceof Error
            ? error.message
            : "Failed to get dataset file previews";
        console.error("Error getting dataset file previews:", errorMessage);

        return {
          success: false,
          message: errorMessage,
        } as ResponseBody<null>;
      }
    }),

  /**
   * 从 Docker 容器查询模型
   */
  queryModels: protectedProcedure
    .input(
      z.object({
        nodeId: resourceNodeIdSchema,
        groupId: z.string().trim().optional(),
        container: z.string().optional().default(""),
      }),
    )
    .mutation(async ({ ctx, input }) => {
      const user = ctx.user as SafeAuthUser;
      const groupTarget =
        user.role === UserRole.ADMIN
          ? await resolveAdminGroupResourceTarget(input.groupId, "model")
          : null;
      const container =
        groupTarget?.container ??
        (await ResourceAccessService.resolveContainerForUser(
          user,
          input.container,
          getDefaultContainerName(),
        ));
      const targetNodeId =
        groupTarget?.nodeId ??
        (await resolveResourceNodeIdForUser(user, input.nodeId));
      await assertResourceNodeAccess(user, targetNodeId);
      try {
        if (user.role === UserRole.ADMIN) {
          return {
            success: true,
            message: groupTarget
              ? "Group models retrieved"
              : "Models aggregated",
            data: await aggregateSnapshots(
              "model",
              container,
              false,
              user,
              targetNodeId,
              !groupTarget,
            ),
          };
        }
        managementRefreshScheduler.registerContainer(
          "model",
          container,
          targetNodeId,
        );
        const models = await managementCacheService.getModels(
          targetNodeId,
          container,
        );
        models.items = sanitizeResourceItemsForUser(
          user,
          "model",
          await ResourceAccessService.filterResourceItems(user, "model", container, models.items),
        ) as typeof models.items;

        return {
          success: true,
          message: "Models snapshot retrieved successfully",
          data: models,
        } as ResponseBody<typeof models>;
      } catch (error) {
        throwIfForbidden(error);
        const errorMessage =
          error instanceof Error ? error.message : "Failed to query models";
        console.error("Error querying models:", errorMessage);

        // 返回错误响应而不是抛出异常，让前端能获取错误信息
        return {
          success: false,
          message: errorMessage,
        } as ResponseBody<null>;
      }
    }),

  refreshModels: protectedProcedure
    .input(
      z.object({
        nodeId: resourceNodeIdSchema,
        groupId: z.string().trim().optional(),
        container: z
          .string()
          .optional()
          .transform(
            (container) => container?.trim() || getDefaultContainerName(),
          ),
      }),
    )
    .mutation(async ({ ctx, input }) => {
      const user = ctx.user as SafeAuthUser;
      const groupTarget =
        user.role === UserRole.ADMIN
          ? await resolveAdminGroupResourceTarget(input.groupId, "model")
          : null;
      const container =
        groupTarget?.container ??
        (await ResourceAccessService.resolveContainerForUser(
          user,
          input.container,
          getDefaultContainerName(),
        ));
      const targetNodeId =
        groupTarget?.nodeId ??
        (await resolveResourceNodeIdForUser(user, input.nodeId));
      await assertResourceNodeAccess(user, targetNodeId);
      try {
        if (user.role === UserRole.ADMIN) {
          return {
            success: true,
            message: groupTarget
              ? "Group models refreshed"
              : "Models refreshed",
            data: await aggregateSnapshots(
              "model",
              container,
              true,
              user,
              targetNodeId,
              !groupTarget,
            ),
          };
        }
        managementRefreshScheduler.registerContainer(
          "model",
          container,
          targetNodeId,
        );
        const models = await managementCacheService.refreshModels(
          targetNodeId,
          container,
        );
        models.items = sanitizeResourceItemsForUser(
          user,
          "model",
          await ResourceAccessService.filterResourceItems(user, "model", container, models.items),
        ) as typeof models.items;

        return {
          success: true,
          message: "Models refreshed successfully",
          data: models,
        } as ResponseBody<typeof models>;
      } catch (error) {
        throwIfForbidden(error);
        const errorMessage =
          error instanceof Error ? error.message : "Failed to refresh models";
        console.error("Error refreshing models:", errorMessage);

        return {
          success: false,
          message: errorMessage,
        } as ResponseBody<null>;
      }
    }),

  queryGrpoResources: protectedProcedure
    .input(
      z.object({
        nodeId: z.string().trim().optional(),
        groupId: z.string().trim().optional(),
        container: z
          .string()
          .optional()
          .transform((container) => container?.trim() || getDefaultGrpoContainerName()),
      }),
    )
    .mutation(async ({ input, ctx }) => {
      try {
        const user = ctx.user as SafeAuthUser;
        const selectedGroup = user.role === UserRole.ADMIN && input.groupId
          ? (await ResourceAccessService.listGroups()).find((group) => group.id === input.groupId)
          : null;
        const targetNodeId = await resolveSingleResourceNodeIdForUser(
          user,
          input.nodeId || selectedGroup?.nodeId,
        );
        const container = selectedGroup
          ? selectedGroup.defaultGrpoContainerName
          : await ResourceAccessService.resolveGrpoContainerForUser(
              user,
              input.container,
              getDefaultGrpoContainerName(),
            );
        const response = await remoteResourceClient.request<{
          data: GrpoResourceInfo;
        }>(targetNodeId, "grpo", undefined, { container });
        const resources = response.data;

        return {
          success: true,
          message: "GRPO resources retrieved successfully",
          data: resources,
        } as ResponseBody<typeof resources>;
      } catch (error) {
        throwIfForbidden(error);
        const errorMessage =
          error instanceof Error
            ? error.message
            : "Failed to query GRPO resources";
        console.error("Error querying GRPO resources:", errorMessage);

        return {
          success: false,
          message: errorMessage,
        } as ResponseBody<null>;
      }
    }),

  deleteModel: protectedProcedure
    .input(
      z.object({
        nodeId: resourceNodeIdSchema,
        container: z
          .string()
          .optional()
          .transform(
            (container) => container?.trim() || getDefaultContainerName(),
          ),
        modelType: z.enum([
          "base_train",
          "batch_trained",
          "daily_trained",
          "inference",
        ]),
        modelName: z.string(),
        modelPath: z.string().optional(),
      }),
    )
    .mutation(async ({ ctx, input }) => {
      const user = ctx.user as SafeAuthUser;
      const container = await ResourceAccessService.resolveContainerForUser(
        user,
        input.container,
        getDefaultContainerName(),
      );
      const targetNodeId = await resolveResourceNodeIdForUser(user, input.nodeId);
      await assertResourceNodeAccess(user, targetNodeId);
      try {
        if (targetNodeId === "all") {
          throw new Error("请选择具体节点后再删除模型");
        }
        await ResourceAccessService.assertResourceWriteAccess(
          user,
          "model",
          container,
          input.modelPath ? `${input.modelPath}/${input.modelName}` : input.modelName,
        );
        const response = await remoteResourceClient.request<{
          data: { success: boolean; message?: string };
        }>(targetNodeId, "models", {
          method: "DELETE",
          body: JSON.stringify({ ...input, nodeId: targetNodeId, container }),
        });
        const result = response.data;
        if (result.success) {
          await ResourceAccessService.removeResourceRecord(
            "model",
            targetNodeId,
            container,
            `${input.modelType}:${input.modelName}`,
            user.id,
          );
        }

        return {
          success: result.success,
          message: result.message || "Model deleted successfully",
        } as ResponseBody<null>;
      } catch (error) {
        throwIfForbidden(error);
        const errorMessage =
          error instanceof Error ? error.message : "Failed to delete model";
        console.error("Error deleting model:", errorMessage);

        return {
          success: false,
          message: errorMessage,
        } as ResponseBody<null>;
      }
    }),

  /**
   * 上传数据集到 Docker 容器
   */
  uploadDataset: protectedProcedure
    .input(
      z.object({
        nodeId: resourceNodeIdSchema,
        container: z.string(),
        datasetType: z.enum(["raw", "sft", "dpo"]),
        datasetName: z
          .string()
          .regex(
            /^[a-zA-Z0-9_-]+$/,
            "数据集名称只能包含字母、数字、下划线和横线",
          ),
        filename: z.string(),
        fileBase64: z.string(),
      }),
    )
    .mutation(async ({ ctx, input }) => {
      const user = ctx.user as SafeAuthUser;
      const container = await ResourceAccessService.resolveContainerForUser(
        user,
        input.container,
        getDefaultContainerName(),
      );
      const targetNodeId = await resolveResourceNodeIdForUser(user, input.nodeId);
      await assertResourceNodeAccess(user, targetNodeId);
      try {
        // Base64 解码为 Buffer
        const fileBuffer = Buffer.from(input.fileBase64, "base64");

        // 检查文件大小（20MB = 20 * 1024 * 1024 bytes）
        const MAX_SIZE = 20 * 1024 * 1024;
        if (fileBuffer.length > MAX_SIZE) {
          return {
            success: false,
            message: `文件大小超过 20MB 限制（当前: ${(fileBuffer.length / 1024 / 1024).toFixed(2)}MB）`,
          } as ResponseBody<null>;
        }

        if (targetNodeId === "all") {
          throw new Error("请选择具体节点后再上传数据集");
        }
        if (user.role !== UserRole.ADMIN) {
          await managementCacheService.refreshDatasets(targetNodeId, container);
        }
        await ResourceAccessService.assertResourceWriteAccess(
          user,
          "dataset",
          container,
          `${input.datasetType}:${input.datasetName}`,
          true,
        );
        const response = await remoteResourceClient.request<{
          data: { success: boolean; message?: string; warning?: string };
        }>(targetNodeId, "datasets/upload", {
          method: "POST",
          body: JSON.stringify({ ...input, nodeId: targetNodeId, container }),
        });
        const result = response.data;
        if (result.success) {
          await ResourceAccessService.recordResourceWrite(
            user,
            "dataset",
            targetNodeId,
            container,
            `${input.datasetType}:${input.datasetName}`,
          );
          try {
            await managementCacheService.refreshDatasets(
              targetNodeId,
              container,
              "manual",
            );
          } catch (refreshError) {
            console.warn(
              "Dataset uploaded but the immediate cache refresh failed:",
              refreshError,
            );
          }
        }

        return {
          success: result.success,
          message: result.message,
          warning: result.warning,
        } as ResponseBody<null>;
      } catch (error) {
        throwIfForbidden(error);
        const errorMessage =
          error instanceof Error ? error.message : "Failed to upload dataset";
        console.error("Error uploading dataset:", errorMessage);

        return {
          success: false,
          message: errorMessage,
        } as ResponseBody<null>;
      }
    }),

  /**
   * 从 Docker 容器下载数据集
   */
  downloadDataset: protectedProcedure
    .input(
      z.object({
        nodeId: resourceNodeIdSchema,
        container: z.string(),
        datasetType: z.enum(["raw", "sft", "dpo"]),
        datasetName: z.string(),
      }),
    )
    .mutation(async ({ ctx, input }) => {
      const user = ctx.user as SafeAuthUser;
      const container = await ResourceAccessService.resolveContainerForUser(
        user,
        input.container,
        getDefaultContainerName(),
      );
      const targetNodeId = await resolveSingleResourceNodeIdForUser(
        user,
        input.nodeId,
      );
      await ResourceAccessService.assertResourceAccess(
        user,
        "dataset",
        container,
        `${input.datasetType}:${input.datasetName}`,
      );
      try {
        const response = await remoteResourceClient.request<{
          data: { filename: string; fileBase64: string };
        }>(
          targetNodeId,
          "datasets/download",
          {
            method: "POST",
            body: JSON.stringify({
              container,
              datasetType: input.datasetType,
              datasetName: input.datasetName,
            }),
          },
        );

        if (!response.data?.fileBase64) {
          return {
            success: false,
            message: "下载失败",
          } as ResponseBody<null>;
        }

        return {
          success: true,
          message: "Dataset downloaded successfully",
          data: response.data,
        } as ResponseBody<{ filename: string; fileBase64: string }>;
      } catch (error) {
        throwIfForbidden(error);
        const errorMessage =
          error instanceof Error ? error.message : "Failed to download dataset";
        console.error("Error downloading dataset:", errorMessage);

        return {
          success: false,
          message: errorMessage,
        } as ResponseBody<null>;
      }
    }),

  deleteDataset: protectedProcedure
    .input(
      z.object({
        nodeId: resourceNodeIdSchema,
        container: z
          .string()
          .optional()
          .transform(
            (container) => container?.trim() || getDefaultContainerName(),
          ),
        datasetType: z.enum(["raw", "sft", "dpo"]),
        datasetName: z.string(),
      }),
    )
    .mutation(async ({ ctx, input }) => {
      const user = ctx.user as SafeAuthUser;
      const container = await ResourceAccessService.resolveContainerForUser(
        user,
        input.container,
        getDefaultContainerName(),
      );
      const targetNodeId = await resolveResourceNodeIdForUser(user, input.nodeId);
      await assertResourceNodeAccess(user, targetNodeId);
      try {
        if (targetNodeId === "all") {
          throw new Error("请选择具体节点后再删除数据集");
        }
        await ResourceAccessService.assertResourceWriteAccess(
          user,
          "dataset",
          container,
          `${input.datasetType}:${input.datasetName}`,
        );
        const response = await remoteResourceClient.request<{
          data: { success: boolean; message?: string };
        }>(targetNodeId, "datasets", {
          method: "DELETE",
          body: JSON.stringify({ ...input, nodeId: targetNodeId, container }),
        });
        const result = response.data;
        if (result.success) {
          await ResourceAccessService.removeResourceRecord(
            "dataset",
            targetNodeId,
            container,
            `${input.datasetType}:${input.datasetName}`,
            user.id,
          );
        }

        return {
          success: result.success,
          message: result.message || "Dataset deleted successfully",
        } as ResponseBody<null>;
      } catch (error) {
        throwIfForbidden(error);
        const errorMessage =
          error instanceof Error ? error.message : "Failed to delete dataset";
        console.error("Error deleting dataset:", errorMessage);

        return {
          success: false,
          message: errorMessage,
        } as ResponseBody<null>;
      }
    }),

  /**
   * 查询医疗评测文件列表
   */
  queryMedicalTests: protectedProcedure
    .input(
      z.object({
        nodeId: resourceNodeIdSchema,
        groupId: z.string().trim().optional(),
        container: z.string().optional().default(""),
      }),
    )
    .mutation(async ({ ctx, input }) => {
      const user = ctx.user as SafeAuthUser;
      const groupTarget =
        user.role === UserRole.ADMIN
          ? await resolveAdminGroupResourceTarget(input.groupId, "medicalTest")
          : null;
      const container =
        groupTarget?.container ??
        (await ResourceAccessService.resolveEvaluateContainerForUser(
          user,
          input.container,
          getDefaultEvaluateContainerName(),
        ));
      const targetNodeId =
        groupTarget?.nodeId ??
        (await resolveResourceNodeIdForUser(user, input.nodeId));
      await assertResourceNodeAccess(user, targetNodeId);
      try {
        if (user.role === UserRole.ADMIN) {
          return {
            success: true,
            message: groupTarget
              ? "Group medical tests retrieved"
              : "Medical tests aggregated",
            data: await aggregateSnapshots(
              "medicalTest",
              container,
              false,
              user,
              targetNodeId,
              !groupTarget,
            ),
          };
        }
        managementRefreshScheduler.registerContainer(
          "medicalTest",
          container,
          targetNodeId,
        );
        const tests = await managementCacheService.getMedicalTests(
          targetNodeId,
          container,
        );
        tests.items = sanitizeResourceItemsForUser(
          user,
          "medicalTest",
          await ResourceAccessService.filterResourceItems(user, "medicalTest", container, tests.items),
        ) as typeof tests.items;

        return {
          success: true,
          message: "Medical tests snapshot retrieved successfully",
          data: tests,
        } as ResponseBody<typeof tests>;
      } catch (error) {
        throwIfForbidden(error);
        const errorMessage =
          error instanceof Error
            ? error.message
            : "Failed to query medical tests";
        console.error("Error querying medical tests:", errorMessage);

        return {
          success: false,
          message: errorMessage,
        } as ResponseBody<null>;
      }
    }),

  refreshMedicalTests: protectedProcedure
    .input(
      z.object({
        nodeId: resourceNodeIdSchema,
        groupId: z.string().trim().optional(),
        container: z
          .string()
          .optional()
          .transform(
            (container) => container?.trim() || getDefaultEvaluateContainerName(),
          ),
      }),
    )
    .mutation(async ({ ctx, input }) => {
      const user = ctx.user as SafeAuthUser;
      const groupTarget =
        user.role === UserRole.ADMIN
          ? await resolveAdminGroupResourceTarget(input.groupId, "medicalTest")
          : null;
      const container =
        groupTarget?.container ??
        (await ResourceAccessService.resolveEvaluateContainerForUser(
          user,
          input.container,
          getDefaultEvaluateContainerName(),
        ));
      const targetNodeId =
        groupTarget?.nodeId ??
        (await resolveResourceNodeIdForUser(user, input.nodeId));
      await assertResourceNodeAccess(user, targetNodeId);
      try {
        if (user.role === UserRole.ADMIN) {
          return {
            success: true,
            message: groupTarget
              ? "Group medical tests refreshed"
              : "Medical tests refreshed",
            data: await aggregateSnapshots(
              "medicalTest",
              container,
              true,
              user,
              targetNodeId,
              !groupTarget,
            ),
          };
        }
        managementRefreshScheduler.registerContainer(
          "medicalTest",
          container,
          targetNodeId,
        );
        const tests = await managementCacheService.refreshMedicalTests(
          targetNodeId,
          container,
        );
        tests.items = sanitizeResourceItemsForUser(
          user,
          "medicalTest",
          await ResourceAccessService.filterResourceItems(user, "medicalTest", container, tests.items),
        ) as typeof tests.items;

        return {
          success: true,
          message: "Medical tests refreshed successfully",
          data: tests,
        } as ResponseBody<typeof tests>;
      } catch (error) {
        throwIfForbidden(error);
        const errorMessage =
          error instanceof Error
            ? error.message
            : "Failed to refresh medical tests";
        console.error("Error refreshing medical tests:", errorMessage);

        return {
          success: false,
          message: errorMessage,
        } as ResponseBody<null>;
      }
    }),

  /**
   * 上传医疗评测文件
   */
  uploadMedicalTest: protectedProcedure
    .input(
      z.object({
        nodeId: resourceNodeIdSchema,
        container: z.string(),
        testType: z.string(),
        filename: z.string(),
        fileBase64: z.string(),
      }),
    )
    .mutation(async ({ ctx, input }) => {
      const user = ctx.user as SafeAuthUser;
      const container = await ResourceAccessService.resolveEvaluateContainerForUser(
        user,
        input.container,
        getDefaultEvaluateContainerName(),
      );
      const targetNodeId = await resolveResourceNodeIdForUser(user, input.nodeId);
      await assertResourceNodeAccess(user, targetNodeId);
      try {
        // Base64 解码为 Buffer
        const fileBuffer = Buffer.from(input.fileBase64, "base64");
        const MAX_SIZE = 20 * 1024 * 1024;
        if (fileBuffer.length > MAX_SIZE) {
          return {
            success: false,
            message: `文件大小超过 20MB 限制（当前: ${(fileBuffer.length / 1024 / 1024).toFixed(2)}MB）`,
          } as ResponseBody<null>;
        }

        if (targetNodeId === "all") {
          throw new Error("请选择具体节点后再上传评测文件");
        }
        if (user.role !== UserRole.ADMIN) {
          await managementCacheService.refreshMedicalTests(targetNodeId, container);
        }
        await ResourceAccessService.assertResourceWriteAccess(
          user,
          "medicalTest",
          container,
          input.filename,
          true,
        );
        const response = await remoteResourceClient.request<{
          data: { success: boolean; message?: string };
        }>(targetNodeId, "medical-tests/upload", {
          method: "POST",
          body: JSON.stringify({ ...input, nodeId: targetNodeId, container }),
        });
        const result = response.data;
        if (result.success) {
          await ResourceAccessService.recordResourceWrite(
            user,
            "medicalTest",
            targetNodeId,
            container,
            `json:${input.filename}`,
          );
        }

        return {
          success: result.success,
          message: result.message,
        } as ResponseBody<null>;
      } catch (error) {
        throwIfForbidden(error);
        const errorMessage =
          error instanceof Error
            ? error.message
            : "Failed to upload medical test";
        console.error("Error uploading medical test:", errorMessage);

        return {
          success: false,
          message: errorMessage,
        } as ResponseBody<null>;
      }
    }),

  /**
   * 下载医疗评测文件
   */
  downloadMedicalTest: protectedProcedure
    .input(
      z.object({
        nodeId: resourceNodeIdSchema,
        container: z.string(),
        filename: z.string(),
      }),
    )
    .mutation(async ({ ctx, input }) => {
      const user = ctx.user as SafeAuthUser;
      const container = await ResourceAccessService.resolveEvaluateContainerForUser(
        user,
        input.container,
        getDefaultEvaluateContainerName(),
      );
      const targetNodeId = await resolveSingleResourceNodeIdForUser(
        user,
        input.nodeId,
      );
      await ResourceAccessService.assertResourceAccess(
        user,
        "medicalTest",
        container,
        input.filename,
      );
      try {
        const response = await remoteResourceClient.request<{
          data: { filename: string; fileBase64: string };
        }>(
          targetNodeId,
          "medical-tests/download",
          {
            method: "POST",
            body: JSON.stringify({
              container,
              filename: input.filename,
            }),
          },
        );

        if (!response.data?.fileBase64) {
          return {
            success: false,
            message: "下载失败",
          } as ResponseBody<null>;
        }

        return {
          success: true,
          message: "Medical test downloaded successfully",
          data: response.data,
        } as ResponseBody<{ filename: string; fileBase64: string }>;
      } catch (error) {
        throwIfForbidden(error);
        const errorMessage =
          error instanceof Error
            ? error.message
            : "Failed to download medical test";
        console.error("Error downloading medical test:", errorMessage);

        return {
          success: false,
          message: errorMessage,
        } as ResponseBody<null>;
      }
    }),

  deleteMedicalTest: protectedProcedure
    .input(
      z.object({
        nodeId: resourceNodeIdSchema,
        container: z
          .string()
          .optional()
          .transform(
            (container) => container?.trim() || getDefaultEvaluateContainerName(),
          ),
        filename: z.string(),
      }),
    )
    .mutation(async ({ ctx, input }) => {
      const user = ctx.user as SafeAuthUser;
      const container = await ResourceAccessService.resolveEvaluateContainerForUser(
        user,
        input.container,
        getDefaultEvaluateContainerName(),
      );
      const targetNodeId = await resolveResourceNodeIdForUser(user, input.nodeId);
      await assertResourceNodeAccess(user, targetNodeId);
      try {
        if (targetNodeId === "all") {
          throw new Error("请选择具体节点后再删除评测文件");
        }
        await ResourceAccessService.assertResourceWriteAccess(
          user,
          "medicalTest",
          container,
          input.filename,
        );
        const response = await remoteResourceClient.request<{
          data: { success: boolean; message?: string };
        }>(targetNodeId, "medical-tests", {
          method: "DELETE",
          body: JSON.stringify({ ...input, nodeId: targetNodeId, container }),
        });
        const result = response.data;
        if (result.success) {
          await ResourceAccessService.removeResourceRecord(
            "medicalTest",
            targetNodeId,
            container,
            `json:${input.filename}`,
            user.id,
          );
        }

        return {
          success: result.success,
          message: result.message || "Dataset deleted successfully",
        } as ResponseBody<null>;
      } catch (error) {
        throwIfForbidden(error);
        const errorMessage =
          error instanceof Error
            ? error.message
            : "Failed to delete medical test";
        console.error("Error deleting medical test:", errorMessage);

        return {
          success: false,
          message: errorMessage,
        } as ResponseBody<null>;
      }
    }),

  /**
   * 查询评测结果列表
   */
  queryEvaluationResults: protectedProcedure
    .input(
      z.object({
        nodeId: resourceNodeIdSchema,
        container: z.string().optional().default(""),
      }),
    )
    .mutation(async ({ ctx, input }) => {
      const user = ctx.user as SafeAuthUser;
      const container = await ResourceAccessService.resolveEvaluateContainerForUser(
        user,
        input.container,
        getDefaultEvaluateContainerName(),
      );
      const targetNodeId = await resolveResourceNodeIdForUser(user, input.nodeId);
      await assertResourceNodeAccess(user, targetNodeId);
      try {
        if (user.role === UserRole.ADMIN) {
          return {
            success: true,
            message: "Evaluation results aggregated",
            data: await aggregateSnapshots(
              "evaluationResult",
              container,
              false,
              user,
              targetNodeId,
            ),
          };
        }
        managementRefreshScheduler.registerContainer(
          "evaluationResult",
          container,
          targetNodeId,
        );
        const results = await managementCacheService.getEvaluationResults(
          targetNodeId,
          container,
        );
        results.items = sanitizeResourceItemsForUser(
          user,
          "evaluationResult",
          await ResourceAccessService.filterResourceItems(user, "evaluationResult", container, results.items),
        ) as typeof results.items;

        return {
          success: true,
          message: "Evaluation results snapshot retrieved successfully",
          data: results,
        } as ResponseBody<typeof results>;
      } catch (error) {
        throwIfForbidden(error);
        const errorMessage =
          error instanceof Error
            ? error.message
            : "Failed to query evaluation results";
        console.error("Error querying evaluation results:", errorMessage);

        return {
          success: false,
          message: errorMessage,
        } as ResponseBody<null>;
      }
    }),

  refreshEvaluationResults: protectedProcedure
    .input(
      z.object({
        nodeId: resourceNodeIdSchema,
        container: z
          .string()
          .optional()
          .transform(
            (container) => container?.trim() || getDefaultEvaluateContainerName(),
          ),
      }),
    )
    .mutation(async ({ ctx, input }) => {
      const user = ctx.user as SafeAuthUser;
      const container = await ResourceAccessService.resolveEvaluateContainerForUser(
        user,
        input.container,
        getDefaultEvaluateContainerName(),
      );
      const targetNodeId = await resolveResourceNodeIdForUser(user, input.nodeId);
      await assertResourceNodeAccess(user, targetNodeId);
      try {
        if (user.role === UserRole.ADMIN) {
          return {
            success: true,
            message: "Evaluation results refreshed",
            data: await aggregateSnapshots(
              "evaluationResult",
              container,
              true,
              user,
              targetNodeId,
            ),
          };
        }
        managementRefreshScheduler.registerContainer(
          "evaluationResult",
          container,
          targetNodeId,
        );
        const results = await managementCacheService.refreshEvaluationResults(
          targetNodeId,
          container,
        );
        results.items = sanitizeResourceItemsForUser(
          user,
          "evaluationResult",
          await ResourceAccessService.filterResourceItems(user, "evaluationResult", container, results.items),
        ) as typeof results.items;

        return {
          success: true,
          message: "Evaluation results refreshed successfully",
          data: results,
        } as ResponseBody<typeof results>;
      } catch (error) {
        throwIfForbidden(error);
        const errorMessage =
          error instanceof Error
            ? error.message
            : "Failed to refresh evaluation results";
        console.error("Error refreshing evaluation results:", errorMessage);

        return {
          success: false,
          message: errorMessage,
        } as ResponseBody<null>;
      }
    }),

  /**
   * 下载评测结果文件
   */
  downloadEvaluationResult: protectedProcedure
    .input(
      z.object({
        nodeId: resourceNodeIdSchema,
        container: z.string(),
        folderPath: z.string(),
        filename: z.string(),
      }),
    )
    .mutation(async ({ ctx, input }) => {
      const user = ctx.user as SafeAuthUser;
      const container = await ResourceAccessService.resolveEvaluateContainerForUser(
        user,
        input.container,
        getDefaultEvaluateContainerName(),
      );
      const targetNodeId = await resolveSingleResourceNodeIdForUser(
        user,
        input.nodeId,
      );
      await ResourceAccessService.assertResourceAccess(
        user,
        "evaluationResult",
        container,
        input.folderPath,
      );
      try {
        const response = await remoteResourceClient.request<{
          data: { filename: string; fileBase64: string };
        }>(
          targetNodeId,
          "evaluation-results/download",
          {
            method: "POST",
            body: JSON.stringify({
              container,
              folderPath: input.folderPath,
              filename: input.filename,
            }),
          },
        );

        if (!response.data?.fileBase64) {
          return {
            success: false,
            message: "下载失败",
          } as ResponseBody<null>;
        }

        return {
          success: true,
          message: "Evaluation result downloaded successfully",
          data: response.data,
        } as ResponseBody<{ filename: string; fileBase64: string }>;
      } catch (error) {
        throwIfForbidden(error);
        const errorMessage =
          error instanceof Error
            ? error.message
            : "Failed to download evaluation result";
        console.error("Error downloading evaluation result:", errorMessage);

        return {
          success: false,
          message: errorMessage,
        } as ResponseBody<null>;
      }
    }),

  deleteEvaluationResult: adminProcedure
    .input(
      z.object({
        nodeId: resourceNodeIdSchema,
        container: z
          .string()
          .optional()
          .transform(
            (container) => container?.trim() || getDefaultEvaluateContainerName(),
          ),
        folderPath: z.string(),
      }),
    )
    .mutation(async ({ input, ctx }) => {
      try {
        const targetNodeId = await resolveSingleResourceNodeIdForUser(
          ctx.user,
          input.nodeId,
        );
        const response = await remoteResourceClient.request<{
          data: { success: boolean; message?: string };
        }>(targetNodeId, "evaluation-results", {
          method: "DELETE",
          body: JSON.stringify({
            container: input.container,
            folderPath: input.folderPath,
          }),
        });
        const result = response.data;

        return {
          success: result.success,
          message: result.message,
        } as ResponseBody<null>;
      } catch (error) {
        throwIfForbidden(error);
        const errorMessage =
          error instanceof Error
            ? error.message
            : "Failed to delete evaluation result";
        console.error("Error deleting evaluation result:", errorMessage);

        return {
          success: false,
          message: errorMessage,
        } as ResponseBody<null>;
      }
    }),
});

export type AppRouter = typeof appRouter;

