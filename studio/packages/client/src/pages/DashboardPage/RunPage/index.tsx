import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Flex, Dropdown, Button, message, Modal, Tooltip, Select } from "antd";
import {
  Route,
  Routes,
  useLocation,
  useNavigate,
  useParams,
} from "react-router-dom";
import {
  DownloadOutlined,
  FileTextOutlined,
  FileMarkdownOutlined,
  CodeOutlined,
} from "@ant-design/icons";
import {
  Zap,
  Server,
  Eraser,
  MoreHorizontal,
  ArrowRight,
  ChevronDown,
  ChevronUp,
  BookOpen,
  MousePointerClick,
  LineChart,
  Activity,
  Square,
  Search,
  X,
  Workflow,
  Filter,
  RefreshCw,
} from "lucide-react";

import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";

import ProjectRunSider from "./ProjectRunSider";
import TrainingMetricsPanel from "./TrainingMetricsPanel";
import type { TrainingMetricsCacheSnapshot } from "./TrainingMetricsPanel";
import InferenceServicePanel from "./InferenceServicePanel";
import DatasetUploadModal from "./DatasetUploadModal";
import EvaluationUploadModal from "./EvaluationUploadModal";
import EnvironmentCheckDialog from "./EnvironmentCheckDialog";
import { TrainingWorkflowGuide } from "@/components/TrainingWorkflowGuide";
import { QuickStartWizard } from "@/components/QuickStartWizard";
import type { QuickStartWizardRef } from "@/components/QuickStartWizard";
import TemplateLibraryDialog from "@/components/chat/AsChat/TemplateLibraryDialog";
import { useTemplateLibrary } from "@/hooks/useTemplateLibrary";
import { useIsMobile } from "@/hooks/use-mobile";

import {
  InputRequestData,
  Reply,
  SystemOverviewData,
  SocketEvents,
  GPUInfo,
  DatasetInfo,
  DatasetFilePreview,
  ModelInfo,
  MedicalTestFile,
  EvaluationResult,
  ManagementCacheMeta,
  EnvironmentCheckResult,
} from "@shared/types/trpc";
import { Status } from "@shared/types/messageForm";
import { ContentType, ContentBlock } from "@shared/types/messageForm";
import {
  ProjectRoomContextProvider,
  useProjectRoom,
} from "@/context/ProjectRoomContext";
import { EmptyRunPage, ProjectNotFoundPage } from "../../DefaultPage";
import { RunRoomContextProvider, useRunRoom } from "@/context/RunRoomContext";
import AsChat from "@/components/chat/AsChat";
import { ContentBlocks } from "@shared/types";
import { useTranslation } from "react-i18next";
import { isMacOs } from "react-device-detect";
import { useMessageApi } from "@/context/MessageApiContext.tsx";
import { useSocket } from "@/context/SocketContext";
import { useFirstTimeGuide } from "@/context/FirstTimeGuideContext";
import { trpc } from "@/api/trpc";
import {
  useNaturalLanguageCommands,
  CommandHandler,
  FormattedResult,
} from "@/hooks/useNaturalLanguageCommands";
import { useStudioSidebar } from "@/context/SidebarContext.tsx";
import { useEnvironmentConfig } from "@/hooks/useEnvironmentConfig";
import { useAuth } from "@/context/AuthContext";
import { useResourceNodeSelection } from "@/hooks/useResourceNodeSelection";

// 桌面端采用更柔和的侧栏宽度，避免中小屏幕被固定 400px 挤压
// 生成随机用户名的函数
const generateRandomUsername = (): string => {
  const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
  let randomPart = "";
  for (let i = 0; i < 4; i++) {
    randomPart += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return `User-${randomPart}`;
};

const CACHE_META_KEYS = {
  datasets: "cached_datasets_meta",
  models: "cached_models_meta",
  tests: "cached_tests_meta",
  evaluationResults: "cached_evaluation_results_meta",
} as const;

const HIDDEN_DATASET_FILES = new Set([
  "preprocessing_audit.json",
  "preprocessing_summary.json",
  "score_audit.json",
  "score_summary.json",
]);

const isVisibleDatasetFile = (filename: string) =>
  !HIDDEN_DATASET_FILES.has(filename.toLowerCase());

const loadCachedMeta = (key: string): ManagementCacheMeta | null => {
  const cached = localStorage.getItem(key);
  if (!cached) return null;

  try {
    return JSON.parse(cached) as ManagementCacheMeta;
  } catch (error) {
    console.error(`Failed to parse cached meta for ${key}:`, error);
    return null;
  }
};

const persistCachedMeta = (
  key: string,
  meta?: ManagementCacheMeta | null,
): void => {
  if (!meta) return;
  localStorage.setItem(key, JSON.stringify(meta));
};

interface TrainingTaskSummary {
  container?: string;
  pid?: string;
  trainType?: string;
  status?: string;
  launchMode?: string;
  isMultinode?: boolean;
  scriptName?: string;
  workflowId?: string;
  wandbUrl?: string;
}

interface WorkflowTaskSummary {
  workflowId?: string;
  workflowStatus?: string;
  workflowUpdatedAt?: number | string;
  workflowDbPath?: string;
  workflowChild?: boolean;
  workflowStage?: string;
  currentStage?: string;
  currentStageStatus?: string;
  currentStageMessage?: string;
  datasetRef?: string;
  evaluationDatasetName?: string;
  container?: string;
  pid?: string;
  modelPath?: string;
  progressPercent?: number;
  trainType?: string;
  status?: string;
  launchMode?: string;
  isMultinode?: boolean;
  scriptName?: string;
  wandbUrl?: string;
  stageStatuses?: Record<string, string>;
  workflowLogs?: Record<string, WorkflowStageLogSummary>;
}

interface WorkflowStageLogSummary {
  stage: string;
  logPath?: string;
  logTail?: string;
  logCommand?: string;
  logUpdatedAt?: number | string;
  stopServiceLogPath?: string;
  stopServiceLogTail?: string;
  stopServiceLogUpdatedAt?: number | string;
}

interface AssessmentTaskSummary {
  container?: string;
  pid?: string;
  script?: string;
  assessmentType?: string;
  assessmentTypeText?: string;
  evalType?: string;
  evalTypeText?: string;
}

interface InferenceInstanceSummary {
  instanceId?: string;
  runtimeNodeId?: string;
  gpus?: string;
  reservationId?: string;
  owner?: string;
  status?: string;
}

interface InferenceTaskSummary {
  sourceKey?: string;
  modelName?: string;
  hostIp?: string;
  inferencePort?: string;
  vllmPort?: string;
  hasConfig?: boolean;
  hasStatus?: boolean;
  preferredView?: InferencePanelView;
  shouldOpenPanel?: boolean;
  stoppedServices?: number;
  runningServices?: number;
  instances?: InferenceInstanceSummary[];
}

interface BenchmarkTaskSummary {
  jobId?: string;
  pid?: string;
  model?: string;
  dataset?: string;
  status?: string;
  resultPath?: string;
}

interface DataFilterTaskSummary {
  sourceKey?: string;
  container?: string;
  outputFolder?: string;
  inputFolder?: string;
  threshold?: number;
  outputDatasetName?: string;
  status?: "running" | "completed" | "failed" | "not_started";
  percent?: number;
  currentFile?: string;
  processedItems?: number;
  totalItems?: number;
  passedItems?: number;
  rejectedItems?: number;
  apiFailedItems?: number;
  invalidItems?: number;
  resumedItems?: number;
  error?: string;
  updatedAt?: string;
  finishedAt?: string;
}

type StatusBarKind =
  | "workflow"
  | "train"
  | "evaluation"
  | "benchmark"
  | "inference"
  | "data_filter";
const WORKFLOW_STAGE_NAMES = [
  "train",
  "evaluate",
  "publish",
  "deploy",
  "benchmark",
] as const;

type InferencePanelView = "config" | "status";

const ADMIN_NL_CAPABILITIES = {
  destructiveResourceActions: [
    /(?:上传|upload).*(?:数据集|评测|evaluation|dataset|test)/i,
    /(?:删除|delete|移除|remove).*(?:数据集|评测|模型|结果|dataset|evaluation|model|result|test)/i,
  ],
  inferenceConfigMutations:
    /(?:修改|应用|保存|modify|apply|save).*(?:推理配置|inference config)/i,
  userEditableInferenceFields: ["model_name"],
  adminOnlyInferenceFields: [
    "vllm_openai_port",
    "inference_port",
    "ui_port",
    "data_annotation_port",
    "host_ip",
    "cuda_visible_devices",
    "model_path",
    "start_script",
    "log_dir",
    "test_dir",
    "benchmark_dir",
    "general_benchmark_dir",
    "tensor_parallel_size",
    "gpu_memory_utilization",
    "max_tokens",
  ],
} as const;

const isAdminOnlyInstruction = (text: string): boolean => {
  const normalized = text.trim().toLowerCase();

  const isUploadOrDelete =
    ADMIN_NL_CAPABILITIES.destructiveResourceActions.some((pattern) =>
      pattern.test(normalized),
    );

  if (isUploadOrDelete) {
    return true;
  }

  const isModifyInferenceConfig =
    ADMIN_NL_CAPABILITIES.inferenceConfigMutations.test(normalized);

  if (!isModifyInferenceConfig) {
    return false;
  }

  const hasUserEditableField =
    ADMIN_NL_CAPABILITIES.userEditableInferenceFields.some((field) =>
      normalized.includes(field),
    );
  if (!hasUserEditableField) {
    return true;
  }

  return ADMIN_NL_CAPABILITIES.adminOnlyInferenceFields.some((field) =>
    normalized.includes(field),
  );
};

const stripUserPrefix = (text: string, username?: string) => {
  if (!username) {
    return text;
  }

  return text.replace(
    new RegExp(`^\\[${username.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\]\\s*`),
    "",
  );
};

const stripUserMarkerFromName = (
  name: string | undefined,
  username?: string,
) => {
  if (!name || !username) {
    return name;
  }

  const escapedUsername = username.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const baseUsername = username.split("#")[0];
  const escapedBaseUsername = baseUsername.replace(
    /[.*+?^${}()|[\]\\]/g,
    "\\$&",
  );

  return name
    .replace(new RegExp(`_\\[${escapedUsername}\\]`, "g"), "")
    .replace(new RegExp(`\\[${escapedUsername}\\]`, "g"), "")
    .replace(new RegExp(`\\[${escapedBaseUsername}\\]`, "g"), "") // 新增：去掉 [baseUsername]
    .replace(new RegExp(`_${escapedUsername}$`), "")
    .replace(new RegExp(`_${escapedBaseUsername}#[a-z0-9]+$`, "i"), "")
    .replace(new RegExp(`_${escapedBaseUsername}$`), "")
    .replace(/_+$/, "")
    .trim(); // 去掉末尾空格（如 "Assistant " → "Assistant"）
};

const parseProtocolFromDisplayText = (
  text: string,
): TrainingJobProtocol | null => {
  const markerIndex = text.lastIndexOf("[协议]");
  if (markerIndex < 0) {
    return null;
  }

  const afterMarker = text.slice(markerIndex + "[协议]".length);
  const jsonStart = afterMarker.indexOf("{");
  if (jsonStart < 0) {
    return null;
  }

  const jsonSource = afterMarker.slice(jsonStart);
  let depth = 0;
  let inString = false;
  let escaped = false;
  let jsonEnd = -1;

  for (let index = 0; index < jsonSource.length; index += 1) {
    const char = jsonSource[index];
    if (escaped) {
      escaped = false;
      continue;
    }
    if (char === "\\") {
      escaped = true;
      continue;
    }
    if (char === '"') {
      inString = !inString;
      continue;
    }
    if (inString) {
      continue;
    }
    if (char === "{") {
      depth += 1;
    }
    if (char === "}") {
      depth -= 1;
      if (depth === 0) {
        jsonEnd = index + 1;
        break;
      }
    }
  }

  if (jsonEnd < 0) {
    return null;
  }

  try {
    return JSON.parse(jsonSource.slice(0, jsonEnd)) as TrainingJobProtocol;
  } catch (error) {
    console.warn("Failed to parse display protocol:", error);
    return null;
  }
};

const normalizeWorkflowDisplayPercent = (
  value: unknown,
): number | undefined => {
  if (value === undefined || value === null || value === "") {
    return undefined;
  }
  const parsed =
    typeof value === "number"
      ? value
      : Number(String(value).replace(/%$/, "").trim());
  if (!Number.isFinite(parsed)) {
    return undefined;
  }
  return Math.max(0, Math.min(100, Math.round(parsed)));
};

const normalizeWorkflowStatusDisplayText = (text: string): string => {
  const protocol = parseProtocolFromDisplayText(text);
  if (protocol?.type !== "workflow_status" || !protocol.message?.trim()) {
    return text;
  }

  let message = protocol.message.trim();
  const stage = protocol.currentStage
    ? protocol.stages?.[protocol.currentStage]
    : undefined;
  const progressPercent = normalizeWorkflowDisplayPercent(
    stage?.progress_percent ?? stage?.progressPercent,
  );

  if (progressPercent !== undefined) {
    const progressLinePattern = /(-\s*进度\s*[:：]\s*)`?[^`\n]*%`?/;
    if (progressLinePattern.test(message)) {
      message = message.replace(
        progressLinePattern,
        `$1\`${progressPercent}%\``,
      );
    } else if (/当前阶段详情[:：]/.test(message)) {
      message = message.replace(
        /(当前阶段详情[:：]\s*)/,
        `$1\n- 进度：\`${progressPercent}%\``,
      );
    }
  }

  if (stage?.message && !message.includes(stage.message)) {
    message = message.replace(
      /(当前阶段详情[:：]\s*)/,
      `$1\n- 阶段说明：\`${stage.message}\``,
    );
  }

  return message;
};

const sanitizeContentForDisplay = (
  content: ContentType,
  username?: string,
): ContentType => {
  if (!username) {
    if (typeof content === "string") {
      return normalizeWorkflowStatusDisplayText(content);
    }
    if (Array.isArray(content)) {
      return content.map((block) => {
        if (block.type === "text") {
          return {
            ...block,
            text: normalizeWorkflowStatusDisplayText(
              (block as { text: string }).text,
            ),
          } as ContentBlock;
        }
        return block;
      }) as ContentType;
    }
    return content;
  }

  if (typeof content === "string") {
    return normalizeWorkflowStatusDisplayText(
      stripUserPrefix(content, username),
    );
  }

  if (Array.isArray(content)) {
    return content.map((block) => {
      if (block.type === "text") {
        return {
          ...block,
          text: normalizeWorkflowStatusDisplayText(
            stripUserPrefix((block as { text: string }).text, username),
          ),
        } as ContentBlock;
      }
      return block;
    }) as ContentType;
  }

  return content;
};

const sanitizeRepliesForDisplay = (replies: Reply[], username?: string) =>
  replies.map((reply) => ({
    ...reply,
    replyName: stripUserMarkerFromName(reply.replyName, username),
    messages: reply.messages.map((msg) => ({
      ...msg,
      name: stripUserMarkerFromName(msg.name, username),
      content: sanitizeContentForDisplay(msg.content, username),
    })),
  }));

const getReplyTime = (reply: Reply): number => {
  const timestamp = reply.createdAt || reply.messages?.[0]?.timestamp;
  const parsed = timestamp ? Date.parse(timestamp) : Number.NaN;
  return Number.isNaN(parsed) ? 0 : parsed;
};

const getAssistantMessageIds = (replies: Reply[]): string[] =>
  replies.flatMap((reply) =>
    reply.replyRole.toLowerCase() === "assistant"
      ? (reply.messages || [])
          .filter((message) => message.role === "assistant" && message.id)
          .map((message) => message.id)
      : [],
  );

const getAssistantReplyCount = (replies: Reply[]): number =>
  replies.filter((reply) => reply.replyRole.toLowerCase() === "assistant")
    .length;

const summarizePendingReplies = (replies: Reply[]) =>
  replies.slice(-5).map((reply) => ({
    replyId: reply.replyId,
    replyRole: reply.replyRole,
    createdAt: reply.createdAt,
    finishedAt: reply.finishedAt,
    messages: (reply.messages || []).map((message) => ({
      id: message.id,
      role: message.role,
      name: message.name,
      timestamp: message.timestamp,
    })),
  }));

const hasAssistantMessageAfterSend = (
  replies: Reply[],
  sentAt: number,
  assistantMessageIdsBeforeSend: string[],
): boolean => {
  const seenBeforeSend = new Set(assistantMessageIdsBeforeSend);

  return replies.some((reply) => {
    if (reply.replyRole.toLowerCase() !== "assistant") {
      return false;
    }

    return reply.messages?.some((messageItem) => {
      if (messageItem.role !== "assistant") {
        return false;
      }

      const timestamp =
        messageItem.timestamp || reply.finishedAt || reply.createdAt;
      const messageTime = timestamp ? Date.parse(timestamp) : Number.NaN;
      if (Number.isFinite(messageTime)) {
        const isCurrentReply = messageTime > sentAt;
        return isCurrentReply;
      }

      const isCurrentReply = false;
      return isCurrentReply;
    });
  });
};

const DEFAULT_CHAT_SESSION_ID = "default";

const buildContextUsername = (username: string, sessionId: string) =>
  sessionId === DEFAULT_CHAT_SESSION_ID ? username : `${username}#${sessionId}`;

const getMetadataString = (
  metadata: object | undefined,
  key: string,
): string | undefined => {
  const value = (metadata as Record<string, unknown> | undefined)?.[key];
  return typeof value === "string" ? value : undefined;
};

const nameMatchesContextUsername = (
  name: string | undefined,
  username: string,
): boolean => {
  if (!name) {
    return false;
  }

  if (name.includes(`[${username}]`)) {
    return true;
  }

  if (username.includes("#")) {
    return name.includes(username);
  }

  return name === username || name.endsWith(`_${username}`);
};

const messageMatchesContextUsername = (
  contentText: string,
  name: string | undefined,
  metadata: object | undefined,
  username: string,
): boolean => {
  const metadataContextUsername = getMetadataString(
    metadata,
    "__medflowContextUsername",
  );
  if (metadataContextUsername === username) {
    return true;
  }

  const metadataUsername = getMetadataString(metadata, "__medflowUsername");
  if (!username.includes("#") && metadataUsername === username) {
    return true;
  }

  return (
    contentText.startsWith(`[${username}]`) ||
    nameMatchesContextUsername(name, username)
  );
};

interface InferenceProtocolConfig {
  ports?: {
    VLLM_OPENAI_PORT?: number | string;
    INFERENCE_PORT?: number | string;
    UI_PORT?: number | string;
    DATA_ANNOTATION_PORT?: number | string;
  };
  env?: {
    HOST_IP?: string;
    CUDA_VISIBLE_DEVICES?: string;
    MODEL_NAME?: string;
    MODEL_PATH?: string;
    START_SCRIPT?: string;
    LOG_DIR?: string;
    TEST_DIR?: string;
    BENCHMARK_DIR?: string;
    GENERAL_BENCHMARK_DIR?: string;
  };
  runtime?: {
    TENSOR_PARALLEL_SIZE?: number | string;
    GPU_MEMORY_UTILIZATION?: number | string;
    MAX_TOKENS?: number | string;
  };
}

interface InferenceServiceInstanceItem {
  instance_id?: string;
  instanceId?: string;
  runtime_node_id?: string;
  runtimeNodeId?: string;
  reservation_id?: string;
  reservationId?: string;
  owner?: string;
  status?: string;
  gpus?: Array<string | number> | string;
  assigned_gpus?: Array<string | number> | string;
  assignedGpus?: Array<string | number> | string;
  node?: string;
  resource?: {
    runtime_node_id?: string;
    runtimeNodeId?: string;
    reservation_id?: string;
    reservationId?: string;
    assigned_gpus?: Array<string | number> | string;
    assignedGpus?: Array<string | number> | string;
  };
}

interface InferenceServiceInstancesPayload {
  operation?: string;
  items?: InferenceServiceInstanceItem[];
  summary?: Record<string, number | string | undefined>;
}

interface BenchmarkStopPayload {
  operation?: string;
  stopped?: boolean;
  status?: string;
  job_id?: string;
  jobId?: string;
}

interface TrainingJobProtocol {
  type?: string;
  jobType?: string;
  agent?: string;
  message?: string;
  errorReason?: string;
  errorRecoverable?: boolean;
  action?: string;
  benchmark_stop?: BenchmarkStopPayload;
  config?: InferenceProtocolConfig;
  service_instances?: InferenceServiceInstancesPayload;
  serviceInstances?: InferenceServiceInstancesPayload;
  service_instance?: Record<string, unknown>;
  serviceInstance?: Record<string, unknown>;
  nodes?: Record<
    string,
    {
      config?: InferenceProtocolConfig;
      services?: Array<{
        name?: string;
        port?: number | string;
        status?: string;
        rawStatus?: string;
        node?: string;
      }>;
      service_instances?: InferenceServiceInstancesPayload;
      serviceInstances?: InferenceServiceInstancesPayload;
      service_instance?: Record<string, unknown>;
      serviceInstance?: Record<string, unknown>;
      allStopped?: boolean;
      allRunning?: boolean;
    }
  >;
  services?: Array<{
    name?: string;
    port?: number | string;
    status?: string;
    rawStatus?: string;
  }>;
  allStopped?: boolean;
  allRunning?: boolean;
  container?: string;
  container_name?: string;
  pid?: string | number;
  trainType?: string;
  trainTypeEn?: string;
  trainTypeText?: string;
  launchMode?: string;
  isMultinode?: boolean;
  workflowId?: string;
  workflowStatus?: string;
  workflowUpdatedAt?: number | string;
  workflowDbPath?: string;
  currentStage?: string;
  datasetRef?: string;
  evaluationDatasetName?: string;
  stages?: Record<
    string,
    {
      status?: string;
      container?: string;
      container_name?: string;
      pid?: string | number;
      trainType?: string;
      train_type?: string;
      launchMode?: string;
      launch_mode?: string;
      isMultinode?: boolean;
      is_multinode?: boolean;
      scriptName?: string;
      script_name?: string;
      message?: string;
      model_path?: string;
      modelPath?: string;
      progress_percent?: number | string;
      progressPercent?: number | string;
      metrics?: {
        container?: string;
        container_name?: string;
        pid?: string | number;
        trainType?: string;
        train_type?: string;
        launchMode?: string;
        launch_mode?: string;
        isMultinode?: boolean;
        is_multinode?: boolean;
        scriptName?: string;
        script_name?: string;
        wandb_url?: string;
        wandbUrl?: string;
      };
      result?: {
        log_path?: string;
        logPath?: string;
        log_tail?: string;
        logTail?: string;
        log_command?: string;
        logCommand?: string;
        log_updated_at?: number | string;
        logUpdatedAt?: number | string;
        stop_service_log_path?: string;
        stopServiceLogPath?: string;
        stop_service_log_tail?: string;
        stopServiceLogTail?: string;
        stop_service_log_updated_at?: number | string;
        stopServiceLogUpdatedAt?: number | string;
      };
      log_path?: string;
      logPath?: string;
      log_tail?: string;
      logTail?: string;
      log_command?: string;
      logCommand?: string;
      log_updated_at?: number | string;
      logUpdatedAt?: number | string;
      stop_service_log_path?: string;
      stopServiceLogPath?: string;
      stop_service_log_tail?: string;
      stopServiceLogTail?: string;
      stop_service_log_updated_at?: number | string;
      stopServiceLogUpdatedAt?: number | string;
    }
  >;
  model_path?: string;
  modelPath?: string;
  progress_percent?: number | string;
  progressPercent?: number | string;
  script?: string;
  scriptName?: string;
  assessmentType?: string;
  assessmentTypeText?: string;
  evalType?: string;
  evalTypeText?: string;
  model?: string;
  dataset?: string;
  status?: string;
  jobId?: string;
  job_id?: string;
  benchmark_job_id?: string;
  outputFolder?: string;
  output_folder?: string;
  inputFolder?: string;
  input_folder?: string;
  outputDatasetName?: string;
  output_dataset_name?: string;
  threshold?: number | string;
  command?: string;
  scriptName?: string;
  scriptArgs?: Record<string, string | number | boolean | undefined>;
  commands?: string[];
  wandbUrl?: string;
  wandbMode?: string;
  wandbUrlPending?: boolean;
}

const isTrainingJobProtocol = (
  value: unknown,
): value is TrainingJobProtocol => {
  if (!value || typeof value !== "object") {
    return false;
  }
  const protocol = value as TrainingJobProtocol;
  return typeof protocol.type === "string";
};

const parseProtocolLikeValue = (value: unknown): TrainingJobProtocol | null => {
  if (typeof value === "string") {
    try {
      return parseProtocolLikeValue(JSON.parse(value));
    } catch {
      return null;
    }
  }
  if (isTrainingJobProtocol(value)) {
    return value;
  }
  if (!value || typeof value !== "object") {
    return null;
  }
  const protocol = (value as { protocol?: unknown }).protocol;
  if (isTrainingJobProtocol(protocol)) {
    return protocol;
  }
  const dataProtocol = (value as { data?: { protocol?: unknown } }).data
    ?.protocol;
  return isTrainingJobProtocol(dataProtocol) ? dataProtocol : null;
};

const extractProtocolFromMetadata = (
  metadata: object | null | undefined,
): TrainingJobProtocol | null => {
  if (!metadata || typeof metadata !== "object") {
    return null;
  }

  return parseProtocolLikeValue(metadata);
};

const isWorkflowProtocol = (protocol: TrainingJobProtocol | null): boolean =>
  Boolean(
    protocol?.type?.startsWith("workflow_") ||
    protocol?.workflowId ||
    protocol?.workflowChild,
  );

const isWorkflowDismissStatus = (status?: string): boolean =>
  [
    "stopped",
    "cancelled",
    "canceled",
    "finished",
    "complete",
    "completed",
    "done",
    "success",
    "succeeded",
  ].includes((status || "").trim().toLowerCase());

const inferTrainingLaunchMode = (
  protocol: TrainingJobProtocol | null,
): string | undefined => {
  if (protocol?.launchMode) {
    return protocol.launchMode;
  }
  if (
    protocol?.isMultinode ||
    protocol?.script?.startsWith("train_multinode_") ||
    protocol?.scriptName?.startsWith("train_multinode_") ||
    protocol?.trainTypeEn?.startsWith("multinode_") ||
    protocol?.trainTypeText?.includes("多机") ||
    protocol?.trainTypeText?.includes("双机")
  ) {
    return "multinode";
  }
  return undefined;
};

const normalizeTrainingTaskStatus = (
  value: unknown,
  fallback: string = "running",
): string => {
  const normalized = String(value || "").trim().toLowerCase();
  if (!normalized) {
    return fallback;
  }
  if (/preparing|prepare|准备/.test(normalized)) {
    return "preparing";
  }
  if (/starting|启动/.test(normalized)) {
    return "starting";
  }
  if (/running|运行/.test(normalized)) {
    return "running";
  }
  if (/interrupted|interrupt|中断/.test(normalized)) {
    return "interrupted";
  }
  if (/^(?:stop|stopped|cancelled|canceled)$|已停止|停止|取消/.test(normalized)) {
    return "stopped";
  }
  if (/finished|complete|completed|done|success|succeeded|完成|成功/.test(normalized)) {
    return "finished";
  }
  if (/failed|failure|fail|error|失败|异常|错误/.test(normalized)) {
    return "failed";
  }
  return fallback;
};

const isTrainingTerminalStatus = (status?: string): boolean =>
  [
    "interrupted",
    "stopped",
    "cancelled",
    "canceled",
    "finished",
    "complete",
    "completed",
    "done",
    "success",
    "succeeded",
    "failed",
    "error",
  ].includes((status || "").trim().toLowerCase());

const isTrainingTerminalProtocol = (
  protocol: TrainingJobProtocol | null,
): boolean => {
  if (!protocol || isWorkflowProtocol(protocol)) {
    return false;
  }
  if (protocol.type === "job_stopped" && protocol.jobType === "train") {
    return true;
  }
  if (
    ["job_started", "job_preparing", "monitor_status"].includes(protocol.type) &&
    protocol.jobType === "train"
  ) {
    return isTrainingTerminalStatus(
      normalizeTrainingTaskStatus(protocol.status, ""),
    );
  }
  return false;
};

const trainingSummaryFromProtocol = (
  protocol: TrainingJobProtocol | null,
): TrainingTaskSummary | null => {
  if (
    protocol?.type !== "job_started" &&
    protocol?.type !== "job_preparing" &&
    protocol?.type !== "monitor_status"
  ) {
    return null;
  }
  if (protocol.jobType !== "train") {
    return null;
  }

  const pid = protocol.pid === undefined ? undefined : String(protocol.pid);
  const wandbUrl =
    typeof protocol.wandbUrl === "string" && protocol.wandbUrl.trim()
      ? protocol.wandbUrl.trim()
      : undefined;
  const container = protocol.container || protocol.container_name;
  if (!container && !pid && !wandbUrl) {
    return null;
  }

  const status = protocol.type === "job_preparing"
    ? "preparing"
    : normalizeTrainingTaskStatus(protocol.status, "running");
  if (isTrainingTerminalStatus(status)) {
    return null;
  }

  return {
    container,
    pid,
    trainType: protocol.trainTypeEn || protocol.trainType || protocol.trainTypeText,
    status,
    launchMode: inferTrainingLaunchMode(protocol),
    isMultinode: Boolean(protocol.isMultinode),
    scriptName: protocol.scriptName || protocol.script,
    wandbUrl,
  };
};

const isExplicitNonTrainingJobProtocol = (
  protocol: TrainingJobProtocol | null,
): boolean => {
  if (
    protocol?.type !== "job_started" &&
    protocol?.type !== "job_preparing" &&
    protocol?.type !== "monitor_status"
  ) {
    return false;
  }
  return Boolean(protocol.jobType && protocol.jobType !== "train");
};

const normalizePercent = (value: unknown): number | undefined => {
  if (value === undefined || value === null || value === "") {
    return undefined;
  }
  const parsed =
    typeof value === "number"
      ? value
      : Number(String(value).replace(/%$/, "").trim());
  if (!Number.isFinite(parsed)) {
    return undefined;
  }
  return Math.max(0, Math.min(100, Math.round(parsed)));
};

const normalizeDataFilterStatus = (
  value: unknown,
): DataFilterTaskSummary["status"] => {
  const status = String(value || "").trim().toLowerCase();
  if (["completed", "complete", "done", "finished", "success", "succeeded"].includes(status)) {
    return "completed";
  }
  if (["failed", "fail", "error"].includes(status)) {
    return "failed";
  }
  if (["running", "processing", "in_progress", "started", "starting"].includes(status)) {
    return "running";
  }
  return "not_started";
};

const workflowSummaryFromProtocol = (
  protocol: TrainingJobProtocol | null,
): WorkflowTaskSummary | null => {
  if (!isWorkflowProtocol(protocol) || !protocol?.workflowId) {
    return null;
  }

  const currentStage = protocol.currentStage;
  const stage = currentStage ? protocol.stages?.[currentStage] : undefined;
  const stageMetrics = stage?.metrics;
  const wandbUrl =
    typeof stageMetrics?.wandb_url === "string" && stageMetrics.wandb_url.trim()
      ? stageMetrics.wandb_url.trim()
      : typeof stageMetrics?.wandbUrl === "string" &&
          stageMetrics.wandbUrl.trim()
        ? stageMetrics.wandbUrl.trim()
        : undefined;
  const fallbackPid =
    protocol.pid === undefined ? undefined : String(protocol.pid);
  const metricsPid = stageMetrics?.pid;
  const pid =
    stage?.pid === undefined
      ? metricsPid === undefined
        ? fallbackPid
        : String(metricsPid)
      : String(stage.pid);
  const progressPercent = normalizePercent(
    stage?.progress_percent ??
      stage?.progressPercent ??
      protocol.progress_percent ??
      protocol.progressPercent,
  );
  const stageStatuses = Object.fromEntries(
    Object.entries(protocol.stages || {}).map(([name, value]) => [
      name,
      value.status || "pending",
    ]),
  );
  const workflowLogs = Object.fromEntries(
    Object.entries(protocol.stages || {})
      .map(([name, value]) => {
        const result = value.result || {};
        const log: WorkflowStageLogSummary = {
          stage: name,
          logPath: value.log_path || value.logPath || result.log_path || result.logPath,
          logTail: value.log_tail || value.logTail || result.log_tail || result.logTail,
          logCommand:
            value.log_command ||
            value.logCommand ||
            result.log_command ||
            result.logCommand,
          logUpdatedAt:
            value.log_updated_at ||
            value.logUpdatedAt ||
            result.log_updated_at ||
            result.logUpdatedAt,
          stopServiceLogPath:
            value.stop_service_log_path ||
            value.stopServiceLogPath ||
            result.stop_service_log_path ||
            result.stopServiceLogPath,
          stopServiceLogTail:
            value.stop_service_log_tail ||
            value.stopServiceLogTail ||
            result.stop_service_log_tail ||
            result.stopServiceLogTail,
          stopServiceLogUpdatedAt:
            value.stop_service_log_updated_at ||
            value.stopServiceLogUpdatedAt ||
            result.stop_service_log_updated_at ||
            result.stopServiceLogUpdatedAt,
        };
        return [name, log] as const;
      })
      .filter(([, log]) =>
        Boolean(
          log.logPath ||
            log.logTail ||
            log.stopServiceLogPath ||
            log.stopServiceLogTail,
        ),
      ),
  );
  if (currentStage && !stageStatuses[currentStage]) {
    stageStatuses[currentStage] = protocol.status || "running";
  }

  return {
    workflowId: protocol.workflowId,
    workflowStatus: protocol.workflowStatus || "running",
    workflowUpdatedAt: protocol.workflowUpdatedAt,
    workflowDbPath: protocol.workflowDbPath,
    currentStage,
    currentStageStatus: stage?.status || protocol.status,
    currentStageMessage: stage?.message || protocol.message,
    datasetRef: protocol.datasetRef,
    evaluationDatasetName: protocol.evaluationDatasetName,
    container:
      stage?.container ||
      stage?.container_name ||
      stageMetrics?.container_name ||
      stageMetrics?.container ||
      protocol.container,
    pid,
    modelPath:
      stage?.model_path ||
      stage?.modelPath ||
      protocol.model_path ||
      protocol.modelPath,
    progressPercent,
    trainType:
      stage?.trainType ||
      stage?.train_type ||
      stageMetrics?.trainType ||
      stageMetrics?.train_type ||
      protocol.trainType ||
      protocol.trainTypeText,
    launchMode:
      stage?.launchMode ||
      stage?.launch_mode ||
      stageMetrics?.launchMode ||
      stageMetrics?.launch_mode ||
      inferTrainingLaunchMode(protocol),
    isMultinode:
      stage?.isMultinode ??
      stage?.is_multinode ??
      stageMetrics?.isMultinode ??
      stageMetrics?.is_multinode ??
      protocol.isMultinode,
    scriptName:
      stage?.scriptName ||
      stage?.script_name ||
      stageMetrics?.scriptName ||
      stageMetrics?.script_name ||
      protocol.scriptName ||
      protocol.script,
    wandbUrl,
    stageStatuses,
    workflowLogs,
  };
};

const WORKFLOW_STAGE_STATUS_RANK: Record<string, number> = {
  pending: 0,
  awaiting_agent: 1,
  starting_external: 2,
  starting: 2,
  running: 3,
  timeout: 3,
  stopping: 4,
  finished: 5,
  failed: 5,
  stopped: 5,
};

const mergeWorkflowTaskSummary = (
  current: WorkflowTaskSummary | null,
  incoming: WorkflowTaskSummary,
): WorkflowTaskSummary => {
  if (!current || current.workflowId !== incoming.workflowId) {
    return incoming;
  }

  // A persisted terminal status must not be overwritten by an older running
  // chat snapshot for the same workflow. New workflow ids still replace above.
  if (isWorkflowDismissStatus(incoming.workflowStatus)) {
    return incoming;
  }
  if (isWorkflowDismissStatus(current.workflowStatus)) {
    return current;
  }
  if (
    current.workflowStatus !== "running" ||
    incoming.workflowStatus !== "running"
  ) {
    return incoming;
  }
  const currentStageIndex = WORKFLOW_STAGE_NAMES.indexOf(
    current.currentStage as (typeof WORKFLOW_STAGE_NAMES)[number],
  );
  const incomingStageIndex = WORKFLOW_STAGE_NAMES.indexOf(
    incoming.currentStage as (typeof WORKFLOW_STAGE_NAMES)[number],
  );
  if (
    currentStageIndex >= 0 &&
    incomingStageIndex >= 0 &&
    incomingStageIndex < currentStageIndex
  ) {
    const hasNewTaskIdentity =
      incoming.currentStage === current.currentStage &&
      ((incoming.pid && incoming.pid !== current.pid) ||
        (incoming.container && incoming.container !== current.container));
    if (!hasNewTaskIdentity) {
      return current;
    }
  }

  const stageStatuses = { ...(incoming.stageStatuses || {}) };
  Object.entries(current.stageStatuses || {}).forEach(([stage, status]) => {
    const incomingStatus = stageStatuses[stage];
    if (
      !incomingStatus ||
      (WORKFLOW_STAGE_STATUS_RANK[incomingStatus] ?? 0) <
        (WORKFLOW_STAGE_STATUS_RANK[status] ?? 0)
    ) {
      stageStatuses[stage] = status;
    }
  });
  const workflowLogs = {
    ...(current.workflowLogs || {}),
    ...(incoming.workflowLogs || {}),
  };
  const hasNewTaskIdentity =
    incoming.currentStage === current.currentStage &&
    ((incoming.pid && incoming.pid !== current.pid) ||
      (incoming.container && incoming.container !== current.container));

  if (incomingStageIndex > currentStageIndex) {
    return { ...incoming, stageStatuses, workflowLogs };
  }

  const currentStatusRank =
    WORKFLOW_STAGE_STATUS_RANK[current.currentStageStatus || ""] ?? 0;
  const incomingStatusRank =
    WORKFLOW_STAGE_STATUS_RANK[incoming.currentStageStatus || ""] ?? 0;
  const stageRegressed = incomingStatusRank < currentStatusRank;
  if (stageRegressed) {
    if (hasNewTaskIdentity) {
      return { ...current, ...incoming, stageStatuses, workflowLogs };
    }
    return { ...current, stageStatuses, workflowLogs };
  }

  if (hasNewTaskIdentity) {
    return { ...current, ...incoming, stageStatuses, workflowLogs };
  }

  const progressPercent =
    current.progressPercent === undefined
      ? incoming.progressPercent
      : incoming.progressPercent === undefined
        ? current.progressPercent
        : Math.max(current.progressPercent, incoming.progressPercent);

  return {
    ...current,
    ...incoming,
    progressPercent,
    stageStatuses,
    workflowLogs,
  };
};

const assessmentSummaryFromProtocol = (
  protocol: TrainingJobProtocol | null,
): AssessmentTaskSummary | null => {
  if (
    (protocol?.type !== "job_started" &&
      protocol?.type !== "monitor_status") ||
    !["assessment", "evaluate"].includes(protocol.jobType || "")
  ) {
    return null;
  }

  const pid = protocol.pid === undefined ? undefined : String(protocol.pid);
  const container = protocol.container || protocol.container_name;
  if (!container && !pid) {
    return null;
  }

  return {
    container,
    pid,
    script: protocol.script,
    assessmentType: protocol.assessmentType || protocol.evalType,
    assessmentTypeText: protocol.assessmentTypeText || protocol.evalTypeText,
    evalType: protocol.evalType,
    evalTypeText: protocol.evalTypeText,
  };
};

const extractMarkdownValue = (
  text: string,
  label: string,
): string | undefined => {
  const escapedLabel = label.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return text
    .match(
      new RegExp(
        `(?:\\*\\*)?${escapedLabel}(?:\\*\\*)?\\s*[:：]\\s*` +
          "`?([^`\\n，,]+)",
        "i",
      ),
    )?.[1]
    ?.trim();
};

const benchmarkSummaryFromText = (
  text: string,
): BenchmarkTaskSummary | null => {
  if (/一键工作流|one-click workflow|workflowId|workflow_/i.test(text)) {
    return null;
  }

  if (
    /(?:最终结果|结果文件路径|完整结果|以上为最终结果|任务已运行完毕|任务已结束)/i.test(text) &&
    /(?:结果|result)/i.test(text)
  ) {
    return null;
  }

  if (!/推理基准测试|基准测试|inference_benchmark|benchmark/i.test(text)) {
    return null;
  }

  const hasRuntimeSignal =
    /(?:任务ID|jobId|job_id|进程ID|PID|任务状态|中间结果路径|resultPath|result_path)/i.test(
      text,
    ) ||
    /(?:已启动|启动成功|正在运行|运行中|执行中|进度|完成|失败).{0,24}(?:基准测试|benchmark)/i.test(
      text,
    ) ||
    /(?:基准测试|benchmark).{0,24}(?:已启动|启动成功|正在运行|运行中|执行中|进度|完成|失败)/i.test(
      text,
    );
  if (!hasRuntimeSignal) {
    return null;
  }

  const jobId = extractMarkdownValue(text, "任务ID");
  const pid =
    extractMarkdownValue(text, "进程ID (PID)") ||
    extractMarkdownValue(text, "PID");
  const model = extractMarkdownValue(text, "模型");
  const dataset = extractMarkdownValue(text, "数据集");
  const status = extractMarkdownValue(text, "任务状态");
  const resultPath = extractMarkdownValue(text, "中间结果路径");

  if (!jobId && !pid && !model && !dataset && !status && !resultPath) {
    return null;
  }

  if (!jobId && !pid && !status && !resultPath && dataset) {
    return null;
  }

  return {
    jobId,
    pid,
    model,
    dataset,
    status,
    resultPath,
  };
};

const benchmarkSummaryFromProtocol = (
  protocol: TrainingJobProtocol | null,
): BenchmarkTaskSummary | null => {
  if (!protocol || protocol.jobType !== "inference_benchmark") {
    return null;
  }

  if (
    protocol.type !== "job_started" &&
    protocol.type !== "inference_benchmark_progress" &&
    protocol.type !== "inference_benchmark_status"
  ) {
    return null;
  }

  const messageSummary = protocol.message
    ? benchmarkSummaryFromText(protocol.message)
    : null;
  const pid =
    protocol.pid === undefined ? messageSummary?.pid : String(protocol.pid);

  return {
    ...messageSummary,
    jobId:
      protocol.jobId ||
      protocol.job_id ||
      protocol.benchmark_job_id ||
      messageSummary?.jobId,
    pid,
    model: protocol.model || messageSummary?.model,
    dataset: protocol.dataset || messageSummary?.dataset,
    status: protocol.status || messageSummary?.status,
  };
};

const isBenchmarkStopProtocol = (
  protocol: TrainingJobProtocol | null,
): boolean => {
  if (!protocol || protocol.jobType !== "inference_benchmark") {
    return false;
  }

  if (protocol.type === "job_stopped") {
    return true;
  }

  return Boolean(
    protocol.type === "inference_benchmark_stop_result" &&
      protocol.action === "benchmark_stop" &&
      protocol.benchmark_stop?.stopped === true,
  );
};

const isBenchmarkTaskStopped = (text: string): boolean =>
  isBenchmarkStopProtocol(extractProtocolFromText(text));

const extractTextFromContent = (content: ContentType): string => {
  if (typeof content === "string") {
    return content;
  }

  return content
    .map((block) => {
      if (block.type === "text") {
        return (block as { text?: string }).text || "";
      }
      return "";
    })
    .join("\n");
};

const extractProtocolFromText = (text: string): TrainingJobProtocol | null => {
  const markerIndex = text.lastIndexOf("[协议]");
  if (markerIndex < 0) {
    const trimmedText = text.trim();
    if (!trimmedText.startsWith("{")) {
      return null;
    }
    try {
      return parseProtocolLikeValue(JSON.parse(trimmedText));
    } catch {
      return null;
    }
  }

  const afterMarker = text.slice(markerIndex + "[协议]".length);
  const jsonStart = afterMarker.indexOf("{");
  if (jsonStart < 0) {
    return null;
  }

  const jsonSource = afterMarker.slice(jsonStart);
  let depth = 0;
  let inString = false;
  let escaped = false;
  let jsonEnd = -1;

  for (let index = 0; index < jsonSource.length; index += 1) {
    const char = jsonSource[index];
    if (escaped) {
      escaped = false;
      continue;
    }
    if (char === "\\") {
      escaped = true;
      continue;
    }
    if (char === '"') {
      inString = !inString;
      continue;
    }
    if (inString) {
      continue;
    }
    if (char === "{") {
      depth += 1;
    }
    if (char === "}") {
      depth -= 1;
      if (depth === 0) {
        jsonEnd = index + 1;
        break;
      }
    }
  }

  if (jsonEnd < 0) {
    return null;
  }

  try {
    return JSON.parse(jsonSource.slice(0, jsonEnd)) as TrainingJobProtocol;
  } catch (error) {
    console.warn("Failed to parse training job protocol:", error);
    return null;
  }
};

const parseTrainingTaskSummary = (text: string): TrainingTaskSummary | null => {
  const protocol = extractProtocolFromText(text);
  if (
    isWorkflowProtocol(protocol) ||
    isExplicitNonTrainingJobProtocol(protocol)
  ) {
    return null;
  }

  const protocolSummary = trainingSummaryFromProtocol(protocol);
  if (protocolSummary) {
    return protocolSummary;
  }

  if (
    /(?:评估类型|评测状态|评估状态|查询评估状态|查询评测状态|评测任务|评估任务)/.test(
      text,
    ) ||
    /(?:single_model_evaluation|compare_between_models|checkpoint(?:_evaluation|评估)?|ckpt评估)/i.test(
      text,
    )
  ) {
    return null;
  }

  const textStatusMatch =
    text.match(/当前状态\*{0,2}\s*[:：]\s*\*{0,2}([^*\n]+)/)?.[1] ||
    text.match(/状态为\s*[:：]?\s*\*{0,2}([^*。\n]+)/)?.[1] ||
    text.match(/status\s*[:：]\s*`?([^`\n，,]+)/i)?.[1];
  const parsedStatus = normalizeTrainingTaskStatus(
    textStatusMatch,
    /训练任务已提交.{0,32}(?:准备|尚未确认训练启动)/.test(text)
      ? "preparing"
      : "running",
  );
  const hasStartedText =
    text.includes("脚本已在Docker容器中后台启动运行") ||
    /训练任务已提交.{0,32}(?:准备|尚未确认训练启动)/.test(text) ||
    /训练任务已成功在后台启动/.test(text) ||
    /训练正在后台进行中/.test(text) ||
    /训练任务监控结果/.test(text) ||
    /当前训练任务.{0,80}状态为\s*[:：]?\s*\*{0,2}(?:启动中|运行中|starting|running|interrupted|stopped|finished|completed|failed|error|中断|停止|完成|失败)/i.test(text) ||
    /当前状态\*{0,2}\s*[:：]\s*\*{0,2}(?:启动中|运行中|starting|running|interrupted|stopped|finished|completed|failed|error|中断|停止|完成|失败)/i.test(text) ||
    /训练任务已启动/.test(text) ||
    /(?:开始|启动|运行).{0,24}(?:训练|微调)/.test(text);
  if (!hasStartedText) {
    return null;
  }

  const container = text.match(
    /(?:(?:运行)?容器(?:名称)?|container)\s*[:：]\s*`?([^\s`\n，,]+)/i,
  )?.[1];
  const pid = text.match(
    /(?:PID|进程ID\s*(?:\(\s*PID\s*\))?)\s*[:：]\s*`?([0-9]+)/i,
  )?.[1];
  const trainTypeText = text.match(
    /(双机\s*LoRA\s*SFT|多机\s*LoRA\s*SFT|LoRA\s*SFT|全参\s*SFT|多机lora批量训练|lora批量训练|全参批量训练|双机增强训练|多机增强训练|增强训练|grpo训练|multinode_lora_sft|multinode_enhanced)/i,
  )?.[1];
  const normalizedTrainType = trainTypeText
    ? trainTypeText.replace(/\s+/g, "").toLowerCase()
    : "";
  const trainType =
    normalizedTrainType === "lorasft"
      ? "lora"
      : normalizedTrainType === "全参sft"
        ? "full"
        : trainTypeText;
  const scriptName =
    text.match(/脚本(?:名称)?\s*[:：]\s*`?([^\s`\n，,]+)/)?.[1] || "";
  const hasTrainingSignal =
    Boolean(trainType) ||
    /(?:train|training|finetune|lora|grpo)/i.test(scriptName) ||
    /训练任务监控结果/.test(text) ||
    /当前训练任务.{0,80}状态为\s*[:：]?\s*\*{0,2}(?:启动中|运行中|starting|running|interrupted|stopped|finished|completed|failed|error|中断|停止|完成|失败)/i.test(text) ||
    /当前状态\*{0,2}\s*[:：]\s*\*{0,2}(?:启动中|运行中|starting|running|interrupted|stopped|finished|completed|failed|error|中断|停止|完成|失败)/i.test(text) ||
    /(?:开始|启动|运行).{0,24}(?:训练|微调)/.test(text);

  if ((!container && !pid) || !hasTrainingSignal) {
    return null;
  }

  return {
    container,
    pid,
    trainType,
    status: parsedStatus,
    launchMode:
      scriptName.startsWith("train_multinode_") ||
      /(?:多机|双机|qingnang_train_multi)/.test(text)
        ? "multinode"
        : undefined,
    isMultinode:
      scriptName.startsWith("train_multinode_") ||
      /(?:多机|双机|qingnang_train_multi)/.test(text),
    scriptName,
  };
};

const parseAssessmentTaskSummary = (
  text: string,
): AssessmentTaskSummary | null => {
  if (/一键工作流|one-click workflow|workflowId|workflow_/i.test(text)) {
    return null;
  }

  const protocolSummary = assessmentSummaryFromProtocol(
    extractProtocolFromText(text),
  );
  if (protocolSummary) {
    return protocolSummary;
  }

  const hasStartedText =
    text.includes("请稍后使用该PID或容器名称查询评测状态") ||
    /评(?:估|测)任务已启动/.test(text) ||
    /(?:开始|启动|运行|执行).{0,24}评(?:估|测)/.test(text);
  if (!hasStartedText) {
    return null;
  }

  const container = text.match(
    /(?:容器(?:名称)?|container)\s*[:：]\s*`?([^\s`\n，,]+)/i,
  )?.[1];
  const pid = text.match(
    /(?:PID|进程ID\s*(?:\(\s*PID\s*\))?)\s*[:：]\s*`?([0-9]+)/i,
  )?.[1];
  const script =
    text.match(/脚本(?:名称)?\s*[:：]\s*`?([^\s`\n，,]+)/)?.[1] || undefined;
  const assessmentTypeText =
    text.match(/(?:assessment|评估)类型\s*[:：]\s*`?([^\s`\n，,]+)/i)?.[1] ||
    undefined;
  const hasEvaluationSignal =
    Boolean(pid) ||
    /(?:evaluate|evaluation|eval|compare_between_models|single_model|ckpt)/i.test(
      script || "",
    ) ||
    /(?:开始|启动|运行|执行).{0,24}评(?:估|测)/.test(text) ||
    text.includes("查询评测状态");

  if ((!container && !pid) || !hasEvaluationSignal) {
    return null;
  }

  return {
    container,
    pid,
    script,
    assessmentTypeText,
    evalTypeText: assessmentTypeText,
  };
};

const parseInferenceTaskSummary = (
  text: string,
): InferenceTaskSummary | null => {
  return inferenceSummaryFromProtocol(extractProtocolFromText(text));
};

const parseInferenceTaskSummaryFromText = (
  sourceText: string,
): InferenceTaskSummary | null => {
  const hasInferenceConfig =
    /当前推理(?:服务)?(?:的)?配置(?:文件|信息)?如下/.test(sourceText) ||
    (sourceText.includes("推理配置") &&
      /(?:MODEL_NAME|VLLM_OPENAI_PORT|INFERENCE_PORT)/.test(sourceText));
  const hasInferenceStatus =
    /当前推理(?:服务)?状态如下/.test(sourceText) ||
    (sourceText.includes("推理服务") && sourceText.includes("服务状态")) ||
    /(?:当前)?所有推理服务均?处于(?:停止|运行)状态/.test(sourceText) ||
    /(?:当前)?所有推理服务(?:已停止|未运行|运行中|已启动)/.test(sourceText);

  if (!hasInferenceConfig && !hasInferenceStatus) {
    return null;
  }

  const modelName =
    extractMarkdownValue(sourceText, "MODEL_NAME") ||
    extractMarkdownValue(sourceText, "模型名称");
  const hostIp =
    extractMarkdownValue(sourceText, "HOST_IP") ||
    extractMarkdownValue(sourceText, "主机IP地址");
  const inferencePort =
    extractMarkdownValue(sourceText, "INFERENCE_PORT") ||
    sourceText.match(
      /推理服务端口\s*[（(]\s*INFERENCE_PORT\s*[）)]\s*[：:]\s*(\d+)/,
    )?.[1] ||
    sourceText.match(/推理服务端口\s*[：:]\s*(\d+)/)?.[1];
  const vllmPort =
    extractMarkdownValue(sourceText, "VLLM_OPENAI_PORT") ||
    sourceText.match(/VLLM(?:_|\s*)OpenAI\s*服务端口\s*[：:]\s*(\d+)/i)?.[1];
  const serviceLineMatches = Array.from(
    sourceText.matchAll(
      /^\s*-\s*(?:[^\w\s(（:：-]+\s*)?.+?\s*[（(]\s*(?:端口\s*)?\d+\s*(?:端口)?\s*[）)]\s*[：:]\s*(.+?)\s*$/gm,
    ),
  );
  const runningServices = serviceLineMatches.filter((match) =>
    /运行中|正在运行|running/i.test(match[1]),
  ).length;
  const stoppedServices = serviceLineMatches.filter((match) =>
    /未运行|未启动|已停止|停止|stopped/i.test(match[1]),
  ).length;
  const hasAggregateStopped =
    /(?:所有推理服务|所有服务|全部服务).{0,12}(?:停止|未运行|未启动|已停止)/.test(
      sourceText,
    );
  const hasAggregateRunning =
    /(?:所有推理服务|所有服务|全部服务).{0,12}(?:运行|已启动)/.test(sourceText);

  return {
    modelName,
    hostIp,
    inferencePort,
    vllmPort,
    hasConfig: hasInferenceConfig,
    hasStatus: hasInferenceStatus,
    stoppedServices: stoppedServices || (hasAggregateStopped ? 4 : 0),
    runningServices: runningServices || (hasAggregateRunning ? 4 : 0),
  };
};

const inferenceServiceInstanceItems = (
  payload?: InferenceServiceInstancesPayload,
): InferenceServiceInstanceItem[] =>
  Array.isArray(payload?.items) ? payload.items : [];

const isInferenceServiceInstancesStatusPayload = (
  payload?: InferenceServiceInstancesPayload,
): boolean => {
  if (!payload) {
    return false;
  }
  const operation = String(payload.operation || "").trim().toLowerCase();
  return !operation || operation === "status";
};

const protocolHasInferenceServiceStatusPayload = (
  protocol: TrainingJobProtocol | null,
): boolean => {
  if (!protocol) {
    return false;
  }
  if (
    isInferenceServiceInstancesStatusPayload(
      protocol.service_instances || protocol.serviceInstances,
    ) ||
    Boolean(protocol.services?.length) ||
    Boolean(protocol.message?.trim())
  ) {
    return true;
  }
  return Object.values(protocol.nodes || {}).some((node) =>
    isInferenceServiceInstancesStatusPayload(
      node?.service_instances || node?.serviceInstances,
    ) || Boolean(node?.services?.length),
  );
};

const inferenceGpuText = (
  value: InferenceServiceInstanceItem["gpus"],
): string | undefined => {
  if (Array.isArray(value)) {
    return value.map((item) => String(item)).filter(Boolean).join(",");
  }
  const text = String(value || "").trim();
  return text || undefined;
};

const inferenceInstanceSummariesFromProtocol = (
  protocol: TrainingJobProtocol,
): InferenceInstanceSummary[] => {
  const items: InferenceServiceInstanceItem[] = [
    ...inferenceServiceInstanceItems(
      protocol.service_instances || protocol.serviceInstances,
    ),
  ];
  Object.entries(protocol.nodes || {}).forEach(([nodeKey, node]) => {
    inferenceServiceInstanceItems(
      node?.service_instances || node?.serviceInstances,
    ).forEach((item) => items.push({ node: nodeKey, ...item }));
  });
  return items
    .map((item) => {
      const resource = item.resource || {};
      return {
        instanceId: String(item.instance_id || item.instanceId || "").trim(),
        runtimeNodeId: String(
          item.runtime_node_id ||
            item.runtimeNodeId ||
            resource.runtime_node_id ||
            resource.runtimeNodeId ||
            item.node ||
            "",
        ).trim(),
        gpus:
          inferenceGpuText(item.gpus) ||
          inferenceGpuText(item.assigned_gpus) ||
          inferenceGpuText(item.assignedGpus) ||
          inferenceGpuText(resource.assigned_gpus) ||
          inferenceGpuText(resource.assignedGpus),
        reservationId: String(
          item.reservation_id ||
            item.reservationId ||
            resource.reservation_id ||
            resource.reservationId ||
            "",
        ).trim(),
        owner: String(item.owner || "").trim(),
        status: String(item.status || "").trim(),
      };
    })
    .filter(
      (item) =>
        item.instanceId || item.runtimeNodeId || item.gpus || item.reservationId,
    )
    .slice(0, 3);
};
const inferenceSummaryFromProtocol = (
  protocol: TrainingJobProtocol | null,
): InferenceTaskSummary | null => {
  const hasServiceInstances = Boolean(
    protocol?.service_instances ||
      protocol?.serviceInstances ||
      Object.values(protocol?.nodes || {}).some(
        (node) => node?.service_instances || node?.serviceInstances,
      ),
  );
  const isLegacyInferenceServiceInstances =
    protocol?.type === "inference_config" &&
    protocol.action === "config_view" &&
    hasServiceInstances &&
    !protocol.config;
  const isInferenceConfigView =
    protocol?.type === "inference_config" &&
    protocol.action === "config_view" &&
    !isLegacyInferenceServiceInstances;
  const isInferenceStatusView =
    protocol?.type === "inference_status" &&
    protocol.action === "service_status" &&
    protocolHasInferenceServiceStatusPayload(protocol);
  const isInferenceServiceJob =
    ["job_started", "job_stopped"].includes(protocol?.type || "") &&
    protocol?.jobType === "inference_service";

  if (
    !protocol ||
    (!isInferenceConfigView && !isInferenceStatusView && !isInferenceServiceJob)
  ) {
    return null;
  }

  if (isInferenceConfigView) {
    const messageSummary = protocol.message
      ? parseInferenceTaskSummaryFromText(protocol.message)
      : null;
    const nodeConfigs = Object.values(protocol.nodes || {})
      .map((node) => node?.config)
      .filter(Boolean);
    const config = (protocol.config ||
      protocol.nodes?.main?.config ||
      nodeConfigs[0]) as
      | {
          ports?: Record<string, number | string | undefined>;
          env?: Record<string, string | undefined>;
          PORTS?: Record<string, number | string | undefined>;
          ENV?: Record<string, string | undefined>;
        }
      | undefined;
    const ports = config?.ports || config?.PORTS;
    const env = config?.env || config?.ENV;
    return {
      modelName: env?.MODEL_NAME || messageSummary?.modelName,
      hostIp: env?.HOST_IP || messageSummary?.hostIp,
      inferencePort:
        ports?.INFERENCE_PORT === undefined
          ? messageSummary?.inferencePort
          : String(ports.INFERENCE_PORT),
      vllmPort:
        ports?.VLLM_OPENAI_PORT === undefined
          ? messageSummary?.vllmPort
          : String(ports.VLLM_OPENAI_PORT),
      hasConfig: true,
      preferredView: `config`,
      shouldOpenPanel: true,
    };
  }

  if (isInferenceStatusView) {
    const nodeServices = Object.values(protocol.nodes || {}).flatMap(
      (node) => node?.services || [],
    );
    const services = protocol.services || nodeServices;
    const instances = inferenceInstanceSummariesFromProtocol(protocol);
    const runningServices = services.length
      ? services.filter((service) => service.status === "running").length
      : instances.filter((instance) => instance.status === "running").length;
    const stoppedServices = services.length
      ? services.filter((service) => service.status !== "running").length
      : instances.filter((instance) => instance.status !== "running").length;
    return {
      hasStatus: true,
      preferredView: `status`,
      shouldOpenPanel: true,
      runningServices,
      stoppedServices,
      instances,
    };
  }
  if (isInferenceServiceJob) {
    const messageSummary = protocol.message
      ? parseInferenceTaskSummaryFromText(protocol.message)
      : null;
    const nodeConfigs = Object.values(protocol.nodes || {})
      .map((node) => node?.config)
      .filter(Boolean);
    const config = (protocol.config ||
      protocol.nodes?.main?.config ||
      nodeConfigs[0]) as InferenceProtocolConfig | undefined;
    const ports = config?.ports;
    const env = config?.env;

    return {
      modelName: env?.MODEL_NAME || messageSummary?.modelName,
      hostIp: env?.HOST_IP || messageSummary?.hostIp,
      inferencePort:
        ports?.INFERENCE_PORT === undefined
          ? messageSummary?.inferencePort
          : String(ports.INFERENCE_PORT),
      vllmPort:
        ports?.VLLM_OPENAI_PORT === undefined
          ? messageSummary?.vllmPort
          : String(ports.VLLM_OPENAI_PORT),
      hasConfig: protocol.type === "job_stopped" && Boolean(config || messageSummary?.hasConfig),
      hasStatus: protocol.type === "job_stopped",
      preferredView: `status`,
      instances: inferenceInstanceSummariesFromProtocol(protocol),
    };
  }
  return null;
};

const mergeInferenceTaskSummary = (
  current: InferenceTaskSummary | null,
  next: InferenceTaskSummary,
): InferenceTaskSummary => ({
  sourceKey: current?.sourceKey || next.sourceKey,
  modelName: current?.modelName || next.modelName,
  hostIp: current?.hostIp || next.hostIp,
  inferencePort: current?.inferencePort || next.inferencePort,
  vllmPort: current?.vllmPort || next.vllmPort,
  hasConfig: Boolean(current?.hasConfig || next.hasConfig),
  hasStatus: Boolean(current?.hasStatus || next.hasStatus),
  preferredView: current?.preferredView || next.preferredView,
  shouldOpenPanel: Boolean(current?.shouldOpenPanel || next.shouldOpenPanel),
  stoppedServices: current?.stoppedServices ?? next.stoppedServices,
  runningServices: current?.runningServices ?? next.runningServices,
  instances: current?.instances?.length ? current.instances : next.instances,
});

const mergeDataFilterTaskSummary = (
  current: DataFilterTaskSummary | null,
  next: DataFilterTaskSummary,
): DataFilterTaskSummary => {
  if (current && next.sourceKey && current.sourceKey !== next.sourceKey) {
    return next;
  }
  return {
    ...current,
    ...next,
  };
};

const parseDataFilterTaskSummaryFromText = (
  text: string,
): DataFilterTaskSummary | null => {
  const isDataFilterLaunch =
    /score_based_filtering/i.test(text) ||
    /高级筛选|数据筛选|score filter|data filter/i.test(text);
  const isBackgroundLaunch =
    /后台启动运行|后台运行|已启动|started|running/i.test(text);
  if (!isDataFilterLaunch || !isBackgroundLaunch) {
    return null;
  }

  const container = text.match(
    /(?:容器(?:名称)?|container)\s*[:：=]\s*`?([^\s`\n，,]+)/i,
  )?.[1];
  const firstPath = (patterns: RegExp[]) => {
    for (const pattern of patterns) {
      const value = text.match(pattern)?.[1]?.replace(/[。/]+$/g, "");
      if (value?.startsWith("/")) {
        return value;
      }
    }
    return undefined;
  };
  const inputFolder = firstPath([
    /(?:input_folder|inputFolder|输入目录|输入路径|输入数据|处理路径(?:为)?)\s*(?:是|为|[:：=])?\s*`?([^\s`\n，,。]+)/i,
  ]);
  const outputFolder = firstPath([
    /(?:output_folder|outputFolder|输出目录|输出路径|输出数据)\s*(?:是|为|[:：=])?\s*`?([^\s`\n，,。]+)/i,
    /(?:保存(?:在|到))\s*`?(\/[^\s`\n，,。]+)/i,
  ]);
  const outputDatasetName = text.match(
    /(?:output_dataset_name|outputDatasetName|输出数据集)\s*[:：=]\s*`?([^\s`\n，,]+)/i,
  )?.[1];
  const thresholdText = text.match(
    /(?:threshold|阈值|筛选阈值)[^0-9\n]{0,16}([0-9.]+)/i,
  )?.[1];

  return {
    container,
    inputFolder,
    outputFolder,
    outputDatasetName,
    threshold: thresholdText ? Number(thresholdText) : undefined,
    status: "running",
  };
};

const dataFilterSummaryFromProtocol = (
  protocol: TrainingJobProtocol | null,
): DataFilterTaskSummary | null => {
  if (
    (protocol?.type !== "job_started" &&
      protocol?.type !== "monitor_status") ||
    protocol?.jobType !== "data_filter"
  ) {
    return null;
  }

  const threshold =
    typeof protocol.threshold === "number"
      ? protocol.threshold
      : typeof protocol.threshold === "string"
        ? Number(protocol.threshold)
        : undefined;

  const scriptArgs = protocol.scriptArgs || {};
  const command = typeof protocol.command === "string" ? protocol.command : "";
  const commandInputFolder = command.match(/--input_folder\s+([^\s]+)/)?.[1];
  const commandOutputFolder = command.match(/--output_folder\s+([^\s]+)/)?.[1];
  const commandThreshold = command.match(/--threshold\s+([^\s]+)/)?.[1];
  const outputFolder =
    (typeof protocol.outputFolder === "string" ? protocol.outputFolder : undefined) ||
    (typeof protocol.output_folder === "string" ? protocol.output_folder : undefined) ||
    (typeof scriptArgs.output_folder === "string" ? scriptArgs.output_folder : undefined) ||
    (typeof scriptArgs.outputFolder === "string" ? scriptArgs.outputFolder : undefined) ||
    commandOutputFolder;
  const inputFolder =
    (typeof protocol.inputFolder === "string" ? protocol.inputFolder : undefined) ||
    (typeof protocol.input_folder === "string" ? protocol.input_folder : undefined) ||
    (typeof scriptArgs.input_folder === "string" ? scriptArgs.input_folder : undefined) ||
    (typeof scriptArgs.inputFolder === "string" ? scriptArgs.inputFolder : undefined) ||
    commandInputFolder;
  const outputDatasetName =
    (typeof protocol.outputDatasetName === "string"
      ? protocol.outputDatasetName
      : undefined) ||
    (typeof protocol.output_dataset_name === "string"
      ? protocol.output_dataset_name
      : undefined) ||
    (outputFolder ? outputFolder.split("/").filter(Boolean).pop() : undefined);

  const protocolStatus = normalizeDataFilterStatus(protocol.status);

  return {
    container: protocol.container || protocol.container_name,
    outputFolder,
    inputFolder,
    threshold: threshold ?? (commandThreshold ? Number(commandThreshold) : undefined),
    outputDatasetName,
    status: protocol.type === "job_started" ? "running" : protocolStatus,
  };
};

const isInferenceServiceStoppedProtocol = (
  protocol: TrainingJobProtocol | null,
): boolean => {
  if (protocol?.type !== "job_stopped") {
    return false;
  }
  if (protocol.jobType) {
    return protocol.jobType === "inference_service";
  }
  return /(?:inference|orchestrator)/i.test(protocol.agent || "");
};

const isTrainingTaskStopped = (text: string): boolean => {
  const protocol = extractProtocolFromText(text);
  if (isWorkflowProtocol(protocol)) {
    return false;
  }
  if (isTrainingTerminalProtocol(protocol)) {
    return true;
  }

  return (
    (/(?:停止|结束).{0,12}训练/.test(text) &&
      /(?:已完成清理|未发现残留进程|已结束)/.test(text)) ||
    /当前训练任务.{0,100}状态为\s*[:：]?\s*\*{0,2}(?:interrupted|stopped|finished|completed|failed|error|中断|已停止|停止|已完成|完成|失败)/i.test(text)
  );
};

const isAssessmentTaskStopped = (text: string): boolean => {
  const protocol = extractProtocolFromText(text);
  if (
    protocol?.type === "job_stopped" &&
    ["assessment", "evaluate"].includes(protocol.jobType || "")
  ) {
    return true;
  }

  return (
    /(?:停止|结束).{0,12}评(?:估|测)/.test(text) &&
    /(?:已完成清理|未发现残留进程|已结束)/.test(text)
  );
};

const isInferenceTaskStopped = (text: string): boolean => {
  const protocol = extractProtocolFromText(text);
  if (protocol?.type === "job_stopped") {
    if (protocol.jobType) {
      return protocol.jobType === "inference_service";
    }
    if (isInferenceServiceStoppedProtocol(protocol)) {
      return true;
    }
  }
  const protocolMessage =
    (protocol?.type === "message" || protocol?.type === "inference_status") &&
    /(?:inference|orchestrator)/i.test(protocol.agent || "")
      ? protocol.message || ""
      : "";
  const sourceText = protocolMessage || text;

  if (/推理基准测试|基准测试|benchmark|评估|评测/i.test(sourceText)) {
    return false;
  }

  return (
    /推理服务.{0,16}(?:已关闭|已停止|已结束|成功关闭|成功停止|关闭成功|停止成功)/.test(
      sourceText,
    ) ||
    /(?:关闭|停止).{0,8}推理服务.{0,12}(?:成功|完成|已完成|已关闭|已停止)?/.test(
      sourceText,
    )
  );
};

const isDataFilterDismissProtocol = (
  protocol: TrainingJobProtocol | null,
): boolean => {
  if (!protocol) {
    return false;
  }
  if (protocol.jobType === "data_filter") {
    if (["job_stopped", "job_finished", "job_completed"].includes(protocol.type || "")) {
      return true;
    }
    if (normalizeDataFilterStatus(protocol.status) === "completed") {
      return true;
    }
  }
  if (isInferenceServiceStoppedProtocol(protocol)) {
    return true;
  }
  return false;
};

const isDataFilterDismissText = (text: string): boolean => {
  const protocol = extractProtocolFromText(text);
  if (isDataFilterDismissProtocol(protocol)) {
    return true;
  }

  return (
    /高级筛选.{0,16}(?:已关闭|已停止|已结束|成功关闭|成功停止|关闭成功|停止成功)/.test(
      text,
    ) ||
    /(?:agent|Agent)\s*(?:服务|service)?.{0,16}(?:已关闭|已停止|已结束|成功关闭|成功停止|关闭成功|停止成功|closed|stopped)/i.test(
      text,
    ) ||
    isInferenceTaskStopped(text)
  );
};

const fileToBase64 = async (file: File): Promise<string> => {
  const arrayBuffer = await file.arrayBuffer();
  const bytes = new Uint8Array(arrayBuffer);
  const chunkSize = 0x8000;
  let binary = "";

  for (let index = 0; index < bytes.length; index += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(index, index + chunkSize));
  }

  return btoa(binary);
};

const isUserRole = (role: string | undefined | null): boolean => {
  const normalizedRole = role?.toLowerCase() || "";
  return normalizedRole === "user" || normalizedRole.endsWith("_user");
};

const TRAINABLE_DATASET_TYPE_CONFIG: Record<
  string,
  { datasetDir: string; trainType: "lora" | "enhanced"; label: string }
> = {
  sft: {
    datasetDir: "/home/workspace/dataset_batch_train",
    trainType: "lora",
    label: "SFT",
  },
  dpo: {
    datasetDir: "/home/workspace/dataset_daily_train",
    trainType: "enhanced",
    label: "DPO",
  },
};

const buildWorkflowDatasetValue = (dataset: DatasetInfo): string =>
  `${(dataset.type || "").toLowerCase()}:${dataset.name}`;

const resolveTrainableDatasetDir = (
  dataset: DatasetInfo,
  config: { datasetDir: string },
): string => {
  const normalizedBaseDir = config.datasetDir.replace(/\/+$/, "");
  const rawPath = dataset.path?.trim().replace(/\/+$/, "");

  if (!rawPath || rawPath === normalizedBaseDir) {
    return `${normalizedBaseDir}/${dataset.name}`;
  }
  if (rawPath.endsWith(`/${dataset.name}`)) {
    return rawPath;
  }
  return `${rawPath}/${dataset.name}`;
};

const buildOneClickWorkflowCommand = (
  dataset: DatasetInfo | string,
  evaluationName?: string,
) => {
  const datasetName = typeof dataset === "string" ? dataset : dataset.name;
  const datasetType =
    typeof dataset === "string" ? "" : (dataset.type || "").toLowerCase();
  const config = TRAINABLE_DATASET_TYPE_CONFIG[datasetType];
  const datasetDir =
    typeof dataset === "string"
      ? undefined
      : config
        ? resolveTrainableDatasetDir(dataset, config)
        : dataset.path?.trim();
  const trainingHint = config
    ? `，训练类型=${config.trainType}，数据类型=${config.label}${datasetDir ? `，dataset_dir=${datasetDir}` : ""}`
    : "";
  const baseCommand = `我想用${datasetName}这个数据一键训练、部署并评测模型${trainingHint}`;
  return evaluationName
    ? `${baseCommand}，评测集使用${evaluationName}`
    : baseCommand;
};

const buildDatasetTrainingCommand = (dataset: DatasetInfo): string => {
  const datasetType = (dataset.type || "").toLowerCase();
  const config = TRAINABLE_DATASET_TYPE_CONFIG[datasetType];
  const baseCommand = `我想用${dataset.name}这个数据训练模型`;

  if (!config) {
    return baseCommand;
  }

  const datasetDir = resolveTrainableDatasetDir(dataset, config);
  return `${baseCommand}，训练类型=${config.trainType}，数据类型=${config.label}，dataset_dir=${datasetDir}`;
};

const resolveRawDatasetPath = (dataset: DatasetInfo): string => {
  const rawPath = dataset.path?.trim().replace(/\\+/g, "/").replace(/\/+$/, "");

  if (!rawPath) {
    return dataset.name;
  }
  if (rawPath === "/home/workspace/dataset/openai") {
    return rawPath;
  }
  if (rawPath.endsWith(`/${dataset.name}`)) {
    return rawPath;
  }
  return `${rawPath}/${dataset.name}`;
};

const buildDatasetPreprocessCommand = (dataset: DatasetInfo): string =>
  `执行数据预处理，${resolveRawDatasetPath(dataset)}`;

interface RunContentPageProps {
  isMetricsSheetOpen: boolean;
  setIsMetricsSheetOpen: (open: boolean) => void;
  setIsInferenceSheetOpen: (open: boolean) => void;
  setInferencePanelView: (view: InferencePanelView) => void;
  randomUsername: string;
  setRandomUsername: React.Dispatch<React.SetStateAction<string>>;
  filteredReplies: Reply[];
  onAskAIRef?: React.MutableRefObject<
    ((blocks: ContentBlocks) => void) | undefined
  >;
  onCurrentInputRequestChange?: (request: InputRequestData | null) => void;
  onCombinedRepliesChange?: (replies: Reply[]) => void;
  onSilentMonitorReplyChange?: (reply: Reply | null) => void;
  onSilentMonitorCacheKeyChange?: (cacheKey: string | null) => void;
  onSilentMonitorStatusChange?: (status: SilentMonitorStatus) => void;
  onMonitorTrainingCommandChange?: (handler: (() => void) | null) => void;
  onRuntimeResourceContextChange?: (context: {
    nodeId?: string;
    resourceGroupId?: string;
    trainingContainerName: string;
    evaluationContainerName: string;
  }) => void;
  // Input text ref for external control
  setInputTextRef?: React.MutableRefObject<
    ((text: string, hint?: string) => void) | undefined
  >;
  // Command handling props
  onTabChange: (
    tab: "runs" | "overview" | "datasets" | "models" | "evaluation",
  ) => void;
  onQueryDatasets: (containerName?: string) => Promise<DatasetInfo[]>;
  onRefreshDatasets: (containerName?: string) => Promise<DatasetInfo[]>;
  onQueryModels: (containerName?: string) => Promise<ModelInfo[]>;
  onRefreshModels: (containerName?: string) => Promise<ModelInfo[]>;
  onQueryTests: () => Promise<MedicalTestFile[]>;
  onRefreshTests: () => Promise<MedicalTestFile[]>;
  onDownloadDataset: (name: string) => Promise<void>;
  onUseDatasetForTraining: (dataset: DatasetInfo) => void;
  onUseEvaluationForBenchmark: (testName: string) => void;
  onDownloadTest: (name: string, test?: MedicalTestFile) => Promise<void>;
  onUpload: () => void;
  onUploadTest: () => void;
  datasets: DatasetInfo[];
  models: ModelInfo[];
  tests: MedicalTestFile[];
  systemOverviewData: SystemOverviewData | null;
  gpuInfo: GPUInfo[] | null;
  isQueryingDatasets: boolean;
  hasQueriedDatasets: boolean;
  hasQueriedModels: boolean;
  hasQueriedTests: boolean;
  datasetCacheMeta?: ManagementCacheMeta | null;
  modelCacheMeta?: ManagementCacheMeta | null;
  testCacheMeta?: ManagementCacheMeta | null;
  // 评测结果相关
  evaluationResults: EvaluationResult[];
  isQueryingEvaluationResults: boolean;
  evaluationResultQueryError: boolean;
  evaluationResultErrorMessage?: string;
  hasQueriedEvaluationResults: boolean;
  onQueryEvaluationResults: (
    containerName: string,
  ) => Promise<EvaluationResult[]>;
  onRefreshEvaluationResults: () => Promise<EvaluationResult[]>;
  evaluationResultCacheMeta?: ManagementCacheMeta | null;
  onQueryEvaluationResultsZero?: () => Promise<EvaluationResult[]>;
  onDownloadEvaluationResult?: (
    folderPath: string,
    filename: string,
    result?: EvaluationResult,
  ) => Promise<void>;
  downloadingResultId?: string | null;
  onDeleteEvaluationResult?: (
    folderPath: string,
    result?: EvaluationResult,
  ) => Promise<void>;
  deletingResultId?: string | null;
  // Wizard props
  isQuickStartWizardOpen: boolean;
  setIsQuickStartWizardOpen: (open: boolean) => void;
  wizardQueryState: "idle" | "querying" | "completed";
  setWizardQueryState: (state: "idle" | "querying" | "completed") => void;
  wizardDatasets: DatasetInfo[];
  setWizardDatasets: (datasets: DatasetInfo[]) => void;
  wizardSelectedDataset: DatasetInfo | null;
  setWizardSelectedDataset: (dataset: DatasetInfo | null) => void;
  focusOnLatestRun: boolean;
  setFocusOnLatestRun: (focus: boolean) => void;
  onClearContext: () => void;
  chatSessionId: string;
  fallbackUsername: string;
}

interface SilentMonitorStatus {
  isQuerying: boolean;
  lastQueryAt?: string;
  lastResultAt?: string;
  lastDataAt?: string;
  hasMetrics?: boolean;
  hasNewData?: boolean;
  message?: string;
}

const RunContentPage = ({
  isMetricsSheetOpen,
  setIsMetricsSheetOpen,
  setIsInferenceSheetOpen,
  setInferencePanelView,
  randomUsername,
  setRandomUsername,
  filteredReplies,
  onAskAIRef,
  onCurrentInputRequestChange,
  onCombinedRepliesChange,
  onSilentMonitorReplyChange,
  onSilentMonitorCacheKeyChange,
  onSilentMonitorStatusChange,
  onMonitorTrainingCommandChange,
  onRuntimeResourceContextChange,
  setInputTextRef,
  onTabChange,
  onQueryDatasets,
  onRefreshDatasets,
  onQueryModels,
  onRefreshModels,
  onQueryTests,
  onRefreshTests,
  onDownloadDataset,
  onUseDatasetForTraining,
  onUseEvaluationForBenchmark,
  onDownloadTest,
  onUpload,
  onUploadTest,
  datasets,
  models,
  tests,
  systemOverviewData,
  gpuInfo,
  isQueryingDatasets,
  hasQueriedDatasets,
  hasQueriedModels,
  hasQueriedTests,
  datasetCacheMeta,
  modelCacheMeta,
  testCacheMeta,
  // 评测结果相关
  evaluationResults,
  isQueryingEvaluationResults,
  evaluationResultQueryError,
  evaluationResultErrorMessage,
  hasQueriedEvaluationResults,
  onQueryEvaluationResults,
  onRefreshEvaluationResults,
  evaluationResultCacheMeta,
  onQueryEvaluationResultsZero,
  onDownloadEvaluationResult,
  downloadingResultId,
  onDeleteEvaluationResult,
  deletingResultId,
  // Wizard props
  isQuickStartWizardOpen,
  setIsQuickStartWizardOpen,
  wizardQueryState,
  setWizardQueryState,
  wizardDatasets,
  setWizardDatasets,
  wizardSelectedDataset,
  setWizardSelectedDataset,
  focusOnLatestRun,
  setFocusOnLatestRun,
  onClearContext,
  chatSessionId,
  fallbackUsername,
}: RunContentPageProps) => {
  const { t } = useTranslation();
  const isMobile = useIsMobile();
  const {
    sendUserInputToServer,
    resetAgentContext,
    cancelRuntimeResponse,
    inputRequests,
    runId,
    runData,
    replies,
  } = useRunRoom();
  const { nodeId: selectedResourceNodeId } = useResourceNodeSelection();
  const { runs } = useProjectRoom();
  const navigate = useNavigate();
  const location = useLocation();
  const { setRunPagePanelOpen } = useStudioSidebar();
  const { defaultContainerName, defaultEvaluateContainerName } =
    useEnvironmentConfig();
  const { isAdmin, user } = useAuth();
  const resourceGroupsQuery = trpc.listResourceGroups.useQuery(undefined, {
    enabled: isAdmin,
  });
  const resourceGroups = resourceGroupsQuery.data?.data || [];
  const [selectedResourceGroupId, setSelectedResourceGroupId] = useState("");
  const [selectedTrainingPoolId, setSelectedTrainingPoolId] = useState("");
  const runResourceGroups = useMemo(() => {
    if (!isAdmin) {
      return [];
    }

    if (runData?.ownerUserId) {
      const ownerGroups = resourceGroups.filter((group) =>
        group.members?.some((member) => member.userId === runData.ownerUserId),
      );
      if (ownerGroups.length) {
        return ownerGroups;
      }
    }

    const runNodeId = runData?.nodeId?.trim();
    if (runNodeId && runNodeId !== "unknown") {
      return resourceGroups.filter((group) => group.nodeId === runNodeId);
    }

    return [];
  }, [isAdmin, resourceGroups, runData?.nodeId, runData?.ownerUserId]);
  const [currentInputRequest, setCurrentInputRequest] =
    useState<InputRequestData | null>(null);
  const selectedResourceGroup = useMemo(
    () =>
      isAdmin && selectedResourceGroupId
        ? resourceGroups.find((group) => group.id === selectedResourceGroupId)
        : undefined,
    [isAdmin, resourceGroups, selectedResourceGroupId],
  );
  const selectedResourceGroupNodeId = selectedResourceGroup?.nodeId?.trim();
  const conversationResourceNodeId = isAdmin
    ? selectedResourceGroupNodeId || undefined
    : runData?.nodeId ||
      (selectedResourceNodeId !== "all" ? selectedResourceNodeId : undefined);
  const conversationResourceGroupId = isAdmin
    ? selectedResourceGroupId || undefined
    : undefined;
  const currentTrainingContainerName =
    (isAdmin
      ? selectedResourceGroup?.defaultContainerName?.trim()
      : undefined) || defaultContainerName;
  const currentEvaluationContainerName =
    (isAdmin
      ? selectedResourceGroup?.defaultEvaluateContainerName?.trim()
      : undefined) || defaultEvaluateContainerName;
  useEffect(() => {
    onRuntimeResourceContextChange?.({
      nodeId: conversationResourceNodeId,
      resourceGroupId: conversationResourceGroupId,
      trainingContainerName: currentTrainingContainerName,
      evaluationContainerName: currentEvaluationContainerName,
    });
  }, [
    conversationResourceGroupId,
    conversationResourceNodeId,
    currentEvaluationContainerName,
    currentTrainingContainerName,
    onRuntimeResourceContextChange,
  ]);
  const refreshConversationDatasetsMutation =
    trpc.refreshDatasets.useMutation();
  const refreshConversationModelsMutation = trpc.refreshModels.useMutation();
  const refreshConversationTestsMutation =
    trpc.refreshMedicalTests.useMutation();
  const queryConversationDatasetsMutation = trpc.queryDatasets.useMutation();
  const queryConversationModelsMutation = trpc.queryModels.useMutation();
  const queryConversationTestsMutation = trpc.queryMedicalTests.useMutation();
  const [conversationNodeDatasets, setConversationNodeDatasets] = useState<
    DatasetInfo[] | null
  >(null);
  const [conversationNodeModels, setConversationNodeModels] = useState<
    ModelInfo[] | null
  >(null);
  const [conversationNodeTests, setConversationNodeTests] = useState<
    MedicalTestFile[] | null
  >(null);
  const filterConversationResources = useCallback(
    <T extends { nodeId?: string; containerName?: string }>(
      items: T[],
      containerName: string,
    ): T[] => {
      if (!conversationResourceNodeId || !containerName) {
        return [];
      }
      return items.filter(
        (item) =>
          item.nodeId === conversationResourceNodeId &&
          Boolean(item.containerName) &&
          item.containerName === containerName,
      );
    },
    [conversationResourceNodeId],
  );
  const refreshConversationDatasets = useCallback(
    async (containerName?: string): Promise<DatasetInfo[]> => {
      if (!conversationResourceNodeId || (isAdmin && !conversationResourceGroupId)) {
        return [];
      }
      const response = await refreshConversationDatasetsMutation.mutateAsync({
        nodeId: conversationResourceNodeId,
        groupId: conversationResourceGroupId,
        container: containerName || currentTrainingContainerName,
      });
      const items =
        response.success && response.data ? response.data.items || [] : [];
      setConversationNodeDatasets(items);
      return items;
    },
    [
      conversationResourceGroupId,
      conversationResourceNodeId,
      currentTrainingContainerName,
      isAdmin,
      refreshConversationDatasetsMutation,
    ],
  );
  const refreshConversationModels = useCallback(
    async (containerName?: string): Promise<ModelInfo[]> => {
      if (!conversationResourceNodeId || (isAdmin && !conversationResourceGroupId)) {
        return [];
      }
      const response = await refreshConversationModelsMutation.mutateAsync({
        nodeId: conversationResourceNodeId,
        groupId: conversationResourceGroupId,
        container: containerName || currentTrainingContainerName,
      });
      const items =
        response.success && response.data ? response.data.items || [] : [];
      setConversationNodeModels(items);
      return items;
    },
    [
      conversationResourceGroupId,
      conversationResourceNodeId,
      currentTrainingContainerName,
      isAdmin,
      refreshConversationModelsMutation,
    ],
  );
  const refreshConversationTests = useCallback(
    async (containerName?: string): Promise<MedicalTestFile[]> => {
      if (!conversationResourceNodeId || (isAdmin && !conversationResourceGroupId)) {
        return [];
      }
      const response = await refreshConversationTestsMutation.mutateAsync({
        nodeId: conversationResourceNodeId,
        groupId: conversationResourceGroupId,
        container: containerName || currentEvaluationContainerName,
      });
      const items =
        response.success && response.data
          ? (response.data.items || []).map((item) => ({
              ...item,
              nodeId: item.nodeId || conversationResourceNodeId,
              containerName:
                item.containerName || containerName || currentEvaluationContainerName,
            }))
          : [];
      setConversationNodeTests(items);
      return items;
    },
    [
      conversationResourceGroupId,
      conversationResourceNodeId,
      currentEvaluationContainerName,
      isAdmin,
      refreshConversationTestsMutation,
    ],
  );
  useEffect(() => {
    setConversationNodeDatasets(null);
    setConversationNodeModels(null);
    setConversationNodeTests(null);
    if (!conversationResourceNodeId || (isAdmin && !conversationResourceGroupId)) {
      return;
    }

    let cancelled = false;
    void Promise.all([
      queryConversationDatasetsMutation.mutateAsync({
        nodeId: conversationResourceNodeId,
        groupId: conversationResourceGroupId,
        container: currentTrainingContainerName,
      }),
      queryConversationModelsMutation.mutateAsync({
        nodeId: conversationResourceNodeId,
        groupId: conversationResourceGroupId,
        container: currentTrainingContainerName,
      }),
      queryConversationTestsMutation.mutateAsync({
        nodeId: conversationResourceNodeId,
        groupId: conversationResourceGroupId,
        container: currentEvaluationContainerName,
      }),
    ])
      .then(([datasetResponse, modelResponse, testResponse]) => {
        if (cancelled) return;
        setConversationNodeDatasets(
          datasetResponse.success && datasetResponse.data
            ? datasetResponse.data.items || []
            : [],
        );
        setConversationNodeModels(
          modelResponse.success && modelResponse.data
            ? modelResponse.data.items || []
            : [],
        );
        setConversationNodeTests(
          testResponse.success && testResponse.data
            ? (testResponse.data.items || []).map((item) => ({
                ...item,
                nodeId: item.nodeId || conversationResourceNodeId,
                containerName:
                  item.containerName || currentEvaluationContainerName,
              }))
            : [],
        );
      })
      .catch((error) => {
        if (cancelled) return;
        console.warn("Failed to load current run node resources:", error);
        setConversationNodeDatasets([]);
        setConversationNodeModels([]);
        setConversationNodeTests([]);
      });

    return () => {
      cancelled = true;
    };
  }, [
    conversationResourceGroupId,
    conversationResourceNodeId,
    currentEvaluationContainerName,
    currentTrainingContainerName,
    isAdmin,
  ]);
  const conversationDatasets = useMemo(
    () =>
      conversationNodeDatasets ??
      filterConversationResources(datasets, currentTrainingContainerName),
    [
      conversationNodeDatasets,
      currentTrainingContainerName,
      datasets,
      filterConversationResources,
    ],
  );
  const conversationModels = useMemo(
    () =>
      conversationNodeModels ??
      filterConversationResources(models, currentTrainingContainerName),
    [
      conversationNodeModels,
      currentTrainingContainerName,
      filterConversationResources,
      models,
    ],
  );
  const conversationTests = useMemo(
    () =>
      conversationNodeTests ??
      filterConversationResources(tests, currentEvaluationContainerName),
    [
      conversationNodeTests,
      currentEvaluationContainerName,
      filterConversationResources,
      tests,
    ],
  );
  const [autoScroll, setAutoScroll] = useState(true);
  const [pendingUserMessage, setPendingUserMessage] = useState<{
    content: string;
    sentAt: number;
    assistantCountBeforeSend: number;
    assistantMessageIdsBeforeSend: string[];
  } | null>(null);
  const filteredRepliesRef = useRef(filteredReplies);
  useEffect(() => {
    filteredRepliesRef.current = filteredReplies;
  }, [filteredReplies]);
  useEffect(() => {
    if (!isAdmin) {
      setSelectedResourceGroupId("");
      return;
    }
    if (!runResourceGroups.length) {
      setSelectedResourceGroupId("");
      return;
    }
    if (
      !selectedResourceGroupId ||
      !runResourceGroups.some((group) => group.id === selectedResourceGroupId)
    ) {
      setSelectedResourceGroupId(runResourceGroups[0].id);
    }
  }, [isAdmin, runResourceGroups, selectedResourceGroupId]);
  const outboundResourceGroupId = isAdmin
    ? selectedResourceGroupId || undefined
    : user?.group?.id;
  const isAdminResourceGroupRequired = isAdmin && !outboundResourceGroupId;
  useEffect(() => {
    setSelectedTrainingPoolId("");
  }, [outboundResourceGroupId]);
  const runnablePoolsQuery = trpc.listRunnableTrainingResourcePools.useQuery(
    isAdmin && outboundResourceGroupId
      ? { groupId: outboundResourceGroupId }
      : undefined,
    {
      enabled: isAdmin ? Boolean(outboundResourceGroupId) : Boolean(user),
    },
  );
  const runnableTrainingPools = useMemo(
    () => runnablePoolsQuery.data?.data || [],
    [runnablePoolsQuery.data?.data],
  );
  useEffect(() => {
    if (!runnableTrainingPools.length) {
      setSelectedTrainingPoolId("");
      return;
    }
    if (runnableTrainingPools.length === 1) {
      setSelectedTrainingPoolId(runnableTrainingPools[0].id);
      return;
    }
    if (
      !runnableTrainingPools.some((pool) => pool.id === selectedTrainingPoolId)
    ) {
      setSelectedTrainingPoolId("");
    }
  }, [runnableTrainingPools, selectedTrainingPoolId]);
  const outboundTrainingPoolId =
    selectedTrainingPoolId ||
    (runnableTrainingPools.length === 1
      ? runnableTrainingPools[0].id
      : undefined);
  const isTrainingPoolRequired =
    runnableTrainingPools.length > 1 && !outboundTrainingPoolId;
  const trainingPoolOptions = useMemo(
    () =>
      runnableTrainingPools.map((pool) => {
        const usageText =
          pool.nodeCount <= 1
            ? t("runpage.trainingPoolSingleNode", {
                gpuCount: pool.totalGpuCount || pool.gpusPerNode,
              })
            : t("runpage.trainingPoolMultiNode", {
                nodeCount:
                  pool.nodeCount === 2
                    ? t("runpage.trainingPoolDualNode")
                    : t("runpage.trainingPoolNodeCount", {
                        count: pool.nodeCount,
                      }),
                gpuCount: pool.gpusPerNode,
              });
        const title =
          pool.name && pool.name !== usageText
            ? `${usageText} · ${pool.name}`
            : usageText;
        return {
          value: pool.id,
          label: title,
          searchLabel: title,
        };
      }),
    [runnableTrainingPools, t],
  );
  const [isTrainingGuideOpen, setIsTrainingGuideOpen] = useState(false);
  const [isPostTrainingGuideChoiceOpen, setIsPostTrainingGuideChoiceOpen] =
    useState(false);
  const [isWorkflowConfigOpen, setIsWorkflowConfigOpen] = useState(false);
  const [isRefreshingWorkflowDatasets, setIsRefreshingWorkflowDatasets] =
    useState(false);
  const [isRefreshingWorkflowEvaluations, setIsRefreshingWorkflowEvaluations] =
    useState(false);
  const [selectedWorkflowDatasetName, setSelectedWorkflowDatasetName] =
    useState<string | undefined>();
  const [selectedWorkflowEvaluationName, setSelectedWorkflowEvaluationName] =
    useState<string | undefined>();
  useEffect(() => {
    setSelectedWorkflowDatasetName(undefined);
    setSelectedWorkflowEvaluationName(undefined);
  }, [selectedResourceGroupId]);
  const [asChatInputText, setAsChatInputText] = useState("");
  const [sendButtonHighlightToken, setSendButtonHighlightToken] = useState(0);
  const [inputInlineHint, setInputInlineHint] = useState("");
  const [wizardResumeState, setWizardResumeState] = useState<
    "hidden" | "filled" | "sent" | "ready"
  >("hidden");
  const [wizardCommandPending, setWizardCommandPending] = useState(false);
  const [wizardProgress, setWizardProgress] = useState({
    step: 0,
    title: "",
  });
  const [wizardCommandProgress, setWizardCommandProgress] = useState({
    step: 0,
    title: "",
  });
  const [wizardResultBaseline, setWizardResultBaseline] = useState({
    assistantReplyCount: 0,
    assistantMessageIdsBeforeSend: [] as string[],
    sentAt: 0,
    commandResultCount: 0,
  });
  const [commandResults, setCommandResults] = useState<
    Array<{ id: string; content: string; timestamp: number }>
  >([]);
  const [wizardPendingSentCommand, setWizardPendingSentCommand] = useState("");
  const [isEnvironmentCheckOpen, setIsEnvironmentCheckOpen] = useState(false);
  const [environmentContainerName, setEnvironmentContainerName] =
    useState(defaultContainerName);
  const [hiddenStatusBars, setHiddenStatusBars] = useState<{
    workflow: boolean;
    train: boolean;
    evaluation: boolean;
    benchmark: boolean;
    inference: boolean;
    data_filter: boolean;
  }>({
    workflow: false,
    train: false,
    evaluation: false,
    benchmark: false,
    inference: false,
    data_filter: false,
  });
  const [closedStatusBars, setClosedStatusBars] = useState<{
    workflow: boolean;
    train: boolean;
    evaluation: boolean;
    benchmark: boolean;
    inference: boolean;
    data_filter: boolean;
  }>({
    workflow: false,
    train: false,
    evaluation: false,
    benchmark: false,
    inference: false,
    data_filter: false,
  });
  const [pendingBenchmarkStop, setPendingBenchmarkStop] = useState(false);
  const [benchmarkTaskSnapshot, setBenchmarkTaskSnapshot] =
    useState<BenchmarkTaskSummary | null>(null);
  const [polledWorkflowTask, setPolledWorkflowTask] =
    useState<WorkflowTaskSummary | null>(null);
  const [isWorkflowDetailsOpen, setIsWorkflowDetailsOpen] = useState(false);
  const [workflowMonitorState, setWorkflowMonitorState] = useState<{
    isPolling: boolean;
    lastUpdatedAt?: number;
    nextRefreshAt?: number;
    error?: string;
  }>({
    isPolling: false,
  });
  const [workflowMonitorNow, setWorkflowMonitorNow] = useState(() =>
    Date.now(),
  );
  const [polledDataFilterTask, setPolledDataFilterTask] =
    useState<DataFilterTaskSummary | null>(null);
  const [dismissedDataFilterTaskSourceKey, setDismissedDataFilterTaskSourceKey] =
    useState<string | null>(null);
  const [dataFilterMonitorState, setDataFilterMonitorState] = useState<{
    isPolling?: boolean;
    isManualRefreshing?: boolean;
    lastUpdatedAt?: number;
    nextRefreshAt?: number;
    error?: string;
    consecutiveErrors?: number;
  }>({});
  const [dataFilterMonitorNow, setDataFilterMonitorNow] = useState(() =>
    Date.now(),
  );
  const [environmentCheckResult, setEnvironmentCheckResult] =
    useState<EnvironmentCheckResult | null>(null);
  const [environmentCheckError, setEnvironmentCheckError] = useState("");
  const checkEnvironmentMutation = trpc.checkEnvironment.useMutation();
  const queryTrainingMetricsMutation = trpc.queryTrainingMetrics.useMutation();
  const { mutateAsync: queryWorkflowStatus } =
    trpc.queryWorkflowStatus.useMutation();
  const queryDataFilterStatusMutation = trpc.queryDataFilterStatus.useMutation();
  const workflowStatusErrorMessage = (error: unknown) => {
    const message =
      error instanceof Error
        ? error.message
        : typeof error === "string"
          ? error
          : "";
    if (/timeout|timed out|abort|超时/i.test(message)) {
      return t("runpage.statusBar.workflowRefreshTimeout");
    }
    return message || t("runpage.query-failed");
  };
  const dataFilterStatusErrorMessage = (error: unknown) => {
    const message =
      error instanceof Error
        ? error.message
        : typeof error === "string"
          ? error
          : "";
    if (/timeout|timed out|abort|超时/i.test(message)) {
      return t("runpage.statusBar.dataFilterRefreshTimeout");
    }
    return message || t("runpage.query-failed");
  };
  const { showPostTourChoice, closePostTourChoice } = useFirstTimeGuide();

  const translateStatusBarTypeValue = useCallback(
    (
      value?: string,
      launchMode?: string,
      container?: string,
      scriptName?: string,
      isMultinode?: boolean,
    ): string => {
      if (!value) {
        return "";
      }

      const normalizedValue = value
        .trim()
        .toLowerCase()
        .replace(/[\s_-]+/g, "");
      const typeKeyMap: Record<string, string> = {
        lora: "loraBatchTraining",
        sft: "loraBatchTraining",
        lora批量训练: "loraBatchTraining",
        多机lora批量训练: "multinodeLoraBatchTraining",
        双机lorasft: "multinodeLoraBatchTraining",
        多机lorasft: "multinodeLoraBatchTraining",
        双机lora批量训练: "multinodeLoraBatchTraining",
        lorabatchtrain: "loraBatchTraining",
        lorabatchtraining: "loraBatchTraining",
        multinodelorasft: "multinodeLoraBatchTraining",
        multinodelorabatchtrain: "multinodeLoraBatchTraining",
        multinodelorabatchtraining: "multinodeLoraBatchTraining",
        full: "fullParamBatchTraining",
        全参批量训练: "fullParamBatchTraining",
        fullparambatchtrain: "fullParamBatchTraining",
        fullparambatchtraining: "fullParamBatchTraining",
        fullparameterbatchtraining: "fullParamBatchTraining",
        enhanced: "enhancedTraining",
        dpo: "enhancedTraining",
        增强训练: "enhancedTraining",
        多机增强训练: "multinodeEnhancedTraining",
        双机增强训练: "multinodeEnhancedTraining",
        enhancedtrain: "enhancedTraining",
        enhancedtraining: "enhancedTraining",
        multinodeenhanced: "multinodeEnhancedTraining",
        multinodeenhancedtrain: "multinodeEnhancedTraining",
        multinodeenhancedtraining: "multinodeEnhancedTraining",
        grpo: "grpoTraining",
        grpo训练: "grpoTraining",
        grpotrain: "grpoTraining",
        grpotraining: "grpoTraining",
        单模型评估: "singleModelEvaluation",
        singlemodel: "singleModelEvaluation",
        singlemodelevaluation: "singleModelEvaluation",
        双模型评估: "dualModelEvaluation",
        comparebetweenmodels: "dualModelEvaluation",
        dualmodel: "dualModelEvaluation",
        dualmodelevaluation: "dualModelEvaluation",
        ckpt评估: "checkpointEvaluation",
        ckptevaluation: "checkpointEvaluation",
        checkpoint评估: "checkpointEvaluation",
        checkpoint: "checkpointEvaluation",
        checkpointevaluation: "checkpointEvaluation",
      };

      let typeKey = typeKeyMap[normalizedValue];
      if (!typeKey) {
        return value;
      }
      const isMultinodeTraining =
        Boolean(isMultinode) ||
        (launchMode || "").trim().toLowerCase() === "multinode" ||
        (scriptName || "").trim().startsWith("train_multinode_") ||
        (container || "").trim().includes("qingnang_train_multi");
      if (isMultinodeTraining && typeKey === "loraBatchTraining") {
        typeKey = "multinodeLoraBatchTraining";
      }
      if (isMultinodeTraining && typeKey === "enhancedTraining") {
        typeKey = "multinodeEnhancedTraining";
      }

      return t(`runpage.statusBar.typeValues.${typeKey}`, {
        defaultValue: value,
      });
    },
    [t],
  );

  useEffect(() => {
    setEnvironmentContainerName(defaultContainerName);
  }, [defaultContainerName]);

  // 将 setAsChatInputText 赋值给 setInputTextRef.current，以便外部可以通过 ref 设置输入框文本
  useEffect(() => {
    if (setInputTextRef) {
      setInputTextRef.current = (text: string, hint?: string) => {
        setAsChatInputText(text);
        setInputInlineHint(hint || "");
        setSendButtonHighlightToken((prev) => prev + 1);
      };
    }
  }, [setInputTextRef, setAsChatInputText]);

  useEffect(() => {
    if (asChatInputText.trim().length === 0) {
      setInputInlineHint("");
    }
  }, [asChatInputText]);

  useEffect(() => {
    const hasNewExecutionResult =
      hasAssistantMessageAfterSend(
        filteredReplies,
        wizardResultBaseline.sentAt,
        wizardResultBaseline.assistantMessageIdsBeforeSend,
      ) ||
      commandResults.length > wizardResultBaseline.commandResultCount;

    if (
      wizardResumeState === "sent" &&
      inputRequests.length > 0 &&
      hasNewExecutionResult
    ) {
      if (wizardPendingSentCommand) {
        wizardRef.current?.handleStepComplete(wizardPendingSentCommand);
        setWizardPendingSentCommand("");
      }
      if (wizardCommandProgress.step >= 3) {
        setWizardResumeState("hidden");
        setIsQuickStartWizardOpen(true);
        return;
      }
      setWizardResumeState("ready");
    }
  }, [
    commandResults.length,
    filteredReplies,
    inputRequests.length,
    wizardPendingSentCommand,
    wizardCommandProgress.step,
    wizardResultBaseline.assistantMessageIdsBeforeSend,
    wizardResultBaseline.sentAt,
    wizardResultBaseline.commandResultCount,
    wizardResumeState,
    setIsQuickStartWizardOpen,
  ]);

  useEffect(() => {
    const shouldOpenTrainingGuide =
      sessionStorage.getItem("medflow_open_training_guide") === "true";
    const shouldOpenQuickStart =
      sessionStorage.getItem("medflow_open_quick_start") === "true";

    if (shouldOpenTrainingGuide) {
      sessionStorage.removeItem("medflow_open_training_guide");
      setIsTrainingGuideOpen(true);
    }

    if (shouldOpenQuickStart) {
      sessionStorage.removeItem("medflow_open_quick_start");
      setIsQuickStartWizardOpen(true);
    }
  }, [location.pathname, setIsQuickStartWizardOpen]);

  // Ref for QuickStartWizard
  const wizardRef = useRef<QuickStartWizardRef>(null);
  // Ref to store the command before it's sent
  const lastSentCommandRef = useRef<string>("");
  const { messageApi } = useMessageApi();

  const openTrainingGuideFromChoice = useCallback(() => {
    closePostTourChoice();
    setIsTrainingGuideOpen(true);
  }, [closePostTourChoice]);

  const openPracticeFromChoice = useCallback(() => {
    closePostTourChoice();
    setIsPostTrainingGuideChoiceOpen(false);
    setIsQuickStartWizardOpen(true);
  }, [closePostTourChoice, setIsQuickStartWizardOpen]);

  const handleWizardStepChange = useCallback((step: number, title: string) => {
    setWizardProgress((prev) => {
      if (prev.step === step && prev.title === title) {
        return prev;
      }

      return { step, title };
    });
  }, []);

  const restartQuickStartWizard = useCallback(() => {
    wizardRef.current?.restart();
    setWizardResumeState("hidden");
    setWizardCommandPending(false);
    setWizardPendingSentCommand("");
    setWizardProgress({ step: 0, title: "" });
    setWizardCommandProgress({ step: 0, title: "" });
    setWizardResultBaseline({
      assistantReplyCount: 0,
      commandResultCount: 0,
    });
    setIsQuickStartWizardOpen(true);
  }, [setIsQuickStartWizardOpen]);

  const closeWizardAndRevealChatInput = useCallback(() => {
    setIsQuickStartWizardOpen(false);
    setRunPagePanelOpen(false);
  }, [setIsQuickStartWizardOpen, setRunPagePanelOpen]);

  const runEnvironmentCheck = useCallback(
    async (containerOverride?: string) => {
      const container =
        (containerOverride ?? defaultContainerName).trim() ||
        defaultContainerName;
      setEnvironmentContainerName(container);
      setEnvironmentCheckError("");

      try {
        const response = await checkEnvironmentMutation.mutateAsync({
          nodeId: conversationResourceNodeId,
          container,
        });

        if (response.success && response.data) {
          setEnvironmentCheckResult(response.data);

          if (response.data.overallStatus === "ok") {
            messageApi.success(t("environmentCheck.toast.ok"));
          } else if (response.data.overallStatus === "warning") {
            messageApi.warning(t("environmentCheck.toast.warning"));
          } else {
            messageApi.error(t("environmentCheck.toast.error"));
          }
        } else {
          const errorMessage = response.message || t("environmentCheck.failed");
          setEnvironmentCheckError(errorMessage);
          messageApi.error(errorMessage);
        }
      } catch (error: any) {
        const errorMessage = error?.message || t("environmentCheck.failed");
        setEnvironmentCheckError(errorMessage);
        messageApi.error(errorMessage);
      }
    },
    [
      checkEnvironmentMutation,
      conversationResourceNodeId,
      defaultContainerName,
      messageApi,
      t,
    ],
  );

  // Refs for data to avoid closure trap
  const datasetsRef = useRef(datasets);
  useEffect(() => {
    datasetsRef.current = datasets;
  }, [datasets]);

  const modelsRef = useRef(models);
  useEffect(() => {
    modelsRef.current = models;
  }, [models]);

  const testsRef = useRef(tests);
  useEffect(() => {
    testsRef.current = tests;
  }, [tests]);

  const systemOverviewDataRef = useRef(systemOverviewData);
  useEffect(() => {
    systemOverviewDataRef.current = systemOverviewData;
  }, [systemOverviewData]);

  const gpuInfoRef = useRef(gpuInfo);
  useEffect(() => {
    gpuInfoRef.current = gpuInfo;
  }, [gpuInfo]);

  // Command input messages state (to show user commands in chat)
  const [commandInputs, setCommandInputs] = useState<
    Array<{ id: string; content: string; timestamp: number }>
  >([]);

  // Command loading messages state (to show loading indicators)
  const [commandLoading, setCommandLoading] = useState<
    Array<{ id: string; inputId: string; content: string; timestamp: number }>
  >([]);

  useEffect(() => {
    setCommandInputs([]);
    setCommandLoading([]);
    setCommandResults([]);
    setPendingUserMessage(null);
    setPolledWorkflowTask(null);
  }, [runId]);

  // 从 content 中提取完整文本（处理 ContentBlocks）
  const extractFullContent = (content: ContentType): string => {
    if (typeof content === "string") {
      return content;
    }
    if (Array.isArray(content)) {
      return content
        .map((block) => {
          if (block.type === "text") {
            return (block as ContentBlock & { text: string }).text || "";
          }
          if (block.type === "image") {
            return "[Image]";
          }
          return "";
        })
        .join("\n");
    }
    return "";
  };

  // 文件下载辅助函数
  const downloadFile = (
    content: string,
    filename: string,
    mimeType: string,
  ) => {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  // 导出为 JSON
  const exportToJSON = () => {
    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    const filename = `${randomUsername}-${runId}-${timestamp}.json`;
    const dataStr = JSON.stringify(
      sanitizeRepliesForDisplay(filteredReplies, randomUsername),
      null,
      2,
    );
    downloadFile(dataStr, filename, "application/json");
    messageApi.success(t("action.export-success"));
  };

  // 导出为 Markdown
  const exportToMarkdown = () => {
    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    const filename = `${randomUsername}-${runId}-${timestamp}.md`;

    let md = `# Chat Export\n\n`;
    md += `**Export Time**: ${new Date().toLocaleString()}\n`;
    md += `**Run ID**: ${runId}\n`;
    md += `**User**: ${randomUsername}\n\n`;
    md += `---\n\n`;

    const displayReplies = sanitizeRepliesForDisplay(
      filteredReplies,
      randomUsername,
    );

    displayReplies.forEach((reply, index) => {
      md += `## Reply ${index + 1}: ${reply.replyName}\n\n`;
      md += `- **Reply ID**: ${reply.replyId}\n`;
      md += `- **Role**: ${reply.replyRole}\n`;
      md += `- **Messages**: ${reply.messages.length}\n\n`;

      reply.messages.forEach((msg) => {
        const roleIcon = isUserRole(msg.role)
          ? "👤"
          : msg.role === "assistant"
            ? "🤖"
            : "🔧";
        md += `### ${roleIcon} ${msg.name} (${msg.role})\n\n`;
        const content = extractFullContent(msg.content);
        md += content || `[No content]\n\n`;
      });
      md += `---\n\n`;
    });

    downloadFile(md, filename, "text/markdown");
    messageApi.success(t("action.export-success"));
  };

  // 导出为 CSV
  const exportToCSV = () => {
    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    const filename = `${randomUsername}-${runId}-${timestamp}.csv`;

    const headers = [
      "Reply ID",
      "Reply Name",
      "Reply Role",
      "Message ID",
      "Message Name",
      "Role",
      "Content",
      "Timestamp",
    ];
    const rows: string[][] = [];

    const displayReplies = sanitizeRepliesForDisplay(
      filteredReplies,
      randomUsername,
    );

    displayReplies.forEach((reply) => {
      reply.messages.forEach((msg) => {
        const content = extractFullContent(msg.content);
        rows.push([
          reply.replyId,
          reply.replyName || "",
          reply.replyRole || "",
          msg.id,
          msg.name || "",
          msg.role,
          content.replace(/"/g, '""'), // 转义引号
          msg.timestamp || reply.createdAt || "",
        ]);
      });
    });

    const csv = [headers, ...rows]
      .map((row) =>
        row.map((cell) => `"${String(cell).replace(/"/g, '""')}"`).join(","),
      )
      .join("\n");

    downloadFile(csv, filename, "text/csv");
    messageApi.success(t("action.export-success"));
  };

  // 导出菜单项
  const handleClearContextClick = () => {
    const previousContextUsername = randomUsername;
    onClearContext();
    void resetAgentContext(previousContextUsername, { silent: true });
  };

  const exportMenuItems = [
    {
      key: "export-records",
      icon: <DownloadOutlined />,
      label: t("common.export-records") || "Export Records",
      children: [
        {
          key: "json",
          icon: <CodeOutlined />,
          label: t("common.export-json") || "Export as JSON",
          onClick: exportToJSON,
        },
        {
          key: "markdown",
          icon: <FileMarkdownOutlined />,
          label: t("common.export-markdown") || "Export as Markdown",
          onClick: exportToMarkdown,
        },
        {
          key: "csv",
          icon: <FileTextOutlined />,
          label: t("common.export-csv") || "Export as CSV",
          onClick: exportToCSV,
        },
      ],
    },
    {
      type: "divider" as const,
    },
    {
      key: "clear-context",
      icon: <Eraser className="h-[15px] w-[15px]" strokeWidth={1.9} />,
      label: t("runpage.toolbar.clearContext") || "清空上下文",
      onClick: handleClearContextClick,
    },
  ];

  const quickStartMenuItems = [
    {
      key: "guide",
      label: t("quickStart.menu.guide") || "学习教程",
      onClick: () => {
        setIsTrainingGuideOpen(true);
      },
    },
    {
      key: "practice",
      label: t("quickStart.menu.practice") || "实战演练",
      onClick: () => {
        setIsQuickStartWizardOpen(true);
      },
    },
  ];

  const workflowDatasetOptions = useMemo(
    () =>
      conversationDatasets
        .filter(
          (dataset) =>
            dataset.size !== "0 B" &&
            (dataset.type || "").toLowerCase() !== "raw",
        )
        .map((dataset) => ({
          label: `${dataset.name}${dataset.type ? ` · ${dataset.type.toUpperCase()}` : ""}`,
          value: buildWorkflowDatasetValue(dataset),
        })),
    [conversationDatasets],
  );
  const workflowEvaluationOptions = useMemo(
    () =>
      conversationTests.map((test) => ({
        label: `${test.filename}${test.category || test.type ? ` · ${test.category || test.type}` : ""}`,
        value: test.filename,
      })),
    [conversationTests],
  );
  const selectedWorkflowDataset = useMemo(
    () =>
      conversationDatasets.find(
        (dataset) =>
          buildWorkflowDatasetValue(dataset) === selectedWorkflowDatasetName ||
          dataset.name === selectedWorkflowDatasetName,
      ),
    [conversationDatasets, selectedWorkflowDatasetName],
  );
  const selectedWorkflowEvaluation = useMemo(
    () =>
      conversationTests.find(
        (test) => test.filename === selectedWorkflowEvaluationName,
      ),
    [conversationTests, selectedWorkflowEvaluationName],
  );
  const workflowCommandPreview = selectedWorkflowDataset
    ? buildOneClickWorkflowCommand(
        selectedWorkflowDataset,
        selectedWorkflowEvaluationName,
      )
    : "";

  useEffect(() => {
    if (
      selectedWorkflowDatasetName &&
      !conversationDatasets.some(
        (dataset) =>
          buildWorkflowDatasetValue(dataset) === selectedWorkflowDatasetName ||
          dataset.name === selectedWorkflowDatasetName,
      )
    ) {
      setSelectedWorkflowDatasetName(undefined);
    }
  }, [conversationDatasets, selectedWorkflowDatasetName]);

  useEffect(() => {
    if (
      selectedWorkflowEvaluationName &&
      !conversationTests.some(
        (test) => test.filename === selectedWorkflowEvaluationName,
      )
    ) {
      setSelectedWorkflowEvaluationName(undefined);
    }
  }, [conversationTests, selectedWorkflowEvaluationName]);

  const openWorkflowConfig = useCallback(() => {
    setIsWorkflowConfigOpen(true);
  }, []);

  const refreshWorkflowDatasets = useCallback(async () => {
    setIsRefreshingWorkflowDatasets(true);
    try {
      const latestDatasets = await refreshConversationDatasets(
        currentTrainingContainerName,
      );
      if (
        selectedWorkflowDatasetName &&
        !latestDatasets.some(
          (dataset) =>
            buildWorkflowDatasetValue(dataset) === selectedWorkflowDatasetName ||
            dataset.name === selectedWorkflowDatasetName,
        )
      ) {
        setSelectedWorkflowDatasetName(undefined);
      }
      messageApi.success(
        t("oneClickWorkflow.dataset.refreshSuccess") || "数据集已刷新",
      );
    } catch (error) {
      console.error(
        "[RunPage][OneClickWorkflow] refresh datasets failed",
        error,
      );
      messageApi.error(
        t("oneClickWorkflow.dataset.refreshFailed") || "刷新数据集失败",
      );
    } finally {
      setIsRefreshingWorkflowDatasets(false);
    }
  }, [
    currentTrainingContainerName,
    messageApi,
    refreshConversationDatasets,
    selectedWorkflowDatasetName,
    t,
  ]);

  const refreshWorkflowEvaluations = useCallback(async () => {
    setIsRefreshingWorkflowEvaluations(true);
    try {
      const latestTests = await refreshConversationTests(
        currentEvaluationContainerName,
      );
      if (
        selectedWorkflowEvaluationName &&
        !latestTests.some(
          (test) => test.filename === selectedWorkflowEvaluationName,
        )
      ) {
        setSelectedWorkflowEvaluationName(undefined);
      }
      messageApi.success(
        t("oneClickWorkflow.evaluation.refreshSuccess", {
          defaultValue: "评测集已刷新",
        }),
      );
    } catch (error) {
      console.error(
        "[RunPage][OneClickWorkflow] refresh evaluations failed",
        error,
      );
      messageApi.error(
        t("oneClickWorkflow.evaluation.refreshFailed", {
          defaultValue: "刷新评测集失败",
        }),
      );
    } finally {
      setIsRefreshingWorkflowEvaluations(false);
    }
  }, [
    currentEvaluationContainerName,
    messageApi,
    refreshConversationTests,
    selectedWorkflowEvaluationName,
    t,
  ]);

  const fillWorkflowStartCommand = useCallback(() => {
    if (!selectedWorkflowDatasetName) {
      message.warning(
        t("oneClickWorkflow.validation.datasetRequired") ||
          "请先选择训练数据集",
      );
      return;
    }
    if (!setInputTextRef?.current) {
      message.warning(t("wizard.resume.inputUnavailable"));
      return;
    }

    if (!selectedWorkflowDataset) {
      message.warning(
        t("oneClickWorkflow.validation.datasetRequired") ||
          "请先选择训练数据集",
      );
      return;
    }

    const command = buildOneClickWorkflowCommand(
      selectedWorkflowDataset,
      selectedWorkflowEvaluationName,
    );
    setInputTextRef.current(
      command,
      t("oneClickWorkflow.inputHint") ||
        "一键工作流命令已填入，点击右下角发送即可开始",
    );
    setIsWorkflowConfigOpen(false);
    setRunPagePanelOpen(false);
    messageApi.success(t("wizard.resume.commandFilledToast"));
  }, [
    messageApi,
    selectedWorkflowDataset,
    selectedWorkflowDatasetName,
    selectedWorkflowEvaluationName,
    setInputTextRef,
    setRunPagePanelOpen,
    t,
  ]);

  // Pop the first input request to receive user input
  useEffect(() => {
    if (inputRequests.length > 0) {
      setCurrentInputRequest(inputRequests[0]);
      onCurrentInputRequestChange?.(inputRequests[0]);
    } else if (runData?.status === Status.RUNNING && runId) {
      const runtimeInputRequest: InputRequestData = {
        requestId: `runtime:${runId}`,
        agentId: "runtime-agent",
        agentName: "Runtime Agent",
        structuredInput: null,
      };
      setCurrentInputRequest(runtimeInputRequest);
      onCurrentInputRequestChange?.(runtimeInputRequest);
    } else {
      setCurrentInputRequest(null);
      onCurrentInputRequestChange?.(null);
    }
  }, [inputRequests, onCurrentInputRequestChange, runData?.status, runId]);

  /*
   * Callback when user sends input in the chat component
   *
   * @param blocksInput - The content blocks input by the user
   * @param structuredInput - The structured input by the user, if any
   *
   * @return void
   */
  const onSendClick = useCallback(
    (
      blocksInput: ContentBlocks,
      structuredInput: Record<string, unknown> | null,
    ) => {
      const originalText = blocksInput
        .filter((b) => b.type === "text")
        .map((b) => (b as { text: string }).text)
        .join("");

      if (!isAdmin && isAdminOnlyInstruction(originalText)) {
        message.warning(t("auth.adminOnlyAction") || "该操作仅管理员可执行");
        return;
      }

      if (!currentInputRequest) {
        message.info(disabledInputHint);
        return;
      }

      if (isAdmin && !outboundResourceGroupId) {
        message.warning(
          t("runpage.adminResourceGroupRequired") ||
            "请先选择本次输入使用的目标用户组",
        );
        return;
      }

      if (isTrainingPoolRequired) {
        message.warning(
          t("runpage.trainingPoolRequired") || "请选择本次训练资源池",
        );
        return;
      }

      if (currentInputRequest) {
        const baseUsername = user?.username ?? fallbackUsername;
        const contextUsername = buildContextUsername(
          baseUsername,
          chatSessionId,
        );
        setRandomUsername(contextUsername);

        // 提取原始命令内容（用于向导验证）
        // 保存原始命令到 ref，供 onSendComplete 使用
        lastSentCommandRef.current = originalText;

        // 将用户名添加到消息内容中，格式: "[user_id] 消息内容"
        const formattedBlocks: ContentBlocks = blocksInput.map(
          (block, index) => {
            if (index === 0 && block.type === "text") {
              return {
                ...block,
                text: `[${contextUsername}] ${block.text}`,
              };
            }
            return block;
          },
        );

        // 记录用户发送的消息内容，用于检测消息是否已显示
        const messageText = formattedBlocks
          .filter((b) => b.type === "text")
          .map((b) => (b as { text: string }).text)
          .join("");
        const sentAt = Date.now();
        const repliesBeforeSend = filteredRepliesRef.current;
        const assistantCountBeforeSend =
          getAssistantReplyCount(repliesBeforeSend);
        const assistantMessageIdsBeforeSend =
          getAssistantMessageIds(repliesBeforeSend);
        setPendingUserMessage({
          content: messageText,
          sentAt,
          assistantCountBeforeSend,
          assistantMessageIdsBeforeSend,
        });

        sendUserInputToServer(currentInputRequest.requestId, formattedBlocks, {
          ...(structuredInput ?? {}),
          __medflowUserId: user?.id,
          __medflowUsername: user?.username,
          __medflowResourceGroupId: outboundResourceGroupId,
          __medflowTrainingPoolId: outboundTrainingPoolId,
          __medflowSessionId: chatSessionId,
          __medflowContextUsername: contextUsername,
        });
      }
    },
    [
      chatSessionId,
      currentInputRequest,
      fallbackUsername,
      filteredReplies,
      isAdmin,
      isTrainingPoolRequired,
      outboundResourceGroupId,
      outboundTrainingPoolId,
      setRandomUsername,
      t,
      user,
    ],
  );

  // 处理AI分析请求的回调
  const handleAskAIFromPanel = useCallback(
    (blocksInput: ContentBlocks) => {
      if (currentInputRequest) {
        if (isAdmin && !outboundResourceGroupId) {
          message.warning(
            t("runpage.adminResourceGroupRequired") ||
              "请先选择本次输入使用的目标用户组",
          );
          return;
        }

        if (isTrainingPoolRequired) {
          message.warning(
            t("runpage.trainingPoolRequired") || "请选择本次训练资源池",
          );
          return;
        }

        const baseUsername = user?.username ?? fallbackUsername;
        const contextUsername = buildContextUsername(
          baseUsername,
          chatSessionId,
        );
        setRandomUsername(contextUsername);

        // 将用户名添加到消息内容中，格式: "[user_id] 消息内容"
        const formattedBlocks: ContentBlocks = blocksInput.map(
          (block, index) => {
            if (index === 0 && block.type === "text") {
              return {
                ...block,
                text: `[${contextUsername}] ${block.text}`,
              };
            }
            return block;
          },
        );

        // 记录用户发送的消息内容
        const messageText = formattedBlocks
          .filter((b) => b.type === "text")
          .map((b) => (b as { text: string }).text)
          .join("");
        const sentAt = Date.now();
        const repliesBeforeSend = filteredRepliesRef.current;
        const assistantCountBeforeSend =
          getAssistantReplyCount(repliesBeforeSend);
        const assistantMessageIdsBeforeSend =
          getAssistantMessageIds(repliesBeforeSend);
        setPendingUserMessage({
          content: messageText,
          sentAt,
          assistantCountBeforeSend,
          assistantMessageIdsBeforeSend,
        });

        sendUserInputToServer(currentInputRequest.requestId, formattedBlocks, {
          __medflowUserId: user?.id,
          __medflowUsername: user?.username,
          __medflowResourceGroupId: outboundResourceGroupId,
          __medflowTrainingPoolId: outboundTrainingPoolId,
          __medflowSessionId: chatSessionId,
          __medflowContextUsername: contextUsername,
        });
      }
    },
    [
      chatSessionId,
      currentInputRequest,
      fallbackUsername,
      filteredReplies,
      isAdmin,
      isTrainingPoolRequired,
      outboundResourceGroupId,
      outboundTrainingPoolId,
      setRandomUsername,
      t,
      user,
    ],
  );

  const hasAssistantReplyAfterPending = useMemo(() => {
    if (!pendingUserMessage) return false;
    const currentAssistantCount = getAssistantReplyCount(filteredReplies);
    const hasAssistantReply =
      currentAssistantCount > pendingUserMessage.assistantCountBeforeSend;
    return hasAssistantReply;
  }, [filteredReplies, pendingUserMessage]);

  const isReplying = useMemo(() => {
    const nextIsReplying = Boolean(
      pendingUserMessage && !hasAssistantReplyAfterPending,
    );
    return nextIsReplying;
  }, [filteredReplies, hasAssistantReplyAfterPending, pendingUserMessage]);

  // 清理过期的 pendingUserMessage
  useEffect(() => {
    if (pendingUserMessage && hasAssistantReplyAfterPending) {
      setPendingUserMessage(null);
    }
  }, [filteredReplies, hasAssistantReplyAfterPending, pendingUserMessage]);

  const onInterruptClick = useCallback(() => {
    if (!isReplying) return;
    const contextUsername = buildContextUsername(
      user?.username ?? fallbackUsername,
      chatSessionId,
    );
    setPendingUserMessage(null);
    void cancelRuntimeResponse().then((response) => {
      if (response?.success) {
        messageApi.info(t("chat.response-stopped") || "已停止生成");
      }
    });
    void resetAgentContext(contextUsername, {
      silent: true,
      cancelWorkflows: true,
    });
  }, [
    cancelRuntimeResponse,
    chatSessionId,
    fallbackUsername,
    isReplying,
    messageApi,
    resetAgentContext,
    t,
    user,
  ]);

  // 将回调存入 ref，供父组件使用
  useEffect(() => {
    if (onAskAIRef) {
      onAskAIRef.current = handleAskAIFromPanel;
    }
  }, [handleAskAIFromPanel, onAskAIRef]);

  // Generate loading message based on command input
  const getLoadingMessage = useCallback((input: string): string => {
    const lowerInput = input.toLowerCase();
    if (lowerInput.includes("刷新") && lowerInput.includes("数据集"))
      return "⏳ 正在刷新数据集列表，请稍候...";
    if (lowerInput.includes("刷新") && lowerInput.includes("模型"))
      return "⏳ 正在刷新模型列表，请稍候...";
    if (lowerInput.includes("刷新") && lowerInput.includes("评测"))
      return "⏳ 正在刷新评测列表，请稍候...";
    if (lowerInput.includes("查询") && lowerInput.includes("数据集"))
      return "⏳ 正在查询数据集列表，请稍候...";
    if (lowerInput.includes("查询") && lowerInput.includes("模型"))
      return "⏳ 正在查询模型列表，请稍候...";
    if (lowerInput.includes("查询") && lowerInput.includes("评测"))
      return "⏳ 正在查询评测列表，请稍候...";
    if (lowerInput.includes("gpu") || lowerInput.includes("显卡"))
      return "⏳ 正在加载GPU状态，请稍候...";
    if (lowerInput.includes("下载") && lowerInput.includes("数据集"))
      return "⏳ 正在准备下载数据集，请稍候...";
    if (lowerInput.includes("下载") && lowerInput.includes("评测"))
      return "⏳ 正在准备下载评测文件，请稍候...";
    if (lowerInput.includes("系统") || lowerInput.includes("概览"))
      return "⏳ 正在加载系统状态，请稍候...";
    return "⏳ 正在执行命令，请稍候...";
  }, []);

  // Handle command result display
  const handleShowCommandResult = useCallback(
    (result: FormattedResult, loadingId?: string) => {
      // Remove loading message if loadingId is provided
      if (loadingId) {
        setCommandLoading((prev) =>
          prev.filter((item) => item.id !== loadingId),
        );
      }

      // Add result message
      const newResult = {
        id: `cmd-${Date.now()}`,
        content: result.content,
        timestamp: Date.now(),
      };
      setCommandResults((prev) => [...prev, newResult]);
    },
    [],
  );

  // Store loadingId in a ref so it's accessible in the handler
  const currentLoadingIdRef = useRef<string>("");

  // Setup /studio command handler
  const commandHandler: CommandHandler = {
    onTabChange,
    onQueryDatasets,
    onRefreshDatasets,
    onQueryModels,
    onRefreshModels,
    onQueryEvaluation: onQueryTests,
    onRefreshEvaluation: onRefreshTests,
    onQueryEvaluationResults:
      onQueryEvaluationResultsZero ||
      (async () => {
        const containerName = defaultEvaluateContainerName;
        return await onQueryEvaluationResults(containerName);
      }),
    onRefreshEvaluationResults,
    onDownloadDataset,
    onDownloadEvaluation: onDownloadTest,
    onUploadDataset: onUpload,
    onUploadEvaluation: onUploadTest,
    getDatasets: () => datasetsRef.current,
    getModels: () => modelsRef.current,
    getEvaluation: () => testsRef.current,
    // 缓存获取函数：返回 null 表示没有缓存
    getCachedDatasets: () => {
      const cached = localStorage.getItem("cached_datasets");
      if (!cached) return null;
      try {
        return JSON.parse(cached);
      } catch {
        return null;
      }
    },
    getCachedModels: () => {
      const cached = localStorage.getItem("cached_models");
      if (!cached) return null;
      try {
        return JSON.parse(cached);
      } catch {
        return null;
      }
    },
    getCachedTests: () => {
      const cached = localStorage.getItem("cached_tests");
      if (!cached) return null;
      try {
        return JSON.parse(cached);
      } catch {
        return null;
      }
    },
    getCachedEvaluationResults: () => {
      const cached = localStorage.getItem("cached_evaluation_results");
      if (!cached) return null;
      try {
        return JSON.parse(cached);
      } catch {
        return null;
      }
    },
    getDatasetCacheMeta: () => loadCachedMeta(CACHE_META_KEYS.datasets),
    getModelCacheMeta: () => loadCachedMeta(CACHE_META_KEYS.models),
    getTestCacheMeta: () => loadCachedMeta(CACHE_META_KEYS.tests),
    getEvaluationResultCacheMeta: () =>
      loadCachedMeta(CACHE_META_KEYS.evaluationResults),
    hasQueriedDatasets,
    hasQueriedModels,
    hasQueriedTests,
    getSystemOverview: () => systemOverviewDataRef.current,
    getGPUInfo: () => gpuInfoRef.current,
    requestGPUInfo: () => {
      return new Promise((resolve, reject) => {
        if (!socket) {
          reject(new Error("Socket not connected"));
          return;
        }

        // 监听一次GPU信息推送
        const handleGPUInfo = (data: GPUInfo[]) => {
          resolve(data);
        };

        socket.once(SocketEvents.server.pushGPUInfo, handleGPUInfo);

        // 发送请求
        socket.emit(SocketEvents.client.requestGPUInfo);

        // 设置超时
        setTimeout(() => {
          socket.off(SocketEvents.server.pushGPUInfo, handleGPUInfo);
          reject(new Error("GPU info request timeout"));
        }, 10000); // 10秒超时
      });
    },
    onShowResult: (result) => {
      // Use the loadingId from ref
      handleShowCommandResult(result, currentLoadingIdRef.current);
      // Clear the ref after showing result
      currentLoadingIdRef.current = "";
    },
  };

  const { processInput } = useNaturalLanguageCommands(commandHandler, {
    isAdmin,
  });

  // Handle command from AsChat
  const handleCommand = useCallback(
    async (input: string): Promise<boolean> => {
      // First, show user command input in chat
      const inputId = `cmd-input-${Date.now()}`;
      setCommandInputs((prev) => [
        ...prev,
        {
          id: inputId,
          content: input,
          timestamp: Date.now(),
        },
      ]);

      // Show loading indicator
      const loadingId = `cmd-loading-${Date.now()}`;
      currentLoadingIdRef.current = loadingId;
      setCommandLoading((prev) => [
        ...prev,
        {
          id: loadingId,
          inputId: inputId,
          content: getLoadingMessage(input),
          timestamp: Date.now(),
        },
      ]);

      // Then process the command
      const result = await processInput(input);

      // If command was not recognized, remove loading immediately
      if (!result) {
        setCommandLoading((prev) =>
          prev.filter((item) => item.id !== loadingId),
        );
        currentLoadingIdRef.current = "";
      }

      return result;
    },
    [processInput, getLoadingMessage],
  );

  const handleAgentWaitingReply = useCallback(
    (text: string) => {
      const trimmedText = text.trim();
      if (!trimmedText) {
        return;
      }

      if (!isAdmin && isAdminOnlyInstruction(trimmedText)) {
        message.warning(t("auth.adminOnlyAction") || "该操作仅管理员可执行");
        return;
      }

      if (currentInputRequest) {
        onSendClick(
          [
            {
              type: "text",
              text: trimmedText,
            } as ContentBlock & { text: string },
          ],
          null,
        );
        setAsChatInputText("");
        setInputInlineHint("");
        return;
      }

      setAsChatInputText(trimmedText);
      setSendButtonHighlightToken((prev) => prev + 1);
      setInputInlineHint(
        t("agentWaiting.replyFilledHint", {
          defaultValue: "已填入回复，等待当前会话可输入后发送",
        }),
      );
    },
    [currentInputRequest, isAdmin, onSendClick, t],
  );

  // Combine filtered replies with command inputs and results
  const combinedReplies = useMemo(() => {
    // Command inputs (user messages)
    const commandInputReplies = commandInputs.map((input) => ({
      replyId: input.id,
      replyName: "User",
      replyRole: "user",
      createdAt: new Date(input.timestamp).toISOString(),
      finishedAt: new Date(input.timestamp).toISOString(),
      messages: [
        {
          id: input.id,
          name: "User",
          role: "user",
          content: input.content,
          timestamp: new Date(input.timestamp).toISOString(),
        },
      ],
    })) as Reply[];

    // Command loading messages (assistant messages showing loading state)
    const commandLoadingReplies = commandLoading.map((loading) => ({
      replyId: loading.id,
      replyName: "System",
      replyRole: "assistant",
      createdAt: new Date(loading.timestamp).toISOString(),
      finishedAt: new Date(loading.timestamp).toISOString(),
      messages: [
        {
          id: loading.id,
          name: "System",
          role: "assistant",
          content: loading.content,
          timestamp: new Date(loading.timestamp).toISOString(),
        },
      ],
    })) as Reply[];

    // Command results (assistant messages)
    const commandResultReplies = commandResults.map((result) => ({
      replyId: result.id,
      replyName: "System",
      replyRole: "assistant",
      createdAt: new Date(result.timestamp).toISOString(),
      finishedAt: new Date(result.timestamp).toISOString(),
      messages: [
        {
          id: result.id,
          name: "System",
          role: "assistant",
          content: result.content,
          timestamp: new Date(result.timestamp).toISOString(),
        },
      ],
    })) as Reply[];

    const displayUsername = randomUsername;
    const displayFilteredReplies = sanitizeRepliesForDisplay(
      filteredReplies,
      displayUsername,
    );

    // Combine all and sort by timestamp
    const allReplies = [
      ...displayFilteredReplies,
      ...commandInputReplies,
      ...commandLoadingReplies,
      ...commandResultReplies,
    ];
    return allReplies.sort((a, b) => {
      const timeA = new Date(a.createdAt).getTime();
      const timeB = new Date(b.createdAt).getTime();
      return timeA - timeB;
    });
  }, [
    filteredReplies,
    commandInputs,
    commandLoading,
    commandResults,
    randomUsername,
    user,
  ]);

  useEffect(() => {
    onCombinedRepliesChange?.(combinedReplies);
  }, [combinedReplies, onCombinedRepliesChange]);

  const shouldHideStoppedRunChatHistory =
    runData?.status === Status.DONE && inputRequests.length === 0;

  const chatDisplayReplies = useMemo(
    () => (shouldHideStoppedRunChatHistory ? [] : combinedReplies),
    [combinedReplies, shouldHideStoppedRunChatHistory],
  );

  const statusBarReplies = chatDisplayReplies;

  const latestWorkflowTask = useMemo(() => {
    for (
      let replyIndex = statusBarReplies.length - 1;
      replyIndex >= 0;
      replyIndex -= 1
    ) {
      const reply = statusBarReplies[replyIndex];
      if (reply.replyRole.toLowerCase() !== "assistant") {
        continue;
      }

      for (
        let messageIndex = reply.messages.length - 1;
        messageIndex >= 0;
        messageIndex -= 1
      ) {
        const messageItem = reply.messages[messageIndex];
        const metadataProtocol = extractProtocolFromMetadata(
          messageItem.metadata,
        );
        if (isWorkflowProtocol(metadataProtocol)) {
          if (isWorkflowDismissStatus(metadataProtocol.workflowStatus)) {
            return null;
          }
          const metadataSummary = workflowSummaryFromProtocol(metadataProtocol);
          if (metadataSummary) {
            return metadataSummary;
          }
        }

        const messageText = extractTextFromContent(messageItem.content);
        const textProtocol = extractProtocolFromText(messageText);
        if (isWorkflowProtocol(textProtocol)) {
          if (isWorkflowDismissStatus(textProtocol.workflowStatus)) {
            return null;
          }
          const textSummary = workflowSummaryFromProtocol(textProtocol);
          if (textSummary) {
            return textSummary;
          }
        }
      }
    }

    return null;
  }, [statusBarReplies]);

  const latestWorkflowDismissSignal = useMemo(() => {
    for (
      let replyIndex = statusBarReplies.length - 1;
      replyIndex >= 0;
      replyIndex -= 1
    ) {
      const reply = statusBarReplies[replyIndex];
      if (reply.replyRole.toLowerCase() !== "assistant") {
        continue;
      }

      for (
        let messageIndex = reply.messages.length - 1;
        messageIndex >= 0;
        messageIndex -= 1
      ) {
        const messageItem = reply.messages[messageIndex];
        const metadataProtocol = extractProtocolFromMetadata(
          messageItem.metadata,
        );
        if (isWorkflowProtocol(metadataProtocol)) {
          return isWorkflowDismissStatus(metadataProtocol.workflowStatus)
            ? metadataProtocol.workflowId || "__workflow__"
            : null;
        }

        const messageText = extractTextFromContent(messageItem.content);
        const textProtocol = extractProtocolFromText(messageText);
        if (isWorkflowProtocol(textProtocol)) {
          return isWorkflowDismissStatus(textProtocol.workflowStatus)
            ? textProtocol.workflowId || "__workflow__"
            : null;
        }
      }
    }

    return null;
  }, [statusBarReplies]);

  useEffect(() => {
    if (!latestWorkflowDismissSignal) {
      return;
    }
    setPolledWorkflowTask((current) =>
      !current?.workflowId || current.workflowId === latestWorkflowDismissSignal
        ? null
        : current,
    );
    setWorkflowMonitorState({ isPolling: false });
    setHiddenStatusBars((prev) =>
      prev.workflow ? prev : { ...prev, workflow: true },
    );
    setClosedStatusBars((prev) =>
      prev.workflow ? prev : { ...prev, workflow: true },
    );
  }, [latestWorkflowDismissSignal]);

  useEffect(() => {
    if (shouldHideStoppedRunChatHistory) {
      setPolledWorkflowTask(null);
      return;
    }
    if (!latestWorkflowTask) {
      return;
    }
    setPolledWorkflowTask((current) => {
      if (
        current?.workflowId === latestWorkflowTask.workflowId &&
        isWorkflowDismissStatus(current.workflowStatus)
      ) {
        return current;
      }
      return mergeWorkflowTaskSummary(current, latestWorkflowTask);
    });
  }, [latestWorkflowTask, shouldHideStoppedRunChatHistory]);

  const displayedWorkflowTask = useMemo(() => {
    if (shouldHideStoppedRunChatHistory) {
      return null;
    }
    if (latestWorkflowTask && polledWorkflowTask) {
      return mergeWorkflowTaskSummary(latestWorkflowTask, polledWorkflowTask);
    }
    return polledWorkflowTask || latestWorkflowTask;
  }, [latestWorkflowTask, polledWorkflowTask, shouldHideStoppedRunChatHistory]);

  const workflowTrainingTask = useMemo<TrainingTaskSummary | null>(() => {
    if (
      !displayedWorkflowTask ||
      displayedWorkflowTask.currentStage !== "train" ||
      (!displayedWorkflowTask.container && !displayedWorkflowTask.pid)
    ) {
      return null;
    }
    return {
      workflowId: displayedWorkflowTask.workflowId,
      container: displayedWorkflowTask.container,
      pid: displayedWorkflowTask.pid,
      trainType: displayedWorkflowTask.trainType,
      launchMode: displayedWorkflowTask.launchMode,
      isMultinode: displayedWorkflowTask.isMultinode,
      scriptName: displayedWorkflowTask.scriptName,
      wandbUrl: displayedWorkflowTask.wandbUrl,
    };
  }, [displayedWorkflowTask]);

  useEffect(() => {
    const intervalId = window.setInterval(
      () => setWorkflowMonitorNow(Date.now()),
      1_000,
    );
    return () => window.clearInterval(intervalId);
  }, []);

  const workflowMonitorTarget = useMemo(
    () =>
      displayedWorkflowTask?.workflowStatus === "running" &&
      displayedWorkflowTask.workflowId
        ? {
            workflowId: displayedWorkflowTask.workflowId,
            runId,
            nodeId: conversationResourceNodeId,
          }
        : null,
    [
      displayedWorkflowTask?.workflowId,
      displayedWorkflowTask?.workflowStatus,
      runId,
      conversationResourceNodeId,
    ],
  );

  useEffect(() => {
    if (!workflowMonitorTarget) {
      setWorkflowMonitorState((current) => ({
        ...current,
        isPolling: false,
        nextRefreshAt: undefined,
      }));
      return;
    }

    let disposed = false;
    let inFlight = false;
    const intervalMs = 5_000;
    const target = workflowMonitorTarget;

    const fetchStatus = async () => {
      if (disposed || inFlight) {
        return;
      }
      inFlight = true;
      setWorkflowMonitorState((current) => ({
        ...current,
        isPolling: true,
        error: undefined,
      }));
      try {
        const result = await queryWorkflowStatus(target);
        if (disposed) {
          return;
        }
        if (!result.success || !result.data) {
          console.warn("[RunPage][WorkflowMonitor] query returned failure", {
            workflowId: target.workflowId,
            message: result.message,
          });
          setWorkflowMonitorState((current) => ({
            ...current,
            isPolling: false,
            error: workflowStatusErrorMessage(result.message),
            nextRefreshAt: Date.now() + intervalMs,
          }));
          return;
        }

        const summary = workflowSummaryFromProtocol(
          result.data as TrainingJobProtocol,
        );
        if (!summary) {
          console.warn("[RunPage][WorkflowMonitor] query returned no workflow summary", {
            workflowId: target.workflowId,
            data: result.data,
          });
          setWorkflowMonitorState((current) => ({
            ...current,
            isPolling: false,
            nextRefreshAt: Date.now() + intervalMs,
          }));
          return;
        }
        if (isWorkflowDismissStatus(summary.workflowStatus)) {
          setPolledWorkflowTask(null);
          setWorkflowMonitorState({
            isPolling: false,
            lastUpdatedAt: Date.now(),
          });
          setHiddenStatusBars((prev) =>
            prev.workflow ? prev : { ...prev, workflow: true },
          );
          setClosedStatusBars((prev) =>
            prev.workflow ? prev : { ...prev, workflow: true },
          );
          return;
        }
        setPolledWorkflowTask((current) =>
          mergeWorkflowTaskSummary(current, summary),
        );
        setWorkflowMonitorState({
          isPolling: false,
          lastUpdatedAt: Date.now(),
          nextRefreshAt: Date.now() + intervalMs,
        });
      } catch (error) {
        if (!disposed) {
          console.warn("[RunPage][WorkflowMonitor] query failed", error);
          setWorkflowMonitorState((current) => ({
            ...current,
            isPolling: false,
            error: workflowStatusErrorMessage(error),
            nextRefreshAt: Date.now() + intervalMs,
          }));
        }
      } finally {
        inFlight = false;
      }
    };

    setWorkflowMonitorState((current) => ({
      ...current,
      nextRefreshAt: Date.now(),
    }));
    void fetchStatus();
    const intervalId = window.setInterval(() => void fetchStatus(), intervalMs);
    return () => {
      disposed = true;
      window.clearInterval(intervalId);
    };
  }, [queryWorkflowStatus, workflowMonitorTarget]);

  const latestDataFilterTask = useMemo(() => {
    const findLatestInputFolderBefore = (beforeReplyIndex: number) => {
      for (let index = beforeReplyIndex; index >= 0; index -= 1) {
        const reply = statusBarReplies[index];
        if (reply.replyRole.toLowerCase() === "assistant") {
          continue;
        }
        for (
          let messageIndex = reply.messages.length - 1;
          messageIndex >= 0;
          messageIndex -= 1
        ) {
          const text = extractTextFromContent(reply.messages[messageIndex].content);
          const inputFolder = text.match(/input_folder\s*[=：:]\s*`?([^\s`\n，,]+)/i)?.[1];
          if (inputFolder) {
            return inputFolder;
          }
        }
      }
      return undefined;
    };

    for (
      let replyIndex = statusBarReplies.length - 1;
      replyIndex >= 0;
      replyIndex -= 1
    ) {
      const reply = statusBarReplies[replyIndex];
      if (reply.replyRole.toLowerCase() !== "assistant") {
        continue;
      }

      for (
        let messageIndex = reply.messages.length - 1;
        messageIndex >= 0;
        messageIndex -= 1
      ) {
        const messageItem = reply.messages[messageIndex];
        const messageText = extractTextFromContent(messageItem.content);
        const metadataProtocol = extractProtocolFromMetadata(
          messageItem.metadata,
        );
        if (
          isWorkflowProtocol(metadataProtocol) ||
          isDataFilterDismissProtocol(metadataProtocol)
        ) {
          return null;
        }

        const metadataSummary = dataFilterSummaryFromProtocol(metadataProtocol);
        if (metadataSummary) {
          const textSummary = parseDataFilterTaskSummaryFromText(messageText);
          const inputFolder =
            metadataSummary.inputFolder ||
            textSummary?.inputFolder ||
            findLatestInputFolderBefore(replyIndex - 1);
          const threshold = metadataSummary.threshold ?? textSummary?.threshold ?? 90;
          const outputFolder =
            metadataSummary.outputFolder ||
            textSummary?.outputFolder ||
            (inputFolder ? `${inputFolder}_${threshold}_score_filter` : undefined);
          const outputDatasetName =
            metadataSummary.outputDatasetName ||
            textSummary?.outputDatasetName ||
            (outputFolder ? outputFolder.split("/").filter(Boolean).pop() : undefined);
          return {
            ...metadataSummary,
            ...textSummary,
            container:
              metadataSummary.container ||
              textSummary?.container ||
              currentTrainingContainerName,
            inputFolder,
            outputFolder,
            outputDatasetName,
            threshold,
            sourceKey:
              messageItem.timestamp ||
              reply.createdAt ||
              `${reply.replyId}:${messageItem.id}:metadata`,
          };
        }

        if (
          isWorkflowProtocol(extractProtocolFromText(messageText)) ||
          isDataFilterDismissText(messageText)
        ) {
          return null;
        }

        const textSummary = parseDataFilterTaskSummaryFromText(messageText);
        if (textSummary) {
          const inputFolder =
            textSummary.inputFolder || findLatestInputFolderBefore(replyIndex - 1);
          const threshold = textSummary.threshold ?? 90;
          const outputFolder =
            textSummary.outputFolder ||
            (inputFolder ? `${inputFolder}_${threshold}_score_filter` : undefined);
          const outputDatasetName =
            textSummary.outputDatasetName ||
            (outputFolder ? outputFolder.split("/").filter(Boolean).pop() : undefined);
          if (!outputFolder && !outputDatasetName) {
            return null;
          }
          return {
            ...textSummary,
            inputFolder,
            outputFolder,
            outputDatasetName,
            threshold,
            sourceKey:
              messageItem.timestamp ||
              reply.createdAt ||
              `${reply.replyId}:${messageItem.id}:text`,
          };
        }
      }
    }

    return null;
  }, [currentTrainingContainerName, statusBarReplies]);

  const latestDataFilterDismissSignal = useMemo(() => {
    if (shouldHideStoppedRunChatHistory) {
      return "run-ended";
    }

    for (
      let replyIndex = statusBarReplies.length - 1;
      replyIndex >= 0;
      replyIndex -= 1
    ) {
      const reply = statusBarReplies[replyIndex];
      if (reply.replyRole.toLowerCase() !== "assistant") {
        continue;
      }

      for (
        let messageIndex = reply.messages.length - 1;
        messageIndex >= 0;
        messageIndex -= 1
      ) {
        const messageItem = reply.messages[messageIndex];
        const metadataProtocol = extractProtocolFromMetadata(
          messageItem.metadata,
        );
        if (isDataFilterDismissProtocol(metadataProtocol)) {
          return `${reply.replyId}:${messageItem.id}:metadata`;
        }
        if (dataFilterSummaryFromProtocol(metadataProtocol)) {
          return "";
        }

        const messageText = extractTextFromContent(messageItem.content);
        if (parseDataFilterTaskSummaryFromText(messageText)) {
          return "";
        }
        if (isDataFilterDismissText(messageText)) {
          return `${reply.replyId}:${messageItem.id}:text`;
        }
      }
    }

    return "";
  }, [statusBarReplies, shouldHideStoppedRunChatHistory]);

  useEffect(() => {
    if (!latestDataFilterDismissSignal) {
      return;
    }
    setDismissedDataFilterTaskSourceKey(
      latestDataFilterTask?.sourceKey || latestDataFilterDismissSignal,
    );
    setPolledDataFilterTask(null);
    setDataFilterMonitorState({ isPolling: false });
    setHiddenStatusBars((prev) =>
      prev.data_filter ? prev : { ...prev, data_filter: true },
    );
    setClosedStatusBars((prev) =>
      prev.data_filter ? prev : { ...prev, data_filter: true },
    );
  }, [latestDataFilterDismissSignal, latestDataFilterTask]);

  useEffect(() => {
    if (!latestDataFilterTask) {
      return;
    }
    if (
      latestDataFilterTask.sourceKey &&
      latestDataFilterTask.sourceKey === dismissedDataFilterTaskSourceKey
    ) {
      return;
    }
    setDismissedDataFilterTaskSourceKey(null);
    setPolledDataFilterTask((current) =>
      mergeDataFilterTaskSummary(current, latestDataFilterTask),
    );
    setHiddenStatusBars((prev) =>
      prev.data_filter ? prev : { ...prev, data_filter: false },
    );
    setClosedStatusBars((prev) =>
      prev.data_filter ? prev : { ...prev, data_filter: false },
    );
  }, [latestDataFilterTask, dismissedDataFilterTaskSourceKey]);

  const activeLatestDataFilterTask =
    latestDataFilterTask?.sourceKey &&
    latestDataFilterTask.sourceKey === dismissedDataFilterTaskSourceKey
      ? null
      : latestDataFilterTask;

  const displayedDataFilterTask = shouldHideStoppedRunChatHistory
    ? null
    : polledDataFilterTask || activeLatestDataFilterTask;

  useEffect(() => {
    const intervalId = window.setInterval(
      () => setDataFilterMonitorNow(Date.now()),
      1_000,
    );
    return () => window.clearInterval(intervalId);
  }, []);

  const fetchDataFilterStatus = useCallback(
    async (manual = false) => {
      if (!displayedDataFilterTask?.container) {
        return;
      }

      setDataFilterMonitorState((current) => ({
        ...current,
        isPolling: !manual,
        isManualRefreshing: manual,
        error: undefined,
      }));

      try {
        const result = await queryDataFilterStatusMutation.mutateAsync({
          container: displayedDataFilterTask.container,
          outputFolder: displayedDataFilterTask.outputFolder,
          inputFolder: displayedDataFilterTask.inputFolder,
          threshold: displayedDataFilterTask.threshold,
          nodeId: conversationResourceNodeId,
          runId,
        });

        if (!result.success || !result.data) {
          setDataFilterMonitorState((current) => ({
            ...current,
            isPolling: false,
            isManualRefreshing: false,
            error: result.message || t("runpage.query-failed"),
            consecutiveErrors: (current.consecutiveErrors || 0) + 1,
          }));
          return;
        }

        const payload = (result.data as { data?: Record<string, unknown> }).data || result.data;
        const updatedSummary: DataFilterTaskSummary = {
          ...displayedDataFilterTask,
          status: normalizeDataFilterStatus(payload.status),
          percent: normalizePercent(payload.percent),
          currentFile:
            typeof payload.current_file === "string"
              ? payload.current_file
              : undefined,
          processedItems:
            typeof payload.processed_items === "number"
              ? payload.processed_items
              : undefined,
          totalItems:
            typeof payload.total_items === "number" ? payload.total_items : undefined,
          passedItems:
            typeof payload.passed_items === "number" ? payload.passed_items : undefined,
          rejectedItems:
            typeof payload.rejected_items === "number"
              ? payload.rejected_items
              : undefined,
          apiFailedItems:
            typeof payload.api_failed_items === "number"
              ? payload.api_failed_items
              : undefined,
          invalidItems:
            typeof payload.invalid_items === "number" ? payload.invalid_items : undefined,
          resumedItems:
            typeof payload.resumed_items === "number" ? payload.resumed_items : undefined,
          error: typeof payload.error === "string" ? payload.error : undefined,
          updatedAt:
            typeof payload.updated_at === "string" ? payload.updated_at : undefined,
          finishedAt:
            typeof payload.finished_at === "string" ? payload.finished_at : undefined,
        };

        if (updatedSummary.status === "completed") {
          setDismissedDataFilterTaskSourceKey(
            updatedSummary.sourceKey ||
              updatedSummary.outputFolder ||
              updatedSummary.outputDatasetName ||
              null,
          );
          setPolledDataFilterTask(null);
          setDataFilterMonitorState({
            isPolling: false,
            isManualRefreshing: false,
            lastUpdatedAt: Date.now(),
          });
          setHiddenStatusBars((prev) =>
            prev.data_filter ? prev : { ...prev, data_filter: true },
          );
          setClosedStatusBars((prev) =>
            prev.data_filter ? prev : { ...prev, data_filter: true },
          );
          return;
        }

        setPolledDataFilterTask((current) =>
          mergeDataFilterTaskSummary(current, updatedSummary),
        );
        setDataFilterMonitorState((current) => ({
          ...current,
          isPolling: false,
          isManualRefreshing: false,
          lastUpdatedAt: Date.now(),
          error: undefined,
          consecutiveErrors: 0,
        }));
      } catch (error) {
        console.warn("[RunPage][DataFilterMonitor] query failed", error);
        setDataFilterMonitorState((current) => ({
          ...current,
          isPolling: false,
          isManualRefreshing: false,
          error: dataFilterStatusErrorMessage(error),
          consecutiveErrors: (current.consecutiveErrors || 0) + 1,
        }));
      }
    },
    [
      displayedDataFilterTask,
      conversationResourceNodeId,
      runId,
      queryDataFilterStatusMutation.mutateAsync,
      t,
    ],
  );

  useEffect(() => {
    if (
      displayedDataFilterTask?.status !== "running" ||
      !displayedDataFilterTask.container
    ) {
      setDataFilterMonitorState((current) => ({
        ...current,
        isPolling: false,
        nextRefreshAt: undefined,
      }));
      return;
    }

    let disposed = false;
    let timeoutId: number | undefined;

    const scheduleNext = () => {
      if (disposed) {
        return;
      }
      const errorCount = dataFilterMonitorState.consecutiveErrors || 0;
      const intervalMs = errorCount >= 3 ? 20_000 : 10_000;
      setDataFilterMonitorState((current) => ({
        ...current,
        nextRefreshAt: Date.now() + intervalMs,
      }));
      timeoutId = window.setTimeout(
        () => void fetchDataFilterStatus(false),
        intervalMs,
      );
    };

    scheduleNext();

    return () => {
      disposed = true;
      if (timeoutId !== undefined) {
        window.clearTimeout(timeoutId);
      }
    };
  }, [
    displayedDataFilterTask?.status,
    displayedDataFilterTask?.container,
    displayedDataFilterTask?.outputFolder,
    displayedDataFilterTask?.inputFolder,
    displayedDataFilterTask?.threshold,
    dataFilterMonitorState.consecutiveErrors,
    fetchDataFilterStatus,
  ]);

  const workflowRefreshCountdown = workflowMonitorState.nextRefreshAt
    ? Math.max(
        0,
        Math.ceil(
          (workflowMonitorState.nextRefreshAt - workflowMonitorNow) / 1_000,
        ),
      )
    : undefined;
  const workflowLastUpdatedText = workflowMonitorState.lastUpdatedAt
    ? new Date(workflowMonitorState.lastUpdatedAt).toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      })
    : undefined;
  const dataFilterRefreshCountdown = dataFilterMonitorState.nextRefreshAt
    ? Math.max(
        0,
        Math.ceil(
          (dataFilterMonitorState.nextRefreshAt - dataFilterMonitorNow) / 1_000,
        ),
      )
    : undefined;
  const dataFilterLastUpdatedText = dataFilterMonitorState.lastUpdatedAt
    ? new Date(dataFilterMonitorState.lastUpdatedAt).toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      })
    : undefined;

  const latestTrainingTask = useMemo(() => {
    for (
      let replyIndex = statusBarReplies.length - 1;
      replyIndex >= 0;
      replyIndex -= 1
    ) {
      const reply = statusBarReplies[replyIndex];
      if (reply.replyRole.toLowerCase() !== "assistant") {
        continue;
      }

      for (
        let messageIndex = reply.messages.length - 1;
        messageIndex >= 0;
        messageIndex -= 1
      ) {
        const messageItem = reply.messages[messageIndex];
        const metadataProtocol = extractProtocolFromMetadata(
          messageItem.metadata,
        );
        if (isWorkflowProtocol(metadataProtocol)) {
          return null;
        }
        if (isTrainingTerminalProtocol(metadataProtocol)) {
          return null;
        }
        if (
          metadataProtocol?.type === "job_stopped" &&
          metadataProtocol.jobType === "train"
        ) {
          return null;
        }

        if (isExplicitNonTrainingJobProtocol(metadataProtocol)) {
          continue;
        }
        const metadataSummary = trainingSummaryFromProtocol(metadataProtocol);
        if (metadataSummary) {
          return metadataSummary;
        }

        const messageText = extractTextFromContent(messageItem.content);
        const textProtocol = extractProtocolFromText(messageText);
        if (isWorkflowProtocol(textProtocol)) {
          return null;
        }
        if (isExplicitNonTrainingJobProtocol(textProtocol)) {
          continue;
        }
        if (isTrainingTaskStopped(messageText)) {
          return null;
        }

        const summary = parseTrainingTaskSummary(messageText);
        if (summary) {
          return summary;
        }
      }
    }

    return null;
  }, [statusBarReplies]);
  const metricsTrainingTask = latestTrainingTask || workflowTrainingTask;
  const metricsCacheKey = useMemo(() => {
    if (!metricsTrainingTask) {
      return null;
    }
    return JSON.stringify([
      runId,
      metricsTrainingTask.workflowId || "",
      metricsTrainingTask.container || "",
      metricsTrainingTask.pid || "",
      metricsTrainingTask.trainType || "",
      metricsTrainingTask.status || "",
      metricsTrainingTask.launchMode || "",
      metricsTrainingTask.isMultinode ? "multinode" : "",
      metricsTrainingTask.scriptName || "",
    ]);
  }, [
    runId,
    metricsTrainingTask?.workflowId,
    metricsTrainingTask?.container,
    metricsTrainingTask?.pid,
    metricsTrainingTask?.trainType,
    metricsTrainingTask?.status,
    metricsTrainingTask?.launchMode,
    metricsTrainingTask?.isMultinode,
    metricsTrainingTask?.scriptName,
  ]);

  useEffect(() => {
    onSilentMonitorCacheKeyChange?.(metricsCacheKey);
  }, [metricsCacheKey, onSilentMonitorCacheKeyChange]);
  const metricsMonitorTarget = useMemo(() => {
    if (!metricsTrainingTask || !metricsCacheKey) {
      return null;
    }

    return {
      key: metricsCacheKey,
      workflowId: metricsTrainingTask.workflowId,
      container: metricsTrainingTask.container,
      pid: metricsTrainingTask.pid,
      trainType: metricsTrainingTask.trainType,
      launchMode: metricsTrainingTask.launchMode,
      isMultinode: metricsTrainingTask.isMultinode,
      scriptName: metricsTrainingTask.scriptName,
    };
  }, [
    metricsCacheKey,
    metricsTrainingTask?.workflowId,
    metricsTrainingTask?.container,
    metricsTrainingTask?.pid,
    metricsTrainingTask?.trainType,
    metricsTrainingTask?.launchMode,
    metricsTrainingTask?.isMultinode,
    metricsTrainingTask?.scriptName,
  ]);

  const latestAssessmentTask = useMemo(() => {
    for (
      let replyIndex = statusBarReplies.length - 1;
      replyIndex >= 0;
      replyIndex -= 1
    ) {
      const reply = statusBarReplies[replyIndex];
      if (reply.replyRole.toLowerCase() !== "assistant") {
        continue;
      }

      for (
        let messageIndex = reply.messages.length - 1;
        messageIndex >= 0;
        messageIndex -= 1
      ) {
        const messageItem = reply.messages[messageIndex];
        const metadataProtocol = extractProtocolFromMetadata(
          messageItem.metadata,
        );
        if (isWorkflowProtocol(metadataProtocol)) {
          return null;
        }
        if (
          metadataProtocol?.type === "job_stopped" &&
          ["assessment", "evaluate"].includes(metadataProtocol.jobType || "")
        ) {
          return null;
        }

        const metadataSummary = assessmentSummaryFromProtocol(metadataProtocol);
        if (metadataSummary) {
          return metadataSummary;
        }

        const messageText = extractTextFromContent(messageItem.content);
        if (isWorkflowProtocol(extractProtocolFromText(messageText))) {
          return null;
        }
        if (isAssessmentTaskStopped(messageText)) {
          return null;
        }

        const summary = parseAssessmentTaskSummary(messageText);
        if (summary) {
          return summary;
        }
      }
    }

    return null;
  }, [statusBarReplies]);

  const latestBenchmarkTask = useMemo(() => {
    for (
      let replyIndex = statusBarReplies.length - 1;
      replyIndex >= 0;
      replyIndex -= 1
    ) {
      const reply = statusBarReplies[replyIndex];
      if (reply.replyRole.toLowerCase() !== "assistant") {
        continue;
      }

      for (
        let messageIndex = reply.messages.length - 1;
        messageIndex >= 0;
        messageIndex -= 1
      ) {
        const messageItem = reply.messages[messageIndex];
        const metadataProtocol = extractProtocolFromMetadata(
          messageItem.metadata,
        );
        if (isWorkflowProtocol(metadataProtocol)) {
          return null;
        }
        if (isBenchmarkStopProtocol(metadataProtocol)) {
          return null;
        }

        const metadataSummary = benchmarkSummaryFromProtocol(metadataProtocol);
        if (metadataSummary) {
          return metadataSummary;
        }

        const messageText = extractTextFromContent(messageItem.content);
        if (isWorkflowProtocol(extractProtocolFromText(messageText))) {
          return null;
        }
        if (isBenchmarkTaskStopped(messageText)) {
          return null;
        }

        const protocolSummary = benchmarkSummaryFromProtocol(
          extractProtocolFromText(messageText),
        );
        if (protocolSummary) {
          return protocolSummary;
        }

        const summary = benchmarkSummaryFromText(messageText);
        if (summary) {
          return summary;
        }
      }
    }

    return null;
  }, [statusBarReplies]);

  const latestBenchmarkStopSignal = useMemo(() => {
    for (
      let replyIndex = statusBarReplies.length - 1;
      replyIndex >= 0;
      replyIndex -= 1
    ) {
      const reply = statusBarReplies[replyIndex];
      if (reply.replyRole.toLowerCase() !== "assistant") {
        continue;
      }

      for (
        let messageIndex = reply.messages.length - 1;
        messageIndex >= 0;
        messageIndex -= 1
      ) {
        const messageItem = reply.messages[messageIndex];
        const metadataProtocol = extractProtocolFromMetadata(
          messageItem.metadata,
        );
        if (isBenchmarkStopProtocol(metadataProtocol)) {
          return `${reply.replyId}:${messageItem.id}:metadata`;
        }

        const messageText = extractTextFromContent(messageItem.content);
        if (isBenchmarkTaskStopped(messageText)) {
          return `${reply.replyId}:${messageItem.id}:text`;
        }
      }
    }

    return "";
  }, [statusBarReplies]);

  useEffect(() => {
    if (latestBenchmarkTask) {
      setBenchmarkTaskSnapshot(latestBenchmarkTask);
    }
  }, [latestBenchmarkTask]);

  useEffect(() => {
    if (latestBenchmarkStopSignal) {
      setPendingBenchmarkStop(false);
    }
  }, [latestBenchmarkStopSignal]);

  const displayedBenchmarkTask =
    latestBenchmarkTask ||
    (pendingBenchmarkStop ? benchmarkTaskSnapshot : null);

  const latestInferenceTask = useMemo(() => {
    let mergedSummary: InferenceTaskSummary | null = null;

    for (
      let replyIndex = statusBarReplies.length - 1;
      replyIndex >= 0;
      replyIndex -= 1
    ) {
      const reply = statusBarReplies[replyIndex];
      if (reply.replyRole.toLowerCase() !== "assistant") {
        continue;
      }

      for (
        let messageIndex = reply.messages.length - 1;
        messageIndex >= 0;
        messageIndex -= 1
      ) {
        const messageItem = reply.messages[messageIndex];
        const metadataProtocol = extractProtocolFromMetadata(
          messageItem.metadata,
        );
        if (isWorkflowProtocol(metadataProtocol)) {
          return mergedSummary;
        }
        if (isInferenceServiceStoppedProtocol(metadataProtocol)) {
          return mergedSummary;
        }

        const metadataMessage =
          metadataProtocol?.type === "message"
            ? metadataProtocol.message || ""
            : "";
        if (metadataMessage && isInferenceTaskStopped(metadataMessage)) {
          return mergedSummary;
        }

        const metadataSummary = inferenceSummaryFromProtocol(metadataProtocol);
        if (metadataSummary) {
          const sourceKey =
            messageItem.timestamp ||
            reply.createdAt ||
            `${replyIndex}:${messageIndex}`;
          mergedSummary = mergeInferenceTaskSummary(mergedSummary, {
            ...metadataSummary,
            sourceKey,
          });
          if (mergedSummary.hasConfig && mergedSummary.hasStatus) {
            return mergedSummary;
          }
          continue;
        }

        const messageText = extractTextFromContent(messageItem.content);
        if (
          isWorkflowProtocol(extractProtocolFromText(messageText)) ||
          /一键工作流|one-click workflow/i.test(messageText)
        ) {
          return mergedSummary;
        }
        const protocolSummary =
          parseInferenceTaskSummary(messageText) ||
          parseInferenceTaskSummaryFromText(messageText);
        if (protocolSummary) {
          const sourceKey =
            messageItem.timestamp ||
            reply.createdAt ||
            `${replyIndex}:${messageIndex}`;
          mergedSummary = mergeInferenceTaskSummary(mergedSummary, {
            ...protocolSummary,
            sourceKey,
          });
          if (mergedSummary.hasConfig && mergedSummary.hasStatus) {
            return mergedSummary;
          }
          continue;
        }

        if (isInferenceTaskStopped(messageText)) {
          return mergedSummary;
        }
      }
    }

    return mergedSummary;
  }, [statusBarReplies]);

  const visibleWorkflowTask = displayedWorkflowTask;
  const hasVisibleTrainingStatusBar =
    !!latestTrainingTask && !displayedWorkflowTask && !closedStatusBars.train;

  const trainingStatusKey = latestTrainingTask
    ? [
        latestTrainingTask.container || "",
        latestTrainingTask.pid || "",
        latestTrainingTask.trainType || "",
        latestTrainingTask.status || "",
        latestTrainingTask.launchMode || "",
      ].join("|")
    : "";
  const workflowStatusKey = displayedWorkflowTask
    ? [
        displayedWorkflowTask.workflowId || "",
        displayedWorkflowTask.workflowStatus || "",
        displayedWorkflowTask.currentStage || "",
        displayedWorkflowTask.currentStageStatus || "",
        displayedWorkflowTask.currentStageMessage || "",
        displayedWorkflowTask.evaluationDatasetName || "",
        displayedWorkflowTask.modelPath || "",
        displayedWorkflowTask.progressPercent ?? "",
        displayedWorkflowTask.container || "",
        displayedWorkflowTask.pid || "",
        ...WORKFLOW_STAGE_NAMES.map(
          (stage) => displayedWorkflowTask.stageStatuses?.[stage] || "",
        ),
      ].join("|")
    : "";
  const assessmentStatusKey = latestAssessmentTask
    ? [
        latestAssessmentTask.container || "",
        latestAssessmentTask.pid || "",
        latestAssessmentTask.assessmentTypeText ||
          latestAssessmentTask.assessmentType ||
          latestAssessmentTask.evalTypeText ||
          latestAssessmentTask.evalType ||
          "",
        latestAssessmentTask.script || "",
      ].join("|")
    : "";
  const benchmarkStatusKey = displayedBenchmarkTask
    ? [
        displayedBenchmarkTask.jobId || "",
        displayedBenchmarkTask.pid || "",
        displayedBenchmarkTask.model || "",
        displayedBenchmarkTask.dataset || "",
        displayedBenchmarkTask.status || "",
        displayedBenchmarkTask.resultPath || "",
      ].join("|")
    : "";
  const inferenceStatusKey = latestInferenceTask
    ? [
        latestInferenceTask.sourceKey || "",
        latestInferenceTask.hasConfig ? "config" : "",
        latestInferenceTask.hasStatus ? "status" : "",
        latestInferenceTask.modelName || "",
        latestInferenceTask.inferencePort || "",
        latestInferenceTask.vllmPort || "",
      ].join("|")
    : "";
  const dataFilterStatusKey = displayedDataFilterTask
    ? [
        displayedDataFilterTask.sourceKey || "",
        displayedDataFilterTask.container || "",
        displayedDataFilterTask.outputFolder || "",
        displayedDataFilterTask.inputFolder || "",
        displayedDataFilterTask.status || "",
        displayedDataFilterTask.outputDatasetName || "",
      ].join("|")
    : "";
  const isStatusCommandDisabled = !currentInputRequest || isReplying;
  const canControlLatestTrainingTask = !!(
    latestTrainingTask &&
    (latestTrainingTask.container || latestTrainingTask.pid) &&
    latestTrainingTask.status !== "failed" &&
    latestTrainingTask.status !== "error"
  );
  const previousStatusKeysRef = useRef({
    workflow: "",
    train: "",
    evaluation: "",
    benchmark: "",
    inference: "",
    data_filter: "",
  });

  useEffect(() => {
    const previous = previousStatusKeysRef.current.workflow;
    previousStatusKeysRef.current.workflow = workflowStatusKey;
    if (workflowStatusKey && workflowStatusKey !== previous) {
      setHiddenStatusBars((prev) =>
        prev.workflow ? { ...prev, workflow: false } : prev,
      );
      setClosedStatusBars((prev) =>
        prev.workflow ? { ...prev, workflow: false } : prev,
      );
    }
  }, [workflowStatusKey]);

  useEffect(() => {
    const previous = previousStatusKeysRef.current.train;
    previousStatusKeysRef.current.train = trainingStatusKey;
    if (trainingStatusKey && trainingStatusKey !== previous) {
      setHiddenStatusBars((prev) =>
        prev.train ? { ...prev, train: false } : prev,
      );
      setClosedStatusBars((prev) =>
        prev.train ? { ...prev, train: false } : prev,
      );
    }
  }, [trainingStatusKey]);

  useEffect(() => {
    const previous = previousStatusKeysRef.current.evaluation;
    previousStatusKeysRef.current.evaluation = assessmentStatusKey;
    if (assessmentStatusKey && assessmentStatusKey !== previous) {
      setHiddenStatusBars((prev) =>
        prev.evaluation ? { ...prev, evaluation: false } : prev,
      );
      setClosedStatusBars((prev) =>
        prev.evaluation ? { ...prev, evaluation: false } : prev,
      );
    }
  }, [assessmentStatusKey]);

  useEffect(() => {
    const previous = previousStatusKeysRef.current.benchmark;
    previousStatusKeysRef.current.benchmark = benchmarkStatusKey;
    if (benchmarkStatusKey && benchmarkStatusKey !== previous) {
      setHiddenStatusBars((prev) =>
        prev.benchmark ? { ...prev, benchmark: false } : prev,
      );
      setClosedStatusBars((prev) =>
        prev.benchmark ? { ...prev, benchmark: false } : prev,
      );
    }
  }, [benchmarkStatusKey]);

  useEffect(() => {
    const previous = previousStatusKeysRef.current.inference;
    previousStatusKeysRef.current.inference = inferenceStatusKey;
    if (inferenceStatusKey && inferenceStatusKey !== previous) {
      setHiddenStatusBars((prev) =>
        prev.inference ? { ...prev, inference: false } : prev,
      );
      setClosedStatusBars((prev) =>
        prev.inference ? { ...prev, inference: false } : prev,
      );
    }
  }, [inferenceStatusKey]);

  useEffect(() => {
    const previous = previousStatusKeysRef.current.data_filter;
    previousStatusKeysRef.current.data_filter = dataFilterStatusKey;
    if (dataFilterStatusKey && dataFilterStatusKey !== previous) {
      setHiddenStatusBars((prev) =>
        prev.data_filter ? { ...prev, data_filter: false } : prev,
      );
      setClosedStatusBars((prev) =>
        prev.data_filter ? { ...prev, data_filter: false } : prev,
      );
    }
  }, [dataFilterStatusKey]);

  const hideStatusBar = useCallback((kind: StatusBarKind) => {
    setHiddenStatusBars((prev) => ({ ...prev, [kind]: true }));
  }, []);

  const expandStatusBar = useCallback((kind: StatusBarKind) => {
    setHiddenStatusBars((prev) => ({ ...prev, [kind]: false }));
  }, []);

  const closeStatusBar = useCallback((kind: StatusBarKind) => {
    setClosedStatusBars((prev) => ({ ...prev, [kind]: true }));
  }, []);

  const openInferencePanel = useCallback(
    (view: InferencePanelView) => {
      setInferencePanelView(view);
      setIsInferenceSheetOpen(true);
    },
    [setInferencePanelView, setIsInferenceSheetOpen],
  );

  const fillTrainingTaskCommand = useCallback(
    (action: "monitor" | "stop") => {
      if (!latestTrainingTask) {
        return;
      }

      const target = [
        latestTrainingTask.container
          ? `容器=${latestTrainingTask.container}`
          : "",
        latestTrainingTask.pid ? `PID=${latestTrainingTask.pid}` : "",
      ]
        .filter(Boolean)
        .join("，");

      handleAgentWaitingReply(
        action === "monitor"
          ? `监控训练状态${target ? `，${target}` : ""}`
          : `停止训练${target ? `，${target}` : ""}`,
      );
    },
    [handleAgentWaitingReply, latestTrainingTask],
  );

  const fillWorkflowTaskCommand = useCallback(
    (action: "monitor" | "stop") => {
      if (!displayedWorkflowTask) {
        return;
      }

      const workflowTarget = displayedWorkflowTask.workflowId
        ? ` ${displayedWorkflowTask.workflowId}`
        : "";
      handleAgentWaitingReply(
        action === "monitor"
          ? `查看一键工作流状态${workflowTarget}`
          : `停止当前一键工作流${workflowTarget}`,
      );
    },
    [displayedWorkflowTask, handleAgentWaitingReply],
  );

  useEffect(() => {
    if (!onMonitorTrainingCommandChange) {
      return;
    }

    if (!latestTrainingTask) {
      onMonitorTrainingCommandChange(null);
      return;
    }

    onMonitorTrainingCommandChange(() => fillTrainingTaskCommand("monitor"));
    return () => {
      onMonitorTrainingCommandChange(null);
    };
  }, [
    fillTrainingTaskCommand,
    latestTrainingTask,
    onMonitorTrainingCommandChange,
  ]);

  useEffect(() => {
    if (!isMetricsSheetOpen || !metricsMonitorTarget) {
      onSilentMonitorStatusChange?.({
        isQuerying: false,
        hasMetrics: false,
        hasNewData: false,
        message: metricsMonitorTarget ? undefined : "暂无训练任务",
      });
      return;
    }

    const monitorTarget = metricsMonitorTarget;
    const silentMonitorIntervalMs = 30_000;
    let disposed = false;
    let inFlight = false;
    let failedAttempts = 0;
    let lastMetricSignature = "";

    const fetchMetrics = async () => {
      if (disposed || inFlight) {
        return;
      }
      inFlight = true;
      const queryStartedAt = new Date().toISOString();
      onSilentMonitorStatusChange?.({
        isQuerying: true,
        lastQueryAt: queryStartedAt,
        message: "正在查询训练指标...",
      });
      try {
        const requestPayload = {
          runId,
          workflowId: monitorTarget.workflowId,
          nodeId: conversationResourceNodeId,
          container: monitorTarget.container,
          pid: monitorTarget.pid,
          trainType: monitorTarget.trainType,
          launchMode: monitorTarget.launchMode,
          isMultinode: monitorTarget.isMultinode,
          scriptName: monitorTarget.scriptName,
          historyLimit: 1000,
          timeWindowMinutes: 1440,
        };
        const result =
          await queryTrainingMetricsMutation.mutateAsync(requestPayload);

        if (disposed) {
          return;
        }

        if (!result.success || !result.data) {
          console.warn("Silent monitor query failed", {
            container: monitorTarget.container,
            pid: monitorTarget.pid,
            trainType: monitorTarget.trainType,
            message: result.message,
          });
          onSilentMonitorStatusChange?.({
            isQuerying: false,
            lastQueryAt: queryStartedAt,
            lastResultAt: new Date().toISOString(),
            hasMetrics: false,
            hasNewData: false,
            message: result.message || "查询失败",
          });
          failedAttempts += 1;
          if (failedAttempts >= 3) {
            disposed = true;
          }
          return;
        }

        failedAttempts = 0;
        const metrics =
          (result.data as { metrics?: Record<string, unknown> }).metrics || {};
        const debug =
          (metrics.debug as Record<string, unknown> | undefined) || {};
        const history = Array.isArray(metrics.history) ? metrics.history : [];
        const toFiniteMetricNumber = (value: unknown): number | undefined => {
          if (typeof value === "number") {
            return Number.isFinite(value) ? value : undefined;
          }
          if (typeof value === "string" && value.trim() !== "") {
            const parsed = Number(value);
            return Number.isFinite(parsed) ? parsed : undefined;
          }
          return undefined;
        };
        const hasLatestMetric =
          toFiniteMetricNumber(metrics.latest_step) !== undefined &&
          toFiniteMetricNumber(metrics.latest_loss) !== undefined;
        const hasHistoryMetric = history.some((item) => {
          if (!item || typeof item !== "object") {
            return false;
          }
          const record = item as Record<string, unknown>;
          return (
            toFiniteMetricNumber(record.step ?? record._step) !== undefined &&
            toFiniteMetricNumber(record.loss) !== undefined
          );
        });
        const metricSignature = [
          metrics.pid,
          metrics.latest_step,
          metrics.latest_loss,
          metrics.history_count,
          metrics.last_update_time,
        ]
          .map((value) => String(value ?? ""))
          .join("|");
        const hasMetrics = hasLatestMetric || hasHistoryMetric;
        const trainingStartedWithoutMetrics =
          !hasMetrics &&
          (metrics.training_process_exists === true ||
            metrics.pid_alive === true ||
            ["starting", "running"].includes(
              String((result.data as { status?: unknown }).status || "").toLowerCase(),
            ));
        const hasNewData =
          hasMetrics && metricSignature !== lastMetricSignature;
        if (hasNewData) {
          lastMetricSignature = metricSignature;
        }
        const resultAt = new Date().toISOString();
        onSilentMonitorStatusChange?.({
          isQuerying: false,
          lastQueryAt: queryStartedAt,
          lastResultAt: resultAt,
          lastDataAt: hasNewData ? resultAt : undefined,
          hasMetrics,
          hasNewData,
          message: hasMetrics
            ? hasNewData
              ? "已获取训练指标"
              : "暂无新指标"
            : trainingStartedWithoutMetrics
              ? "训练已启动，等待指标写入"
              : "尚未获取到训练指标",
        });
        const timestamp = new Date().toISOString();
        const reply: Reply = {
          replyId: `silent-monitor-${timestamp}`,
          replyName: "System",
          replyRole: "assistant",
          createdAt: timestamp,
          finishedAt: timestamp,
          messages: [
            {
              id: `silent-monitor-message-${timestamp}`,
              name: "silent-monitor",
              role: "assistant",
              content: JSON.stringify(result.data),
              timestamp,
              metadata: {},
            },
          ],
        };
        onSilentMonitorReplyChange?.(reply);
      } finally {
        inFlight = false;
      }
    };

    void fetchMetrics();
    const intervalId = window.setInterval(() => {
      void fetchMetrics();
    }, silentMonitorIntervalMs);

    return () => {
      disposed = true;
      window.clearInterval(intervalId);
    };
  }, [
    isMetricsSheetOpen,
    metricsMonitorTarget,
    runId,
    conversationResourceNodeId,
    onSilentMonitorReplyChange,
    onSilentMonitorStatusChange,
    queryTrainingMetricsMutation.mutateAsync,
  ]);

  const fillAssessmentTaskCommand = useCallback(
    (action: "monitor" | "stop") => {
      if (!latestAssessmentTask) {
        return;
      }

      const target = [
        latestAssessmentTask.container
          ? `容器=${latestAssessmentTask.container}`
          : "",
        latestAssessmentTask.pid ? `PID=${latestAssessmentTask.pid}` : "",
      ]
        .filter(Boolean)
        .join("，");

      handleAgentWaitingReply(
        action === "monitor"
          ? `监控评估状态${target ? `，${target.replace("PID=", "pid=")}` : ""}`
          : `停止评估${target ? `，${target}` : ""}`,
      );
    },
    [handleAgentWaitingReply, latestAssessmentTask],
  );

  const fillBenchmarkTaskCommand = useCallback(
    (action: "status" | "stop") => {
      if (!displayedBenchmarkTask) {
        return;
      }

      if (action === "stop") {
        setBenchmarkTaskSnapshot(displayedBenchmarkTask);
        setPendingBenchmarkStop(true);
      }

      const target = [
        displayedBenchmarkTask.jobId
          ? `任务ID=${displayedBenchmarkTask.jobId}`
          : "",
        displayedBenchmarkTask.dataset
          ? `数据集=${displayedBenchmarkTask.dataset}`
          : "",
        displayedBenchmarkTask.pid ? `PID=${displayedBenchmarkTask.pid}` : "",
      ]
        .filter(Boolean)
        .join("，");

      if (action === "status") {
        const datasetStatusCommand =
          !displayedBenchmarkTask.jobId && displayedBenchmarkTask.dataset
            ? `查看推理基准测试${displayedBenchmarkTask.dataset}状态`
            : "查看推理基准测试状态";
        handleAgentWaitingReply(
          `${datasetStatusCommand}${target ? `，${target}` : ""}`,
        );
        return;
      }

      handleAgentWaitingReply(
        `停止推理基准测试${target ? `，${target}` : ""}`,
      );
    },
    [displayedBenchmarkTask, handleAgentWaitingReply],
  );

  const placeholder = useMemo(() => {
    if (currentInputRequest) {
      return t("placeholder.input-as-user", {
        name: currentInputRequest.agentName,
      });
    }

    if (runs.length === 0) {
      return t("placeholder.input-no-runs", {
        defaultValue: "暂无运行实例，请先创建或选择一个运行",
      });
    }

    if (isReplying || runData?.status === Status.RUNNING) {
      return t("placeholder.input-model-working", {
        defaultValue: "后台模型正在处理，暂时不需要用户输入",
      });
    }

    if (runData?.status === Status.DONE) {
      return t("placeholder.input-run-finished", {
        defaultValue: "当前运行已结束，暂无输入请求",
      });
    }

    return t("placeholder.input-disable");
  }, [currentInputRequest, isReplying, runData?.status, runs.length, t]);

  const disabledInputHint = currentInputRequest ? "" : placeholder;

  const shortcutKeys = isMacOs ? "Command + Enter" : "fCtrl + Enter";
  const nextStepRecommendation = useMemo(() => {
    if (displayedWorkflowTask?.workflowStatus?.toLowerCase() === "running") {
      return {
        message:
          t("runpage.toolbar.nextStep.workflowRunning") ||
          "当前一键工作流正在运行，建议先查看工作流状态或监控进度。",
        actionLabel:
          t("runpage.toolbar.nextStep.action.viewWorkflow") || "查看状态",
        action: () => fillWorkflowTaskCommand("monitor"),
      };
    }

    const normalizedTrainingStatus = latestTrainingTask?.status
      ?.trim()
      .toLowerCase();
    const inactiveTrainingStatuses = new Set([
      "stopped",
      "stop",
      "finished",
      "completed",
      "done",
      "failed",
      "error",
      "cancelled",
      "canceled",
      "timeout",
    ]);
    if (
      latestTrainingTask &&
      (!normalizedTrainingStatus ||
        !inactiveTrainingStatuses.has(normalizedTrainingStatus))
    ) {
      return {
        message:
          t("runpage.toolbar.nextStep.trainingRunning") ||
          "训练任务正在运行，建议打开监控查看 loss、学习率和 GPU 使用情况。",
        actionLabel:
          t("runpage.toolbar.nextStep.action.openMonitor") || "打开监控",
        action: () => setIsMetricsSheetOpen(true),
      };
    }

    if (displayedDataFilterTask?.status?.toLowerCase() === "running") {
      return {
        message:
          t("runpage.toolbar.nextStep.dataFilterRunning") ||
          "数据筛选正在运行，建议等待处理完成或在状态卡片中刷新进度。",
      };
    }

    if (latestInferenceTask) {
      return {
        message:
          t("runpage.toolbar.nextStep.inferenceReady") ||
          "推理服务已有配置或状态信息，可查看服务状态后继续操作。",
        actionLabel:
          t("runpage.toolbar.nextStep.action.viewInference") || "查看服务",
        action: () =>
          openInferencePanel(latestInferenceTask.hasStatus ? "status" : "config"),
      };
    }

    if (!hasQueriedDatasets || datasets.length === 0) {
      return {
        message:
          t("runpage.toolbar.nextStep.queryDataset") ||
          "下一步推荐：先查询数据集",
        actionLabel:
          t("runpage.toolbar.nextStep.action.queryDataset") || "去查询",
        action: () => onTabChange("datasets"),
      };
    }

    if (!hasQueriedModels || models.length === 0) {
      return {
        message:
          t("runpage.toolbar.nextStep.queryModel") || "下一步推荐：先查询模型",
        actionLabel:
          t("runpage.toolbar.nextStep.action.queryModel") || "去查询",
        action: () => onTabChange("models"),
      };
    }

    if (!hasQueriedTests) {
      return {
        message:
          t("runpage.toolbar.nextStep.queryEvaluation") ||
          "下一步推荐：先查询评测",
        actionLabel:
          t("runpage.toolbar.nextStep.action.queryEvaluation") || "去查询",
        action: () => onTabChange("evaluation"),
      };
    }

    return {
      message:
        t("runpage.toolbar.nextStep.readyToTrain") ||
        "下一步推荐：开始训练。不清楚如何训练？试试输入框上方的常用命令",
    };
  }, [
    t,
    displayedWorkflowTask?.workflowStatus,
    fillWorkflowTaskCommand,
    latestTrainingTask,
    setIsMetricsSheetOpen,
    displayedDataFilterTask?.status,
    latestInferenceTask,
    openInferencePanel,
    onTabChange,
    hasQueriedDatasets,
    datasets.length,
    hasQueriedModels,
    models.length,
    hasQueriedTests,
  ]);
  return (
    <div className="animate-slide-in-right h-full">
      <Flex
        style={{
          minHeight: 0,
          height: "100%",
        }}
        flex={1}
        vertical={true}
      >
        <div className="border-b border-border/15 bg-[linear-gradient(180deg,rgba(255,255,255,0.92)_0%,rgba(248,250,252,0.88)_100%)] px-4 py-3 sm:px-5 sm:py-3.5 shrink-0 z-10 dark:border-white/8 dark:bg-[linear-gradient(180deg,rgba(15,23,42,0.96)_0%,rgba(17,24,39,0.92)_100%)]">
          <div className="flex flex-col gap-3 rounded-[22px] border border-white/75 bg-white/72 px-4 py-3 shadow-[0_18px_42px_-34px_rgba(15,23,42,0.3)] sm:flex-row sm:items-center sm:justify-between sm:px-5 dark:border-white/10 dark:bg-slate-900/72 dark:shadow-[0_18px_42px_-34px_rgba(2,6,23,0.82)]">
            <div className="min-w-0">
              <div className="flex flex-wrap items-center gap-2">
                <div className="text-[11px] leading-5 font-normal text-muted-foreground/80 dark:text-slate-400">
                  {nextStepRecommendation.message}
                </div>
                {nextStepRecommendation.actionLabel &&
                  nextStepRecommendation.action && (
                    <Button
                      type="link"
                      size="small"
                      className="h-auto px-0 text-[11px] font-medium"
                      onClick={nextStepRecommendation.action}
                    >
                      <span className="inline-flex items-center gap-1">
                        {nextStepRecommendation.actionLabel}
                        <ArrowRight className="h-3.5 w-3.5" />
                      </span>
                    </Button>
                  )}
              </div>
            </div>

            <div className="flex w-full items-start gap-2 sm:w-auto sm:items-center sm:justify-end">
              <div className="flex min-w-0 flex-1 flex-wrap items-center gap-2 sm:flex-none sm:justify-end">
                <div className="quick-start-wrapper inline-flex min-w-[132px] flex-1 sm:min-w-0 sm:flex-none">
                  <Dropdown
                    menu={{ items: quickStartMenuItems }}
                    trigger={["click"]}
                    placement="bottomLeft"
                    overlayClassName="runpage-quickstart-dropdown"
                  >
                    <Button
                      type="primary"
                      size="small"
                      className="flex h-9 w-full items-center justify-center rounded-lg px-3.5 text-sm font-medium shadow-[0_12px_24px_-18px_rgba(37,99,235,0.72)] transition-all sm:w-auto"
                      style={{ background: "var(--primary)" }}
                      title={t("quickStart.button") || "新手引导"}
                    >
                      <span className="truncate">
                        {t("quickStart.button") || "新手引导"}
                      </span>
                      <ChevronDown className="ml-1 h-3.5 w-3.5 shrink-0" />
                    </Button>
                  </Dropdown>
                </div>

                <div className="one-click-workflow-wrapper inline-flex min-w-[156px] flex-1 sm:min-w-0 sm:flex-none">
                  <Button
                    size="small"
                    className="flex h-9 w-full items-center justify-center rounded-lg border border-slate-200/80 bg-slate-50/88 px-3.5 text-sm font-medium text-slate-800 shadow-none transition-all hover:border-slate-300 hover:bg-white sm:w-auto dark:border-white/10 dark:bg-slate-800/82 dark:text-slate-100 dark:hover:bg-slate-700/78"
                    onClick={openWorkflowConfig}
                    title={t("oneClickWorkflow.button") || "一键工作流"}
                  >
                    <span className="inline-flex min-w-0 items-center gap-1.5">
                      <Workflow className="h-3.5 w-3.5 shrink-0" />
                      <span className="truncate">
                        {t("oneClickWorkflow.button") || "一键工作流"}
                      </span>
                    </span>
                  </Button>
                </div>

              </div>

              <Dropdown
                menu={{ items: exportMenuItems }}
                placement="bottomRight"
                overlayClassName="runpage-more-dropdown"
              >
                <Button
                  size="small"
                  aria-label={t("common.more-actions") || "更多操作"}
                  className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border border-slate-200/70 bg-slate-50/70 p-0 text-slate-500 shadow-none transition-all hover:bg-white dark:border-white/10 dark:bg-slate-800/78 dark:text-slate-300 dark:hover:bg-slate-700/78"
                >
                  <MoreHorizontal className="h-3.5 w-3.5" />
                </Button>
              </Dropdown>
            </div>
          </div>
        </div>

        <div className="relative flex flex-col w-full h-full overflow-hidden bg-[radial-gradient(circle_at_top,rgba(191,219,254,0.1),transparent_24%),linear-gradient(180deg,rgba(248,250,252,0.76)_0%,rgba(241,245,249,0.94)_100%)] dark:bg-[radial-gradient(circle_at_top,rgba(56,189,248,0.08),transparent_22%),linear-gradient(180deg,rgba(2,6,23,0.94)_0%,rgba(15,23,42,0.98)_100%)]">
          <div className="h-full flex justify-center w-full px-4 py-4 sm:px-6 sm:py-5 lg:px-8 lg:py-6">
            <div className="flex h-full w-full max-w-6xl flex-col overflow-hidden rounded-[28px] border border-white/75 bg-[linear-gradient(180deg,rgba(255,255,255,0.9)_0%,rgba(248,250,252,0.98)_100%)] shadow-[0_24px_60px_-36px_rgba(15,23,42,0.26)] dark:border-white/8 dark:bg-[linear-gradient(180deg,rgba(15,23,42,0.94)_0%,rgba(17,24,39,0.98)_100%)] dark:shadow-[0_24px_60px_-36px_rgba(2,6,23,0.86)]">
              {visibleWorkflowTask &&
                !closedStatusBars.workflow &&
                !hiddenStatusBars.workflow && (
                  <div className="relative shrink-0 overflow-hidden border-b border-violet-100/80 bg-[linear-gradient(90deg,rgba(252,250,255,0.98)_0%,rgba(247,243,255,0.76)_48%,rgba(255,255,255,0.96)_100%)] px-4 py-3 shadow-[0_8px_22px_-24px_rgba(124,58,237,0.24)] backdrop-blur-sm dark:border-violet-400/12 dark:bg-[linear-gradient(90deg,rgba(76,29,149,0.22)_0%,rgba(109,40,217,0.14)_52%,rgba(15,23,42,0.62)_100%)]">
                    <div className="absolute inset-y-0 left-0 w-1 bg-violet-300 dark:bg-violet-300/80" />
                    <button
                      type="button"
                      aria-label={t("runpage.statusBar.collapseWorkflow")}
                      className="absolute right-2 top-2 inline-flex h-6 w-6 items-center justify-center rounded-full text-violet-600/70 transition hover:bg-white/70 hover:text-violet-900 dark:text-violet-100/60 dark:hover:bg-violet-950/38 dark:hover:text-violet-50"
                      onClick={() => hideStatusBar("workflow")}
                    >
                      <ChevronUp className="h-3.5 w-3.5" />
                    </button>
                    <button
                      type="button"
                      aria-label="Close workflow status bar"
                      className="absolute right-9 top-2 inline-flex h-6 w-6 items-center justify-center rounded-full text-violet-600/70 transition hover:bg-white/70 hover:text-violet-900 dark:text-violet-100/60 dark:hover:bg-violet-950/38 dark:hover:text-violet-50"
                      onClick={() => closeStatusBar("workflow")}
                    >
                      <X className="h-3.5 w-3.5" />
                    </button>
                    <div className="flex flex-col gap-3 pl-2 xl:flex-row xl:items-start xl:justify-between">
                      <div className="min-w-0 flex-1">
                        <div className="flex flex-wrap items-center gap-2 text-[15px] font-semibold text-violet-950 dark:text-violet-50">
                          <span className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-violet-500/90 text-white shadow-[0_8px_18px_-15px_rgba(124,58,237,0.58)] dark:bg-violet-400/90 dark:text-violet-950">
                            <Zap className="h-4 w-4" />
                          </span>
                          {t("runpage.statusBar.workflowTitle")}
                          <span className="inline-flex items-center gap-1.5 rounded-full border border-emerald-200/80 bg-emerald-50/88 px-2 py-0.5 text-[10px] font-semibold text-emerald-700 dark:border-emerald-400/20 dark:bg-emerald-950/30 dark:text-emerald-200">
                            <span
                              className={`h-1.5 w-1.5 rounded-full bg-emerald-500 ${workflowMonitorState.isPolling ? "animate-pulse" : ""}`}
                            />
                            {workflowMonitorState.isPolling
                              ? t("runpage.statusBar.workflowRefreshing")
                              : t("runpage.statusBar.workflowAutoRefresh")}
                            {!workflowMonitorState.isPolling &&
                              workflowRefreshCountdown !== undefined &&
                              ` · ${workflowRefreshCountdown}s`}
                          </span>
                          {workflowLastUpdatedText && (
                            <span className="text-[10px] font-medium text-violet-500/80 dark:text-violet-200/65">
                              {t("runpage.statusBar.workflowLastUpdated")}
                              {workflowLastUpdatedText}
                            </span>
                          )}
                        </div>
                        <div className="mt-1 flex flex-wrap items-center gap-1.5 pl-10 text-[11px] font-medium text-violet-800 dark:text-violet-100">
                          {displayedWorkflowTask.currentStage && (
                            <span className="rounded-md bg-white/42 px-2 py-0.5 ring-1 ring-violet-100/70 dark:bg-violet-950/20 dark:ring-violet-300/12">
                              {t("runpage.statusBar.currentStage")}
                              {t("runpage.statusBar.labelSeparator")}
                              {t(
                                `runpage.statusBar.workflowStages.${displayedWorkflowTask.currentStage}`,
                                {
                                  defaultValue:
                                    displayedWorkflowTask.currentStage,
                                },
                              )}
                              {displayedWorkflowTask.currentStageStatus &&
                                ` / ${t(
                                  `runpage.statusBar.workflowStatuses.${displayedWorkflowTask.currentStageStatus}`,
                                  {
                                    defaultValue:
                                      displayedWorkflowTask.currentStageStatus,
                                  },
                                )}`}
                            </span>
                          )}
                          {displayedWorkflowTask.progressPercent !==
                            undefined && (
                            <span className="rounded-md bg-white/42 px-2 py-0.5 ring-1 ring-violet-100/70 dark:bg-violet-950/20 dark:ring-violet-300/12">
                              {t("runpage.statusBar.progress")}
                              {t("runpage.statusBar.labelSeparator")}
                              {displayedWorkflowTask.progressPercent}%
                            </span>
                          )}
                          {displayedWorkflowTask.datasetRef && (
                            <span className="rounded-md bg-white/42 px-2 py-0.5 ring-1 ring-violet-100/70 dark:bg-violet-950/20 dark:ring-violet-300/12">
                              {t("runpage.statusBar.datasetRef")}
                              {t("runpage.statusBar.labelSeparator")}
                              {displayedWorkflowTask.datasetRef}
                            </span>
                          )}
                          {displayedWorkflowTask.evaluationDatasetName && (
                            <span className="rounded-md bg-white/42 px-2 py-0.5 ring-1 ring-violet-100/70 dark:bg-violet-950/20 dark:ring-violet-300/12">
                              {t("runpage.statusBar.evaluationDataset")}
                              {t("runpage.statusBar.labelSeparator")}
                              {displayedWorkflowTask.evaluationDatasetName}
                            </span>
                          )}
                          <button
                            type="button"
                            className="inline-flex items-center gap-0.5 px-1 text-[11px] font-semibold text-violet-600 transition hover:text-violet-900 dark:text-violet-200 dark:hover:text-violet-50"
                            onClick={() =>
                              setIsWorkflowDetailsOpen((open) => !open)
                            }
                          >
                            {isWorkflowDetailsOpen
                              ? t("runpage.statusBar.hideDetails")
                              : t("runpage.statusBar.showDetails")}
                            {isWorkflowDetailsOpen ? (
                              <ChevronUp className="h-3 w-3" />
                            ) : (
                              <ChevronDown className="h-3 w-3" />
                            )}
                          </button>
                        </div>
                        {isWorkflowDetailsOpen && (
                          <div className="mt-2 flex flex-wrap gap-1.5 pl-10 text-[11px] font-medium text-violet-700/90 dark:text-violet-200/90">
                            {displayedWorkflowTask.workflowId && (
                              <span className="rounded-md bg-white/36 px-2 py-0.5 ring-1 ring-violet-100/60 dark:bg-violet-950/18 dark:ring-violet-300/10">
                                {t("runpage.statusBar.workflowId")}
                                {t("runpage.statusBar.labelSeparator")}
                                {displayedWorkflowTask.workflowId}
                              </span>
                            )}
                            {displayedWorkflowTask.container && (
                              <span className="rounded-md bg-white/36 px-2 py-0.5 ring-1 ring-violet-100/60 dark:bg-violet-950/18 dark:ring-violet-300/10">
                                {t("runpage.statusBar.container")}
                                {t("runpage.statusBar.labelSeparator")}
                                {displayedWorkflowTask.container}
                              </span>
                            )}
                            {displayedWorkflowTask.pid && (
                              <span className="rounded-md bg-white/36 px-2 py-0.5 ring-1 ring-violet-100/60 dark:bg-violet-950/18 dark:ring-violet-300/10">
                                {t("runpage.statusBar.pid")}
                                {t("runpage.statusBar.labelSeparator")}
                                {displayedWorkflowTask.pid}
                              </span>
                            )}
                            {Object.values(
                              displayedWorkflowTask.workflowLogs || {},
                            ).map((log) => {
                              const stageLabel = t(
                                `runpage.statusBar.workflowStages.${log.stage}`,
                                { defaultValue: log.stage },
                              );
                              const hasStopLog =
                                Boolean(log.stopServiceLogPath) ||
                                Boolean(log.stopServiceLogTail);
                              return (
                                <button
                                  key={log.stage}
                                  type="button"
                                  className="inline-flex items-center gap-1 rounded-md bg-white/48 px-2 py-0.5 text-[11px] font-semibold text-violet-700 ring-1 ring-violet-100/70 transition hover:bg-white hover:text-violet-950 dark:bg-violet-950/24 dark:text-violet-100 dark:ring-violet-300/12 dark:hover:bg-violet-900/42"
                                  onClick={() =>
                                    Modal.info({
                                      title: `${stageLabel}日志`,
                                      width: 760,
                                      content: (
                                        <div className="space-y-3 text-xs">
                                          {log.logPath && (
                                            <div>
                                              <div className="mb-1 font-semibold text-slate-600">
                                                日志路径
                                              </div>
                                              <code className="block max-h-24 overflow-auto rounded bg-slate-100 p-2 text-[11px] text-slate-800">
                                                {log.logPath}
                                              </code>
                                            </div>
                                          )}
                                          {log.logTail && (
                                            <pre className="max-h-72 overflow-auto whitespace-pre-wrap rounded bg-slate-950 p-3 text-[11px] leading-relaxed text-slate-100">
                                              {log.logTail}
                                            </pre>
                                          )}
                                          {hasStopLog && (
                                            <div>
                                              <div className="mb-1 font-semibold text-slate-600">
                                                关闭服务日志
                                              </div>
                                              {log.stopServiceLogPath && (
                                                <code className="mb-2 block max-h-24 overflow-auto rounded bg-slate-100 p-2 text-[11px] text-slate-800">
                                                  {log.stopServiceLogPath}
                                                </code>
                                              )}
                                              {log.stopServiceLogTail && (
                                                <pre className="max-h-72 overflow-auto whitespace-pre-wrap rounded bg-slate-950 p-3 text-[11px] leading-relaxed text-slate-100">
                                                  {log.stopServiceLogTail}
                                                </pre>
                                              )}
                                            </div>
                                          )}
                                          {!log.logTail &&
                                            !log.stopServiceLogTail && (
                                              <div className="text-slate-500">
                                                暂无日志内容，稍后刷新状态后再查看。
                                              </div>
                                            )}
                                        </div>
                                      ),
                                    })
                                  }
                                >
                                  <FileTextOutlined />
                                  {stageLabel}日志
                                </button>
                              );
                            })}
                          </div>
                        )}
                        <div className="mt-3 flex max-w-full flex-wrap items-center gap-1 pl-10 pr-2 text-[11px] font-medium">
                          {WORKFLOW_STAGE_NAMES.map((stage, index) => {
                            const status =
                              displayedWorkflowTask.stageStatuses?.[stage] ||
                              "pending";
                            const tone =
                              status === "finished"
                                ? "border-emerald-200 bg-emerald-50/86 text-emerald-700 dark:border-emerald-400/20 dark:bg-emerald-950/30 dark:text-emerald-200"
                                : status === "failed"
                                  ? "border-red-200 bg-red-50/86 text-red-700 dark:border-red-400/20 dark:bg-red-950/30 dark:text-red-200"
                                  : status === "timeout"
                                    ? "border-amber-200 bg-amber-50/90 text-amber-800 dark:border-amber-300/25 dark:bg-amber-950/32 dark:text-amber-100"
                                  : status === "running" ||
                                      status.startsWith("starting")
                                    ? "border-violet-200 bg-violet-100/86 text-violet-800 dark:border-violet-300/20 dark:bg-violet-900/38 dark:text-violet-100"
                                    : "border-slate-200 bg-white/58 text-slate-500 dark:border-white/10 dark:bg-slate-900/24 dark:text-slate-400";
                            return (
                              <span
                                key={stage}
                                className="inline-flex shrink-0 items-center gap-1"
                              >
                                <span
                                  className={`inline-flex h-7 items-center whitespace-nowrap rounded-full border px-2.5 py-0 ${tone}`}
                                >
                                  {t(
                                    `runpage.statusBar.workflowStages.${stage}`,
                                  )}
                                  {t("runpage.statusBar.labelSeparator")}
                                  {t(
                                    `runpage.statusBar.workflowStatuses.${status}`,
                                    { defaultValue: status },
                                  )}
                                </span>
                                {index < WORKFLOW_STAGE_NAMES.length - 1 && (
                                  <span className="text-violet-300/80 dark:text-violet-400/45">
                                    ›
                                  </span>
                                )}
                              </span>
                            );
                          })}
                        </div>
                        {workflowMonitorState.error && (
                          <div className="mt-2 pl-10 text-[11px] font-medium text-red-600 dark:text-red-300">
                            {t("runpage.statusBar.workflowRefreshFailed")}
                            {`：${workflowMonitorState.error}`}
                          </div>
                        )}
                      </div>
                      <div className="flex shrink-0 flex-nowrap items-center justify-start gap-1.5 pl-10 sm:justify-end sm:pl-0 sm:pr-12">
                        {workflowTrainingTask && (
                          <Button
                            size="small"
                            className="h-7 shrink-0 rounded-md border border-violet-200 bg-white/58 px-2 text-[11px] font-semibold text-violet-950 shadow-none hover:border-violet-300 hover:bg-white dark:border-violet-300/18 dark:bg-violet-950/34 dark:text-violet-50 dark:hover:bg-violet-900/46"
                            onClick={() => setIsMetricsSheetOpen(true)}
                          >
                            <span className="inline-flex items-center gap-1">
                              <LineChart className="h-3 w-3" />
                              {t("runpage.statusBar.openTrainingMonitor")}
                            </span>
                          </Button>
                        )}
                        <Button
                          size="small"
                          className="h-7 shrink-0 rounded-md border border-violet-200 bg-white/58 px-2 text-[11px] font-semibold text-violet-950 shadow-none hover:border-violet-300 hover:bg-white dark:border-violet-300/18 dark:bg-violet-950/34 dark:text-violet-50 dark:hover:bg-violet-900/46"
                          disabled={isStatusCommandDisabled}
                          onClick={() => fillWorkflowTaskCommand("monitor")}
                        >
                          <span className="inline-flex items-center gap-1">
                            <Search className="h-3 w-3" />
                            {t("runpage.statusBar.viewWorkflowStatus")}
                          </span>
                        </Button>
                        {displayedWorkflowTask.workflowStatus === "running" && (
                          <Button
                            danger
                            size="small"
                            className="h-7 shrink-0 rounded-md bg-white/88 px-2 text-[11px] font-semibold shadow-sm"
                            disabled={isStatusCommandDisabled}
                            onClick={() => fillWorkflowTaskCommand("stop")}
                          >
                            <span className="inline-flex items-center gap-1">
                              <Square className="h-3 w-3" />
                              {t("runpage.statusBar.stopWorkflow")}
                            </span>
                          </Button>
                        )}
                      </div>
                    </div>
                  </div>
                )}
              {visibleWorkflowTask &&
                !closedStatusBars.workflow &&
                hiddenStatusBars.workflow && (
                  <div className="relative shrink-0 border-b border-violet-100/80 bg-violet-50/62 px-4 py-2 dark:border-violet-400/12 dark:bg-violet-950/20">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div className="flex min-w-0 flex-wrap items-center gap-2 text-xs font-medium text-violet-900 dark:text-violet-100">
                        <span className="inline-flex items-center gap-1.5 font-semibold">
                          <Zap className="h-3.5 w-3.5" />
                          {t("runpage.statusBar.workflowTitle")}
                        </span>
                        {displayedWorkflowTask.workflowId && (
                          <span className="rounded-md bg-white/70 px-2 py-0.5 ring-1 ring-violet-100/80 dark:bg-violet-950/40 dark:ring-violet-300/12">
                            {t("runpage.statusBar.workflowId")}
                            {t("runpage.statusBar.labelSeparator")}
                            {displayedWorkflowTask.workflowId}
                          </span>
                        )}
                        {displayedWorkflowTask.currentStage && (
                          <span className="rounded-md bg-white/70 px-2 py-0.5 ring-1 ring-violet-100/80 dark:bg-violet-950/40 dark:ring-violet-300/12">
                            {t(
                              `runpage.statusBar.workflowStages.${displayedWorkflowTask.currentStage}`,
                              {
                                defaultValue:
                                  displayedWorkflowTask.currentStage,
                              },
                            )}
                            {displayedWorkflowTask.currentStageStatus &&
                              ` / ${t(
                                `runpage.statusBar.workflowStatuses.${displayedWorkflowTask.currentStageStatus}`,
                                {
                                  defaultValue:
                                    displayedWorkflowTask.currentStageStatus,
                                },
                              )}`}
                          </span>
                        )}
                        {displayedWorkflowTask.progressPercent !==
                          undefined && (
                          <span className="rounded-md bg-white/70 px-2 py-0.5 ring-1 ring-violet-100/80 dark:bg-violet-950/40 dark:ring-violet-300/12">
                            {t("runpage.statusBar.progress")}
                            {t("runpage.statusBar.labelSeparator")}
                            {displayedWorkflowTask.progressPercent}%
                          </span>
                        )}
                        {displayedWorkflowTask.evaluationDatasetName && (
                          <span className="rounded-md bg-white/70 px-2 py-0.5 ring-1 ring-violet-100/80 dark:bg-violet-950/40 dark:ring-violet-300/12">
                            {t("runpage.statusBar.evaluationDataset")}
                            {t("runpage.statusBar.labelSeparator")}
                            {displayedWorkflowTask.evaluationDatasetName}
                          </span>
                        )}
                        <span className="inline-flex items-center gap-1.5 rounded-full border border-emerald-200/80 bg-emerald-50/88 px-2 py-0.5 text-[10px] font-semibold text-emerald-700 dark:border-emerald-400/20 dark:bg-emerald-950/30 dark:text-emerald-200">
                          <span
                            className={`h-1.5 w-1.5 rounded-full bg-emerald-500 ${workflowMonitorState.isPolling ? "animate-pulse" : ""}`}
                          />
                          {workflowMonitorState.isPolling
                            ? t("runpage.statusBar.workflowRefreshing")
                            : t("runpage.statusBar.workflowAutoRefresh")}
                          {!workflowMonitorState.isPolling &&
                            workflowRefreshCountdown !== undefined &&
                            ` · ${workflowRefreshCountdown}s`}
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        <Button
                          size="small"
                          className="h-7 w-7 rounded-lg border border-violet-200 bg-white/70 p-0 text-violet-900 shadow-none hover:bg-white dark:border-violet-300/18 dark:bg-violet-950/34 dark:text-violet-50 dark:hover:bg-violet-900/46"
                          aria-label="Close workflow status bar"
                          onClick={() => closeStatusBar("workflow")}
                        >
                          <X className="h-3.5 w-3.5" />
                        </Button>
                        <Button
                          size="small"
                          className="h-7 rounded-lg border border-violet-200 bg-white/70 px-2.5 text-xs font-semibold text-violet-900 shadow-none hover:bg-white dark:border-violet-300/18 dark:bg-violet-950/34 dark:text-violet-50 dark:hover:bg-violet-900/46"
                          onClick={() => expandStatusBar("workflow")}
                        >
                          <span className="inline-flex items-center gap-1">
                            {t("runpage.statusBar.expand")}
                            <ChevronDown className="h-3.5 w-3.5" />
                          </span>
                        </Button>
                      </div>
                    </div>
                  </div>
                )}
              {!displayedWorkflowTask &&
                latestTrainingTask &&
                !closedStatusBars.train &&
                !hiddenStatusBars.train && (
                  <div className="relative shrink-0 overflow-hidden border-b border-sky-100/80 bg-[linear-gradient(90deg,rgba(248,252,255,0.98)_0%,rgba(241,248,255,0.74)_48%,rgba(255,255,255,0.96)_100%)] px-4 py-3 shadow-[0_8px_22px_-24px_rgba(37,99,235,0.24)] backdrop-blur-sm dark:border-sky-400/12 dark:bg-[linear-gradient(90deg,rgba(12,74,110,0.22)_0%,rgba(30,64,175,0.14)_52%,rgba(15,23,42,0.62)_100%)]">
                    <div className="absolute inset-y-0 left-0 w-1 bg-sky-300 dark:bg-sky-300/80" />
                    <button
                      type="button"
                      aria-label={t("runpage.statusBar.collapseTraining")}
                      className="absolute right-2 top-2 inline-flex h-6 w-6 items-center justify-center rounded-full text-sky-600/70 transition hover:bg-white/70 hover:text-sky-900 dark:text-sky-100/60 dark:hover:bg-sky-950/38 dark:hover:text-sky-50"
                      onClick={() => hideStatusBar("train")}
                    >
                      <ChevronUp className="h-3.5 w-3.5" />
                    </button>
                    <button
                      type="button"
                      aria-label="Close training status bar"
                      className="absolute right-9 top-2 inline-flex h-6 w-6 items-center justify-center rounded-full text-sky-600/70 transition hover:bg-white/70 hover:text-sky-900 dark:text-sky-100/60 dark:hover:bg-sky-950/38 dark:hover:text-sky-50"
                      onClick={() => closeStatusBar("train")}
                    >
                      <X className="h-3.5 w-3.5" />
                    </button>
                    <div className="flex flex-col gap-2 pl-2 sm:flex-row sm:items-center sm:justify-between">
                      <div className="min-w-0">
                        <div className="flex items-center gap-2 text-[15px] font-semibold text-sky-950 dark:text-sky-50">
                          <span className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-sky-500/90 text-white shadow-[0_8px_18px_-15px_rgba(37,99,235,0.58)] dark:bg-sky-400/90 dark:text-sky-950">
                            <Activity className="h-4 w-4" />
                          </span>
                          {t("runpage.statusBar.trainingTitle")}
                        </div>
                        <div className="mt-1 flex flex-wrap gap-1.5 pl-10 text-[11px] font-medium text-sky-800 dark:text-sky-100">
                          {latestTrainingTask.container && (
                            <span className="rounded-md bg-white/42 px-2 py-0.5 ring-1 ring-sky-100/70 dark:bg-sky-950/20 dark:ring-sky-300/12">
                              {t("runpage.statusBar.container")}
                              {t("runpage.statusBar.labelSeparator")}
                              {latestTrainingTask.container}
                            </span>
                          )}
                          {latestTrainingTask.pid && (
                            <span className="rounded-md bg-white/42 px-2 py-0.5 ring-1 ring-sky-100/70 dark:bg-sky-950/20 dark:ring-sky-300/12">
                              {t("runpage.statusBar.pid")}
                              {t("runpage.statusBar.labelSeparator")}
                              {latestTrainingTask.pid}
                            </span>
                          )}
                          {latestTrainingTask.status && (
                            <span className="rounded-md bg-white/42 px-2 py-0.5 ring-1 ring-sky-100/70 dark:bg-sky-950/20 dark:ring-sky-300/12">
                              {t("runpage.statusBar.status")}
                              {t("runpage.statusBar.labelSeparator")}
                              {t(
                                `runpage.statusBar.trainingStatuses.${latestTrainingTask.status}`,
                                { defaultValue: latestTrainingTask.status },
                              )}
                            </span>
                          )}
                          {latestTrainingTask.trainType && (
                            <span className="rounded-md bg-white/42 px-2 py-0.5 ring-1 ring-sky-100/70 dark:bg-sky-950/20 dark:ring-sky-300/12">
                              {t("runpage.statusBar.trainingType")}
                              {t("runpage.statusBar.labelSeparator")}
                              {translateStatusBarTypeValue(
                                latestTrainingTask.trainType,
                                latestTrainingTask.launchMode,
                                latestTrainingTask.container,
                                latestTrainingTask.scriptName,
                                latestTrainingTask.isMultinode,
                              )}
                            </span>
                          )}
                        </div>
                      </div>
                      {canControlLatestTrainingTask && (
                        <div className="flex flex-wrap items-center gap-2 pl-10 sm:pl-0">
                          <Button
                            size="small"
                            className="h-8 rounded-lg border border-sky-200 bg-white/58 px-3 text-xs font-semibold text-sky-950 shadow-none hover:border-sky-300 hover:bg-white dark:border-sky-300/18 dark:bg-sky-950/34 dark:text-sky-50 dark:hover:bg-sky-900/46"
                            disabled={isStatusCommandDisabled}
                            onClick={() => fillTrainingTaskCommand("monitor")}
                          >
                            <span className="inline-flex items-center gap-1.5">
                              <Search className="h-3.5 w-3.5" />
                              {t("runpage.statusBar.queryStatus")}
                            </span>
                          </Button>
                          <Button
                            size="small"
                            className="h-8 rounded-lg border border-sky-200 bg-white/58 px-3 text-xs font-semibold text-sky-950 shadow-none hover:border-sky-300 hover:bg-white dark:border-sky-300/18 dark:bg-sky-950/34 dark:text-sky-50 dark:hover:bg-sky-900/46"
                            onClick={() => setIsMetricsSheetOpen(true)}
                          >
                            <span className="inline-flex items-center gap-1.5">
                              <LineChart className="h-3.5 w-3.5" />
                              {t("runpage.statusBar.openMonitor")}
                            </span>
                          </Button>
                          <Button
                            danger
                            size="small"
                            className="h-8 rounded-lg bg-white/88 px-3 text-xs font-semibold shadow-sm"
                            disabled={isStatusCommandDisabled}
                            onClick={() => fillTrainingTaskCommand("stop")}
                          >
                            <span className="inline-flex items-center gap-1.5">
                              <Square className="h-3.5 w-3.5" />
                              {t("runpage.statusBar.stopTraining")}
                            </span>
                          </Button>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              {!displayedWorkflowTask &&
                latestTrainingTask &&
                !closedStatusBars.train &&
                hiddenStatusBars.train && (
                  <div className="relative shrink-0 border-b border-sky-100/80 bg-sky-50/62 px-4 py-2 dark:border-sky-400/12 dark:bg-sky-950/20">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div className="flex min-w-0 flex-wrap items-center gap-2 text-xs font-medium text-sky-900 dark:text-sky-100">
                        <span className="inline-flex items-center gap-1.5 font-semibold">
                          <Activity className="h-3.5 w-3.5" />
                          {t("runpage.statusBar.trainingTitle")}
                        </span>
                        {latestTrainingTask.container && (
                          <span className="rounded-md bg-white/70 px-2 py-0.5 ring-1 ring-sky-100/80 dark:bg-sky-950/40 dark:ring-sky-300/12">
                            {t("runpage.statusBar.container")}
                            {t("runpage.statusBar.labelSeparator")}
                            {latestTrainingTask.container}
                          </span>
                        )}
                        {latestTrainingTask.pid && (
                          <span className="rounded-md bg-white/70 px-2 py-0.5 ring-1 ring-sky-100/80 dark:bg-sky-950/40 dark:ring-sky-300/12">
                            {t("runpage.statusBar.pid")}
                            {t("runpage.statusBar.labelSeparator")}
                            {latestTrainingTask.pid}
                          </span>
                        )}
                        {latestTrainingTask.status && (
                          <span className="rounded-md bg-white/70 px-2 py-0.5 ring-1 ring-sky-100/80 dark:bg-sky-950/40 dark:ring-sky-300/12">
                            {t("runpage.statusBar.status")}
                            {t("runpage.statusBar.labelSeparator")}
                            {t(
                              `runpage.statusBar.trainingStatuses.${latestTrainingTask.status}`,
                              { defaultValue: latestTrainingTask.status },
                            )}
                          </span>
                        )}
                      </div>
                      <div className="flex items-center gap-2">
                        <Button
                          size="small"
                          className="h-7 w-7 rounded-lg border border-sky-200 bg-white/70 p-0 text-sky-900 shadow-none hover:bg-white dark:border-sky-300/18 dark:bg-sky-950/34 dark:text-sky-50 dark:hover:bg-sky-900/46"
                          aria-label="Close training status bar"
                          onClick={() => closeStatusBar("train")}
                        >
                          <X className="h-3.5 w-3.5" />
                        </Button>
                        <Button
                          size="small"
                          className="h-7 rounded-lg border border-sky-200 bg-white/70 px-2.5 text-xs font-semibold text-sky-900 shadow-none hover:bg-white dark:border-sky-300/18 dark:bg-sky-950/34 dark:text-sky-50 dark:hover:bg-sky-900/46"
                          onClick={() => expandStatusBar("train")}
                        >
                          <span className="inline-flex items-center gap-1">
                            {t("runpage.statusBar.expand")}
                            <ChevronDown className="h-3.5 w-3.5" />
                          </span>
                        </Button>
                      </div>
                    </div>
                  </div>
                )}
              {!displayedWorkflowTask &&
                latestAssessmentTask &&
                !closedStatusBars.evaluation &&
                !hiddenStatusBars.evaluation && (
                  <div className="relative shrink-0 overflow-hidden border-b border-indigo-100/80 bg-[linear-gradient(90deg,rgba(250,251,255,0.98)_0%,rgba(242,245,255,0.74)_48%,rgba(255,255,255,0.96)_100%)] px-4 py-3 shadow-[0_8px_22px_-24px_rgba(79,70,229,0.22)] backdrop-blur-sm dark:border-indigo-400/12 dark:bg-[linear-gradient(90deg,rgba(49,46,129,0.22)_0%,rgba(67,56,202,0.14)_52%,rgba(15,23,42,0.62)_100%)]">
                    <div className="absolute inset-y-0 left-0 w-1 bg-indigo-300 dark:bg-indigo-300/80" />
                    <button
                      type="button"
                      aria-label={t("runpage.statusBar.collapseEvaluation")}
                      className="absolute right-2 top-2 inline-flex h-6 w-6 items-center justify-center rounded-full text-indigo-600/70 transition hover:bg-white/70 hover:text-indigo-900 dark:text-indigo-100/60 dark:hover:bg-indigo-950/38 dark:hover:text-indigo-50"
                      onClick={() => hideStatusBar("evaluation")}
                    >
                      <ChevronUp className="h-3.5 w-3.5" />
                    </button>
                    <button
                      type="button"
                      aria-label="Close evaluation status bar"
                      className="absolute right-9 top-2 inline-flex h-6 w-6 items-center justify-center rounded-full text-indigo-600/70 transition hover:bg-white/70 hover:text-indigo-900 dark:text-indigo-100/60 dark:hover:bg-indigo-950/38 dark:hover:text-indigo-50"
                      onClick={() => closeStatusBar("evaluation")}
                    >
                      <X className="h-3.5 w-3.5" />
                    </button>
                    <div className="flex flex-col gap-2 pl-2 sm:flex-row sm:items-center sm:justify-between">
                      <div className="min-w-0">
                        <div className="flex items-center gap-2 text-[15px] font-semibold text-indigo-950 dark:text-indigo-50">
                          <span className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-indigo-500/90 text-white shadow-[0_8px_18px_-15px_rgba(79,70,229,0.54)] dark:bg-indigo-400/90 dark:text-indigo-950">
                            <Activity className="h-4 w-4" />
                          </span>
                          {t("runpage.statusBar.evaluationTitle")}
                        </div>
                        <div className="mt-1 flex flex-wrap gap-1.5 pl-10 text-[11px] font-medium text-indigo-800 dark:text-indigo-100">
                          {latestAssessmentTask.container && (
                            <span className="rounded-md bg-white/42 px-2 py-0.5 ring-1 ring-indigo-100/70 dark:bg-indigo-950/20 dark:ring-indigo-300/12">
                              {t("runpage.statusBar.container")}
                              {t("runpage.statusBar.labelSeparator")}
                              {latestAssessmentTask.container}
                            </span>
                          )}
                          {latestAssessmentTask.pid && (
                            <span className="rounded-md bg-white/42 px-2 py-0.5 ring-1 ring-indigo-100/70 dark:bg-indigo-950/20 dark:ring-indigo-300/12">
                              {t("runpage.statusBar.pid")}
                              {t("runpage.statusBar.labelSeparator")}
                              {latestAssessmentTask.pid}
                            </span>
                          )}
                          {(latestAssessmentTask.assessmentTypeText ||
                            latestAssessmentTask.assessmentType ||
                            latestAssessmentTask.evalTypeText ||
                            latestAssessmentTask.evalType) && (
                            <span className="rounded-md bg-white/42 px-2 py-0.5 ring-1 ring-indigo-100/70 dark:bg-indigo-950/20 dark:ring-indigo-300/12">
                              {t("runpage.statusBar.evaluationType")}
                              {t("runpage.statusBar.labelSeparator")}
                              {translateStatusBarTypeValue(
                                latestAssessmentTask.assessmentTypeText ||
                                  latestAssessmentTask.assessmentType ||
                                  latestAssessmentTask.evalTypeText ||
                                  latestAssessmentTask.evalType,
                              )}
                            </span>
                          )}
                        </div>
                      </div>
                      <div className="flex flex-wrap items-center gap-2 pl-10 sm:pl-0">
                        <Button
                          size="small"
                          className="h-8 rounded-lg border border-indigo-200 bg-white/58 px-3 text-xs font-semibold text-indigo-950 shadow-none hover:border-indigo-300 hover:bg-white dark:border-indigo-300/18 dark:bg-indigo-950/34 dark:text-indigo-50 dark:hover:bg-indigo-900/46"
                          disabled={isStatusCommandDisabled}
                          onClick={() => fillAssessmentTaskCommand("monitor")}
                        >
                          <span className="inline-flex items-center gap-1.5">
                            <Search className="h-3.5 w-3.5" />
                            {t("runpage.statusBar.viewStatus") || "View Status"}
                          </span>
                        </Button>
                        <Button
                          danger
                          size="small"
                          className="h-8 rounded-lg bg-white/88 px-3 text-xs font-semibold shadow-sm"
                          disabled={isStatusCommandDisabled}
                          onClick={() => fillAssessmentTaskCommand("stop")}
                        >
                          <span className="inline-flex items-center gap-1.5">
                            <Square className="h-3.5 w-3.5" />
                            {t("runpage.statusBar.stopEvaluation")}
                          </span>
                        </Button>
                      </div>
                    </div>
                  </div>
                )}
              {!displayedWorkflowTask &&
                latestAssessmentTask &&
                !closedStatusBars.evaluation &&
                hiddenStatusBars.evaluation && (
                  <div className="relative shrink-0 border-b border-indigo-100/80 bg-indigo-50/62 px-4 py-2 dark:border-indigo-400/12 dark:bg-indigo-950/20">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div className="flex min-w-0 flex-wrap items-center gap-2 text-xs font-medium text-indigo-900 dark:text-indigo-100">
                        <span className="inline-flex items-center gap-1.5 font-semibold">
                          <Activity className="h-3.5 w-3.5" />
                          {t("runpage.statusBar.evaluationTitle")}
                        </span>
                        {latestAssessmentTask.container && (
                          <span className="rounded-md bg-white/70 px-2 py-0.5 ring-1 ring-indigo-100/80 dark:bg-indigo-950/40 dark:ring-indigo-300/12">
                            {t("runpage.statusBar.container")}
                            {t("runpage.statusBar.labelSeparator")}
                            {latestAssessmentTask.container}
                          </span>
                        )}
                        {latestAssessmentTask.pid && (
                          <span className="rounded-md bg-white/70 px-2 py-0.5 ring-1 ring-indigo-100/80 dark:bg-indigo-950/40 dark:ring-indigo-300/12">
                            {t("runpage.statusBar.pid")}
                            {t("runpage.statusBar.labelSeparator")}
                            {latestAssessmentTask.pid}
                          </span>
                        )}
                      </div>
                      <div className="flex items-center gap-2">
                        <Button
                          size="small"
                          className="h-7 w-7 rounded-lg border border-indigo-200 bg-white/70 p-0 text-indigo-900 shadow-none hover:bg-white dark:border-indigo-300/18 dark:bg-indigo-950/34 dark:text-indigo-50 dark:hover:bg-indigo-900/46"
                          aria-label="Close evaluation status bar"
                          onClick={() => closeStatusBar("evaluation")}
                        >
                          <X className="h-3.5 w-3.5" />
                        </Button>
                        <Button
                          size="small"
                          className="h-7 rounded-lg border border-indigo-200 bg-white/70 px-2.5 text-xs font-semibold text-indigo-900 shadow-none hover:bg-white dark:border-indigo-300/18 dark:bg-indigo-950/34 dark:text-indigo-50 dark:hover:bg-indigo-900/46"
                          onClick={() => expandStatusBar("evaluation")}
                        >
                          <span className="inline-flex items-center gap-1">
                            {t("runpage.statusBar.expand")}
                            <ChevronDown className="h-3.5 w-3.5" />
                          </span>
                        </Button>
                      </div>
                    </div>
                  </div>
                )}
              {!displayedWorkflowTask &&
                displayedBenchmarkTask &&
                !closedStatusBars.benchmark &&
                !hiddenStatusBars.benchmark && (
                  <div className="relative shrink-0 overflow-hidden border-b border-emerald-100/80 bg-[linear-gradient(90deg,rgba(247,254,251,0.98)_0%,rgba(236,253,245,0.74)_48%,rgba(255,255,255,0.96)_100%)] px-4 py-3 shadow-[0_8px_22px_-24px_rgba(5,150,105,0.22)] backdrop-blur-sm dark:border-emerald-400/12 dark:bg-[linear-gradient(90deg,rgba(6,78,59,0.22)_0%,rgba(4,120,87,0.14)_52%,rgba(15,23,42,0.62)_100%)]">
                    <div className="absolute inset-y-0 left-0 w-1 bg-emerald-300 dark:bg-emerald-300/80" />
                    <button
                      type="button"
                      aria-label={t("runpage.statusBar.collapseBenchmark")}
                      className="absolute right-2 top-2 inline-flex h-6 w-6 items-center justify-center rounded-full text-emerald-600/70 transition hover:bg-white/70 hover:text-emerald-900 dark:text-emerald-100/60 dark:hover:bg-emerald-950/38 dark:hover:text-emerald-50"
                      onClick={() => hideStatusBar("benchmark")}
                    >
                      <ChevronUp className="h-3.5 w-3.5" />
                    </button>
                    <button
                      type="button"
                      aria-label="Close benchmark status bar"
                      className="absolute right-9 top-2 inline-flex h-6 w-6 items-center justify-center rounded-full text-emerald-600/70 transition hover:bg-white/70 hover:text-emerald-900 dark:text-emerald-100/60 dark:hover:bg-emerald-950/38 dark:hover:text-emerald-50"
                      onClick={() => closeStatusBar("benchmark")}
                    >
                      <X className="h-3.5 w-3.5" />
                    </button>
                    <div className="flex flex-col gap-2 pl-2 sm:flex-row sm:items-center sm:justify-between">
                      <div className="min-w-0">
                        <div className="flex items-center gap-2 text-[15px] font-semibold text-emerald-950 dark:text-emerald-50">
                          <span className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-emerald-500/90 text-white shadow-[0_8px_18px_-15px_rgba(5,150,105,0.54)] dark:bg-emerald-400/90 dark:text-emerald-950">
                            <Activity className="h-4 w-4" />
                          </span>
                          {t("runpage.statusBar.benchmarkTitle")}
                        </div>
                      </div>
                      <div className="flex flex-wrap items-center gap-2 pl-10 sm:pl-0">
                        <Button
                          size="small"
                          className="h-8 rounded-lg border border-emerald-200 bg-white/88 px-3 text-xs font-semibold text-emerald-900 shadow-sm hover:bg-white dark:border-emerald-300/18 dark:bg-emerald-950/34 dark:text-emerald-50 dark:hover:bg-emerald-900/46"
                          disabled={isStatusCommandDisabled}
                          onClick={() => fillBenchmarkTaskCommand("status")}
                        >
                          <span className="inline-flex items-center gap-1.5">
                            <Search className="h-3.5 w-3.5" />
                            {t("runpage.statusBar.viewBenchmarkStatus")}
                          </span>
                        </Button>
                        <Button
                          danger
                          size="small"
                          className="h-8 rounded-lg bg-white/88 px-3 text-xs font-semibold shadow-sm"
                          disabled={isStatusCommandDisabled}
                          onClick={() => fillBenchmarkTaskCommand("stop")}
                        >
                          <span className="inline-flex items-center gap-1.5">
                            <Square className="h-3.5 w-3.5" />
                            {t("runpage.statusBar.stopBenchmark")}
                          </span>
                        </Button>
                      </div>
                    </div>
                  </div>
                )}
              {!displayedWorkflowTask &&
                displayedBenchmarkTask &&
                !closedStatusBars.benchmark &&
                hiddenStatusBars.benchmark && (
                  <div className="relative shrink-0 border-b border-emerald-100/80 bg-emerald-50/62 px-4 py-2 dark:border-emerald-400/12 dark:bg-emerald-950/20">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div className="flex min-w-0 flex-wrap items-center gap-2 text-xs font-medium text-emerald-900 dark:text-emerald-100">
                        <span className="inline-flex items-center gap-1.5 font-semibold">
                          <Activity className="h-3.5 w-3.5" />
                          {t("runpage.statusBar.benchmarkTitle")}
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        <Button
                          size="small"
                          className="h-7 w-7 rounded-lg border border-emerald-200 bg-white/70 p-0 text-emerald-900 shadow-none hover:bg-white dark:border-emerald-300/18 dark:bg-emerald-950/34 dark:text-emerald-50 dark:hover:bg-emerald-900/46"
                          aria-label="Close benchmark status bar"
                          onClick={() => closeStatusBar("benchmark")}
                        >
                          <X className="h-3.5 w-3.5" />
                        </Button>
                        <Button
                          size="small"
                          className="h-7 rounded-lg border border-emerald-200 bg-white/70 px-2.5 text-xs font-semibold text-emerald-900 shadow-none hover:bg-white dark:border-emerald-300/18 dark:bg-emerald-950/34 dark:text-emerald-50 dark:hover:bg-emerald-900/46"
                          onClick={() => expandStatusBar("benchmark")}
                        >
                          <span className="inline-flex items-center gap-1">
                            {t("runpage.statusBar.expand")}
                            <ChevronDown className="h-3.5 w-3.5" />
                          </span>
                        </Button>
                      </div>
                    </div>
                  </div>
                )}
              {!displayedWorkflowTask &&
                latestInferenceTask &&
                !closedStatusBars.inference &&
                !hiddenStatusBars.inference && (
                  <div className="relative shrink-0 overflow-hidden border-b border-cyan-100/80 bg-[linear-gradient(90deg,rgba(248,253,255,0.98)_0%,rgba(239,250,252,0.74)_48%,rgba(255,255,255,0.96)_100%)] px-4 py-3 shadow-[0_8px_22px_-24px_rgba(14,116,144,0.22)] backdrop-blur-sm dark:border-cyan-400/12 dark:bg-[linear-gradient(90deg,rgba(8,51,68,0.24)_0%,rgba(12,74,110,0.14)_52%,rgba(15,23,42,0.62)_100%)]">
                    <div className="absolute inset-y-0 left-0 w-1 bg-cyan-300 dark:bg-cyan-300/80" />
                    <button
                      type="button"
                      aria-label={t("runpage.statusBar.collapseInference")}
                      className="absolute right-2 top-2 inline-flex h-6 w-6 items-center justify-center rounded-full text-cyan-600/70 transition hover:bg-white/70 hover:text-cyan-900 dark:text-cyan-100/60 dark:hover:bg-cyan-950/38 dark:hover:text-cyan-50"
                      onClick={() => hideStatusBar("inference")}
                    >
                      <ChevronUp className="h-3.5 w-3.5" />
                    </button>
                    <button
                      type="button"
                      aria-label="Close inference status bar"
                      className="absolute right-9 top-2 inline-flex h-6 w-6 items-center justify-center rounded-full text-cyan-600/70 transition hover:bg-white/70 hover:text-cyan-900 dark:text-cyan-100/60 dark:hover:bg-cyan-950/38 dark:hover:text-cyan-50"
                      onClick={() => closeStatusBar("inference")}
                    >
                      <X className="h-3.5 w-3.5" />
                    </button>
                    <div className="flex flex-col gap-2 pl-2 sm:flex-row sm:items-center sm:justify-between">
                      <div className="min-w-0">
                        <div className="flex items-center gap-2 text-[15px] font-semibold text-cyan-950 dark:text-cyan-50">
                          <span className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-cyan-500/90 text-white shadow-[0_8px_18px_-15px_rgba(14,116,144,0.54)] dark:bg-cyan-400/90 dark:text-cyan-950">
                            <Server className="h-4 w-4" />
                          </span>
                          {t("runpage.statusBar.inferenceTitle")}
                        </div>
                        <div className="mt-1 flex flex-wrap gap-1.5 pl-10 text-[11px] font-medium text-cyan-800 dark:text-cyan-100">
                          <span className="rounded-md bg-white/42 px-2 py-0.5 ring-1 ring-cyan-100/70 dark:bg-cyan-950/20 dark:ring-cyan-300/12">
                            {t("runpage.statusBar.config")}
                            {latestInferenceTask.hasConfig
                              ? t("runpage.statusBar.queried")
                              : t("runpage.statusBar.notQueried")}
                          </span>
                          <span className="rounded-md bg-white/42 px-2 py-0.5 ring-1 ring-cyan-100/70 dark:bg-cyan-950/20 dark:ring-cyan-300/12">
                            {t("runpage.statusBar.serviceStatus")}
                            {latestInferenceTask.hasStatus
                              ? t("runpage.statusBar.queried")
                              : t("runpage.statusBar.notQueried")}
                          </span>
                        </div>
                      </div>
                      <div className="flex flex-wrap items-center gap-2 pl-10 sm:pl-0">
                        <Button
                          size="small"
                          className="h-8 rounded-lg border border-cyan-200 bg-white/58 px-3 text-xs font-semibold text-cyan-950 shadow-none hover:border-cyan-300 hover:bg-white dark:border-cyan-300/18 dark:bg-cyan-950/34 dark:text-cyan-50 dark:hover:bg-cyan-900/46"
                          onClick={() => openInferencePanel("config")}
                        >
                          <span className="inline-flex items-center gap-1.5">
                            <FileTextOutlined />
                            {t("runpage.statusBar.configFile")}
                          </span>
                        </Button>
                        <Button
                          size="small"
                          className="h-8 rounded-lg border border-cyan-200 bg-white/58 px-3 text-xs font-semibold text-cyan-950 shadow-none hover:border-cyan-300 hover:bg-white dark:border-cyan-300/18 dark:bg-cyan-950/34 dark:text-cyan-50 dark:hover:bg-cyan-900/46"
                          onClick={() => openInferencePanel("status")}
                        >
                          <span className="inline-flex items-center gap-1.5">
                            <Search className="h-3.5 w-3.5" />
                            {t("runpage.statusBar.serviceStatus")}
                          </span>
                        </Button>
                        <Button
                          size="small"
                          className="h-8 rounded-lg border border-cyan-200 bg-white/58 px-3 text-xs font-semibold text-cyan-950 shadow-none hover:border-cyan-300 hover:bg-white dark:border-cyan-300/18 dark:bg-cyan-950/34 dark:text-cyan-50 dark:hover:bg-cyan-900/46"
                          disabled={isStatusCommandDisabled}
                          onClick={() =>
                            handleAgentWaitingReply("启动推理服务")
                          }
                        >
                          <span className="inline-flex items-center gap-1.5">
                            <Zap className="h-3.5 w-3.5" />
                            {t("runpage.statusBar.startService")}
                          </span>
                        </Button>
                        <Button
                          danger
                          size="small"
                          className="h-8 rounded-lg bg-white/88 px-3 text-xs font-semibold shadow-sm"
                          disabled={isStatusCommandDisabled}
                          onClick={() =>
                            handleAgentWaitingReply("关闭推理服务")
                          }
                        >
                          <span className="inline-flex items-center gap-1.5">
                            <Square className="h-3.5 w-3.5" />
                            {t("runpage.statusBar.stopService")}
                          </span>
                        </Button>
                      </div>
                    </div>
                  </div>
                )}
              {!displayedWorkflowTask &&
                latestInferenceTask &&
                !closedStatusBars.inference &&
                hiddenStatusBars.inference && (
                  <div className="relative shrink-0 border-b border-cyan-100/80 bg-cyan-50/62 px-4 py-2 dark:border-cyan-400/12 dark:bg-cyan-950/20">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div className="flex min-w-0 flex-wrap items-center gap-2 text-xs font-medium text-cyan-900 dark:text-cyan-100">
                        <span className="inline-flex items-center gap-1.5 font-semibold">
                          <Server className="h-3.5 w-3.5" />
                          {t("runpage.statusBar.inferenceTitle")}
                        </span>
                        <span className="rounded-md bg-white/70 px-2 py-0.5 ring-1 ring-cyan-100/80 dark:bg-cyan-950/40 dark:ring-cyan-300/12">
                          {t("runpage.statusBar.config")}
                          {latestInferenceTask.hasConfig
                            ? t("runpage.statusBar.queried")
                            : t("runpage.statusBar.notQueried")}
                        </span>
                        <span className="rounded-md bg-white/70 px-2 py-0.5 ring-1 ring-cyan-100/80 dark:bg-cyan-950/40 dark:ring-cyan-300/12">
                          {t("runpage.statusBar.status")}
                          {latestInferenceTask.hasStatus
                            ? t("runpage.statusBar.queried")
                            : t("runpage.statusBar.notQueried")}
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        <Button
                          size="small"
                          className="h-7 w-7 rounded-lg border border-cyan-200 bg-white/70 p-0 text-cyan-900 shadow-none hover:bg-white dark:border-cyan-300/18 dark:bg-cyan-950/34 dark:text-cyan-50 dark:hover:bg-cyan-900/46"
                          aria-label="Close inference status bar"
                          onClick={() => closeStatusBar("inference")}
                        >
                          <X className="h-3.5 w-3.5" />
                        </Button>
                        <Button
                          size="small"
                          className="h-7 rounded-lg border border-cyan-200 bg-white/70 px-2.5 text-xs font-semibold text-cyan-900 shadow-none hover:bg-white dark:border-cyan-300/18 dark:bg-cyan-950/34 dark:text-cyan-50 dark:hover:bg-cyan-900/46"
                          onClick={() => expandStatusBar("inference")}
                        >
                          <span className="inline-flex items-center gap-1">
                            {t("runpage.statusBar.expand")}
                            <ChevronDown className="h-3.5 w-3.5" />
                          </span>
                        </Button>
                      </div>
                    </div>
                  </div>
                )}
              {displayedDataFilterTask &&
                !closedStatusBars.data_filter &&
                !hiddenStatusBars.data_filter && (
                  <div className="relative shrink-0 overflow-hidden border-b border-violet-100/80 bg-[linear-gradient(90deg,rgba(250,245,255,0.98)_0%,rgba(243,232,255,0.74)_48%,rgba(255,255,255,0.96)_100%)] px-4 py-3 shadow-[0_8px_22px_-24px_rgba(124,58,237,0.22)] backdrop-blur-sm dark:border-violet-400/12 dark:bg-[linear-gradient(90deg,rgba(46,16,101,0.24)_0%,rgba(76,29,149,0.14)_52%,rgba(15,23,42,0.62)_100%)]">
                    <div className="absolute inset-y-0 left-0 w-1 bg-violet-400 dark:bg-violet-400/80" />
                    <button
                      type="button"
                      aria-label={t("runpage.statusBar.collapseDataFilter")}
                      className="absolute right-2 top-2 inline-flex h-6 w-6 items-center justify-center rounded-full text-violet-600/70 transition hover:bg-white/70 hover:text-violet-900 dark:text-violet-100/60 dark:hover:bg-violet-950/38 dark:hover:text-violet-50"
                      onClick={() => hideStatusBar("data_filter")}
                    >
                      <ChevronUp className="h-3.5 w-3.5" />
                    </button>
                    <button
                      type="button"
                      aria-label={t("runpage.statusBar.closeDataFilter")}
                      className="absolute right-9 top-2 inline-flex h-6 w-6 items-center justify-center rounded-full text-violet-600/70 transition hover:bg-white/70 hover:text-violet-900 dark:text-violet-100/60 dark:hover:bg-violet-950/38 dark:hover:text-violet-50"
                      onClick={() => closeStatusBar("data_filter")}
                    >
                      <X className="h-3.5 w-3.5" />
                    </button>
                    <div className="flex flex-col gap-2 pl-2 sm:flex-row sm:items-center sm:justify-between">
                      <div className="min-w-0 flex-1">
                        <div className="flex flex-wrap items-center gap-2 text-[15px] font-semibold text-violet-950 dark:text-violet-50">
                          <span className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-violet-500/90 text-white shadow-[0_8px_18px_-15px_rgba(124,58,237,0.54)] dark:bg-violet-400/90 dark:text-violet-950">
                            <Filter className="h-4 w-4" />
                          </span>
                          {t("runpage.statusBar.dataFilterTitle")}
                        </div>
                        <div className="mt-1 flex flex-col gap-1.5 pl-10">
                          <div className="flex flex-wrap items-center gap-1.5 text-[11px] font-medium text-violet-800 dark:text-violet-100">
                            <span className="rounded-md bg-white/42 px-2 py-0.5 ring-1 ring-violet-100/70 dark:bg-violet-950/20 dark:ring-violet-300/12">
                              {t("runpage.statusBar.status")}
                              {t("runpage.statusBar.labelSeparator")}
                              {displayedDataFilterTask.status === "running"
                                ? t("runpage.statusBar.dataFilterStatusRunning")
                                : displayedDataFilterTask.status === "completed"
                                  ? t("runpage.statusBar.dataFilterStatusCompleted")
                                  : displayedDataFilterTask.status === "failed"
                                    ? t("runpage.statusBar.dataFilterStatusFailed")
                                    : t("runpage.statusBar.dataFilterStatusNotStarted")}
                            </span>
                            {displayedDataFilterTask.outputDatasetName && (
                              <span className="rounded-md bg-white/42 px-2 py-0.5 ring-1 ring-violet-100/70 dark:bg-violet-950/20 dark:ring-violet-300/12">
                                {t("runpage.statusBar.dataFilterOutputDataset")}
                                {t("runpage.statusBar.labelSeparator")}
                                {displayedDataFilterTask.outputDatasetName}
                              </span>
                            )}
                          </div>
                          {displayedDataFilterTask.status === "running" &&
                            (displayedDataFilterTask.percent !== undefined ||
                              displayedDataFilterTask.processedItems !== undefined ||
                              displayedDataFilterTask.totalItems !== undefined ||
                              displayedDataFilterTask.currentFile) && (
                            <>
                              {displayedDataFilterTask.percent !== undefined && (
                                <div className="flex flex-wrap items-center gap-2 text-[11px] font-medium text-violet-800 dark:text-violet-100">
                                  <div className="h-1.5 w-28 overflow-hidden rounded-full bg-violet-200/70 dark:bg-violet-900/40">
                                    <div
                                      className="h-full rounded-full bg-violet-500 transition-all duration-500 dark:bg-violet-400"
                                      style={{
                                        width: `${Math.max(0, Math.min(100, displayedDataFilterTask.percent))}%`,
                                      }}
                                    />
                                  </div>
                                  <span className="text-[11px] font-semibold text-violet-900 dark:text-violet-100">
                                    {displayedDataFilterTask.percent}%
                                  </span>
                                </div>
                              )}
                              {(displayedDataFilterTask.processedItems !== undefined ||
                                displayedDataFilterTask.totalItems !== undefined) && (
                                <div className="text-[11px] font-medium text-violet-800 dark:text-violet-200">
                                  {t("runpage.statusBar.dataFilterProgressText", {
                                    processed: displayedDataFilterTask.processedItems ?? 0,
                                    total: displayedDataFilterTask.totalItems ?? 0,
                                  })}
                                </div>
                              )}
                              {displayedDataFilterTask.currentFile && (
                                <div className="text-[11px] font-medium text-violet-800 dark:text-violet-200">
                                  {t("runpage.statusBar.dataFilterCurrentFile", {
                                    file: displayedDataFilterTask.currentFile,
                                  })}
                                </div>
                              )}
                            </>
                          )}
                          {displayedDataFilterTask.status === "completed" && (
                            <div className="text-[11px] font-medium text-violet-800 dark:text-violet-200">
                              {t("runpage.statusBar.dataFilterCompletedHint", {
                                dataset: displayedDataFilterTask.outputDatasetName || "-",
                              })}
                            </div>
                          )}
                          {displayedDataFilterTask.status === "failed" && (
                            <div className="text-[11px] font-medium text-red-600 dark:text-red-400">
                              {t("runpage.statusBar.dataFilterFailedHint")}
                              {displayedDataFilterTask.error
                                ? `：${displayedDataFilterTask.error}`
                                : ""}
                            </div>
                          )}
                          <div className="flex flex-wrap items-center gap-1.5 text-[11px] font-medium text-violet-700 dark:text-violet-300">
                            {dataFilterLastUpdatedText && (
                              <span>
                                {t("runpage.statusBar.dataFilterLastUpdated")}
                                {dataFilterLastUpdatedText}
                              </span>
                            )}
                            {dataFilterMonitorState.error && (
                              <span className="text-amber-600 dark:text-amber-400">
                                {t("runpage.statusBar.dataFilterRefreshFailed")}
                              </span>
                            )}
                            {dataFilterRefreshCountdown !== undefined &&
                              displayedDataFilterTask.status === "running" && (
                                <span>
                                  {t("runpage.statusBar.dataFilterNextRefresh", {
                                    seconds: dataFilterRefreshCountdown,
                                  })}
                                </span>
                              )}
                          </div>
                        </div>
                      </div>
                      <div className="flex flex-wrap items-center gap-2 pl-10 sm:pl-0">
                        <Button
                          size="small"
                          loading={dataFilterMonitorState.isManualRefreshing}
                          className="h-8 rounded-lg border border-violet-200 bg-white/58 px-3 text-xs font-semibold text-violet-950 shadow-none hover:border-violet-300 hover:bg-white dark:border-violet-300/18 dark:bg-violet-950/34 dark:text-violet-50 dark:hover:bg-violet-900/46"
                          onClick={() => void fetchDataFilterStatus(true)}
                        >
                          <span className="inline-flex items-center gap-1.5">
                            <RefreshCw className="h-3.5 w-3.5" />
                            {t("runpage.statusBar.dataFilterRefresh")}
                          </span>
                        </Button>
                      </div>
                    </div>
                  </div>
                )}
              {displayedDataFilterTask &&
                !closedStatusBars.data_filter &&
                hiddenStatusBars.data_filter && (
                  <div className="relative shrink-0 border-b border-violet-100/80 bg-violet-50/62 px-4 py-2 dark:border-violet-400/12 dark:bg-violet-950/20">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div className="flex min-w-0 flex-wrap items-center gap-2 text-xs font-medium text-violet-900 dark:text-violet-100">
                        <span className="inline-flex items-center gap-1.5 font-semibold">
                          <Filter className="h-3.5 w-3.5" />
                          {t("runpage.statusBar.dataFilterTitle")}
                        </span>
                        <span className="rounded-md bg-white/70 px-2 py-0.5 ring-1 ring-violet-100/80 dark:bg-violet-950/40 dark:ring-violet-300/12">
                          {displayedDataFilterTask.status === "running"
                            ? t("runpage.statusBar.dataFilterStatusRunning")
                            : displayedDataFilterTask.status === "completed"
                              ? t("runpage.statusBar.dataFilterStatusCompleted")
                              : displayedDataFilterTask.status === "failed"
                                ? t("runpage.statusBar.dataFilterStatusFailed")
                                : t("runpage.statusBar.dataFilterStatusNotStarted")}
                        </span>
                        {displayedDataFilterTask.status === "running" &&
                          displayedDataFilterTask.percent !== undefined && (
                          <span className="rounded-md bg-white/70 px-2 py-0.5 ring-1 ring-violet-100/80 dark:bg-violet-950/40 dark:ring-violet-300/12">
                            {displayedDataFilterTask.percent}%
                          </span>
                        )}
                      </div>
                      <div className="flex items-center gap-2">
                        <Button
                          size="small"
                          className="h-7 w-7 rounded-lg border border-violet-200 bg-white/70 p-0 text-violet-900 shadow-none hover:bg-white dark:border-violet-300/18 dark:bg-violet-950/34 dark:text-violet-50 dark:hover:bg-violet-900/46"
                          aria-label={t("runpage.statusBar.closeDataFilter")}
                          onClick={() => closeStatusBar("data_filter")}
                        >
                          <X className="h-3.5 w-3.5" />
                        </Button>
                        <Button
                          size="small"
                          className="h-7 rounded-lg border border-violet-200 bg-white/70 px-2.5 text-xs font-semibold text-violet-900 shadow-none hover:bg-white dark:border-violet-300/18 dark:bg-violet-950/34 dark:text-violet-50 dark:hover:bg-violet-900/46"
                          onClick={() => expandStatusBar("data_filter")}
                        >
                          <span className="inline-flex items-center gap-1">
                            {t("runpage.statusBar.expand")}
                            <ChevronDown className="h-3.5 w-3.5" />
                          </span>
                        </Button>
                      </div>
                    </div>
                  </div>
                )}
              <div className="flex min-h-0 flex-1 flex-col">
                {(isAdmin ||
                  runnablePoolsQuery.isLoading ||
                  outboundResourceGroupId) && (
                  <div
                    className={
                      hasVisibleTrainingStatusBar
                        ? "flex shrink-0 flex-wrap items-center gap-2 border-b border-slate-200/60 bg-slate-50/62 px-4 py-2 text-xs text-slate-600 dark:border-white/8 dark:bg-slate-950/34 dark:text-slate-300"
                        : "mb-2 flex shrink-0 flex-wrap items-center gap-2 rounded-xl border border-slate-200/70 bg-white/70 px-3 py-2 text-xs text-slate-600 dark:border-white/10 dark:bg-slate-900/62 dark:text-slate-300"
                    }
                  >
                    {isAdmin && (
                      <>
                        <span className="font-medium">
                          {t("runpage.adminResourceGroup") || "目标用户组"}
                        </span>
                        <Select
                          size="small"
                          className="min-w-48"
                          value={selectedResourceGroupId || undefined}
                          placeholder={
                            t("runpage.adminResourceGroupPlaceholder") ||
                            "选择本次输入使用的用户组"
                          }
                          loading={resourceGroupsQuery.isLoading}
                          disabled={!runResourceGroups.length}
                          notFoundContent={
                            resourceGroupsQuery.isLoading
                              ? undefined
                              : t("runpage.adminResourceGroupPlaceholder") ||
                                "选择本次输入使用的用户组"
                          }
                          options={runResourceGroups.map((group) => ({
                            value: group.id,
                            label: group.name,
                          }))}
                          onChange={setSelectedResourceGroupId}
                        />
                      </>
                    )}
                    {runnableTrainingPools.length > 0 && (
                      <>
                        <span className="font-medium">
                          {t("runpage.trainingPool") || "训练资源池"}
                        </span>
                        <Select
                          size="small"
                          className="min-w-56"
                          value={outboundTrainingPoolId}
                          placeholder={
                            t("runpage.trainingPoolPlaceholder") ||
                            "选择本次训练资源池"
                          }
                          loading={runnablePoolsQuery.isLoading}
                          disabled={runnableTrainingPools.length <= 1}
                          optionFilterProp="searchLabel"
                          options={trainingPoolOptions}
                          onChange={setSelectedTrainingPoolId}
                        />
                      </>
                    )}
                    {outboundResourceGroupId &&
                      !runnablePoolsQuery.isLoading &&
                      !runnableTrainingPools.length && (
                        <span className="text-slate-400 dark:text-slate-500">
                          {t("runpage.trainingPoolEmpty") ||
                            "该用户组暂无训练资源池配额"}
                        </span>
                      )}
                    <span className="text-slate-400 dark:text-slate-500">
                      {isTrainingPoolRequired
                        ? t("runpage.trainingPoolRequired") ||
                          "请选择本次训练资源池"
                        : t("runpage.trainingPoolHint") ||
                          "训练/评估启动会使用所选资源池"}
                    </span>
                  </div>
                )}
                <div className="min-h-0 flex-1">
                  <AsChat
                    replies={chatDisplayReplies}
                    isReplying={isReplying}
                    onSendClick={onSendClick}
                    disableSendBtn={
                      !currentInputRequest ||
                      isReplying ||
                      isAdminResourceGroupRequired ||
                      isTrainingPoolRequired
                    }
                    allowInterrupt
                    onInterruptClick={onInterruptClick}
                    placeholder={placeholder}
                    tooltips={{
                      sendButton:
                        isReplying
                          ? t("tooltip.button.stop-response") || "停止生成"
                          : currentInputRequest
                            ? t("tooltip.button.send-message", {
                                shortcutKeys,
                              })
                            : disabledInputHint,
                      interruptButton:
                        t("tooltip.button.stop-response") || "停止生成",
                      attachButton: t("tooltip.button.attachment-add"),
                      expandTextarea: t("tooltip.button.expand-textarea"),
                    }}
                    attachMaxFileSize={20 * 1024 * 1024} // 20 MB
                    attachAccept={["image/*", "video/*", "audio/*"]}
                    onError={async (error) => {
                      messageApi.error(error);
                    }}
                    stripUsernamePrefix={true}
                    autoScroll={autoScroll}
                    onCommand={handleCommand}
                    enableCommandDetection={true}
                    inputText={asChatInputText}
                    onChange={setAsChatInputText}
                    onAgentWaitingReply={handleAgentWaitingReply}
                    agentDatasets={conversationDatasets}
                    agentModels={conversationModels}
                    onRefreshAgentDatasets={refreshConversationDatasets}
                    onRefreshAgentModels={refreshConversationModels}
                    resourceGroupId={conversationResourceGroupId}
                    inlineHintText={inputInlineHint}
                    sendButtonHighlightToken={sendButtonHighlightToken}
                    onSendComplete={() => {
                      // 使用发送前的命令内容进行验证
                      const sentCommand =
                        lastSentCommandRef.current || asChatInputText;
                      setInputInlineHint("");
                      if (wizardCommandPending) {
                        setWizardResultBaseline({
                          assistantReplyCount: filteredReplies.filter((reply) =>
                            reply.messages?.some(
                              (message) => message.role === "assistant",
                            ),
                          ).length,
                          assistantMessageIdsBeforeSend:
                            getAssistantMessageIds(filteredReplies),
                          sentAt: Date.now(),
                          commandResultCount: commandResults.length,
                        });
                        setWizardPendingSentCommand(sentCommand);
                        setWizardResumeState("sent");
                        setWizardCommandPending(false);
                      }
                    }}
                  />
                </div>
              </div>
            </div>
            {wizardResumeState !== "hidden" && !isQuickStartWizardOpen && (
              <div className="pointer-events-none absolute inset-x-0 bottom-24 z-30 flex justify-center px-4">
                <div className="pointer-events-auto flex w-full max-w-[720px] items-center justify-between gap-3 rounded-2xl border border-sky-200/80 bg-white/95 px-4 py-3 shadow-[0_18px_48px_-28px_rgba(15,23,42,0.35)] backdrop-blur-md dark:border-sky-400/20 dark:bg-slate-900/95">
                  <div className="min-w-0">
                    <div className="text-sm font-medium text-slate-900 dark:text-slate-100">
                      {t("wizard.resume.progress", {
                        step: wizardCommandProgress.step + 1,
                        total: 4,
                        title: wizardCommandProgress.title,
                      })}
                    </div>
                    <div className="mt-1 text-sm font-medium text-slate-900 dark:text-slate-100">
                      {wizardResumeState === "ready"
                        ? t("wizard.resume.readyTitle")
                        : wizardResumeState === "filled"
                          ? t("wizard.resume.filledTitle")
                          : t("wizard.resume.sentTitle")}
                    </div>
                  </div>
                  <div className="flex shrink-0 items-center gap-2">
                    <Button
                      onClick={restartQuickStartWizard}
                      disabled={wizardResumeState === "sent"}
                      className="h-9 rounded-xl px-4"
                    >
                      {t("wizard.resume.restartButton")}
                    </Button>
                    <Button
                      type="primary"
                      disabled={wizardResumeState !== "ready"}
                      onClick={() => {
                        setIsQuickStartWizardOpen(true);
                        setWizardResumeState("hidden");
                      }}
                      className="h-9 rounded-xl px-4"
                    >
                      {t("wizard.resume.continueButton")}
                    </Button>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* 训练流程指南 */}
        <TrainingWorkflowGuide
          open={isTrainingGuideOpen}
          onClose={() => setIsTrainingGuideOpen(false)}
          onComplete={() => setIsPostTrainingGuideChoiceOpen(true)}
        />
        <Modal
          title={null}
          open={showPostTourChoice}
          onCancel={closePostTourChoice}
          footer={null}
          centered
          width={720}
          className="post-tour-choice-modal"
        >
          <div className="px-1 py-2">
            <div className="rounded-[28px] border border-slate-200/70 bg-[linear-gradient(180deg,rgba(255,255,255,0.98)_0%,rgba(248,250,252,0.96)_100%)] p-5 shadow-[0_24px_70px_-44px_rgba(15,23,42,0.36)] dark:border-white/10 dark:bg-[linear-gradient(180deg,rgba(15,23,42,0.98)_0%,rgba(17,24,39,0.96)_100%)]">
              <div className="mb-5">
                <div className="text-xs font-semibold uppercase tracking-[0.16em] text-sky-600 dark:text-sky-300">
                  {t("onboarding.postTour.eyebrow") || "Next Step"}
                </div>
                <div className="mt-2 text-2xl font-semibold tracking-tight text-slate-950 dark:text-slate-50">
                  {t("onboarding.postTour.title") || "接下来想怎么继续？"}
                </div>
                <div className="mt-2 text-sm leading-6 text-slate-500 dark:text-slate-400">
                  {t("onboarding.postTour.subtitle") ||
                    "你已经认识了页面布局，可以选择继续学习、进入实战，或先自由探索。"}
                </div>
              </div>

              <div className="grid gap-3 md:grid-cols-3">
                <button
                  type="button"
                  onClick={openTrainingGuideFromChoice}
                  className="group rounded-[22px] border border-emerald-100 bg-emerald-50/70 p-4 text-left transition-all hover:-translate-y-0.5 hover:border-emerald-200 hover:bg-emerald-50 hover:shadow-[0_18px_38px_-30px_rgba(16,185,129,0.45)] dark:border-emerald-400/20 dark:bg-emerald-500/10"
                >
                  <div className="mb-4 flex h-10 w-10 items-center justify-center rounded-2xl bg-white text-emerald-600 shadow-sm dark:bg-slate-900">
                    <BookOpen className="h-5 w-5" />
                  </div>
                  <div className="text-base font-semibold text-slate-950 dark:text-slate-50">
                    {t("onboarding.postTour.trainingGuide.title") || "训练教程"}
                  </div>
                  <div className="mt-2 min-h-[42px] text-sm leading-6 text-slate-600 dark:text-slate-300">
                    {t("onboarding.postTour.trainingGuide.desc") ||
                      "先理解完整训练流程。"}
                  </div>
                </button>

                <button
                  type="button"
                  onClick={openPracticeFromChoice}
                  className="group rounded-[22px] border border-orange-100 bg-orange-50/70 p-4 text-left transition-all hover:-translate-y-0.5 hover:border-orange-200 hover:bg-orange-50 hover:shadow-[0_18px_38px_-30px_rgba(249,115,22,0.45)] dark:border-orange-400/20 dark:bg-orange-500/10"
                >
                  <div className="mb-4 flex h-10 w-10 items-center justify-center rounded-2xl bg-white text-orange-600 shadow-sm dark:bg-slate-900">
                    <Zap className="h-5 w-5" />
                  </div>
                  <div className="text-base font-semibold text-slate-950 dark:text-slate-50">
                    {t("onboarding.postTour.practice.title") || "实战演练"}
                  </div>
                  <div className="mt-2 min-h-[42px] text-sm leading-6 text-slate-600 dark:text-slate-300">
                    {t("onboarding.postTour.practice.desc") ||
                      "跟着示例一步步操作。"}
                  </div>
                </button>

                <button
                  type="button"
                  onClick={closePostTourChoice}
                  className="group rounded-[22px] border border-slate-200 bg-slate-50/80 p-4 text-left transition-all hover:-translate-y-0.5 hover:border-slate-300 hover:bg-white hover:shadow-[0_18px_38px_-32px_rgba(15,23,42,0.35)] dark:border-white/10 dark:bg-slate-800/70"
                >
                  <div className="mb-4 flex h-10 w-10 items-center justify-center rounded-2xl bg-white text-slate-600 shadow-sm dark:bg-slate-900 dark:text-slate-300">
                    <MousePointerClick className="h-5 w-5" />
                  </div>
                  <div className="text-base font-semibold text-slate-950 dark:text-slate-50">
                    {t("onboarding.postTour.explore.title") || "先自己看看"}
                  </div>
                  <div className="mt-2 min-h-[42px] text-sm leading-6 text-slate-600 dark:text-slate-300">
                    {t("onboarding.postTour.explore.desc") ||
                      "回到页面自由探索。"}
                  </div>
                </button>
              </div>
            </div>
          </div>
        </Modal>
        <Modal
          title={
            t("onboarding.postTrainingGuide.title") ||
            "已完成训练教程。是否进入实战演练？"
          }
          open={isPostTrainingGuideChoiceOpen}
          onCancel={() => setIsPostTrainingGuideChoiceOpen(false)}
          footer={null}
          centered
          width={480}
        >
          <div className="flex justify-end gap-3 py-2">
            <Button onClick={() => setIsPostTrainingGuideChoiceOpen(false)}>
              {t("onboarding.postTrainingGuide.explore") || "先自己看看"}
            </Button>
            <Button type="primary" onClick={openPracticeFromChoice}>
              {t("onboarding.postTrainingGuide.practice") || "进入实战演练"}
            </Button>
          </div>
        </Modal>
        <EnvironmentCheckDialog
          open={isEnvironmentCheckOpen}
          onOpenChange={setIsEnvironmentCheckOpen}
          containerName={environmentContainerName}
          result={environmentCheckResult}
          isChecking={checkEnvironmentMutation.isPending}
          errorMessage={environmentCheckError}
          onRunCheck={() => {
            void runEnvironmentCheck();
          }}
        />
        <Sheet
          open={isWorkflowConfigOpen}
          onOpenChange={setIsWorkflowConfigOpen}
        >
          <SheetContent
            side={isMobile ? "bottom" : "right"}
            className={
              isMobile
                ? "w-full max-w-full rounded-t-[24px] p-0"
                : "inset-y-4 right-4 h-auto w-[min(520px,calc(100%-2rem))] max-w-[520px] rounded-[28px] border border-slate-200/70 bg-[linear-gradient(180deg,rgba(255,255,255,0.97)_0%,rgba(248,250,252,0.99)_100%)] p-0 shadow-[0_28px_60px_-40px_rgba(15,23,42,0.3),0_18px_32px_-28px_rgba(15,23,42,0.2)] dark:border-white/10 dark:bg-[linear-gradient(180deg,rgba(15,23,42,0.97)_0%,rgba(17,24,39,0.97)_100%)]"
            }
          >
            <div className="flex h-full min-h-0 flex-col overflow-hidden">
              <SheetHeader className="border-b border-slate-200/70 px-5 py-4 text-left dark:border-white/10">
                <div className="flex items-center gap-2">
                  <span className="inline-flex h-8 w-8 items-center justify-center rounded-lg bg-violet-500/90 text-white dark:bg-violet-400 dark:text-violet-950">
                    <Workflow className="h-4 w-4" />
                  </span>
                  <div>
                    <SheetTitle className="text-base font-semibold text-slate-950 dark:text-slate-50">
                      {t("oneClickWorkflow.title") || "一键工作流"}
                    </SheetTitle>
                    <p className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">
                      {t("oneClickWorkflow.subtitle") ||
                        "选择数据集和评测集，生成命令后手动发送执行。"}
                    </p>
                  </div>
                </div>
              </SheetHeader>

              <div className="min-h-0 flex-1 space-y-4 overflow-y-auto px-5 py-4">
                <section className="space-y-2">
                  <div className="flex items-center justify-between gap-2">
                    <label className="text-xs font-semibold text-slate-700 dark:text-slate-200">
                      {t("oneClickWorkflow.dataset.label") || "训练数据集"}
                      <span className="ml-1 text-red-500">*</span>
                    </label>
                    <Tooltip
                      title={t("oneClickWorkflow.refreshCurrentRunNodeResources", {
                        defaultValue: "刷新当前 Run 节点资源",
                      })}
                    >
                      <Button
                        type="link"
                        size="small"
                        className="h-auto px-0 text-xs"
                        loading={isRefreshingWorkflowDatasets}
                        onClick={() => {
                          void refreshWorkflowDatasets();
                        }}
                      >
                        {isRefreshingWorkflowDatasets
                          ? t("oneClickWorkflow.dataset.refreshing") ||
                            "刷新中..."
                          : t("oneClickWorkflow.refreshCurrentRunNodeResources", {
                              defaultValue: "刷新当前 Run 节点资源",
                            })}
                      </Button>
                    </Tooltip>
                  </div>
                  <Select
                    showSearch
                    className="w-full"
                    getPopupContainer={(triggerNode) =>
                      triggerNode.parentElement || document.body
                    }
                    listHeight={240}
                    value={selectedWorkflowDatasetName}
                    options={workflowDatasetOptions}
                    placeholder={
                      t("oneClickWorkflow.dataset.placeholder") ||
                      "请选择训练数据集"
                    }
                    optionFilterProp="label"
                    onSelect={(value) => setSelectedWorkflowDatasetName(value)}
                    onChange={setSelectedWorkflowDatasetName}
                    notFoundContent={
                      hasQueriedDatasets
                        ? t("oneClickWorkflow.dataset.empty") ||
                          "暂无可用数据集"
                        : t("oneClickWorkflow.dataset.notQueried") ||
                          "请先查询数据集"
                    }
                  />
                  {selectedWorkflowDataset && (
                    <p className="text-xs text-slate-500 dark:text-slate-400">
                      {t("oneClickWorkflow.dataset.selected") || "已选择"}
                      {t("runpage.statusBar.labelSeparator")}
                      {selectedWorkflowDataset.name}
                      {selectedWorkflowDataset.type
                        ? ` · ${selectedWorkflowDataset.type.toUpperCase()}`
                        : ""}
                    </p>
                  )}
                </section>

                <section className="space-y-2">
                  <div className="flex items-center justify-between gap-2">
                    <label className="text-xs font-semibold text-slate-700 dark:text-slate-200">
                      {t("oneClickWorkflow.evaluation.label") || "评测集"}
                    </label>
                    <Tooltip
                      title={t("oneClickWorkflow.refreshCurrentRunNodeResources", {
                        defaultValue: "刷新当前 Run 节点资源",
                      })}
                    >
                      <Button
                        type="link"
                        size="small"
                        className="h-auto px-0 text-xs"
                        loading={isRefreshingWorkflowEvaluations}
                        onClick={() => {
                          void refreshWorkflowEvaluations();
                        }}
                      >
                        {isRefreshingWorkflowEvaluations
                          ? t("oneClickWorkflow.evaluation.refreshing", {
                              defaultValue: "刷新中...",
                            })
                          : t("oneClickWorkflow.refreshCurrentRunNodeResources", {
                              defaultValue: "刷新当前 Run 节点资源",
                            })}
                      </Button>
                    </Tooltip>
                  </div>
                  <Select
                    allowClear
                    showSearch
                    className="w-full"
                    getPopupContainer={(triggerNode) =>
                      triggerNode.parentElement || document.body
                    }
                    listHeight={240}
                    value={selectedWorkflowEvaluationName}
                    options={workflowEvaluationOptions}
                    placeholder={
                      t("oneClickWorkflow.evaluation.placeholder") ||
                      "可选；不选则由 Agent 默认使用 2024.json"
                    }
                    optionFilterProp="label"
                    onSelect={(value) =>
                      setSelectedWorkflowEvaluationName(value)
                    }
                    onChange={setSelectedWorkflowEvaluationName}
                    notFoundContent={
                      hasQueriedTests
                        ? t("oneClickWorkflow.evaluation.empty") ||
                          "暂无可用评测集"
                        : t("oneClickWorkflow.evaluation.notQueried") ||
                          "请先查询评测集"
                    }
                  />
                  <p className="text-xs text-slate-500 dark:text-slate-400">
                    {selectedWorkflowEvaluation
                      ? `${t("oneClickWorkflow.evaluation.selected") || "已选择"}${t("runpage.statusBar.labelSeparator")}${selectedWorkflowEvaluation.filename}`
                      : t("oneClickWorkflow.evaluation.defaultHint") ||
                        "未指定，默认使用 2024.json"}
                  </p>
                </section>

                <section className="rounded-xl border border-slate-200/70 bg-slate-50/80 p-3 dark:border-white/10 dark:bg-slate-900/60">
                  <div className="mb-2 text-xs font-semibold text-slate-700 dark:text-slate-200">
                    {t("oneClickWorkflow.preview") || "命令预览"}
                  </div>
                  <div className="min-h-[56px] rounded-lg border border-slate-200/70 bg-white px-3 py-2 text-xs leading-5 text-slate-700 dark:border-white/10 dark:bg-slate-950/70 dark:text-slate-200">
                    {workflowCommandPreview ||
                      t("oneClickWorkflow.previewEmpty") ||
                      "选择训练数据集后将生成命令"}
                  </div>
                </section>
              </div>

              <div className="flex items-center justify-end gap-2 border-t border-slate-200/70 px-5 py-4 dark:border-white/10">
                <Button onClick={() => setIsWorkflowConfigOpen(false)}>
                  {t("oneClickWorkflow.cancel") || "取消"}
                </Button>
                <Button
                  type="primary"
                  disabled={!selectedWorkflowDatasetName}
                  onClick={fillWorkflowStartCommand}
                >
                  {t("oneClickWorkflow.generate") || "生成命令"}
                </Button>
              </div>
            </div>
          </SheetContent>
        </Sheet>
        {/* 新手引导向导 */}
        <QuickStartWizard
          ref={wizardRef}
          open={isQuickStartWizardOpen}
          onClose={() => setIsQuickStartWizardOpen(false)}
          onTabChange={onTabChange}
          onSendCommand={(command) => {
            // 将命令填入 AsChat 输入框
            setAsChatInputText(command);
            setInputInlineHint("");
            setWizardCommandPending(true);
            setWizardCommandProgress(wizardProgress);
            setWizardResumeState("filled");
            setSendButtonHighlightToken((prev) => prev + 1);
            closeWizardAndRevealChatInput();
            // 提示用户手动点击发送
            messageApi.success(t("wizard.resume.commandFilledToast"));
          }}
          onOpenMetrics={() => setIsMetricsSheetOpen(true)}
          currentTab="datasets"
          queryState={wizardQueryState}
          datasets={wizardDatasets}
          selectedDataset={wizardSelectedDataset}
          onSelectDataset={setWizardSelectedDataset}
          onStepChange={handleWizardStepChange}
          requiredDatasetId="medical-example"
          inputRequests={inputRequests}
          runs={runs.map((r) => ({
            id: r.id,
            name: r.name,
            status: r.status,
            timestamp: r.timestamp,
          }))}
          onSelectRun={(runId) => {
            navigate(`/projects/${projectName}/runs/${runId}`, {
              replace: true,
            });
          }}
          focusOnLatestRun={focusOnLatestRun}
          setFocusOnLatestRun={setFocusOnLatestRun}
        />
      </Flex>
    </div>
  );
};

interface RunPageWithMetricsProps {
  systemOverviewData: SystemOverviewData | null;
  gpuInfo: GPUInfo[] | null;
  onRefreshGPUInfo: () => void;
  activeTab: "runs" | "overview" | "datasets" | "models" | "evaluation";
  onTabChange: (
    tab: "runs" | "overview" | "datasets" | "models" | "evaluation",
  ) => void;
  isMetricsSheetOpen: boolean;
  setIsMetricsSheetOpen: (open: boolean) => void;
  isInferenceSheetOpen: boolean;
  setIsInferenceSheetOpen: (open: boolean) => void;
  inferencePanelView: InferencePanelView;
  setInferencePanelView: (view: InferencePanelView) => void;
  // Dataset management props
  datasets: DatasetInfo[];
  isQueryingDatasets: boolean;
  datasetQueryError: boolean;
  datasetErrorMessage?: string;
  hasQueriedDatasets: boolean;
  onQueryDatasets: (containerName: string) => void;
  onRefreshDatasets?: (containerName: string) => void;
  datasetCacheMeta?: ManagementCacheMeta | null;
  onUpload: () => void;
  onDownload: (dataset: DatasetInfo) => void;
  onDeleteDataset: (dataset: DatasetInfo) => Promise<void>;
  onUseDatasetForTraining: (dataset: DatasetInfo) => void;
  onUseDatasetForPreprocess: (dataset: DatasetInfo) => void;
  onUseEvaluationForBenchmark: (testName: string) => void;
  onLoadDatasetPreviews: (dataset: DatasetInfo) => Promise<void>;
  isUploading: boolean;
  downloadingId: string | null;
  deletingDatasetId: string | null;
  // Model management props
  models: ModelInfo[];
  isQueryingModels: boolean;
  modelQueryError: boolean;
  modelErrorMessage?: string;
  hasQueriedModels: boolean;
  onQueryModels: (containerName: string) => void;
  onRefreshModels?: (containerName: string) => void;
  modelCacheMeta?: ManagementCacheMeta | null;
  onDeleteModel: (model: ModelInfo) => Promise<void>;
  deletingModelId: string | null;
  // Evaluation management props
  tests: MedicalTestFile[];
  isQueryingTests: boolean;
  testQueryError: boolean;
  testErrorMessage?: string;
  hasQueriedTests: boolean;
  onQueryTests: (containerName: string) => void;
  onRefreshTests?: (containerName: string) => void;
  testCacheMeta?: ManagementCacheMeta | null;
  onUploadTest: () => void;
  onDownloadTest: (testName: string, test?: MedicalTestFile) => void;
  onDeleteTest: (testName: string, test?: MedicalTestFile) => Promise<void>;
  isUploadingTest: boolean;
  downloadingTestId: string | null;
  deletingTestId: string | null;
  // Evaluation results props
  evaluationResults: EvaluationResult[];
  isQueryingEvaluationResults: boolean;
  evaluationResultQueryError: boolean;
  evaluationResultErrorMessage?: string;
  hasQueriedEvaluationResults: boolean;
  onQueryEvaluationResults: (
    containerName: string,
  ) => Promise<EvaluationResult[]>;
  onRefreshEvaluationResults?: (
    containerName: string,
  ) => Promise<EvaluationResult[]>;
  evaluationResultCacheMeta?: ManagementCacheMeta | null;
  onQueryEvaluationResultsZero?: () => Promise<EvaluationResult[]>;
  onDownloadEvaluationResult?: (
    folderPath: string,
    filename: string,
    result?: EvaluationResult,
  ) => Promise<void>;
  downloadingResultId?: string | null;
  onDeleteEvaluationResult?: (
    folderPath: string,
    result?: EvaluationResult,
  ) => Promise<void>;
  deletingResultId?: string | null;
  // Upload modals props
  uploadModalOpen: boolean;
  setUploadModalOpen: (open: boolean) => void;
  handleUpload: (params: {
    containerName: string;
    datasetType: "raw" | "sft" | "dpo";
    datasetName: string;
    file: File;
  }) => Promise<void>;
  uploadProgress: number;
  evaluationUploadModalOpen: boolean;
  setEvaluationUploadModalOpen: (open: boolean) => void;
  handleUploadTest: (params: {
    containerName: string;
    testType: string;
    filename: string;
    file: File;
  }) => Promise<void>;
  evaluationUploadProgress: number;
  // Wizard props
  isQuickStartWizardOpen: boolean;
  setIsQuickStartWizardOpen: (open: boolean) => void;
  wizardQueryState: "idle" | "querying" | "completed";
  setWizardQueryState: (state: "idle" | "querying" | "completed") => void;
  wizardDatasets: DatasetInfo[];
  setWizardDatasets: (datasets: DatasetInfo[]) => void;
  wizardSelectedDataset: DatasetInfo | null;
  setWizardSelectedDataset: (dataset: DatasetInfo | null) => void;
  focusOnLatestRun: boolean;
  setFocusOnLatestRun: (focus: boolean) => void;
  setInputTextRef?: React.MutableRefObject<
    ((text: string, hint?: string) => void) | undefined
  >;
  randomUsername: string;
  setRandomUsername: React.Dispatch<React.SetStateAction<string>>;
  chatSessionId: string;
  fallbackUsername: string;
  clearContextAfter?: string | null;
  onClearContext: () => void;
  onRuntimeResourceGroupIdChange?: (resourceGroupId?: string) => void;
}

const RunPageWithMetrics = ({
  systemOverviewData,
  gpuInfo,
  onRefreshGPUInfo,
  activeTab,
  onTabChange,
  isMetricsSheetOpen,
  setIsMetricsSheetOpen,
  isInferenceSheetOpen,
  setIsInferenceSheetOpen,
  inferencePanelView,
  setInferencePanelView,
  // Dataset management
  datasets,
  isQueryingDatasets,
  datasetQueryError,
  datasetErrorMessage,
  hasQueriedDatasets,
  onQueryDatasets,
  onRefreshDatasets,
  datasetCacheMeta,
  onUpload,
  onDownload,
  onDeleteDataset,
  onUseDatasetForTraining,
  onUseDatasetForPreprocess,
  onUseEvaluationForBenchmark,
  onLoadDatasetPreviews,
  isUploading,
  downloadingId,
  deletingDatasetId,
  // Model management
  models,
  isQueryingModels,
  modelQueryError,
  modelErrorMessage,
  hasQueriedModels,
  onQueryModels,
  onRefreshModels,
  modelCacheMeta,
  onDeleteModel,
  deletingModelId,
  // Evaluation management
  tests,
  isQueryingTests,
  testQueryError,
  testErrorMessage,
  hasQueriedTests,
  onQueryTests,
  onRefreshTests,
  testCacheMeta,
  onUploadTest,
  onDownloadTest,
  onDeleteTest,
  isUploadingTest,
  downloadingTestId,
  deletingTestId,
  // Evaluation results
  evaluationResults,
  isQueryingEvaluationResults,
  evaluationResultQueryError,
  evaluationResultErrorMessage,
  hasQueriedEvaluationResults,
  onQueryEvaluationResults,
  onRefreshEvaluationResults,
  evaluationResultCacheMeta,
  onQueryEvaluationResultsZero,
  onDownloadEvaluationResult,
  downloadingResultId,
  onDeleteEvaluationResult,
  deletingResultId,
  // Upload modals
  uploadModalOpen,
  setUploadModalOpen,
  handleUpload,
  uploadProgress,
  evaluationUploadModalOpen,
  setEvaluationUploadModalOpen,
  handleUploadTest,
  evaluationUploadProgress,
  // Wizard
  isQuickStartWizardOpen,
  setIsQuickStartWizardOpen,
  wizardQueryState,
  setWizardQueryState,
  wizardDatasets,
  setWizardDatasets,
  wizardSelectedDataset,
  setWizardSelectedDataset,
  focusOnLatestRun,
  setFocusOnLatestRun,
  setInputTextRef,
  randomUsername,
  setRandomUsername,
  chatSessionId,
  fallbackUsername,
  clearContextAfter,
  onClearContext,
  onRuntimeResourceGroupIdChange,
}: RunPageWithMetricsProps) => {
  const isMobile = useIsMobile();
  const { t } = useTranslation();
  const { projectName } = useParams<{ projectName: string }>();
  const navigate = useNavigate();
  const { replies, runData } = useRunRoom();
  const { runs } = useProjectRoom();
  const { isRunPagePanelOpen, setRunPagePanelOpen } = useStudioSidebar();
  const { defaultContainerName, defaultEvaluateContainerName } =
    useEnvironmentConfig();
  const { isAdmin, user } = useAuth();
  const { nodeId: selectedResourceNodeId } = useResourceNodeSelection();
  const runNodeId = runData?.nodeId?.trim();
  const fallbackRunNodeId =
    user?.assignedNodeId?.trim() ||
    (selectedResourceNodeId !== "all" ? selectedResourceNodeId.trim() : "");
  const currentRunNodeId =
    runNodeId && runNodeId !== "unknown"
      ? runNodeId
      : fallbackRunNodeId || undefined;
  const [runtimeResourceContext, setRuntimeResourceContext] = useState({
    nodeId: currentRunNodeId,
    resourceGroupId: undefined as string | undefined,
    trainingContainerName: defaultContainerName,
    evaluationContainerName: defaultEvaluateContainerName,
  });
  useEffect(() => {
    setRuntimeResourceContext({
      nodeId: currentRunNodeId,
      resourceGroupId: undefined,
      trainingContainerName: defaultContainerName,
      evaluationContainerName: defaultEvaluateContainerName,
    });
    onRuntimeResourceGroupIdChange?.(undefined);
  }, [
    currentRunNodeId,
    defaultContainerName,
    defaultEvaluateContainerName,
    onRuntimeResourceGroupIdChange,
  ]);
  const currentTrainingContainerName = runtimeResourceContext.trainingContainerName;
  const currentEvaluationContainerName =
    runtimeResourceContext.evaluationContainerName;
  const handleRuntimeResourceContextChange = useCallback(
    (context: {
      nodeId?: string;
      resourceGroupId?: string;
      trainingContainerName: string;
      evaluationContainerName: string;
    }) => {
      setRuntimeResourceContext(context);
      onRuntimeResourceGroupIdChange?.(context.resourceGroupId);
    },
    [onRuntimeResourceGroupIdChange],
  );

  // 用于存储 AI 分析回调的 ref
  const askAIRef = useRef<((blocks: ContentBlocks) => void) | undefined>(
    undefined,
  );

  // 用于跟踪当前输入请求状态（传递给TrainingMetricsPanel控制询问AI按钮）
  const [currentInputRequest, setCurrentInputRequest] =
    useState<InputRequestData | null>(null);
  const isInputDisabled = !currentInputRequest;
  const [combinedRepliesForPanels, setCombinedRepliesForPanels] = useState<
    Reply[]
  >([]);
  const [silentMonitorReply, setSilentMonitorReply] = useState<Reply | null>(
    null,
  );
  const [trainingMetricsCacheKey, setTrainingMetricsCacheKey] = useState<
    string | null
  >(null);
  const trainingMetricsCacheKeyRef = useRef<string | null>(null);
  const [trainingMetricsCacheByKey, setTrainingMetricsCacheByKey] = useState<
    Map<string, TrainingMetricsCacheSnapshot>
  >(() => new Map());
  const [silentMonitorStatus, setSilentMonitorStatus] =
    useState<SilentMonitorStatus>({ isQuerying: false });
  const [monitorTrainingCommand, setMonitorTrainingCommand] = useState<
    (() => void) | null
  >(null);

  const handleSilentMonitorStatusChange = useCallback(
    (status: SilentMonitorStatus) => {
      setSilentMonitorStatus((previous) => ({
        ...previous,
        ...status,
      }));
    },
    [],
  );

  const handleSilentMonitorCacheKeyChange = useCallback(
    (cacheKey: string | null) => {
      if (trainingMetricsCacheKeyRef.current !== cacheKey) {
        trainingMetricsCacheKeyRef.current = cacheKey;
        setSilentMonitorReply(null);
      }
      setTrainingMetricsCacheKey(cacheKey);
    },
    [],
  );

  const handleMonitorTrainingCommandChange = useCallback(
    (handler: (() => void) | null) => {
      setMonitorTrainingCommand(handler ? () => handler : null);
    },
    [],
  );

  const handleTrainingMetricsCacheChange = useCallback(
    (snapshot: TrainingMetricsCacheSnapshot) => {
      if (!trainingMetricsCacheKey) {
        return;
      }
      setTrainingMetricsCacheByKey((previous) => {
        const next = new Map(previous);
        next.set(trainingMetricsCacheKey, snapshot);
        return next;
      });
    },
    [trainingMetricsCacheKey],
  );

  const guardedUseDatasetForTraining = useCallback(
    (dataset: DatasetInfo) => {
      const datasetNodeId = dataset.nodeId?.trim();
      const runNodeId = runData?.nodeId?.trim();
      const fallbackNodeId =
        user?.assignedNodeId?.trim() ||
        (selectedResourceNodeId !== "all" ? selectedResourceNodeId.trim() : "");
      const currentNodeId =
        runNodeId && runNodeId !== "unknown" ? runNodeId : fallbackNodeId;
      const datasetContainerName = dataset.containerName?.trim();
      const currentContainerName = (currentTrainingContainerName || "").trim();

      if (!datasetNodeId) {
        message.warning(
          t("dataset.train.missingNode") ||
            "数据集缺少节点信息，请刷新数据集列表后重试",
        );
        return;
      }

      if (!currentNodeId || currentNodeId === "unknown") {
        message.warning(
          t("dataset.train.missingRunNode") ||
            "当前运行实例缺少节点信息，不能直接启动训练",
        );
        return;
      }

      if (datasetNodeId !== currentNodeId) {
        const targetRun = runs.find(
          (run) =>
            run.nodeId === datasetNodeId &&
            (run.status === Status.RUNNING || run.status === Status.PENDING),
        );
        const datasetNodeLabel =
          dataset.nodeName || datasetNodeId || t("resourceNode.unknown");
        const currentNodeLabel = currentNodeId || t("resourceNode.unknown");

        Modal.confirm({
          title: t("dataset.train.crossNodeTitle") || "数据集不在当前运行节点",
          content: (
            <div className="space-y-2 text-sm">
              <p>
                {t("dataset.train.crossNodeDesc", {
                  dataset: dataset.name,
                  datasetNode: datasetNodeLabel,
                  runNode: currentNodeLabel,
                }) ||
                  `数据集 ${dataset.name} 位于 ${datasetNodeLabel}，当前运行实例位于 ${currentNodeLabel}。训练请求会由当前运行节点执行，因此不能直接使用跨节点数据集。`}
              </p>
              <p>
                {targetRun
                  ? t("dataset.train.crossNodeSwitchHint") ||
                    "可以切换到该节点当前可运行的 Run 后再启动训练。"
                  : t("dataset.train.crossNodeNoRunHint") ||
                    "该节点当前没有可运行的 Run，请先启动对应 Runtime/run。"}
              </p>
            </div>
          ),
          okText: targetRun
            ? t("dataset.train.switchRun") || "切换到该节点 Run"
            : t("dataset.train.viewRuns") || "查看运行列表",
          cancelText: t("common.cancel") || "取消",
          onOk: () => {
            if (targetRun) {
              navigate(`/projects/${projectName}/runs/${targetRun.id}`, {
                replace: true,
              });
              setRunPagePanelOpen(false);
              return;
            }
            onTabChange("runs");
            setRunPagePanelOpen(true);
          },
        });
        return;
      }

      if (
        datasetContainerName &&
        currentContainerName &&
        datasetContainerName !== currentContainerName
      ) {
        Modal.warning({
          title:
            t("dataset.train.crossContainerTitle") || "数据集不在当前训练容器",
          content:
            t("dataset.train.crossContainerDesc", {
              dataset: dataset.name,
              datasetContainer: datasetContainerName,
              runContainer: currentContainerName,
            }) ||
            `数据集 ${dataset.name} 来自容器 ${datasetContainerName}，当前运行实例使用训练容器 ${currentContainerName}。请先将数据集同步到当前训练容器后再启动训练。`,
          okText: t("common.ok") || "知道了",
        });
        return;
      }

      onUseDatasetForTraining(dataset);
    },
    [
      defaultContainerName,
      navigate,
      onTabChange,
      onUseDatasetForTraining,
      projectName,
      runData?.nodeId,
      selectedResourceNodeId,
      runs,
      setRunPagePanelOpen,
      t,
      user?.assignedNodeId,
    ],
  );

  const guardedUseDatasetForPreprocess = useCallback(
    (dataset: DatasetInfo) => {
      const datasetNodeId = dataset.nodeId?.trim();
      const runNodeId = runData?.nodeId?.trim();
      const fallbackNodeId =
        user?.assignedNodeId?.trim() ||
        (selectedResourceNodeId !== "all" ? selectedResourceNodeId.trim() : "");
      const currentNodeId =
        runNodeId && runNodeId !== "unknown" ? runNodeId : fallbackNodeId;
      const datasetContainerName = dataset.containerName?.trim();
      const currentContainerName = (currentTrainingContainerName || "").trim();

      if (!datasetNodeId) {
        message.warning(
          t("dataset.train.missingNode") ||
            "数据集缺少节点信息，请刷新数据集列表后重试",
        );
        return;
      }

      if (!currentNodeId || currentNodeId === "unknown") {
        message.warning(
          t("dataset.train.missingRunNode") ||
            "当前运行实例缺少节点信息，不能直接启动训练",
        );
        return;
      }

      if (datasetNodeId !== currentNodeId) {
        const targetRun = runs.find(
          (run) =>
            run.nodeId === datasetNodeId &&
            (run.status === Status.RUNNING || run.status === Status.PENDING),
        );
        const datasetNodeLabel =
          dataset.nodeName || datasetNodeId || t("resourceNode.unknown");
        const currentNodeLabel = currentNodeId || t("resourceNode.unknown");

        Modal.confirm({
          title: t("dataset.train.crossNodeTitle") || "数据集不在当前运行节点",
          content: (
            <div className="space-y-2 text-sm">
              <p>
                {t("dataset.train.crossNodeDesc", {
                  dataset: dataset.name,
                  datasetNode: datasetNodeLabel,
                  runNode: currentNodeLabel,
                }) ||
                  `数据集 ${dataset.name} 位于 ${datasetNodeLabel}，当前运行实例位于 ${currentNodeLabel}。训练请求会由当前运行节点执行，因此不能直接使用跨节点数据集。`}
              </p>
              <p>
                {targetRun
                  ? t("dataset.train.crossNodeSwitchHint") ||
                    "可以切换到该节点当前可运行的 Run 后再启动训练。"
                  : t("dataset.train.crossNodeNoRunHint") ||
                    "该节点当前没有可运行的 Run，请先启动对应 Runtime/run。"}
              </p>
            </div>
          ),
          okText: targetRun
            ? t("dataset.train.switchRun") || "切换到该节点 Run"
            : t("dataset.train.viewRuns") || "查看运行列表",
          cancelText: t("common.cancel") || "取消",
          onOk: () => {
            if (targetRun) {
              navigate(`/projects/${projectName}/runs/${targetRun.id}`, {
                replace: true,
              });
              setRunPagePanelOpen(false);
              return;
            }
            onTabChange("runs");
            setRunPagePanelOpen(true);
          },
        });
        return;
      }

      if (
        datasetContainerName &&
        currentContainerName &&
        datasetContainerName !== currentContainerName
      ) {
        Modal.warning({
          title:
            t("dataset.train.crossContainerTitle") || "数据集不在当前训练容器",
          content:
            t("dataset.train.crossContainerDesc", {
              dataset: dataset.name,
              datasetContainer: datasetContainerName,
              runContainer: currentContainerName,
            }) ||
            `数据集 ${dataset.name} 来自容器 ${datasetContainerName}，当前运行实例使用训练容器 ${currentContainerName}。请先将数据集同步到当前训练容器后再启动训练。`,
          okText: t("common.ok") || "知道了",
        });
        return;
      }

      onUseDatasetForPreprocess(dataset);
    },
    [
      currentTrainingContainerName,
      navigate,
      onTabChange,
      onUseDatasetForPreprocess,
      projectName,
      runData?.nodeId,
      selectedResourceNodeId,
      runs,
      setRunPagePanelOpen,
      t,
      user?.assignedNodeId,
    ],
  );

  // 从 content 中提取文本（处理 string 或 ContentBlocks）
  const extractTextFromContent = (content: ContentType): string => {
    if (typeof content === "string") {
      return content;
    }
    // 如果是 ContentBlocks 数组，提取第一个 text block 的文本
    if (Array.isArray(content)) {
      const textBlock = content.find(
        (b): b is ContentBlock & { text: string } =>
          b.type === "text" && "text" in b,
      );
      return textBlock?.text || "";
    }
    return "";
  };

  // 过滤只显示当前用户相关的消息
  const filteredReplies = useMemo(() => {
    const displayUsername = randomUsername;
    if (!displayUsername) {
      return replies;
    }

    const clearAfterTime = clearContextAfter
      ? Date.parse(clearContextAfter)
      : 0;

    return replies.filter((reply: Reply) => {
      if (clearAfterTime && getReplyTime(reply) < clearAfterTime) {
        return false;
      }

      // 检查是否有消息
      if (!reply.messages || reply.messages.length === 0) {
        return false;
      }

      const firstMsg = reply.messages[0];
      const msgRole = firstMsg.role;

      // 1. System 消息：不显示
      if (msgRole === "system") {
        return false;
      }

      // 2. 用户消息：检查 content 是否以 [User-XXXX] 开头
      if (isUserRole(msgRole)) {
        const contentText = extractTextFromContent(firstMsg.content);
        return messageMatchesContextUsername(
          contentText,
          firstMsg.name,
          firstMsg.metadata,
          displayUsername,
        );
      }

      // 3. Agent 回复：检查 msg.name 或 metadata 是否属于当前会话用户名
      if (msgRole === "assistant") {
        return messageMatchesContextUsername(
          extractTextFromContent(firstMsg.content),
          firstMsg.name,
          firstMsg.metadata,
          displayUsername,
        );
      }

      return false;
    });
  }, [replies, randomUsername, user, clearContextAfter]);
  const isModelReplying = useMemo(() => {
    if (runData?.status !== Status.RUNNING || filteredReplies.length === 0) {
      return false;
    }

    const lastVisibleReply = filteredReplies[filteredReplies.length - 1];
    const firstMessage = lastVisibleReply.messages?.[0];
    return isUserRole(firstMessage?.role);
  }, [filteredReplies, runData?.status]);
  const isCommandFillDisabled = isModelReplying;
  const commandFillDisabledHint = isModelReplying
    ? t("placeholder.input-model-working", {
        defaultValue: "后台模型正在处理，请等待当前回复完成后再填入命令。",
      })
    : "";
  const panelReplies = useMemo(() => {
    const baseReplies =
      combinedRepliesForPanels.length > 0
        ? combinedRepliesForPanels
        : filteredReplies;
    return silentMonitorReply
      ? [...baseReplies, silentMonitorReply]
      : baseReplies;
  }, [combinedRepliesForPanels, filteredReplies, silentMonitorReply]);
  const trainingMetricsInitialCache = useMemo(
    () =>
      trainingMetricsCacheKey
        ? (trainingMetricsCacheByKey.get(trainingMetricsCacheKey) ?? null)
        : null,
    [trainingMetricsCacheByKey, trainingMetricsCacheKey],
  );

  const layoutClassName = isMobile
    ? "h-screen w-full flex flex-col bg-muted/20 dark:bg-slate-950"
    : "relative h-screen w-full overflow-hidden bg-muted/20 dark:bg-slate-950";
  const contentWrapperClassName = isMobile
    ? "overflow-hidden min-h-0 flex-1 dark:bg-slate-950"
    : "overflow-hidden h-full min-w-0 bg-muted/30 dark:bg-slate-900/80";
  const panelSheetClassName = isMobile
    ? "w-full max-w-full p-0 rounded-t-[24px]"
    : "inset-y-4 right-4 h-auto w-[min(600px,calc(100%-2rem))] max-w-[600px] rounded-[28px] border border-slate-200/70 bg-[linear-gradient(180deg,rgba(255,255,255,0.96)_0%,rgba(248,250,252,0.98)_100%)] p-0 shadow-[0_28px_60px_-40px_rgba(15,23,42,0.3),0_18px_32px_-28px_rgba(15,23,42,0.2)] data-[state=closed]:duration-300 data-[state=open]:duration-300 dark:border-white/10 dark:bg-[linear-gradient(180deg,rgba(15,23,42,0.96)_0%,rgba(17,24,39,0.96)_100%)] dark:shadow-[0_28px_60px_-40px_rgba(2,6,23,0.86),0_18px_32px_-28px_rgba(2,6,23,0.72)]";

  return (
    <div className={layoutClassName}>
      <div className={contentWrapperClassName}>
        <RunContentPage
          isMetricsSheetOpen={isMetricsSheetOpen}
          setIsMetricsSheetOpen={setIsMetricsSheetOpen}
          setIsInferenceSheetOpen={setIsInferenceSheetOpen}
          setInferencePanelView={setInferencePanelView}
          randomUsername={randomUsername}
          setRandomUsername={setRandomUsername}
          filteredReplies={filteredReplies}
          onAskAIRef={askAIRef}
          setInputTextRef={setInputTextRef}
          onCurrentInputRequestChange={setCurrentInputRequest}
          onCombinedRepliesChange={setCombinedRepliesForPanels}
          onSilentMonitorReplyChange={setSilentMonitorReply}
          onSilentMonitorCacheKeyChange={handleSilentMonitorCacheKeyChange}
          onSilentMonitorStatusChange={handleSilentMonitorStatusChange}
          onMonitorTrainingCommandChange={handleMonitorTrainingCommandChange}
          onRuntimeResourceContextChange={handleRuntimeResourceContextChange}
          onTabChange={onTabChange}
          onQueryDatasets={async () => {
            const containerName = currentTrainingContainerName;
            return await onQueryDatasets(containerName);
          }}
          onRefreshDatasets={async (containerOverride?: string) => {
            const containerName =
              (containerOverride ?? currentTrainingContainerName).trim() ||
              currentTrainingContainerName;
            return await (onRefreshDatasets
              ? onRefreshDatasets(containerName)
              : onQueryDatasets(containerName));
          }}
          onQueryModels={async () => {
            const containerName = currentTrainingContainerName;
            return await onQueryModels(containerName);
          }}
          onRefreshModels={async (containerOverride?: string) => {
            const containerName =
              (containerOverride ?? currentTrainingContainerName).trim() ||
              currentTrainingContainerName;
            return await (onRefreshModels
              ? onRefreshModels(containerName)
              : onQueryModels(containerName));
          }}
          onQueryTests={async () => {
            const containerName = currentEvaluationContainerName;
            return await onQueryTests(containerName);
          }}
          onRefreshTests={async () => {
            const containerName = currentEvaluationContainerName;
            return await (onRefreshTests
              ? onRefreshTests(containerName)
              : onQueryTests(containerName));
          }}
          onDownloadDataset={(name) => {
            const dataset = datasets.find((item) => item.name === name);
            if (!dataset) {
              return Promise.reject(new Error(`未找到数据集: ${name}`));
            }
            return Promise.resolve(onDownload(dataset));
          }}
          onUseDatasetForTraining={guardedUseDatasetForTraining}
          onUseDatasetForPreprocess={guardedUseDatasetForPreprocess}
          onUseEvaluationForBenchmark={onUseEvaluationForBenchmark}
          onDownloadTest={(name, test) => {
            onDownloadTest(name, test);
          }}
          onUpload={onUpload}
          onUploadTest={onUploadTest}
          datasets={datasets}
          models={models}
          tests={tests}
          systemOverviewData={systemOverviewData}
          gpuInfo={gpuInfo}
          isQueryingDatasets={isQueryingDatasets}
          hasQueriedDatasets={hasQueriedDatasets}
          hasQueriedModels={hasQueriedModels}
          hasQueriedTests={hasQueriedTests}
          // 评测结果相关
          evaluationResults={evaluationResults}
          isQueryingEvaluationResults={isQueryingEvaluationResults}
          evaluationResultQueryError={evaluationResultQueryError}
          evaluationResultErrorMessage={evaluationResultErrorMessage}
          hasQueriedEvaluationResults={hasQueriedEvaluationResults}
          onQueryEvaluationResults={onQueryEvaluationResults}
          onRefreshEvaluationResults={async () => {
            const containerName = defaultEvaluateContainerName;
            return await (onRefreshEvaluationResults
              ? onRefreshEvaluationResults(containerName)
              : onQueryEvaluationResults(containerName));
          }}
          onQueryEvaluationResultsZero={onQueryEvaluationResultsZero}
          onDownloadEvaluationResult={onDownloadEvaluationResult}
          downloadingResultId={downloadingResultId}
          onDeleteEvaluationResult={
            isAdmin ? onDeleteEvaluationResult : undefined
          }
          deletingResultId={deletingResultId}
          isQuickStartWizardOpen={isQuickStartWizardOpen}
          setIsQuickStartWizardOpen={setIsQuickStartWizardOpen}
          wizardQueryState={wizardQueryState}
          setWizardQueryState={setWizardQueryState}
          wizardDatasets={wizardDatasets}
          setWizardDatasets={setWizardDatasets}
          wizardSelectedDataset={wizardSelectedDataset}
          setWizardSelectedDataset={setWizardSelectedDataset}
          focusOnLatestRun={focusOnLatestRun}
          setFocusOnLatestRun={setFocusOnLatestRun}
          onClearContext={onClearContext}
          chatSessionId={chatSessionId}
          fallbackUsername={fallbackUsername}
        />
      </div>

      {isRunPagePanelOpen && (
        <>
          <button
            type="button"
            aria-label="Close management panel"
            onClick={() => setRunPagePanelOpen(false)}
            className="absolute inset-0 z-10 hidden bg-slate-900/8 backdrop-blur-[1px] md:block"
          />
          <div className="absolute inset-y-4 left-4 z-20 w-[min(360px,calc(100%-2rem))] max-w-[360px]">
            <ProjectRunSider
              systemOverviewData={systemOverviewData}
              gpuInfo={gpuInfo}
              onRefreshGPUInfo={onRefreshGPUInfo}
              activeTab={activeTab}
              onTabChange={onTabChange}
              onClose={() => setRunPagePanelOpen(false)}
              onRunClick={(runId) =>
                navigate(`/projects/${projectName}/runs/${runId}`, {
                  replace: true,
                })
              }
              datasets={datasets}
              isQueryingDatasets={isQueryingDatasets}
              datasetQueryError={datasetQueryError}
              datasetErrorMessage={datasetErrorMessage}
              hasQueriedDatasets={hasQueriedDatasets}
              onQueryDatasets={onQueryDatasets}
              onRefreshDatasets={onRefreshDatasets}
              datasetCacheMeta={datasetCacheMeta}
              onUpload={onUpload}
              onDownload={onDownload}
              onDeleteDataset={onDeleteDataset}
              onUseDatasetForTraining={guardedUseDatasetForTraining}
              onUseDatasetForPreprocess={guardedUseDatasetForPreprocess}
              onUseEvaluationForBenchmark={onUseEvaluationForBenchmark}
              onLoadDatasetPreviews={onLoadDatasetPreviews}
              isInputDisabled={isCommandFillDisabled}
              inputDisabledHint={commandFillDisabledHint}
              isUploading={isUploading}
              downloadingId={downloadingId}
              deletingDatasetId={deletingDatasetId}
              models={models}
              isQueryingModels={isQueryingModels}
              modelQueryError={modelQueryError}
              modelErrorMessage={modelErrorMessage}
              hasQueriedModels={hasQueriedModels}
              onQueryModels={onQueryModels}
              onRefreshModels={onRefreshModels}
              modelCacheMeta={modelCacheMeta}
              onDeleteModel={onDeleteModel}
              deletingModelId={deletingModelId}
              tests={tests}
              isQueryingTests={isQueryingTests}
              testQueryError={testQueryError}
              testErrorMessage={testErrorMessage}
              hasQueriedTests={hasQueriedTests}
              onQueryTests={onQueryTests}
              onRefreshTests={onRefreshTests}
              testCacheMeta={testCacheMeta}
              onUploadTest={onUploadTest}
              onDownloadTest={onDownloadTest}
              onDeleteTest={onDeleteTest}
              isUploadingTest={isUploadingTest}
              downloadingTestId={downloadingTestId}
              deletingTestId={deletingTestId}
              evaluationResults={evaluationResults}
              isQueryingEvaluationResults={isQueryingEvaluationResults}
              evaluationResultQueryError={evaluationResultQueryError}
              evaluationResultErrorMessage={evaluationResultErrorMessage}
              hasQueriedEvaluationResults={hasQueriedEvaluationResults}
              onQueryEvaluationResults={onQueryEvaluationResults}
              onRefreshEvaluationResults={onRefreshEvaluationResults}
              evaluationResultCacheMeta={evaluationResultCacheMeta}
              onQueryEvaluationResultsZero={onQueryEvaluationResultsZero}
              onDownloadEvaluationResult={onDownloadEvaluationResult}
              downloadingResultId={downloadingResultId}
              onDeleteEvaluationResult={
                isAdmin ? onDeleteEvaluationResult : undefined
              }
              deletingResultId={deletingResultId}
              currentRunNodeId={currentRunNodeId}
              currentTrainingContainerName={currentTrainingContainerName}
              currentEvaluationContainerName={currentEvaluationContainerName}
              focusOnLatestRun={focusOnLatestRun}
            />
          </div>
        </>
      )}

      {/* Sheet 弹窗: 训练指标面板 */}
      <Sheet open={isMetricsSheetOpen} onOpenChange={setIsMetricsSheetOpen}>
        <SheetContent
          side={isMobile ? "bottom" : "right"}
          className={panelSheetClassName}
        >
          <TrainingMetricsPanel
            onClose={() => setIsMetricsSheetOpen(false)}
            replies={panelReplies}
            cacheKey={trainingMetricsCacheKey ?? undefined}
            initialCache={trainingMetricsInitialCache}
            onCacheChange={handleTrainingMetricsCacheChange}
            onAskAI={askAIRef.current}
            onMonitorTraining={monitorTrainingCommand ?? undefined}
            monitorStatus={silentMonitorStatus}
            isInputDisabled={isInputDisabled}
          />
        </SheetContent>
      </Sheet>

      {/* Sheet 弹窗: 推理服务面板 */}
      <Sheet open={isInferenceSheetOpen} onOpenChange={setIsInferenceSheetOpen}>
        <SheetContent
          side={isMobile ? "bottom" : "right"}
          className={panelSheetClassName}
        >
          <InferenceServicePanel
            onClose={() => setIsInferenceSheetOpen(false)}
            replies={
              combinedRepliesForPanels.length > 0
                ? combinedRepliesForPanels
                : filteredReplies
            }
            onAskAI={askAIRef.current}
            isInputDisabled={isInputDisabled}
            view={inferencePanelView}
            isAdmin={isAdmin}
          />
        </SheetContent>
      </Sheet>

      {/* 数据集上传弹窗 */}
      <DatasetUploadModal
        open={uploadModalOpen}
        onCancel={() => setUploadModalOpen(false)}
        onUpload={handleUpload}
        isUploading={isUploading}
        uploadProgress={uploadProgress}
      />

      {/* 评测文件上传弹窗 */}
      <EvaluationUploadModal
        open={evaluationUploadModalOpen}
        onCancel={() => setEvaluationUploadModalOpen(false)}
        onUpload={handleUploadTest}
        isUploading={isUploadingTest}
        uploadProgress={evaluationUploadProgress}
      />
    </div>
  );
};

const RunPage = () => {
  const isMobile = useIsMobile();
  const { t } = useTranslation();
  const { projectName } = useParams<{ projectName: string }>();
  const navigate = useNavigate();
  const location = useLocation();
  const socket = useSocket();
  const {
    runPageSection,
    isRunPagePanelOpen,
    setRunPagePanelOpen,
    setRunPageSection,
  } = useStudioSidebar();
  const { defaultContainerName, defaultEvaluateContainerName } =
    useEnvironmentConfig();
  const { isAdmin, user } = useAuth();
  const activeTab = runPageSection;

  const [systemOverviewData, setSystemOverviewData] =
    useState<SystemOverviewData | null>(null);
  const systemOverviewDataRef = useRef(systemOverviewData);
  useEffect(() => {
    systemOverviewDataRef.current = systemOverviewData;
  }, [systemOverviewData]);

  const [gpuInfo, setGPUInfo] = useState<GPUInfo[] | null>(null);
  const gpuInfoRef = useRef(gpuInfo);
  useEffect(() => {
    gpuInfoRef.current = gpuInfo;
  }, [gpuInfo]);

  const [isMetricsSheetOpen, setIsMetricsSheetOpen] = useState(false);
  const [isInferenceSheetOpen, setIsInferenceSheetOpen] = useState(false);
  const [inferencePanelView, setInferencePanelView] =
    useState<InferencePanelView>("config");
  const [focusOnLatestRun, setFocusOnLatestRun] = useState(false);
  const [chatSessionKey, setChatSessionKey] = useState(0);
  const [clearContextAfter, setClearContextAfter] = useState<string | null>(
    null,
  );
  const currentRunId = useMemo(() => {
    const match = location.pathname.match(/\/runs\/([^/]+)/);
    return match?.[1] ?? "";
  }, [location.pathname]);
  const trpcUtils = trpc.useUtils();
  const chatSessionQuery = trpc.getChatSession.useQuery(
    { runId: currentRunId },
    { enabled: Boolean(currentRunId), retry: false },
  );
  const resetChatSessionMutation = trpc.resetChatSession.useMutation();
  const chatSessionId =
    chatSessionQuery.data?.data?.sessionId || DEFAULT_CHAT_SESSION_ID;
  const usernameRef = useRef<string>("");
  if (!usernameRef.current) {
    const username = user?.username ?? generateRandomUsername();
    usernameRef.current = username;
  }
  const [randomUsername, setRandomUsername] = useState<string>(() =>
    buildContextUsername(usernameRef.current, DEFAULT_CHAT_SESSION_ID),
  );
  const gpuRefreshInterval = useRef<NodeJS.Timeout | null>(null);
  const autoRefreshProjectRef = useRef<string | null>(null);
  const [cacheHydrated, setCacheHydrated] = useState(false);

  const layoutClassName = isMobile
    ? "h-screen w-full flex flex-col bg-muted/20"
    : "relative h-screen w-full overflow-hidden bg-muted/20";
  const layoutStyle = undefined;
  const contentWrapperClassName = isMobile
    ? "overflow-hidden min-h-0 flex-1"
    : "overflow-hidden h-full min-w-0 bg-muted/30";

  // Ref for setting input text from external components (e.g., TemplateLibrary)
  const setInputTextRef = useRef<
    ((text: string, hint?: string) => void) | undefined
  >(undefined);

  useEffect(() => {
    const baseUsername = user?.username ?? usernameRef.current;
    setRandomUsername(buildContextUsername(baseUsername, chatSessionId));
    setClearContextAfter(chatSessionQuery.data?.data?.clearedAt || null);
  }, [chatSessionId, chatSessionQuery.data?.data?.clearedAt, user]);

  const handleUseDatasetForTraining = useCallback(
    (dataset: DatasetInfo) => {
      const command = buildDatasetTrainingCommand(dataset);
      const datasetType = (dataset.type || "").toLowerCase();
      const trainConfig = TRAINABLE_DATASET_TYPE_CONFIG[datasetType];

      if (setInputTextRef.current) {
        setInputTextRef.current(
          command,
          t("dataset.train.inputHint") ||
            "训练命令已填入，点击右下角发送即可开始",
        );
        setRunPagePanelOpen(false);
        message.success(t("wizard.resume.commandFilledToast"));
        return;
      }

      message.warning(t("wizard.resume.inputUnavailable"));
    },
    [setRunPagePanelOpen, t],
  );

  const handleUseDatasetForPreprocess = useCallback(
    (dataset: DatasetInfo) => {
      const command = buildDatasetPreprocessCommand(dataset);

      if (setInputTextRef.current) {
        setInputTextRef.current(
          command,
          t("dataset.preprocess.inputHint") ||
            "预处理命令已填入，点击右下角发送即可开始",
        );
        setRunPagePanelOpen(false);
        message.success(t("wizard.resume.commandFilledToast"));
        return;
      }

      message.warning(t("wizard.resume.inputUnavailable"));
    },
    [setRunPagePanelOpen, t],
  );

  const handleUseEvaluationForBenchmark = useCallback(
    (testName: string) => {
      const command = `运行推理基准测试${testName}`;

      if (setInputTextRef.current) {
        setInputTextRef.current(
          command,
          t("evaluation.benchmark.inputHint") ||
            "评测命令已填入，点击右下角发送即可开始",
        );
        setRunPagePanelOpen(false);
        message.success(t("wizard.resume.commandFilledToast"));
        return;
      }

      message.warning(t("wizard.resume.inputUnavailable"));
    },
    [setRunPagePanelOpen, t],
  );

  const handleClearContext = useCallback(async () => {
    if (!currentRunId) return;
    const result = await resetChatSessionMutation.mutateAsync({
      runId: currentRunId,
    });
    const baseUsername = user?.username ?? usernameRef.current;
    const nextSessionId = result.data.sessionId;
    usernameRef.current = baseUsername;
    setClearContextAfter(result.data.clearedAt || null);
    setRandomUsername(buildContextUsername(baseUsername, nextSessionId));
    await trpcUtils.getChatSession.invalidate({ runId: currentRunId });
    setChatSessionKey((prev) => prev + 1);
    message.success(
      t("runpage.toolbar.clearContextSuccess") || "已清空当前页面上下文",
    );
  }, [currentRunId, resetChatSessionMutation, t, trpcUtils, user]);

  // 数据集管理状态
  const { nodeId: selectedManagementResourceNodeId } =
    useResourceNodeSelection();
  const resourceNodeId = isAdmin
    ? selectedManagementResourceNodeId
    : user?.assignedNodeId || selectedManagementResourceNodeId;
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const datasetsRef = useRef(datasets);
  useEffect(() => {
    datasetsRef.current = datasets;
  }, [datasets]);

  const [isQueryingDatasets, setIsQueryingDatasets] = useState(false);
  const [datasetQueryError, setDatasetQueryError] = useState(false);
  const [datasetErrorMessage, setDatasetErrorMessage] = useState<string>("");
  const [hasQueriedDatasets, setHasQueriedDatasets] = useState(false);
  const [datasetCacheMeta, setDatasetCacheMeta] =
    useState<ManagementCacheMeta | null>(null);
  const withDatasetRuntimeContext = useCallback(
    (items: DatasetInfo[], containerName: string): DatasetInfo[] => {
      const fallbackNodeId = resourceNodeId !== "all" ? resourceNodeId : undefined;
      return items.map((item) => ({
        ...item,
        nodeId: item.nodeId || fallbackNodeId,
        containerName: item.containerName || containerName,
      }));
    },
    [resourceNodeId],
  );

  // 模型管理状态
  const [models, setModels] = useState<ModelInfo[]>([]);
  const modelsRef = useRef(models);
  useEffect(() => {
    modelsRef.current = models;
  }, [models]);

  const [isQueryingModels, setIsQueryingModels] = useState(false);
  const [modelQueryError, setModelQueryError] = useState(false);
  const [modelErrorMessage, setModelErrorMessage] = useState<string>("");
  const [hasQueriedModels, setHasQueriedModels] = useState(false);
  const [modelCacheMeta, setModelCacheMeta] =
    useState<ManagementCacheMeta | null>(null);
  const [managementResourceGroupId, setManagementResourceGroupId] =
    useState<string | undefined>(undefined);

  // 评测管理状态
  const [tests, setTests] = useState<MedicalTestFile[]>([]);
  const testsRef = useRef(tests);
  useEffect(() => {
    testsRef.current = tests;
  }, [tests]);

  const [isQueryingTests, setIsQueryingTests] = useState(false);
  const [testQueryError, setTestQueryError] = useState(false);
  const [testErrorMessage, setTestErrorMessage] = useState<string>("");
  const [hasQueriedTests, setHasQueriedTests] = useState(false);
  const [testCacheMeta, setTestCacheMeta] =
    useState<ManagementCacheMeta | null>(null);
  const withTestRuntimeContext = useCallback(
    (items: MedicalTestFile[], containerName: string): MedicalTestFile[] => {
      const fallbackNodeId = resourceNodeId !== "all" ? resourceNodeId : undefined;
      return items.map((item) => ({
        ...item,
        nodeId: item.nodeId || fallbackNodeId,
        containerName: item.containerName || containerName,
      }));
    },
    [resourceNodeId],
  );

  // 评测结果状态
  const [evaluationResults, setEvaluationResults] = useState<
    EvaluationResult[]
  >([]);
  const evaluationResultsRef = useRef(evaluationResults);
  useEffect(() => {
    evaluationResultsRef.current = evaluationResults;
  }, [evaluationResults]);

  const [isQueryingEvaluationResults, setIsQueryingEvaluationResults] =
    useState(false);
  const [evaluationResultQueryError, setEvaluationResultQueryError] =
    useState(false);
  const [evaluationResultErrorMessage, setEvaluationResultErrorMessage] =
    useState<string>("");
  const [hasQueriedEvaluationResults, setHasQueriedEvaluationResults] =
    useState(false);
  const [evaluationResultCacheMeta, setEvaluationResultCacheMeta] =
    useState<ManagementCacheMeta | null>(null);

  // tRPC API hooks
  const queryDatasetsMutation = trpc.queryDatasets.useMutation();
  const refreshDatasetsMutation = trpc.refreshDatasets.useMutation();
  const getDatasetFilePreviewsMutation =
    trpc.getDatasetFilePreviews.useMutation();
  const queryModelsMutation = trpc.queryModels.useMutation();
  const refreshModelsMutation = trpc.refreshModels.useMutation();
  const uploadDatasetMutation = trpc.uploadDataset.useMutation();
  const downloadDatasetMutation = trpc.downloadDataset.useMutation();
  const deleteDatasetMutation = trpc.deleteDataset.useMutation();
  const queryMedicalTestsMutation = trpc.queryMedicalTests.useMutation();
  const refreshMedicalTestsMutation = trpc.refreshMedicalTests.useMutation();
  const uploadMedicalTestMutation = trpc.uploadMedicalTest.useMutation();
  const downloadMedicalTestMutation = trpc.downloadMedicalTest.useMutation();
  const deleteMedicalTestMutation = trpc.deleteMedicalTest.useMutation();
  const queryEvaluationResultsMutation =
    trpc.queryEvaluationResults.useMutation();
  const refreshEvaluationResultsMutation =
    trpc.refreshEvaluationResults.useMutation();
  const downloadEvaluationResultMutation =
    trpc.downloadEvaluationResult.useMutation();
  const deleteEvaluationResultMutation =
    trpc.deleteEvaluationResult.useMutation();
  const deleteModelMutation = trpc.deleteModel.useMutation();

  // 上传/下载状态
  const [uploadModalOpen, setUploadModalOpen] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [downloadingId, setDownloadingId] = useState<string | null>(null);
  const [deletingDatasetId, setDeletingDatasetId] = useState<string | null>(
    null,
  );
  const [deletingModelId, setDeletingModelId] = useState<string | null>(null);

  // 评测上传/下载状态
  const [evaluationUploadModalOpen, setEvaluationUploadModalOpen] =
    useState(false);
  const [isUploadingEvaluation, setIsUploadingEvaluation] = useState(false);
  const [evaluationUploadProgress, setEvaluationUploadProgress] = useState(0);
  const [downloadingTestId, setDownloadingTestId] = useState<string | null>(
    null,
  );
  const [deletingTestId, setDeletingTestId] = useState<string | null>(null);

  // 向导相关状态
  const [isQuickStartWizardOpen, setIsQuickStartWizardOpen] = useState(false);
  const [wizardQueryState, setWizardQueryState] = useState<
    "idle" | "querying" | "completed"
  >("idle");
  const [wizardDatasets, setWizardDatasets] = useState<DatasetInfo[]>([]);
  const [wizardSelectedDataset, setWizardSelectedDataset] =
    useState<DatasetInfo | null>(null);

  // Template Library 状态
  const [isLibraryOpen, setIsLibraryOpen] = useState(false);
  const [libraryInitialCategory, setLibraryInitialCategory] = useState<
    string | null
  >(null);
  const {
    getUnifiedCategories,
    getRecentTemplates,
    addToRecent,
    getIconComponent,
  } = useTemplateLibrary({ isAdmin });

  // 打开 Template Library 的回调
  const handleOpenTemplateLibrary = useCallback((category?: string) => {
    setLibraryInitialCategory(category || null);
    setIsLibraryOpen(true);
  }, []);

  // 组件挂载或默认容器变化时，只加载与当前默认容器匹配的缓存。
  useEffect(() => {
    const isCacheForCurrentContainer = (metaKey: string) => {
      const meta = loadCachedMeta(metaKey);
      const metaContainer = meta?.containerName?.trim();
      return (
        Boolean(metaContainer) &&
        metaContainer === defaultContainerName &&
        meta?.nodeId === resourceNodeId
      );
    };
    const isCacheForCurrentEvaluateContainer = (metaKey: string) => {
      const meta = loadCachedMeta(metaKey);
      const metaContainer = meta?.containerName?.trim();
      return (
        Boolean(metaContainer) &&
        metaContainer === defaultEvaluateContainerName &&
        meta?.nodeId === resourceNodeId
      );
    };

    // 加载缓存的数据集
    const cachedDatasets = localStorage.getItem("cached_datasets");
    if (
      cachedDatasets &&
      isCacheForCurrentContainer(CACHE_META_KEYS.datasets)
    ) {
      try {
        const parsed = JSON.parse(cachedDatasets);
        setDatasets(withDatasetRuntimeContext(parsed, defaultContainerName));
        setDatasetCacheMeta(loadCachedMeta(CACHE_META_KEYS.datasets));
        setHasQueriedDatasets(true);
      } catch (e) {
        console.error("Failed to parse cached datasets:", e);
      }
    } else {
      setDatasets([]);
      setDatasetCacheMeta(null);
      setHasQueriedDatasets(false);
    }

    // 加载缓存的模型
    const cachedModels = localStorage.getItem("cached_models");
    if (cachedModels && isCacheForCurrentContainer(CACHE_META_KEYS.models)) {
      try {
        const parsed = JSON.parse(cachedModels);
        setModels(parsed);
        setModelCacheMeta(loadCachedMeta(CACHE_META_KEYS.models));
        setHasQueriedModels(true);
      } catch (e) {
        console.error("Failed to parse cached models:", e);
      }
    } else {
      setModels([]);
      setModelCacheMeta(null);
      setHasQueriedModels(false);
    }

    // 加载缓存的评测文件
    const cachedTests = localStorage.getItem("cached_tests");
    if (
      cachedTests &&
      isCacheForCurrentEvaluateContainer(CACHE_META_KEYS.tests)
    ) {
      try {
        const parsed = JSON.parse(cachedTests);
        setTests(withTestRuntimeContext(parsed, defaultEvaluateContainerName));
        setTestCacheMeta(loadCachedMeta(CACHE_META_KEYS.tests));
        setHasQueriedTests(true);
      } catch (e) {
        console.error("Failed to parse cached tests:", e);
      }
    } else {
      setTests([]);
      setTestCacheMeta(null);
      setHasQueriedTests(false);
    }

    // 加载缓存的评测结果
    const cachedEvaluationResults = localStorage.getItem(
      "cached_evaluation_results",
    );
    if (
      cachedEvaluationResults &&
      isCacheForCurrentEvaluateContainer(CACHE_META_KEYS.evaluationResults)
    ) {
      try {
        const parsed = JSON.parse(cachedEvaluationResults);
        setEvaluationResults(parsed);
        setEvaluationResultCacheMeta(
          loadCachedMeta(CACHE_META_KEYS.evaluationResults),
        );
        setHasQueriedEvaluationResults(true);
      } catch (e) {
        console.error("Failed to parse cached evaluation results:", e);
      }
    } else {
      setEvaluationResults([]);
      setEvaluationResultCacheMeta(null);
      setHasQueriedEvaluationResults(false);
    }

    setCacheHydrated(true);
  }, [
    defaultContainerName,
    defaultEvaluateContainerName,
    resourceNodeId,
    withDatasetRuntimeContext,
    withTestRuntimeContext,
  ]);

  // 将后端返回的数据集格式转换为前端格式
  const convertBackendDatasetsToFrontend = (
    backendData: Record<string, any[]>,
  ): DatasetInfo[] => {
    const result: DatasetInfo[] = [];

    for (const [type, datasets] of Object.entries(backendData)) {
      for (const ds of datasets) {
        // 获取第一个文件的预览作为 sampleContent
        let sampleContent = "";
        let fileName = "";

        const files = (ds.files || []).filter(isVisibleDatasetFile);
        const filePreviews = (ds.filePreviews || []).filter(
          (item: DatasetFilePreview) => isVisibleDatasetFile(item.filename),
        );

        if (filePreviews.length > 0) {
          sampleContent = filePreviews[0].preview;
          fileName = filePreviews[0].filename;
        }

        result.push({
          name: ds.name,
          type: ds.type || type,
          path: ds.path,
          description: ds.description,
          files,
          filePreviews,
          sampleContent,
          fileName,
          size: ds.size,
          createdAt: ds.createdAt,
        });
      }
    }

    return result;
  };

  // 将后端返回的模型格式转换为前端格式
  const convertBackendModelsToFrontend = (
    backendData: Record<string, any[]>,
  ): ModelInfo[] => {
    const result: ModelInfo[] = [];

    for (const [type, models] of Object.entries(backendData)) {
      for (const model of models) {
        result.push({
          name: model.name,
          type: model.type || type,
          path: model.path,
          merged: model.merged,
          checkpoints: model.checkpoints,
          size: model.size,
          createdAt: model.createdAt,
        });
      }
    }

    return result;
  };

  // 查询数据集快照
  const handleQueryDatasets = useCallback(
    async (
      containerName: string,
      options?: {
        background?: boolean;
        hydrateState?: boolean;
        persistCache?: boolean;
      },
    ): Promise<DatasetInfo[]> => {
      const background = options?.background ?? false;
      const hydrateState = options?.hydrateState ?? true;
      const persistCache = options?.persistCache ?? true;
      if (!background) {
        setIsQueryingDatasets(true);
        setDatasetQueryError(false);
        setDatasetErrorMessage("");
        setHasQueriedDatasets(true);
      }

      // 如果向导打开，更新向导查询状态
      if (!background && isQuickStartWizardOpen) {
        setWizardQueryState("querying");
      }

      try {
        const response = await queryDatasetsMutation.mutateAsync({
          nodeId: resourceNodeId,
          container: containerName,
        });

        if (response.success && response.data) {
          const frontendDatasets = withDatasetRuntimeContext(
            response.data.items || [],
            containerName,
          );
          if (hydrateState) {
            setDatasets(frontendDatasets);
            setDatasetCacheMeta(response.data.meta);
            setHasQueriedDatasets(true);
          }
          if (!background) {
            setDatasetQueryError(false);
            setDatasetErrorMessage("");
          }

          if (persistCache) {
            localStorage.setItem(
              "cached_datasets",
              JSON.stringify(frontendDatasets),
            );
            localStorage.setItem("cached_datasets_container", containerName);
            persistCachedMeta(CACHE_META_KEYS.datasets, response.data.meta);
          }

          // 如果向导打开，更新向导数据集和状态
          if (!background && isQuickStartWizardOpen) {
            setWizardDatasets(frontendDatasets);
            setWizardQueryState("completed");
          }

          // 检查是否需要重新打开向导（用户可能在查询时关闭了向导）
          const wizardActive = sessionStorage.getItem("wizard_active");
          const wizardStep = sessionStorage.getItem("wizard_step");
          if (
            !background &&
            wizardActive === "true" &&
            !isQuickStartWizardOpen &&
            wizardStep === "0"
          ) {
            // 延迟重新打开向导，确保查询完成
            setTimeout(() => {
              setIsQuickStartWizardOpen(true);
              setWizardDatasets(frontendDatasets);
              setWizardQueryState("completed");
              // 注意：不要在这里清除 sessionStorage
              // 只有在完成整个向导流程后才清除
            }, 500);
          }

          // 返回查询到的数据
          return frontendDatasets;
        } else {
          if (!background) {
            setDatasetQueryError(true);
            setDatasetErrorMessage(response.message || "查询失败");
          }
          if (!background && isQuickStartWizardOpen) {
            setWizardQueryState("idle");
          }
          return [];
        }
      } catch (error: any) {
        console.error("Error querying datasets:", error);
        if (!background) {
          setDatasetQueryError(true);
          setDatasetErrorMessage(error.message || "查询过程中发生错误");
        }
        if (!background && isQuickStartWizardOpen) {
          setWizardQueryState("idle");
        }
        return [];
      } finally {
        if (!background) {
          setIsQueryingDatasets(false);
        }
      }
    },
    [
      queryDatasetsMutation,
      isQuickStartWizardOpen,
      resourceNodeId,
      withDatasetRuntimeContext,
    ],
  );

  const handleRefreshDatasets = useCallback(
    async (
      containerName: string,
      options?: {
        background?: boolean;
        hydrateState?: boolean;
        persistCache?: boolean;
      },
    ): Promise<DatasetInfo[]> => {
      const background = options?.background ?? false;
      const hydrateState = options?.hydrateState ?? true;
      const persistCache = options?.persistCache ?? true;
      if (!background) {
        setIsQueryingDatasets(true);
        setDatasetQueryError(false);
        setDatasetErrorMessage("");
        setHasQueriedDatasets(true);
      }

      try {
        const response = await refreshDatasetsMutation.mutateAsync({
          nodeId: resourceNodeId,
          container: containerName,
        });

        if (response.success && response.data) {
          const latestDatasets = withDatasetRuntimeContext(
            response.data.items || [],
            containerName,
          );
          if (persistCache) {
            localStorage.setItem(
              "cached_datasets",
              JSON.stringify(latestDatasets),
            );
            localStorage.setItem("cached_datasets_container", containerName);
            persistCachedMeta(CACHE_META_KEYS.datasets, response.data.meta);
          }
          if (hydrateState) {
            setDatasets(latestDatasets);
            setDatasetCacheMeta(response.data.meta);
            setHasQueriedDatasets(true);
          }
          return latestDatasets;
        }

        console.warn(
          "Dataset refresh failed",
          {
            projectName,
            containerName,
            message: response.message,
          },
        );
        if (!background) {
          setDatasetQueryError(true);
          setDatasetErrorMessage(response.message || "刷新失败");
        }
        return [];
      } catch (error: any) {
        console.error("Error refreshing datasets:", error);
        console.error("Dataset refresh error", {
          projectName,
          containerName,
          message: error?.message,
        });
        if (!background) {
          setDatasetQueryError(true);
          setDatasetErrorMessage(error.message || "刷新过程中发生错误");
        }
        return [];
      } finally {
        if (!background) {
          setIsQueryingDatasets(false);
        }
      }
    },
    [
      refreshDatasetsMutation,
      projectName,
      resourceNodeId,
      withDatasetRuntimeContext,
    ],
  );

  const handleLoadDatasetPreviews = useCallback(
    async (dataset: DatasetInfo): Promise<void> => {
      const jsonFileCount = (dataset.files || []).filter(
        (file) =>
          isVisibleDatasetFile(file) &&
          file.endsWith(".json"),
      ).length;
      if ((dataset.filePreviews || []).length >= jsonFileCount) {
        return;
      }

      const response = await getDatasetFilePreviewsMutation.mutateAsync({
        nodeId: dataset.nodeId || resourceNodeId,
        container: dataset.containerName || defaultContainerName,
        datasetType: dataset.type as "raw" | "sft" | "dpo",
        datasetName: dataset.name,
      });

      if (!response.success || !response.data) {
        throw new Error(response.message || "预览加载失败");
      }

      const previews = (response.data as DatasetFilePreview[]).filter(
        (item) => isVisibleDatasetFile(item.filename),
      );
      const firstPreview = previews[0];

      setDatasets((prev) => {
        const next = prev.map((item) =>
          item.name === dataset.name &&
          item.type === dataset.type &&
          item.nodeId === dataset.nodeId &&
          item.containerName === dataset.containerName
            ? {
                ...item,
                filePreviews: previews,
                sampleContent: firstPreview?.preview || item.sampleContent,
                fileName: firstPreview?.filename || item.fileName,
              }
            : item,
        );
        localStorage.setItem("cached_datasets", JSON.stringify(next));
        localStorage.setItem("cached_datasets_container", defaultContainerName);
        return next;
      });
    },
    [defaultContainerName, getDatasetFilePreviewsMutation, resourceNodeId],
  );

  // 查询模型快照
  const handleQueryModels = useCallback(
    async (
      containerName: string,
      options?: {
        background?: boolean;
        hydrateState?: boolean;
        persistCache?: boolean;
      },
    ): Promise<ModelInfo[]> => {
      const background = options?.background ?? false;
      const hydrateState = options?.hydrateState ?? true;
      const persistCache = options?.persistCache ?? true;
      if (!background) {
        setIsQueryingModels(true);
        setModelQueryError(false);
        setModelErrorMessage("");
        setHasQueriedModels(true);
      }

      try {
        const response = await queryModelsMutation.mutateAsync({
          nodeId: resourceNodeId,
          groupId: isAdmin ? managementResourceGroupId || undefined : undefined,
          container: containerName,
        });

        if (response.success && response.data) {
          const frontendModels = response.data.items || [];
          if (hydrateState) {
            setModels(frontendModels);
            setModelCacheMeta(response.data.meta);
            setHasQueriedModels(true);
          }
          if (!background) {
            setModelQueryError(false);
            setModelErrorMessage("");
          }

          if (persistCache) {
            localStorage.setItem(
              "cached_models",
              JSON.stringify(frontendModels),
            );
            localStorage.setItem("cached_models_container", containerName);
            persistCachedMeta(CACHE_META_KEYS.models, response.data.meta);
          }

          // 返回查询到的数据
          return frontendModels;
        } else {
          if (!background) {
            setModelQueryError(true);
            setModelErrorMessage(response.message || "查询失败");
          }
          return [];
        }
      } catch (error: any) {
        console.error("Error querying models:", error);
        if (!background) {
          setModelQueryError(true);
          setModelErrorMessage(error.message || "查询过程中发生错误");
        }
        return [];
      } finally {
        if (!background) {
          setIsQueryingModels(false);
        }
      }
    },
    [isAdmin, queryModelsMutation, resourceNodeId, managementResourceGroupId],
  );

  const handleRefreshModels = useCallback(
    async (
      containerName: string,
      options?: {
        background?: boolean;
        hydrateState?: boolean;
        persistCache?: boolean;
      },
    ): Promise<ModelInfo[]> => {
      const background = options?.background ?? false;
      const hydrateState = options?.hydrateState ?? true;
      const persistCache = options?.persistCache ?? true;
      if (!background) {
        setIsQueryingModels(true);
        setModelQueryError(false);
        setModelErrorMessage("");
        setHasQueriedModels(true);
      }

      try {
        const response = await refreshModelsMutation.mutateAsync({
          nodeId: resourceNodeId,
          groupId: isAdmin ? managementResourceGroupId || undefined : undefined,
          container: containerName,
        });

        if (response.success && response.data) {
          const latestModels = response.data.items || [];
          if (persistCache) {
            localStorage.setItem("cached_models", JSON.stringify(latestModels));
            localStorage.setItem("cached_models_container", containerName);
            persistCachedMeta(CACHE_META_KEYS.models, response.data.meta);
          }
          if (hydrateState) {
            setModels(latestModels);
            setModelCacheMeta(response.data.meta);
            setHasQueriedModels(true);
          }
          return latestModels;
        }

        console.warn(
          "Model refresh failed",
          {
            projectName,
            containerName,
            message: response.message,
          },
        );
        if (!background) {
          setModelQueryError(true);
          setModelErrorMessage(response.message || "刷新失败");
        }
        return [];
      } catch (error: any) {
        console.error("Error refreshing models:", error);
        console.error("Model refresh error", {
          projectName,
          containerName,
          message: error?.message,
        });
        if (!background) {
          setModelQueryError(true);
          setModelErrorMessage(error.message || "刷新过程中发生错误");
        }
        return [];
      } finally {
        if (!background) {
          setIsQueryingModels(false);
        }
      }
    },
    [isAdmin, refreshModelsMutation, projectName, resourceNodeId, managementResourceGroupId],
  );

  // 查询评测文件快照
  const handleQueryTests = useCallback(
    async (
      containerName: string,
      options?: {
        background?: boolean;
        hydrateState?: boolean;
        persistCache?: boolean;
      },
    ): Promise<MedicalTestFile[]> => {
      const background = options?.background ?? false;
      const hydrateState = options?.hydrateState ?? true;
      const persistCache = options?.persistCache ?? true;
      if (!background) {
        setIsQueryingTests(true);
        setTestQueryError(false);
        setTestErrorMessage("");
        setHasQueriedTests(true);
      }

      try {
        const response = await queryMedicalTestsMutation.mutateAsync({
          nodeId: resourceNodeId,
          container: containerName,
        });

        if (response.success && response.data) {
          const latestTests = withTestRuntimeContext(
            response.data.items || [],
            containerName,
          );
          if (hydrateState) {
            setTests(latestTests);
            setTestCacheMeta(response.data.meta);
            setHasQueriedTests(true);
          }
          if (!background) {
            setTestQueryError(false);
            setTestErrorMessage("");
          }

          if (persistCache) {
            localStorage.setItem(
              "cached_tests",
              JSON.stringify(latestTests),
            );
            localStorage.setItem("cached_tests_container", containerName);
            persistCachedMeta(CACHE_META_KEYS.tests, response.data.meta);
          }

          // 返回查询到的数据
          return latestTests;
        } else {
          if (!background) {
            setTestQueryError(true);
            setTestErrorMessage(response.message || "查询失败");
          }
          return [];
        }
      } catch (error: any) {
        console.error("Error querying medical tests:", error);
        if (!background) {
          setTestQueryError(true);
          setTestErrorMessage(error.message || "查询过程中发生错误");
        }
        return [];
      } finally {
        if (!background) {
          setIsQueryingTests(false);
        }
      }
    },
    [queryMedicalTestsMutation, resourceNodeId, withTestRuntimeContext],
  );

  const handleRefreshTests = useCallback(
    async (
      containerName: string,
      options?: {
        background?: boolean;
        hydrateState?: boolean;
        persistCache?: boolean;
      },
    ): Promise<MedicalTestFile[]> => {
      const background = options?.background ?? false;
      const hydrateState = options?.hydrateState ?? true;
      const persistCache = options?.persistCache ?? true;
      if (!background) {
        setIsQueryingTests(true);
        setTestQueryError(false);
        setTestErrorMessage("");
        setHasQueriedTests(true);
      }

      try {
        const response = await refreshMedicalTestsMutation.mutateAsync({
          nodeId: resourceNodeId,
          container: containerName,
        });

        if (response.success && response.data) {
          const latestTests = withTestRuntimeContext(
            response.data.items || [],
            containerName,
          );
          if (persistCache) {
            localStorage.setItem("cached_tests", JSON.stringify(latestTests));
            localStorage.setItem("cached_tests_container", containerName);
            persistCachedMeta(CACHE_META_KEYS.tests, response.data.meta);
          }
          if (hydrateState) {
            setTests(latestTests);
            setTestCacheMeta(response.data.meta);
            setHasQueriedTests(true);
          }
          return latestTests;
        }

        console.warn(
          "Test refresh failed",
          {
            projectName,
            containerName,
            message: response.message,
          },
        );
        if (!background) {
          setTestQueryError(true);
          setTestErrorMessage(response.message || "刷新失败");
        }
        return [];
      } catch (error: any) {
        console.error("Error refreshing medical tests:", error);
        console.error("Test refresh error", {
          projectName,
          containerName,
          message: error?.message,
        });
        if (!background) {
          setTestQueryError(true);
          setTestErrorMessage(error.message || "刷新过程中发生错误");
        }
        return [];
      } finally {
        if (!background) {
          setIsQueryingTests(false);
        }
      }
    },
    [refreshMedicalTestsMutation, projectName, resourceNodeId, withTestRuntimeContext],
  );

  // 上传数据集
  const handleUpload = useCallback(
    async (params: {
      containerName: string;
      datasetType: "raw" | "sft" | "dpo";
      datasetName: string;
      file: File;
    }) => {
      setIsUploading(true);
      setUploadProgress(0);

      try {
        const fileBase64 = await fileToBase64(params.file);

        setUploadProgress(30);

        const response = await uploadDatasetMutation.mutateAsync({
          nodeId: resourceNodeId,
          container: params.containerName,
          datasetType: params.datasetType,
          datasetName: params.datasetName,
          filename: params.file.name,
          fileBase64,
        });


        setUploadProgress(100);

        if (response.success) {
          message.success(response.message || "上传成功");
          // 如果有警告，显示警告信息
          if (response.warning) {
            message.warning(response.warning);
          }
          setUploadModalOpen(false);
          // 上传后强制重扫容器，避免查询接口返回上传前的缓存快照。
          await handleRefreshDatasets(params.containerName, {
            hydrateState: true,
            persistCache: true,
          });
        } else {
          // 上传失败，抛出错误让 DatasetUploadModal 捕获并显示
          // 不在这里显示 message.error，避免重复提示
          throw new Error(response.message || "上传失败");
        }
      } catch (error: any) {
        console.error("Upload error:", error);
        // 重新抛出错误，让 DatasetUploadModal 捕获并显示
        // 不在这里显示 message.error，避免重复提示
        throw error;
      } finally {
        setIsUploading(false);
        setUploadProgress(0);
      }
    },
    [uploadDatasetMutation, handleRefreshDatasets, resourceNodeId],
  );

  // 上传评测文件
  const handleUploadTest = useCallback(
    async (params: {
      containerName: string;
      testType: string;
      filename: string;
      file: File;
    }) => {
      setIsUploadingEvaluation(true);
      setEvaluationUploadProgress(0);

      try {
        const fileBase64 = await fileToBase64(params.file);

        setEvaluationUploadProgress(30);

        const response = await uploadMedicalTestMutation.mutateAsync({
          nodeId: resourceNodeId,
          container: params.containerName,
          testType: params.testType,
          filename: params.filename,
          fileBase64,
        });


        setEvaluationUploadProgress(100);

        if (response.success) {
          message.success(response.message || "上传成功");
          setEvaluationUploadModalOpen(false);
          // 刷新评测列表
          handleQueryTests(params.containerName);
        } else {
          throw new Error(response.message || "上传失败");
        }
      } catch (error: any) {
        console.error("Upload test error:", error);
        throw error;
      } finally {
        setIsUploadingEvaluation(false);
        setEvaluationUploadProgress(0);
      }
    },
    [uploadMedicalTestMutation, handleQueryTests, resourceNodeId],
  );

  // 下载数据集
  const handleDownload = useCallback(
    async (dataset: DatasetInfo) => {
      setDownloadingId(dataset.name);

      try {
        const containerName = dataset.containerName || defaultContainerName;

        const response = await downloadDatasetMutation.mutateAsync({
          nodeId: dataset.nodeId || resourceNodeId,
          container: containerName,
          datasetType: dataset.type as "raw" | "sft" | "dpo",
          datasetName: dataset.name,
        });

        if (response.success && response.data) {
          // 将 base64 转换为 Blob 并下载
          const byteCharacters = atob(response.data.fileBase64);
          const byteNumbers = new Array(byteCharacters.length);
          for (let i = 0; i < byteCharacters.length; i++) {
            byteNumbers[i] = byteCharacters.charCodeAt(i);
          }
          const byteArray = new Uint8Array(byteNumbers);
          const blob = new Blob([byteArray], { type: "application/zip" });

          // 创建下载链接
          const url = window.URL.createObjectURL(blob);
          const link = document.createElement("a");
          link.href = url;
          link.download = response.data.filename;
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
          window.URL.revokeObjectURL(url);

          message.success("下载成功");
        } else {
          throw new Error(response.message || "下载失败");
        }
      } catch (error: any) {
        console.error("Download error:", error);
        message.error(error.message || "下载过程中发生错误");
        throw error;
      } finally {
        setDownloadingId(null);
      }
    },
    [defaultContainerName, downloadDatasetMutation, resourceNodeId],
  );

  // 下载评测文件
  const handleDownloadTest = useCallback(
    async (testName: string, sourceTest?: MedicalTestFile) => {
      const test =
        sourceTest ||
        testsRef.current.find((t) => (t.filename || t.name) === testName);
      if (!test) {
        throw new Error(`未找到评测文件: ${testName}`);
      }

      setDownloadingTestId(test.filename);

      try {
        const containerName =
          test.containerName || defaultEvaluateContainerName;

        const response = await downloadMedicalTestMutation.mutateAsync({
          nodeId: test.nodeId || resourceNodeId,
          container: containerName,
          filename: test.filename,
        });

        if (response.success && response.data) {
          // 将 base64 转换为 Blob 并下载
          const byteCharacters = atob(response.data.fileBase64);
          const byteNumbers = new Array(byteCharacters.length);
          for (let i = 0; i < byteCharacters.length; i++) {
            byteNumbers[i] = byteCharacters.charCodeAt(i);
          }
          const byteArray = new Uint8Array(byteNumbers);
          const mimeType = response.data.filename.endsWith(".tar.gz")
            ? "application/gzip"
            : "application/json";
          const blob = new Blob([byteArray], { type: mimeType });

          // 创建下载链接
          const url = window.URL.createObjectURL(blob);
          const link = document.createElement("a");
          link.href = url;
          link.download = response.data.filename;
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
          window.URL.revokeObjectURL(url);

          message.success("下载成功");
        } else {
          throw new Error(response.message || "下载失败");
        }
      } catch (error: any) {
        console.error("Download test error:", error);
        message.error(error.message || "下载过程中发生错误");
        throw error;
      } finally {
        setDownloadingTestId(null);
      }
    },
    [defaultEvaluateContainerName, downloadMedicalTestMutation, resourceNodeId],
  );

  const handleDeleteDataset = useCallback(
    async (dataset: DatasetInfo) => {
      const deleteId = `${dataset.type}:${dataset.name}`;
      setDeletingDatasetId(deleteId);

      try {
        const response = await deleteDatasetMutation.mutateAsync({
          nodeId: dataset.nodeId || resourceNodeId,
          container: dataset.containerName || defaultContainerName,
          datasetType: dataset.type as "raw" | "sft" | "dpo",
          datasetName: dataset.name,
        });

        if (!response.success) {
          throw new Error(response.message || "删除失败");
        }

        message.success(response.message || "删除成功");
        await handleRefreshDatasets(dataset.containerName || defaultContainerName, {
          hydrateState: true,
          persistCache: true,
        });
      } catch (error: any) {
        console.error("Delete dataset error:", error);
        message.error(error.message || "删除数据集失败");
        throw error;
      } finally {
        setDeletingDatasetId(null);
      }
    },
    [
      defaultContainerName,
      deleteDatasetMutation,
      handleRefreshDatasets,
      resourceNodeId,
    ],
  );

  const handleDeleteModel = useCallback(
    async (model: ModelInfo) => {
      const deleteId = `${model.type}:${model.name}`;
      setDeletingModelId(deleteId);

      try {
        const response = await deleteModelMutation.mutateAsync({
          nodeId: model.nodeId || resourceNodeId,
          container: model.containerName || defaultContainerName,
          modelType: model.type as
            | "base_train"
            | "batch_trained"
            | "daily_trained"
            | "inference",
          modelName: model.name,
          modelPath: model.path,
        });

        if (!response.success) {
          throw new Error(response.message || "删除失败");
        }

        message.success(response.message || "删除成功");
        await handleRefreshModels(defaultContainerName, {
          hydrateState: true,
          persistCache: true,
        });
      } catch (error: any) {
        console.error("Delete model error:", error);
        message.error(error.message || "删除模型失败");
        throw error;
      } finally {
        setDeletingModelId(null);
      }
    },
    [
      defaultContainerName,
      deleteModelMutation,
      handleRefreshModels,
      resourceNodeId,
    ],
  );

  const handleDeleteTest = useCallback(
    async (testName: string, sourceTest?: MedicalTestFile) => {
      setDeletingTestId(testName);

      try {
        const test =
          sourceTest ||
          testsRef.current.find(
            (item) => (item.filename || item.name) === testName,
          );
        const response = await deleteMedicalTestMutation.mutateAsync({
          nodeId: test?.nodeId || resourceNodeId,
          container: test?.containerName || defaultEvaluateContainerName,
          filename: testName,
        });

        if (!response.success) {
          throw new Error(response.message || "删除失败");
        }

        message.success(response.message || "删除成功");
        await handleRefreshTests(defaultEvaluateContainerName, {
          hydrateState: true,
          persistCache: true,
        });
      } catch (error: any) {
        console.error("Delete test error:", error);
        message.error(error.message || "删除评测集失败");
        throw error;
      } finally {
        setDeletingTestId(null);
      }
    },
    [
      defaultEvaluateContainerName,
      deleteMedicalTestMutation,
      handleRefreshTests,
      resourceNodeId,
    ],
  );

  // 查询评测结果快照
  const handleQueryEvaluationResults = useCallback(
    async (
      containerName: string,
      options?: {
        background?: boolean;
        hydrateState?: boolean;
        persistCache?: boolean;
      },
    ): Promise<EvaluationResult[]> => {
      const background = options?.background ?? false;
      const hydrateState = options?.hydrateState ?? true;
      const persistCache = options?.persistCache ?? true;
      if (!background) {
        setIsQueryingEvaluationResults(true);
        setEvaluationResultQueryError(false);
        setEvaluationResultErrorMessage("");
        setHasQueriedEvaluationResults(true);
      }

      try {
        const response = await queryEvaluationResultsMutation.mutateAsync({
          nodeId: resourceNodeId,
          container: containerName,
        });

        if (response.success && response.data) {
          if (hydrateState) {
            setEvaluationResults(response.data.items || []);
            setEvaluationResultCacheMeta(response.data.meta);
            setHasQueriedEvaluationResults(true);
          }
          if (!background) {
            setEvaluationResultQueryError(false);
            setEvaluationResultErrorMessage("");
          }

          if (persistCache) {
            localStorage.setItem(
              "cached_evaluation_results",
              JSON.stringify(response.data.items || []),
            );
            localStorage.setItem(
              "cached_evaluation_results_container",
              containerName,
            );
            persistCachedMeta(
              CACHE_META_KEYS.evaluationResults,
              response.data.meta,
            );
          }

          // 返回查询到的数据
          return response.data.items || [];
        } else {
          if (!background) {
            setEvaluationResultQueryError(true);
            setEvaluationResultErrorMessage(response.message || "查询失败");
          }
          return [];
        }
      } catch (error: any) {
        console.error("Error querying evaluation results:", error);
        if (!background) {
          setEvaluationResultQueryError(true);
          setEvaluationResultErrorMessage(
            error.message || "查询过程中发生错误",
          );
        }
        return [];
      } finally {
        if (!background) {
          setIsQueryingEvaluationResults(false);
        }
      }
    },
    [queryEvaluationResultsMutation, resourceNodeId],
  );

  const handleRefreshEvaluationResults = useCallback(
    async (
      containerName: string,
      options?: {
        background?: boolean;
        hydrateState?: boolean;
        persistCache?: boolean;
      },
    ): Promise<EvaluationResult[]> => {
      const background = options?.background ?? false;
      const hydrateState = options?.hydrateState ?? true;
      const persistCache = options?.persistCache ?? true;
      if (!background) {
        setIsQueryingEvaluationResults(true);
        setEvaluationResultQueryError(false);
        setEvaluationResultErrorMessage("");
        setHasQueriedEvaluationResults(true);
      }

      try {
        const response = await refreshEvaluationResultsMutation.mutateAsync({
          nodeId: resourceNodeId,
          container: containerName,
        });

        if (response.success && response.data) {
          const latestResults = response.data.items || [];
          if (persistCache) {
            localStorage.setItem(
              "cached_evaluation_results",
              JSON.stringify(latestResults),
            );
            localStorage.setItem(
              "cached_evaluation_results_container",
              containerName,
            );
            persistCachedMeta(
              CACHE_META_KEYS.evaluationResults,
              response.data.meta,
            );
          }
          if (hydrateState) {
            setEvaluationResults(latestResults);
            setEvaluationResultCacheMeta(response.data.meta);
            setHasQueriedEvaluationResults(true);
          }
          return latestResults;
        }

        console.warn(
          "Evaluation result refresh failed",
          {
            projectName,
            containerName,
            message: response.message,
          },
        );
        if (!background) {
          setEvaluationResultQueryError(true);
          setEvaluationResultErrorMessage(response.message || "刷新失败");
        }
        return [];
      } catch (error: any) {
        console.error("Error refreshing evaluation results:", error);
        console.error(
          "Evaluation result refresh error",
          {
            projectName,
            containerName,
            message: error?.message,
          },
        );
        if (!background) {
          setEvaluationResultQueryError(true);
          setEvaluationResultErrorMessage(
            error.message || "刷新过程中发生错误",
          );
        }
        return [];
      } finally {
        if (!background) {
          setIsQueryingEvaluationResults(false);
        }
      }
    },
    [
      handleQueryEvaluationResults,
      refreshEvaluationResultsMutation,
      projectName,
      resourceNodeId,
    ],
  );

  // 进入项目页时，基于最近使用的容器静默预热一轮管理数据。
  // 默认容器变化时，当前页面也要同步刷新，避免展示旧容器的缓存数据。
  useEffect(() => {
    if (!cacheHydrated || !projectName) return;
    const refreshKey = `${projectName}:${defaultContainerName}:${defaultEvaluateContainerName}`;
    if (autoRefreshProjectRef.current === refreshKey) return;

    autoRefreshProjectRef.current = refreshKey;

    const timer = window.setTimeout(() => {
      const hasCurrentContainerCache = (cacheKey: string, metaKey: string) => {
        const metaContainer = loadCachedMeta(metaKey)?.containerName?.trim();
        return (
          Boolean(localStorage.getItem(cacheKey)) &&
          Boolean(metaContainer) &&
          metaContainer === defaultContainerName
        );
      };
      const hasCurrentEvaluateContainerCache = (
        cacheKey: string,
        metaKey: string,
      ) => {
        const metaContainer = loadCachedMeta(metaKey)?.containerName?.trim();
        return (
          Boolean(localStorage.getItem(cacheKey)) &&
          Boolean(metaContainer) &&
          metaContainer === defaultEvaluateContainerName
        );
      };
      const hasDatasetCache = hasCurrentContainerCache(
        "cached_datasets",
        CACHE_META_KEYS.datasets,
      );
      const hasModelCache = hasCurrentContainerCache(
        "cached_models",
        CACHE_META_KEYS.models,
      );
      const hasTestCache = hasCurrentEvaluateContainerCache(
        "cached_tests",
        CACHE_META_KEYS.tests,
      );
      const hasEvaluationCache = hasCurrentEvaluateContainerCache(
        "cached_evaluation_results",
        CACHE_META_KEYS.evaluationResults,
      );
      const datasetContainer = defaultContainerName;
      const modelContainer = defaultContainerName;
      const testContainer = defaultEvaluateContainerName;
      const evaluationContainer = defaultEvaluateContainerName;


      const syncTasks: Promise<unknown>[] = [];

      syncTasks.push(
        handleRefreshDatasets(datasetContainer, {
          background: true,
          hydrateState: hasDatasetCache,
          persistCache: hasDatasetCache,
        }),
      );
      syncTasks.push(
        handleRefreshModels(modelContainer, {
          background: true,
          hydrateState: hasModelCache,
          persistCache: hasModelCache,
        }),
      );
      syncTasks.push(
        handleRefreshTests(testContainer, {
          background: true,
          hydrateState: hasTestCache,
          persistCache: hasTestCache,
        }),
      );
      syncTasks.push(
        handleRefreshEvaluationResults(evaluationContainer, {
          background: true,
          hydrateState: hasEvaluationCache,
          persistCache: hasEvaluationCache,
        }),
      );

      if (syncTasks.length > 0) {
        void Promise.allSettled(syncTasks);
      }
    }, 0);

    return () => window.clearTimeout(timer);
  }, [
    cacheHydrated,
    projectName,
    defaultContainerName,
    defaultEvaluateContainerName,
    handleRefreshDatasets,
    handleRefreshModels,
    handleRefreshTests,
    handleRefreshEvaluationResults,
  ]);

  useEffect(() => {
    if (!cacheHydrated || !projectName) return;

    const inFlight = new Set<string>();
    const runRefresh = async (key: string, refresh: () => Promise<unknown>) => {
      if (inFlight.has(key)) return;
      inFlight.add(key);
      try {
        await refresh();
      } finally {
        inFlight.delete(key);
      }
    };
    const intervals = [
      window.setInterval(
        () =>
          void runRefresh("datasets", () =>
            handleRefreshDatasets(defaultContainerName, {
              background: true,
              hydrateState: true,
              persistCache: true,
            }),
          ),
        10 * 60 * 1000,
      ),
      window.setInterval(
        () =>
          void runRefresh("models", () =>
            handleRefreshModels(defaultContainerName, {
              background: true,
              hydrateState: true,
              persistCache: true,
            }),
          ),
        15 * 60 * 1000,
      ),
      window.setInterval(
        () =>
          void runRefresh("tests", () =>
            handleRefreshTests(defaultEvaluateContainerName, {
              background: true,
              hydrateState: true,
              persistCache: true,
            }),
          ),
        10 * 60 * 1000,
      ),
      window.setInterval(
        () =>
          void runRefresh("evaluation-results", () =>
            handleRefreshEvaluationResults(defaultEvaluateContainerName, {
              background: true,
              hydrateState: true,
              persistCache: true,
            }),
          ),
        5 * 60 * 1000,
      ),
    ];

    return () =>
      intervals.forEach((interval) => window.clearInterval(interval));
  }, [
    cacheHydrated,
    projectName,
    defaultContainerName,
    defaultEvaluateContainerName,
    handleRefreshDatasets,
    handleRefreshModels,
    handleRefreshTests,
    handleRefreshEvaluationResults,
  ]);

  // 用于自然语言命令的评测结果查询（无参版本）
  const onQueryEvaluationResultsZero = useCallback(async (): Promise<
    EvaluationResult[]
  > => {
    const containerName = defaultEvaluateContainerName;
    return await handleQueryEvaluationResults(containerName);
  }, [defaultEvaluateContainerName, handleQueryEvaluationResults]);

  // 下载评测结果文件
  const [downloadingResultId, setDownloadingResultId] = useState<string | null>(
    null,
  );
  const [deletingResultId, setDeletingResultId] = useState<string | null>(null);

  const handleDownloadEvaluationResult = useCallback(
    async (
      folderPath: string,
      filename: string,
      sourceResult?: EvaluationResult,
    ) => {
      const result =
        sourceResult ||
        evaluationResults.find((item) => item.folderPath === folderPath);
      setDownloadingResultId(`${folderPath}/${filename}`);

      try {
        const containerName =
          result?.containerName || defaultEvaluateContainerName;

        const response = await downloadEvaluationResultMutation.mutateAsync({
          nodeId: result?.nodeId || resourceNodeId,
          container: containerName,
          folderPath,
          filename,
        });

        if (response.success && response.data) {
          // 将 base64 转换为 Blob 并下载
          const byteCharacters = atob(response.data.fileBase64);
          const byteNumbers = new Array(byteCharacters.length);
          for (let i = 0; i < byteCharacters.length; i++) {
            byteNumbers[i] = byteCharacters.charCodeAt(i);
          }
          const byteArray = new Uint8Array(byteNumbers);

          // 根据文件类型设置 MIME
          const mimeType = filename.endsWith(".json")
            ? "application/json"
            : "text/plain";
          const blob = new Blob([byteArray], { type: mimeType });

          // 创建下载链接
          const url = window.URL.createObjectURL(blob);
          const link = document.createElement("a");
          link.href = url;
          link.download = response.data.filename;
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
          window.URL.revokeObjectURL(url);

          message.success("下载成功");
        } else {
          throw new Error(response.message || "下载失败");
        }
      } catch (error: any) {
        console.error("Download evaluation result error:", error);
        message.error(error.message || "下载过程中发生错误");
        throw error;
      } finally {
        setDownloadingResultId(null);
      }
    },
    [
      defaultEvaluateContainerName,
      downloadEvaluationResultMutation,
      evaluationResults,
      resourceNodeId,
    ],
  );

  const handleDeleteEvaluationResult = useCallback(
    async (folderPath: string, sourceResult?: EvaluationResult) => {
      setDeletingResultId(folderPath);

      try {
        const result =
          sourceResult ||
          evaluationResults.find((item) => item.folderPath === folderPath);
        const response = await deleteEvaluationResultMutation.mutateAsync({
          nodeId: result?.nodeId || resourceNodeId,
          container: result?.containerName || defaultEvaluateContainerName,
          folderPath,
        });

        if (!response.success) {
          throw new Error(response.message || "删除失败");
        }

        message.success(response.message || "删除成功");
        await handleRefreshEvaluationResults(defaultEvaluateContainerName, {
          hydrateState: true,
          persistCache: true,
        });
      } catch (error: any) {
        console.error("Delete evaluation result error:", error);
        message.error(error.message || "删除评测结果失败");
        throw error;
      } finally {
        setDeletingResultId(null);
      }
    },
    [
      defaultEvaluateContainerName,
      deleteEvaluationResultMutation,
      evaluationResults,
      handleRefreshEvaluationResults,
      resourceNodeId,
    ],
  );

  // 获取系统概览数据（不包含 GPU）
  useEffect(() => {
    if (!socket) return;

    // 加入 OverviewRoom
    socket.emit(SocketEvents.client.joinOverviewRoom);

    // 监听系统概览数据更新
    socket.on(
      SocketEvents.server.pushSystemOverviewData,
      (data: SystemOverviewData) => {
        setSystemOverviewData(data);
      },
    );

    // 监听在线用户数更新
    socket.on(
      SocketEvents.server.pushOnlineUsersCount,
      ({ count }: { count: number }) => {
        setSystemOverviewData((prev) =>
          prev ? { ...prev, onlineUsers: count } : null,
        );
      },
    );

    // 监听 GPU 信息更新
    socket.on(SocketEvents.server.pushGPUInfo, (data: GPUInfo[]) => {
      setGPUInfo(data);
    });

    // 每分钟请求一次系统概览数据（不包含 GPU）
    const interval = setInterval(() => {
      socket.emit(SocketEvents.client.requestSystemOverviewData);
    }, 60000);

    return () => {
      clearInterval(interval);
      if (gpuRefreshInterval.current) {
        clearInterval(gpuRefreshInterval.current);
      }
      socket.off(SocketEvents.server.pushSystemOverviewData);
      socket.off(SocketEvents.server.pushOnlineUsersCount);
      socket.off(SocketEvents.server.pushGPUInfo);
    };
  }, [socket]);

  const handleRefreshGPUInfo = useCallback(() => {
    socket?.emit(SocketEvents.client.requestGPUInfo);
  }, [socket]);

  useEffect(() => {
    if (activeTab === "overview" && socket) {
      socket.emit(SocketEvents.client.requestGPUInfo);

      if (gpuRefreshInterval.current) {
        clearInterval(gpuRefreshInterval.current);
      }

      gpuRefreshInterval.current = setInterval(() => {
        socket.emit(SocketEvents.client.requestGPUInfo);
      }, 60000);

      return () => {
        if (gpuRefreshInterval.current) {
          clearInterval(gpuRefreshInterval.current);
          gpuRefreshInterval.current = null;
        }
      };
    }

    if (gpuRefreshInterval.current) {
      clearInterval(gpuRefreshInterval.current);
      gpuRefreshInterval.current = null;
    }
  }, [activeTab, socket]);

  // 处理标签切换
  const handleTabChange = useCallback(
    (tab: "runs" | "overview" | "datasets" | "models" | "evaluation") => {
      setRunPageSection(tab);
      setRunPagePanelOpen(true);
    },
    [setRunPagePanelOpen, setRunPageSection],
  );

  const renderProjectRunSider = () => (
    <ProjectRunSider
      systemOverviewData={systemOverviewData}
      gpuInfo={gpuInfo}
      onRefreshGPUInfo={handleRefreshGPUInfo}
      activeTab={activeTab}
      onTabChange={handleTabChange}
      onRunClick={(runId) =>
        navigate(`/projects/${projectName}/runs/${runId}`, {
          replace: true,
        })
      }
      datasets={datasets}
      isQueryingDatasets={isQueryingDatasets}
      datasetQueryError={datasetQueryError}
      datasetErrorMessage={datasetErrorMessage}
      hasQueriedDatasets={hasQueriedDatasets}
      onQueryDatasets={handleQueryDatasets}
      onRefreshDatasets={handleRefreshDatasets}
      datasetCacheMeta={datasetCacheMeta}
      onUpload={() => setUploadModalOpen(true)}
      onDownload={handleDownload}
      onDeleteDataset={handleDeleteDataset}
      onUseDatasetForTraining={handleUseDatasetForTraining}
      onUseDatasetForPreprocess={handleUseDatasetForPreprocess}
      onLoadDatasetPreviews={handleLoadDatasetPreviews}
      isUploading={isUploading}
      downloadingId={downloadingId}
      deletingDatasetId={deletingDatasetId}
      models={models}
      isQueryingModels={isQueryingModels}
      modelQueryError={modelQueryError}
      modelErrorMessage={modelErrorMessage}
      hasQueriedModels={hasQueriedModels}
      onQueryModels={handleQueryModels}
      onRefreshModels={handleRefreshModels}
      modelCacheMeta={modelCacheMeta}
      onDeleteModel={handleDeleteModel}
      deletingModelId={deletingModelId}
      tests={tests}
      isQueryingTests={isQueryingTests}
      testQueryError={testQueryError}
      testErrorMessage={testErrorMessage}
      hasQueriedTests={hasQueriedTests}
      onQueryTests={handleQueryTests}
      onRefreshTests={handleRefreshTests}
      testCacheMeta={testCacheMeta}
      onUploadTest={() => setEvaluationUploadModalOpen(true)}
      onDownloadTest={handleDownloadTest}
      onDeleteTest={handleDeleteTest}
      onUseEvaluationForBenchmark={handleUseEvaluationForBenchmark}
      isUploadingTest={isUploadingEvaluation}
      downloadingTestId={downloadingTestId}
      deletingTestId={deletingTestId}
      evaluationResults={evaluationResults}
      isQueryingEvaluationResults={isQueryingEvaluationResults}
      evaluationResultQueryError={evaluationResultQueryError}
      evaluationResultErrorMessage={evaluationResultErrorMessage}
      hasQueriedEvaluationResults={hasQueriedEvaluationResults}
      onQueryEvaluationResults={handleQueryEvaluationResults}
      onRefreshEvaluationResults={handleRefreshEvaluationResults}
      evaluationResultCacheMeta={evaluationResultCacheMeta}
      onDownloadEvaluationResult={handleDownloadEvaluationResult}
      downloadingResultId={downloadingResultId}
      onDeleteEvaluationResult={
        isAdmin ? handleDeleteEvaluationResult : undefined
      }
      deletingResultId={deletingResultId}
    />
  );

  return (
    <ProjectRoomContextProvider project={projectName}>
      <Routes>
        <Route
          index
          element={
            <div className={layoutClassName} style={layoutStyle}>
              <div className={contentWrapperClassName}>
                <EmptyRunPage />
              </div>
              {isRunPagePanelOpen && (
                <>
                  <button
                    type="button"
                    aria-label="Close management panel"
                    onClick={() => setRunPagePanelOpen(false)}
                    className="absolute inset-0 z-10 hidden bg-slate-900/8 backdrop-blur-[1px] md:block"
                  />
                  <div className="absolute inset-y-4 left-4 z-20 w-[min(360px,calc(100%-2rem))] max-w-[360px]">
                    {renderProjectRunSider()}
                  </div>
                </>
              )}

              {/* 数据集上传弹窗 */}
              <DatasetUploadModal
                open={uploadModalOpen}
                onCancel={() => setUploadModalOpen(false)}
                onUpload={handleUpload}
                isUploading={isUploading}
                uploadProgress={uploadProgress}
              />

              {/* 评测文件上传弹窗 */}
              <EvaluationUploadModal
                open={evaluationUploadModalOpen}
                onCancel={() => setEvaluationUploadModalOpen(false)}
                onUpload={handleUploadTest}
                isUploading={isUploadingEvaluation}
                uploadProgress={evaluationUploadProgress}
              />
            </div>
          }
        />
        <Route
          path="runs"
          element={
            <div className={layoutClassName} style={layoutStyle}>
              <div className={contentWrapperClassName}>
                <EmptyRunPage />
              </div>
              {isRunPagePanelOpen && (
                <>
                  <button
                    type="button"
                    aria-label="Close management panel"
                    onClick={() => setRunPagePanelOpen(false)}
                    className="absolute inset-0 z-10 hidden bg-slate-900/8 backdrop-blur-[1px] md:block"
                  />
                  <div className="absolute inset-y-4 left-4 z-20 w-[min(360px,calc(100%-2rem))] max-w-[360px]">
                    {renderProjectRunSider()}
                  </div>
                </>
              )}

              {/* 数据集上传弹窗 */}
              <DatasetUploadModal
                open={uploadModalOpen}
                onCancel={() => setUploadModalOpen(false)}
                onUpload={handleUpload}
                isUploading={isUploading}
                uploadProgress={uploadProgress}
              />

              {/* 评测文件上传弹窗 */}
              <EvaluationUploadModal
                open={evaluationUploadModalOpen}
                onCancel={() => setEvaluationUploadModalOpen(false)}
                onUpload={handleUploadTest}
                isUploading={isUploadingEvaluation}
                uploadProgress={evaluationUploadProgress}
              />
            </div>
          }
        />
        <Route
          path="runs/:runId"
          element={
            <RunRoomContextProvider key={chatSessionKey}>
              <RunPageWithMetrics
                systemOverviewData={systemOverviewData}
                gpuInfo={gpuInfo}
                onRefreshGPUInfo={handleRefreshGPUInfo}
                activeTab={activeTab}
                onTabChange={handleTabChange}
                isMetricsSheetOpen={isMetricsSheetOpen}
                setIsMetricsSheetOpen={setIsMetricsSheetOpen}
                isInferenceSheetOpen={isInferenceSheetOpen}
                setIsInferenceSheetOpen={setIsInferenceSheetOpen}
                inferencePanelView={inferencePanelView}
                setInferencePanelView={setInferencePanelView}
                datasets={datasets}
                isQueryingDatasets={isQueryingDatasets}
                datasetQueryError={datasetQueryError}
                datasetErrorMessage={datasetErrorMessage}
                hasQueriedDatasets={hasQueriedDatasets}
                onQueryDatasets={handleQueryDatasets}
                onRefreshDatasets={handleRefreshDatasets}
                datasetCacheMeta={datasetCacheMeta}
                onUpload={() => setUploadModalOpen(true)}
                onDownload={handleDownload}
                onDeleteDataset={handleDeleteDataset}
                onUseDatasetForTraining={handleUseDatasetForTraining}
                onUseDatasetForPreprocess={handleUseDatasetForPreprocess}
                onLoadDatasetPreviews={handleLoadDatasetPreviews}
                isUploading={isUploading}
                downloadingId={downloadingId}
                deletingDatasetId={deletingDatasetId}
                models={models}
                isQueryingModels={isQueryingModels}
                modelQueryError={modelQueryError}
                modelErrorMessage={modelErrorMessage}
                hasQueriedModels={hasQueriedModels}
                onQueryModels={handleQueryModels}
                onRefreshModels={handleRefreshModels}
                modelCacheMeta={modelCacheMeta}
                onDeleteModel={handleDeleteModel}
                deletingModelId={deletingModelId}
                tests={tests}
                isQueryingTests={isQueryingTests}
                testQueryError={testQueryError}
                testErrorMessage={testErrorMessage}
                hasQueriedTests={hasQueriedTests}
                onQueryTests={handleQueryTests}
                onRefreshTests={handleRefreshTests}
                testCacheMeta={testCacheMeta}
                onUploadTest={() => setEvaluationUploadModalOpen(true)}
                onDownloadTest={handleDownloadTest}
                onDeleteTest={handleDeleteTest}
                isUploadingTest={isUploadingEvaluation}
                downloadingTestId={downloadingTestId}
                deletingTestId={deletingTestId}
                // 评测结果相关
                evaluationResults={evaluationResults}
                isQueryingEvaluationResults={isQueryingEvaluationResults}
                evaluationResultQueryError={evaluationResultQueryError}
                evaluationResultErrorMessage={evaluationResultErrorMessage}
                hasQueriedEvaluationResults={hasQueriedEvaluationResults}
                onQueryEvaluationResults={handleQueryEvaluationResults}
                onRefreshEvaluationResults={handleRefreshEvaluationResults}
                evaluationResultCacheMeta={evaluationResultCacheMeta}
                onQueryEvaluationResultsZero={onQueryEvaluationResultsZero}
                onDownloadEvaluationResult={handleDownloadEvaluationResult}
                downloadingResultId={downloadingResultId}
                onDeleteEvaluationResult={
                  isAdmin ? handleDeleteEvaluationResult : undefined
                }
                deletingResultId={deletingResultId}
                uploadModalOpen={uploadModalOpen}
                setUploadModalOpen={setUploadModalOpen}
                handleUpload={handleUpload}
                uploadProgress={uploadProgress}
                evaluationUploadModalOpen={evaluationUploadModalOpen}
                setEvaluationUploadModalOpen={setEvaluationUploadModalOpen}
                handleUploadTest={handleUploadTest}
                evaluationUploadProgress={evaluationUploadProgress}
                isQuickStartWizardOpen={isQuickStartWizardOpen}
                setIsQuickStartWizardOpen={setIsQuickStartWizardOpen}
                wizardQueryState={wizardQueryState}
                setWizardQueryState={setWizardQueryState}
                wizardDatasets={wizardDatasets}
                setWizardDatasets={setWizardDatasets}
                wizardSelectedDataset={wizardSelectedDataset}
                setWizardSelectedDataset={setWizardSelectedDataset}
                focusOnLatestRun={focusOnLatestRun}
                setFocusOnLatestRun={setFocusOnLatestRun}
                onUseEvaluationForBenchmark={handleUseEvaluationForBenchmark}
                setInputTextRef={setInputTextRef}
                randomUsername={randomUsername}
                setRandomUsername={setRandomUsername}
                chatSessionId={chatSessionId}
                fallbackUsername={usernameRef.current}
                clearContextAfter={clearContextAfter}
                onClearContext={handleClearContext}
                onRuntimeResourceGroupIdChange={setManagementResourceGroupId}
              />
            </RunRoomContextProvider>
          }
        />
      </Routes>

      {/* Template Library 对话框 */}
      <TemplateLibraryDialog
        open={isLibraryOpen}
        onOpenChange={setIsLibraryOpen}
        categories={getUnifiedCategories()}
        recentTemplates={getRecentTemplates()}
        currentInput=""
        onInsert={(content) => {
          // 设置输入框文本
          if (setInputTextRef.current) {
            setInputTextRef.current(content);
          }
          // 关闭对话框
          setIsLibraryOpen(false);
        }}
        onSaveNew={() => {
          message.info("保存新模板功能暂未实现");
        }}
        addToRecent={addToRecent}
        getIconComponent={getIconComponent}
        initialCategory={libraryInitialCategory}
      />
    </ProjectRoomContextProvider>
  );
};

export default memo(RunPage);
