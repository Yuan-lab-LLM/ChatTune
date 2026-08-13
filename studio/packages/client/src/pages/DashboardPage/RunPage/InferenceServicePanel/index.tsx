import { memo, useMemo, useCallback, useState, useEffect } from "react";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button.tsx";
import { trpc } from "@/api/trpc";
import {
  PanelLeftClose,
  RefreshCw,
  Settings,
  Activity,
  Server,
  Cpu,
  Check,
  RotateCcw,
  Gauge,
  AlertTriangle,
  AlertCircle,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent } from "@/components/ui/tabs";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  ContentType,
  ContentBlocks,
  BlockType,
} from "@shared/types/messageForm";
import { Reply } from "@shared/types";
import { useMessageApi } from "@/context/MessageApiContext.tsx";
import { ScrollArea } from "@/components/ui/scroll-area";
import { InferenceConfig, ServiceStatus } from "./types";
import { validateConfig, ValidationResult } from "./validation";

interface Props {
  onClose: () => void;
  replies: Reply[];
  onAskAI?: (blocks: ContentBlocks) => void;
  isInputDisabled?: boolean;
  view?: "config" | "status";
  isAdmin?: boolean;
}

// 从消息内容中提取文本
const extractTextFromContent = (content: ContentType): string => {
  if (typeof content === "string") {
    return content;
  }
  if (Array.isArray(content)) {
    return content
      .map((block) => {
        if (block.type === "text" && "text" in block) {
          return (block as { text: string }).text || "";
        }
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
};

const parsePercentageOrDecimal = (value: string): number | undefined => {
  const normalized = value.trim();
  if (!normalized) {
    return undefined;
  }

  if (normalized.endsWith("%")) {
    const percentValue = parseFloat(normalized.replace("%", "").trim());
    return Number.isNaN(percentValue) ? undefined : percentValue / 100;
  }

  const decimalValue = parseFloat(normalized);
  return Number.isNaN(decimalValue) ? undefined : decimalValue;
};

type ProtocolServiceStatusItem = {
  key?: string;
  name?: string;
  service?: string;
  displayName?: string;
  port?: number | string;
  status?: string;
  rawStatus?: string;
  raw_status?: string;
  node?: string;
};

type ProtocolServiceInstances = {
  items?: Array<Record<string, unknown>>;
  services?: ProtocolServiceStatusItem[];
  template_services?: ProtocolServiceStatusItem[];
  templateServices?: ProtocolServiceStatusItem[];
  port_statuses?: ProtocolServiceStatusItem[];
  portStatuses?: ProtocolServiceStatusItem[];
  ports?: Record<string, number | string>;
  summary?: Record<string, unknown>;
};

interface InferenceProtocol {
  type?: string;
  jobType?: string;
  agent?: string;
  message?: string;
  action?: string;
  config?: InferenceConfig;
  service_start?: Record<string, unknown>;
  serviceStart?: Record<string, unknown>;
  nodes?: Record<
    string,
    {
      config?: InferenceConfig;
      services?: ProtocolServiceStatusItem[];
      ports?: Record<string, number | string>;
      service_instances?: ProtocolServiceInstances;
      serviceInstances?: ProtocolServiceInstances;
      serviceStart?: Record<string, unknown>;
      service_start?: Record<string, unknown>;
      allStopped?: boolean;
      allRunning?: boolean;
    }
  >;
  services?: ProtocolServiceStatusItem[];
  ports?: Record<string, number | string>;
  service_instances?: ProtocolServiceInstances;
  serviceInstances?: ProtocolServiceInstances;
  allStopped?: boolean;
  allRunning?: boolean;
}

const extractInferenceProtocolFromText = (
  text: string,
): InferenceProtocol | null => {
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
    return JSON.parse(jsonSource.slice(0, jsonEnd)) as InferenceProtocol;
  } catch (error) {
    console.warn("Failed to parse inference protocol:", error);
    return null;
  }
};

const extractInferenceProtocolFromMetadata = (
  metadata: object | null | undefined,
): InferenceProtocol | null => {
  if (!metadata || typeof metadata !== "object") {
    return null;
  }

  const protocol = (metadata as { protocol?: unknown }).protocol;
  if (!protocol || typeof protocol !== "object") {
    return null;
  }

  return protocol as InferenceProtocol;
};

const isInferenceServiceStartProtocol = (
  protocol: InferenceProtocol | null,
): boolean =>
  protocol?.type === "job_started" &&
  protocol.jobType === "inference_service" &&
  protocol.action === "service_start";

const serviceDisplayName = (name: string) => {
  const labels: Record<string, string> = {
    VLLM_OPENAI_PORT: "VLLM OpenAI API",
    INFERENCE_PORT: "推理服务",
    UI_PORT: "UI 服务",
    DATA_ANNOTATION_PORT: "数据标注服务",
  };
  return labels[name] || name;
};

const serviceDescription = (name: string) => {
  const labels: Record<string, string> = {
    VLLM_OPENAI_PORT: "vLLM OpenAI 兼容 API 服务",
    INFERENCE_PORT: "主推理服务",
    UI_PORT: "Web UI 服务",
    DATA_ANNOTATION_PORT: "数据标注工具服务",
  };
  return labels[name];
};

const STANDARD_SERVICE_KEYS = [
  "VLLM_OPENAI_PORT",
  "INFERENCE_PORT",
  "UI_PORT",
  "DATA_ANNOTATION_PORT",
] as const;

type StandardServiceKey = (typeof STANDARD_SERVICE_KEYS)[number];

const isStandardServiceKey = (name: string): name is StandardServiceKey =>
  (STANDARD_SERVICE_KEYS as readonly string[]).includes(name);

const normalizeServiceName = (name: string): string => {
  const normalized = name.replace(/[*`]/g, "").replace(/\s+/g, "");
  const upper = normalized.toUpperCase();
  if (isStandardServiceKey(upper)) {
    return upper;
  }

  const aliases: Record<string, string> = {
    vllm: "VLLM_OPENAI_PORT",
    VLLM服务: "VLLM_OPENAI_PORT",
    vLLM服务: "VLLM_OPENAI_PORT",
    VLLMOPENAI服务: "VLLM_OPENAI_PORT",
    VLLMOPENAI端口: "VLLM_OPENAI_PORT",
    VLLMOpenAI服务: "VLLM_OPENAI_PORT",
    VLLMOpenAI端口: "VLLM_OPENAI_PORT",
    VLLMOpenAIAPI: "VLLM_OPENAI_PORT",
    VLLMAPI: "VLLM_OPENAI_PORT",
    VLLMOpenAI接口: "VLLM_OPENAI_PORT",
    vLLMAPI服务: "VLLM_OPENAI_PORT",
    vLLMOpenAIAPI: "VLLM_OPENAI_PORT",
    vLLMAPI: "VLLM_OPENAI_PORT",
    vLLMOpenAI兼容API服务: "VLLM_OPENAI_PORT",
    VLLM开放API端口: "VLLM_OPENAI_PORT",
    InferenceServer: "INFERENCE_PORT",
    INFERENCE服务端口: "INFERENCE_PORT",
    inference: "INFERENCE_PORT",
    推理服务: "INFERENCE_PORT",
    推理服务端口: "INFERENCE_PORT",
    推理接口: "INFERENCE_PORT",
    推理接口端口: "INFERENCE_PORT",
    推理引擎: "INFERENCE_PORT",
    主推理服务: "INFERENCE_PORT",
    Web界面: "UI_PORT",
    UI界面: "UI_PORT",
    UI界面服务: "UI_PORT",
    UI端口: "UI_PORT",
    UI服务端口: "UI_PORT",
    WebUI: "UI_PORT",
    ui: "UI_PORT",
    UI服务: "UI_PORT",
    DataAnnotation: "DATA_ANNOTATION_PORT",
    DATAANNOTATION端口: "DATA_ANNOTATION_PORT",
    Case2Chat: "DATA_ANNOTATION_PORT",
    case2chat: "DATA_ANNOTATION_PORT",
    数据标注: "DATA_ANNOTATION_PORT",
    数据标注端口: "DATA_ANNOTATION_PORT",
    数据标注接口: "DATA_ANNOTATION_PORT",
    数据标注接口端口: "DATA_ANNOTATION_PORT",
    数据标注服务: "DATA_ANNOTATION_PORT",
    数据标注服务端口: "DATA_ANNOTATION_PORT",
  };
  return aliases[normalized] || name;
};
const parseServiceStatusesFromText = (text: string): ServiceStatus[] | null => {
  const statuses: ServiceStatus[] = [];
  const seen = new Set<string>();
  const tablePattern = /^\s*\|(.+?)\|\s*$/gm;
  for (const match of text.matchAll(tablePattern)) {
    const cells = match[1]
      .split("|")
      .map((cell) => cell.trim().replace(/^[*`\s]+|[*`\s]+$/g, ""));
    if (cells.length < 3) {
      continue;
    }
    if (cells.every((cell) => /^:?-{2,}:?$/.test(cell.replace(/\s+/g, "")))) {
      continue;
    }
    const [rawName, rawPort, rawStatus] = cells;
    const headerText = cells.join("").toLowerCase();
    if (
      headerText.includes("服务组件端口状态") ||
      headerText.includes("servicecomponentportstatus")
    ) {
      continue;
    }
    const portMatch = rawPort.match(/\d{2,5}/);
    if (!rawName || !portMatch) {
      continue;
    }
    const normalizedName = normalizeServiceName(rawName);
    if (!isStandardServiceKey(normalizedName)) {
      continue;
    }
    const port = Number(portMatch[0]);
    const key = `${normalizedName}:${port}`;
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    statuses.push({
      name: serviceDisplayName(normalizedName),
      port,
      status: parseStatus(rawStatus),
      rawStatus: rawStatus.trim(),
      description: serviceDescription(normalizedName),
      serviceKey: normalizedName,
    });
  }

  const linePatterns = [
    /^\s*-?\s*(?:[^\w\s(（:：-]+\s*)?(.+?)\s*[（(]\s*(?:端口\s*)?(\d+)\s*(?:端口)?\s*[）)]\s*[：:]\s*(.+?)\s*$/gm,
    /^\s*-?\s*(?:[^\w\s(（:：-]+\s*)?(.+?)\s*[：:]\s*(\d+)\s*[（(]\s*(.+?)\s*[）)]\s*$/gm,
    /^\s*-?\s*(?:[^\w\s(（:：-]+\s*)?(.+?)\s*[：:]\s*(\d+)\s+(.+?)\s*$/gm,
  ];

  for (const linePattern of linePatterns) {
    for (const match of text.matchAll(linePattern)) {
      const rawName = match[1].trim();
      const normalizedName = normalizeServiceName(rawName);
      if (!isStandardServiceKey(normalizedName)) {
        continue;
      }
      const port = Number(match[2]);
      if (!Number.isFinite(port)) {
        continue;
      }
      const key = `${normalizedName}:${port}`;
      if (seen.has(key)) {
        continue;
      }
      seen.add(key);
      statuses.push({
        name: serviceDisplayName(normalizedName),
        port,
        status: parseStatus(match[3]),
        rawStatus: match[3].trim(),
        description: serviceDescription(normalizedName),
        serviceKey: normalizedName,
      });
    }
  }
  return statuses.length > 0 ? statuses : null;
};

const cleanProtocolStatusText = (value: unknown): string =>
  String(value || "")
    .replace(/[*`]/g, "")
    .trim();

const serviceInstanceStatus = (
  value: unknown,
): ServiceStatus["status"] => {
  const normalized = cleanProtocolStatusText(value).toUpperCase();
  if (normalized.includes("STARTING") || normalized.includes("启动中")) {
    return "starting";
  }
  if (
    normalized.includes("FAILED") ||
    normalized.includes("失败") ||
    normalized.includes("异常")
  ) {
    return "failed";
  }
  if (normalized.includes("DEGRADED") || normalized.includes("降级")) {
    return "degraded";
  }
  if (
    normalized.includes("RUNNING") ||
    normalized.includes("正在运行") ||
    normalized.includes("运行中")
  ) {
    return "running";
  }
  return "stopped";
};

const protocolServiceKey = (item: ProtocolServiceStatusItem): StandardServiceKey | null => {
  const candidates = [item.key, item.name, item.service, item.displayName];
  for (const candidate of candidates) {
    if (!candidate) {
      continue;
    }
    const normalized = normalizeServiceName(String(candidate).trim());
    if (isStandardServiceKey(normalized)) {
      return normalized;
    }
  }
  return null;
};

const orderedProtocolStatuses = (
  statusesByKey: Partial<Record<StandardServiceKey, ServiceStatus>>,
): ServiceStatus[] =>
  STANDARD_SERVICE_KEYS.flatMap((key) => {
    const status = statusesByKey[key];
    return status ? [status] : [];
  });

const serviceStatusesFromProtocolItems = (
  items: ProtocolServiceStatusItem[],
  nodeName = "",
): ServiceStatus[] | null => {
  const statusesByKey: Partial<Record<StandardServiceKey, ServiceStatus>> = {};

  items.forEach((item) => {
    const serviceKey = protocolServiceKey(item);
    const port = Number(item.port);
    if (!serviceKey || !Number.isFinite(port) || statusesByKey[serviceKey]) {
      return;
    }

    const rawStatus = cleanProtocolStatusText(
      item.rawStatus || item.raw_status || item.status,
    );
    statusesByKey[serviceKey] = {
      name: serviceDisplayName(serviceKey),
      port,
      status: serviceInstanceStatus(item.status || rawStatus),
      rawStatus,
      description: serviceDescription(serviceKey),
      serviceKey,
      node: nodeName || item.node || "",
      isPortStatus: true,
    };
  });

  const ordered = orderedProtocolStatuses(statusesByKey);
  return ordered.length > 0 ? ordered : null;
};

const inferredProtocolPortStatus = (
  serviceInstances: ProtocolServiceInstances | undefined,
  protocol: InferenceProtocol,
): ServiceStatus["status"] => {
  if (protocol.allRunning) {
    return "running";
  }
  if (protocol.allStopped) {
    return "stopped";
  }

  const summary = serviceInstances?.summary || {};
  const running = Number(summary.running || 0);
  const starting = Number(summary.starting || 0);
  const failed = Number(summary.failed || 0);
  const stopped = Number(summary.stopped || 0);
  if (starting > 0 && running === 0) {
    return "starting";
  }
  if (running > 0 && stopped === 0 && failed === 0) {
    return "running";
  }
  if (failed > 0 && running === 0) {
    return "failed";
  }
  return "stopped";
};

const serviceStatusesFromProtocolPorts = (
  ports: Record<string, number | string> | undefined,
  defaultStatus: ServiceStatus["status"],
  nodeName = "",
): ServiceStatus[] | null => {
  const statusesByKey: Partial<Record<StandardServiceKey, ServiceStatus>> = {};

  Object.entries(ports || {}).forEach(([name, portValue]) => {
    if (name.toLowerCase() === "master") {
      return;
    }
    const serviceKey = normalizeServiceName(name);
    const port = Number(portValue);
    if (!isStandardServiceKey(serviceKey) || !Number.isFinite(port)) {
      return;
    }
    statusesByKey[serviceKey] = {
      name: serviceDisplayName(serviceKey),
      port,
      status: defaultStatus,
      rawStatus: defaultStatus.toUpperCase(),
      description: serviceDescription(serviceKey),
      serviceKey,
      node: nodeName,
      isPortStatus: true,
    };
  });

  const ordered = orderedProtocolStatuses(statusesByKey);
  return ordered.length > 0 ? ordered : null;
};

const protocolArray = (
  value: ProtocolServiceStatusItem[] | undefined,
): ProtocolServiceStatusItem[] => (Array.isArray(value) ? value : []);

const serviceStatusesFromProtocol = (
  protocol: InferenceProtocol | null,
  nodeName = "",
): ServiceStatus[] | null => {
  if (protocol?.type !== "inference_status") {
    return null;
  }

  const serviceInstances = protocol.service_instances || protocol.serviceInstances;
  let sawStructuredSource = false;
  const structuredSources = [
    protocolArray(serviceInstances?.port_statuses),
    protocolArray(serviceInstances?.portStatuses),
    protocolArray(serviceInstances?.template_services),
    protocolArray(serviceInstances?.templateServices),
    protocolArray(serviceInstances?.services),
  ];

  for (const source of structuredSources) {
    if (source.length === 0) {
      continue;
    }
    sawStructuredSource = true;
    const statuses = serviceStatusesFromProtocolItems(source, nodeName);
    if (statuses) {
      return statuses;
    }
  }

  const ports = serviceInstances?.ports || protocol.ports;
  if (ports && Object.keys(ports).length > 0) {
    sawStructuredSource = true;
    const statuses = serviceStatusesFromProtocolPorts(
      ports,
      inferredProtocolPortStatus(serviceInstances, protocol),
      nodeName,
    );
    if (statuses) {
      return statuses;
    }
  }

  if (protocol.services?.length) {
    sawStructuredSource = true;
    const statuses = serviceStatusesFromProtocolItems(protocol.services, nodeName);
    if (statuses) {
      return statuses;
    }
    return null;
  }

  if (sawStructuredSource) {
    return null;
  }

  return protocol.message ? parseServiceStatusesFromText(protocol.message) : null;
};
type InferenceNodeStatuses = Record<string, ServiceStatus[]>;

const nodeStatusesFromProtocol = (
  protocol: InferenceProtocol | null,
): InferenceNodeStatuses | null => {
  if (protocol?.type !== "inference_status") {
    return null;
  }

  const statuses: InferenceNodeStatuses = {};
  Object.entries(protocol.nodes || {}).forEach(([nodeName, node]) => {
    const normalized = serviceStatusesFromProtocol(
      {
        type: "inference_status",
        services: node?.services,
        ports: node?.ports,
        service_instances: node?.service_instances || node?.serviceInstances,
        allRunning: node?.allRunning,
        allStopped: node?.allStopped,
      },
      nodeName,
    );
    if (normalized) {
      statuses[nodeName] = normalized;
    }
  });

  if (Object.keys(statuses).length === 0) {
    const legacyStatuses = serviceStatusesFromProtocol(protocol);
    if (legacyStatuses) {
      statuses.main = legacyStatuses;
    }
  }

  return Object.keys(statuses).length > 0 ? statuses : null;
};

const numberFromProtocolValue = (
  value: number | string | undefined,
): number | undefined => {
  if (typeof value === "number") {
    return Number.isNaN(value) ? undefined : value;
  }
  if (typeof value === "string") {
    const normalized = value.trim();
    const percentMatch = normalized.match(/^(-?\d+(?:\.\d+)?)\s*%/);
    if (percentMatch) {
      const parsedPercent = Number(percentMatch[1]);
      return Number.isNaN(parsedPercent) ? undefined : parsedPercent / 100;
    }
    const numberMatch = normalized.match(/-?\d+(?:\.\d+)?/);
    const parsed = numberMatch ? Number(numberMatch[0]) : Number(normalized);
    return Number.isNaN(parsed) ? undefined : parsed;
  }
  return undefined;
};

const cleanConfigStringValue = (
  value: unknown,
): string | undefined => {
  if (value == null) {
    return undefined;
  }
  return String(value)
    .trim()
    .replace(/^["'`]+|["'`,]+$/g, "")
    .replace(/\s*[（(][^）)]*[）)]\s*$/g, "")
    .trim();
};

const isInferenceConfigProtocol = (
  protocol: InferenceProtocol | null,
): boolean =>
  protocol?.type === "inference_config" ||
  protocol?.type === "inference_config_updated";

const configFromProtocol = (
  protocol: InferenceProtocol | null,
): InferenceConfig | null => {
  if (!isInferenceConfigProtocol(protocol)) {
    return null;
  }

  const nodeConfigs = protocol.nodes
    ? Object.values(protocol.nodes)
        .map((node) => node?.config)
        .filter((config): config is InferenceConfig => Boolean(config))
    : [];
  const source =
    protocol.config || protocol.nodes?.main?.config || nodeConfigs[0];
  if (!source) {
    return null;
  }
  const config: InferenceConfig = {
    ports: {},
    env: {},
    runtime: {},
  };

  config.ports.VLLM_OPENAI_PORT = numberFromProtocolValue(
    source.ports?.VLLM_OPENAI_PORT,
  );
  config.ports.INFERENCE_PORT = numberFromProtocolValue(
    source.ports?.INFERENCE_PORT,
  );
  config.ports.UI_PORT = numberFromProtocolValue(source.ports?.UI_PORT);
  config.ports.DATA_ANNOTATION_PORT = numberFromProtocolValue(
    source.ports?.DATA_ANNOTATION_PORT,
  );

  config.env.HOST_IP = cleanConfigStringValue(source.env?.HOST_IP);
  config.env.CUDA_VISIBLE_DEVICES = cleanConfigStringValue(
    source.env?.CUDA_VISIBLE_DEVICES,
  );
  config.env.MODEL_NAME = cleanConfigStringValue(source.env?.MODEL_NAME);
  config.env.MODEL_PARAM_B = cleanConfigStringValue(source.env?.MODEL_PARAM_B);
  config.env.MODEL_PATH = cleanConfigStringValue(source.env?.MODEL_PATH);
  config.env.START_SCRIPT = cleanConfigStringValue(source.env?.START_SCRIPT);
  config.env.LOG_DIR = cleanConfigStringValue(source.env?.LOG_DIR);
  config.env.TEST_DIR = cleanConfigStringValue(source.env?.TEST_DIR);
  config.env.BENCHMARK_DIR = cleanConfigStringValue(source.env?.BENCHMARK_DIR);
  config.env.GENERAL_BENCHMARK_DIR = cleanConfigStringValue(
    source.env?.GENERAL_BENCHMARK_DIR,
  );
  config.env.MASTER_PORT = numberFromProtocolValue(source.env?.MASTER_PORT);

  config.runtime.TENSOR_PARALLEL_SIZE = numberFromProtocolValue(
    source.runtime?.TENSOR_PARALLEL_SIZE,
  );
  config.runtime.GPU_MEMORY_UTILIZATION = numberFromProtocolValue(
    source.runtime?.GPU_MEMORY_UTILIZATION,
  );
  config.runtime.GPU_UTILIZATION_THRESHOLD = numberFromProtocolValue(
    source.runtime?.GPU_UTILIZATION_THRESHOLD,
  );
  config.runtime.MAX_TOKENS = numberFromProtocolValue(
    source.runtime?.MAX_TOKENS,
  );

  Object.keys(config.ports).forEach((key) => {
    if (config.ports[key as keyof InferenceConfig["ports"]] === undefined) {
      delete config.ports[key as keyof InferenceConfig["ports"]];
    }
  });
  Object.keys(config.env).forEach((key) => {
    if (config.env[key as keyof InferenceConfig["env"]] === undefined) {
      delete config.env[key as keyof InferenceConfig["env"]];
    }
  });
  Object.keys(config.runtime).forEach((key) => {
    if (config.runtime[key as keyof InferenceConfig["runtime"]] === undefined) {
      delete config.runtime[key as keyof InferenceConfig["runtime"]];
    }
  });

  return Object.keys(config.ports).length > 0 ||
    Object.keys(config.env).length > 0 ||
    Object.keys(config.runtime).length > 0
    ? config
    : null;
};

type InferenceNodeConfigs = Record<string, InferenceConfig>;

const nodeConfigsFromProtocol = (
  protocol: InferenceProtocol | null,
): InferenceNodeConfigs | null => {
  if (!isInferenceConfigProtocol(protocol)) {
    return null;
  }

  const configs: InferenceNodeConfigs = {};
  Object.entries(protocol.nodes || {}).forEach(([nodeName, node]) => {
    if (!node?.config) {
      return;
    }
    const normalized = configFromProtocol({
      type: "inference_config",
      config: node.config,
    });
    if (normalized) {
      configs[nodeName] = normalized;
    }
  });

  if (Object.keys(configs).length === 0 && protocol.config) {
    const normalized = configFromProtocol(protocol);
    if (normalized) {
      configs.main = normalized;
    }
  }

  return Object.keys(configs).length > 0 ? configs : null;
};

const asRecord = (value: unknown): Record<string, unknown> =>
  value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};

const hasAnyKey = (record: Record<string, unknown>, keys: string[]): boolean =>
  keys.some((key) => record[key] !== undefined);

const stringFromJsonValue = (value: unknown): string | undefined => {
  if (value === undefined || value === null) {
    return undefined;
  }
  return cleanConfigStringValue(String(value));
};

const configFromJsonPayload = (payload: unknown): InferenceConfig | null => {
  const root = asRecord(payload);
  const ports = asRecord(root.ports ?? root.PORTS);
  const env = asRecord(root.env ?? root.ENV);
  const runtime = asRecord(root.runtime ?? root.RUNTIME);
  const config: InferenceConfig = {
    ports: {},
    env: {},
    runtime: {},
  };

  config.ports.VLLM_OPENAI_PORT = numberFromProtocolValue(
    ports.VLLM_OPENAI_PORT as number | string | undefined,
  );
  config.ports.INFERENCE_PORT = numberFromProtocolValue(
    ports.INFERENCE_PORT as number | string | undefined,
  );
  config.ports.UI_PORT = numberFromProtocolValue(
    ports.UI_PORT as number | string | undefined,
  );
  config.ports.DATA_ANNOTATION_PORT = numberFromProtocolValue(
    ports.DATA_ANNOTATION_PORT as number | string | undefined,
  );

  config.env.HOST_IP = stringFromJsonValue(env.HOST_IP);
  config.env.CUDA_VISIBLE_DEVICES = stringFromJsonValue(
    env.CUDA_VISIBLE_DEVICES,
  );
  config.env.MODEL_NAME = stringFromJsonValue(env.MODEL_NAME);
  config.env.MODEL_PARAM_B = stringFromJsonValue(env.MODEL_PARAM_B);
  config.env.MODEL_PATH = stringFromJsonValue(env.MODEL_PATH);
  config.env.START_SCRIPT = stringFromJsonValue(env.START_SCRIPT);
  config.env.LOG_DIR = stringFromJsonValue(env.LOG_DIR);
  config.env.TEST_DIR = stringFromJsonValue(env.TEST_DIR);
  config.env.BENCHMARK_DIR = stringFromJsonValue(env.BENCHMARK_DIR);
  config.env.GENERAL_BENCHMARK_DIR = stringFromJsonValue(
    env.GENERAL_BENCHMARK_DIR,
  );
  config.env.MASTER_PORT = numberFromProtocolValue(env.MASTER_PORT as any);

  config.runtime.TENSOR_PARALLEL_SIZE = numberFromProtocolValue(
    runtime.TENSOR_PARALLEL_SIZE as number | string | undefined,
  );
  config.runtime.GPU_MEMORY_UTILIZATION = numberFromProtocolValue(
    runtime.GPU_MEMORY_UTILIZATION as any,
  );
  config.runtime.GPU_UTILIZATION_THRESHOLD = numberFromProtocolValue(
    runtime.GPU_UTILIZATION_THRESHOLD as any,
  );
  config.runtime.MAX_TOKENS = numberFromProtocolValue(
    runtime.MAX_TOKENS as number | string | undefined,
  );

  Object.keys(config.ports).forEach((key) => {
    if (config.ports[key as keyof InferenceConfig["ports"]] === undefined) {
      delete config.ports[key as keyof InferenceConfig["ports"]];
    }
  });
  Object.keys(config.env).forEach((key) => {
    if (config.env[key as keyof InferenceConfig["env"]] === undefined) {
      delete config.env[key as keyof InferenceConfig["env"]];
    }
  });
  Object.keys(config.runtime).forEach((key) => {
    if (config.runtime[key as keyof InferenceConfig["runtime"]] === undefined) {
      delete config.runtime[key as keyof InferenceConfig["runtime"]];
    }
  });

  return Object.keys(config.ports).length > 0 ||
    Object.keys(config.env).length > 0 ||
    Object.keys(config.runtime).length > 0
    ? config
    : null;
};

const mergeConfig = (
  target: InferenceConfig,
  source: InferenceConfig | null,
): void => {
  if (!source) {
    return;
  }
  Object.assign(target.ports, source.ports);
  Object.assign(target.env, source.env);
  Object.assign(target.runtime, source.runtime);
};

const configFromJsonBlocks = (text: string): InferenceConfig | null => {
  const merged: InferenceConfig = {
    ports: {},
    env: {},
    runtime: {},
  };
  for (const match of text.matchAll(/```json\s*\n([\s\S]*?)```/gi)) {
    try {
      const payload = JSON.parse(match[1]);
      const wrappedConfig = configFromJsonPayload(payload);
      if (wrappedConfig) {
        mergeConfig(merged, wrappedConfig);
      }
      const record = asRecord(payload);
      const context = text.slice(Math.max(0, match.index - 180), match.index);
      const contextUpper = context.toUpperCase();
      if (
        contextUpper.includes("PORT") ||
        context.includes("端口") ||
        hasAnyKey(record, [
          "VLLM_OPENAI_PORT",
          "INFERENCE_PORT",
          "UI_PORT",
          "DATA_ANNOTATION_PORT",
        ])
      ) {
        merged.ports.VLLM_OPENAI_PORT =
          numberFromProtocolValue(
            record.VLLM_OPENAI_PORT as number | string | undefined,
          ) ?? merged.ports.VLLM_OPENAI_PORT;
        merged.ports.INFERENCE_PORT =
          numberFromProtocolValue(
            record.INFERENCE_PORT as number | string | undefined,
          ) ?? merged.ports.INFERENCE_PORT;
        merged.ports.UI_PORT =
          numberFromProtocolValue(
            record.UI_PORT as number | string | undefined,
          ) ?? merged.ports.UI_PORT;
        merged.ports.DATA_ANNOTATION_PORT =
          numberFromProtocolValue(
            record.DATA_ANNOTATION_PORT as number | string | undefined,
          ) ?? merged.ports.DATA_ANNOTATION_PORT;
      } else if (
        contextUpper.includes("RUNTIME") ||
        context.includes("运行时") ||
        context.includes("参数") ||
        hasAnyKey(record, [
          "TENSOR_PARALLEL_SIZE",
          "GPU_MEMORY_UTILIZATION",
          "GPU_UTILIZATION_THRESHOLD",
          "MAX_TOKENS",
        ])
      ) {
        merged.runtime.TENSOR_PARALLEL_SIZE =
          numberFromProtocolValue(
            record.TENSOR_PARALLEL_SIZE as number | string | undefined,
          ) ?? merged.runtime.TENSOR_PARALLEL_SIZE;
        merged.runtime.GPU_MEMORY_UTILIZATION =
          numberFromProtocolValue(
            record.GPU_MEMORY_UTILIZATION as any,
          ) ?? merged.runtime.GPU_MEMORY_UTILIZATION;
        merged.runtime.GPU_UTILIZATION_THRESHOLD =
          numberFromProtocolValue(
            record.GPU_UTILIZATION_THRESHOLD as any,
          ) ?? merged.runtime.GPU_UTILIZATION_THRESHOLD;
        merged.runtime.MAX_TOKENS =
          numberFromProtocolValue(
            record.MAX_TOKENS as number | string | undefined,
          ) ?? merged.runtime.MAX_TOKENS;
      } else {
        merged.env.HOST_IP =
          stringFromJsonValue(record.HOST_IP) ?? merged.env.HOST_IP;
        merged.env.CUDA_VISIBLE_DEVICES =
          stringFromJsonValue(record.CUDA_VISIBLE_DEVICES) ??
          merged.env.CUDA_VISIBLE_DEVICES;
        merged.env.MODEL_NAME =
          stringFromJsonValue(record.MODEL_NAME) ?? merged.env.MODEL_NAME;
        merged.env.MODEL_PARAM_B =
          stringFromJsonValue(record.MODEL_PARAM_B) ?? merged.env.MODEL_PARAM_B;
        merged.env.MODEL_PATH =
          stringFromJsonValue(record.MODEL_PATH) ?? merged.env.MODEL_PATH;
        merged.env.START_SCRIPT =
          stringFromJsonValue(record.START_SCRIPT) ?? merged.env.START_SCRIPT;
        merged.env.LOG_DIR =
          stringFromJsonValue(record.LOG_DIR) ?? merged.env.LOG_DIR;
        merged.env.TEST_DIR =
          stringFromJsonValue(record.TEST_DIR) ?? merged.env.TEST_DIR;
        merged.env.BENCHMARK_DIR =
          stringFromJsonValue(record.BENCHMARK_DIR) ?? merged.env.BENCHMARK_DIR;
        merged.env.GENERAL_BENCHMARK_DIR =
          stringFromJsonValue(record.GENERAL_BENCHMARK_DIR) ??
          merged.env.GENERAL_BENCHMARK_DIR;
        merged.env.MASTER_PORT =
          numberFromProtocolValue(record.MASTER_PORT as any) ?? merged.env.MASTER_PORT;
      }
    } catch {
      // Ignore malformed JSON snippets and keep trying other parsers.
    }
  }
  return Object.keys(merged.ports).length > 0 ||
    Object.keys(merged.env).length > 0 ||
    Object.keys(merged.runtime).length > 0
    ? merged
    : null;
};

// 从消息历史中解析推理配置
const parseInferenceConfigFromReplies = (
  replies: Reply[],
): InferenceConfig | null => {
  for (let i = replies.length - 1; i >= 0; i--) {
    const reply = replies[i];
    // Prefer structured protocol payloads from the newest reply before falling back to prose parsing.
    for (
      let messageIndex = reply.messages.length - 1;
      messageIndex >= 0;
      messageIndex -= 1
    ) {
      const msg = reply.messages[messageIndex];
      const metadataConfig = configFromProtocol(
        extractInferenceProtocolFromMetadata(msg.metadata),
      );
      if (metadataConfig) {
        return metadataConfig;
      }

      const text = extractTextFromContent(msg.content);
      const protocolConfig = configFromProtocol(
        extractInferenceProtocolFromText(text),
      );
      if (protocolConfig) {
        return protocolConfig;
      }
    }

    for (
      let messageIndex = reply.messages.length - 1;
      messageIndex >= 0;
      messageIndex -= 1
    ) {
      const msg = reply.messages[messageIndex];
      const text = extractTextFromContent(msg.content);
      const jsonBlockConfig = configFromJsonBlocks(text);
      if (jsonBlockConfig) {
        return jsonBlockConfig;
      }

      const normalizedText = text.replace(/\*\*/g, "");

      // 检查是否包含推理配置信息
      const hasStructuredConfig =
        normalizedText.includes("VLLM_OPENAI_PORT") ||
        normalizedText.includes("推理配置") ||
        normalizedText.includes("当前推理服务配置如下") ||
        (normalizedText.includes("端口配置") &&
          normalizedText.includes("环境配置"));

      if (hasStructuredConfig) {
        const config: InferenceConfig = {
          ports: {},
          env: {},
          runtime: {},
        };

        // 解析端口配置（支持带反引号的Markdown格式）
        const vllmPortMatch = normalizedText.match(
          /[`"]?VLLM_OPENAI_PORT[`"]?[:：]\s*(\d+)/,
        );
        if (vllmPortMatch)
          config.ports.VLLM_OPENAI_PORT = parseInt(vllmPortMatch[1]);

        const inferencePortMatch = normalizedText.match(
          /[`"]?INFERENCE_PORT[`"]?[:：]\s*(\d+)/,
        );
        if (inferencePortMatch)
          config.ports.INFERENCE_PORT = parseInt(inferencePortMatch[1]);

        const uiPortMatch = normalizedText.match(
          /[`"]?UI_PORT[`"]?[:：]\s*(\d+)/,
        );
        if (uiPortMatch) config.ports.UI_PORT = parseInt(uiPortMatch[1]);

        const dataPortMatch = normalizedText.match(
          /[`"]?DATA_ANNOTATION_PORT[`"]?[:：]\s*(\d+)/,
        );
        if (dataPortMatch)
          config.ports.DATA_ANNOTATION_PORT = parseInt(dataPortMatch[1]);

        const vllmServicePortMatch = text.match(
          /VLLM\s*(?:OpenAI\s*)?服务端口[:：]\s*(\d+)/i,
        );
        if (vllmServicePortMatch)
          config.ports.VLLM_OPENAI_PORT = parseInt(vllmServicePortMatch[1], 10);

        const vllmOpenApiPortMatch = text.match(/VLLM开放API端口[:：]\s*(\d+)/);
        if (vllmOpenApiPortMatch)
          config.ports.VLLM_OPENAI_PORT = parseInt(vllmOpenApiPortMatch[1], 10);

        const inferenceServicePortMatch =
          text.match(/推理服务端口[:：]\s*(\d+)/);
        if (inferenceServicePortMatch)
          config.ports.INFERENCE_PORT = parseInt(
            inferenceServicePortMatch[1],
            10,
          );

        const uiServicePortMatch = text.match(
          /(?:UI服务|用户界面)端口[:：]\s*(\d+)/,
        );
        if (uiServicePortMatch)
          config.ports.UI_PORT = parseInt(uiServicePortMatch[1], 10);

        const annotationServicePortMatch = text.match(
          /数据标注(?:服务)?端口[:：]\s*(\d+)/,
        );
        if (annotationServicePortMatch)
          config.ports.DATA_ANNOTATION_PORT = parseInt(
            annotationServicePortMatch[1],
            10,
          );

        // 解析环境配置
        const hostIpMatch = normalizedText.match(
          /[`"]?HOST_IP[`"]?\s*[:：]\s*["']?([\d.]+)["']?,?/,
        );
        if (hostIpMatch) config.env.HOST_IP = hostIpMatch[1];

        const cudaMatch = normalizedText.match(
          /[`"]?CUDA_VISIBLE_DEVICES[`"]?\s*[:：]\s*["']?([\d,]+)["']?,?/,
        );
        if (cudaMatch) config.env.CUDA_VISIBLE_DEVICES = cudaMatch[1];

        const modelNameMatch = normalizedText.match(
          /[`"]?MODEL_NAME[`"]?\s*[:：]\s*["']?([^"',\n\r]+)["']?,?/,
        );
        if (modelNameMatch)
          config.env.MODEL_NAME = cleanConfigStringValue(modelNameMatch[1]);

        const modelParamMatch = normalizedText.match(
          /[`"]?MODEL_PARAM_B[`"]?\s*[:：]\s*["']?([^"',\n\r]+)["']?,?/,
        );
        if (modelParamMatch)
          config.env.MODEL_PARAM_B = cleanConfigStringValue(modelParamMatch[1]);

        const modelPathMatch = normalizedText.match(
          /[`"]?MODEL_PATH[`"]?\s*[:：]\s*["']?([^"',\n\r]+)["']?,?/,
        );
        if (modelPathMatch)
          config.env.MODEL_PATH = cleanConfigStringValue(modelPathMatch[1]);

        const startScriptMatch = normalizedText.match(
          /[`"]?START_SCRIPT[`"]?\s*[:：]\s*["']?([^"',\n\r]+)["']?,?/,
        );
        if (startScriptMatch)
          config.env.START_SCRIPT = cleanConfigStringValue(startScriptMatch[1]);

        const logDirMatch = normalizedText.match(
          /[`"]?LOG_DIR[`"]?\s*[:：]\s*["']?([^"',\n\r]+)["']?,?/,
        );
        if (logDirMatch)
          config.env.LOG_DIR = cleanConfigStringValue(logDirMatch[1]);

        const testDirMatch = normalizedText.match(
          /[`"]?TEST_DIR[`"]?\s*[:：]\s*["']?([^"',\n\r]+)["']?,?/,
        );
        if (testDirMatch)
          config.env.TEST_DIR = cleanConfigStringValue(testDirMatch[1]);

        const benchmarkDirMatch = normalizedText.match(
          /[`"]?BENCHMARK_DIR[`"]?\s*[:：]\s*["']?([^"',\n\r]+)["']?,?/,
        );
        if (benchmarkDirMatch)
          config.env.BENCHMARK_DIR = cleanConfigStringValue(
            benchmarkDirMatch[1],
          );

        const generalBenchmarkDirMatch = normalizedText.match(
          /[`"]?GENERAL_BENCHMARK_DIR[`"]?\s*[:：]\s*["']?([^"',\n\r]+)["']?,?/,
        );
        if (generalBenchmarkDirMatch)
          config.env.GENERAL_BENCHMARK_DIR = cleanConfigStringValue(
            generalBenchmarkDirMatch[1],
          );

        const masterPortMatch = normalizedText.match(
          /[`"]?MASTER_PORT[`"]?\s*[:：]\s*(\d+)/,
        );
        if (masterPortMatch) config.env.MASTER_PORT = parseInt(masterPortMatch[1], 10);

        const hostIpLabelMatch = text.match(/主机\s*IP(?:地址)?[:：]\s*([^\s]+)/);
        if (hostIpLabelMatch) config.env.HOST_IP = hostIpLabelMatch[1];

        const gpuDevicesLabelMatch = text.match(
          /GPU(?:设备|分配)[:：]\s*([^\n\r]+)/,
        );
        if (gpuDevicesLabelMatch)
          config.env.CUDA_VISIBLE_DEVICES = cleanConfigStringValue(
            gpuDevicesLabelMatch[1],
          );

        const visibleGpuDevicesLabelMatch = text.match(
          /可见\s*GPU\s*设备[:：]\s*([^\n\r]+)/,
        );
        if (visibleGpuDevicesLabelMatch)
          config.env.CUDA_VISIBLE_DEVICES = cleanConfigStringValue(
            visibleGpuDevicesLabelMatch[1],
          );

        const modelNameLabelMatch = text.match(/模型名称[:：]\s*([^\n\r]+)/);
        if (modelNameLabelMatch)
          config.env.MODEL_NAME = cleanConfigStringValue(
            modelNameLabelMatch[1],
          );

        const modelParamLabelMatch = text.match(/模型参数量[:：]\s*([^\n\r]+)/);
        if (modelParamLabelMatch)
          config.env.MODEL_PARAM_B = cleanConfigStringValue(
            modelParamLabelMatch[1],
          );

        const modelPathLabelMatch = text.match(/模型路径[:：]\s*([^\n\r]+)/);
        if (modelPathLabelMatch)
          config.env.MODEL_PATH = cleanConfigStringValue(
            modelPathLabelMatch[1],
          );

        const startScriptLabelMatch = text.match(
          /启动脚本路径[:：]\s*([^\n\r]+)/,
        );
        if (startScriptLabelMatch)
          config.env.START_SCRIPT = cleanConfigStringValue(
            startScriptLabelMatch[1],
          );

        const logDirLabelMatch = text.match(/日志目录[:：]\s*([^\n\r]+)/);
        if (logDirLabelMatch)
          config.env.LOG_DIR = cleanConfigStringValue(logDirLabelMatch[1]);

        const testDirLabelMatch = text.match(/测试目录[:：]\s*([^\n\r]+)/);
        if (testDirLabelMatch)
          config.env.TEST_DIR = cleanConfigStringValue(testDirLabelMatch[1]);

        const benchmarkDirLabelMatch = text.match(
          /医疗基准测试目录[:：]\s*([^\n\r]+)/,
        );
        if (benchmarkDirLabelMatch)
          config.env.BENCHMARK_DIR = cleanConfigStringValue(
            benchmarkDirLabelMatch[1],
          );

        const generalBenchmarkDirLabelMatch = text.match(
          /通用基准测试目录[:：]\s*([^\n\r]+)/,
        );
        if (generalBenchmarkDirLabelMatch)
          config.env.GENERAL_BENCHMARK_DIR = cleanConfigStringValue(
            generalBenchmarkDirLabelMatch[1],
          );

        const masterPortLabelMatch = text.match(/主端口[:：]\s*(\d+)/);
        if (masterPortLabelMatch)
          config.env.MASTER_PORT = parseInt(masterPortLabelMatch[1], 10);

        // 解析运行时配置
        const tensorMatch = normalizedText.match(
          /[`"]?TENSOR_PARALLEL_SIZE[`"]?[:：]\s*(\d+)/,
        );
        if (tensorMatch)
          config.runtime.TENSOR_PARALLEL_SIZE = parseInt(tensorMatch[1]);

        const gpuMemMatch = normalizedText.match(
          /[`"]?GPU_MEMORY_UTILIZATION[`"]?[:：]\s*([0-9.]+%?)/,
        );
        if (gpuMemMatch) {
          const parsedGpuMem = parsePercentageOrDecimal(gpuMemMatch[1]);
          if (parsedGpuMem !== undefined) {
            config.runtime.GPU_MEMORY_UTILIZATION = parsedGpuMem;
          }
        }

        const gpuThresholdMatch = normalizedText.match(
          /[`"]?GPU_UTILIZATION_THRESHOLD[`"]?\s*[:：]\s*([0-9.]+%?)/,
        );
        if (gpuThresholdMatch) {
          const parsedThreshold = parsePercentageOrDecimal(gpuThresholdMatch[1]);
          if (parsedThreshold !== undefined) {
            config.runtime.GPU_UTILIZATION_THRESHOLD = parsedThreshold;
          }
        }

        const maxTokensMatch = normalizedText.match(
          /[`"]?MAX_TOKENS[`"]?[:：]\s*(\d+)/,
        );
        if (maxTokensMatch)
          config.runtime.MAX_TOKENS = parseInt(maxTokensMatch[1]);

        const tensorLabelMatch = text.match(
          /张量并行(?:度|数|规模|尺寸|大小)(?:\s*[（(][^）)]*[）)])?[:：]\s*(\d+)/,
        );
        if (tensorLabelMatch)
          config.runtime.TENSOR_PARALLEL_SIZE = parseInt(
            tensorLabelMatch[1],
            10,
          );

        const gpuMemLabelMatch = text.match(/GPU\s*(?:内存|显存)利用率[:：]\s*([0-9.]+%?)/);
        if (gpuMemLabelMatch) {
          const parsedGpuMem = parsePercentageOrDecimal(gpuMemLabelMatch[1]);
          if (parsedGpuMem !== undefined) {
            config.runtime.GPU_MEMORY_UTILIZATION = parsedGpuMem;
          }
        }

        const gpuThresholdLabelMatch = text.match(/GPU\s*利用率阈值[:：]\s*([0-9.]+%?)/);
        if (gpuThresholdLabelMatch) {
          const parsedThreshold = parsePercentageOrDecimal(gpuThresholdLabelMatch[1]);
          if (parsedThreshold !== undefined) {
            config.runtime.GPU_UTILIZATION_THRESHOLD = parsedThreshold;
          }
        }

        const maxTokensLabelMatch = text.match(
          /最大\s*(?:上下文令牌数|令牌数|token长度|Token长度|Token数|tokens?(?:限制)?|Tokens?(?:限制)?)[:：]\s*(\d+)/,
        );
        if (maxTokensLabelMatch)
          config.runtime.MAX_TOKENS = parseInt(maxTokensLabelMatch[1], 10);

        // 如果解析到了任何配置，返回配置对象
        if (
          Object.keys(config.ports).length > 0 ||
          Object.keys(config.env).length > 0 ||
          Object.keys(config.runtime).length > 0
        ) {
          return config;
        }
      }
    }
  }
  return null;
};

const parseInferenceNodeConfigsFromReplies = (
  replies: Reply[],
): InferenceNodeConfigs | null => {
  for (let replyIndex = replies.length - 1; replyIndex >= 0; replyIndex -= 1) {
    const reply = replies[replyIndex];
    for (
      let messageIndex = reply.messages.length - 1;
      messageIndex >= 0;
      messageIndex -= 1
    ) {
      const message = reply.messages[messageIndex];
      const metadataConfigs = nodeConfigsFromProtocol(
        extractInferenceProtocolFromMetadata(message.metadata),
      );
      if (metadataConfigs) {
        return metadataConfigs;
      }

      const textConfigs = nodeConfigsFromProtocol(
        extractInferenceProtocolFromText(
          extractTextFromContent(message.content),
        ),
      );
      if (textConfigs) {
        return textConfigs;
      }
    }
  }

  const legacyConfig = parseInferenceConfigFromReplies(replies);
  return legacyConfig ? { main: legacyConfig } : null;
};

const sliceRepliesAfterMessageCount = (
  replies: Reply[],
  messageCount: number,
): Reply[] => {
  let remainingMessages = messageCount;
  const slicedReplies: Reply[] = [];

  replies.forEach((reply) => {
    if (remainingMessages >= reply.messages.length) {
      remainingMessages -= reply.messages.length;
      return;
    }

    if (remainingMessages > 0) {
      slicedReplies.push({
        ...reply,
        messages: reply.messages.slice(remainingMessages),
      });
      remainingMessages = 0;
      return;
    }

    slicedReplies.push(reply);
  });

  return slicedReplies;
};

const selectConfigForNode = (
  configs: InferenceNodeConfigs,
  selectedNode: string,
): InferenceConfig | null => {
  const names = Object.keys(configs);
  return configs[selectedNode] || configs.main || configs[names[0]] || null;
};

// 状态映射：将中英文状态统一转换为 running/stopped
const parseStatus = (statusText: string): ServiceStatus["status"] => {
  const normalized = statusText.trim().toUpperCase();

  if (normalized.includes("STARTING") || normalized.includes("启动中")) {
    return "starting";
  }

  if (normalized.includes("FAILED") || normalized.includes("失败") || normalized.includes("异常")) {
    return "failed";
  }

  if (normalized.includes("DEGRADED") || normalized.includes("降级")) {
    return "degraded";
  }

  if (
    normalized.includes("RUNNING") ||
    normalized.includes("正在运行") ||
    normalized.includes("运行中")
  ) {
    return "running";
  }

  return "stopped";
};

const statusTokenPattern =
  "(正在运行|运行中|启动中|已停止|停止|未运行|未启动|失败|异常|降级|RUNNING|STARTING|STOPPED|FAILED|DEGRADED)";

const parsePortStatusLine = (
  line: string,
  portName: string,
): { port: number; status: ServiceStatus["status"] } | null => {
  const normalizedLine = line
    .replace(/[*`]/g, "")
    .replace(/^[\s-]+/, "")
    .trim();
  const escapedPortName = portName.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const patterns = [
    // 兼容 markdown 列表/加粗格式，例如：
    // - **VLLM_OPENAI_PORT (8010)**: 正在运行
    new RegExp(
      `^${escapedPortName}\\s*\\(\\s*(\\d+)\\s*\\)\\s*[:：]\\s*${statusTokenPattern}`,
      "i",
    ),
    // 兼容 "PORT: 正在运行 (8010)" 这类格式
    new RegExp(
      `^${escapedPortName}\\s*[:：]\\s*(${statusTokenPattern}).*?\\(\\s*(\\d+)\\s*\\)`,
      "i",
    ),
    // 兼容 "PORT (端口 8010): 停止"
    new RegExp(
      `^${escapedPortName}\\s*\\(\\s*端口\\s*(\\d+)\\s*\\)\\s*[:：]\\s*${statusTokenPattern}`,
      "i",
    ),
    // 兜底：允许端口名和端口之间有任意少量字符
    new RegExp(
      `${escapedPortName}[^\\n\\r:：]*?(\\d+)[^\\n\\r:：]*?[:：]\\s*${statusTokenPattern}`,
      "i",
    ),
  ];

  for (const pattern of patterns) {
    const match = normalizedLine.match(pattern);
    if (!match) {
      continue;
    }

    const first = match[1];
    const second = match[2];

    const port = /^\d+$/.test(first)
      ? parseInt(first, 10)
      : parseInt(second || "", 10);
    const statusText = /^\d+$/.test(first) ? second || "" : first;

    if (!Number.isNaN(port)) {
      return {
        port,
        status: parseStatus(statusText),
      };
    }
  }

  return null;
};

// 从消息历史中解析服务状态
const parseServiceStatusFromReplies = (replies: Reply[]): ServiceStatus[] => {
  const emptyStatuses: ServiceStatus[] = [];
  const parsedConfig = parseInferenceConfigFromReplies(replies);
  const serviceDefinitions = [
    {
      portName: "VLLM_OPENAI_PORT",
      name: "VLLM OpenAI API",
      description: "vLLM OpenAI 兼容 API 服务",
    },
    {
      portName: "INFERENCE_PORT",
      name: "推理服务",
      description: "主推理服务",
    },
    {
      portName: "UI_PORT",
      name: "UI 服务",
      description: "Web UI 服务",
    },
    {
      portName: "DATA_ANNOTATION_PORT",
      name: "数据标注服务",
      description: "数据标注工具服务",
    },
  ] as const;
  const portMap: Record<
    (typeof serviceDefinitions)[number]["portName"],
    number | undefined
  > = {
    VLLM_OPENAI_PORT: parsedConfig?.ports.VLLM_OPENAI_PORT ?? 7111,
    INFERENCE_PORT: parsedConfig?.ports.INFERENCE_PORT ?? 7013,
    UI_PORT: parsedConfig?.ports.UI_PORT ?? 7860,
    DATA_ANNOTATION_PORT: parsedConfig?.ports.DATA_ANNOTATION_PORT ?? 7016,
  };

  for (let i = replies.length - 1; i >= 0; i--) {
    const reply = replies[i];
    // 遍历 reply 中的所有 messages
    for (const msg of reply.messages) {
      const metadataProtocol = extractInferenceProtocolFromMetadata(msg.metadata);
      if (isInferenceServiceStartProtocol(metadataProtocol)) {
        continue;
      }
      const metadataStatuses = serviceStatusesFromProtocol(metadataProtocol);
      if (metadataStatuses) {
        return metadataStatuses;
      }

      const text = extractTextFromContent(msg.content);
      const textProtocol = extractInferenceProtocolFromText(text);
      if (isInferenceServiceStartProtocol(textProtocol)) {
        continue;
      }
      const protocolStatuses = serviceStatusesFromProtocol(textProtocol);
      if (protocolStatuses) {
        return protocolStatuses;
      }

      const textStatuses = parseServiceStatusesFromText(text);
      if (textStatuses) {
        return textStatuses;
      }

      // 检查是否包含服务状态信息（支持中英文）
      const hasStatusInfo =
        (text.includes("推理服务") || text.includes("服务状态")) &&
        (text.includes("运行") ||
          text.includes("停止") ||
          text.includes("RUNNING") ||
          text.includes("STOPPED"));

      if (hasStatusInfo) {
        const lines = text
          .split(/\r?\n/)
          .map((line) => line.trim())
          .filter(Boolean);
        const statuses: ServiceStatus[] = [];

        for (const service of serviceDefinitions) {
          for (const line of lines) {
            const parsed = parsePortStatusLine(line, service.portName);
            if (!parsed) {
              continue;
            }

            statuses.push({
              name: service.name,
              port: parsed.port,
              status: parsed.status,
              description: service.description,
            });
            break;
          }
        }

        if (statuses.length > 0) {
          return statuses;
        }

        const normalizedText = text.toUpperCase();
        const stoppedPatterns = [
          "所有服务均处于 STOPPED 状态",
          "所有服务均处于停止状态",
          "全部服务已停止",
          "当前所有服务未启动",
          "当前所有推理服务端口均处于停止状态",
          "所有推理服务端口均处于停止状态",
          "服务未运行",
          "所有服务未运行",
          "全部服务未运行",
          "所有服务已停止",
        ];
        const runningPatterns = [
          "所有服务均处于 RUNNING 状态",
          "所有服务均处于运行状态",
          "当前所有推理服务端口均处于运行状态",
          "所有推理服务端口均处于运行状态",
          "全部服务运行中",
          "所有服务运行中",
          "所有服务已启动",
          "全部服务已启动",
        ];

        const aggregateStatus = stoppedPatterns.some(
          (pattern) =>
            text.includes(pattern) || normalizedText.includes(pattern),
        )
          ? "stopped"
          : runningPatterns.some(
                (pattern) =>
                  text.includes(pattern) || normalizedText.includes(pattern),
              )
            ? "running"
            : null;

        if (aggregateStatus) {
          return serviceDefinitions
            .map((service) => {
              const port = portMap[service.portName];
              if (!port) {
                return null;
              }

              return {
                name: service.name,
                port,
                status: aggregateStatus,
                description: service.description,
                serviceKey: service.portName,
              } satisfies ServiceStatus;
            })
            .filter((service): service is ServiceStatus => service !== null);
        }
      }
    }
  }

  return emptyStatuses;
};

const parseNodeServiceStatusesFromReplies = (
  replies: Reply[],
): InferenceNodeStatuses | null => {
  for (let replyIndex = replies.length - 1; replyIndex >= 0; replyIndex -= 1) {
    const reply = replies[replyIndex];
    for (
      let messageIndex = reply.messages.length - 1;
      messageIndex >= 0;
      messageIndex -= 1
    ) {
      const message = reply.messages[messageIndex];
      const metadataStatuses = nodeStatusesFromProtocol(
        extractInferenceProtocolFromMetadata(message.metadata),
      );
      if (metadataStatuses) {
        return metadataStatuses;
      }

      const textStatuses = nodeStatusesFromProtocol(
        extractInferenceProtocolFromText(
          extractTextFromContent(message.content),
        ),
      );
      if (textStatuses) {
        return textStatuses;
      }
    }
  }

  return null;
};

const inferenceStatusLabelKey = (status: ServiceStatus["status"]): string => {
  const labels: Record<ServiceStatus["status"], string> = {
    running: "inference.statusLabels.running",
    stopped: "inference.statusLabels.stopped",
    starting: "inference.statusLabels.starting",
    failed: "inference.statusLabels.failed",
    degraded: "inference.statusLabels.degraded",
  };
  return labels[status];
};

const InferenceServicePanel = memo<Props>(function InferenceServicePanel({
  onClose,
  replies,
  onAskAI,
  isInputDisabled,
  view = "config",
  isAdmin = false,
}) {
  const { t } = useTranslation();
  const { messageApi } = useMessageApi();
  const [isRefreshingStatus, setIsRefreshingStatus] = useState(false);
  const [isRefreshingConfig, setIsRefreshingConfig] = useState(false);
  const [configRefreshReplyCount, setConfigRefreshReplyCount] = useState(0);
  const [statusRefreshReplyCount, setStatusRefreshReplyCount] = useState(0);
  const [selectedNode, setSelectedNode] = useState("main");
  const [selectedStatusNode, setSelectedStatusNode] = useState("main");
  const inferenceAdminStopServiceMutation =
    trpc.inferenceAdminStopServiceApply.useMutation();


  // 编辑状态
  const [editingConfig, setEditingConfig] = useState<Partial<InferenceConfig>>(
    {},
  );
  const [modifiedFields, setModifiedFields] = useState<Set<string>>(new Set());
  const [showSuccessAlert, setShowSuccessAlert] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [validationResult, setValidationResult] = useState<ValidationResult>({
    isValid: true,
    errors: [],
    warnings: [],
  });

  // 从消息历史中解析配置和状态
  const replyCount = useMemo(
    () => replies.reduce((count, reply) => count + reply.messages.length, 0),
    [replies],
  );
  const nodeConfigs = useMemo(
    () => parseInferenceNodeConfigsFromReplies(replies),
    [replies],
  );
  const nodeNames = useMemo(
    () => Object.keys(nodeConfigs || {}),
    [nodeConfigs],
  );
  const config =
    nodeConfigs?.[selectedNode] ||
    nodeConfigs?.main ||
    nodeConfigs?.[nodeNames[0]] ||
    null;
  const refreshedNodeConfigs = useMemo(() => {
    if (!isRefreshingConfig || replyCount <= configRefreshReplyCount) {
      return null;
    }
    return parseInferenceNodeConfigsFromReplies(
      sliceRepliesAfterMessageCount(replies, configRefreshReplyCount),
    );
  }, [configRefreshReplyCount, isRefreshingConfig, replies, replyCount]);

  useEffect(() => {
    if (nodeNames.length === 0 || nodeNames.includes(selectedNode)) {
      return;
    }
    setSelectedNode(nodeNames.includes("main") ? "main" : nodeNames[0]);
  }, [nodeNames, selectedNode]);
  const nodeStatuses = useMemo(
    () => parseNodeServiceStatusesFromReplies(replies),
    [replies],
  );
  const statusNodeNames = useMemo(
    () => Object.keys(nodeStatuses || {}),
    [nodeStatuses],
  );
  const serviceStatuses =
    nodeStatuses?.[selectedStatusNode] ||
    nodeStatuses?.main ||
    nodeStatuses?.[statusNodeNames[0]] ||
    [];
  const refreshedNodeStatuses = useMemo(() => {
    if (!isRefreshingStatus || replyCount <= statusRefreshReplyCount) {
      return null;
    }
    return parseNodeServiceStatusesFromReplies(
      sliceRepliesAfterMessageCount(replies, statusRefreshReplyCount),
    );
  }, [isRefreshingStatus, replies, replyCount, statusRefreshReplyCount]);

  useEffect(() => {
    if (
      statusNodeNames.length === 0 ||
      statusNodeNames.includes(selectedStatusNode)
    ) {
      return;
    }
    setSelectedStatusNode(
      statusNodeNames.includes("main") ? "main" : statusNodeNames[0],
    );
  }, [selectedStatusNode, statusNodeNames]);

  // 初始化编辑配置 - 只在用户未编辑时更新
  useEffect(() => {
    if (config && modifiedFields.size === 0) {
      setEditingConfig({
        ports: { ...config.ports },
        env: { ...config.env },
        runtime: { ...config.runtime },
      });
    }
  }, [config, modifiedFields.size]);

  // 智能检测配置刷新状态 - 只接受本次刷新/应用之后返回的新配置
  useEffect(() => {
    if (!isRefreshingConfig || !refreshedNodeConfigs) {
      return;
    }

    const refreshedConfig = selectConfigForNode(
      refreshedNodeConfigs,
      selectedNode,
    );
    if (!refreshedConfig) {
      return;
    }

    setEditingConfig({
      ports: { ...refreshedConfig.ports },
      env: { ...refreshedConfig.env },
      runtime: { ...refreshedConfig.runtime },
    });
    setModifiedFields(new Set());
    setSubmitError(null);
    setValidationResult({ isValid: true, errors: [], warnings: [] });
    setIsRefreshingConfig(false);
  }, [isRefreshingConfig, refreshedNodeConfigs, selectedNode]);

  // 智能检测服务状态刷新状态 - 只接受本次刷新之后返回的新状态
  useEffect(() => {
    if (isRefreshingStatus && refreshedNodeStatuses) {
      setIsRefreshingStatus(false);
    }
  }, [isRefreshingStatus, refreshedNodeStatuses]);

  // 处理配置项修改
  const handleConfigChange = useCallback(
    (section: keyof InferenceConfig, field: string, value: string | number) => {
      setEditingConfig((prev) => {
        // 确保 section 存在，如果不存在则从 config 初始化
        const currentSection = prev[section] || config?.[section] || {};

        const newConfig = {
          ...prev,
          [section]: {
            ...currentSection,
            [field]: value,
          },
        };

        // 实时验证配置
        const validation = validateConfig(newConfig);
        setValidationResult(validation);

        return newConfig;
      });
      setModifiedFields((prev) => new Set(prev).add(`${section}.${field}`));
      setShowSuccessAlert(false);
      setSubmitError(null); // 清除提交错误
    },
    [config],
  );

  // 应用配置修改
  const handleApplyConfig = useCallback(() => {
    if (isInputDisabled || !onAskAI || modifiedFields.size === 0) {
      return;
    }

    // 生成配置修改指令
    const changes: string[] = [];
    modifiedFields.forEach((field) => {
      const [section, key] = field.split(".");
      const value =
        editingConfig[section as keyof InferenceConfig]?.[
          key as keyof (typeof editingConfig)[keyof InferenceConfig]
        ];
      const originalValue =
        config?.[section as keyof InferenceConfig]?.[
          key as keyof (typeof config)[keyof InferenceConfig]
        ];
      if (value !== originalValue) {
        changes.push(`${key}: ${value}`);
      }
    });

    if (changes.length === 0) {
      messageApi.info(t("inference.noChanges") || "没有检测到修改");
      return;
    }

    // 使用验证器检查配置
    const validation = validateConfig(editingConfig);
    setValidationResult(validation);

    // 如果有错误，阻止提交
    if (!validation.isValid) {
      setSubmitError(
        t("inference.validationFailed") || "配置验证失败，请检查并修正错误",
      );
      return;
    }

    const nodeHint = nodeNames.length > 1 ? `（节点：${selectedNode}）` : "";
    const modifyText = `请帮我修改推理配置${nodeHint}：\n${changes.join("\n")}，并查看推理配置文件`;

    onAskAI([
      {
        type: BlockType.TEXT,
        text: modifyText,
      },
    ]);

    // 合并指令已包含查看配置，等待同一次请求返回最新配置，避免重复发送打断修改请求。
    setConfigRefreshReplyCount(replyCount);
    setIsRefreshingConfig(true);

    setShowSuccessAlert(true);
    setSubmitError(null);
    setValidationResult({ isValid: true, errors: [], warnings: [] });
  }, [
    onAskAI,
    isInputDisabled,
    editingConfig,
    config,
    modifiedFields,
    messageApi,
    replyCount,
    t,
    nodeNames.length,
    selectedNode,
  ]);

  // 重置配置
  const handleResetConfig = useCallback(() => {
    if (config) {
      setEditingConfig({
        ports: { ...config.ports },
        env: { ...config.env },
        runtime: { ...config.runtime },
      });
      setModifiedFields(new Set());
      setShowSuccessAlert(false);
      setSubmitError(null);
      setValidationResult({ isValid: true, errors: [], warnings: [] });
    }
  }, [config]);

  // 刷新服务状态（单独）
  const handleRefreshStatus = useCallback(() => {
    if (isInputDisabled || !onAskAI) {
      messageApi.warning(t("inference.waiting") || "请等待当前对话完成");
      return;
    }

    setStatusRefreshReplyCount(replyCount);
    setIsRefreshingStatus(true);
    onAskAI([
      {
        type: BlockType.TEXT,
        text: "查看推理服务状态",
      },
    ]);
  }, [onAskAI, isInputDisabled, messageApi, replyCount, t]);


  const handleAdminStopServiceInstance = useCallback(
    async (service: ServiceStatus) => {
      if (!service.instanceId) return;
      const confirmed = window.confirm(
        t("inference.forceStopConfirm", { instanceId: service.instanceId }),
      );
      if (!confirmed) return;
      try {
        const response = await inferenceAdminStopServiceMutation.mutateAsync({
          instanceId: service.instanceId,
        });
        if (!response.success) {
          throw new Error(response.message || t("inference.stopFailed"));
        }
        messageApi.success(response.message || t("inference.stopSubmitted"));
        handleRefreshStatus();
      } catch (error: any) {
        messageApi.error(error?.message || t("inference.stopInstanceFailed"));
      }
    },
    [handleRefreshStatus, inferenceAdminStopServiceMutation, messageApi, t],
  );
  // 刷新配置（单独）
  const handleRefreshConfig = useCallback(() => {
    if (isInputDisabled || !onAskAI) {
      messageApi.warning(t("inference.waiting") || "请等待当前对话完成");
      return;
    }

    setConfigRefreshReplyCount(replyCount);
    setIsRefreshingConfig(true);
    onAskAI([
      {
        type: BlockType.TEXT,
        text: "查看推理配置文件",
      },
    ]);
  }, [onAskAI, isInputDisabled, messageApi, replyCount, t]);

  const panelTitle =
    view === "status"
      ? t("inference.serviceStatus") || "服务状态"
      : t("inference.configInfo") || "配置文件";
  const panelIcon =
    view === "status" ? (
      <Activity className="h-5 w-5 text-primary" />
    ) : (
      <Server className="h-5 w-5 text-primary" />
    );
  const isPanelRefreshing =
    view === "status" ? isRefreshingStatus : isRefreshingConfig;
  const handlePanelRefresh =
    view === "status" ? handleRefreshStatus : handleRefreshConfig;
  const panelRefreshText =
    view === "status"
      ? t("inference.refreshStatus") || "刷新状态"
      : t("inference.refreshConfig") || "刷新配置";

  return (
    <div className="flex flex-col h-full min-h-0 bg-background text-sm">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b bg-card">
        <div className="flex items-center gap-3">
          {panelIcon}
          <h2 className="text-base font-semibold">{panelTitle}</h2>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={handlePanelRefresh}
            disabled={isPanelRefreshing || isInputDisabled}
            className="h-8 gap-1.5 text-xs"
          >
            <RefreshCw
              className={`h-4 w-4 ${isPanelRefreshing ? "animate-spin" : ""}`}
            />
            {isPanelRefreshing
              ? t("inference.loading") || "加载中..."
              : panelRefreshText}
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={onClose}
            className="h-8 w-8 p-0"
          >
            <PanelLeftClose className="h-5 w-5" />
          </Button>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
        <Tabs value={view} className="flex-1 flex flex-col min-h-0">
          {/* Config Tab */}
          <TabsContent
            value="config"
            className="flex-1 min-h-0 overflow-hidden mt-0"
          >
            {config ? (
              <div className="h-full min-h-0 overflow-y-auto p-4 pt-4">
                <div className="space-y-4 pr-4">
                  {nodeNames.length > 1 && (
                    <div className="flex items-center gap-2 rounded-lg border bg-muted/30 p-2">
                      <span className="px-1 text-xs text-muted-foreground">
                        {t("inference.inferenceNode") || "推理节点"}
                      </span>
                      <div className="flex flex-wrap gap-1.5">
                        {nodeNames.map((nodeName) => (
                          <Button
                            key={nodeName}
                            type="button"
                            size="sm"
                            variant={
                              selectedNode === nodeName ? "default" : "ghost"
                            }
                            className="h-7 px-3 font-mono text-xs"
                            onClick={() => {
                              setSelectedNode(nodeName);
                              setModifiedFields(new Set());
                              setShowSuccessAlert(false);
                              setSubmitError(null);
                              setValidationResult({
                                isValid: true,
                                errors: [],
                                warnings: [],
                              });
                            }}
                          >
                            {nodeName}
                          </Button>
                        ))}
                      </div>
                    </div>
                  )}
                  {/* Success Alert */}
                  {showSuccessAlert && (
                    <Alert className="bg-green-50 border-green-200">
                      <Check className="h-4 w-4 text-green-600" />
                      <AlertDescription className="text-xs text-green-800">
                        {t("inference.modifySent") ||
                          "修改请求已提交，正在等待配置返回"}
                      </AlertDescription>
                    </Alert>
                  )}

                  {/* Configuration Card - 外围 Card 包含所有配置 */}
                  <Card>
                    <CardHeader className="pb-3">
                      <CardTitle className="text-sm flex items-center gap-2">
                        <Settings className="h-4 w-4" />
                        {t("inference.configInfo") || "配置信息"}
                        {nodeNames.length > 1 && (
                          <Badge
                            variant="outline"
                            className="font-mono font-normal"
                          >
                            {selectedNode}
                          </Badge>
                        )}
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-6">
                      {/* Ports Configuration */}
                      <div>
                        <h4 className="text-xs font-medium flex items-center gap-2 mb-3 text-muted-foreground">
                          <Server className="h-4 w-4" />
                          {t("inference.ports") || "端口配置"}
                        </h4>
                        <div className="grid grid-cols-2 gap-3">
                          <div className="space-y-1.5">
                            <Label className="text-xs text-muted-foreground">
                              VLLM_OPENAI_PORT
                            </Label>
                            <Input
                              type="number"
                              value={
                                editingConfig.ports?.VLLM_OPENAI_PORT || ""
                              }
                              onChange={(e) =>
                                handleConfigChange(
                                  "ports",
                                  "VLLM_OPENAI_PORT",
                                  parseInt(e.target.value) || 0,
                                )
                              }
                              readOnly={!isAdmin}
                              className={`h-9 font-mono text-xs ${modifiedFields.has("ports.VLLM_OPENAI_PORT") ? "border-yellow-500 bg-yellow-50" : ""}`}
                            />
                          </div>
                          <div className="space-y-1.5">
                            <Label className="text-xs text-muted-foreground">
                              INFERENCE_PORT
                            </Label>
                            <Input
                              type="number"
                              value={editingConfig.ports?.INFERENCE_PORT || ""}
                              onChange={(e) =>
                                handleConfigChange(
                                  "ports",
                                  "INFERENCE_PORT",
                                  parseInt(e.target.value) || 0,
                                )
                              }
                              readOnly={!isAdmin}
                              className={`h-9 font-mono text-xs ${modifiedFields.has("ports.INFERENCE_PORT") ? "border-yellow-500 bg-yellow-50" : ""}`}
                            />
                          </div>
                          <div className="space-y-1.5">
                            <Label className="text-xs text-muted-foreground">
                              UI_PORT
                            </Label>
                            <Input
                              type="number"
                              value={editingConfig.ports?.UI_PORT || ""}
                              onChange={(e) =>
                                handleConfigChange(
                                  "ports",
                                  "UI_PORT",
                                  parseInt(e.target.value) || 0,
                                )
                              }
                              readOnly={!isAdmin}
                              className={`h-9 font-mono text-xs ${modifiedFields.has("ports.UI_PORT") ? "border-yellow-500 bg-yellow-50" : ""}`}
                            />
                          </div>
                          <div className="space-y-1.5">
                            <Label className="text-xs text-muted-foreground">
                              DATA_ANNOTATION_PORT
                            </Label>
                            <Input
                              type="number"
                              value={
                                editingConfig.ports?.DATA_ANNOTATION_PORT || ""
                              }
                              onChange={(e) =>
                                handleConfigChange(
                                  "ports",
                                  "DATA_ANNOTATION_PORT",
                                  parseInt(e.target.value) || 0,
                                )
                              }
                              readOnly={!isAdmin}
                              className={`h-9 font-mono text-xs ${modifiedFields.has("ports.DATA_ANNOTATION_PORT") ? "border-yellow-500 bg-yellow-50" : ""}`}
                            />
                          </div>
                        </div>
                      </div>

                      <div className="border-t border-border pt-6">
                        {/* Environment Configuration */}
                        <h4 className="text-xs font-medium flex items-center gap-2 mb-3 text-muted-foreground">
                          <Cpu className="h-4 w-4" />
                          {t("inference.environment") || "环境配置"}
                        </h4>
                        <div className="space-y-3">
                          <div className="space-y-1.5">
                            <Label className="text-xs text-muted-foreground">
                              HOST_IP
                            </Label>
                            <Input
                              value={editingConfig.env?.HOST_IP || ""}
                              onChange={(e) =>
                                handleConfigChange(
                                  "env",
                                  "HOST_IP",
                                  e.target.value,
                                )
                              }
                              readOnly={!isAdmin}
                              className={`h-9 font-mono text-xs ${modifiedFields.has("env.HOST_IP") ? "border-yellow-500 bg-yellow-50" : ""}`}
                            />
                          </div>
                          <div className="space-y-1.5">
                            <Label className="text-xs text-muted-foreground">
                              CUDA_VISIBLE_DEVICES
                            </Label>
                            <Input
                              value={
                                editingConfig.env?.CUDA_VISIBLE_DEVICES || ""
                              }
                              onChange={(e) =>
                                handleConfigChange(
                                  "env",
                                  "CUDA_VISIBLE_DEVICES",
                                  e.target.value,
                                )
                              }
                              readOnly={!isAdmin}
                              className={`h-9 font-mono text-xs ${modifiedFields.has("env.CUDA_VISIBLE_DEVICES") ? "border-yellow-500 bg-yellow-50" : ""}`}
                              placeholder="0,1,2,3"
                            />
                          </div>
                          <div className="space-y-1.5">
                            <Label className="text-xs text-muted-foreground">
                              MODEL_NAME
                            </Label>
                            <Input
                              value={editingConfig.env?.MODEL_NAME || ""}
                              onChange={(e) =>
                                handleConfigChange(
                                  "env",
                                  "MODEL_NAME",
                                  e.target.value,
                                )
                              }
                              readOnly={!isAdmin}
                              className={`h-9 font-mono text-xs ${modifiedFields.has("env.MODEL_NAME") ? "border-yellow-500 bg-yellow-50" : ""}`}
                            />
                          </div>
                        {config.env.MODEL_PARAM_B !== undefined && (
                          <div className="space-y-1.5">
                            <Label className="text-xs text-muted-foreground">MODEL_PARAM_B</Label>
                            <Input value={editingConfig.env?.MODEL_PARAM_B || ""} onChange={(e) => handleConfigChange("env", "MODEL_PARAM_B", e.target.value)} readOnly={!isAdmin} className="h-9 font-mono text-xs" />
                          </div>
                        )}
                          <div className="space-y-1.5">
                            <Label className="text-xs text-muted-foreground">
                              MODEL_PATH
                            </Label>
                            <Input
                              value={editingConfig.env?.MODEL_PATH || ""}
                              onChange={(e) =>
                                handleConfigChange(
                                  "env",
                                  "MODEL_PATH",
                                  e.target.value,
                                )
                              }
                              readOnly={!isAdmin}
                              className={`h-9 font-mono text-xs ${modifiedFields.has("env.MODEL_PATH") ? "border-yellow-500 bg-yellow-50" : ""}`}
                            />
                          </div>
                          {config.env.START_SCRIPT !== undefined && (
                            <div className="space-y-1.5">
                              <Label className="text-xs text-muted-foreground">
                                START_SCRIPT
                              </Label>
                              <Input
                                value={editingConfig.env?.START_SCRIPT || ""}
                                onChange={(e) =>
                                  handleConfigChange(
                                    "env",
                                    "START_SCRIPT",
                                    e.target.value,
                                  )
                                }
                                readOnly={!isAdmin}
                                className={`h-9 font-mono text-xs ${modifiedFields.has("env.START_SCRIPT") ? "border-yellow-500 bg-yellow-50" : ""}`}
                              />
                            </div>
                          )}
                          {config.env.LOG_DIR !== undefined && (
                            <div className="space-y-1.5">
                              <Label className="text-xs text-muted-foreground">
                                LOG_DIR
                              </Label>
                              <Input
                                value={editingConfig.env?.LOG_DIR || ""}
                                onChange={(e) =>
                                  handleConfigChange(
                                    "env",
                                    "LOG_DIR",
                                    e.target.value,
                                  )
                                }
                                readOnly={!isAdmin}
                                className={`h-9 font-mono text-xs ${modifiedFields.has("env.LOG_DIR") ? "border-yellow-500 bg-yellow-50" : ""}`}
                              />
                            </div>
                          )}
                          {config.env.TEST_DIR !== undefined && (
                            <div className="space-y-1.5">
                              <Label className="text-xs text-muted-foreground">
                                TEST_DIR
                              </Label>
                              <Input
                                value={editingConfig.env?.TEST_DIR || ""}
                                onChange={(e) =>
                                  handleConfigChange(
                                    "env",
                                    "TEST_DIR",
                                    e.target.value,
                                  )
                                }
                                readOnly={!isAdmin}
                                className={`h-9 font-mono text-xs ${modifiedFields.has("env.TEST_DIR") ? "border-yellow-500 bg-yellow-50" : ""}`}
                              />
                            </div>
                          )}
                          {config.env.BENCHMARK_DIR !== undefined && (
                            <div className="space-y-1.5">
                              <Label className="text-xs text-muted-foreground">
                                BENCHMARK_DIR
                              </Label>
                              <Input
                                value={editingConfig.env?.BENCHMARK_DIR || ""}
                                onChange={(e) =>
                                  handleConfigChange(
                                    "env",
                                    "BENCHMARK_DIR",
                                    e.target.value,
                                  )
                                }
                                readOnly={!isAdmin}
                                className={`h-9 font-mono text-xs ${modifiedFields.has("env.BENCHMARK_DIR") ? "border-yellow-500 bg-yellow-50" : ""}`}
                              />
                            </div>
                          )}
                          {config.env.GENERAL_BENCHMARK_DIR !== undefined && (
                            <div className="space-y-1.5">
                              <Label className="text-xs text-muted-foreground">
                                GENERAL_BENCHMARK_DIR
                              </Label>
                              <Input
                                value={
                                  editingConfig.env?.GENERAL_BENCHMARK_DIR || ""
                                }
                                onChange={(e) =>
                                  handleConfigChange(
                                    "env",
                                    "GENERAL_BENCHMARK_DIR",
                                    e.target.value,
                                  )
                                }
                                readOnly={!isAdmin}
                                className={`h-9 font-mono text-xs ${modifiedFields.has("env.GENERAL_BENCHMARK_DIR") ? "border-yellow-500 bg-yellow-50" : ""}`}
                              />
                            </div>
                          )}
                        {config.env.MASTER_PORT !== undefined && (
                          <div className="space-y-1.5">
                            <Label className="text-xs text-muted-foreground">MASTER_PORT</Label>
                            <Input type="number" value={editingConfig.env?.MASTER_PORT || ""} onChange={(e) => handleConfigChange("env", "MASTER_PORT", parseInt(e.target.value) || 0)} readOnly={!isAdmin} className="h-9 font-mono text-xs" />
                          </div>
                        )}
                        </div>
                      </div>

                      <div className="border-t border-border pt-6">
                        {/* Runtime Configuration */}
                        <h4 className="text-xs font-medium flex items-center gap-2 mb-3 text-muted-foreground">
                          <Gauge className="h-4 w-4" />
                          {t("inference.runtime") || "运行时配置"}
                        </h4>
                        <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
                          <div className="space-y-1.5">
                            <Label className="text-xs text-muted-foreground">
                              Tensor Parallel
                            </Label>
                            <Input
                              type="number"
                              value={
                                editingConfig.runtime?.TENSOR_PARALLEL_SIZE ||
                                ""
                              }
                              onChange={(e) =>
                                handleConfigChange(
                                  "runtime",
                                  "TENSOR_PARALLEL_SIZE",
                                  parseInt(e.target.value) || 0,
                                )
                              }
                              readOnly={!isAdmin}
                              className={`h-9 font-mono text-xs text-center ${modifiedFields.has("runtime.TENSOR_PARALLEL_SIZE") ? "border-yellow-500 bg-yellow-50" : ""}`}
                            />
                          </div>
                          <div className="space-y-1.5">
                            <Label className="text-xs text-muted-foreground">
                              GPU Memory
                            </Label>
                            <Input
                              type="number"
                              step="0.1"
                              min="0"
                              max="1"
                              value={
                                editingConfig.runtime?.GPU_MEMORY_UTILIZATION ||
                                ""
                              }
                              onChange={(e) =>
                                handleConfigChange(
                                  "runtime",
                                  "GPU_MEMORY_UTILIZATION",
                                  parseFloat(e.target.value) || 0,
                                )
                              }
                              readOnly={!isAdmin}
                              className={`h-9 font-mono text-xs text-center ${modifiedFields.has("runtime.GPU_MEMORY_UTILIZATION") ? "border-yellow-500 bg-yellow-50" : ""}`}
                              placeholder="0.9"
                            />
                          </div>
                            {config.runtime.GPU_UTILIZATION_THRESHOLD !== undefined && (
                              <div className="space-y-1.5">
                                <Label className="text-xs text-muted-foreground">GPU Threshold</Label>
                                <Input type="number" step="1" value={editingConfig.runtime?.GPU_UTILIZATION_THRESHOLD || ""} onChange={(e) => handleConfigChange("runtime", "GPU_UTILIZATION_THRESHOLD", parseFloat(e.target.value) || 0)} readOnly={!isAdmin} className="h-9 font-mono text-xs text-center" />
                              </div>
                            )}
                          <div className="space-y-1.5">
                            <Label className="text-xs text-muted-foreground">
                              Max Tokens
                            </Label>
                            <Input
                              type="number"
                              value={editingConfig.runtime?.MAX_TOKENS || ""}
                              onChange={(e) =>
                                handleConfigChange(
                                  "runtime",
                                  "MAX_TOKENS",
                                  parseInt(e.target.value) || 0,
                                )
                              }
                              readOnly={!isAdmin}
                              className={`h-9 font-mono text-xs text-center ${modifiedFields.has("runtime.MAX_TOKENS") ? "border-yellow-500 bg-yellow-50" : ""}`}
                            />
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  {/* Validation Summary */}
                  {(validationResult.errors.length > 0 ||
                    validationResult.warnings.length > 0) && (
                    <Card
                      className={
                        validationResult.errors.length > 0
                          ? "border-red-200 bg-red-50/30"
                          : "border-yellow-200 bg-yellow-50/30"
                      }
                    >
                      <CardContent className="pt-4 pb-4">
                        {/* 错误列表 */}
                        {validationResult.errors.length > 0 && (
                          <div className="mb-3 space-y-2">
                            <div className="flex items-center gap-2 text-red-800 font-medium text-xs">
                              <AlertCircle className="h-4 w-4" />
                              <span>
                                {t("inference.validationErrors") || "配置错误"}{" "}
                                ({validationResult.errors.length})
                              </span>
                            </div>
                            <div className="pl-6 space-y-1">
                              {validationResult.errors.map((error, index) => (
                                <p key={index} className="text-xs text-red-700">
                                  • {error.message}
                                </p>
                              ))}
                            </div>
                          </div>
                        )}
                        {/* 警告列表 */}
                        {validationResult.warnings.length > 0 && (
                          <div
                            className={
                              validationResult.errors.length > 0
                                ? "mt-3 pt-3 border-t border-red-200/50 space-y-2"
                                : "space-y-2"
                            }
                          >
                            <div className="flex items-center gap-2 text-yellow-800 font-medium text-xs">
                              <AlertTriangle className="h-4 w-4" />
                              <span>
                                {t("inference.validationWarnings") ||
                                  "配置警告"}{" "}
                                ({validationResult.warnings.length})
                              </span>
                            </div>
                            <div className="pl-6 space-y-1">
                              {validationResult.warnings.map(
                                (warning, index) => (
                                  <p
                                    key={index}
                                    className="text-xs text-yellow-700"
                                  >
                                    • {warning.message}
                                  </p>
                                ),
                              )}
                            </div>
                          </div>
                        )}
                      </CardContent>
                    </Card>
                  )}

                  {/* Action Buttons */}
                  {isAdmin && modifiedFields.size > 0 && (
                    <Card className="border-yellow-200 bg-yellow-50/50">
                      <CardContent className="pt-4 pb-4">
                        {/* 错误提示 */}
                        {submitError && (
                          <div className="mb-3 p-3 bg-red-50 border border-red-200 rounded-lg">
                            <p className="text-xs text-red-800 flex items-center gap-2">
                              <AlertCircle className="h-4 w-4" />
                              {submitError}
                            </p>
                          </div>
                        )}
                        <div className="flex items-center justify-between">
                          <div className="text-xs text-yellow-800">
                            <span className="font-medium">
                              {modifiedFields.size}
                            </span>{" "}
                            {t("inference.fieldsModified") || "个字段已修改"}
                          </div>
                          <div className="flex gap-2">
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={handleResetConfig}
                              disabled={isInputDisabled || isRefreshingConfig}
                              className="gap-1.5 text-xs"
                            >
                              <RotateCcw className="h-4 w-4" />
                              {t("inference.reset") || "重置"}
                            </Button>
                            <Button
                              size="sm"
                              onClick={handleApplyConfig}
                              disabled={
                                isInputDisabled ||
                                isRefreshingConfig ||
                                !validationResult.isValid
                              }
                              className="gap-1.5 text-xs bg-yellow-600 hover:bg-yellow-700"
                              title={
                                !validationResult.isValid
                                  ? t("inference.fixErrorsFirst") ||
                                    "请先修正配置错误"
                                  : undefined
                              }
                            >
                              <Check className="h-4 w-4" />
                              {t("inference.applyChanges") || "应用修改"}
                            </Button>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  )}
                </div>
              </div>
            ) : (
              <div className="h-full min-h-0 overflow-y-auto p-4 pt-4">
                <div className="pr-4">
                  <Card>
                    <CardHeader className="pb-3">
                      <CardTitle className="text-sm flex items-center gap-2">
                        <Settings className="h-4 w-4" />
                        {t("inference.configInfo") || "配置信息"}
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      <div className="text-center py-12 text-muted-foreground">
                        <Settings className="h-12 w-12 mx-auto mb-3 opacity-50" />
                        <p>{t("inference.noConfigData") || "暂无配置数据"}</p>
                        <p className="text-xs mt-1">
                          {t("inference.clickRefreshConfig") ||
                            "点击刷新按钮获取配置信息"}
                        </p>
                      </div>
                    </CardContent>
                  </Card>
                </div>
              </div>
            )}
          </TabsContent>

          {/* Status Tab */}
          <TabsContent
            value="status"
            className="flex-1 min-h-0 overflow-hidden mt-0"
          >
            <div className="h-full min-h-0 overflow-y-auto p-4 pt-4">
              <div className="space-y-4 pr-4">
                {statusNodeNames.length > 1 && (
                  <div className="flex items-center gap-2 rounded-lg border bg-muted/30 p-2">
                    <span className="px-1 text-xs text-muted-foreground">
                      {t("inference.inferenceNode") || "推理节点"}
                    </span>
                    <div className="flex flex-wrap gap-1.5">
                      {statusNodeNames.map((nodeName) => (
                        <Button
                          key={nodeName}
                          type="button"
                          size="sm"
                          variant={
                            selectedStatusNode === nodeName
                              ? "default"
                              : "ghost"
                          }
                          className="h-7 px-3 font-mono text-xs"
                          onClick={() => setSelectedStatusNode(nodeName)}
                        >
                          {nodeName}
                        </Button>
                      ))}
                    </div>
                  </div>
                )}
                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm flex items-center gap-2">
                      <Activity className="h-4 w-4" />
                      {t("inference.serviceStatus") || "服务状态"}
                      {statusNodeNames.length > 1 && (
                        <Badge
                          variant="outline"
                          className="font-mono font-normal"
                        >
                          {selectedStatusNode}
                        </Badge>
                      )}
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    {serviceStatuses.length > 0 ? (
                      serviceStatuses.map((service, index) => (
                        <div
                          key={`${service.instanceId || "service"}:${service.serviceKey || service.name}:${service.port || index}`}
                          className="flex flex-col gap-3 rounded-lg border bg-muted/50 p-3 sm:flex-row sm:items-center sm:justify-between"
                        >
                          <div className="flex min-w-0 items-start gap-3">
                            <div
                              className={`mt-1.5 h-2 w-2 shrink-0 rounded-full ${
                                service.status === "running"
                                  ? "bg-green-500"
                                  : service.status === "starting"
                                    ? "bg-yellow-500"
                                    : service.status === "failed"
                                      ? "bg-red-500"
                                      : "bg-muted-foreground"
                              }`}
                            />
                            <div className="min-w-0 space-y-1">
                              <div className="truncate text-sm font-medium">
                                {t(`inference.services.${service.serviceKey || service.name}.name`, { defaultValue: service.name })}
                              </div>
                              <div className="text-xs text-muted-foreground">
                                {service.port
                                  ? t("inference.servicePortDescription", {
                                      description: t(`inference.services.${service.serviceKey || service.name}.description`, {
                                        defaultValue: service.description || "",
                                      }),
                                      port: service.port,
                                    })
                                  : t(`inference.services.${service.serviceKey || service.name}.description`, {
                                      defaultValue: service.description || "",
                                    })}
                              </div>
                            </div>
                          </div>
                          <div className="flex shrink-0 items-center gap-2 self-end sm:self-auto">
                            <Badge
                              variant={
                                service.status === "running"
                                  ? "default"
                                  : "secondary"
                              }
                            >
                              {t(inferenceStatusLabelKey(service.status))}
                            </Badge>
                            {isAdmin && service.instanceId && (
                              <Button
                                type="button"
                                variant="outline"
                                size="sm"
                                className="h-7 px-2 text-xs"
                                disabled={
                                  service.status === "stopped" ||
                                  inferenceAdminStopServiceMutation.isPending
                                }
                                onClick={() => handleAdminStopServiceInstance(service)}
                              >
                                {t("inference.forceStop")}
                              </Button>
                            )}
                          </div>
                        </div>
                      ))
                    ) : (
                      <div className="text-center py-12 text-muted-foreground">
                        <Activity className="h-12 w-12 mx-auto mb-3 opacity-50" />
                        <p>
                          {t("inference.noStatusData") || "暂无服务状态数据"}
                        </p>
                        <p className="text-xs mt-1">
                          {t("inference.clickRefresh") ||
                            "点击刷新按钮获取最新状态"}
                        </p>
                      </div>
                    )}
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
});

export default InferenceServicePanel;
