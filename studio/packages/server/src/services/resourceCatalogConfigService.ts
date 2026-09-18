import type {
  DatasetCatalogConfigItem,
  EvaluationCatalogConfigItem,
  ModelCatalogConfigItem,
  QueryResourceCatalogConfigData,
} from "../../../shared/src";
import { ConfigManager } from "../../../shared/src/config/server";

const DATASETS: DatasetCatalogConfigItem[] = [
  {
    id: "raw",
    path: "/home/workspace/dataset",
    aliases: ["raw"],
    dataTypes: ["raw", "custom"],
    description: "原始上传数据目录",
  },
  {
    id: "sft",
    path: "/home/workspace/dataset_batch_train",
    aliases: ["batch_train", "sft"],
    dataTypes: ["sft"],
    description: "通用 SFT 数据目录",
  },
  {
    id: "dpo",
    path: "/home/workspace/dataset_daily_train",
    aliases: ["daily_train", "dpo"],
    dataTypes: ["dpo"],
    description: "DPO 偏好数据目录",
  },
  {
    id: "pt",
    path: "/home/workspace/dataset_pretrain",
    aliases: ["text", "pretrain", "pt"],
    dataTypes: ["pt"],
    description: "继续预训练文本数据目录，text 作为 PT alias 保留",
  },
  {
    id: "rl",
    path: "/home/workspace/verl/examples/data_preprocess/data",
    aliases: ["grpo", "verl"],
    dataTypes: ["rl"],
    description: "GRPO/RL 训练数据目录",
  },
  {
    id: "eval",
    path: "/home/workspace/eval",
    aliases: ["evaluation", "medicalTest"],
    dataTypes: ["eval"],
    description: "评估资源目录",
  },
];

const MODELS: ModelCatalogConfigItem[] = [
  {
    id: "base",
    path: "/home/workspace/models/base",
    modelTypes: ["base"],
    legacyTypes: ["base_train"],
    aliases: ["base_model"],
    description: "基础模型目录",
  },
  {
    id: "batchTrain",
    path: "/home/workspace/models/batch_train",
    modelTypes: ["adapter", "merged", "checkpoint"],
    legacyTypes: ["batch_trained"],
    aliases: ["sftOutput"],
    description: "SFT 训练输出目录",
  },
  {
    id: "pretrain",
    path: "/home/workspace/models/pretrain",
    modelTypes: ["adapter", "merged", "checkpoint"],
    legacyTypes: ["pretrain"],
    aliases: ["ptOutput"],
    description: "PT 训练输出目录",
  },
  {
    id: "dpoSaves",
    path: "/home/workspace/models/dpo_train/internal/saves",
    modelTypes: ["adapter", "checkpoint"],
    legacyTypes: ["daily_trained"],
    aliases: ["dpoOutput"],
    description: "DPO adapter/checkpoint 输出目录",
  },
  {
    id: "dpoExport",
    path: "/home/workspace/models/dpo_train/internal/export",
    modelTypes: ["merged"],
    legacyTypes: ["daily_trained"],
    aliases: ["dpoMerged"],
    description: "DPO 合并导出模型目录",
  },
  {
    id: "inference",
    path: "/home/workspace/medical_models",
    modelTypes: ["inference", "merged"],
    legacyTypes: ["inference"],
    description: "推理服务模型目录",
  },
  {
    id: "grpo",
    path: "/home/workspace/models/grpo_train",
    modelTypes: ["checkpoint", "reward"],
    legacyTypes: ["grpo"],
    aliases: ["rlOutput"],
    description: "GRPO 训练产物目录",
  },
];

const EVALUATION: EvaluationCatalogConfigItem[] = [
  {
    id: "evaluationDataset",
    legacyTypes: ["medicalTest"],
    categories: ["medical", "general", "custom"],
    path: "/home/workspace/tests",
  },
];

const managedPath = (value: unknown, label: string): string => {
  const path = String(value || "").trim().replace(/\\/g, "/").replace(/\/+$/, "");
  if (!path.startsWith("/") || path === "" || path === "/" || path.split("/").includes("..")) {
    throw new Error(`${label} must be an absolute managed directory`);
  }
  return path;
};

const mergeById = <T extends { id: string; path?: string }>(
  defaults: T[],
  supplied: unknown,
  section: string,
): T[] => {
  if (supplied === undefined) return defaults.map((item) => ({ ...item }));
  if (!Array.isArray(supplied)) throw new Error(`resource catalog ${section} must be a list`);
  const entries = new Map(defaults.map((item) => [item.id, { ...item }]));
  for (const rawItem of supplied) {
    if (!rawItem || typeof rawItem !== "object") {
      throw new Error(`resource catalog ${section} entry requires id`);
    }
    const item = rawItem as Record<string, unknown>;
    const id = String(item.id || "").trim();
    if (!id) throw new Error(`resource catalog ${section} entry requires id`);
    const merged = { ...(entries.get(id) || {}), ...item, id } as T;
    merged.path = managedPath(merged.path, `${section}.${id}.path`);
    entries.set(id, merged);
  }
  return [...entries.values()];
};

const configuredCatalog = (): {
  datasets: DatasetCatalogConfigItem[];
  models: ModelCatalogConfigItem[];
  evaluation: EvaluationCatalogConfigItem[];
} => {
  const raw = process.env.MEDFLOW_RESOURCE_CATALOG_CONFIG?.trim();
  if (!raw) return {
    datasets: DATASETS.map((item) => ({ ...item })),
    models: MODELS.map((item) => ({ ...item })),
    evaluation: EVALUATION.map((item) => ({ ...item })),
  };
  let parsed: Record<string, unknown>;
  try {
    const value = JSON.parse(raw) as unknown;
    if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("not an object");
    parsed = value as Record<string, unknown>;
  } catch {
    throw new Error("MEDFLOW_RESOURCE_CATALOG_CONFIG must be a valid JSON object");
  }
  return {
    datasets: mergeById(DATASETS, parsed.datasets, "datasets").map((item) => ({
      ...item,
      aliases: Array.isArray(item.aliases) ? item.aliases : [],
      dataTypes: Array.isArray(item.dataTypes) && item.dataTypes.length > 0
        ? item.dataTypes
        : ["custom"],
    })),
    models: mergeById(MODELS, parsed.models, "models").map((item) => ({
      ...item,
      modelTypes: Array.isArray(item.modelTypes) && item.modelTypes.length > 0
        ? item.modelTypes
        : ["custom"],
      legacyTypes: Array.isArray(item.legacyTypes) && item.legacyTypes.length > 0
        ? item.legacyTypes
        : [item.id],
    })),
    evaluation: mergeById(EVALUATION, parsed.evaluation, "evaluation"),
  };
};

export class ResourceCatalogConfigService {
  static getConfig(): QueryResourceCatalogConfigData {
    const config = ConfigManager.getInstance();
    const catalog = configuredCatalog();
    return {
      datasets: catalog.datasets.map((item) => ({
        ...item,
        aliases: [...item.aliases],
        dataTypes: [...item.dataTypes],
      })),
      models: catalog.models.map((item) => ({
        ...item,
        modelTypes: [...item.modelTypes],
        legacyTypes: [...item.legacyTypes],
        aliases: item.aliases ? [...item.aliases] : undefined,
      })),
      evaluation: catalog.evaluation.map((item) => ({
        ...item,
        legacyTypes: [...item.legacyTypes] as ["medicalTest"],
        categories: [...item.categories],
      })),
      containers: {
        training: config.getDefaultContainerName(),
        multinodeTraining: config.getDefaultMultinodeContainerName(),
        grpoTraining: config.getDefaultGrpoContainerName(),
        evaluation: config.getDefaultEvaluateContainerName(),
      },
      supportedDatasetTypes: ["raw", "sft", "dpo", "pt", "rl", "eval", "custom"],
      supportedModelTypes: [
        "base",
        "adapter",
        "merged",
        "checkpoint",
        "inference",
        "reward",
        "custom",
      ],
    };
  }
}
