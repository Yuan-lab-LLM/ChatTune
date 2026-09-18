export type DatasetDataType =
  | "raw"
  | "sft"
  | "dpo"
  | "pt"
  | "rl"
  | "eval"
  | "custom";

export type DatasetCatalogId = DatasetDataType | (string & {});

export type ModelCatalogType =
  | "base"
  | "adapter"
  | "merged"
  | "checkpoint"
  | "inference"
  | "reward"
  | "custom";

export interface DatasetCatalogConfigItem {
  id: DatasetCatalogId;
  path: string;
  aliases: string[];
  description?: string;
  dataTypes: DatasetDataType[];
}

export interface ModelCatalogConfigItem {
  id: string;
  path: string;
  modelTypes: ModelCatalogType[];
  legacyTypes: string[];
  aliases?: string[];
  description?: string;
}

export interface EvaluationCatalogConfigItem {
  id: "evaluationDataset";
  legacyTypes: ["medicalTest"];
  categories: Array<"medical" | "general" | "custom">;
  path?: string;
}

export interface ResourceCatalogContainerConfig {
  training: string;
  multinodeTraining: string;
  grpoTraining: string;
  evaluation: string;
}

export interface QueryResourceCatalogConfigData {
  datasets: DatasetCatalogConfigItem[];
  models: ModelCatalogConfigItem[];
  evaluation: EvaluationCatalogConfigItem[];
  containers: ResourceCatalogContainerConfig;
  supportedDatasetTypes: DatasetDataType[];
  supportedModelTypes: ModelCatalogType[];
}
