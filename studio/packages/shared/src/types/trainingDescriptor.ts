export type TrainingBackend = "llamafactory" | "verl";
export type TrainingStage = "sft" | "pt" | "dpo" | "grpo";
export type TrainingFinetuningType = "lora" | "full" | "rl";
export type TrainingInvocationStyle = "env" | "argv" | "keyValue";

export type TrainingParamType =
  | "text"
  | "number"
  | "boolean"
  | "enum"
  | "model"
  | "dataset"
  | "file"
  | "template"
  | "checkpoint"
  | "argvRemainder";

export interface TrainingParamDescriptor {
  name: string;
  label?: string;
  type: TrainingParamType;
  required: boolean;
  aliases?: string[];
  default?: string | number | boolean | string[];
  options?: string[];
  multiple?: boolean;
  advanced?: boolean;
  resourceOnly?: boolean;
}

export interface TrainingEntryDescriptor {
  sourceScript: string;
  containerCommand: string;
  invocationStyle: TrainingInvocationStyle;
}

export interface TrainingPathDescriptor {
  defaultDatasetRoot?: string;
  outputRoot?: string;
  logRoot?: string;
  exportRoot?: string;
}

export interface TrainingMonitoringDescriptor {
  primaryMetric: string;
  fallbackMetrics?: string[];
  logFiles?: string[];
  wandbDir?: string;
  completionStates?: string[];
}

export interface TrainingArtifactPolicy {
  produces: Array<"adapter" | "checkpoint" | "merged">;
  deployable: "merged" | "checkpoint" | "none";
}

export interface TrainingResourceRequirements {
  gpuCount?: number;
  nodeCount?: number;
  gpusPerNode?: number;
  containerRole: "training" | "multinodeTraining" | "grpoTraining";
  resourcePool?: boolean;
}

export interface TrainingDescriptor {
  id: string;
  displayName: string;
  aliases: string[];
  backend: TrainingBackend;
  stage: TrainingStage;
  finetuningType: TrainingFinetuningType;
  entry: TrainingEntryDescriptor;
  params: TrainingParamDescriptor[];
  requiredParams: string[];
  resourceSelectors: string[];
  resourceRequirements: TrainingResourceRequirements;
  paths: TrainingPathDescriptor;
  monitoring: TrainingMonitoringDescriptor;
  artifactPolicy: TrainingArtifactPolicy;
}

export interface QueryTrainingMethodsData {
  methods: TrainingDescriptor[];
}
