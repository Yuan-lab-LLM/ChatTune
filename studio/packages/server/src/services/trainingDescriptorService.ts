import { readFileSync } from "fs";
import { basename, dirname, resolve } from "path";
import { z } from "zod";
import type { TrainingDescriptor } from "../../../shared/src";
import { PATHS } from "../../../shared/src/config/server";

const MANIFEST_SCHEMA_VERSION = 1;
const MANIFEST_ENV = "MEDFLOW_TRAINING_DESCRIPTOR_PATH";

const paramSchema = z.object({
  name: z.string().trim().min(1),
  label: z.string().trim().min(1).optional(),
  type: z.enum([
    "text", "number", "boolean", "enum", "model", "dataset", "file",
    "template", "checkpoint", "argvRemainder",
  ]),
  required: z.boolean(),
  aliases: z.array(z.string().trim().min(1)).optional(),
  default: z.union([
    z.string(), z.number(), z.boolean(), z.array(z.string()),
  ]).optional(),
  options: z.array(z.string().trim().min(1)).min(1).optional(),
  multiple: z.boolean().optional(),
  advanced: z.boolean().optional(),
  resourceOnly: z.boolean().optional(),
}).strict().superRefine((param, context) => {
  if (param.options && param.type !== "enum") {
    context.addIssue({
      code: z.ZodIssueCode.custom,
      message: `param ${param.name} options require enum type`,
    });
  }
  if (param.options && param.default !== undefined && !param.options.includes(String(param.default))) {
    context.addIssue({
      code: z.ZodIssueCode.custom,
      message: `param ${param.name} default must be one of options`,
    });
  }
});

const rawDescriptorSchema = z.object({
  id: z.string().trim().min(1),
  displayName: z.string().trim().min(1),
  aliases: z.array(z.string().trim().min(1)),
  backend: z.enum(["llamafactory", "verl"]),
  stage: z.enum(["sft", "pt", "dpo", "grpo"]),
  finetuningType: z.enum(["lora", "full", "rl"]),
  entry: z.object({
    sourceScript: z.string().trim().min(1),
    containerCommand: z.string().trim().min(1),
    invocationStyle: z.enum(["env", "argv", "keyValue"]),
  }).strict(),
  params: z.array(paramSchema),
  resourceRequirements: z.object({
    gpuCount: z.number().int().positive().optional(),
    nodeCount: z.number().int().positive().optional(),
    gpusPerNode: z.number().int().positive().optional(),
    containerRole: z.enum(["training", "multinodeTraining", "grpoTraining"]),
    resourcePool: z.boolean().optional(),
  }).strict(),
  paths: z.object({
    defaultDatasetRoot: z.string().trim().min(1).optional(),
    outputRoot: z.string().trim().min(1).optional(),
    logRoot: z.string().trim().min(1).optional(),
    exportRoot: z.string().trim().min(1).optional(),
  }).strict(),
  monitoring: z.object({
    primaryMetric: z.string().trim().min(1),
    fallbackMetrics: z.array(z.string().trim().min(1)).optional(),
    logFiles: z.array(z.string().trim().min(1)).optional(),
    wandbDir: z.string().trim().min(1).optional(),
    completionStates: z.array(z.string().trim().min(1)).optional(),
  }).strict(),
  artifactPolicy: z.object({
    produces: z.array(z.enum(["adapter", "checkpoint", "merged"])).min(1),
    deployable: z.enum(["merged", "checkpoint", "none"]),
  }).strict(),
}).strict();

const manifestSchema = z.object({
  schemaVersion: z.literal(MANIFEST_SCHEMA_VERSION),
  descriptors: z.array(rawDescriptorSchema).min(1),
}).strict();

const normalizeIdentity = (value: string): string =>
  value.trim().toLowerCase().replace(/\s+/g, "");

const descriptorIdentityValues = (
  descriptor: z.infer<typeof rawDescriptorSchema>,
): string[] => [
  descriptor.id,
  descriptor.displayName,
  descriptor.entry.sourceScript,
  basename(descriptor.entry.sourceScript).replace(/\.py$/i, ""),
  descriptor.entry.containerCommand,
  descriptor.entry.containerCommand.split("/").filter(Boolean).pop() || "",
  ...descriptor.aliases,
];

const trainingDescriptorManifestPath = (): string => {
  const configured = process.env[MANIFEST_ENV]?.trim();
  if (configured) return resolve(configured);
  return resolve(dirname(PATHS.getAgentConfigPath()), "training_descriptors.v1.json");
};

const loadDescriptors = (): TrainingDescriptor[] => {
  const manifestPath = trainingDescriptorManifestPath();
  let encoded: string;
  try {
    encoded = readFileSync(manifestPath, "utf8");
  } catch (error) {
    const detail = error instanceof Error ? `: ${error.message}` : "";
    throw new Error(`Training descriptor manifest not found or unreadable: ${manifestPath}${detail}`);
  }

  let manifest: z.infer<typeof manifestSchema>;
  try {
    manifest = manifestSchema.parse(JSON.parse(encoded));
  } catch (error) {
    const detail = error instanceof Error ? `: ${error.message}` : "";
    throw new Error(`Invalid training descriptor manifest: ${manifestPath}${detail}`);
  }

  const ids = new Set<string>();
  const identityOwners = new Map<string, string>();
  for (const descriptor of manifest.descriptors) {
    if (ids.has(descriptor.id)) {
      throw new Error(`Duplicate training descriptor id: ${descriptor.id}`);
    }
    ids.add(descriptor.id);

    const paramNames = new Set<string>();
    for (const param of descriptor.params) {
      if (paramNames.has(param.name)) {
        throw new Error(`Training descriptor ${descriptor.id} has duplicate param name: ${param.name}`);
      }
      paramNames.add(param.name);
    }

    for (const identity of descriptorIdentityValues(descriptor)) {
      const normalized = normalizeIdentity(identity);
      const owner = identityOwners.get(normalized);
      if (owner && owner !== descriptor.id) {
        throw new Error(
          `Training descriptor alias conflict: ${identity} belongs to ${owner} and ${descriptor.id}`,
        );
      }
      identityOwners.set(normalized, descriptor.id);
    }
  }

  return manifest.descriptors.map((descriptor) => ({
    ...descriptor,
    requiredParams: descriptor.params
      .filter((param) => param.required)
      .map((param) => param.name),
    resourceSelectors: [
      descriptor.resourceRequirements.containerRole,
      ...(descriptor.resourceRequirements.resourcePool ? ["trainingPool"] : []),
    ],
  }));
};

const trainingDescriptors = loadDescriptors();

export class TrainingDescriptorService {
  static listDescriptors(): TrainingDescriptor[] {
    return trainingDescriptors.map((descriptor) => ({
      ...descriptor,
      aliases: [...descriptor.aliases],
      params: descriptor.params.map((param) => ({
        ...param,
        aliases: param.aliases ? [...param.aliases] : undefined,
        options: param.options ? [...param.options] : undefined,
      })),
      requiredParams: [...descriptor.requiredParams],
      resourceSelectors: [...descriptor.resourceSelectors],
      entry: { ...descriptor.entry },
      resourceRequirements: { ...descriptor.resourceRequirements },
      paths: { ...descriptor.paths },
      monitoring: {
        ...descriptor.monitoring,
        fallbackMetrics: descriptor.monitoring.fallbackMetrics
          ? [...descriptor.monitoring.fallbackMetrics]
          : undefined,
        logFiles: descriptor.monitoring.logFiles
          ? [...descriptor.monitoring.logFiles]
          : undefined,
        completionStates: descriptor.monitoring.completionStates
          ? [...descriptor.monitoring.completionStates]
          : undefined,
      },
      artifactPolicy: {
        ...descriptor.artifactPolicy,
        produces: [...descriptor.artifactPolicy.produces],
      },
    }));
  }

  static findByAlias(value: string): TrainingDescriptor | null {
    const normalized = normalizeIdentity(value);
    if (!normalized) return null;
    return (
      trainingDescriptors.find((descriptor) =>
        descriptorIdentityValues(descriptor).some(
          (alias) => normalizeIdentity(alias) === normalized,
        ),
      ) || null
    );
  }
}
