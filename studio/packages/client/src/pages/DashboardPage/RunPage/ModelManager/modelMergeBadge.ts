import type { ModelInfo } from "@shared/types";

export type ModelMergeBadgeKind =
  | "merged-artifact"
  | "merged-output-exists"
  | "merged-output-missing";

export const getModelMergeBadgeKind = (
  model: ModelInfo,
): ModelMergeBadgeKind | null => {
  if (!["batch_trained", "pretrain", "daily_trained"].includes(model.type || "")) {
    return null;
  }

  if (
    model.artifactType === "merged" ||
    (!model.artifactType && model.artifactTypes?.includes("merged"))
  ) {
    return "merged-artifact";
  }

  // Older cached entries may not have artifact metadata yet.
  if (!model.artifactType) {
    const normalizedPath = (model.path || "")
      .replace(/\\/g, "/")
      .replace(/\/+$/, "");
    if (
      /(?:^|[_-])merged$/i.test(model.name) ||
      (model.type === "daily_trained" && /\/export$/i.test(normalizedPath))
    ) {
      return "merged-artifact";
    }
  }

  if (typeof model.merged !== "boolean") return null;
  return model.merged ? "merged-output-exists" : "merged-output-missing";
};
