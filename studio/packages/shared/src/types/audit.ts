export const AUDIT_RESULTS = ["success", "warning", "failure"] as const;
export type AuditResult = (typeof AUDIT_RESULTS)[number];

export const AUDIT_MODULES = [
  "auth",
  "resource",
  "resourceGroup",
  "trainingResource",
  "inferenceAdmin",
  "other",
] as const;
export type AuditModule = (typeof AUDIT_MODULES)[number];

export const AUDIT_LEVELS = ["low", "medium", "high"] as const;
export type AuditLevel = (typeof AUDIT_LEVELS)[number];

export const AUDIT_OPERATION_TYPES = [
  "CREATE",
  "UPDATE",
  "DELETE",
  "APPROVE",
  "REJECT",
  "RELEASE",
  "RENEW",
] as const;
export type AuditOperationType = (typeof AUDIT_OPERATION_TYPES)[number];

export type AuditClassification = {
  result: AuditResult;
  module: AuditModule;
  level: AuditLevel;
  operationType: AuditOperationType;
};

export type AuditClassificationFilters = Partial<AuditClassification>;

export const classifyAuditEventType = (eventType: string): AuditClassification => {
  let module: AuditModule = "other";
  if (eventType.startsWith("admin_user_")) module = "auth";
  else if (eventType.startsWith("admin_resource_group_")) module = "resourceGroup";
  else if (eventType.startsWith("admin_training_") || eventType.startsWith("training_reservation_")) {
    module = "trainingResource";
  } else if (eventType.startsWith("inference_admin_")) module = "inferenceAdmin";
  else if (eventType.startsWith("resource_") || eventType.startsWith("publication_")) module = "resource";

  let operationType: AuditOperationType = "UPDATE";
  if (eventType.includes("created") || eventType.endsWith("_created") || eventType.endsWith("_requested")) {
    operationType = "CREATE";
  } else if (eventType.includes("deleted") || eventType.endsWith("_delete") || eventType.endsWith("_deleted")) {
    operationType = "DELETE";
  } else if (eventType.includes("rejected")) operationType = "REJECT";
  else if (eventType.includes("approved") || eventType.includes("published")) operationType = "APPROVE";
  else if (eventType.includes("released")) operationType = "RELEASE";
  else if (eventType.includes("renewed")) operationType = "RENEW";

  let level: AuditLevel = "medium";
  if (
    eventType.includes("delete") ||
    eventType.includes("deleted") ||
    eventType.includes("password_reset") ||
    eventType.includes("force_released") ||
    eventType.includes("failed")
  ) {
    level = "high";
  } else if (eventType.includes("requested") || eventType.includes("renewed") || eventType.includes("created")) {
    level = "low";
  }

  const result: AuditResult = eventType.includes("failed")
    ? "failure"
    : eventType.includes("rejected")
      ? "warning"
      : "success";

  return { result, module, level, operationType };
};

export const matchesAuditClassification = (
  eventType: string,
  filters: AuditClassificationFilters,
) => {
  const classification = classifyAuditEventType(eventType);
  return (
    (!filters.result || classification.result === filters.result) &&
    (!filters.module || classification.module === filters.module) &&
    (!filters.level || classification.level === filters.level) &&
    (!filters.operationType || classification.operationType === filters.operationType)
  );
};
