import {
  ChevronDownIcon,
  ChevronRightIcon,
  ClipboardListIcon,
  DownloadIcon,
  FilterIcon,
  SearchIcon,
  XIcon,
} from "lucide-react";
import { Fragment, useLayoutEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";

import { trpc } from "@/api/trpc";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { translateAuthError } from "@/utils/authErrors";
import {
  AUDIT_LEVELS,
  AUDIT_MODULES,
  AUDIT_OPERATION_TYPES,
  AUDIT_RESULTS,
  classifyAuditEventType,
} from "@shared/types/audit";
import type {
  AuditLevel,
  AuditModule,
  AuditOperationType,
  AuditResult,
} from "@shared/types/audit";

type AuditDetails = Record<string, unknown>;
type AuditEvent = {
  id: string;
  eventType: string;
  resourceId?: string | null;
  actorUserId?: string | null;
  details?: AuditDetails | null;
  createdAt: string | Date;
};
type AuditEntry = { key: string; value: string };
type AuditExportFormat = "csv" | "json";

const sensitiveKeyPattern = /password|hash|salt|token|secret/i;
const hiddenDetailKeys = new Set([
  "actorUsername", "actorRole", "actorGroupName", "actorGroupId",
  "targetUsername", "targetUserId", "targetRole",
  "requesterUsername", "requesterUserId", "ownerUsername", "ownerUserId",
  "groupName", "groupId", "previousGroupName", "previousGroupId",
  "poolName", "poolId", "nodeName", "nodeId", "homeNodeName", "homeNodeId",
  "resourceId", "resourceIds", "reservationId", "requestId", "nodeNames",
  "before", "after",
]);
const hiddenChangeKeys = new Set([
  "userId", "username", "targetUserId", "targetUsername",
  "requesterUserId", "requesterUsername", "ownerUserId", "ownerUsername",
  "groupId", "groupName", "name", "poolId", "poolName",
  "resourceId", "resourceIds", "reservationId", "requestId",
  "nodeId", "nodeName", "nodeIds", "nodeNames", "homeNodeId", "homeNodeName",
]);
const targetKeyPriority = [
  "targetUsername", "targetUserId", "requesterUsername", "requesterUserId",
  "groupName", "groupId", "poolName", "poolId", "itemKey",
  "resourceId", "reservationId", "requestId",
];
const auditEventTypes = [
  "resource_write", "resource_delete", "resource_published",
  "publication_requested", "publication_approved", "publication_rejected",
  "admin_user_created", "admin_user_role_updated", "admin_user_password_reset",
  "admin_user_disabled_set", "admin_user_sessions_revoked", "admin_user_deleted",
  "admin_user_resource_group_moved", "admin_resource_group_created",
  "admin_resource_group_deleted", "admin_resource_group_container_set",
  "admin_resource_group_evaluate_container_set", "admin_resource_group_grpo_container_set",
  "admin_resource_group_multinode_container_set", "admin_resource_group_node_set",
  "admin_resource_group_containers_validated", "admin_training_pool_upserted",
  "admin_training_pool_status_changed", "admin_training_pool_nodes_set",
  "admin_training_group_quota_set", "admin_training_group_quota_deleted",
  "admin_training_resources_reserved", "admin_training_resources_released",
  "admin_training_resources_stop_and_released", "training_reservation_created",
  "training_reservation_released", "training_reservation_force_released",
  "training_reservation_stopped_and_released", "training_reservation_release_failed",
  "training_reservation_renewed", "training_reservation_expired",
];

const compactJson = (value: unknown) => {
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
};
const valueText = (value: unknown) => {
  if (value === null || value === undefined || value === "") return "-";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (Array.isArray(value)) return value.length ? value.map(valueText).join(", ") : "[]";
  if (typeof value === "object") return compactJson(value);
  return String(value);
};
const parseAuditDateInput = (value: string) => {
  const normalized = value.trim().replace(/\//g, "-").replace(" ", "T");
  if (!normalized) return undefined;
  const withSeconds = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$/.test(normalized)
    ? `${normalized}:00`
    : normalized;
  const parsed = new Date(withSeconds);
  return Number.isNaN(parsed.getTime()) ? undefined : parsed.toISOString();
};
const namedValue = (details: AuditDetails, nameKey: string, idKey: string) => {
  const name = details[nameKey];
  if (typeof name === "string" && name.trim()) return name;
  return valueText(details[idKey]);
};
const isDefaultGroupValue = (value: unknown) =>
  value === "default-users" || value === "默认用户组" || value === "Default User Group";
const groupText = (value: unknown, t: ReturnType<typeof useTranslation>["t"]) =>
  isDefaultGroupValue(value) ? t("resourceAccess.defaultGroupName") : valueText(value);
const namedGroupValue = (details: AuditDetails, t: ReturnType<typeof useTranslation>["t"]) => {
  const name = details.groupName;
  if (typeof name === "string" && name.trim()) return groupText(name, t);
  return groupText(details.groupId, t);
};
const shortId = (value?: string | null) =>
  value && value.length > 12 ? `${value.slice(0, 8)}...${value.slice(-4)}` : value || "";
const sourceForEvent = (event: AuditEvent) => {
  if (event.details?.actorRole === "admin") return "admin";
  if (event.eventType.startsWith("admin_") || event.eventType.startsWith("inference_admin_")) return "admin";
  if (event.eventType.startsWith("training_reservation_") && !event.actorUserId) return "runtime";
  return event.actorUserId ? "user" : "system";
};
const isPlainRecord = (value: unknown): value is AuditDetails =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value);
const changedEntries = (before: unknown, after: unknown): AuditEntry[] => {
  if (isPlainRecord(before) && isPlainRecord(after)) {
    const keys = Array.from(new Set([...Object.keys(before), ...Object.keys(after)]));
    return keys
      .filter((key) => !sensitiveKeyPattern.test(key))
      .filter((key) => !hiddenChangeKeys.has(key))
      .filter((key) => valueText(before[key]) !== valueText(after[key]))
      .map((key) => ({ key, value: `${valueText(before[key])} -> ${valueText(after[key])}` }));
  }
  if (before !== undefined || after !== undefined) {
    return [{ key: "change", value: `${valueText(before)} -> ${valueText(after)}` }];
  }
  return [];
};
const fallbackEntries = (event: AuditEvent): AuditEntry[] => {
  const details = event.details || {};
  return Object.entries(details)
    .filter(([key]) => !hiddenDetailKeys.has(key) && !sensitiveKeyPattern.test(key))
    .map(([key, value]) => ({ key, value: valueText(value) }));
};
const buildEntries = (event: AuditEvent): AuditEntry[] => {
  const details = event.details || {};
  const changes = changedEntries(details.before, details.after);
  return changes.length ? changes : fallbackEntries(event);
};
const shouldShowEntrySummary = (event: AuditEvent, entries: AuditEntry[]) =>
  entries.length > 0 && (isPlainRecord(event.details?.before) || isPlainRecord(event.details?.after));
const targetForEvent = (event: AuditEvent, t: ReturnType<typeof useTranslation>["t"]) => {
  const details = event.details || {};
  if (details.targetUsername || details.targetUserId) return namedValue(details, "targetUsername", "targetUserId");
  if (details.groupName || details.groupId) return namedGroupValue(details, t);
  if (details.poolName || details.poolId) return namedValue(details, "poolName", "poolId");
  if (Array.isArray(details.resourceIds) && details.resourceIds.length) {
    return t("resourceAccess.auditResourceCount", { count: details.resourceIds.length });
  }
  const targetKey = targetKeyPriority.find((key) => details[key] || (key === "resourceId" && event.resourceId));
  const target = targetKey === "resourceId" ? event.resourceId : targetKey ? details[targetKey] : event.resourceId;
  return valueText(target);
};
const actorForEvent = (event: AuditEvent) => {
  const actorUsername = event.details?.actorUsername;
  if (typeof actorUsername === "string" && actorUsername.trim()) return actorUsername;
  return shortId(event.actorUserId) || "";
};
const objectForEvent = (event: AuditEvent) => {
  const type = event.eventType;
  if (type.startsWith("admin_user_") || type === "admin_user_resource_group_moved") return "user";
  if (type.startsWith("admin_resource_group_")) return "resourceGroup";
  if (type.startsWith("admin_training_pool_")) return "trainingPool";
  if (type.startsWith("admin_training_group_quota_")) return "trainingQuota";
  if (type.startsWith("admin_training_resources_") || type.startsWith("training_reservation_")) return "trainingReservation";
  if (type.startsWith("publication_")) return "publicationRequest";
  if (type.startsWith("resource_")) return "resource";
  return "systemObject";
};
const resultClassName = (result: string) => {
  if (result === "failure") return "border-red-200 bg-red-50 text-red-700";
  if (result === "warning") return "border-amber-200 bg-amber-50 text-amber-700";
  return "border-emerald-200 bg-emerald-50 text-emerald-700";
};
const levelClassName = (level: string) => {
  if (level === "high") return "border-red-200 bg-red-50 text-red-700";
  if (level === "medium") return "border-sky-200 bg-sky-50 text-sky-700";
  return "border-slate-200 bg-slate-50 text-slate-600";
};
const primaryDataId = (event: AuditEvent) => {
  const details = event.details || {};
  const value =
    event.resourceId ||
    details.reservationId ||
    details.requestId ||
    details.targetUserId ||
    details.groupId ||
    details.poolId ||
    event.id;
  return shortId(typeof value === "string" ? value : String(value));
};
const fieldLabel = (key: string, t: ReturnType<typeof useTranslation>["t"]) =>
  t(`resourceAccess.auditFields.${key}`, { defaultValue: key });
const entrySummary = (entries: AuditEntry[], t: ReturnType<typeof useTranslation>["t"]) =>
  entries.map((entry) => `${fieldLabel(entry.key, t)}: ${entry.value}`).join("; ");
const sanitizeDetails = (value: unknown): unknown => {
  if (Array.isArray(value)) return value.map(sanitizeDetails);
  if (!isPlainRecord(value)) return value;
  return Object.fromEntries(
    Object.entries(value)
      .filter(([key]) => !sensitiveKeyPattern.test(key))
      .map(([key, detailValue]) => [key, sanitizeDetails(detailValue)]),
  );
};
const prettyJson = (value: unknown) => {
  try {
    return JSON.stringify(sanitizeDetails(value), null, 2);
  } catch {
    return valueText(value);
  }
};
const operationContent = (
  event: AuditEvent,
  entries: AuditEntry[],
  target: string,
  t: ReturnType<typeof useTranslation>["t"],
) => {
  const details = event.details || {};
  const node = namedValue(details, "nodeName", "nodeId");
  const group = namedGroupValue(details, t);
  const pool = namedValue(details, "poolName", "poolId");
  const resourceCount = Array.isArray(details.resourceIds) && details.resourceIds.length
    ? t("resourceAccess.auditResourceCount", { count: details.resourceIds.length })
    : target;
  switch (event.eventType) {
    case "admin_user_created":
      return t("resourceAccess.auditContent.userCreated", { target, role: valueText(details.role), group });
    case "admin_user_password_reset":
      return t("resourceAccess.auditContent.passwordReset", { target });
    case "admin_user_role_updated":
      return t("resourceAccess.auditContent.roleUpdated", { target, changes: entrySummary(entries, t) || valueText(details.role) });
    case "admin_user_disabled_set":
      return t("resourceAccess.auditContent.userStatusChanged", {
        target,
        status: details.disabled ? t("resourceAccess.auditDisabled") : t("resourceAccess.auditEnabled"),
      });
    case "admin_user_sessions_revoked":
      return t("resourceAccess.auditContent.sessionsRevoked", { target });
    case "admin_user_deleted":
      return t("resourceAccess.auditContent.userDeleted", { target });
    case "admin_user_resource_group_moved":
      return t("resourceAccess.auditContent.userMoved", { target, changes: entrySummary(entries, t) || group });
    case "admin_resource_group_node_set":
      return t("resourceAccess.auditContent.groupNodeSet", { group: target, node });
    case "admin_resource_group_container_set":
    case "admin_resource_group_evaluate_container_set":
    case "admin_resource_group_grpo_container_set":
    case "admin_resource_group_multinode_container_set":
      return t("resourceAccess.auditContent.groupContainerSet", {
        group: target,
        container: valueText(details.containerName),
        changes: entrySummary(entries, t),
      });
    case "admin_training_pool_nodes_set":
      return t("resourceAccess.auditContent.poolNodesSet", { pool: target, nodes: valueText(details.nodeNames || details.nodeIds) });
    case "admin_training_group_quota_set":
      return t("resourceAccess.auditContent.quotaSet", { group, pool, changes: entrySummary(entries, t) });
    case "admin_training_group_quota_deleted":
      return t("resourceAccess.auditContent.quotaDeleted", { group, pool });
    case "resource_write":
      return t("resourceAccess.auditContent.resourceWrite", {
        bizType: valueText(details.bizType),
        itemKey: valueText(details.itemKey),
        container: valueText(details.containerName),
        node,
      });
    case "resource_delete":
      return t("resourceAccess.auditContent.resourceDelete", {
        bizType: valueText(details.bizType),
        itemKey: valueText(details.itemKey),
        container: valueText(details.containerName),
        node,
      });
    case "resource_published":
      return t("resourceAccess.auditContent.resourcePublished", { count: resourceCount });
    case "publication_requested":
      return t("resourceAccess.auditContent.publicationRequested", { target });
    case "publication_approved":
      return t("resourceAccess.auditContent.publicationApproved", { target });
    case "publication_rejected":
      return t("resourceAccess.auditContent.publicationRejected", { target, note: valueText(details.note) });
    default:
      return entrySummary(entries, t) || t(`resourceAccess.auditEvents.${event.eventType}`, { defaultValue: event.eventType });
  }
};
const downloadTextFile = (filename: string, mimeType: string, content: string) => {
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

const ResourceAuditPanel = () => {
  const { t, i18n } = useTranslation();
  const [search, setSearch] = useState("");
  const [eventType, setEventType] = useState("");
  const [resultFilter, setResultFilter] = useState<AuditResult | "">("");
  const [moduleFilter, setModuleFilter] = useState<AuditModule | "">("");
  const [levelFilter, setLevelFilter] = useState<AuditLevel | "">("");
  const [operationTypeFilter, setOperationTypeFilter] = useState<AuditOperationType | "">("");
  const [actorUserId, setActorUserId] = useState("");
  const [resourceId, setResourceId] = useState("");
  const [startAt, setStartAt] = useState("");
  const [endAt, setEndAt] = useState("");
  const [limit, setLimit] = useState("200");
  const [showAdvancedFilters, setShowAdvancedFilters] = useState(false);
  const [expandedEventId, setExpandedEventId] = useState<string | null>(null);
  const [exportFormat, setExportFormat] = useState<AuditExportFormat>("csv");
  const [statusMessage, setStatusMessage] = useState("");
  const [hasHorizontalOverflow, setHasHorizontalOverflow] = useState(false);
  const tableScrollRef = useRef<HTMLDivElement | null>(null);
  const floatingScrollbarRef = useRef<HTMLDivElement | null>(null);
  const floatingScrollbarInnerRef = useRef<HTMLDivElement | null>(null);
  const exportAudit = trpc.exportResourceAuditEvents.useMutation();
  const auditFilters = useMemo(
    () => ({
      search: search.trim() || undefined,
      eventType: eventType || undefined,
      result: resultFilter || undefined,
      module: moduleFilter || undefined,
      level: levelFilter || undefined,
      operationType: operationTypeFilter || undefined,
      actorUserId: actorUserId.trim() || undefined,
      resourceId: resourceId.trim() || undefined,
      startAt: parseAuditDateInput(startAt),
      endAt: parseAuditDateInput(endAt),
      limit: Number(limit) || 200,
    }),
    [
      actorUserId,
      endAt,
      eventType,
      levelFilter,
      limit,
      moduleFilter,
      operationTypeFilter,
      resourceId,
      resultFilter,
      search,
      startAt,
    ],
  );
  const datePlaceholder = i18n.language.startsWith("zh") ? "YYYY/MM/DD HH:mm" : "YYYY-MM-DD HH:mm";
  const audit = trpc.listResourceAuditEvents.useQuery(auditFilters);
  const events = (audit.data?.data || []) as AuditEvent[];
  const renderedEvents = useMemo(
    () =>
      events.map((event) => {
        const classification = classifyAuditEventType(event.eventType);
        return {
          event,
          entries: buildEntries(event),
          source: sourceForEvent(event),
          target: targetForEvent(event, t),
          objectType: objectForEvent(event),
          ...classification,
          dataId: primaryDataId(event),
        };
      }),
    [events, t],
  );

  useLayoutEffect(() => {
    const tableScroll = tableScrollRef.current;
    const floatingScrollbar = floatingScrollbarRef.current;
    const floatingScrollbarInner = floatingScrollbarInnerRef.current;
    if (!tableScroll || !floatingScrollbar || !floatingScrollbarInner) return;

    let syncingFromTable = false;
    let syncingFromFloatingScrollbar = false;
    const updateScrollbar = () => {
      const isOverflowing = tableScroll.scrollWidth > tableScroll.clientWidth + 1;
      setHasHorizontalOverflow(isOverflowing);
      floatingScrollbar.style.width = `${tableScroll.clientWidth}px`;
      floatingScrollbarInner.style.width = `${tableScroll.scrollWidth}px`;
      floatingScrollbar.scrollLeft = tableScroll.scrollLeft;
    };
    const handleTableScroll = () => {
      if (syncingFromFloatingScrollbar) return;
      syncingFromTable = true;
      floatingScrollbar.scrollLeft = tableScroll.scrollLeft;
      requestAnimationFrame(() => {
        syncingFromTable = false;
      });
    };
    const handleFloatingScrollbarScroll = () => {
      if (syncingFromTable) return;
      syncingFromFloatingScrollbar = true;
      tableScroll.scrollLeft = floatingScrollbar.scrollLeft;
      requestAnimationFrame(() => {
        syncingFromFloatingScrollbar = false;
      });
    };

    const frameId = requestAnimationFrame(updateScrollbar);
    const resizeObserver = new ResizeObserver(updateScrollbar);
    resizeObserver.observe(tableScroll);
    const table = tableScroll.querySelector("table");
    if (table) resizeObserver.observe(table);
    tableScroll.addEventListener("scroll", handleTableScroll, { passive: true });
    floatingScrollbar.addEventListener("scroll", handleFloatingScrollbarScroll, { passive: true });
    window.addEventListener("resize", updateScrollbar);

    return () => {
      cancelAnimationFrame(frameId);
      resizeObserver.disconnect();
      tableScroll.removeEventListener("scroll", handleTableScroll);
      floatingScrollbar.removeEventListener("scroll", handleFloatingScrollbarScroll);
      window.removeEventListener("resize", updateScrollbar);
    };
  }, [expandedEventId, i18n.language, renderedEvents.length]);

  const handleExport = async () => {
    setStatusMessage("");
    try {
      const result = await exportAudit.mutateAsync({
        ...auditFilters,
        limit: Math.min(Math.max(Number(auditFilters.limit) || 200, events.length || 1), 5000),
        format: exportFormat,
      });
      const payload = result.data;
      if (payload?.content && payload.filename && payload.mimeType) {
        downloadTextFile(payload.filename, payload.mimeType, payload.content);
        setStatusMessage(t("resourceAccess.auditExportSuccess"));
      }
    } catch (error) {
      setStatusMessage(translateAuthError(error, t, "resourceAccess.auditExportFailed"));
    }
  };

  return (
    <section className="w-full min-w-0 overflow-hidden rounded-2xl border border-border/50 bg-card/95 p-4 shadow-sm">
      <div className="mb-4">
        <div className="flex items-center gap-2">
          <ClipboardListIcon className="size-4 text-primary" />
          <h3 className="text-base font-semibold">{t("resourceAccess.auditLogTitle")}</h3>
        </div>
        <p className="mt-1 text-xs text-muted-foreground">{t("resourceAccess.auditLogDescription")}</p>
      </div>
      {statusMessage && (
        <div className="mb-3 rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2 text-xs text-emerald-700">
          {statusMessage}
        </div>
      )}
      <div className="mb-3 grid min-w-0 gap-2 rounded-lg border border-border/45 bg-muted/10 p-3">
        <div className="grid min-w-0 gap-2 md:grid-cols-2 xl:grid-cols-[minmax(220px,1fr)_minmax(160px,220px)_minmax(90px,110px)_minmax(90px,110px)]">
          <label className="text-xs text-muted-foreground">
            {t("resourceAccess.auditSearch")}
            <div className="relative mt-1">
              <SearchIcon className="pointer-events-none absolute left-2 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground" />
              <Input className="h-8 pl-7 text-xs" value={search} onChange={(event) => setSearch(event.target.value)} placeholder={t("resourceAccess.auditSearchPlaceholder")} />
            </div>
          </label>
          <label className="text-xs text-muted-foreground">
            {t("resourceAccess.auditEventType")}
            <select className="mt-1 h-8 w-full rounded-md border border-input bg-background px-2 text-xs text-foreground" value={eventType} onChange={(event) => setEventType(event.target.value)}>
              <option value="">{t("resourceAccess.auditAllEventTypes")}</option>
              {auditEventTypes.map((type) => (
                <option key={type} value={type}>{t(`resourceAccess.auditEvents.${type}`, { defaultValue: type })}</option>
              ))}
            </select>
          </label>
          <label className="text-xs text-muted-foreground">
            {t("resourceAccess.auditLimit")}
            <Input className="mt-1 h-8 text-xs" min={1} max={5000} type="number" value={limit} onChange={(event) => setLimit(event.target.value)} />
          </label>
          <label className="text-xs text-muted-foreground">
            {t("resourceAccess.auditExportFormat")}
            <select className="mt-1 h-8 w-full rounded-md border border-input bg-background px-2 text-xs text-foreground" value={exportFormat} onChange={(event) => setExportFormat(event.target.value as AuditExportFormat)}>
              <option value="csv">CSV</option>
              <option value="json">JSON</option>
            </select>
          </label>
        </div>
        <div className="grid min-w-0 gap-2 md:grid-cols-2 xl:grid-cols-4">
          <label className="text-xs text-muted-foreground">
            {t("resourceAccess.auditColumnResult")}
            <select className="mt-1 h-8 w-full rounded-md border border-input bg-background px-2 text-xs text-foreground" value={resultFilter} onChange={(event) => setResultFilter(event.target.value as AuditResult | "")}>
              <option value="">{t("resourceAccess.auditAllResults")}</option>
              {AUDIT_RESULTS.map((result) => (
                <option key={result} value={result}>{t(`resourceAccess.auditResults.${result}`)}</option>
              ))}
            </select>
          </label>
          <label className="text-xs text-muted-foreground">
            {t("resourceAccess.auditColumnModule")}
            <select className="mt-1 h-8 w-full rounded-md border border-input bg-background px-2 text-xs text-foreground" value={moduleFilter} onChange={(event) => setModuleFilter(event.target.value as AuditModule | "")}>
              <option value="">{t("resourceAccess.auditAllModules")}</option>
              {AUDIT_MODULES.map((module) => (
                <option key={module} value={module}>{t(`resourceAccess.auditModules.${module}`, { defaultValue: module })}</option>
              ))}
            </select>
          </label>
          <label className="text-xs text-muted-foreground">
            {t("resourceAccess.auditColumnLevel")}
            <select className="mt-1 h-8 w-full rounded-md border border-input bg-background px-2 text-xs text-foreground" value={levelFilter} onChange={(event) => setLevelFilter(event.target.value as AuditLevel | "")}>
              <option value="">{t("resourceAccess.auditAllLevels")}</option>
              {AUDIT_LEVELS.map((level) => (
                <option key={level} value={level}>{t(`resourceAccess.auditLevels.${level}`)}</option>
              ))}
            </select>
          </label>
          <label className="text-xs text-muted-foreground">
            {t("resourceAccess.auditColumnType")}
            <select className="mt-1 h-8 w-full rounded-md border border-input bg-background px-2 font-mono text-xs text-foreground" value={operationTypeFilter} onChange={(event) => setOperationTypeFilter(event.target.value as AuditOperationType | "")}>
              <option className="font-sans" value="">{t("resourceAccess.auditAllOperationTypes")}</option>
              {AUDIT_OPERATION_TYPES.map((operationType) => (
                <option key={operationType} value={operationType}>{operationType}</option>
              ))}
            </select>
          </label>
        </div>
        <div className="flex min-w-0 flex-wrap items-end gap-2">
            <Button className="h-8 shrink-0 gap-1 px-2 text-xs" type="button" variant="outline" onClick={() => setShowAdvancedFilters((value) => !value)}>
              <FilterIcon className="size-3.5" />
              {t("resourceAccess.auditAdvancedFilters")}
            </Button>
            <Button
              className="h-8 shrink-0 gap-1 px-2 text-xs"
              type="button"
              variant="outline"
              onClick={() => {
                setSearch("");
                setEventType("");
                setResultFilter("");
                setModuleFilter("");
                setLevelFilter("");
                setOperationTypeFilter("");
                setActorUserId("");
                setResourceId("");
                setStartAt("");
                setEndAt("");
                setLimit("200");
              }}
            >
              <XIcon className="size-3.5" />
              {t("resourceAccess.auditClearFilters")}
            </Button>
            <Button className="h-8 shrink-0 gap-1 px-2 text-xs" type="button" onClick={() => void handleExport()} disabled={exportAudit.isPending}>
              <DownloadIcon className="size-3.5" />
              {exportAudit.isPending ? t("resourceAccess.auditExporting") : t("resourceAccess.auditExport")}
            </Button>
        </div>
        {showAdvancedFilters && (
          <div className="grid gap-2 md:grid-cols-2 lg:grid-cols-4">
            <label className="text-xs text-muted-foreground">
              {t("resourceAccess.auditActorId")}
              <Input className="mt-1 h-8 text-xs" value={actorUserId} onChange={(event) => setActorUserId(event.target.value)} placeholder={t("resourceAccess.auditActorIdPlaceholder")} />
            </label>
            <label className="text-xs text-muted-foreground">
              {t("resourceAccess.auditResourceId")}
              <Input className="mt-1 h-8 text-xs" value={resourceId} onChange={(event) => setResourceId(event.target.value)} placeholder={t("resourceAccess.auditResourceIdPlaceholder")} />
            </label>
            <label className="text-xs text-muted-foreground">
              {t("resourceAccess.auditStartAt")}
              <Input className="mt-1 h-8 text-xs" value={startAt} onChange={(event) => setStartAt(event.target.value)} placeholder={datePlaceholder} />
              <span className="mt-1 block text-[10px] text-muted-foreground">{t("resourceAccess.auditDateFormatHint")}</span>
            </label>
            <label className="text-xs text-muted-foreground">
              {t("resourceAccess.auditEndAt")}
              <Input className="mt-1 h-8 text-xs" value={endAt} onChange={(event) => setEndAt(event.target.value)} placeholder={datePlaceholder} />
              <span className="mt-1 block text-[10px] text-muted-foreground">{t("resourceAccess.auditDateFormatHint")}</span>
            </label>
          </div>
        )}
      </div>
      <div className="relative w-full max-w-full">
        <div
          ref={tableScrollRef}
          className={`max-h-[52vh] w-full max-w-full overflow-x-auto overflow-y-auto rounded-lg border border-border/45 ${hasHorizontalOverflow ? "pb-4" : ""}`}
        >
          <table className="w-[96rem] min-w-full table-fixed border-collapse text-left text-xs">
          <colgroup>
            <col className="w-[2.5rem]" />
            <col className="w-[9.5rem]" />
            <col className="w-[8rem]" />
            <col className="w-[8rem]" />
            <col className="w-[10rem]" />
            <col className="w-[24rem]" />
            <col className="w-[6rem]" />
            <col className="w-[8.5rem]" />
            <col className="w-[5.5rem]" />
            <col className="w-[6rem]" />
            <col className="w-[7.5rem]" />
          </colgroup>
          <thead className="sticky top-0 z-10 bg-muted/55 text-muted-foreground">
            <tr className="[&>th]:whitespace-nowrap [&>th]:border-b [&>th]:border-border/45 [&>th]:px-3 [&>th]:py-2 [&>th]:font-medium">
              <th aria-label={t("resourceAccess.auditColumnDetails")} />
              <th>{t("resourceAccess.auditColumnTime")}</th>
              <th>{t("resourceAccess.auditColumnActor")}</th>
              <th>{t("resourceAccess.auditColumnGroup")}</th>
              <th>{t("resourceAccess.auditColumnObject")}</th>
              <th>{t("resourceAccess.auditColumnContent")}</th>
              <th>{t("resourceAccess.auditColumnResult")}</th>
              <th>{t("resourceAccess.auditColumnModule")}</th>
              <th>{t("resourceAccess.auditColumnLevel")}</th>
              <th>{t("resourceAccess.auditColumnType")}</th>
              <th>{t("resourceAccess.auditColumnDataId")}</th>
            </tr>
          </thead>
          <tbody>
            {renderedEvents.map(({ event, entries, source, target, objectType, module, operationType, result, level, dataId }) => {
              const details = event.details || {};
              const isExpanded = expandedEventId === event.id;
              const hasBeforeAfter = details.before !== undefined || details.after !== undefined;
              const hasExpandedDetails = entries.length > 0 || hasBeforeAfter || Object.keys(details).length > 0;
              return (
                <Fragment key={event.id}>
                  <tr className="border-b border-border/35 hover:bg-muted/20">
                    <td className="px-2 py-3 align-top">
                      <Button aria-expanded={isExpanded} aria-label={t("resourceAccess.auditToggleDetails")} className="size-7 p-0" disabled={!hasExpandedDetails} type="button" variant="ghost" onClick={() => setExpandedEventId(isExpanded ? null : event.id)}>
                        {isExpanded ? <ChevronDownIcon className="size-3.5" /> : <ChevronRightIcon className="size-3.5" />}
                      </Button>
                    </td>
                    <td className="break-words px-3 py-3 align-top text-muted-foreground">{new Date(event.createdAt).toLocaleString()}</td>
                    <td className="break-words px-3 py-3 align-top">
                      <div className="font-medium text-foreground">{actorForEvent(event) || t("resourceAccess.systemActor")}</div>
                      <Badge variant="outline" className="mt-1 h-5 px-1.5 text-[10px] uppercase">
                        {t(`resourceAccess.auditSources.${source}`, { defaultValue: source })}
                      </Badge>
                    </td>
                    <td className="break-words px-3 py-3 align-top text-muted-foreground">{groupText(details.actorGroupName || details.actorGroupId, t)}</td>
                    <td className="break-words px-3 py-3 align-top">
                      <div className="font-medium text-foreground">{t(`resourceAccess.auditObjects.${objectType}`, { defaultValue: objectType })}</div>
                      <div className="mt-1 text-[11px] text-muted-foreground">{target}</div>
                    </td>
                    <td className="break-words px-3 py-3 align-top text-foreground">
                      <div>{operationContent(event, entries, target, t)}</div>
                      {shouldShowEntrySummary(event, entries) && <div className="mt-1 text-[11px] text-muted-foreground">{entrySummary(entries, t)}</div>}
                    </td>
                    <td className="whitespace-nowrap px-3 py-3 align-top">
                      <Badge variant="outline" className={`h-5 px-2 text-[10px] ${resultClassName(result)}`}>{t(`resourceAccess.auditResults.${result}`)}</Badge>
                    </td>
                    <td className="break-words px-3 py-3 align-top text-muted-foreground">{t(`resourceAccess.auditModules.${module}`, { defaultValue: module })}</td>
                    <td className="whitespace-nowrap px-3 py-3 align-top">
                      <Badge variant="outline" className={`h-5 px-2 text-[10px] ${levelClassName(level)}`}>{t(`resourceAccess.auditLevels.${level}`)}</Badge>
                    </td>
                    <td className="whitespace-nowrap px-3 py-3 align-top font-mono text-[11px] text-muted-foreground">{operationType}</td>
                    <td className="whitespace-nowrap px-3 py-3 align-top font-mono text-[11px] text-muted-foreground">{dataId}</td>
                  </tr>
                  {isExpanded && (
                    <tr className="border-b border-border/35 bg-muted/10">
                      <td />
                      <td className="px-3 pb-4 pt-1" colSpan={10}>
                        <div className="grid gap-3 lg:grid-cols-[minmax(220px,0.8fr)_minmax(320px,1.2fr)]">
                          <div className="rounded-md border border-border/45 bg-background p-3">
                            <div className="text-[11px] font-medium uppercase text-muted-foreground">{t("resourceAccess.auditDetailSummary")}</div>
                            {entries.length ? (
                              <dl className="mt-2 grid gap-2 sm:grid-cols-2">
                                {entries.map((entry) => (
                                  <div key={entry.key} className="rounded-md bg-muted/25 p-2">
                                    <dt className="text-[10px] uppercase text-muted-foreground">{fieldLabel(entry.key, t)}</dt>
                                    <dd className="mt-1 break-words font-mono text-[11px] text-foreground">{entry.value}</dd>
                                  </div>
                                ))}
                              </dl>
                            ) : (
                              <div className="mt-2 text-xs text-muted-foreground">{t("resourceAccess.auditNoDetails")}</div>
                            )}
                          </div>
                          <div className="grid gap-3 md:grid-cols-2">
                            {hasBeforeAfter && (
                              <>
                                <div className="rounded-md border border-border/45 bg-background p-3">
                                  <div className="text-[11px] font-medium uppercase text-muted-foreground">{t("resourceAccess.auditDetailBefore")}</div>
                                  <pre className="mt-2 max-h-56 overflow-auto whitespace-pre-wrap break-words text-[11px] leading-5 text-foreground">{prettyJson(details.before)}</pre>
                                </div>
                                <div className="rounded-md border border-border/45 bg-background p-3">
                                  <div className="text-[11px] font-medium uppercase text-muted-foreground">{t("resourceAccess.auditDetailAfter")}</div>
                                  <pre className="mt-2 max-h-56 overflow-auto whitespace-pre-wrap break-words text-[11px] leading-5 text-foreground">{prettyJson(details.after)}</pre>
                                </div>
                              </>
                            )}
                            <div className="rounded-md border border-border/45 bg-background p-3 md:col-span-2">
                              <div className="text-[11px] font-medium uppercase text-muted-foreground">{t("resourceAccess.auditDetailRaw")}</div>
                              <pre className="mt-2 max-h-56 overflow-auto whitespace-pre-wrap break-words text-[11px] leading-5 text-foreground">{prettyJson(details)}</pre>
                            </div>
                          </div>
                        </div>
                      </td>
                    </tr>
                  )}
                </Fragment>
              );
            })}
            {!events.length && (
              <tr>
                <td className="px-3 py-4 text-muted-foreground" colSpan={11}>{t("resourceAccess.noAuditRecords")}</td>
              </tr>
            )}
          </tbody>
          </table>
        </div>
        <div
          ref={floatingScrollbarRef}
          aria-hidden={!hasHorizontalOverflow}
          aria-label={t("resourceAccess.auditHorizontalScroll")}
          className={`absolute bottom-px left-px z-20 h-4 overflow-x-scroll overflow-y-hidden rounded-b-lg border-t border-border/60 bg-background/95 shadow-[0_-2px_6px_rgba(15,23,42,0.08)] backdrop-blur-sm ${hasHorizontalOverflow ? "" : "pointer-events-none invisible"}`}
          role="region"
          tabIndex={hasHorizontalOverflow ? 0 : -1}
        >
          <div ref={floatingScrollbarInnerRef} className="h-px" />
        </div>
      </div>
    </section>
  );
};

export default ResourceAuditPanel;
