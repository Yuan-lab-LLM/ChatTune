import { useEffect, useMemo, useState } from "react";
import { AlertTriangleIcon, ChevronDownIcon, ChevronRightIcon, CircleStopIcon, CpuIcon, RefreshCwIcon, ServerCogIcon, UnlockIcon } from "lucide-react";
import { useTranslation } from "react-i18next";

import { trpc } from "@/api/trpc";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { translateAuthError } from "@/utils/authErrors";

type NodeDraft = {
  enabled: boolean;
  sshAlias: string;
  trainAddress: string;
  allowedGpuIndexes: string;
  ncclSocketIfname: string;
};

type TrainingResourcePanelProps = {
  embedded?: boolean;
};

const numberValue = (value: string, fallback: number) => {
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed >= 0 ? parsed : fallback;
};

const activeReservationStatuses = ["preparing", "reserved", "running"];
const parseGpuIndexes = (value: string) => {
  const trimmed = value.trim();
  if (!trimmed) return [];
  if (!/^\d+(,\d+)*$/.test(trimmed)) return null;
  const indexes = trimmed.split(",").map(Number);
  return new Set(indexes).size === indexes.length ? indexes : null;
};
const hasDuplicate = (values: string[]) => new Set(values).size !== values.length;
const formatDateTime = (value?: string | Date | null) => {
  if (!value) return "-";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString();
};

const normalizedReleaseResult = (value?: string | null) => {
  const raw = String(value || "").trim().toLowerCase();
  if (!raw) return "";
  if (["success", "released", "gpu 释放成功", "gpu released", "gpu release succeeded"].includes(raw)) return "success";
  if (["force_released", "forced", "force released", "强制释放"].includes(raw)) return "force_released";
  if (["stopped_and_released", "stop_and_released", "stopped and released", "停止进程并释放"].includes(raw)) return "stopped_and_released";
  if (["failed", "failure"].includes(raw) || raw.includes("失败")) return "failed";
  return raw;
};

const normalizedReservationReason = (value?: string | null) => {
  const raw = String(value || "").trim();
  if (!raw) return "";
  if (raw === "Runtime 主动释放预约" || raw.toLowerCase() === "runtime released reservation") return "runtime_released";
  if (raw === "管理员强制释放" || raw.toLowerCase() === "admin force released") return "admin_force_released";
  if (raw === "管理员停止进程并释放预约") return "admin_stopped_process_and_released";
  if (raw === "管理员停止推理服务并释放预约") return "admin_stopped_inference_and_released";
  if (raw === "训练资源租约已过期") return "training_lease_expired";
  if (raw === "推理资源租约已过期") return "inference_lease_expired";
  if (raw.includes("已分配 GPU 仍有占用")) return "lease_expired_gpu_busy";
  return "";
};

const endedReservationStatuses = ["released", "failed"];

const normalizedTaskCategory = (value?: string | null) => {
  const raw = String(value || "").trim().toLowerCase();
  if (["assessment", "evaluation", "eval", "evaluate"].includes(raw)) return "assessment";
  if (["training", "train"].includes(raw)) return "training";
  if (["inference", "infer", "推理"].includes(raw)) return "inference";
  return raw;
};

const normalizedTaskType = (value?: string | null) => {
  const raw = String(value || "").trim();
  const lower = raw.toLowerCase();
  if (!lower) return "";
  if (raw === "推理服务" || lower === "inference" || lower === "inference_service") return "inference";
  if (["单模型评估", "单模型评测"].includes(raw) || lower.includes("single_model")) return "single_model_evaluation";
  if (raw.includes("双模型") || lower.includes("compare_between_models")) return "compare_between_models";
  if (raw.includes("checkpoint") || raw.includes("检查点") || lower.includes("ckpt_eval") || lower.includes("checkpoint")) return "ckpt_eval";
  if (lower === "lora" || lower === "lora sft") return "lora";
  if (raw.includes("全参") || lower.includes("full")) return "full";
  if (raw.includes("增强") || lower.includes("enhanced") || lower.includes("dpo")) return "enhanced";
  if (raw.includes("定时") || lower.includes("scheduled")) return "scheduled";
  if (lower.includes("grpo")) return "grpo";
  return lower;
};
const TrainingResourcePanel = ({ embedded = false }: TrainingResourcePanelProps) => {
  const { t } = useTranslation();
  const poolsQuery = trpc.listTrainingResourcePools.useQuery();
  const reservationsQuery = trpc.listTrainingReservations.useQuery();
  const groupsQuery = trpc.listResourceGroups.useQuery();
  const nodesQuery = trpc.listNodeAssignments.useQuery();
  const upsertPool = trpc.upsertTrainingResourcePool.useMutation();
  const setPoolEnabled = trpc.setTrainingResourcePoolEnabled.useMutation();
  const setPoolNodes = trpc.setTrainingResourcePoolNodes.useMutation();
  const setQuota = trpc.setTrainingGroupQuota.useMutation();
  const deleteQuota = trpc.deleteTrainingGroupQuota.useMutation();
  const releaseReservation = trpc.releaseTrainingResources.useMutation();
  const stopAndReleaseReservation = trpc.stopAndReleaseTrainingReservation.useMutation();

  const pools = poolsQuery.data?.data || [];
  const groups = groupsQuery.data?.data || [];
  const nodes = nodesQuery.data?.data || [];
  const reservations = reservationsQuery.data?.data || [];
  const [poolName, setPoolName] = useState("");
  const [poolDescription, setPoolDescription] = useState("");
  const [selectedPoolId, setSelectedPoolId] = useState("");
  const [selectedGroupId, setSelectedGroupId] = useState("");
  const [nodeDrafts, setNodeDrafts] = useState<Record<string, NodeDraft>>({});
  const [guaranteedGpuCount, setGuaranteedGpuCount] = useState("0");
  const [maxGpuCount, setMaxGpuCount] = useState("1");
  const [maxConcurrentJobs, setMaxConcurrentJobs] = useState("1");
  const [maxNodesPerJob, setMaxNodesPerJob] = useState("2");
  const [status, setStatus] = useState("");
  const [poolStatus, setPoolStatus] = useState<{ poolId: string; message: string } | null>(null);
  const [reservationFilter, setReservationFilter] = useState<"active" | "all" | "released" | "failed">("active");
  const [expandedReservationId, setExpandedReservationId] = useState<string | null>(null);
  const [reservationStatus, setReservationStatus] = useState("");
  const [forceReleasingReservationId, setForceReleasingReservationId] = useState<string | null>(null);
  const [stopReleasingReservationId, setStopReleasingReservationId] = useState<string | null>(null);

  const selectedPool = pools.find((pool) => pool.id === selectedPoolId);
  const selectedGroup = groups.find((group) => group.id === selectedGroupId);
  const selectedQuota = selectedPool?.quotas.find((quota) => quota.groupId === selectedGroupId);
  const groupNames = new Map(groups.map((group) => [group.id, group.name]));
  const selectedPoolSummary = selectedPool?.summary;
  const selectedPoolEnabledNodeIds = useMemo(
    () => new Set((selectedPool?.nodes || []).filter((node) => node.enabled).map((node) => node.nodeId)),
    [selectedPool?.nodes],
  );

  useEffect(() => {
    if (!selectedPoolId && pools[0]) setSelectedPoolId(pools[0].id);
  }, [pools, selectedPoolId]);

  useEffect(() => {
    if (!selectedGroupId && groups[0]) setSelectedGroupId(groups[0].id);
  }, [groups, selectedGroupId]);

  useEffect(() => {
    const configured = new Map((selectedPool?.nodes || []).map((node) => [node.nodeId, node]));
    setNodeDrafts(Object.fromEntries(nodes.map((node) => {
      const current = configured.get(node.id);
      return [node.id, {
        enabled: current?.enabled ?? false,
        sshAlias: current?.sshAlias || node.id,
        trainAddress: current?.trainAddress || "",
        allowedGpuIndexes: current?.allowedGpuIndexes?.join(",") || "",
        ncclSocketIfname: current?.ncclSocketIfname || "",
      }];
    })));
  }, [selectedPoolId, poolsQuery.data, nodesQuery.data]);

  useEffect(() => {
    setGuaranteedGpuCount(String(selectedQuota?.guaranteedGpuCount ?? 0));
    setMaxGpuCount(String(selectedQuota?.maxGpuCount ?? 1));
    setMaxConcurrentJobs(String(selectedQuota?.maxConcurrentJobs ?? 1));
    setMaxNodesPerJob(String(selectedQuota?.maxNodesPerJob ?? 2));
  }, [selectedQuota?.id]);

  const activeReservations = useMemo(
    () => reservations.filter((item) => activeReservationStatuses.includes(item.status)),
    [reservations],
  );
  const selectedPoolActiveReservations = useMemo(
    () => activeReservations.filter((item) => item.poolId === selectedPoolId),
    [activeReservations, selectedPoolId],
  );
  const selectedGroupActiveReservations = useMemo(
    () => selectedPoolActiveReservations.filter((item) => item.groupId === selectedGroupId),
    [selectedPoolActiveReservations, selectedGroupId],
  );
  const configLocked = selectedPoolActiveReservations.length > 0;
  const quotaLocked = selectedGroupActiveReservations.length > 0;
  const filteredReservations = useMemo(() => {
    if (reservationFilter === "all") return reservations;
    if (reservationFilter === "active") return activeReservations;
    return reservations.filter((item) => item.status === reservationFilter);
  }, [activeReservations, reservationFilter, reservations]);

  const quotaValidationMessage = useMemo(() => {
    if (!selectedPool) return "";
    if (selectedGroup?.nodeId && !selectedPoolEnabledNodeIds.has(selectedGroup.nodeId)) {
      return t("trainingResource.homeRuntimeNodeMustBeEnabledInPool", {
        nodeId: selectedGroup.nodeId,
      });
    }
    const guaranteed = numberValue(guaranteedGpuCount, 0);
    const maxGpu = numberValue(maxGpuCount, 1);
    if (guaranteed > maxGpu) return t("trainingResource.guaranteedGpuCannotExceedMax");
    const capacity = selectedPoolSummary?.capacityGpuCount ?? 0;
    if (capacity > 0 && maxGpu > capacity) return t("trainingResource.maxGpuCannotExceedPool");
    const otherGuaranteed = selectedPool.quotas
      .filter((quota) => quota.groupId !== selectedGroupId)
      .reduce((sum, quota) => sum + quota.guaranteedGpuCount, 0);
    if (capacity > 0 && otherGuaranteed + guaranteed > capacity) {
      return t("trainingResource.guaranteedGpuTotalCannotExceedPool");
    }
    return "";
  }, [
    guaranteedGpuCount,
    maxGpuCount,
    selectedGroup?.nodeId,
    selectedGroupId,
    selectedPool,
    selectedPoolEnabledNodeIds,
    selectedPoolSummary?.capacityGpuCount,
    t,
  ]);

  const refresh = async () => {
    await Promise.all([
      poolsQuery.refetch(),
      reservationsQuery.refetch(),
      groupsQuery.refetch(),
      nodesQuery.refetch(),
    ]);
  };

  const createPool = async () => {
    if (!poolName.trim()) return;
    try {
      const response = await upsertPool.mutateAsync({
        name: poolName.trim(),
        description: poolDescription.trim() || undefined,
      });
      setPoolName("");
      setPoolDescription("");
      setSelectedPoolId(response.data.id);
      setStatus(t("trainingResource.poolSaved"));
      await refresh();
    } catch (error) {
      setStatus(translateAuthError(error, t, "auth.adminActionFailed"));
    }
  };

  const changePoolEnabled = async (poolId: string, enabled: boolean) => {
    try {
      await setPoolEnabled.mutateAsync({ poolId, enabled });
      setPoolStatus({
        poolId,
        message: t(enabled ? "trainingResource.poolEnabledMessage" : "trainingResource.poolDisabledMessage"),
      });
      await refresh();
    } catch (error) {
      setPoolStatus({ poolId, message: translateAuthError(error, t, "auth.adminActionFailed") });
    }
  };

  const saveNodes = async () => {
    if (!selectedPoolId) return;
    if (configLocked) {
      setStatus(t("trainingResource.activeReservationsBlockConfig"));
      return;
    }
    const selectedNodes = nodes.flatMap((node) => {
      const draft = nodeDrafts[node.id];
      const allowedGpuIndexes = parseGpuIndexes(draft?.allowedGpuIndexes || "");
      return draft?.enabled ? [{
        nodeId: node.id,
        sshAlias: draft.sshAlias.trim(),
        trainAddress: draft.trainAddress.trim(),
        allowedGpuIndexes: allowedGpuIndexes || undefined,
        ncclSocketIfname: draft.ncclSocketIfname.trim() || undefined,
        enabled: true,
      }] : [];
    });
    if (selectedNodes.some((node) => node.allowedGpuIndexes === undefined
      && nodeDrafts[node.nodeId]?.allowedGpuIndexes.trim())) {
      setStatus(t("trainingResource.allowedGpuIndexesInvalid"));
      return;
    }
    if (!selectedNodes.length || selectedNodes.some((node) => !node.sshAlias || !node.trainAddress)) {
      setStatus(t("trainingResource.nodeConfigRequired"));
      return;
    }
    if (hasDuplicate(selectedNodes.map((node) => node.sshAlias))) {
      setStatus(t("trainingResource.poolNodeFieldDuplicate", { field: "sshAlias" }));
      return;
    }
    if (hasDuplicate(selectedNodes.map((node) => node.trainAddress))) {
      setStatus(t("trainingResource.poolNodeFieldDuplicate", { field: "trainAddress" }));
      return;
    }
    try {
      await setPoolNodes.mutateAsync({ poolId: selectedPoolId, nodes: selectedNodes });
      setStatus(t("trainingResource.poolNodesSaved"));
      await refresh();
    } catch (error) {
      setStatus(translateAuthError(error, t, "auth.adminActionFailed"));
    }
  };

  const saveQuota = async () => {
    if (configLocked || quotaLocked) {
      setStatus(t("trainingResource.activeReservationsBlockConfig"));
      return;
    }
    if (!selectedPoolId || !selectedGroupId || !selectedGroup?.nodeId) {
      setStatus(t("trainingResource.groupRuntimeNodeRequired"));
      return;
    }
    if (quotaValidationMessage) {
      setStatus(quotaValidationMessage);
      return;
    }
    try {
      await setQuota.mutateAsync({
        groupId: selectedGroupId,
        poolId: selectedPoolId,
        homeNodeId: selectedGroup.nodeId,
        guaranteedGpuCount: numberValue(guaranteedGpuCount, 0),
        maxGpuCount: numberValue(maxGpuCount, 1),
        maxConcurrentJobs: numberValue(maxConcurrentJobs, 1),
        maxNodesPerJob: numberValue(maxNodesPerJob, 1),
      });
      setStatus(t("trainingResource.quotaSaved"));
      await refresh();
    } catch (error) {
      setStatus(translateAuthError(error, t, "auth.adminActionFailed"));
    }
  };

  const removeQuota = async () => {
    if (!selectedPoolId || !selectedGroupId || !selectedQuota) return;
    if (configLocked || quotaLocked) {
      setStatus(t("trainingResource.activeReservationsBlockConfig"));
      return;
    }
    try {
      await deleteQuota.mutateAsync({
        groupId: selectedGroupId,
        poolId: selectedPoolId,
      });
      setStatus(t("trainingResource.quotaDeleted"));
      await refresh();
    } catch (error) {
      setStatus(translateAuthError(error, t, "auth.adminActionFailed"));
    }
  };

  const forceReleaseReservation = async (reservationId: string) => {
    if (!window.confirm(t("trainingResource.forceReleaseConfirm"))) return;
    setForceReleasingReservationId(reservationId);
    setReservationStatus("");
    try {
      await releaseReservation.mutateAsync({ reservationId, force: true });
      setReservationStatus(t("trainingResource.forceReleaseSuccess"));
      await refresh();
    } catch (error) {
      setReservationStatus(translateAuthError(error, t, "auth.adminActionFailed"));
    } finally {
      setForceReleasingReservationId(null);
    }
  };

  const stopAndReleaseReservationAction = async (reservationId: string, isInference: boolean) => {
    if (!window.confirm(t(isInference ? "trainingResource.stopInferenceAndReleaseConfirm" : "trainingResource.stopAndReleaseConfirm"))) return;
    setStopReleasingReservationId(reservationId);
    setReservationStatus("");
    try {
      const response = await stopAndReleaseReservation.mutateAsync({ reservationId });
      const stopResult = response.data?.stopResult;
      setReservationStatus(t(
        isInference
          ? "trainingResource.stopInferenceAndReleaseSuccess"
          : stopResult?.gpuIdle && stopResult?.remainingNonGpuPids?.length
            ? "trainingResource.stopAndReleaseGpuIdleSuccess"
            : "trainingResource.stopAndReleaseSuccess",
      ));
      await refresh();
    } catch (error) {
      setReservationStatus(translateAuthError(error, t, "auth.adminActionFailed"));
    } finally {
      setStopReleasingReservationId(null);
    }
  };
  const sectionClassName = embedded
    ? "rounded-xl border border-border/45 bg-background p-3"
    : "rounded-2xl border border-border/50 bg-card/95 p-4 shadow-sm";

  const renderPoolEditor = () => (
    <div className="grid gap-3 xl:grid-cols-2">
      <div className="rounded-xl border border-border/40 p-3">
        <h4 className="mb-2 text-sm font-semibold">{t("trainingResource.poolNodes")}</h4>
        {configLocked && (
          <div className="mb-3 flex items-center gap-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800">
            <AlertTriangleIcon className="size-3.5" />
            {t("trainingResource.activeReservationsBlockConfig")}
          </div>
        )}
        <div className="grid gap-2">
          {nodes.map((node) => {
            const draft = nodeDrafts[node.id] || { enabled: false, sshAlias: "", trainAddress: "", allowedGpuIndexes: "", ncclSocketIfname: "" };
            const runtimeSummary = selectedPoolSummary?.nodes?.find((item) => item.nodeId === node.id);
            const isSelectedGroupHomeNode = selectedGroup?.nodeId === node.id;
            return (
              <div key={node.id} className="grid gap-2 rounded-lg bg-muted/20 p-2 md:grid-cols-[auto_1fr_1fr_1fr]">
                <label className="flex items-center gap-2 text-sm">
                  <input type="checkbox" checked={draft.enabled} disabled={configLocked} onChange={(event) => setNodeDrafts((current) => ({ ...current, [node.id]: { ...draft, enabled: event.target.checked } }))} />
                  {node.name}
                  {isSelectedGroupHomeNode && (
                    <span className="rounded-full bg-primary/10 px-2 py-0.5 text-[11px] font-medium text-primary">
                      {t("trainingResource.masterNodeBadge")}
                    </span>
                  )}
                </label>
                <Input value={draft.sshAlias} onChange={(event) => setNodeDrafts((current) => ({ ...current, [node.id]: { ...draft, sshAlias: event.target.value } }))} placeholder={t("trainingResource.sshAliasPlaceholder")} />
                <Input value={draft.trainAddress} onChange={(event) => setNodeDrafts((current) => ({ ...current, [node.id]: { ...draft, trainAddress: event.target.value } }))} placeholder={t("trainingResource.trainAddressPlaceholder")} />
                <Input value={draft.allowedGpuIndexes} onChange={(event) => setNodeDrafts((current) => ({ ...current, [node.id]: { ...draft, allowedGpuIndexes: event.target.value } }))} placeholder={t("trainingResource.allowedGpuIndexesPlaceholder")} />
                {draft.enabled && runtimeSummary && (
                  <div className={`md:col-span-4 rounded-md px-2 py-1 text-xs ${runtimeSummary.status === "online" ? "bg-emerald-50 text-emerald-700" : "bg-rose-50 text-rose-700"}`}>
                    {runtimeSummary.status === "online"
                      ? t("trainingResource.nodeOnlineSummary", {
                        gpu: runtimeSummary.gpuCount,
                        available: runtimeSummary.availableGpuCount,
                      })
                      : t("trainingResource.nodeOfflineSummary", {
                        reason: runtimeSummary.errorMessage || t("trainingResource.unknownError"),
                      })}
                  </div>
                )}
                <details className="md:col-span-4">
                  <summary className="cursor-pointer text-xs font-medium text-muted-foreground">
                    {t("trainingResource.advancedNodeSettings")}
                  </summary>
                  <Input
                    className="mt-2"
                    value={draft.ncclSocketIfname}
                    onChange={(event) => setNodeDrafts((current) => ({ ...current, [node.id]: { ...draft, ncclSocketIfname: event.target.value } }))}
                    placeholder={t("trainingResource.ncclSocketPlaceholder")}
                  />
                </details>
              </div>
            );
          })}
        </div>
        {status && (
          <div className="mt-3 rounded-lg border border-sky-200 bg-sky-50 px-3 py-2 text-xs text-sky-700">
            {status}
          </div>
        )}
        <Button type="button" className="mt-3" onClick={saveNodes} disabled={setPoolNodes.isPending || configLocked}>{t("trainingResource.savePoolNodes")}</Button>
      </div>

      <div className="rounded-xl border border-border/40 p-3">
        <div className="mb-3 grid gap-2">
          <label className="grid gap-1 text-xs text-muted-foreground">
            {t("trainingResource.configureGroupQuota")}
            <select className="h-9 rounded-md border border-input bg-background px-3 text-sm" value={selectedGroupId} onChange={(event) => setSelectedGroupId(event.target.value)}>
              {groups.map((group) => <option key={group.id} value={group.id}>{group.name}</option>)}
            </select>
          </label>
          <div className="rounded-lg bg-muted/20 px-3 py-2 text-xs text-muted-foreground">
            {t("trainingResource.homeRuntimeMaster", { nodeId: selectedGroup?.nodeId || t("trainingResource.unassigned") })}
          </div>
        </div>
        {(quotaLocked || quotaValidationMessage) && (
          <div className="mb-3 flex items-center gap-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800">
            <AlertTriangleIcon className="size-3.5" />
            {quotaLocked ? t("trainingResource.activeReservationsBlockConfig") : quotaValidationMessage}
          </div>
        )}
        <h4 className="mb-2 text-sm font-semibold">{t("trainingResource.groupQuota")}</h4>
        <div className="grid gap-2 sm:grid-cols-2">
          <label className="grid gap-1 text-xs font-medium text-muted-foreground">
            {t("trainingResource.guaranteedGpuPlaceholder")}
            <Input type="number" value={guaranteedGpuCount} onChange={(event) => setGuaranteedGpuCount(event.target.value)} placeholder={t("trainingResource.guaranteedGpuPlaceholder")} />
          </label>
          <label className="grid gap-1 text-xs font-medium text-muted-foreground">
            {t("trainingResource.maxGpuPlaceholder")}
            <Input type="number" value={maxGpuCount} onChange={(event) => setMaxGpuCount(event.target.value)} placeholder={t("trainingResource.maxGpuPlaceholder")} />
          </label>
          <label className="grid gap-1 text-xs font-medium text-muted-foreground">
            {t("trainingResource.maxConcurrentJobsPlaceholder")}
            <Input type="number" value={maxConcurrentJobs} onChange={(event) => setMaxConcurrentJobs(event.target.value)} placeholder={t("trainingResource.maxConcurrentJobsPlaceholder")} />
          </label>
          <label className="grid gap-1 text-xs font-medium text-muted-foreground">
            {t("trainingResource.maxNodesPerJobPlaceholder")}
            <Input type="number" value={maxNodesPerJob} onChange={(event) => setMaxNodesPerJob(event.target.value)} placeholder={t("trainingResource.maxNodesPerJobPlaceholder")} />
          </label>
        </div>
        <div className="mt-3 flex gap-2">
          <Button type="button" onClick={saveQuota} disabled={setQuota.isPending || configLocked || quotaLocked || Boolean(quotaValidationMessage)}>{t("trainingResource.saveGroupQuota")}</Button>
          {selectedQuota && <Button type="button" variant="outline" onClick={removeQuota} disabled={deleteQuota.isPending || configLocked || quotaLocked}>{t("trainingResource.deleteQuota")}</Button>}
        </div>
      </div>
    </div>
  );

  return (
    <>
      <section className={sectionClassName}>
        <div className="mb-4 flex items-start justify-between gap-3">
          {!embedded && (
            <div>
              <div className="flex items-center gap-2">
                <ServerCogIcon className="size-4 text-primary" />
                <h3 className="text-base font-semibold">{t("trainingResource.poolTitle")}</h3>
              </div>
              <p className="mt-1 text-xs text-muted-foreground">
                {t("trainingResource.poolDescription")}
              </p>
            </div>
          )}
          <Button type="button" size="sm" variant="outline" onClick={refresh}>
            <RefreshCwIcon className="size-3.5" />{t("trainingResource.refresh")}
          </Button>
        </div>
        {status && <div className="mb-3 rounded-lg border border-sky-200 bg-sky-50 px-3 py-2 text-xs text-sky-700">{status}</div>}
        <div className="grid gap-2 md:grid-cols-[1fr_2fr_auto]">
          <Input value={poolName} onChange={(event) => setPoolName(event.target.value)} placeholder={t("trainingResource.poolNamePlaceholder")} />
          <Input value={poolDescription} onChange={(event) => setPoolDescription(event.target.value)} placeholder={t("trainingResource.poolDescriptionPlaceholder")} />
          <Button type="button" onClick={createPool} disabled={upsertPool.isPending}>{t("trainingResource.createPool")}</Button>
        </div>

        <div className="mt-4">
          <div className="mb-2 flex items-center justify-between gap-2">
            <h4 className="text-sm font-semibold">{t("trainingResource.createdPools")}</h4>
            <span className="text-xs text-muted-foreground">{t("trainingResource.poolCount", { count: pools.length })}</span>
          </div>
          <div className="grid gap-2">
            {pools.map((pool) => {
              const enabledNodes = pool.nodes.filter((node) => node.enabled);
              const summary = pool.summary;
              return (
                <div
                  key={pool.id}
                  className="rounded-xl border border-border/45 bg-background p-3"
                >
                  <div className="flex flex-wrap items-start justify-between gap-2">
                    <div>
                      <div className="flex items-center gap-2 text-sm font-medium">
                        <span>{pool.name}</span>
                        <span className={pool.enabled
                          ? "rounded-full bg-emerald-100 px-2 py-0.5 text-[10px] text-emerald-700"
                          : "rounded-full bg-slate-200 px-2 py-0.5 text-[10px] text-slate-600"}
                        >
                          {t(pool.enabled ? "trainingResource.poolEnabledLabel" : "trainingResource.poolDisabledLabel")}
                        </span>
                      </div>
                      <div className="mt-1 text-xs text-muted-foreground">
                        {pool.description || t("trainingResource.noPoolDescription")}
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <Button
                        type="button"
                        size="sm"
                        variant="outline"
                        disabled={setPoolEnabled.isPending}
                        onClick={() => changePoolEnabled(pool.id, !pool.enabled)}
                      >
                        {t(pool.enabled ? "trainingResource.disablePool" : "trainingResource.enablePool")}
                      </Button>
                      <Button
                        type="button"
                        size="sm"
                        variant={pool.id === selectedPoolId ? "default" : "outline"}
                        onClick={() => setSelectedPoolId(pool.id)}
                      >
                        {pool.id === selectedPoolId ? t("trainingResource.configuringPool") : t("trainingResource.configurePool")}
                      </Button>
                    </div>
                  </div>
                  {poolStatus?.poolId === pool.id && (
                    <div className="mt-3 rounded-lg border border-sky-200 bg-sky-50 px-3 py-2 text-xs text-sky-700">
                      {poolStatus.message}
                    </div>
                  )}
                  <div className="mt-3 grid gap-2 text-xs text-muted-foreground md:grid-cols-3">
                    <div>{t("trainingResource.poolNodeSummary", { enabled: enabledNodes.length, total: pool.nodes.length })}</div>
                    <div>{t("trainingResource.poolQuotaSummary", { count: pool.quotas.length })}</div>
                    <div className="truncate">{pool.id}</div>
                  </div>
                  {summary && (
                    <div className="mt-3 grid gap-2 text-xs sm:grid-cols-2 lg:grid-cols-5">
                      <div className="rounded-lg bg-muted/20 px-3 py-2">
                        <div className="text-muted-foreground">{t("trainingResource.capacityGpu")}</div>
                        <div className="text-base font-semibold">{summary.capacityGpuCount}</div>
                      </div>
                      <div className="rounded-lg bg-muted/20 px-3 py-2">
                        <div className="text-muted-foreground">{t("trainingResource.availableGpu")}</div>
                        <div className="text-base font-semibold">{summary.availableGpuCount}</div>
                      </div>
                      <div className="rounded-lg bg-muted/20 px-3 py-2">
                        <div className="text-muted-foreground">{t("trainingResource.reservedGpu")}</div>
                        <div className="text-base font-semibold">{summary.reservedGpuCount}</div>
                      </div>
                      <div className="rounded-lg bg-muted/20 px-3 py-2">
                        <div className="text-muted-foreground">{t("trainingResource.guaranteedGpu")}</div>
                        <div className="text-base font-semibold">{summary.guaranteedGpuCount}</div>
                      </div>
                      <div className="rounded-lg bg-muted/20 px-3 py-2">
                        <div className="text-muted-foreground">{t("trainingResource.activeReservationCount")}</div>
                        <div className="text-base font-semibold">{summary.activeReservationCount}</div>
                      </div>
                    </div>
                  )}
                  {summary?.offlineNodeIds?.length > 0 && (
                    <div className="mt-3 flex items-center gap-2 rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-xs text-rose-700">
                      <AlertTriangleIcon className="size-3.5" />
                      {t("trainingResource.offlineNodes", { nodes: summary.offlineNodeIds.join(", ") })}
                    </div>
                  )}
                  <div className="mt-3 grid gap-2 lg:grid-cols-2">
                    <div className="rounded-lg bg-muted/20 p-2 text-xs">
                      <div className="mb-1 font-medium text-foreground">{t("trainingResource.poolNodes")}</div>
                      <div className="grid gap-1 text-muted-foreground">
                        {enabledNodes.map((node) => (
                          <div key={node.id} className="truncate">
                            {node.nodeId} · {node.sshAlias} · {node.trainAddress} · {t("trainingResource.allowedGpuIndexesInline", {
                              indexes: node.allowedGpuIndexes?.join(",") || t("trainingResource.allGpuIndexes"),
                            })}
                          </div>
                        ))}
                        {!enabledNodes.length && <div>{t("trainingResource.noPoolNodes")}</div>}
                      </div>
                    </div>
                    <div className="rounded-lg bg-muted/20 p-2 text-xs">
                      <div className="mb-1 font-medium text-foreground">{t("trainingResource.groupQuota")}</div>
                      <div className="grid gap-1 text-muted-foreground">
                        {pool.quotas.map((quota) => (
                          <div key={quota.id} className="truncate">
                            {groupNames.get(quota.groupId) || quota.groupId} · {t("trainingResource.quotaInlineSummary", {
                              guaranteed: quota.guaranteedGpuCount,
                              max: quota.maxGpuCount,
                              jobs: quota.maxConcurrentJobs,
                              nodes: quota.maxNodesPerJob,
                            })}
                          </div>
                        ))}
                        {!pool.quotas.length && <div>{t("trainingResource.noPoolQuotas")}</div>}
                      </div>
                    </div>
                  </div>
                  {pool.id === selectedPoolId && (
                    <div className="mt-3">
                      {renderPoolEditor()}
                    </div>
                  )}
                </div>
              );
            })}
            {!pools.length && (
              <div className="rounded-xl border border-dashed border-border/60 bg-background p-4 text-center text-xs text-muted-foreground">
                {t("trainingResource.noPools")}
              </div>
            )}
          </div>
        </div>
      </section>

      <section className={sectionClassName}>
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <CpuIcon className="size-4 text-primary" />
            <h3 className="text-base font-semibold">{t("trainingResource.reservationAudit")}</h3>
            <span className="text-xs text-muted-foreground">{t("trainingResource.reservationSummary", { active: activeReservations.length, total: reservations.length })}</span>
          </div>
          <select
            className="h-8 rounded-md border border-input bg-background px-2 text-xs"
            value={reservationFilter}
            onChange={(event) => setReservationFilter(event.target.value as typeof reservationFilter)}
          >
            <option value="active">{t("trainingResource.filterActive")}</option>
            <option value="all">{t("trainingResource.filterAll")}</option>
            <option value="released">{t("trainingResource.filterReleased")}</option>
            <option value="failed">{t("trainingResource.filterFailed")}</option>
          </select>
        </div>
        {reservationStatus && (
          <div className="mb-3 rounded-lg border border-sky-200 bg-sky-50 px-3 py-2 text-xs text-sky-700">{reservationStatus}</div>
        )}
        <div className="grid gap-2">
          {filteredReservations.slice(0, 30).map((reservation) => {
            const showMasterBadge = reservation.nodes.length > 1;
            const isExpanded = expandedReservationId === reservation.id;
            const reason = reservation.expiredReason || reservation.errorMessage;
            const reasonLabel = (value?: string | null) => {
              const normalized = normalizedReservationReason(value);
              return normalized
                ? t(`trainingResource.reason.${normalized}`, { defaultValue: value || "" })
                : value || t("trainingResource.noDiagnosticValue");
            };
            const taskCategory = normalizedTaskCategory(reservation.taskCategory);
            const taskCategoryLabel = taskCategory
              ? t(`trainingResource.taskCategory.${taskCategory}`, { defaultValue: reservation.taskCategory })
              : t("trainingResource.noDiagnosticValue");
            const taskType = normalizedTaskType(reservation.taskType || reservation.taskTypeText);
            const taskTypeLabel = taskType
              ? t(`trainingResource.taskType.${taskType}`, { defaultValue: reservation.taskTypeText || reservation.taskType })
              : t("trainingResource.noDiagnosticValue");
            const releaseResult = normalizedReleaseResult(reservation.releaseResult);
            const isInferenceReservation = taskCategory === "inference" || taskType === "inference";
            const releaseResultLabel = releaseResult
              ? t(`trainingResource.releaseResult.${releaseResult}`, { defaultValue: reservation.releaseResult })
              : t("trainingResource.noDiagnosticValue");
            const isActiveReservation = activeReservationStatuses.includes(reservation.status);
            const endedAt = reservation.endedAt || reservation.releasedAt || (endedReservationStatuses.includes(reservation.status) ? reservation.updatedAt : null);
            const endedAtLabel = endedAt
              ? formatDateTime(endedAt)
              : isActiveReservation
                ? t("trainingResource.notEnded")
                : t("trainingResource.noDiagnosticValue");
            const endReason = isActiveReservation
              ? ""
              : reservation.endReason || reservation.expiredReason || reservation.errorMessage || (releaseResult ? releaseResultLabel : "");
            const endReasonLabel = endReason
              ? reasonLabel(endReason)
              : isActiveReservation
                ? t("trainingResource.notEnded")
                : t("trainingResource.noDiagnosticValue");
            return (
              <div key={reservation.id} className="rounded-lg border border-border/45 bg-background p-3 text-xs shadow-xs">
                <div className="flex flex-wrap items-start justify-between gap-2">
                  <div className="min-w-0">
                    <div className="flex flex-wrap items-center gap-2 font-medium">
                      <span>{reservation.groupName} · {reservation.poolName}</span>
                      <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-slate-600">
                        {reservation.status}
                      </span>
                      <span className="rounded-full bg-sky-50 px-2 py-0.5 text-[10px] font-medium text-sky-700">
                        {taskCategoryLabel} · {taskTypeLabel}
                      </span>
                    </div>
                    <div className="mt-1 text-muted-foreground">
                      {t("trainingResource.reservationDetails", {
                        master: reservation.homeNodeId,
                        nodes: reservation.requestedNodeCount,
                        gpus: reservation.gpusPerNode,
                        expiresAt: formatDateTime(reservation.expiresAt),
                      })}
                    </div>
                    <div className="mt-1 text-muted-foreground">
                      {reservation.nodes.map((node) => `${node.nodeId}[${node.gpuIndexes.join(",")}]${showMasterBadge && node.isMaster ? "(master)" : ""}`).join(" · ")}
                    </div>
                  </div>
                  <div className="flex shrink-0 flex-wrap items-center gap-2">
                    {activeReservationStatuses.includes(reservation.status) && (
                      <>
                        <Button
                          type="button"
                          size="sm"
                          variant="destructive"
                          className="h-8 rounded-full px-3 text-xs shadow-xs"
                          disabled={stopAndReleaseReservation.isPending && stopReleasingReservationId === reservation.id}
                          onClick={() => stopAndReleaseReservationAction(reservation.id, isInferenceReservation)}
                        >
                          <CircleStopIcon className="size-3.5" />
                          {t(isInferenceReservation ? "trainingResource.stopInferenceAndRelease" : "trainingResource.stopAndRelease")}
                        </Button>
                        <Button
                          type="button"
                          size="sm"
                          variant="outline"
                          className="h-8 rounded-full px-3 text-xs shadow-xs"
                          disabled={releaseReservation.isPending && forceReleasingReservationId === reservation.id}
                          onClick={() => forceReleaseReservation(reservation.id)}
                        >
                          <UnlockIcon className="size-3.5" />
                          {t("trainingResource.forceRelease")}
                        </Button>
                      </>
                    )}
                    <Button
                      type="button"
                      size="sm"
                      variant="outline"
                      className="h-8 rounded-full px-3 text-xs shadow-xs"
                      onClick={() => setExpandedReservationId(isExpanded ? null : reservation.id)}
                    >
                      {isExpanded ? <ChevronDownIcon className="size-3.5" /> : <ChevronRightIcon className="size-3.5" />}
                      {t(isExpanded ? "trainingResource.collapseDiagnostics" : "trainingResource.expandDiagnostics")}
                    </Button>
                  </div>
                </div>
                {reservation.errorMessage && (
                  <div className="mt-3 rounded-md border border-rose-200 bg-rose-50 px-2 py-1 text-rose-700">
                    {reservation.errorMessage}
                  </div>
                )}
                {isExpanded && (
                  <div className="mt-3 grid gap-3 rounded-xl border border-border/50 bg-slate-50/60 p-3">
                    <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-4">
                      <div className="rounded-lg bg-background px-3 py-2 shadow-xs">
                        <div className="text-muted-foreground">{t("trainingResource.statusLabel")}</div>
                        <div className="mt-1 font-semibold text-foreground">{reservation.status}</div>
                      </div>
                      <div className="rounded-lg bg-background px-3 py-2 shadow-xs">
                        <div className="text-muted-foreground">{t("trainingResource.createdAtLabel")}</div>
                        <div className="mt-1 font-semibold text-foreground">{formatDateTime(reservation.createdAt)}</div>
                      </div>
                      <div className="rounded-lg bg-background px-3 py-2 shadow-xs">
                        <div className="text-muted-foreground">{t("trainingResource.taskCategoryLabel")}</div>
                        <div className="mt-1 font-semibold text-foreground">{taskCategoryLabel}</div>
                      </div>
                      <div className="rounded-lg bg-background px-3 py-2 shadow-xs">
                        <div className="text-muted-foreground">{t("trainingResource.taskTypeLabel")}</div>
                        <div className="mt-1 font-semibold text-foreground">{taskTypeLabel}</div>
                      </div>
                      <div className="rounded-lg bg-background px-3 py-2 shadow-xs">
                        <div className="text-muted-foreground">{t("trainingResource.expiresAtLabel")}</div>
                        <div className="mt-1 font-semibold text-foreground">{formatDateTime(reservation.expiresAt)}</div>
                      </div>
                      <div className="rounded-lg bg-background px-3 py-2 shadow-xs">
                        <div className="text-muted-foreground">{t("trainingResource.lastRenewedAtLabel")}</div>
                        <div className="mt-1 font-semibold text-foreground">{formatDateTime(reservation.lastRenewedAt)}</div>
                      </div>
                      <div className="rounded-lg bg-background px-3 py-2 shadow-xs">
                        <div className="text-muted-foreground">{t("trainingResource.endedAtLabel")}</div>
                        <div className="mt-1 font-semibold text-foreground">{endedAtLabel}</div>
                      </div>
                      <div className="rounded-lg bg-background px-3 py-2 shadow-xs xl:col-span-2">
                        <div className="text-muted-foreground">{t("trainingResource.endReasonLabel")}</div>
                        <div className="mt-1 font-semibold text-foreground">{endReasonLabel}</div>
                      </div>
                      <div className="rounded-lg bg-background px-3 py-2 shadow-xs">
                        <div className="text-muted-foreground">{t("trainingResource.releaseResultLabel")}</div>
                        <div className="mt-1 font-semibold text-foreground">{releaseResultLabel}</div>
                      </div>
                    </div>

                    <div className="grid gap-2 md:grid-cols-3">
                      <div className="rounded-lg border border-border/40 bg-background p-3 shadow-xs">
                        <div className="font-medium text-foreground">{t("trainingResource.expiredReasonLabel")}</div>
                        <div className="mt-1 text-muted-foreground">{reasonLabel(reservation.expiredReason)}</div>
                      </div>
                      <div className="rounded-lg border border-border/40 bg-background p-3 shadow-xs">
                        <div className="font-medium text-foreground">{t("trainingResource.errorReasonLabel")}</div>
                        <div className="mt-1 text-muted-foreground">{reasonLabel(reason)}</div>
                      </div>
                    </div>

                    <div className="rounded-lg border border-border/40 bg-background p-3 shadow-xs">
                      <div className="mb-2 font-medium text-foreground">{t("trainingResource.gpuAllocationDetails")}</div>
                      <div className="grid gap-2 md:grid-cols-2">
                        {reservation.nodes.map((node) => (
                          <div key={node.id} className="rounded-md border border-border/35 bg-muted/10 p-2">
                            <div className="flex flex-wrap items-center gap-2 font-medium text-foreground">
                              <span>{t("trainingResource.nodeLabel")}: {node.nodeId}</span>
                              {node.isMaster && (
                                <span className="rounded-full bg-primary/10 px-2 py-0.5 text-[10px] text-primary">
                                  {t("trainingResource.masterNodeBadge")}
                                </span>
                              )}
                            </div>
                            <div className="mt-1 text-muted-foreground">
                              {t("trainingResource.gpuIndexesLabel")}: {node.gpuIndexes.length ? node.gpuIndexes.join(", ") : t("trainingResource.noDiagnosticValue")}
                            </div>
                          </div>
                        ))}
                        {!reservation.nodes.length && (
                          <div className="rounded-md border border-dashed border-border/50 p-3 text-muted-foreground">
                            {t("trainingResource.noDiagnosticValue")}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
          {!filteredReservations.length && <div className="py-6 text-center text-xs text-muted-foreground">{t("trainingResource.noReservations")}</div>}
        </div>
      </section>
    </>
  );
};

export default TrainingResourcePanel;
