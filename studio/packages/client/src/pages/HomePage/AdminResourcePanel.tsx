import { useState } from "react";
import { BoxesIcon, RefreshCwIcon, ServerIcon, Trash2Icon } from "lucide-react";
import { useTranslation } from "react-i18next";

import { trpc } from "@/api/trpc";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { translateAuthError } from "@/utils/authErrors";

const secondaryActionClassName =
  "border-sky-200 bg-sky-50 text-sky-700 shadow-xs hover:border-sky-300 hover:bg-sky-100 hover:text-sky-800 dark:border-sky-400/25 dark:bg-sky-400/10 dark:text-sky-300 dark:hover:bg-sky-400/15";

const AdminResourcePanel = () => {
  const { t } = useTranslation();
  const [groupName, setGroupName] = useState("");
  const [containerName, setContainerName] = useState("");
  const [evaluateContainerName, setEvaluateContainerName] = useState("");
  const [grpoContainerName, setGrpoContainerName] = useState("");
  const [multinodeContainerName, setMultinodeContainerName] = useState("");
  const [containerDrafts, setContainerDrafts] = useState<Record<string, string>>({});
  const [evaluateContainerDrafts, setEvaluateContainerDrafts] = useState<Record<string, string>>({});
  const [grpoContainerDrafts, setGrpoContainerDrafts] = useState<Record<string, string>>({});
  const [multinodeContainerDrafts, setMultinodeContainerDrafts] = useState<Record<string, string>>({});
  const [validatingGroupId, setValidatingGroupId] = useState<string | null>(null);
  const [savingContainerKey, setSavingContainerKey] = useState<string | null>(null);
  const [resourceActionKey, setResourceActionKey] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState("");
  const groups = trpc.listResourceGroups.useQuery();
  const nodes = trpc.listNodeAssignments.useQuery();
  const createGroup = trpc.createResourceGroup.useMutation();
  const deleteGroup = trpc.deleteResourceGroup.useMutation();
  const setGroupContainer = trpc.setResourceGroupContainer.useMutation();
  const setGroupEvaluateContainer = trpc.setResourceGroupEvaluateContainer.useMutation();
  const setGroupGrpoContainer = trpc.setResourceGroupGrpoContainer.useMutation();
  const setGroupMultinodeContainer = trpc.setResourceGroupMultinodeContainer.useMutation();
  const setNode = trpc.setResourceGroupNode.useMutation();
  const validateContainers = trpc.validateResourceGroupContainers.useMutation();

  const refresh = async () => {
    await Promise.all([groups.refetch(), nodes.refetch()]);
  };

  const groupItems = groups.data?.data || [];
  const nodeItems = nodes.data?.data || [];
  const getGroupDisplayName = (group: { id: string; name: string }) =>
    group.id === "default-users"
      ? t("resourceAccess.defaultGroupName")
      : group.name;

  const createNewGroup = async () => {
    if (
      !groupName.trim() ||
      !containerName.trim() ||
      !evaluateContainerName.trim() ||
      !grpoContainerName.trim() ||
      !multinodeContainerName.trim()
    ) return;
    setStatusMessage("");
    try {
      await createGroup.mutateAsync({
        name: groupName.trim(),
        containerName: containerName.trim(),
        evaluateContainerName: evaluateContainerName.trim(),
        grpoContainerName: grpoContainerName.trim(),
        multinodeContainerName: multinodeContainerName.trim(),
      });
      setGroupName("");
      setContainerName("");
      setEvaluateContainerName("");
      setGrpoContainerName("");
      setMultinodeContainerName("");
      setStatusMessage(t("resourceAccess.groupCreated"));
      await refresh();
    } catch (error) {
      setStatusMessage(translateAuthError(error, t, "auth.adminActionFailed"));
    }
  };

  const saveTrainingContainer = async (groupId: string, currentContainerName: string) => {
    const nextContainer = (containerDrafts[groupId] ?? currentContainerName).trim();
    if (!nextContainer || nextContainer === currentContainerName) return;
    setSavingContainerKey(`${groupId}:training`);
    setStatusMessage("");
    try {
      await setGroupContainer.mutateAsync({
        groupId,
        containerName: nextContainer,
      });
      setContainerDrafts((current) => {
        const next = { ...current };
        delete next[groupId];
        return next;
      });
      setStatusMessage(t("resourceAccess.groupDockerUpdated"));
      await refresh();
    } catch (error) {
      setStatusMessage(translateAuthError(error, t, "auth.adminActionFailed"));
    } finally {
      setSavingContainerKey(null);
    }
  };

  const saveEvaluationContainer = async (groupId: string, currentContainerName: string) => {
    const nextContainer = (evaluateContainerDrafts[groupId] ?? currentContainerName).trim();
    if (!nextContainer || nextContainer === currentContainerName) return;
    setSavingContainerKey(`${groupId}:evaluation`);
    setStatusMessage("");
    try {
      await setGroupEvaluateContainer.mutateAsync({
        groupId,
        containerName: nextContainer,
      });
      setEvaluateContainerDrafts((current) => {
        const next = { ...current };
        delete next[groupId];
        return next;
      });
      setStatusMessage(t("resourceAccess.groupEvaluationDockerUpdated"));
      await refresh();
    } catch (error) {
      setStatusMessage(translateAuthError(error, t, "auth.adminActionFailed"));
    } finally {
      setSavingContainerKey(null);
    }
  };

  const saveGrpoContainer = async (groupId: string, currentContainerName: string) => {
    const nextContainer = (grpoContainerDrafts[groupId] ?? currentContainerName).trim();
    if (!nextContainer || nextContainer === currentContainerName) return;
    setSavingContainerKey(`${groupId}:grpo`);
    setStatusMessage("");
    try {
      await setGroupGrpoContainer.mutateAsync({
        groupId,
        containerName: nextContainer,
      });
      setGrpoContainerDrafts((current) => {
        const next = { ...current };
        delete next[groupId];
        return next;
      });
      setStatusMessage(t("resourceAccess.groupGrpoDockerUpdated"));
      await refresh();
    } catch (error) {
      setStatusMessage(translateAuthError(error, t, "auth.adminActionFailed"));
    } finally {
      setSavingContainerKey(null);
    }
  };

  const saveMultinodeContainer = async (groupId: string, currentContainerName: string) => {
    const nextContainer = (multinodeContainerDrafts[groupId] ?? currentContainerName).trim();
    if (!nextContainer || nextContainer === currentContainerName) return;
    setSavingContainerKey(`${groupId}:multinode`);
    setStatusMessage("");
    try {
      await setGroupMultinodeContainer.mutateAsync({
        groupId,
        containerName: nextContainer,
      });
      setMultinodeContainerDrafts((current) => {
        const next = { ...current };
        delete next[groupId];
        return next;
      });
      setStatusMessage(t("resourceAccess.groupMultinodeDockerUpdated"));
      await refresh();
    } catch (error) {
      setStatusMessage(translateAuthError(error, t, "auth.adminActionFailed"));
    } finally {
      setSavingContainerKey(null);
    }
  };

  const deleteEmptyGroup = async (
    group: { id: string; name: string },
  ) => {
    const groupNameText = getGroupDisplayName(group);
    if (
      !window.confirm(
        t("resourceAccess.deleteGroupConfirm", { name: groupNameText }),
      )
    ) {
      return;
    }
    setResourceActionKey(`${group.id}:delete`);
    setStatusMessage("");
    try {
      await deleteGroup.mutateAsync({ groupId: group.id });
      setStatusMessage(t("resourceAccess.groupDeleted"));
      await refresh();
    } catch (error) {
      setStatusMessage(translateAuthError(error, t, "auth.adminActionFailed"));
    } finally {
      setResourceActionKey(null);
    }
  };

  const updateAssignedNode = async (groupId: string, nodeId: string | null) => {
    setResourceActionKey(`${groupId}:node`);
    setStatusMessage("");
    try {
      await setNode.mutateAsync({ groupId, nodeId });
      setStatusMessage(t("resourceAccess.groupNodeUpdated"));
      await refresh();
    } catch (error) {
      setStatusMessage(translateAuthError(error, t, "auth.adminActionFailed"));
    } finally {
      setResourceActionKey(null);
    }
  };

  const revalidateDocker = async (groupId: string) => {
    setValidatingGroupId(groupId);
    setStatusMessage("");
    try {
      const response = await validateContainers.mutateAsync({ groupId });
      await refresh();
      const result = response.data;
      const failed =
        result?.trainingContainerStatus === "failed" ||
        result?.evaluationContainerStatus === "failed" ||
        result?.grpoContainerStatus === "failed" ||
        result?.multinodeContainerStatus === "failed";
      if (failed) {
        const reason = [
          result?.trainingContainerError,
          result?.evaluationContainerError,
          result?.grpoContainerError,
          result?.multinodeContainerError,
        ].filter(Boolean).join("; ");
        setStatusMessage(
          reason
            ? t("resourceAccess.dockerValidationFailedWithReason", { reason })
            : t("resourceAccess.dockerValidationFailed"),
        );
      } else {
        setStatusMessage(t("resourceAccess.dockerRevalidated"));
      }
    } catch (error) {
      const reason = translateAuthError(error, t, "resourceAccess.dockerValidationFailed");
      setStatusMessage(
        reason === t("resourceAccess.dockerValidationFailed")
          ? reason
          : t("resourceAccess.dockerValidationFailedWithReason", { reason }),
      );
    } finally {
      setValidatingGroupId(null);
    }
  };

  return (
    <div className="space-y-4">
      <section className="rounded-2xl border border-border/50 bg-card/95 p-4 shadow-sm">
        <div className="mb-4 flex items-start justify-between gap-3">
          <div>
            <div className="flex items-center gap-2">
              <BoxesIcon className="size-4 text-primary" />
              <h3 className="text-base font-semibold">{t("resourceAccess.groupsTitle")}</h3>
            </div>
            <p className="mt-1 text-xs text-muted-foreground">
              {t("resourceAccess.groupsDescription")}
            </p>
          </div>
        </div>

        <div className="rounded-xl border border-border/40 bg-muted/20 p-3">
          <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-[1fr_1fr_1fr_1fr_1fr_auto]">
            <label className="grid gap-1 text-xs font-medium text-muted-foreground">
              {t("resourceAccess.groupNameLabel")}
              <Input
                value={groupName}
                onChange={(event) => setGroupName(event.target.value)}
                placeholder={t("resourceAccess.groupNamePlaceholder")}
                className="placeholder:text-muted-foreground/50"
              />
            </label>
            <label className="grid gap-1 text-xs font-medium text-muted-foreground">
              {t("resourceAccess.trainingDockerLabel")}
              <Input
                value={containerName}
                onChange={(event) => setContainerName(event.target.value)}
                placeholder={t("resourceAccess.defaultContainerPlaceholder")}
                className="placeholder:text-muted-foreground/50"
              />
            </label>
            <label className="grid gap-1 text-xs font-medium text-muted-foreground">
              {t("resourceAccess.evaluationDockerLabel")}
              <Input
                value={evaluateContainerName}
                onChange={(event) => setEvaluateContainerName(event.target.value)}
                placeholder={t("resourceAccess.evaluateContainerPlaceholder")}
                className="placeholder:text-muted-foreground/50"
              />
            </label>
            <label className="grid gap-1 text-xs font-medium text-muted-foreground">
              {t("resourceAccess.grpoDockerLabel")}
              <Input
                value={grpoContainerName}
                onChange={(event) => setGrpoContainerName(event.target.value)}
                placeholder={t("resourceAccess.grpoContainerPlaceholder")}
                className="placeholder:text-muted-foreground/50"
              />
            </label>
            <label className="grid gap-1 text-xs font-medium text-muted-foreground">
              {t("resourceAccess.multinodeDockerLabel")}
              <Input
                value={multinodeContainerName}
                onChange={(event) => setMultinodeContainerName(event.target.value)}
                placeholder={t("resourceAccess.multinodeContainerPlaceholder")}
                className="placeholder:text-muted-foreground/50"
              />
            </label>
            <Button
              type="button"
              onClick={createNewGroup}
              disabled={createGroup.isPending}
              className="min-w-32 self-end"
            >
              {createGroup.isPending
                ? t("resourceAccess.creatingGroup")
                : t("resourceAccess.createGroup")}
            </Button>
          </div>
        </div>
      </section>

      <section className="rounded-2xl border border-border/50 bg-card/95 p-4 shadow-sm">
        <div className="mb-4 flex items-center gap-2">
          <ServerIcon className="size-4 text-primary" />
          <h3 className="text-base font-semibold">
            {t("resourceAccess.currentGroups")}
          </h3>
        </div>

        {statusMessage && (
          <div className="mb-3 rounded-lg border border-sky-200 bg-sky-50 px-3 py-2 text-xs text-sky-700">
            {statusMessage}
          </div>
        )}

        <div className="grid gap-3">
        {groupItems.map((group) => {
          const selectedNode = nodeItems.find((node) => node.id === group.nodeId);
          const isValidating = validatingGroupId === group.id;
          const trainingDraft = containerDrafts[group.id] ?? group.defaultContainerName;
          const evaluationDraft =
            evaluateContainerDrafts[group.id] ?? group.defaultEvaluateContainerName;
          const grpoDraft = grpoContainerDrafts[group.id] ?? group.defaultGrpoContainerName;
          const multinodeDraft =
            multinodeContainerDrafts[group.id] ?? group.defaultMultinodeContainerName;
          const isSavingTraining = savingContainerKey === `${group.id}:training`;
          const isSavingEvaluation = savingContainerKey === `${group.id}:evaluation`;
          const isSavingGrpo = savingContainerKey === `${group.id}:grpo`;
          const isSavingMultinode = savingContainerKey === `${group.id}:multinode`;
          const isDeletingGroup = resourceActionKey === `${group.id}:delete`;
          const isUpdatingNode = resourceActionKey === `${group.id}:node`;
          const deleteBlockedReason = group.members.length
            ? t("auth.groupNotEmpty")
            : group.nodeId
              ? t("auth.groupNodeAssigned")
              : "";
          const trainingChanged =
            trainingDraft.trim() !== "" &&
            trainingDraft.trim() !== group.defaultContainerName;
          const evaluationChanged =
            evaluationDraft.trim() !== "" &&
            evaluationDraft.trim() !== group.defaultEvaluateContainerName;
          const grpoChanged =
            grpoDraft.trim() !== "" &&
            grpoDraft.trim() !== group.defaultGrpoContainerName;
          const multinodeChanged =
            multinodeDraft.trim() !== "" &&
            multinodeDraft.trim() !== group.defaultMultinodeContainerName;

          return (
            <div key={group.id} className="rounded-xl border border-border/45 bg-background p-4 shadow-xs">
              <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
                <div className="min-w-0">
                  <div className="flex flex-wrap items-center gap-2">
                    <div className="text-lg font-semibold leading-tight">
                      {getGroupDisplayName(group)}
                    </div>
                    <span className="rounded-full bg-muted px-2 py-0.5 text-xs text-muted-foreground">
                      {t("resourceAccess.memberCount", { count: group.members.length })}
                    </span>
                  </div>
                  {group.members.length > 0 && (
                    <div className="mt-1 truncate text-xs text-muted-foreground">
                      {group.members.map((member) => member.username).join(", ")}
                    </div>
                  )}
                </div>

                {group.id !== "default-users" && (
                  <Button
                    type="button"
                    size="sm"
                    variant="outline"
                    disabled={isDeletingGroup || Boolean(deleteBlockedReason)}
                    title={deleteBlockedReason || undefined}
                    onClick={() => deleteEmptyGroup(group)}
                  >
                    <Trash2Icon className="size-3.5" />
                    {isDeletingGroup
                      ? t("resourceAccess.deletingGroup")
                      : t("resourceAccess.deleteEmptyGroup")}
                  </Button>
                )}
              </div>

              <div className="mt-3 grid gap-3 lg:grid-cols-2 xl:grid-cols-5">
                <div className="flex min-h-[172px] flex-col rounded-lg border border-border/35 bg-muted/15 p-3">
                  <div className="mb-2 flex items-center gap-2 text-sm font-medium">
                    <ServerIcon className="size-4 text-primary" />
                    {t("resourceAccess.assignedNode")}
                  </div>
                  <select
                    className="h-9 w-full rounded-md border border-input bg-background px-3 text-sm outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
                    value={group.nodeId || ""}
                    disabled={isUpdatingNode}
                    onChange={(event) =>
                      updateAssignedNode(group.id, event.target.value || null)
                    }
                  >
                    <option value="">{t("resourceAccess.noNode")}</option>
                    {nodeItems.map((node) => (
                      <option key={node.id} value={node.id}>
                        {node.name}
                      </option>
                    ))}
                  </select>
                  <div className="mt-2 text-xs text-muted-foreground">
                    {selectedNode?.name || t("resourceAccess.noNode")}
                  </div>
                  {group.nodeId && (
                    <Button
                      type="button"
                      size="sm"
                      variant="outline"
                      disabled={isValidating}
                      onClick={() => revalidateDocker(group.id)}
                      className={`mt-auto w-full ${secondaryActionClassName}`}
                    >
                      <RefreshCwIcon className={`size-3.5 ${isValidating ? "animate-spin" : ""}`} />
                      {isValidating
                        ? t("resourceAccess.revalidatingDocker")
                        : t("resourceAccess.revalidateDocker")}
                    </Button>
                  )}
                </div>

                <div className="flex min-h-[172px] flex-col rounded-lg border border-border/35 bg-muted/15 p-3">
                  <div className="mb-2 text-sm font-medium">
                    {t("resourceAccess.trainingDockerLabel")}
                  </div>
                  <div className="mb-2 truncate text-xs text-muted-foreground">
                    {group.defaultContainerName}
                  </div>
                  <div className="mt-auto grid gap-2">
                    <Input
                      value={trainingDraft}
                      onChange={(event) =>
                        setContainerDrafts((current) => ({
                          ...current,
                          [group.id]: event.target.value,
                        }))
                      }
                      placeholder={group.defaultContainerName}
                      className="placeholder:text-muted-foreground/50"
                    />
                    <Button
                      type="button"
                      size="sm"
                      variant="outline"
                      disabled={!trainingChanged || isSavingTraining}
                      className={`w-full ${secondaryActionClassName}`}
                      onClick={() => saveTrainingContainer(group.id, group.defaultContainerName)}
                    >
                      {isSavingTraining
                        ? t("resourceAccess.savingDocker")
                        : t("resourceAccess.saveTrainingDocker")}
                    </Button>
                  </div>
                </div>

                <div className="flex min-h-[172px] flex-col rounded-lg border border-border/35 bg-muted/15 p-3">
                  <div className="mb-2 text-sm font-medium">
                    {t("resourceAccess.evaluationDockerLabel")}
                  </div>
                  <div className="mb-2 truncate text-xs text-muted-foreground">
                    {group.defaultEvaluateContainerName}
                  </div>
                  <div className="mt-auto grid gap-2">
                    <Input
                      value={evaluationDraft}
                      onChange={(event) =>
                        setEvaluateContainerDrafts((current) => ({
                          ...current,
                          [group.id]: event.target.value,
                        }))
                      }
                      placeholder={group.defaultEvaluateContainerName}
                      className="placeholder:text-muted-foreground/50"
                    />
                    <Button
                      type="button"
                      size="sm"
                      variant="outline"
                      disabled={!evaluationChanged || isSavingEvaluation}
                      className={`w-full ${secondaryActionClassName}`}
                      onClick={() =>
                        saveEvaluationContainer(
                          group.id,
                          group.defaultEvaluateContainerName,
                        )
                      }
                    >
                      {isSavingEvaluation
                        ? t("resourceAccess.savingDocker")
                        : t("resourceAccess.saveEvaluationDocker")}
                    </Button>
                  </div>
                </div>

                <div className="flex min-h-[172px] flex-col rounded-lg border border-border/35 bg-muted/15 p-3">
                  <div className="mb-2 text-sm font-medium">
                    {t("resourceAccess.grpoDockerLabel")}
                  </div>
                  <div className="mb-2 truncate text-xs text-muted-foreground">
                    {group.defaultGrpoContainerName}
                  </div>
                  <div className="mt-auto grid gap-2">
                    <Input
                      value={grpoDraft}
                      onChange={(event) =>
                        setGrpoContainerDrafts((current) => ({
                          ...current,
                          [group.id]: event.target.value,
                        }))
                      }
                      placeholder={group.defaultGrpoContainerName}
                      className="placeholder:text-muted-foreground/50"
                    />
                    <Button
                      type="button"
                      size="sm"
                      variant="outline"
                      disabled={!grpoChanged || isSavingGrpo}
                      className={`w-full ${secondaryActionClassName}`}
                      onClick={() =>
                        saveGrpoContainer(
                          group.id,
                          group.defaultGrpoContainerName,
                        )
                      }
                    >
                      {isSavingGrpo
                        ? t("resourceAccess.savingDocker")
                        : t("resourceAccess.saveGrpoDocker")}
                    </Button>
                  </div>
                </div>

                <div className="flex min-h-[172px] flex-col rounded-lg border border-border/35 bg-muted/15 p-3">
                  <div className="mb-2 text-sm font-medium">
                    {t("resourceAccess.multinodeDockerLabel")}
                  </div>
                  <div className="mb-2 truncate text-xs text-muted-foreground">
                    {group.defaultMultinodeContainerName}
                  </div>
                  <div className="mt-auto grid gap-2">
                    <Input
                      value={multinodeDraft}
                      onChange={(event) =>
                        setMultinodeContainerDrafts((current) => ({
                          ...current,
                          [group.id]: event.target.value,
                        }))
                      }
                      placeholder={group.defaultMultinodeContainerName}
                      className="placeholder:text-muted-foreground/50"
                    />
                    <Button
                      type="button"
                      size="sm"
                      variant="outline"
                      disabled={!multinodeChanged || isSavingMultinode}
                      className={`w-full ${secondaryActionClassName}`}
                      onClick={() =>
                        saveMultinodeContainer(
                          group.id,
                          group.defaultMultinodeContainerName,
                        )
                      }
                    >
                      {isSavingMultinode
                        ? t("resourceAccess.savingDocker")
                        : t("resourceAccess.saveMultinodeDocker")}
                    </Button>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
        </div>
      </section>
    </div>
  );
};

export default AdminResourcePanel;
