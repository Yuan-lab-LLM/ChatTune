import {
  DatasetInfo,
  EvaluationResult,
  ManagementBizType,
  ManagementCacheMeta,
  ManagementCacheResponse,
  MedicalTestFile,
  ModelInfo,
  RefreshTriggerType,
} from "../../../shared/src";
import { In } from "typeorm";
import { ManagementCacheTable } from "../models/ManagementCache";
import { RefreshJobTable } from "../models/RefreshJob";
import {
  remoteResourceClient,
  resourceNodeRegistry,
} from "./resourceNodeService";
import {
  ResourceCatalogTable,
  ResourceShareScopeTable,
  ResourceVisibility,
} from "../models/ResourceAccess";

type CachePayloadMap = {
  dataset: DatasetInfo;
  model: ModelInfo;
  medicalTest: MedicalTestFile;
  evaluationResult: EvaluationResult;
};

type CachePayload<T extends ManagementBizType> = CachePayloadMap[T];

const RUNNING_JOB_STALE_MS = 5 * 60 * 1000;

const createId = (...parts: string[]): string =>
  parts.map((part) => part.replace(/[^a-zA-Z0-9._-]/g, "_")).join(":");

const createJobId = (): string =>
  `job_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;

export class ManagementCacheService {
  async getDatasets(
    nodeId: string,
    containerName: string,
  ): Promise<ManagementCacheResponse<DatasetInfo>> {
    return this.getSnapshotOrBootstrap("dataset", nodeId, containerName);
  }

  async refreshDatasets(
    nodeId: string,
    containerName: string,
    triggerType: RefreshTriggerType = "manual",
  ): Promise<ManagementCacheResponse<DatasetInfo>> {
    await this.refreshCache("dataset", nodeId, containerName, triggerType);
    return this.getSnapshot("dataset", nodeId, containerName);
  }

  async getModels(
    nodeId: string,
    containerName: string,
  ): Promise<ManagementCacheResponse<ModelInfo>> {
    return this.getSnapshotOrBootstrap("model", nodeId, containerName);
  }

  async refreshModels(
    nodeId: string,
    containerName: string,
    triggerType: RefreshTriggerType = "manual",
  ): Promise<ManagementCacheResponse<ModelInfo>> {
    await this.refreshCache("model", nodeId, containerName, triggerType);
    return this.getSnapshot("model", nodeId, containerName);
  }

  async getMedicalTests(
    nodeId: string,
    containerName: string,
  ): Promise<ManagementCacheResponse<MedicalTestFile>> {
    return this.getSnapshotOrBootstrap("medicalTest", nodeId, containerName);
  }

  async refreshMedicalTests(
    nodeId: string,
    containerName: string,
    triggerType: RefreshTriggerType = "manual",
  ): Promise<ManagementCacheResponse<MedicalTestFile>> {
    await this.refreshCache("medicalTest", nodeId, containerName, triggerType);
    return this.getSnapshot("medicalTest", nodeId, containerName);
  }

  async getEvaluationResults(
    nodeId: string,
    containerName: string,
  ): Promise<ManagementCacheResponse<EvaluationResult>> {
    return this.getSnapshotOrBootstrap(
      "evaluationResult",
      nodeId,
      containerName,
    );
  }

  async refreshEvaluationResults(
    nodeId: string,
    containerName: string,
    triggerType: RefreshTriggerType = "manual",
  ): Promise<ManagementCacheResponse<EvaluationResult>> {
    await this.refreshCache(
      "evaluationResult",
      nodeId,
      containerName,
      triggerType,
    );
    return this.getSnapshot("evaluationResult", nodeId, containerName);
  }

  async getKnownContainers(bizType: ManagementBizType): Promise<string[]> {
    const cacheRows = await ManagementCacheTable.find({
      where: {
        bizType,
      },
    });
    const jobRows = await RefreshJobTable.find({
      where: {
        bizType,
      },
    });

    return [
      ...new Set([...cacheRows, ...jobRows].map((row) => row.containerName)),
    ];
  }

  async getSnapshot<T extends ManagementBizType>(
    bizType: T,
    nodeId: string,
    containerName: string,
  ): Promise<ManagementCacheResponse<CachePayload<T>>> {
    const records = await ManagementCacheTable.find({
      where: {
        bizType,
        nodeId,
        containerName,
      },
      order: {
        updatedAt: "DESC",
      },
    });

    const meta = await this.buildMeta(
      bizType,
      nodeId,
      containerName,
      records.length,
    );
    const items = this.sortPayloads(
      bizType,
      records.map((record) => record.payload as unknown as CachePayload<T>),
    );

    const node = (() => {
      try {
        return resourceNodeRegistry.get(nodeId);
      } catch {
        return undefined;
      }
    })();

    const itemsWithNode = items.map((item) => ({
      ...item,
      nodeId,
      nodeName: node?.name,
      containerName,
    }));

    return {
      items: itemsWithNode as CachePayload<T>[],
      meta,
    };
  }

  async getSnapshotOrBootstrap<T extends ManagementBizType>(
    bizType: T,
    nodeId: string,
    containerName: string,
  ): Promise<ManagementCacheResponse<CachePayload<T>>> {
    const snapshot = await this.getSnapshot(bizType, nodeId, containerName);
    if (snapshot.meta.updatedAt || snapshot.meta.isRefreshing) {
      return snapshot;
    }

    await this.refreshCache(bizType, nodeId, containerName, "bootstrap");
    return this.getSnapshot(bizType, nodeId, containerName);
  }

  async refreshCache(
    bizType: ManagementBizType,
    nodeId: string,
    containerName: string,
    triggerType: RefreshTriggerType,
  ): Promise<void> {
    let runningJob = await RefreshJobTable.findOne({
      where: {
        bizType,
        nodeId,
        containerName,
        status: "running",
      },
      order: {
        startedAt: "DESC",
      },
    });

    if (runningJob) {
      const startedAt = new Date(runningJob.startedAt).getTime();
      const isStale =
        Number.isFinite(startedAt) &&
        Date.now() - startedAt > RUNNING_JOB_STALE_MS;

      if (!isStale) {
        return;
      }

      console.warn(
        `[ManagementCacheService] Marking stale running job as failed: ${runningJob.id} (${bizType}/${containerName})`,
      );
      runningJob.status = "failed";
      runningJob.finishedAt = new Date();
      runningJob.errorMessage =
        "Marked as failed because the running refresh job exceeded the stale timeout.";
      await runningJob.save();
      runningJob = null;
    }

    const job = RefreshJobTable.create({
      id: createJobId(),
      bizType,
      nodeId,
      containerName,
      triggerType,
      status: "running",
      errorMessage: null,
    });
    await job.save();

    try {
      const items = await this.fetchPayloads(bizType, nodeId, containerName);
      await this.persistItems(bizType, nodeId, containerName, items);

      job.status = "success";
      job.finishedAt = new Date();
      job.errorMessage = null;
      await job.save();
    } catch (error) {
      job.status = "failed";
      job.finishedAt = new Date();
      job.errorMessage =
        error instanceof Error ? error.message : "Unknown refresh error";
      await job.save();
      throw error;
    }
  }

  private async buildMeta(
    bizType: ManagementBizType,
    nodeId: string,
    containerName: string,
    itemCount: number,
  ): Promise<ManagementCacheMeta> {
    const [runningJob, latestJob, latestSuccessJob] = await Promise.all([
      RefreshJobTable.findOne({
        where: {
          bizType,
          nodeId,
          containerName,
          status: "running",
        },
        order: {
          startedAt: "DESC",
        },
      }),
      RefreshJobTable.findOne({
        where: {
          bizType,
          nodeId,
          containerName,
        },
        order: {
          startedAt: "DESC",
        },
      }),
      RefreshJobTable.findOne({
        where: {
          bizType,
          nodeId,
          containerName,
          status: "success",
        },
        order: {
          finishedAt: "DESC",
        },
      }),
    ]);

    const node = (() => {
      try {
        return resourceNodeRegistry.get(nodeId);
      } catch {
        return undefined;
      }
    })();

    return {
      bizType,
      nodeId,
      nodeName: node?.name,
      containerName,
      itemCount,
      updatedAt: latestSuccessJob?.finishedAt?.toISOString() ?? null,
      lastRefreshStartedAt: latestJob?.startedAt?.toISOString() ?? null,
      lastRefreshFinishedAt: latestJob?.finishedAt?.toISOString() ?? null,
      lastRefreshStatus: runningJob ? "running" : (latestJob?.status ?? "idle"),
      isRefreshing: Boolean(runningJob),
      lastErrorMessage:
        latestJob?.status === "failed"
          ? (latestJob.errorMessage ?? null)
          : null,
    };
  }

  private async fetchPayloads<T extends ManagementBizType>(
    bizType: T,
    nodeId: string,
    containerName: string,
  ): Promise<CachePayload<T>[]> {
    switch (bizType) {
      case "dataset": {
        const response = await remoteResourceClient.request<{
          data: Record<string, any[]>;
        }>(nodeId, "datasets", undefined, { container: containerName });
        const datasets = response.data;
        return this.flattenDatasets(datasets) as CachePayload<T>[];
      }
      case "model": {
        const response = await remoteResourceClient.request<{
          data: Record<string, any[]>;
        }>(nodeId, "models", undefined, { container: containerName });
        const models = response.data;
        return this.flattenModels(models) as CachePayload<T>[];
      }
      case "medicalTest": {
        const response = await remoteResourceClient.request<{
          data: MedicalTestFile[];
        }>(nodeId, "medical-tests", undefined, { container: containerName });
        return response.data as CachePayload<T>[];
      }
      case "evaluationResult": {
        const response = await remoteResourceClient.request<{
          data: EvaluationResult[];
        }>(nodeId, "evaluation-results", undefined, {
          container: containerName,
        });
        return response.data as CachePayload<T>[];
      }
      default:
        return [];
    }
  }

  private async persistItems<T extends ManagementBizType>(
    bizType: T,
    nodeId: string,
    containerName: string,
    items: CachePayload<T>[],
  ): Promise<void> {
    const records = items.map((item) => {
      const itemKey = this.getItemKey(bizType, item);
      return ManagementCacheTable.create({
        id: createId(bizType, nodeId, containerName, itemKey),
        bizType,
        nodeId,
        containerName,
        itemKey,
        sourcePath: this.getSourcePath(bizType, item),
        payload: item as unknown as Record<string, unknown>,
      });
    });

    if (records.length > 0) {
      await ManagementCacheTable.save(records);
      for (const record of records) {
        const existing = await ResourceCatalogTable.findOne({
          where: {
            bizType,
            nodeId,
            containerName,
            itemKey: record.itemKey,
          },
        });
        if (!existing) {
          const privateOwnerMatch = record.sourcePath?.match(
            /\/medflow\/users\/([^/]+)\//,
          );
          const created = await ResourceCatalogTable.create({
            id: record.id,
            bizType,
            nodeId,
            containerName,
            itemKey: record.itemKey,
            visibility: privateOwnerMatch
              ? ResourceVisibility.PRIVATE
              : ResourceVisibility.PUBLIC,
            ownerUserId: privateOwnerMatch?.[1] || null,
            groupId: null,
            sourcePath: record.sourcePath,
            payload: record.payload,
          }).save();
          if (created.sourcePath) {
            const sharedReplicas = await ResourceCatalogTable.createQueryBuilder("resource")
              .where("resource.bizType = :bizType", { bizType })
              .andWhere("resource.sourcePath = :sourcePath", { sourcePath: created.sourcePath })
              .andWhere("resource.id != :id", { id: created.id })
              .getMany();
            if (sharedReplicas.length) {
              const inheritedScopes = await ResourceShareScopeTable.find({
                where: { resourceId: In(sharedReplicas.map((replica) => replica.id)) },
              });
              if (inheritedScopes.length) {
                created.visibility = ResourceVisibility.PUBLIC;
                created.groupId = null;
                await created.save();
                const uniqueScopes = [...new Map(inheritedScopes.map((scope) => [scope.scopeKey, scope])).values()];
                await ResourceShareScopeTable.save(uniqueScopes.map((scope) =>
                  ResourceShareScopeTable.create({
                    id: createId(created.id, "share", scope.scopeKey),
                    resourceId: created.id,
                    scopeKey: scope.scopeKey,
                    groupId: scope.groupId || null,
                  })));
              }
            }
          }
        } else {
          existing.sourcePath = record.sourcePath;
          existing.payload = record.payload;
          await existing.save();
        }
      }
    }

    const currentIds = records.map((record) => record.id);
    const deleteQuery = ManagementCacheTable.createQueryBuilder()
      .delete()
      .where("bizType = :bizType", { bizType })
      .andWhere("nodeId = :nodeId", { nodeId })
      .andWhere("containerName = :containerName", { containerName });

    if (currentIds.length > 0) {
      await deleteQuery
        .andWhere("id NOT IN (:...currentIds)", { currentIds })
        .execute();
      return;
    }

    await deleteQuery.execute();
  }

  private flattenDatasets(backendData: Record<string, any[]>): DatasetInfo[] {
    const result: DatasetInfo[] = [];

    for (const [type, datasets] of Object.entries(backendData)) {
      for (const dataset of datasets) {
        const firstPreview = dataset.filePreviews?.[0];
        result.push({
          name: dataset.name,
          type: dataset.type || type,
          path: dataset.path,
          description: dataset.description,
          files: dataset.files,
          filePreviews: dataset.filePreviews,
          sampleContent: firstPreview?.preview,
          fileName: firstPreview?.filename,
          size: dataset.size,
          createdAt: dataset.createdAt,
        });
      }
    }

    return result;
  }

  private flattenModels(backendData: Record<string, any[]>): ModelInfo[] {
    const result: ModelInfo[] = [];

    for (const [type, models] of Object.entries(backendData)) {
      for (const model of models) {
        result.push({
          name: model.name,
          type: model.type || type,
          path: model.path,
          size: model.size,
          createdAt: model.createdAt,
          version: model.version,
          description: model.description,
          checkpoints: model.checkpoints,
          ...(typeof model.merged === "boolean"
            ? { merged: model.merged }
            : {}),
        } as ModelInfo);
      }
    }

    return result;
  }

  private getItemKey<T extends ManagementBizType>(
    bizType: T,
    item: CachePayload<T>,
  ): string {
    switch (bizType) {
      case "dataset":
        return `${(item as DatasetInfo).type}:${(item as DatasetInfo).name}`;
      case "model":
        return `${(item as ModelInfo).type || "model"}:${(item as ModelInfo).name}`;
      case "medicalTest":
        return `${(item as MedicalTestFile).type}:${(item as MedicalTestFile).filename}`;
      case "evaluationResult":
        return (item as EvaluationResult).jobId;
      default:
        return JSON.stringify(item);
    }
  }

  private getSourcePath<T extends ManagementBizType>(
    bizType: T,
    item: CachePayload<T>,
  ): string | undefined {
    switch (bizType) {
      case "dataset":
        return `${(item as DatasetInfo).path || ""}/${(item as DatasetInfo).name}`;
      case "model":
        return `${(item as ModelInfo).path || ""}/${(item as ModelInfo).name}`;
      case "evaluationResult":
        return (item as EvaluationResult).folderPath;
      default:
        return undefined;
    }
  }

  private sortPayloads<T extends ManagementBizType>(
    bizType: T,
    items: CachePayload<T>[],
  ): CachePayload<T>[] {
    const sorted = [...items];

    switch (bizType) {
      case "evaluationResult":
        return sorted.sort(
          (a, b) =>
            new Date((b as EvaluationResult).startTime || 0).getTime() -
            new Date((a as EvaluationResult).startTime || 0).getTime(),
        );
      case "dataset":
      case "model":
        return sorted.sort(
          (a, b) =>
            new Date((b as DatasetInfo | ModelInfo).createdAt || 0).getTime() -
            new Date((a as DatasetInfo | ModelInfo).createdAt || 0).getTime(),
        );
      default:
        return sorted;
    }
  }
}

export const managementCacheService = new ManagementCacheService();
