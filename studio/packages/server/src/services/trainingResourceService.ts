import crypto from 'crypto';
import { In } from 'typeorm';
import { SafeAuthUser } from '../dao/Auth';
import { UserRole } from '../models/Auth';
import { GroupNodeAssignmentTable, UserGroupMemberTable, UserGroupTable } from '../models/ResourceAccess';
import {
    GroupResourceQuotaTable,
    ResourcePoolNodeTable,
    ResourcePoolTable,
    TrainingReservationNodeTable,
    TrainingReservationStatus,
    TrainingReservationTable,
    TrainingResourceLockTable,
} from '../models/TrainingResource';
import { remoteResourceClient, resourceNodeRegistry } from './resourceNodeService';
import { ResourceAccessService } from './resourceAccessService';

type NodeGpu = { index: number; available?: boolean; memoryUsed?: number };
type PoolRuntimeNodeSummary = {
    nodeId: string;
    status: 'online' | 'offline';
    gpuCount: number;
    availableGpuCount: number;
    reservedGpuCount: number;
    errorMessage?: string;
};
type AllocationNode = {
    nodeId: string;
    sshAlias: string;
    trainAddress: string;
    ncclSocketIfname?: string | null;
    gpuIndexes: number[];
    isMaster: boolean;
};
type PoolNodeInput = {
    nodeId: string;
    sshAlias: string;
    trainAddress: string;
    ncclSocketIfname?: string;
    allowedGpuIndexes?: number[];
    enabled?: boolean;
};
type RunnablePoolSummary = {
    id: string;
    name: string;
    description?: string | null;
    nodeIds: string[];
    nodeCount: number;
    gpusPerNode: number;
    totalGpuCount: number;
    maxGpuCount: number;
    maxConcurrentJobs: number;
    maxNodesPerJob: number;
};
type TrainingResourcePreflightResult = {
    ok: true;
    groupId: string;
    poolId: string;
    nodeCount: number;
    gpusPerNode: number;
    availableGpuCount: number;
    diagnostics: string;
};
type RuntimeGpuSnapshot = {
    gpus?: NodeGpu[];
    collectedAt?: string;
    ageSeconds?: number;
    maxAgeSeconds?: number;
};
type TrainingResourceReservationInput = {
    poolId: string;
    nodeCount: number;
    gpusPerNode: number;
    runtimeNodeId?: string;
    runtimeGpuSnapshot?: RuntimeGpuSnapshot;
    taskCategory?: string | null;
    taskType?: string | null;
    taskTypeText?: string | null;
    dryRun?: false;
};
type TrainingResourcePreflightInput = Omit<TrainingResourceReservationInput, 'dryRun'> & {
    dryRun: true;
};
type TrainingResourceAllocation = {
    reservationId: string;
    groupId: string;
    poolId: string;
    masterNodeId: string;
    masterAddr: string;
    masterPort: number;
    expiresAt: string;
    nodes: AllocationNode[];
    allocationFile?: string;
    allocationContainer?: string;
};
type TrainingResourceReleaseOptions = {
    force?: boolean;
    reason?: string | null;
    releaseResult?: string;
    source?: 'admin' | 'runtime' | 'system';
    actorUserId?: string | null;
};
type RuntimeStopProcessResult = {
    reservationId: string;
    container?: string;
    pid?: string;
    alreadyExited?: boolean;
    stopped?: boolean;
    remainingPids?: string[];
    remainingNonGpuPids?: string[];
    gpuIdle?: boolean;
    message?: string;
};
type InferenceResourceContext = {
    reservation_id: string;
    resource_group_id: string;
    training_pool_id: string;
    runtime_node_id: string;
    assigned_gpus: string[];
    cuda_visible_devices: string;
    tensor_parallel_size: number;
    gpus_per_node: number;
    expires_at: string;
    nodes: Array<{
        runtime_node_id: string;
        assigned_gpus: string[];
        cuda_visible_devices: string;
        tensor_parallel_size: number;
        gpus_per_node: number;
        is_master: boolean;
    }>;
};

type RuntimeStopInferenceServiceResult = {
    reservationId: string;
    container?: string | null;
    stopped?: boolean;
    releaseReady?: boolean;
    message?: string;
    inferenceResponse?: unknown;
};

const trainingReservationTtlSeconds = () =>
    Math.max(60, Number(process.env.MEDFLOW_TRAINING_RESERVATION_TTL_SECONDS || 300));
const inferenceReservationTtlSeconds = () =>
    Math.max(300, Number(process.env.MEDFLOW_INFERENCE_RESERVATION_TTL_SECONDS || 86400));
const resourceLockTtlSeconds = () =>
    Math.max(30, Number(process.env.MEDFLOW_TRAINING_RESOURCE_LOCK_TTL_SECONDS || 300));
const gpuSnapshotRequestTimeoutMs = () =>
    Math.max(1000, Number(process.env.MEDFLOW_RESOURCE_SNAPSHOT_TIMEOUT_MS || 15000));
const gpuBusyMemoryThresholdMb = () =>
    Math.max(0, Number(process.env.MEDFLOW_RESOURCE_GPU_BUSY_MEMORY_MB || 200));
const activeReservationStatuses = [
    TrainingReservationStatus.PREPARING,
    TrainingReservationStatus.RESERVED,
    TrainingReservationStatus.RUNNING,
];
const isInferenceReservation = (reservation: Pick<TrainingReservationTable, 'taskCategory' | 'taskType' | 'taskTypeText'>) => {
    const category = String(reservation.taskCategory || '').trim().toLowerCase();
    const type = String(reservation.taskType || reservation.taskTypeText || '').trim().toLowerCase();
    return category === 'inference' || type === 'inference';
};
const reservationTtlSeconds = (reservation?: Pick<TrainingReservationTable, 'taskCategory' | 'taskType' | 'taskTypeText'>) =>
    reservation && isInferenceReservation(reservation)
        ? inferenceReservationTtlSeconds()
        : trainingReservationTtlSeconds();
const reservationExpiredReason = (reservation: Pick<TrainingReservationTable, 'taskCategory' | 'taskType' | 'taskTypeText'>) =>
    isInferenceReservation(reservation) ? '推理资源租约已过期' : '训练资源租约已过期';
const reservationAuditDetails = (reservation: TrainingReservationTable, extra?: Record<string, unknown>) => ({
    groupId: reservation.groupId,
    poolId: reservation.poolId,
    homeNodeId: reservation.homeNodeId,
    status: reservation.status,
    taskCategory: reservation.taskCategory || null,
    taskType: reservation.taskType || reservation.taskTypeText || null,
    expiresAt: reservation.expiresAt,
    ...(extra || {}),
});
const recordReservationAuditEvent = async (
    eventType: string,
    reservation: TrainingReservationTable,
    actorUserId?: string | null,
    details?: Record<string, unknown>,
) => ResourceAccessService.recordAuditEvent(
    eventType,
    reservation.id,
    actorUserId || null,
    reservationAuditDetails(reservation, details),
);
const normalizeAllowedGpuIndexes = (gpuIndexes?: number[]) => {
    if (!gpuIndexes?.length) return null;
    if (gpuIndexes.some((index) => !Number.isInteger(index) || index < 0)) {
        throw new Error('允许 GPU 卡号必须是非负整数');
    }
    if (new Set(gpuIndexes).size !== gpuIndexes.length) {
        throw new Error('允许 GPU 卡号不能重复');
    }
    return [...gpuIndexes].sort((left, right) => left - right);
};
const gpusAllowedForNode = (node: ResourcePoolNodeTable, gpus: NodeGpu[]) => {
    if (!node.allowedGpuIndexes?.length) return gpus;
    const allowed = new Set(node.allowedGpuIndexes);
    return gpus.filter((gpu) => allowed.has(gpu.index));
};
const configuredGpuIndexesForNode = (node: ResourcePoolNodeTable) => {
    if (!node.allowedGpuIndexes?.length) {
        throw new Error(`资源池节点 ${node.nodeId} 未配置允许 GPU 卡号，无法进行训练分配`);
    }
    return [...node.allowedGpuIndexes].sort((left, right) => left - right);
};
const configuredOrSnapshotGpuIndexesForNode = (
    node: ResourcePoolNodeTable,
    gpus: NodeGpu[],
) => {
    if (node.allowedGpuIndexes?.length) {
        return [...node.allowedGpuIndexes].sort((left, right) => left - right);
    }
    return gpus.map((gpu) => gpu.index).sort((left, right) => left - right);
};
const formatGpuIndexes = (indexes: number[]) => indexes.length ? indexes.join(',') : '无';
const formatGpuDiagnostics = (items: Array<{
    node: ResourcePoolNodeTable;
    gpuIndexes: number[];
    reservedGpuIndexes: number[];
    busyGpuIndexes: number[];
    missingGpuIndexes: number[];
}>) => items.map((item) =>
    `${resourceNodeRegistry.get(item.node.nodeId).name || item.node.nodeId}: 可用GPU[${formatGpuIndexes(item.gpuIndexes)}]` +
    `, 已占用/不可用GPU[${formatGpuIndexes(
        [...new Set([...item.reservedGpuIndexes, ...item.busyGpuIndexes])]
            .sort((left, right) => left - right),
    )}]`,
).join('；');
const runtimeSnapshotGpus = (nodeId: string, snapshot?: RuntimeGpuSnapshot) => {
    if (!Array.isArray(snapshot?.gpus)) {
        throw new Error(`归属 Runtime 节点 ${nodeId} GPU 快照不可用，请稍后重试或检查节点探测服务`);
    }
    const ageSeconds = Number(snapshot.ageSeconds ?? 0);
    const maxAgeSeconds = Number(snapshot.maxAgeSeconds ?? 0);
    if (maxAgeSeconds > 0 && ageSeconds > maxAgeSeconds) {
        throw new Error(
            `归属 Runtime 节点 ${nodeId} GPU 快照已过期：age=${ageSeconds}s max=${maxAgeSeconds}s，请等待本地缓存刷新`,
        );
    }
    return snapshot.gpus;
};
const uniquePoolNodeGpus = async (nodes: ResourcePoolNodeTable[]) => {
    const byNodeId = new Map<string, Promise<NodeGpu[]>>();
    for (const node of nodes) {
        if (!byNodeId.has(node.nodeId)) {
            byNodeId.set(
                node.nodeId,
                remoteResourceClient
                    .request<{ data: NodeGpu[] | { gpus?: NodeGpu[] } }>(
                        node.nodeId,
                        'gpus/snapshot',
                        undefined,
                        undefined,
                        gpuSnapshotRequestTimeoutMs(),
                    )
                    .then((response) => {
                        if (Array.isArray(response.data)) return response.data;
                        if (Array.isArray(response.data?.gpus)) return response.data.gpus;
                        throw new Error('GPU snapshot response is invalid');
                    })
                    .catch((error) => {
                        const message = error instanceof Error ? error.message : String(error);
                        throw new Error(
                            `资源节点 ${node.nodeId} GPU 状态不可用，请稍后重试或检查节点探测服务：${message}`,
                        );
                    }),
            );
        }
    }
    return Promise.all(nodes.map(async (node) => ({
        node,
        gpus: await byNodeId.get(node.nodeId)!,
    })));
};

export class TrainingResourceService {
    private static readonly reservationLocks = new Map<string, Promise<void>>();

    private static async withReservationLock<T>(key: string, action: () => Promise<T>): Promise<T> {
        const previous = this.reservationLocks.get(key) || Promise.resolve();
        let release = () => {};
        const current = new Promise<void>((resolve) => {
            release = resolve;
        });
        const queued = previous.then(() => current);
        this.reservationLocks.set(key, queued);
        await previous;
        const lockOwnerId = crypto.randomUUID();
        let dbLockAcquired = false;
        try {
            await this.acquireDatabaseLock(key, lockOwnerId);
            dbLockAcquired = true;
            return await action();
        } finally {
            if (dbLockAcquired) {
                await this.releaseDatabaseLock(key, lockOwnerId);
            }
            release();
            if (this.reservationLocks.get(key) === queued) this.reservationLocks.delete(key);
        }
    }

    private static async acquireDatabaseLock(lockKey: string, ownerId: string) {
        const retryMs = 100;
        const timeoutAt = Date.now() + resourceLockTtlSeconds() * 1000;
        while (Date.now() < timeoutAt) {
            const expiresAt = new Date(Date.now() + resourceLockTtlSeconds() * 1000).toISOString();
            try {
                await TrainingResourceLockTable.create({ lockKey, ownerId, expiresAt }).save();
                return;
            } catch {
                const current = await TrainingResourceLockTable.findOne({ where: { lockKey } });
                if (!current || new Date(current.expiresAt).getTime() <= Date.now()) {
                    await TrainingResourceLockTable.delete({ lockKey });
                    continue;
                }
                await new Promise((resolve) => setTimeout(resolve, retryMs));
            }
        }
        throw new Error(`训练资源锁等待超时：${lockKey}，请稍后重试`);
    }

    private static async releaseDatabaseLock(lockKey: string, ownerId: string) {
        try {
            await TrainingResourceLockTable.delete({ lockKey, ownerId });
        } catch (error) {
            console.warn('Failed to release training resource lock:', error);
        }
    }

    private static async assignedGpuBusyByNode(reservationId: string) {
        const reservationNodes = await TrainingReservationNodeTable.find({ where: { reservationId } });
        const busyByNode: Array<{ nodeId: string; gpuIndexes: number[] }> = [];
        await Promise.all(reservationNodes.map(async (node) => {
            if (!node.gpuIndexes.length) return;
            try {
                const response = await remoteResourceClient.request<{ data: NodeGpu[] | { gpus?: NodeGpu[] } }>(
                    node.nodeId,
                    'gpus/snapshot',
                    undefined,
                    undefined,
                    gpuSnapshotRequestTimeoutMs(),
                );
                const gpus = Array.isArray(response.data) ? response.data : response.data?.gpus;
                if (!Array.isArray(gpus)) return;
                const snapshotByIndex = new Map(gpus.map((gpu) => [gpu.index, gpu]));
                const busyGpuIndexes = node.gpuIndexes.filter((index) => {
                    const gpu = snapshotByIndex.get(index);
                    return gpu && (
                        gpu.available === false ||
                        Number(gpu.memoryUsed || 0) >= gpuBusyMemoryThresholdMb()
                    );
                });
                if (busyGpuIndexes.length) {
                    busyByNode.push({ nodeId: node.nodeId, gpuIndexes: busyGpuIndexes });
                }
            } catch (error) {
                console.warn(`[TrainingResource] Failed to inspect expired reservation GPUs: ${reservationId}/${node.nodeId}`, error);
            }
        }));
        return busyByNode;
    }

    private static async activeReservations(where: { groupId?: string; poolId?: string }) {
        const candidates = await TrainingReservationTable.find({
            where: { ...where, status: In(activeReservationStatuses) },
        });
        const now = Date.now();
        const active: TrainingReservationTable[] = [];
        await Promise.all(candidates.map(async (item) => {
            if (new Date(item.expiresAt).getTime() > now) {
                active.push(item);
                return;
            }
            const busyByNode = await this.assignedGpuBusyByNode(item.id);
            if (busyByNode.length) {
                const busyDetails = busyByNode.map((node) => `${node.nodeId}[${node.gpuIndexes.join(',')}]`).join(' · ');
                item.errorMessage = `${reservationExpiredReason(item)}，但已分配 GPU 仍有占用：${busyDetails}`;
                item.expiredReason = item.errorMessage;
                await item.save();
                active.push(item);
                return;
            }
            const reason = reservationExpiredReason(item);
            item.status = TrainingReservationStatus.FAILED;
            item.errorMessage = reason;
            item.expiredReason = reason;
            await item.save();
            await this.releaseReservationNodes(item.id);
            await recordReservationAuditEvent('training_reservation_expired', item, null, {
                reason,
                releasedGpuAllocation: true,
            });
        }));
        return active;
    }

    private static async assertNoActiveReservations(where: { groupId?: string; poolId?: string }) {
        if ((await this.activeReservations(where)).length) {
            throw new Error('存在活跃训练预约，当前配置暂不可修改');
        }
    }

    static async listPools() {
        const [pools, nodes, quotas] = await Promise.all([
            ResourcePoolTable.find({ order: { name: 'ASC' } }),
            ResourcePoolNodeTable.find(),
            GroupResourceQuotaTable.find(),
        ]);
        const activeReservations = await this.activeReservations({});
        const activeReservationNodes = activeReservations.length
            ? await TrainingReservationNodeTable.find({
                where: { reservationId: In(activeReservations.map((item) => item.id)) },
            })
            : [];
        const activeByPool = new Map(activeReservations.map((item) => [item.id, item.poolId]));
        return pools.map((pool) => ({
            ...pool,
            nodes: nodes.filter((node) => node.poolId === pool.id),
            quotas: quotas.filter((quota) => quota.poolId === pool.id),
            summary: {
                guaranteedGpuCount: quotas
                    .filter((quota) => quota.poolId === pool.id)
                    .reduce((sum, quota) => sum + quota.guaranteedGpuCount, 0),
                activeReservationCount: activeReservations.filter((item) => item.poolId === pool.id).length,
                reservedGpuCount: activeReservationNodes
                    .filter((node) => activeByPool.get(node.reservationId) === pool.id)
                    .reduce((sum, node) => sum + node.gpuIndexes.length, 0),
                capacityGpuCount: 0,
                availableGpuCount: 0,
                offlineNodeIds: [] as string[],
                nodes: [] as PoolRuntimeNodeSummary[],
            },
        }));
    }

    static async listPoolsWithRuntimeSummary() {
        const pools = await this.listPools();
        const activeReservations = await this.activeReservations({});
        const activeReservationNodes = activeReservations.length
            ? await TrainingReservationNodeTable.find({
                where: { reservationId: In(activeReservations.map((item) => item.id)) },
            })
            : [];
        const globallyReservedByNode = new Map<string, Set<number>>();
        for (const node of activeReservationNodes) {
            const reserved = globallyReservedByNode.get(node.nodeId) || new Set<number>();
            node.gpuIndexes.forEach((index) => reserved.add(index));
            globallyReservedByNode.set(node.nodeId, reserved);
        }
        return Promise.all(pools.map(async (pool) => {
            const enabledNodes = pool.nodes.filter((node) => node.enabled);
            const nodeSummaries = await Promise.all(enabledNodes.map(async (node) => {
                try {
                    const response = await remoteResourceClient.request<{ data: NodeGpu[] | { gpus?: NodeGpu[] } }>(
                        node.nodeId,
                        'gpus/snapshot',
                        undefined,
                        undefined,
                        gpuSnapshotRequestTimeoutMs(),
                    );
                    const gpus = Array.isArray(response.data) ? response.data : response.data?.gpus || [];
                    const poolGpus = gpusAllowedForNode(node, gpus);
                    const globallyReserved = globallyReservedByNode.get(node.nodeId);
                    const gpuCount = poolGpus.length;
                    const availableGpuCount = poolGpus
                        .filter((gpu) => gpu.available !== false && !globallyReserved?.has(gpu.index))
                        .length;
                    return {
                        nodeId: node.nodeId,
                        status: 'online' as const,
                        gpuCount,
                        availableGpuCount,
                        reservedGpuCount: Math.max(0, gpuCount - availableGpuCount),
                    };
                } catch (error) {
                    return {
                        nodeId: node.nodeId,
                        status: 'offline' as const,
                        gpuCount: 0,
                        availableGpuCount: 0,
                        reservedGpuCount: 0,
                        errorMessage: error instanceof Error ? error.message : String(error),
                    };
                }
            }));
            return {
                ...pool,
                summary: {
                    ...pool.summary,
                    capacityGpuCount: nodeSummaries.reduce((sum, node) => sum + node.gpuCount, 0),
                    availableGpuCount: nodeSummaries.reduce((sum, node) => sum + node.availableGpuCount, 0),
                    offlineNodeIds: nodeSummaries
                        .filter((node) => node.status === 'offline')
                        .map((node) => node.nodeId),
                    nodes: nodeSummaries,
                },
            };
        }));
    }

    static async listRunnablePoolsForUser(user: SafeAuthUser, requestedGroupId?: string | null) {
        const groupId = await this.groupIdForUser(user, requestedGroupId || undefined);
        return this.listRunnablePoolsForGroup(groupId);
    }

    static async listRunnablePoolsForGroup(groupId: string): Promise<RunnablePoolSummary[]> {
        const quotas = await GroupResourceQuotaTable.find({ where: { groupId } });
        if (!quotas.length) return [];
        const poolIds = quotas.map((quota) => quota.poolId);
        const [pools, nodes] = await Promise.all([
            ResourcePoolTable.find({ where: { id: In(poolIds), enabled: true }, order: { name: 'ASC' } }),
            ResourcePoolNodeTable.find({ where: { poolId: In(poolIds), enabled: true } }),
        ]);
        const poolsById = new Map(pools.map((pool) => [pool.id, pool]));
        const runnablePools = await Promise.all(quotas
            .map(async (quota): Promise<RunnablePoolSummary | null> => {
                const pool = poolsById.get(quota.poolId);
                if (!pool) return null;
                const poolNodes = nodes
                    .filter((node) => node.poolId === pool.id)
                    .sort((left, right) => left.nodeId.localeCompare(right.nodeId));
                const gpuCounts = await Promise.all(poolNodes.map(async (node) => {
                    if (node.allowedGpuIndexes?.length) return node.allowedGpuIndexes.length;
                    try {
                        const [{ gpus }] = await uniquePoolNodeGpus([node]);
                        return gpusAllowedForNode(node, gpus).length;
                    } catch {
                        return 0;
                    }
                }));
                return {
                    id: pool.id,
                    name: pool.name,
                    description: pool.description,
                    nodeIds: poolNodes.map((node) => node.nodeId),
                    nodeCount: poolNodes.length,
                    gpusPerNode: gpuCounts.length ? Math.min(...gpuCounts) : 0,
                    totalGpuCount: gpuCounts.reduce((sum, count) => sum + count, 0),
                    maxGpuCount: quota.maxGpuCount,
                    maxConcurrentJobs: quota.maxConcurrentJobs,
                    maxNodesPerJob: quota.maxNodesPerJob,
                };
            }));
        return runnablePools
            .filter((pool): pool is RunnablePoolSummary => Boolean(pool))
            .sort((left, right) => left.name.localeCompare(right.name));
    }

    static async listReservations() {
        await this.activeReservations({});
        const [reservations, reservationNodes, groups, pools] = await Promise.all([
            TrainingReservationTable.find({ order: { createdAt: 'DESC' }, take: 200 }),
            TrainingReservationNodeTable.find(),
            UserGroupTable.find(),
            ResourcePoolTable.find(),
        ]);
        const groupNames = new Map(groups.map((group) => [group.id, group.name]));
        const poolNames = new Map(pools.map((pool) => [pool.id, pool.name]));
        return reservations.map((reservation) => {
            const isEnded = [
                TrainingReservationStatus.RELEASED,
                TrainingReservationStatus.FAILED,
            ].includes(reservation.status);
            return {
                ...reservation,
                groupName: groupNames.get(reservation.groupId) || reservation.groupId,
                poolName: poolNames.get(reservation.poolId) || reservation.poolId,
                nodes: reservationNodes.filter((node) => node.reservationId === reservation.id),
                endedAt: reservation.releasedAt || (isEnded ? reservation.updatedAt?.toISOString?.() || String(reservation.updatedAt) : null),
                endReason: reservation.expiredReason || reservation.errorMessage || reservation.releaseResult || null,
            };
        });
    }

    static async upsertPool(input: { id?: string; name: string; description?: string }) {
        const pool = input.id
            ? await ResourcePoolTable.findOne({ where: { id: input.id } })
            : null;
        return ResourcePoolTable.create({
            ...pool,
            id: pool?.id || crypto.randomUUID(),
            name: input.name.trim(),
            description: input.description?.trim() || null,
        }).save();
    }

    static async setPoolEnabled(poolId: string, enabled: boolean) {
        return this.withReservationLock('training-resources', async () => {
            const pool = await ResourcePoolTable.findOne({ where: { id: poolId } });
            if (!pool) throw new Error('资源池不存在');
            pool.enabled = enabled;
            return pool.save();
        });
    }

    static async setPoolNodes(
        poolId: string,
        nodes: PoolNodeInput[],
    ) {
        return this.withReservationLock(`pool:${poolId}`, () => this.setPoolNodesUnlocked(poolId, nodes));
    }

    private static async setPoolNodesUnlocked(
        poolId: string,
        nodes: PoolNodeInput[],
    ) {
        if (!(await ResourcePoolTable.findOne({ where: { id: poolId } }))) {
            throw new Error('资源池不存在');
        }
        await this.assertNoActiveReservations({ poolId });
        nodes.forEach((node) => resourceNodeRegistry.get(node.nodeId));
        for (const [field, values] of [
            ['nodeId', nodes.map((node) => node.nodeId.trim())],
            ['sshAlias', nodes.map((node) => node.sshAlias.trim())],
            ['trainAddress', nodes.map((node) => node.trainAddress.trim())],
        ] as const) {
            if (values.some((value) => !value)) throw new Error(`资源池节点 ${field} 不能为空`);
            if (new Set(values).size !== values.length) throw new Error(`资源池节点 ${field} 不能重复`);
        }
        const normalizedGpuIndexes = new Map(nodes.map((node) => [
            node.nodeId,
            normalizeAllowedGpuIndexes(node.allowedGpuIndexes),
        ]));
        await ResourcePoolNodeTable.delete({ poolId });
        return ResourcePoolNodeTable.save(nodes.map((node) => ResourcePoolNodeTable.create({
            id: crypto.randomUUID(),
            poolId,
            nodeId: node.nodeId,
            sshAlias: node.sshAlias.trim(),
            trainAddress: node.trainAddress.trim(),
            ncclSocketIfname: node.ncclSocketIfname?.trim() || null,
            allowedGpuIndexes: normalizedGpuIndexes.get(node.nodeId),
            enabled: node.enabled ?? true,
        })));
    }

    static async setGroupQuota(input: {
        groupId: string;
        poolId: string;
        homeNodeId: string;
        guaranteedGpuCount: number;
        maxGpuCount: number;
        maxConcurrentJobs: number;
        maxNodesPerJob: number;
    }) {
        return this.withReservationLock(`pool:${input.poolId}`, () => this.setGroupQuotaUnlocked(input));
    }

    private static async setGroupQuotaUnlocked(input: {
        groupId: string;
        poolId: string;
        homeNodeId: string;
        guaranteedGpuCount: number;
        maxGpuCount: number;
        maxConcurrentJobs: number;
        maxNodesPerJob: number;
    }) {
        await this.assertNoActiveReservations({ groupId: input.groupId, poolId: input.poolId });
        const [group, poolNode, existing, runtimeAssignment] = await Promise.all([
            UserGroupTable.findOne({ where: { id: input.groupId } }),
            ResourcePoolNodeTable.findOne({ where: { poolId: input.poolId, nodeId: input.homeNodeId } }),
            GroupResourceQuotaTable.findOne({ where: { groupId: input.groupId, poolId: input.poolId } }),
            GroupNodeAssignmentTable.findOne({ where: { groupId: input.groupId } }),
        ]);
        if (!group) throw new Error('用户组不存在');
        if (!poolNode) throw new Error('归属 Runtime 节点必须属于资源池');
        if (!runtimeAssignment || runtimeAssignment.nodeId !== input.homeNodeId) {
            throw new Error('资源池归属节点必须与用户组当前 Runtime 节点一致');
        }
        if (!Number.isInteger(input.guaranteedGpuCount) || input.guaranteedGpuCount < 0) {
            throw new Error('保底 GPU 数必须是非负整数');
        }
        for (const [label, value] of [
            ['最大 GPU 数', input.maxGpuCount],
            ['最大并发任务数', input.maxConcurrentJobs],
            ['单任务最大节点数', input.maxNodesPerJob],
        ] as const) {
            if (!Number.isInteger(value) || value < 1) throw new Error(`${label}必须是正整数`);
        }
        if (input.guaranteedGpuCount > input.maxGpuCount) throw new Error('保底 GPU 数不能超过最大 GPU 数');
        const poolNodes = await ResourcePoolNodeTable.find({ where: { poolId: input.poolId, enabled: true } });
        const gpuCapacity = (await uniquePoolNodeGpus(poolNodes))
            .reduce((sum, { node, gpus }) => sum + gpusAllowedForNode(node, gpus).length, 0);
        if (input.maxGpuCount > gpuCapacity) throw new Error('用户组最大 GPU 数不能超过资源池总容量');
        const otherGuaranteed = (await GroupResourceQuotaTable.find({ where: { poolId: input.poolId } }))
            .filter((quota) => quota.groupId !== input.groupId)
            .reduce((sum, quota) => sum + quota.guaranteedGpuCount, 0);
        if (otherGuaranteed + input.guaranteedGpuCount > gpuCapacity) {
            throw new Error('资源池内各用户组保底 GPU 数之和不能超过资源池总容量');
        }
        return GroupResourceQuotaTable.create({
            ...existing,
            id: existing?.id || crypto.randomUUID(),
            ...input,
        }).save();
    }

    static async deleteGroupQuota(groupId: string, poolId: string) {
        return this.withReservationLock(`pool:${poolId}`, async () => {
            await this.assertNoActiveReservations({ groupId, poolId });
            await GroupResourceQuotaTable.delete({ groupId, poolId });
            return { groupId, poolId };
        });
    }

    private static async groupIdForUser(user: SafeAuthUser, requestedGroupId?: string) {
        if (user.role === UserRole.ADMIN && requestedGroupId) return requestedGroupId;
        const membership = await UserGroupMemberTable.findOne({ where: { userId: user.id } });
        if (!membership) throw new Error('用户未加入用户组');
        return membership.groupId;
    }

    static async reserve(
        user: SafeAuthUser,
        input: { groupId?: string; poolId: string; nodeCount: number; gpusPerNode: number; taskCategory?: string | null; taskType?: string | null; taskTypeText?: string | null },
    ) {
        const groupId = await this.groupIdForUser(user, input.groupId);
        return this.reserveForGroup(groupId, user.id, input);
    }

    static async reserveForRuntime(input: {
        groupId: string;
        poolId?: string;
        runtimeNodeId: string;
        nodeCount: number;
        gpusPerNode: number;
        runtimeGpuSnapshot?: RuntimeGpuSnapshot;
        taskCategory?: string | null;
        taskType?: string | null;
        taskTypeText?: string | null;
    }) {
        const assignment = await GroupNodeAssignmentTable.findOne({ where: { groupId: input.groupId } });
        if (!assignment || assignment.nodeId !== input.runtimeNodeId) {
            throw new Error('当前 Runtime 不是该用户组的归属节点');
        }
        const poolId = input.poolId?.trim() || await this.resolvePoolIdForGroup(input.groupId);
        return this.reserveForGroup(input.groupId, null, { ...input, poolId, runtimeNodeId: input.runtimeNodeId });
    }

    static async preflightForRuntime(input: {
        groupId: string;
        poolId?: string;
        runtimeNodeId: string;
        nodeCount: number;
        gpusPerNode: number;
        runtimeGpuSnapshot?: RuntimeGpuSnapshot;
        taskCategory?: string | null;
        taskType?: string | null;
        taskTypeText?: string | null;
    }): Promise<TrainingResourcePreflightResult> {
        const assignment = await GroupNodeAssignmentTable.findOne({ where: { groupId: input.groupId } });
        if (!assignment || assignment.nodeId !== input.runtimeNodeId) {
            throw new Error('当前 Runtime 不是该用户组的归属节点');
        }
        const poolId = input.poolId?.trim() || await this.resolvePoolIdForGroup(input.groupId);
        return this.reserveForGroup(input.groupId, null, {
            ...input,
            poolId,
            runtimeNodeId: input.runtimeNodeId,
            dryRun: true,
        });
    }

    static async listRuntimeGroupsForNode(runtimeNodeId: string) {
        const assignments = await GroupNodeAssignmentTable.find({ where: { nodeId: runtimeNodeId } });
        if (!assignments.length) return [];
        const groups = await UserGroupTable.find({
            where: { id: In(assignments.map((assignment) => assignment.groupId)) },
        });
        const groupNames = new Map(groups.map((group) => [group.id, group.name]));
        return assignments.map((assignment) => ({
            groupId: assignment.groupId,
            groupName: groupNames.get(assignment.groupId) || assignment.groupId,
            runtimeNodeId: assignment.nodeId,
        }));
    }

    private static async resolvePoolIdForGroup(groupId: string) {
        const quotas = await GroupResourceQuotaTable.find({ where: { groupId } });
        if (!quotas.length) throw new Error('用户组未配置训练资源池配额');
        const pools = await ResourcePoolTable.find({
            where: { id: In(quotas.map((quota) => quota.poolId)), enabled: true },
        });
        if (!pools.length) throw new Error('用户组没有已启用的训练资源池');
        if (pools.length > 1) throw new Error('用户组配置了多个已启用训练资源池，启动时必须指定资源池');
        return pools[0].id;
    }

    private static async reserveForGroup(
        groupId: string,
        requestedByUserId: string | null,
        input: TrainingResourcePreflightInput,
    ): Promise<TrainingResourcePreflightResult>;
    private static async reserveForGroup(
        groupId: string,
        requestedByUserId: string | null,
        input: TrainingResourceReservationInput,
    ): Promise<TrainingResourceAllocation>;
    private static async reserveForGroup(
        groupId: string,
        requestedByUserId: string | null,
        input: TrainingResourceReservationInput | TrainingResourcePreflightInput,
    ): Promise<TrainingResourceAllocation | TrainingResourcePreflightResult> {
        if (input.dryRun) {
            return this.withReservationLock('training-resources', () =>
                this.reserveForGroupUnlocked(groupId, requestedByUserId, input),
            );
        }
        return this.withReservationLock('training-resources', () =>
            this.reserveForGroupUnlocked(groupId, requestedByUserId, input),
        );
    }

    private static async reserveForGroupUnlocked(
        groupId: string,
        requestedByUserId: string | null,
        input: TrainingResourcePreflightInput,
    ): Promise<TrainingResourcePreflightResult>;
    private static async reserveForGroupUnlocked(
        groupId: string,
        requestedByUserId: string | null,
        input: TrainingResourceReservationInput,
    ): Promise<TrainingResourceAllocation>;
    private static async reserveForGroupUnlocked(
        groupId: string,
        requestedByUserId: string | null,
        input: TrainingResourceReservationInput | TrainingResourcePreflightInput,
    ): Promise<TrainingResourceAllocation | TrainingResourcePreflightResult> {
        const quota = await GroupResourceQuotaTable.findOne({ where: { groupId, poolId: input.poolId } });
        if (!quota) throw new Error('用户组未配置该资源池配额');
        const pool = await ResourcePoolTable.findOne({ where: { id: input.poolId } });
        if (!pool) throw new Error('资源池不存在');
        if (!pool.enabled) throw new Error('资源池已禁用');
        if (!Number.isInteger(input.nodeCount) || input.nodeCount < 1) throw new Error('请求节点数必须是正整数');
        if (!Number.isInteger(input.gpusPerNode) || input.gpusPerNode < 1) throw new Error('每节点 GPU 数必须是正整数');
        if (input.nodeCount < 1 || input.nodeCount > quota.maxNodesPerJob) throw new Error('请求节点数超过用户组配额');
        const requestedGpuCount = input.nodeCount * input.gpusPerNode;
        if (requestedGpuCount > quota.maxGpuCount) throw new Error('请求 GPU 数超过用户组最大配额');

        const allActive = await this.activeReservations({});
        const poolActive = allActive.filter((reservation) => reservation.poolId === input.poolId);
        const active = poolActive.filter((reservation) => reservation.groupId === groupId);
        if (active.length >= quota.maxConcurrentJobs) throw new Error('用户组并发训练任务数已达到上限');
        const allActiveNodes = allActive.length
            ? await TrainingReservationNodeTable.find({ where: { reservationId: In(allActive.map((item) => item.id)) } })
            : [];
        const poolActiveIds = new Set(poolActive.map((item) => item.id));
        const poolActiveNodes = allActiveNodes.filter((node) => poolActiveIds.has(node.reservationId));
        const reservedByNode = new Map<string, Set<number>>();
        for (const node of allActiveNodes) {
            const reserved = reservedByNode.get(node.nodeId) || new Set<number>();
            node.gpuIndexes.forEach((index) => reserved.add(index));
            reservedByNode.set(node.nodeId, reserved);
        }
        const activeReservationIds = new Set(active.map((item) => item.id));
        const activeNodes = poolActiveNodes.filter((node) => activeReservationIds.has(node.reservationId));
        const usedGpuCount = activeNodes.reduce((sum, item) => sum + item.gpuIndexes.length, 0);
        if (usedGpuCount + requestedGpuCount > quota.maxGpuCount) throw new Error('用户组 GPU 使用量将超过最大配额');

        const poolNodes = await ResourcePoolNodeTable.find({ where: { poolId: input.poolId, enabled: true } });
        const home = poolNodes.find((node) => node.nodeId === quota.homeNodeId);
        if (!home) throw new Error('归属 Runtime 节点不在可用资源池中');

        const singleNodeRuntimeAllocation = input.nodeCount <= 1 && !!input.runtimeNodeId;
        const allocationPoolNodes = singleNodeRuntimeAllocation
            ? poolNodes.filter((node) => node.nodeId === input.runtimeNodeId)
            : poolNodes;
        const localRuntimeNodeIds = new Set<string>();
        if (input.runtimeNodeId) localRuntimeNodeIds.add(input.runtimeNodeId);
        const localSnapshots = allocationPoolNodes
            .filter((node) => localRuntimeNodeIds.has(node.nodeId))
            .map((node) => {
                const snapshotGpus = runtimeSnapshotGpus(node.nodeId, input.runtimeGpuSnapshot);
                const configuredIndexes = configuredOrSnapshotGpuIndexesForNode(node, snapshotGpus);
                const configured = new Set(configuredIndexes);
                const reserved = reservedByNode.get(node.nodeId);
                const snapshotByIndex = new Map(snapshotGpus.map((gpu) => [gpu.index, gpu]));
                const available = snapshotGpus
                    .filter((gpu) =>
                        configured.has(gpu.index) &&
                        gpu.available !== false &&
                        Number(gpu.memoryUsed || 0) < gpuBusyMemoryThresholdMb() &&
                        !reserved?.has(gpu.index),
                    )
                    .map((gpu) => gpu.index)
                    .sort((left, right) => left - right);
                const reservedGpuIndexes = configuredIndexes.filter((index) => reserved?.has(index));
                const busyGpuIndexes = configuredIndexes.filter((index) => {
                    const gpu = snapshotByIndex.get(index);
                    return gpu && !reserved?.has(index) && (
                        gpu.available === false ||
                        Number(gpu.memoryUsed || 0) >= gpuBusyMemoryThresholdMb()
                    );
                });
                const missingGpuIndexes = configuredIndexes.filter((index) => !snapshotByIndex.has(index));
                console.info(
                    `[TrainingResource] GPU local-runtime node=${node.nodeId} ` +
                    `allowed=${configuredIndexes.length} reserved=${reserved?.size || 0} ` +
                    `available=${available.length} thresholdMb=${gpuBusyMemoryThresholdMb()} ` +
                    `snapshotAge=${input.runtimeGpuSnapshot?.ageSeconds ?? 'unknown'}; use runtime request snapshot`,
                );
                return { node, gpuIndexes: available, reservedGpuIndexes, busyGpuIndexes, missingGpuIndexes };
            });
        const remoteSnapshotNodes = allocationPoolNodes.filter((node) => !localRuntimeNodeIds.has(node.nodeId));
        const remoteSnapshots = (await uniquePoolNodeGpus(remoteSnapshotNodes)).map(({ node, gpus }) => {
            const configuredIndexes = configuredOrSnapshotGpuIndexesForNode(node, gpus);
            const configured = new Set(configuredIndexes);
            const reserved = reservedByNode.get(node.nodeId);
            const snapshotByIndex = new Map(gpus.map((gpu) => [gpu.index, gpu]));
            const available = gpus
                .filter((gpu) =>
                    configured.has(gpu.index) &&
                    gpu.available !== false &&
                    Number(gpu.memoryUsed || 0) < gpuBusyMemoryThresholdMb() &&
                    !reserved?.has(gpu.index),
                )
                .map((gpu) => gpu.index)
                .sort((left, right) => left - right);
            const reservedGpuIndexes = configuredIndexes.filter((index) => reserved?.has(index));
            const busyGpuIndexes = configuredIndexes.filter((index) => {
                const gpu = snapshotByIndex.get(index);
                return gpu && !reserved?.has(index) && (
                    gpu.available === false ||
                    Number(gpu.memoryUsed || 0) >= gpuBusyMemoryThresholdMb()
                );
            });
            const missingGpuIndexes = configuredIndexes.filter((index) => !snapshotByIndex.has(index));
            console.info(
                `[TrainingResource] GPU snapshot node=${node.nodeId} raw=${gpus.length} ` +
                `allowed=${configuredIndexes.length} reserved=${reserved?.size || 0} ` +
                `available=${available.length} thresholdMb=${gpuBusyMemoryThresholdMb()}`,
            );
            return {
                node,
                gpuIndexes: available,
                reservedGpuIndexes,
                busyGpuIndexes,
                missingGpuIndexes,
            };
        });
        const snapshots = [...localSnapshots, ...remoteSnapshots];
        const usageByGroup = new Map<string, number>();
        const groupByReservation = new Map(poolActive.map((reservation) => [reservation.id, reservation.groupId]));
        for (const node of poolActiveNodes) {
            const reservationGroupId = groupByReservation.get(node.reservationId);
            if (reservationGroupId) {
                usageByGroup.set(
                    reservationGroupId,
                    (usageByGroup.get(reservationGroupId) || 0) + node.gpuIndexes.length,
                );
            }
        }
        const protectedForOtherGroups = (await GroupResourceQuotaTable.find({ where: { poolId: input.poolId } }))
            .filter((item) => item.groupId !== groupId)
            .reduce(
                (sum, item) => sum + Math.max(0, item.guaranteedGpuCount - (usageByGroup.get(item.groupId) || 0)),
                0,
            );
        const availableGpuCount = snapshots.reduce((sum, item) => sum + item.gpuIndexes.length, 0);
        if (requestedGpuCount > availableGpuCount) {
            throw new Error(
                `资源池当前可用 GPU 不足：需要 ${requestedGpuCount} 张，当前可用 ${availableGpuCount} 张。` +
                `GPU 明细：${formatGpuDiagnostics(snapshots)}`,
            );
        }
        if (requestedGpuCount > Math.max(0, availableGpuCount - protectedForOtherGroups)) {
            throw new Error('请求会占用其他用户组的保底 GPU 配额');
        }
        const candidates = [
            snapshots.find((item) => item.node.nodeId === quota.homeNodeId)!,
            ...snapshots.filter((item) => item.node.nodeId !== quota.homeNodeId),
        ];
        const homeSnapshot = candidates[0];
        if (!homeSnapshot || homeSnapshot.gpuIndexes.length < input.gpusPerNode) {
            throw new Error(
                `归属 Runtime 节点没有足够的可用 GPU，无法作为多机训练 master。` +
                `需要 ${input.gpusPerNode} 张，当前可用 ${homeSnapshot?.gpuIndexes.length || 0} 张。` +
                `GPU 明细：${formatGpuDiagnostics(snapshots)}`,
            );
        }
        const selectedCandidates = [
            homeSnapshot,
            ...candidates.slice(1).filter((item) => item.gpuIndexes.length >= input.gpusPerNode),
        ].slice(0, input.nodeCount)
            .map((item) => ({ ...item, gpuIndexes: item.gpuIndexes.slice(0, input.gpusPerNode) }));
        console.info(
            `[TrainingResource] selected group=${groupId} pool=${input.poolId} ` +
            `nodes=${selectedCandidates.map((item) => `${item.node.nodeId}:${item.gpuIndexes.join(',')}`).join('|')}`,
        );
        if (selectedCandidates.length !== input.nodeCount) {
            const eligibleNodeCount = snapshots.filter((item) => item.gpuIndexes.length >= input.gpusPerNode).length;
            throw new Error(
                `资源池没有足够的可用节点和 GPU：需要 ${input.nodeCount} 个节点，每节点 ${input.gpusPerNode} 张 GPU，` +
                `当前满足条件节点 ${eligibleNodeCount} 个。GPU 明细：${formatGpuDiagnostics(snapshots)}`,
            );
        }
        if (input.dryRun) {
            return {
                ok: true as const,
                groupId,
                poolId: input.poolId,
                nodeCount: input.nodeCount,
                gpusPerNode: input.gpusPerNode,
                availableGpuCount,
                diagnostics: formatGpuDiagnostics(snapshots),
            };
        }
        const usedMasterPorts = new Set(
            poolActive
                .filter((item) => item.homeNodeId === quota.homeNodeId)
                .map((item) => item.masterPort),
        );
        const masterPort = Array.from({ length: 1000 }, (_, index) => 29500 + index)
            .find((port) => !usedMasterPorts.has(port));
        if (!masterPort) throw new Error('归属 Runtime 没有可用的多机训练 master 端口');

        const reservation = await TrainingReservationTable.create({
            id: crypto.randomUUID(),
            groupId,
            poolId: input.poolId,
            homeNodeId: quota.homeNodeId,
            requestedByUserId,
            status: TrainingReservationStatus.PREPARING,
            requestedNodeCount: input.nodeCount,
            gpusPerNode: input.gpusPerNode,
            masterPort,
            taskCategory: input.taskCategory?.trim() || null,
            taskType: input.taskType?.trim() || null,
            taskTypeText: input.taskTypeText?.trim() || null,
            expiresAt: new Date(Date.now() + reservationTtlSeconds(input) * 1000).toISOString(),
            errorMessage: null,
            expiredReason: null,
            lastRenewedAt: null,
            releaseResult: null,
            releasedAt: null,
        }).save();
        await recordReservationAuditEvent('training_reservation_created', reservation, requestedByUserId, {
            requestedNodeCount: input.nodeCount,
            gpusPerNode: input.gpusPerNode,
        });

        const allocationNodes: AllocationNode[] = selectedCandidates.map(({ node, gpuIndexes }) => ({
            nodeId: node.nodeId,
            sshAlias: node.sshAlias,
            trainAddress: node.trainAddress,
            ncclSocketIfname: node.ncclSocketIfname,
            gpuIndexes,
            isMaster: node.nodeId === quota.homeNodeId,
        }));
        try {
            await TrainingReservationNodeTable.save(allocationNodes.map((node) => TrainingReservationNodeTable.create({
                id: crypto.randomUUID(),
                reservationId: reservation.id,
                nodeId: node.nodeId,
                gpuIndexes: node.gpuIndexes,
                isMaster: node.isMaster,
            })));
            reservation.status = TrainingReservationStatus.RESERVED;
            await reservation.save();
            const allocation = {
                reservationId: reservation.id,
                groupId,
                poolId: input.poolId,
                masterNodeId: quota.homeNodeId,
                masterAddr: home.trainAddress,
                masterPort: reservation.masterPort,
                expiresAt: reservation.expiresAt,
                nodes: allocationNodes,
            };
            if (input.nodeCount <= 1) return allocation;
            const allocationFile = await remoteResourceClient.request<{
                data: { allocationFile: string; container: string };
            }>(quota.homeNodeId, 'training-allocations/write', {
                method: 'POST',
                body: JSON.stringify({ reservationId: reservation.id, allocation }),
            });
            return {
                ...allocation,
                allocationFile: allocationFile.data.allocationFile,
                allocationContainer: allocationFile.data.container,
            };
        } catch (error) {
            reservation.status = TrainingReservationStatus.FAILED;
            reservation.errorMessage = error instanceof Error ? error.message : String(error);
            await reservation.save();
            throw error;
        }
    }

    static async release(user: SafeAuthUser, reservationId: string, options: TrainingResourceReleaseOptions = {}) {
        const reservation = await TrainingReservationTable.findOne({ where: { id: reservationId } });
        if (!reservation) throw new Error('训练预约不存在');
        if (user.role !== UserRole.ADMIN && reservation.requestedByUserId !== user.id) {
            throw new Error('无权释放其他用户的训练预约');
        }
        return this.releaseReservation(reservation, {
            ...options,
            source: options.source || 'admin',
            actorUserId: options.actorUserId ?? user.id,
        });
    }

    static async stopProcessAndRelease(user: SafeAuthUser, reservationId: string) {
        const reservation = await TrainingReservationTable.findOne({ where: { id: reservationId } });
        if (!reservation) throw new Error('训练预约不存在');
        if (user.role !== UserRole.ADMIN) {
            throw new Error('该操作仅管理员可执行');
        }
        if (!activeReservationStatuses.includes(reservation.status)) {
            throw new Error('当前训练预约状态不允许停止进程并释放');
        }
        if (isInferenceReservation(reservation)) {
            return this.stopInferenceServiceAndRelease(reservation, user);
        }
        const stopResponse = await remoteResourceClient.request<{ data: RuntimeStopProcessResult }>(
            reservation.homeNodeId,
            'training-reservations/stop-process',
            {
                method: 'POST',
                body: JSON.stringify({ reservationId: reservation.id }),
            },
            undefined,
            Math.max(1000, Number(process.env.MEDFLOW_RESOURCE_STOP_PROCESS_TIMEOUT_MS || 60000)),
        );
        const stopResult = stopResponse.data;
        if (!stopResult || stopResult.remainingPids?.length || (!stopResult.stopped && !stopResult.alreadyExited)) {
            if (stopResult?.remainingPids?.length) {
                throw new Error(`Runtime 停止进程后仍有残留 PID：${stopResult.remainingPids.join(", ")}`);
            }
            throw new Error(stopResult?.message || "Runtime 未确认训练进程已退出");
        }
        const released = await this.releaseReservation(reservation, {
            reason: '管理员停止进程并释放预约',
            releaseResult: 'stopped_and_released',
            source: 'admin',
            actorUserId: user.id,
        });
        return { reservation: released, stopResult };
    }

    private static async inferenceResourceContext(reservation: TrainingReservationTable): Promise<InferenceResourceContext> {
        const reservationNodes = await TrainingReservationNodeTable.find({ where: { reservationId: reservation.id } });
        if (!reservationNodes.length) {
            throw new Error('推理预约缺少资源节点分配，无法停止推理服务');
        }
        const masterNode = reservationNodes.find((node) => node.isMaster) || reservationNodes[0];
        const assignedGpus = masterNode.gpuIndexes.map((index) => String(index));
        const nodes = reservationNodes.map((node) => {
            const nodeGpus = node.gpuIndexes.map((index) => String(index));
            return {
                runtime_node_id: node.nodeId,
                assigned_gpus: nodeGpus,
                cuda_visible_devices: nodeGpus.join(','),
                tensor_parallel_size: nodeGpus.length,
                gpus_per_node: reservation.gpusPerNode,
                is_master: node.isMaster,
            };
        });
        return {
            reservation_id: reservation.id,
            resource_group_id: reservation.groupId,
            training_pool_id: reservation.poolId,
            runtime_node_id: masterNode.nodeId,
            assigned_gpus: assignedGpus,
            cuda_visible_devices: assignedGpus.join(','),
            tensor_parallel_size: assignedGpus.length,
            gpus_per_node: reservation.gpusPerNode,
            expires_at: reservation.expiresAt,
            nodes,
        };
    }

    private static async stopInferenceServiceAndRelease(reservation: TrainingReservationTable, user: SafeAuthUser) {
        const resourceContext = await this.inferenceResourceContext(reservation);
        const stopResponse = await remoteResourceClient.request<{ data: RuntimeStopInferenceServiceResult }>(
            reservation.homeNodeId,
            'inference-reservations/stop-service',
            {
                method: 'POST',
                body: JSON.stringify({ reservationId: reservation.id, resourceContext }),
            },
            undefined,
            Math.max(1000, Number(process.env.MEDFLOW_RESOURCE_STOP_PROCESS_TIMEOUT_MS || 60000)),
        );
        const stopResult = stopResponse.data;
        if (!stopResult || stopResult.stopped !== true || stopResult.releaseReady === false) {
            throw new Error(stopResult?.message || 'Runtime 未确认推理服务已停止');
        }
        const released = await this.releaseReservation(reservation, {
            reason: '管理员停止推理服务并释放预约',
            releaseResult: 'stopped_and_released',
            source: 'admin',
            actorUserId: user.id,
        });
        return { reservation: released, stopResult };
    }

    static async releaseForRuntime(reservationId: string, runtimeNodeId: string) {
        const reservation = await TrainingReservationTable.findOne({ where: { id: reservationId } });
        if (!reservation) throw new Error('训练预约不存在');
        if (reservation.homeNodeId !== runtimeNodeId) {
            throw new Error('当前 Runtime 无权释放该训练预约');
        }

        return this.releaseReservation(reservation, {
            reason: 'Runtime 主动释放预约',
            source: 'runtime',
            releaseResult: 'success',
        });
    }

    private static async releaseReservation(
        reservation: TrainingReservationTable,
        options: TrainingResourceReleaseOptions = {},
    ) {
        return this.withReservationLock(`reservation:${reservation.id}`, () =>
            this.releaseReservationUnlocked(reservation.id, options),
        );
    }

    private static async releaseReservationUnlocked(
        reservationId: string,
        options: TrainingResourceReleaseOptions = {},
    ) {
        const reservation = await TrainingReservationTable.findOne({ where: { id: reservationId } });
        if (!reservation) throw new Error('训练预约不存在');
        if (reservation.status === TrainingReservationStatus.RELEASED) {
            return reservation;
        }
        const { failures } = await this.releaseReservationNodes(reservation.id);
        const releasedAt = new Date().toISOString();
        reservation.status = failures.length
            ? TrainingReservationStatus.FAILED
            : TrainingReservationStatus.RELEASED;
        reservation.errorMessage = failures.length ? `GPU 释放失败：${failures.join('; ')}` : null;
        reservation.releaseResult = failures.length
            ? 'failed'
            : options.releaseResult || (options.force ? 'force_released' : 'success');
        reservation.releasedAt = releasedAt;
        if (!failures.length && options.reason?.trim()) {
            reservation.expiredReason = options.reason.trim();
        }
        await reservation.save();
        const eventType = failures.length
            ? 'training_reservation_release_failed'
            : reservation.releaseResult === 'stopped_and_released'
                ? 'training_reservation_stopped_and_released'
                : reservation.releaseResult === 'force_released'
                    ? 'training_reservation_force_released'
                    : 'training_reservation_released';
        await recordReservationAuditEvent(eventType, reservation, options.actorUserId || null, {
            source: options.source || (options.force ? 'admin' : 'runtime'),
            releaseResult: reservation.releaseResult,
            releasedAt,
            reason: reservation.expiredReason || reservation.errorMessage || null,
            failures,
        });
        return reservation;
    }

    private static async releaseReservationNodes(reservationId: string) {
        const nodes = await TrainingReservationNodeTable.find({ where: { reservationId } });
        if (!nodes.length) return { failures: [] };
        // Keep node/GPU allocation rows as an immutable audit trail. Active resource
        // accounting is based on reservation status, so released/failed rows do not
        // continue to reserve GPUs.
        return { failures: [] };
    }

    static async renewForRuntime(reservationId: string, runtimeNodeId: string) {
        return this.withReservationLock(`reservation:${reservationId}`, () =>
            this.renewForRuntimeUnlocked(reservationId, runtimeNodeId),
        );
    }

    private static async renewForRuntimeUnlocked(reservationId: string, runtimeNodeId: string) {
        const reservation = await TrainingReservationTable.findOne({ where: { id: reservationId } });
        if (!reservation) throw new Error('训练预约不存在');
        if (reservation.homeNodeId !== runtimeNodeId) {
            throw new Error('当前 Runtime 无权续期该训练预约');
        }
        if (![TrainingReservationStatus.RESERVED, TrainingReservationStatus.RUNNING].includes(reservation.status)) {
            throw new Error('当前训练预约状态不允许续期');
        }
        const now = new Date().toISOString();
        const previousExpiresAt = reservation.expiresAt;
        const expiresAt = new Date(Date.now() + reservationTtlSeconds(reservation) * 1000).toISOString();
        reservation.status = TrainingReservationStatus.RUNNING;
        reservation.expiresAt = expiresAt;
        reservation.lastRenewedAt = now;
        await reservation.save();
        await recordReservationAuditEvent('training_reservation_renewed', reservation, null, {
            runtimeNodeId,
            renewedAt: now,
            previousExpiresAt,
            expiresAt,
        });
        return reservation;
    }
}


