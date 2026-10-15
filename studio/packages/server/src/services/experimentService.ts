import crypto from 'crypto';
import { TRPCError } from '@trpc/server';
import { Buffer } from 'node:buffer';
import { EntityManager, In, LessThan, LessThanOrEqual, MoreThanOrEqual } from 'typeorm';
import { z } from 'zod';
import { appendExperimentLogSchema, assetRegistrationSchema, attemptStatusSchema, evaluationSchema, experimentIdSchema,
    experimentMetadataSchema, searchExperimentLogsSchema, experimentLogContextSchema, serviceOperationSchema,
    AppendExperimentLogInput, ExperimentEventInput, PrepareExperimentInput } from '../../../shared/src/experiment';
import { AuthDao, SafeAuthUser } from '../dao/Auth';
import { UserRole } from '../models/Auth';
import { remoteResourceClient, resourceNodeRegistry } from './resourceNodeService';
import { ResourceAccessService } from './resourceAccessService';
import { TrainingResourceService } from './trainingResourceService';
import { equivalentModelVersions, modelAssetKinds, assetContentKey, modelShardSummary } from './experimentModelIdentity';
import { applyExperimentListFilters } from './experimentListQuery';
import { experimentStatus } from './experimentStatus';
import { searchArchivedLogs, suspectedLogErrors } from './experimentLogSearch';
import type { ListExperimentsInput } from '../../../shared/src/experiment';
import type { ServiceOperationHistory } from '../../../shared/src/experiment';
import { TrainingReservationTable } from '../models/TrainingResource';
import { AssetVersionTable, EvaluationMetricTable, EvaluationResultTable, ExperimentEventTable,
    ExperimentLogSourceTable, ExperimentLogChunkTable,
    ExperimentTable, ExperimentTaskTable, TaskAssetTable, TaskAttemptTable } from '../models/Experiment';

const now = () => new Date().toISOString();
const id = (prefix: string) => `${prefix}-${crypto.randomUUID()}`;
const terminal = new Set(['succeeded', 'failed', 'cancelled']);
type AssociationScope = 'all' | 'model' | 'dataset' | 'report';
type OwnedAssetRelation = { link: TaskAssetTable; task: ExperimentTaskTable; record: ExperimentTable };
// Each public read owns its context; no cached permissions or records survive the request.
const readContext = () => ({
    fingerprints: new Map<string, AssetVersionTable[]>(),
    links: new Map<string, OwnedAssetRelation[]>(),
    peers: new Map<string, TaskAssetTable[]>(),
    assets: new Map<string, AssetVersionTable>(),
});
type ReadContext = ReturnType<typeof readContext>;
const inAssociationScope = (asset: AssetVersionTable, scope: AssociationScope) =>
    scope === 'all' || (scope === 'model' ? modelAssetKinds.includes(asset.kind) : asset.kind === scope);
const comparisonIneligibility = (attempt: TaskAttemptTable | undefined, result: EvaluationResultTable) =>
    attempt?.status !== 'succeeded' ? 'execution-not-succeeded'
        : attempt.runtime.inputIntegrityAfterExecution !== 'verified' ? 'inputs-not-verified'
        : !result.protocolComplete ? 'protocol-incomplete'
        : result.expectedCount === null || result.expectedCount <= 0 ||
            result.succeededCount !== result.expectedCount || result.failedCount !== 0 ? 'samples-incomplete' : null;
const deny = (message: string): never => { throw new TRPCError({ code: 'FORBIDDEN', message }); };
const conflict = (message: string): never => { throw new TRPCError({ code: 'CONFLICT', message }); };
let transactionTail: Promise<void> = Promise.resolve();
async function transaction<T>(operation: (manager: EntityManager) => Promise<T>): Promise<T> {
    const previous = transactionTail;
    let release!: () => void;
    transactionTail = new Promise<void>(resolve => { release = resolve; });
    await previous;
    try { return await ExperimentTable.getRepository().manager.transaction(operation); }
    finally { release(); }
}

export function canonicalJson(value: unknown): string {
    if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
    if (value !== null && typeof value === 'object') {
        return `{${Object.entries(value).filter(([, item]) => item !== undefined)
            .sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0)
            .map(([key, item]) => `${JSON.stringify(key)}:${canonicalJson(item)}`).join(',')}}`;
    }
    return JSON.stringify(value) ?? 'null';
}
const digest = (value: unknown) => crypto.createHash('sha256').update(canonicalJson(value)).digest('hex');
const proposal = (spec: Record<string, unknown>) => {
    if (spec.schemaVersion !== 1 || typeof spec.script !== 'string' || !spec.resolved) return spec;
    const versions = Array.isArray(spec.inputVersions) ? spec.inputVersions : [];
    if (spec.kind === 'benchmark' && spec.executionBackend === 'inference_worker') {
        const protocol = z.object({ runnerSha256: z.string().regex(/^[a-f0-9]{64}$/),
            arguments: z.array(z.string()) }).passthrough().safeParse(spec.protocol);
        if (!protocol.success) return spec; // Legacy records without a fixed protocol cannot ignore command changes.
        // Job directories, service instances and API addresses change on retry.
        // The runner/protocol and exact input versions define the same task.
        return { script: spec.script, kind: spec.kind, executionBackend: spec.executionBackend,
            protocol: protocol.data, inputVersions: versions.map(canonicalJson).sort() };
    }
    if (spec.kind !== 'train') return { script: spec.script, kind: spec.kind, resolved: spec.resolved,
        inputVersions: versions.map(canonicalJson).sort() };
    const resolved = z.record(z.unknown()).parse(spec.resolved);
    const resumeKeys = new Set(['RESUME', 'resume-from-checkpoint', 'resume_from_checkpoint']);
    const withoutResume = (value: unknown) => Object.fromEntries(Object.entries(
        z.record(z.unknown()).parse(value || {})).filter(([key]) => !resumeKeys.has(key)));
    const fixedResolved = { ...resolved };
    if (spec.script === 'grpo_train') {
        delete fixedResolved.resumeCheckpointName;
        fixedResolved.positionalArguments = z.array(z.string()).parse(resolved.positionalArguments || [])
            .filter(argument => !['trainer.resume_from_path', 'trainer.resume_mode'].includes(argument.split('=', 1)[0]));
    }
    // A checkpoint selects where this attempt continues; it does not change the
    // fixed data, base model or hyperparameters of its parent training task.
    return { script: spec.script, kind: spec.kind, resolved: { ...fixedResolved,
        arguments: withoutResume(resolved.arguments), environment: withoutResume(resolved.environment) },
        inputVersions: versions.filter(item => item?.role !== 'resume_checkpoint').map(canonicalJson).sort() };
};

// Defence in depth: collectors must use explicit environment allowlists as well.
export function redactExperimentValue(value: unknown): any {
    if (Array.isArray(value)) {
        let hideNext = false;
        return value.map(item => {
            if (hideNext) { hideNext = false; return '[REDACTED]'; }
            hideNext = typeof item === 'string' && item.startsWith('--') && !item.includes('=') &&
                /password|secret|authorization|api[_-]?key|(?:access|runtime|auth|hf(?:[_-]?hub)?|huggingface|hub)[_-]?token|^token$/i.test(item.replace(/^--/, ''));
            return redactExperimentValue(item);
        });
    }
    if (value !== null && typeof value === 'object') {
        return Object.fromEntries(Object.entries(value).map(([key, item]) => [key,
            /password|secret|authorization|api[_-]?key|(?:access|runtime|auth|hf(?:[_-]?hub)?|huggingface|hub)[_-]?token|^token$/i.test(key)
                ? '[REDACTED]' : redactExperimentValue(item)]));
    }
    if (typeof value === 'string') {
        return value.replace(/Bearer\s+[^\s"']+/gi, 'Bearer [REDACTED]')
            .replace(/(--(?:api[_-]?key|password|secret|authorization|(?:access|runtime|auth|hf(?:[_-]?hub)?|huggingface|hub)[_-]?token))\s+\S+/gi, '$1 [REDACTED]')
            .replace(/(["']?(?:api[_-]?key|password|secret|authorization|(?:access|runtime|auth|hf(?:[_-]?hub)?|huggingface|hub)[_-]?token)["']?\s*[:=]\s*)(["'])(.*?)\2/gi, '$1"[REDACTED]"')
            .replace(/(https?:\/\/)[^\s/@]+:[^\s/@]+@/g, '$1[REDACTED]@')
            .replace(/((?:api[_-]?key|password|secret|authorization|(?:access|runtime|auth|hf(?:[_-]?hub)?|huggingface|hub)[_-]?token)\s*[=:]\s*)[^\s,;"']+/gi, '$1[REDACTED]');
    }
    return value;
}

const statusPayload = z.object({
    status: attemptStatusSchema,
    runtime: z.record(z.unknown()).optional(),
    error: z.string().max(16000).optional(),
});
const workflowStatusPayload = z.object({
    workflowId: experimentIdSchema,
    status: z.enum(['running', 'stopping', 'finished', 'failed', 'stopped']),
    currentStage: z.enum(['train', 'evaluate', 'publish', 'deploy', 'benchmark']),
    updatedAt: z.number().finite().nonnegative(),
});
const configurationPayload = z.object({
    configuration: z.record(z.unknown()),
    environment: z.record(z.unknown()).default({}),
});

export class ExperimentService {
    static async runtimeActor(nodeId: string, userId: string, reservationId?: string) {
        resourceNodeRegistry.get(nodeId);
        const user = await AuthDao.getUserById(userId);
        if (!user || user.disabled) {
            return deny('Experiment actor is not authorized for this Runtime node');
        }
        if (reservationId) {
            const reservation = await TrainingResourceService.inferenceAllocationForRuntime(reservationId, nodeId);
            if (user.role !== UserRole.ADMIN) {
                const group = await ResourceAccessService.getGroupForUser(user.id);
                if (!group || reservation.groupId !== group.id || reservation.homeNodeId !== user.assignedNodeId ||
                    (reservation.requestedByUserId && reservation.requestedByUserId !== user.id)) {
                    return deny('Inference reservation is not authorized for this experiment actor');
                }
            }
        } else if (user.role !== UserRole.ADMIN && user.assignedNodeId !== nodeId) {
            return deny('Experiment actor is not authorized for this Runtime node');
        }
        return user;
    }

    static async owned(user: SafeAuthUser, experimentId: string, manager = ExperimentTable.getRepository().manager) {
        const experiment = await manager.findOneBy(ExperimentTable, { id: experimentId, ownerUserId: user.id });
        if (!experiment) return deny('Experiment is unavailable');
        return experiment;
    }

    private static async asset(manager: EntityManager, assetId: string, ownerId: string) {
        const asset = await manager.findOneBy(AssetVersionTable, { id: assetId, ownerUserId: ownerId });
        if (!asset) return deny('Asset version is unavailable');
        return asset;
    }

    static async prepare(input: PrepareExperimentInput) {
        if (!/^[A-Za-z0-9][A-Za-z0-9_.-]*$/.test(input.container)) deny('Invalid experiment container');
        const inferenceWorker = input.container === 'inference-worker' && input.spec.executionBackend === 'inference_worker';
        if (inferenceWorker && !['benchmark', 'publish', 'deploy'].includes(input.kind)) deny('Invalid inference worker task kind');
        if (input.resourceReservationId && !inferenceWorker) deny('Inference reservation requires a native inference worker task');
        const user = await this.runtimeActor(input.runtimeNodeId, input.actorUserId, input.resourceReservationId);
        const transfer = input.spec.executionBackend === 'container_transfer';
        let targetContainer: string | undefined;
        if (transfer) {
            if (input.kind !== 'publish') deny('Container transfer requires a publication task');
            const resolved = z.object({ sourceContainer: z.string(), destinationContainer: z.string().regex(/^[A-Za-z0-9][A-Za-z0-9_.-]*$/),
                destination: z.string().min(2) }).parse(input.spec.resolved);
            if (resolved.sourceContainer !== input.container || !resolved.destination.startsWith('/') ||
                resolved.destination.split('/').includes('..')) deny('Invalid publication execution locations');
            targetContainer = resolved.destinationContainer;
        }
        if (user.role !== UserRole.ADMIN && !inferenceWorker) {
            const group = await ResourceAccessService.getGroupForUser(user.id);
            const allowed = group ? [group.defaultContainerName, group.defaultEvaluateContainerName,
                group.defaultGrpoContainerName, group.defaultMultinodeContainerName] : [];
            if (!allowed.includes(input.container) || (targetContainer && !allowed.includes(targetContainer))) {
                deny('Experiment container is not assigned to this user');
            }
        }
        const spec = redactExperimentValue(input.spec);
        const requestHash = digest({ ...input, spec });
        return transaction(async (manager) => {
            const existing = await manager.findOneBy(TaskAttemptTable, {
                nodeId: input.runtimeNodeId, requestKey: input.requestKey,
            });
            if (existing) {
                const task = await manager.findOneByOrFail(ExperimentTaskTable, { id: existing.taskId });
                await this.owned(user, task.experimentId, manager);
                if (existing.requestHash !== requestHash) conflict('Request key was already used with different inputs');
                return { experimentId: task.experimentId, taskId: task.id, attemptId: existing.id,
                    attemptNumber: existing.attemptNumber, status: existing.status, reused: true,
                    logCaptureEnabled: existing.runtime.logCaptureVersion === 1 };
            }
            const timestamp = now();
            let experiment: ExperimentTable;
            if (input.experimentId) {
                experiment = await this.owned(user, input.experimentId, manager);
                if (input.workflowId && experiment.workflowId !== input.workflowId) conflict('Workflow does not match the experiment');
            } else {
                experiment = manager.create(ExperimentTable, {
                    id: id('exp'), name: input.name, project: input.project, ownerUserId: user.id,
                    groupId: user.group?.id ?? null, conversationRunId: input.conversationRunId ?? null,
                    workflowId: input.workflowId ?? null, parentExperimentId: null, status: 'prepared',
                    spec, createdAt: timestamp, updatedAt: timestamp,
                });
                await manager.save(experiment);
            }
            let task: ExperimentTaskTable;
            if (input.taskId) {
                task = await manager.findOneByOrFail(ExperimentTaskTable, { id: input.taskId, experimentId: experiment.id });
                if (task.kind !== input.kind || digest(proposal(task.spec)) !== digest(proposal(spec))) {
                    conflict('Retry must preserve the task specification; clone the experiment to change parameters');
                }
                if (!terminal.has(task.status)) conflict('Previous execution is not terminal; reconcile it before retrying');
            } else {
                // An experiment owns one fixed training proposal. Additional evaluation/merge tasks are allowed.
                if (input.kind === 'train' && await manager.countBy(ExperimentTaskTable, { experimentId: experiment.id, kind: 'train' })) {
                    conflict('Use taskId to retry training or clone the experiment for another training proposal');
                }
                if (experiment.status === 'draft') {
                    const normalizedInputs = (value: unknown) => (Array.isArray(value) ? value : []).map(canonicalJson).sort();
                    if (digest(normalizedInputs(experiment.spec.inputVersions)) !== digest(normalizedInputs(spec.inputVersions))) {
                        conflict('Draft input versions differ from the submitted versions');
                    }
                    experiment.spec = spec; // Freeze the resolved launch proposal when the draft first executes.
                }
                task = manager.create(ExperimentTaskTable, {
                    id: id('task'), experimentId: experiment.id, kind: input.kind, status: 'prepared', spec,
                    createdAt: timestamp, updatedAt: timestamp,
                });
                await manager.save(task);
            }
            for (const item of input.inputs) {
                const asset = await this.asset(manager, item.assetVersionId, user.id);
                if (asset.integrity !== 'verified') conflict('Training inputs must have verified immutable versions');
            }
            if (input.resumeFromVersionId) {
                const checkpoint = await this.asset(manager, input.resumeFromVersionId, user.id);
                if (checkpoint.kind !== 'checkpoint' || checkpoint.integrity !== 'verified') {
                    conflict('Resume requires a verified checkpoint version');
                }
            }
            const attemptNumber = (await manager.countBy(TaskAttemptTable, { taskId: task.id })) + 1;
            const attempt = manager.create(TaskAttemptTable, {
                id: id('attempt'), taskId: task.id, nodeId: input.runtimeNodeId, container: input.container,
                requestKey: input.requestKey, requestHash, attemptNumber, status: 'prepared', lastSequence: 0,
                resumeFromVersionId: input.resumeFromVersionId ?? null,
                resourceReservationId: input.resourceReservationId ?? null,
                configuration: { resolved: spec, completeness: 'incomplete' }, environment: {}, runtime: { logCaptureVersion: 1 },
                error: null, startedAt: null, endedAt: null, createdAt: timestamp, updatedAt: timestamp,
            });
            await manager.save(attempt);
            const inputs = [...input.inputs, ...(input.resumeFromVersionId
                ? [{ assetVersionId: input.resumeFromVersionId, role: 'resume_checkpoint' }] : [])];
            for (const item of inputs) {
                await this.linkAsset(manager, task.id, attempt.id, item.assetVersionId, 'input', item.role);
            }
            task.status = 'prepared'; task.updatedAt = timestamp;
            await manager.save(task);
            await this.refreshStatus(manager, experiment);
            return { experimentId: experiment.id, taskId: task.id, attemptId: attempt.id,
                attemptNumber, status: attempt.status, reused: false, logCaptureEnabled: true };
        });
    }

    private static async refreshStatus(manager: EntityManager, experiment: ExperimentTable) {
        const tasks = await manager.findBy(ExperimentTaskTable, { experimentId: experiment.id });
        const workflow = experiment.workflowId ? await manager.createQueryBuilder(ExperimentEventTable, 'event')
            .innerJoin(TaskAttemptTable, 'workflowAttempt', 'workflowAttempt.id = event.attemptId')
            .innerJoin(ExperimentTaskTable, 'workflowTask', 'workflowTask.id = workflowAttempt.taskId')
            .where('workflowTask.experimentId = :experimentId AND event.type = :type', {
                experimentId: experiment.id, type: 'workflow_status',
            })
            // Different stages/nodes may flush older snapshots after newer ones.
            .orderBy("CAST(json_extract(event.payload, '$.updatedAt') AS REAL)", 'DESC')
            .addOrderBy('event.sequence', 'DESC').getOne() : null;
        experiment.status = experimentStatus(tasks, experiment.workflowId, workflow?.payload.status as string | undefined);
        experiment.updatedAt = now();
        await manager.save(experiment);
    }

    private static async linkAsset(manager: EntityManager, taskId: string, attemptId: string,
        assetVersionId: string, direction: string, role: string, sourcePath?: string) {
        const relation = { attemptId, assetVersionId, direction, role };
        const existing = await manager.findOneBy(TaskAssetTable, relation);
        if (!existing) {
            await manager.save(manager.create(TaskAssetTable, { id: id('link'), taskId, ...relation,
                sourcePaths: sourcePath ? [sourcePath] : [], createdAt: now() }));
        } else if (sourcePath && !existing.sourcePaths.includes(sourcePath)) {
            existing.sourcePaths.push(sourcePath);
            await manager.save(existing);
        }
    }

    static async append(nodeId: string, events: ExperimentEventInput[]) {
        const acknowledged: string[] = [];
        for (const event of events) {
            await transaction(async (manager) => {
                const attempt = await manager.findOneBy(TaskAttemptTable, { id: event.attemptId, nodeId });
                if (!attempt) return deny('Execution is not assigned to this Runtime node');
                const payload = redactExperimentValue(event.payload);
                const payloadHash = digest({ ...event, payload });
                const prior = await manager.findOneBy(ExperimentEventTable, { id: event.eventId });
                if (prior) {
                    if (prior.nodeId !== nodeId || prior.payloadHash !== payloadHash) conflict('Event ID payload conflict');
                    return;
                }
                if (event.sequence !== attempt.lastSequence + 1) conflict(`Expected event sequence ${attempt.lastSequence + 1}`);
                const task = await manager.findOneByOrFail(ExperimentTaskTable, { id: attempt.taskId });
                const experiment = await manager.findOneByOrFail(ExperimentTable, { id: task.experimentId });
                if (event.type === 'workflow_status') {
                    const data = workflowStatusPayload.parse(payload);
                    if (!experiment.workflowId || experiment.workflowId !== data.workflowId) {
                        conflict('Workflow does not match the experiment');
                    }
                } else if (event.type === 'status') {
                    const data = statusPayload.parse(payload);
                    if (terminal.has(attempt.status) && data.status !== attempt.status) conflict('Terminal execution cannot change state');
                    const order: Record<string, number> = { prepared: 0, starting: 1, running: 2 };
                    if (data.status in order && attempt.status in order && order[data.status] < order[attempt.status]) {
                        conflict('Execution state cannot move backwards');
                    }
                    attempt.status = data.status;
                    attempt.runtime = { ...attempt.runtime, ...data.runtime };
                    if (data.error !== undefined) attempt.error = data.error;
                    else if (data.status === 'succeeded' || data.status === 'running') attempt.error = null;
                    if (data.status === 'running') attempt.startedAt ??= event.occurredAt;
                    if (terminal.has(data.status)) attempt.endedAt ??= event.occurredAt;
                } else if (event.type === 'service_operation') {
                    const data = serviceOperationSchema.parse(payload);
                    if (task.kind !== 'deploy' || task.spec.serviceInstanceId !== data.serviceInstanceId ||
                        attempt.runtime.serviceRunId !== data.serviceRunId ||
                        (attempt.resourceReservationId || null) !== data.reservationId) {
                        conflict('Service operation must match its deployment instance, run and reservation');
                    }
                    const previous = await manager.createQueryBuilder(ExperimentEventTable, 'event')
                        .where('event.attemptId = :attemptId AND event.type = :type', { attemptId: attempt.id, type: 'service_operation' })
                        .andWhere("json_extract(event.payload, '$.operationId') = :operationId", { operationId: data.operationId })
                        .orderBy('event.sequence', 'DESC').getOne();
                    if (previous && previous.payload.status !== 'requested') conflict('Completed service operation is immutable');
                    if (!previous && data.status !== 'requested') conflict('Service operation requires a recorded request');
                    if (previous && ['serviceInstanceId', 'serviceRunId', 'actorUserId', 'trigger', 'cause', 'reservationId']
                        .some(key => previous.payload[key] !== data[key as keyof typeof data])) {
                        conflict('Service operation identity is immutable');
                    }
                    // The deployment outcome, timestamps and error describe startup only.
                } else if (event.type === 'configuration') {
                    const data = configurationPayload.parse(payload);
                    if (attempt.configuration.completeness === 'complete' &&
                        digest(attempt.configuration) !== digest(data.configuration)) {
                        conflict('Final execution configuration is immutable');
                    }
                    attempt.configuration = data.configuration;
                    attempt.environment = data.environment;
                } else if (event.type === 'artifact') {
                    const data = assetRegistrationSchema.parse(payload);
                    if (data.integrity === 'verified' && (!data.fingerprint || !Array.isArray(data.manifest.files))) {
                        conflict('Verified asset requires a content fingerprint and file manifest');
                    }
                    if (data.integrity === 'verified') {
                        const files = z.array(z.object({ path: z.string().min(1), size: z.number().int().nonnegative(),
                            sha256: z.string().regex(/^[a-f0-9]{64}$/) }).strict()).min(1).parse(data.manifest.files);
                        if (files.some(file => file.path.startsWith('/') || file.path.includes('\\') ||
                            file.path.split('/').some(part => !part || part === '..' || part === '.')) ||
                            new Set(files.map(file => file.path)).size !== files.length || digest(files) !== data.fingerprint) {
                            conflict('Invalid asset content manifest or fingerprint');
                        }
                    }
                    const publicationTarget = task.kind === 'publish' && task.spec.executionBackend === 'container_transfer' &&
                        data.direction === 'output' && data.role === 'published_model' && data.kind === 'model' &&
                        data.location.container === (task.spec.resolved as Record<string, unknown>)?.destinationContainer;
                    if (data.location.container !== attempt.container && !publicationTarget) conflict('Artifact container must match its execution');
                    const values = {
                        ownerUserId: experiment.ownerUserId, nodeId, resourceId: data.resourceId ?? null,
                        kind: data.kind, logicalName: data.logicalName, fingerprint: data.fingerprint ?? null,
                        integrity: data.integrity, location: data.location, manifest: data.manifest, metadata: data.metadata,
                    };
                    const asset = await manager.findOneBy(AssetVersionTable, { id: data.id });
                    if (asset) {
                        // Identical content can be used under another display/source name.
                        // Preserve the first record; only its immutable identity must agree.
                        if (asset.ownerUserId !== values.ownerUserId || asset.nodeId !== nodeId ||
                            asset.kind !== data.kind || asset.fingerprint !== values.fingerprint ||
                            digest(asset.manifest) !== digest(values.manifest) ||
                            digest(asset.location) !== digest(values.location) || asset.integrity !== values.integrity) {
                            conflict('Asset version is immutable');
                        }
                    } else {
                        await manager.save(manager.create(AssetVersionTable, { id: data.id, ...values, createdAt: now() }));
                    }
                    await this.linkAsset(manager, task.id, attempt.id, data.id, data.direction, data.role,
                        typeof data.metadata.sourcePath === 'string' ? data.metadata.sourcePath : data.location.path);
                    if (data.direction === 'input' && data.role === 'resume_checkpoint') {
                        if (data.kind !== 'checkpoint' || data.integrity !== 'verified') conflict('Resume input must be a verified checkpoint');
                        if (attempt.resumeFromVersionId && attempt.resumeFromVersionId !== data.id) conflict('Resume version is immutable');
                        attempt.resumeFromVersionId = data.id;
                    }
                } else if (event.type === 'evaluation') {
                    if (!['evaluate', 'benchmark'].includes(task.kind)) conflict('Evaluation result requires an evaluation task');
                    const data = evaluationSchema.parse(payload);
                    if (data.expectedCount !== null &&
                        (data.succeededCount ?? 0) + (data.failedCount ?? 0) > data.expectedCount) conflict('Invalid evaluation sample counts');
                    const model = await this.asset(manager, data.modelVersionId, experiment.ownerUserId);
                    if (!['model', 'adapter', 'base_model', 'checkpoint'].includes(model.kind)) conflict('Invalid evaluated model kind');
                    if (data.protocolComplete && model.integrity !== 'verified') conflict('Complete evaluations require a verified model');
                    if (data.reportVersionId) {
                        const report = await this.asset(manager, data.reportVersionId, experiment.ownerUserId);
                        if (report.kind !== 'report') conflict('Invalid report kind');
                        await this.linkAsset(manager, task.id, attempt.id, report.id, 'output', 'report');
                    }
                    const datasetIds = [...new Set(data.metrics.map(metric => metric.datasetVersionId))].sort();
                    const datasetFingerprints: string[] = [];
                    for (const datasetId of datasetIds) {
                        const dataset = await this.asset(manager, datasetId, experiment.ownerUserId);
                        if (dataset.kind !== 'dataset') conflict('Metric dataset must reference a dataset version');
                        if (data.protocolComplete && dataset.integrity !== 'verified') conflict('Complete evaluations require verified datasets');
                        datasetFingerprints.push(dataset.fingerprint || `unverified:${dataset.id}`);
                        await this.linkAsset(manager, task.id, attempt.id, datasetId, 'input', 'evaluation_dataset');
                    }
                    await this.linkAsset(manager, task.id, attempt.id, model.id, 'input', 'evaluated_model');
                    const metricKeys = data.metrics.map(m => canonicalJson([m.datasetVersionId, m.subset, m.name]));
                    if (new Set(metricKeys).size !== metricKeys.length) conflict('Duplicate evaluation metric');
                    const processedCount = data.succeededCount !== null && data.failedCount !== null
                        ? data.succeededCount + data.failedCount : data.expectedCount;
                    if (processedCount !== null && data.metrics.some(m => m.sampleCount !== null && m.sampleCount > processedCount)) {
                        conflict('Metric sample count exceeds processed samples');
                    }
                    if (await manager.findOneBy(EvaluationResultTable, { id: data.id })) conflict('Evaluation result is immutable');
                    await manager.save(manager.create(EvaluationResultTable, {
                        id: data.id, taskId: task.id, attemptId: attempt.id, modelVersionId: data.modelVersionId,
                        reportVersionId: data.reportVersionId ?? null, protocol: data.protocol,
                        protocolComplete: data.protocolComplete,
                        comparisonKey: digest({ protocol: data.protocol, datasetFingerprints: datasetFingerprints.sort() }),
                        expectedCount: data.expectedCount, succeededCount: data.succeededCount, failedCount: data.failedCount,
                        createdAt: now(),
                    }));
                    for (const metric of data.metrics) {
                        await manager.save(manager.create(EvaluationMetricTable, { id: id('metric'), resultId: data.id,
                            ...metric, createdAt: now() }));
                    }
                }
                attempt.lastSequence = event.sequence; attempt.updatedAt = now();
                await manager.save(attempt);
                const latest = await manager.findOne(TaskAttemptTable, { where: { taskId: task.id }, order: { attemptNumber: 'DESC' } });
                task.status = latest!.status; task.updatedAt = now();
                await manager.save(task);
                await manager.save(manager.create(ExperimentEventTable, {
                    id: event.eventId, attemptId: attempt.id, nodeId, sequence: event.sequence, type: event.type,
                    occurredAt: event.occurredAt, payloadHash, payload, createdAt: now(),
                }));
                await this.refreshStatus(manager, experiment);
            });
            acknowledged.push(event.eventId);
        }
        return { acknowledged };
    }

    static async appendLog(input: AppendExperimentLogInput) {
        appendExperimentLogSchema.parse(input);
        return transaction(async manager => {
            const attempt = await manager.findOneBy(TaskAttemptTable, { id: input.attemptId, nodeId: input.runtimeNodeId });
            if (!attempt) return deny('Execution is not assigned to this Runtime node');
            // The flag is written only for executions prepared after this feature ships.
            if (attempt.runtime.logCaptureVersion !== 1) return deny('Log capture is unavailable for this execution');
            let source = await manager.findOneBy(ExperimentLogSourceTable, { id: input.source.id });
            if (source && (source.attemptId !== attempt.id || source.nodeId !== input.runtimeNodeId ||
                source.label !== input.source.label || source.kind !== input.source.kind)) {
                conflict('Log source identity conflict');
            }
            if (!source) {
                source = manager.create(ExperimentLogSourceTable, { id: input.source.id, attemptId: attempt.id,
                    nodeId: input.runtimeNodeId, label: input.source.label, kind: input.source.kind,
                    status: 'streaming', error: null, nextOffset: 0, createdAt: now(), updatedAt: now() });
            }
            for (const chunk of input.chunks) {
                if (chunk.endOffset <= chunk.startOffset || Buffer.byteLength(chunk.content, 'utf8') > 65536) {
                    conflict('Invalid log chunk size or offset');
                }
                const content = redactExperimentValue(chunk.content) as string;
                const contentHash = digest({ endOffset: chunk.endOffset, content });
                const existing = await manager.findOneBy(ExperimentLogChunkTable, {
                    sourceId: source.id, startOffset: chunk.startOffset,
                });
                if (existing) {
                    if (existing.endOffset !== chunk.endOffset || existing.contentHash !== contentHash) {
                        conflict('Log chunk payload conflict');
                    }
                    continue;
                }
                if (chunk.startOffset !== source.nextOffset) conflict(`Expected log offset ${source.nextOffset}`);
                await manager.save(manager.create(ExperimentLogChunkTable, {
                    id: id('log'), sourceId: source.id, attemptId: attempt.id,
                    startOffset: chunk.startOffset, endOffset: chunk.endOffset,
                    contentHash, content, createdAt: now(),
                }));
                source.nextOffset = chunk.endOffset;
            }
            if (input.source.status === 'complete' || source.status !== 'complete') {
                source.status = input.source.status;
                source.error = input.source.error ? String(redactExperimentValue(input.source.error)) : null;
            }
            source.updatedAt = now();
            await manager.save(source);
            return { sourceId: source.id, nextOffset: source.nextOffset };
        });
    }

    private static async ownedAttempt(user: SafeAuthUser, experimentId: string, attemptId: string) {
        await this.owned(user, experimentId);
        const attempt = await TaskAttemptTable.findOneBy({ id: attemptId });
        const task = attempt && await ExperimentTaskTable.findOneBy({ id: attempt.taskId, experimentId });
        if (!task) return deny('Execution is unavailable');
        return attempt!;
    }

    static async logSources(user: SafeAuthUser, experimentId: string, attemptId: string) {
        await this.ownedAttempt(user, experimentId, attemptId);
        return ExperimentLogSourceTable.find({ where: { attemptId }, order: { createdAt: 'ASC', id: 'ASC' } });
    }

    static async businessLogSource(user: SafeAuthUser, experimentId: string, attemptId: string, sourceId: string) {
        await this.ownedAttempt(user, experimentId, attemptId);
        const source = await ExperimentLogSourceTable.findOneBy({ id: sourceId, attemptId });
        if (!source || ['execution', 'operation'].includes(source.kind)) return deny('Business log source is unavailable');
        return source;
    }

    static async *archivedLogChunks(sourceId: string, snapshotEndOffset: number, startOffset = 0) {
        let next = startOffset;
        while (true) {
            const chunks = await ExperimentLogChunkTable.find({
                where: { sourceId, startOffset: MoreThanOrEqual(next), endOffset: LessThanOrEqual(snapshotEndOffset) },
                order: { startOffset: 'ASC' }, take: 16,
            });
            if (!chunks.length) return;
            for (const chunk of chunks) yield chunk;
            next = chunks[chunks.length - 1].endOffset;
        }
    }

    static async searchLogs(user: SafeAuthUser, input: z.infer<typeof searchExperimentLogsSchema>) {
        const source = await this.businessLogSource(user, input.experimentId, input.attemptId, input.sourceId);
        const snapshotEndOffset = Math.min(input.snapshotEndOffset ?? source.nextOffset, source.nextOffset);
        if (input.cursor) {
            const block = await ExperimentLogChunkTable.findOneBy({ sourceId: source.id, startOffset: input.cursor.startOffset });
            if (!block || block.endOffset > snapshotEndOffset || input.cursor.characterOffset > block.content.length)
                throw new TRPCError({ code: 'BAD_REQUEST', message: 'Invalid log cursor' });
        }
        const page = await searchArchivedLogs(this.archivedLogChunks(source.id, snapshotEndOffset, input.cursor?.startOffset), {
            keywords: input.suspectedErrors ? suspectedLogErrors : [input.keyword],
            caseSensitive: input.caseSensitive, cursor: input.cursor,
        });
        return { ...page, snapshotEndOffset };
    }

    static async logContext(user: SafeAuthUser, input: z.infer<typeof experimentLogContextSchema>) {
        const source = await this.businessLogSource(user, input.experimentId, input.attemptId, input.sourceId);
        const snapshot = Math.min(input.snapshotEndOffset, source.nextOffset);
        const target = await ExperimentLogChunkTable.findOneBy({ sourceId: source.id, startOffset: input.position.startOffset });
        if (!target || target.endOffset > snapshot || input.position.characterOffset >= target.content.length)
            throw new TRPCError({ code: 'BAD_REQUEST', message: 'Log position is unavailable' });
        const preceding = await ExperimentLogChunkTable.find({ where: { sourceId: source.id, startOffset: LessThan(target.startOffset) },
            order: { startOffset: 'DESC' }, take: 2 });
        const chunks = [...preceding.reverse(), target];
        let remaining = input.matchLength + 200 - (target.content.length - input.position.characterOffset);
        if (remaining > 0) for await (const block of this.archivedLogChunks(source.id, snapshot, target.endOffset)) {
            chunks.push(block); remaining -= block.content.length;
            if (remaining <= 0 || chunks.length >= 256) break;
        }
        return { chunks, position: input.position, matchLength: input.matchLength,
            latestOffset: source.nextOffset, status: source.status };
    }

    static async updateMetadata(user: SafeAuthUser, input: z.infer<typeof experimentMetadataSchema>) {
        const values = experimentMetadataSchema.parse(input);
        return transaction(async manager => {
            await this.owned(user, values.experimentId, manager);
            // Serialize with runtime events, which also save the experiment entity.
            await manager.update(ExperimentTable, { id: values.experimentId, ownerUserId: user.id },
                { notes: values.notes, tags: values.tags });
            return { notes: values.notes, tags: values.tags };
        });
    }

    static async logChunks(user: SafeAuthUser, experimentId: string, attemptId: string, sourceId: string,
        beforeOffset?: number, afterOffset?: number) {
        await this.ownedAttempt(user, experimentId, attemptId);
        const source = await ExperimentLogSourceTable.findOneBy({ id: sourceId, attemptId });
        if (!source) return deny('Log source is unavailable');
        const forward = afterOffset !== undefined;
        const chunks = await ExperimentLogChunkTable.find({
            where: { sourceId, ...(forward ? { startOffset: MoreThanOrEqual(afterOffset!) }
                : beforeOffset !== undefined ? { startOffset: LessThan(beforeOffset) } : {}) },
            order: { startOffset: forward ? 'ASC' : 'DESC' }, take: 8,
        });
        if (!forward) chunks.reverse();
        return { chunks, nextBefore: chunks[0]?.startOffset ?? null, latestOffset: source.nextOffset,
            status: source.status, error: source.error };
    }

    static async list(user: SafeAuthUser, input: ListExperimentsInput) {
        const query = applyExperimentListFilters(ExperimentTable.createQueryBuilder('experiment'), user.id, input);
        const [items, total] = await query.orderBy('experiment.createdAt', 'DESC').addOrderBy('experiment.id', 'DESC')
            .skip(input.offset).take(input.limit).getManyAndCount();
        return { items, total };
    }

    private static async equivalentAssets(user: SafeAuthUser, assets: AssetVersionTable[], context: ReadContext) {
        const fingerprints = [...new Set(assets.filter(asset => assetContentKey(asset)).map(asset => asset.fingerprint!))];
        const missing = fingerprints.filter(fingerprint => !context.fingerprints.has(fingerprint));
        if (missing.length) {
            const matches = await AssetVersionTable.findBy({ ownerUserId: user.id,
                integrity: 'verified', kind: In([...modelAssetKinds, 'dataset']), fingerprint: In(missing) });
            for (const fingerprint of missing) context.fingerprints.set(fingerprint, []);
            for (const asset of matches) {
                context.fingerprints.get(asset.fingerprint!)!.push(asset);
                context.assets.set(asset.id, asset);
            }
        }
        const matches = fingerprints.flatMap(fingerprint => context.fingerprints.get(fingerprint)!);
        for (const asset of assets) context.assets.set(asset.id, asset);
        return equivalentModelVersions([...new Map([...assets, ...matches].map(asset => [asset.id, asset])).values()]);
    }

    private static async ownedAssetLinks(user: SafeAuthUser, versionIds: string[], context: ReadContext) {
        if (!versionIds.length) return [];
        const missing = [...new Set(versionIds)].filter(versionId => !context.links.has(versionId));
        if (!missing.length) return [...new Set(versionIds)].flatMap(versionId => context.links.get(versionId)!);
        const links = await TaskAssetTable.findBy({ assetVersionId: In(missing) });
        const tasks = links.length ? await ExperimentTaskTable.findBy({ id: In([...new Set(links.map(link => link.taskId))]) }) : [];
        const records = tasks.length ? await ExperimentTable.findBy({ ownerUserId: user.id,
            id: In([...new Set(tasks.map(task => task.experimentId))]) }) : [];
        const recordMap = new Map(records.map(record => [record.id, record]));
        const taskMap = new Map(tasks.map(task => [task.id, task]));
        for (const versionId of missing) context.links.set(versionId, []);
        for (const link of links) {
            const task = taskMap.get(link.taskId);
            const record = task && recordMap.get(task.experimentId);
            if (task && record) context.links.get(link.assetVersionId)!.push({ link, task, record });
        }
        return [...new Set(versionIds)].flatMap(versionId => context.links.get(versionId)!);
    }

    private static async associationPeers(attemptIds: string[], direction: 'input' | 'output', context: ReadContext) {
        const missing = attemptIds.filter(attemptId => !context.peers.has(`${attemptId}:${direction}`));
        if (missing.length) {
            const peers = await TaskAssetTable.findBy({ attemptId: In(missing), direction });
            for (const attemptId of missing) context.peers.set(`${attemptId}:${direction}`, []);
            for (const peer of peers) context.peers.get(`${peer.attemptId}:${direction}`)!.push(peer);
        }
        return attemptIds.flatMap(attemptId => context.peers.get(`${attemptId}:${direction}`)!);
    }

    private static async associationAssets(user: SafeAuthUser, versionIds: string[], context: ReadContext) {
        const missing = [...new Set(versionIds)].filter(versionId => !context.assets.has(versionId));
        if (missing.length) {
            for (const asset of await AssetVersionTable.findBy({ ownerUserId: user.id, id: In(missing) })) {
                context.assets.set(asset.id, asset);
            }
        }
        return [...new Set(versionIds)].flatMap(versionId => {
            const asset = context.assets.get(versionId);
            return asset ? [asset] : [];
        });
    }

    private static async walkAssets(user: SafeAuthUser, roots: AssetVersionTable[], direction: 'upstream' | 'downstream',
        scope: AssociationScope, context: ReadContext) {
        const found = new Map<string, AssetVersionTable>();
        let pending = roots;
        while (pending.length) {
            const equivalents = await this.equivalentAssets(user, pending, context);
            const frontier = [...new Map(pending.flatMap(asset => equivalents.get(asset.id) || [asset])
                .map(asset => [asset.id, asset])).values()].filter(asset => !found.has(asset.id));
            if (!frontier.length) break;
            frontier.forEach(asset => found.set(asset.id, asset));
            const relations = (await this.ownedAssetLinks(user, frontier.map(asset => asset.id), context))
                .filter(({ link }) => link.direction === (direction === 'upstream' ? 'output' : 'input'));
            const attemptIds = [...new Set(relations.map(({ link }) => link.attemptId))];
            const peers = await this.associationPeers(attemptIds, direction === 'upstream' ? 'input' : 'output', context);
            const peerAssets = await this.associationAssets(user, peers.map(link => link.assetVersionId), context);
            const peerMap = new Map(peerAssets.map(asset => [asset.id, asset]));
            const peersByAttempt = new Map<string, TaskAssetTable[]>();
            for (const peer of peers) {
                const key = `${peer.attemptId}:${peer.taskId}`;
                const group = peersByAttempt.get(key) || [];
                group.push(peer);
                peersByAttempt.set(key, group);
            }
            const next = new Map<string, AssetVersionTable>();
            for (const { link, task } of relations) for (const peer of peersByAttempt.get(`${link.attemptId}:${link.taskId}`) || []) {
                const current = found.get(link.assetVersionId)!;
                const adjacent = peerMap.get(peer.assetVersionId);
                if (!adjacent || !inAssociationScope(adjacent, scope)) continue;
                const output = direction === 'upstream' ? current : adjacent;
                const input = direction === 'upstream' ? adjacent : current;
                if (!Array.isArray(output.metadata.inputVersionIds) || !output.metadata.inputVersionIds.includes(input.id)) continue;
                // Never traverse forwards through a shared base model into unrelated training.
                // Dataset transformations and explicit model merge/publication steps are directional.
                if (direction === 'downstream' && !(task.kind === 'preprocess' && input.kind === 'dataset' && output.kind === 'dataset') &&
                    !(['merge', 'publish', 'evaluate'].includes(task.kind) && modelAssetKinds.includes(input.kind) &&
                        modelAssetKinds.includes(output.kind) && (task.kind !== 'evaluate' || link.role === 'evaluated_model'))) continue;
                if (!found.has(adjacent.id)) next.set(adjacent.id, adjacent);
            }
            pending = [...next.values()];
        }
        return found;
    }

    private static async assetRelations(user: SafeAuthUser, assets: AssetVersionTable[], scope: AssociationScope = 'all',
        context: ReadContext = readContext()) {
        const rows: Array<{ assetVersionId: string; relatedAssetVersionId: string; match: string;
            experimentId: string; name: string; taskId: string; attemptId: string; direction: string; role: string;
            taskKind: string; relatedAssetKind: string }> = [];
        const roots = assets.filter(asset => inAssociationScope(asset, scope));
        const equivalents = await this.equivalentAssets(user, roots, context);
        await this.ownedAssetLinks(user, [...new Set(roots.flatMap(asset =>
            (equivalents.get(asset.id) || [asset]).map(item => item.id)))], context);
        for (const asset of roots) {
            const direct = new Set((equivalents.get(asset.id) || [asset]).map(item => item.id));
            const upstream = await this.walkAssets(user, [asset], 'upstream', scope, context);
            const downstream = await this.walkAssets(user, [asset], 'downstream', scope, context);
            const ids = [...new Set([...upstream.keys(), ...downstream.keys()])];
            for (const { link, record, task } of await this.ownedAssetLinks(user, ids, context)) {
                const match = direct.has(link.assetVersionId) ? (asset.id === link.assetVersionId ? 'version' : 'content')
                    : downstream.has(link.assetVersionId) ? 'downstream' : 'upstream';
                // Upstream consumers include sibling runs; only expose actual producers.
                if (match === 'upstream' && link.direction !== 'output') continue;
                rows.push({ assetVersionId: asset.id, relatedAssetVersionId: link.assetVersionId, match,
                    experimentId: record.id, name: record.name, taskId: link.taskId, attemptId: link.attemptId,
                    direction: link.direction, role: link.role, taskKind: task.kind,
                    relatedAssetKind: (upstream.get(link.assetVersionId) || downstream.get(link.assetVersionId))!.kind });
            }
        }
        return rows;
    }

    private static async readDetail(user: SafeAuthUser, experimentId: string, context: ReadContext, includeLineage = true, includeConfiguration = true) {
        const experiment = await this.owned(user, experimentId);
        const tasks = await ExperimentTaskTable.find({ where: { experimentId }, order: { createdAt: 'ASC' },
            ...(!includeConfiguration ? { select: ['id', 'experimentId', 'kind', 'status', 'createdAt', 'updatedAt'] as Array<keyof ExperimentTaskTable> } : {}) });
        const taskIds = tasks.map(task => task.id);
        const attempts = taskIds.length ? await TaskAttemptTable.find({ where: { taskId: In(taskIds) }, order: { createdAt: 'ASC' },
            ...(!includeConfiguration ? { select: ['id', 'taskId', 'nodeId', 'container', 'attemptNumber', 'status', 'lastSequence', 'error',
                'resumeFromVersionId', 'resourceReservationId', 'startedAt', 'endedAt', 'createdAt', 'updatedAt', 'runtime'] as Array<keyof TaskAttemptTable> } : {}) }) : [];
        if (!includeConfiguration) {
            for (const task of tasks) task.spec = {};
            for (const attempt of attempts) { attempt.configuration = {}; attempt.environment = {}; }
        }
        const links = taskIds.length ? await TaskAssetTable.findBy({ taskId: In(taskIds) }) : [];
        const draftInputs = experiment.status === 'draft' && Array.isArray(experiment.spec.inputVersions)
            ? experiment.spec.inputVersions.map((item: any) => String(item.assetVersionId)) : [];
        const assetIds = [...new Set([...links.map(link => link.assetVersionId), ...draftInputs])];
        const assets = assetIds.length ? await AssetVersionTable.findBy({ id: In(assetIds), ownerUserId: user.id }) : [];
        const evaluations = taskIds.length ? await EvaluationResultTable.findBy({ taskId: In(taskIds) }) : [];
        const resultIds = evaluations.map(result => result.id);
        const metrics = resultIds.length ? await EvaluationMetricTable.findBy({ resultId: In(resultIds) }) : [];
        const lineage = includeLineage ? await this.assetRelations(user, assets, 'all', context) : [];
        return { experiment, tasks, attempts, links, assets, evaluations, metrics, lineage,
            serviceOperations: await this.serviceOperations(attempts) };
    }

    static async detail(user: SafeAuthUser, experimentId: string) {
        const context = readContext();
        const detail = await this.readDetail(user, experimentId, context);
        const { linkedEvaluations } = await this.trainingEvaluations(user, [detail], context);
        return { ...detail, linkedEvaluations };
    }

    private static async serviceOperations(attempts: TaskAttemptTable[]): Promise<ServiceOperationHistory[]> {
        if (!attempts.length) return [];
        const events = await ExperimentEventTable.find({ where: { attemptId: In(attempts.map(attempt => attempt.id)),
            type: 'service_operation' }, order: { sequence: 'ASC' } });
        const operations = new Map<string, ServiceOperationHistory>();
        for (const event of events) {
            const payload = serviceOperationSchema.parse(event.payload);
            const key = `${event.attemptId}:${payload.operationId}`;
            const previous = operations.get(key);
            operations.set(key, { ...payload, id: key, attemptId: event.attemptId,
                requestedAt: previous?.requestedAt || event.occurredAt,
                completedAt: payload.status === 'requested' ? null : event.occurredAt,
                reservationStatus: null, releasedAt: null });
        }
        const reservationIds = [...new Set([...operations.values()].map(item => item.reservationId).filter((id): id is string => Boolean(id)))];
        const reservations = reservationIds.length ? await TrainingReservationTable.find({ where: { id: In(reservationIds) },
            select: ['id', 'status', 'releasedAt'] }) : [];
        const reservationMap = new Map(reservations.map(item => [item.id, item]));
        return [...operations.values()].map(item => ({ ...item,
            reservationStatus: item.reservationId ? reservationMap.get(item.reservationId)?.status || null : null,
            releasedAt: item.reservationId ? reservationMap.get(item.reservationId)?.releasedAt || null : null,
        })).sort((a, b) => b.requestedAt.localeCompare(a.requestedAt) || a.id.localeCompare(b.id));
    }

    static async summary(user: SafeAuthUser, experimentId: string) {
        const experiment = await this.owned(user, experimentId);
        const spec = Object.fromEntries(['kind', 'script', 'operation', 'filename'].filter(key => key in experiment.spec)
            .map(key => [key, experiment.spec[key]]));
        const tasks = await ExperimentTaskTable.find({ where: { experimentId },
            select: ['id', 'experimentId', 'kind', 'status', 'createdAt', 'updatedAt'], order: { createdAt: 'ASC' } });
        const taskIds = tasks.map(task => task.id);
        const attempts = taskIds.length ? await TaskAttemptTable.find({ where: { taskId: In(taskIds) },
            select: ['id', 'taskId', 'nodeId', 'container', 'attemptNumber', 'status', 'lastSequence', 'error',
                'resumeFromVersionId', 'resourceReservationId', 'startedAt', 'endedAt', 'createdAt', 'updatedAt'],
            order: { createdAt: 'ASC' } }) : [];
        const runtimeKeys = ['archiveStatus', 'archiveError', 'archiveWarning', 'exitCode', 'signals', 'scriptName',
            'logCaptureVersion', 'inputIntegrityAfterExecution', 'uploadedVersionId', 'upload', 'dataProcessing'];
        const annotations = taskIds.length ? await TaskAttemptTable.createQueryBuilder('a').select('a.id', 'id')
            .addSelect("json_extract(a.configuration, '$.completeness')", 'completeness')
            .addSelect(`json_object(${runtimeKeys.map(key => `'${key}', json_extract(a.runtime, '$.${key}')`).join(', ')})`, 'runtime')
            .where('a.taskId IN (:...taskIds)', { taskIds }).getRawMany<{ id: string; completeness: string | null; runtime: string }>() : [];
        const annotationMap = new Map(annotations.map(item => [item.id, item]));
        for (const attempt of attempts) {
            const annotation = annotationMap.get(attempt.id);
            attempt.configuration = { completeness: annotation?.completeness ?? 'incomplete' };
            attempt.environment = {};
            attempt.runtime = annotation ? JSON.parse(annotation.runtime) : {};
        }
        const links = taskIds.length ? await TaskAssetTable.find({ where: { taskId: In(taskIds) },
            select: ['id', 'taskId', 'attemptId', 'assetVersionId', 'direction', 'role'] }) : [];
        const ids = [...new Set(links.map(link => link.assetVersionId))];
        const assets = ids.length ? await AssetVersionTable.find({ where: { id: In(ids), ownerUserId: user.id },
            select: ['id', 'kind', 'logicalName', 'integrity', 'fingerprint'] }) : [];
        for (const asset of assets) { asset.manifest = {}; asset.metadata = {}; asset.location = { container: '', path: '' }; }
        return { experiment: { ...experiment, spec }, tasks: tasks.map(task => ({ ...task, spec: {} as Record<string, unknown> })),
            attempts, links, assets, serviceOperations: await this.serviceOperations(attempts),
            evaluations: [] as EvaluationResultTable[], metrics: [] as EvaluationMetricTable[],
            lineage: [] as Awaited<ReturnType<typeof ExperimentService.assetRelations>>,
            linkedEvaluations: [] as Awaited<ReturnType<typeof ExperimentService.trainingEvaluations>>['linkedEvaluations'] };
    }

    static async configuration(user: SafeAuthUser, experimentId: string) {
        const experiment = await this.owned(user, experimentId);
        const tasks = await ExperimentTaskTable.find({ where: { experimentId }, order: { createdAt: 'ASC' } });
        const attempts = tasks.length ? await TaskAttemptTable.find({ where: { taskId: In(tasks.map(task => task.id)) },
            order: { createdAt: 'ASC' } }) : [];
        return { spec: experiment.spec, tasks, attempts };
    }

    static async assets(user: SafeAuthUser, experimentId: string) {
        const experiment = await this.owned(user, experimentId);
        const tasks = await ExperimentTaskTable.find({ where: { experimentId }, select: ['id'] });
        const taskIds = tasks.map(task => task.id);
        const links = taskIds.length ? await TaskAssetTable.findBy({ taskId: In(taskIds) }) : [];
        const draftInputs = experiment.status === 'draft' && Array.isArray(experiment.spec.inputVersions)
            ? experiment.spec.inputVersions.map((item: any) => String(item.assetVersionId)) : [];
        const ids = [...new Set([...links.map(link => link.assetVersionId), ...draftInputs])];
        const assets = ids.length ? await AssetVersionTable.findBy({ id: In(ids), ownerUserId: user.id }) : [];
        const observations = taskIds.length ? await TaskAttemptTable.createQueryBuilder('a').select('a.id', 'id')
            .addSelect("CASE WHEN json_type(a.runtime, '$.evaluationObservation') = 'object' THEN json_extract(a.runtime, '$.evaluationObservation') ELSE NULL END", 'observation')
            .where('a.taskId IN (:...taskIds)', { taskIds }).getRawMany<{ id: string; observation: string | null }>() : [];
        return { assets, links, lineage: await this.assetRelations(user, assets, 'all', readContext()),
            observations: observations.map(item => ({ id: item.id,
                evaluationObservation: item.observation ? JSON.parse(item.observation) : undefined })) };
    }

    static async evaluations(user: SafeAuthUser, experimentId: string) {
        const context = readContext();
        const detail = await this.readDetail(user, experimentId, context, false, false);
        const { linkedEvaluations } = await this.trainingEvaluations(user, [detail], context, false);
        const compact = (asset: AssetVersionTable) => ({ ...asset,
            manifest: { shardSummary: modelShardSummary(asset.manifest) } as Record<string, unknown> });
        return { evaluations: detail.evaluations, metrics: detail.metrics, assets: detail.assets.map(compact),
            attempts: detail.attempts.map(attempt => ({ ...attempt, configuration: {}, environment: {} })),
            linkedEvaluations: linkedEvaluations.map(item => ({ ...item, model: compact(item.model), datasets: item.datasets.map(compact),
                evaluationAttempt: { ...item.evaluationAttempt, configuration: {}, environment: {} } })) };
    }

    static async compare(user: SafeAuthUser, experimentIds: string[], lightweight = false) {
        const context = readContext();
        const details: Array<Awaited<ReturnType<typeof ExperimentService.readDetail>>> = [];
        for (const experimentId of experimentIds) details.push(await this.readDetail(user, experimentId, context, !lightweight));
        const groups: Record<string, Array<{ experimentId: string; result: EvaluationResultTable; metrics: EvaluationMetricTable[];
            ineligibleReason: string | null }>> = {};
        for (const detail of details) for (const result of detail.evaluations) {
            // Results may arrive before final archival, or survive a failed/cancelled
            // execution. Only the result's own successfully verified attempt qualifies.
            const attempt = detail.attempts.find(item => item.id === result.attemptId);
            const ineligibleReason = comparisonIneligibility(attempt, result);
            const complete = ineligibleReason === null;
            const key = complete ? result.comparisonKey : `incomplete:${result.id}`;
            (groups[key] ||= []).push({ experimentId: detail.experiment.id, result, ineligibleReason,
                metrics: detail.metrics.filter(metric => metric.resultId === result.id) });
        }
        const linked = await this.trainingEvaluations(user, details, context, !lightweight);
        if (lightweight) {
            // Equivalence checks need manifests internally; omit them only after associations are resolved.
            for (const detail of details) for (const asset of detail.assets) asset.manifest = {};
            for (const item of linked.linkedEvaluations) {
                item.model.manifest = {};
                for (const asset of item.datasets) asset.manifest = {};
            }
            for (const run of linked.trainingRuns) for (const item of [...run.inputs, ...run.outputs]) item.asset.manifest = {};
        }
        return { details, groups, ...linked };
    }

    private static async trainingEvaluations(user: SafeAuthUser,
        details: Array<Awaited<ReturnType<typeof ExperimentService.readDetail>>>, context: ReadContext, includeConfiguration = true) {
        const modelKinds = new Set(modelAssetKinds);
        const trainingRuns = details.flatMap(detail => {
            const trainingTasks = new Set(detail.tasks.filter(task => task.kind === 'train').map(task => task.id));
            return detail.attempts.filter(attempt => trainingTasks.has(attempt.taskId)).map(attempt => {
                const inputs = detail.links.filter(link => link.attemptId === attempt.id && link.direction === 'input')
                    .map(link => ({ link, asset: detail.assets.find(asset => asset.id === link.assetVersionId) }))
                    .filter((item): item is { link: TaskAssetTable; asset: AssetVersionTable } => Boolean(item.asset));
                const outputs = detail.links.filter(link => link.attemptId === attempt.id && link.direction === 'output')
                    .map(link => ({ link, asset: detail.assets.find(asset => asset.id === link.assetVersionId) }))
                    .filter((item): item is { link: TaskAssetTable; asset: AssetVersionTable } => Boolean(item.asset));
                return { experimentId: detail.experiment.id, attempt, inputs, outputs };
            });
        });
        const origins = new Map<string, Set<string>>();
        for (const run of trainingRuns) for (const output of run.outputs) {
            if (!modelKinds.has(output.asset.kind)) continue;
            const owners = origins.get(output.asset.id) || new Set<string>();
            owners.add(run.attempt.id);
            origins.set(output.asset.id, owners);
        }
        // Expand verified copies across kinds/containers, then follow explicit
        // transformations. A directory name alone never establishes provenance.
        while (true) {
            const pending = [...origins.keys()];
            if (!pending.length) break;
            const sourceAssets = await this.associationAssets(user, pending, context);
            const equivalents = await this.equivalentAssets(user, sourceAssets, context);
            let changed = false;
            for (const source of sourceAssets) for (const equivalent of equivalents.get(source.id) || []) {
                const owners = origins.get(equivalent.id) || new Set<string>();
                for (const owner of origins.get(source.id) || []) {
                    if (!owners.has(owner)) changed = true;
                    owners.add(owner);
                }
                origins.set(equivalent.id, owners);
            }
            const inputLinks = await TaskAssetTable.findBy({ assetVersionId: In([...origins.keys()]), direction: 'input' });
            if (!inputLinks.length) break;
            const tasks = await ExperimentTaskTable.findBy({ id: In([...new Set(inputLinks.map(link => link.taskId))]),
                kind: In(['merge', 'publish', 'evaluate']) });
            if (!tasks.length) break;
            const owned = await ExperimentTable.findBy({ id: In([...new Set(tasks.map(task => task.experimentId))]), ownerUserId: user.id });
            const ownedIds = new Set(owned.map(item => item.id));
            const mergeIds = new Set(tasks.filter(task => ownedIds.has(task.experimentId)).map(task => task.id));
            const taskKinds = new Map(tasks.map(task => [task.id, task.kind]));
            const mergeInputs = inputLinks.filter(link => mergeIds.has(link.taskId) &&
                (taskKinds.get(link.taskId) !== 'evaluate' || link.role === 'evaluated_model'));
            const mergeAttempts = [...new Set(mergeInputs.map(link => link.attemptId))];
            const outputLinks = mergeAttempts.length ? await TaskAssetTable.findBy({ attemptId: In(mergeAttempts), direction: 'output' }) : [];
            const outputAssets = outputLinks.length ? await AssetVersionTable.findBy({
                id: In([...new Set(outputLinks.map(link => link.assetVersionId))]), ownerUserId: user.id,
            }) : [];
            const outputMap = new Map(outputAssets.filter(asset => modelKinds.has(asset.kind)).map(asset => [asset.id, asset]));
            for (const input of mergeInputs) for (const output of outputLinks) {
                const outputAsset = outputMap.get(output.assetVersionId);
                const sourceIds = outputAsset?.metadata.inputVersionIds;
                if (input.attemptId !== output.attemptId || !Array.isArray(sourceIds) ||
                    !sourceIds.includes(input.assetVersionId)) continue;
                const owners = origins.get(output.assetVersionId) || new Set<string>();
                for (const owner of origins.get(input.assetVersionId) || []) {
                    if (!owners.has(owner)) changed = true;
                    owners.add(owner);
                }
                origins.set(output.assetVersionId, owners);
            }
            if (!changed) break;
        }
        const linkedEvaluations: Array<{
            trainingAttemptId: string; trainingExperimentId: string; sourceExperimentId: string; sourceExperimentName: string;
            evaluationTaskKind: string;
            evaluationAttempt: TaskAttemptTable; result: EvaluationResultTable; metrics: EvaluationMetricTable[];
            model: AssetVersionTable; datasets: AssetVersionTable[]; eligible: boolean; ineligibleReason: string | null;
        }> = [];
        if (origins.size) {
            const versionIds = [...origins.keys()];
            const modelInputLinks = await TaskAssetTable.findBy({ assetVersionId: In(versionIds), direction: 'input' });
            const candidateAttempts = [...new Set(modelInputLinks.map(link => link.attemptId))];
            const directlyEvaluated = await EvaluationResultTable.findBy({ modelVersionId: In(versionIds) });
            const evaluatedInAttempt = candidateAttempts.length
                ? await EvaluationResultTable.findBy({ attemptId: In(candidateAttempts) }) : [];
            const results = [...new Map([...directlyEvaluated, ...evaluatedInAttempt].map(result => [result.id, result])).values()];
            if (results.length) {
                const tasks = await ExperimentTaskTable.findBy({ id: In([...new Set(results.map(result => result.taskId))]) });
                const evaluationTasks = new Map(tasks.filter(task => ['evaluate', 'benchmark'].includes(task.kind))
                    .map(task => [task.id, task]));
                const sourceExperiments = tasks.length ? await ExperimentTable.findBy({
                    id: In([...new Set(tasks.map(task => task.experimentId))]), ownerUserId: user.id,
                }) : [];
                const sourceMap = new Map(sourceExperiments.map(experiment => [experiment.id, experiment]));
                const attempts = await TaskAttemptTable.find({ where: { id: In([...new Set(results.map(result => result.attemptId))]) },
                    ...(!includeConfiguration ? { select: ['id', 'taskId', 'nodeId', 'container', 'attemptNumber', 'status', 'lastSequence',
                        'error', 'resumeFromVersionId', 'resourceReservationId', 'startedAt', 'endedAt', 'createdAt', 'updatedAt', 'runtime'] as Array<keyof TaskAttemptTable> } : {}) });
                if (!includeConfiguration) for (const attempt of attempts) { attempt.configuration = {}; attempt.environment = {}; }
                const attemptMap = new Map(attempts.map(attempt => [attempt.id, attempt]));
                const outputLinks = await TaskAssetTable.findBy({ attemptId: In([...new Set(results.map(result => result.attemptId))]), direction: 'output' });
                const outputKeys = new Set(outputLinks.map(link => `${link.attemptId}:${link.assetVersionId}`));
                const metrics = await EvaluationMetricTable.findBy({ resultId: In(results.map(result => result.id)) });
                const assetIds = [...new Set([...results.map(result => result.modelVersionId), ...metrics.map(metric => metric.datasetVersionId)])];
                const assets = await AssetVersionTable.findBy({ id: In(assetIds), ownerUserId: user.id });
                const assetMap = new Map(assets.map(asset => [asset.id, asset]));
                const runMap = new Map(trainingRuns.map(run => [run.attempt.id, run]));
                for (const result of results) {
                    const task = evaluationTasks.get(result.taskId);
                    const source = task && sourceMap.get(task.experimentId);
                    const attempt = attemptMap.get(result.attemptId);
                    const model = assetMap.get(result.modelVersionId);
                    if (!task || !source || !attempt || !model) continue;
                    const sourceIds = model.metadata.inputVersionIds;
                    const candidateVersions = origins.has(result.modelVersionId) ? [result.modelVersionId] :
                        outputKeys.has(`${result.attemptId}:${result.modelVersionId}`) && Array.isArray(sourceIds)
                            ? modelInputLinks.filter(link => link.attemptId === result.attemptId &&
                                sourceIds.includes(link.assetVersionId)).map(link => link.assetVersionId) : [];
                    const owners = new Set(candidateVersions.flatMap(versionId => [...(origins.get(versionId) || [])]));
                    const resultMetrics = metrics.filter(metric => metric.resultId === result.id);
                    const datasets = [...new Set(resultMetrics.map(metric => metric.datasetVersionId))]
                        .map(versionId => assetMap.get(versionId)).filter((asset): asset is AssetVersionTable => Boolean(asset));
                    const ineligibleReason = comparisonIneligibility(attempt, result);
                    for (const owner of owners) {
                        const run = runMap.get(owner);
                        if (!run) continue;
                        linkedEvaluations.push({ trainingAttemptId: owner, trainingExperimentId: run.experimentId,
                            sourceExperimentId: source.id, sourceExperimentName: source.name,
                            evaluationTaskKind: task.kind,
                            evaluationAttempt: attempt, result, metrics: resultMetrics, model, datasets,
                            eligible: ineligibleReason === null, ineligibleReason });
                    }
                }
            }
        }
        return { trainingRuns, linkedEvaluations };
    }

    static async launchDraft(user: SafeAuthUser, experimentId: string) {
        const draft = await this.owned(user, experimentId);
        if (draft.status !== 'draft') {
            const task = await ExperimentTaskTable.findOneBy({ experimentId });
            const attempt = task ? await TaskAttemptTable.findOne({ where: { taskId: task.id }, order: { attemptNumber: 'DESC' } }) : null;
            if (!attempt) conflict('Only experiment drafts can be launched');
            if (attempt!.status !== 'prepared' || attempt!.requestKey !== `draft-${experimentId}`) {
                return { success: true, experimentId, status: draft.status, reused: true };
            }
            // A previous prepare response may have been lost before Runtime claimed the launch.
            // Re-submit the same request key so the node can reconcile its durable intent.
        }
        const versions = z.array(z.object({ assetVersionId: experimentIdSchema,
            role: z.string(), fingerprint: z.string() })).min(1).parse(draft.spec.inputVersions);
        const assets = await Promise.all(versions.map(version => this.asset(ExperimentTable.getRepository().manager, version.assetVersionId, user.id)));
        const nodeId = assets[0].nodeId, container = assets[0].location.container;
        await this.runtimeActor(nodeId, user.id);
        for (const [index, asset] of assets.entries()) {
            if (asset.nodeId !== nodeId || asset.location.container !== container || asset.integrity !== 'verified' ||
                asset.fingerprint !== versions[index].fingerprint || asset.metadata.storage !== 'snapshot') {
                conflict('Draft requires available immutable input versions in the same execution container');
            }
        }
        const group = await ResourceAccessService.getGroupForUser(user.id);
        if (user.role !== UserRole.ADMIN && (!group || ![group.defaultContainerName, group.defaultEvaluateContainerName,
            group.defaultGrpoContainerName, group.defaultMultinodeContainerName].includes(container))) deny('Draft container is no longer assigned to this user');
        const kind = z.enum(['train', 'evaluate']).parse(draft.spec.kind);
        const result = await remoteResourceClient.request<{ data: Record<string, unknown> }>(nodeId, 'experiments/launch', {
            method: 'POST', body: JSON.stringify({ nodeId, container, actorUserId: user.id, groupId: group?.id,
                experimentId: draft.id, name: draft.name, project: draft.project, kind, spec: draft.spec,
                inputAssets: assets.map((asset, index) => ({ id: asset.id, kind: asset.kind,
                    role: versions[index].role, fingerprint: asset.fingerprint,
                    location: asset.location, metadata: asset.metadata })) }),
        }, undefined, 180000);
        return { success: true, experimentId, ...result.data };
    }

    private static async experimentsForAssets(user: SafeAuthUser, assets: AssetVersionTable[], scope: AssociationScope = 'all') {
        const relations = await this.assetRelations(user, assets, scope);
        const ids = [...new Set(relations.map(relation => relation.experimentId))];
        const records = ids.length ? await ExperimentTable.findBy({ id: In(ids), ownerUserId: user.id }) : [];
        return records.map(record => ({ ...record, assetRelations: relations.filter(relation => relation.experimentId === record.id) }));
    }

    static async forAsset(user: SafeAuthUser, assetVersionId: string, scope: AssociationScope = 'all') {
        const asset = await this.asset(ExperimentTable.getRepository().manager, assetVersionId, user.id);
        return this.experimentsForAssets(user, [asset], scope);
    }

    static async forPath(user: SafeAuthUser, nodeId: string, container: string, path: string, scope: AssociationScope = 'all') {
        const assets = (await AssetVersionTable.findBy({ ownerUserId: user.id, nodeId })).filter(asset => asset.location.container === container);
        if (!assets.length) return [];
        const direct = new Set(assets.filter(asset => asset.location.path === path || asset.metadata.sourcePath === path).map(asset => asset.id));
        const links = (await TaskAssetTable.findBy({ assetVersionId: In(assets.map(asset => asset.id)) }))
            .filter(link => direct.has(link.assetVersionId) || link.sourcePaths.includes(path));
        // Resolve recorded versions at this exact location first, then apply the
        // same verified model-content matching used by experiment details.
        const matchedIds = new Set([...direct, ...links.map(link => link.assetVersionId)]);
        return this.experimentsForAssets(user, assets.filter(asset => matchedIds.has(asset.id)), scope);
    }
}
