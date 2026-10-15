// Run in the existing remote dependency environment; uses only an in-memory database.
import 'reflect-metadata';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import { TRPCError } from '@trpc/server';
import { DataSource } from 'typeorm';
import { prepareExperimentSchema, listExperimentsSchema, ExperimentEventInput } from '../../shared/src/experiment';
import { AuthDao, SafeAuthUser } from '../src/dao/Auth';
import { UserRole } from '../src/models/Auth';
import { UserGroupTable, ResourceAuditEventTable } from '../src/models/ResourceAccess';
import { TrainingReservationTable, TrainingReservationNodeTable, TrainingReservationStatus, TrainingResourceLockTable } from '../src/models/TrainingResource';
import { experimentEntities, ExperimentTable, TaskAttemptTable, ExperimentEventTable, ExperimentLogSourceTable,
    ExperimentLogChunkTable } from '../src/models/Experiment';
import { CreateExperimentRecords1930000000000 } from '../src/migrations/1930000000000-CreateExperimentRecords';
import { AddExperimentReservation1940000000000 } from '../src/migrations/1940000000000-AddExperimentReservation';
import { CreateExperimentLogs1950000000000 } from '../src/migrations/1950000000000-CreateExperimentLogs';
import { OptimizeExperimentQueries1960000000000 } from '../src/migrations/1960000000000-OptimizeExperimentQueries';
import { AddExperimentMetadata1970000000000 } from '../src/migrations/1970000000000-AddExperimentMetadata';
import { experimentMetadataSchema, searchExperimentLogsSchema, serviceOperationSchema, experimentEventSchema } from '../../shared/src/experiment';
import { TrainingResourceService } from '../src/services/trainingResourceService';
import { ExperimentService, canonicalJson, redactExperimentValue } from '../src/services/experimentService';
import { ResourceAccessService } from '../src/services/resourceAccessService';
import { remoteResourceClient, resourceNodeRegistry } from '../src/services/resourceNodeService';

async function main() {
    const database = new DataSource({ type: 'better-sqlite3', database: ':memory:', entities: experimentEntities,
        migrations: [CreateExperimentRecords1930000000000, AddExperimentReservation1940000000000,
            CreateExperimentLogs1950000000000, OptimizeExperimentQueries1960000000000,
            AddExperimentMetadata1970000000000], synchronize: false });
    // Resource fixtures have their own temporary database so synchronization
    // cannot silently repair an invalid experiment migration under test.
    const resources = new DataSource({ type: 'better-sqlite3', database: ':memory:', synchronize: true,
        entities: [TrainingReservationTable, TrainingReservationNodeTable, TrainingResourceLockTable, ResourceAuditEventTable] });
    const user: SafeAuthUser = { id: 'owner-a', username: 'test', role: UserRole.USER, assignedNodeId: 'node-a',
        disabled: false, mustChangePassword: false, createdAt: new Date().toISOString() };
    const originals = { getUser: AuthDao.getUserById, getNode: resourceNodeRegistry.get,
        getGroup: ResourceAccessService.getGroupForUser, request: remoteResourceClient.request };
    AuthDao.getUserById = async id => id === user.id ? user : null;
    resourceNodeRegistry.get = nodeId => {
        if (!['node-a', 'node-b', 'node-c'].includes(nodeId)) throw new Error('Unknown node');
        return { id: nodeId, name: nodeId, baseUrl: 'http://127.0.0.1', defaultContainer: 'trainer', status: 'online' };
    };
    ResourceAccessService.getGroupForUser = async () => ({ id: 'group-a', defaultContainerName: 'trainer',
        defaultEvaluateContainerName: 'evaluator', defaultGrpoContainerName: 'grpo', defaultMultinodeContainerName: 'multi' } as UserGroupTable);
    try {
        await database.initialize();
        await database.runMigrations();
        await resources.initialize();
        const experimentIndexes = () => database.query("SELECT name FROM sqlite_master WHERE type = 'index' AND name IN ('IDX_asset_owner_node', 'IDX_task_asset_task', 'IDX_evaluation_model', 'IDX_evaluation_attempt')");
        assert.equal((await experimentIndexes()).length, 4);
        const input = prepareExperimentSchema.parse({ runtimeNodeId: 'node-a', actorUserId: user.id,
            requestKey: 'request-1', name: 'Example', kind: 'train', spec: { kind: 'train' }, container: 'trainer' });
        const [first, replay] = await Promise.all([ExperimentService.prepare(input), ExperimentService.prepare(input)]);
        assert.equal(first.attemptId, replay.attemptId, 'Concurrent replay must not create another attempt');
        assert.equal(await TaskAttemptTable.count(), 1);
        assert.equal(listExperimentsSchema.safeParse({ status: 'invalid' }).success, false);
        assert.equal(listExperimentsSchema.safeParse({ taskKind: 'invalid' }).success, false);
        assert.equal(listExperimentsSchema.safeParse({ search: 'x'.repeat(201) }).success, false);
        assert.equal(listExperimentsSchema.safeParse({ createdFrom: 'invalid' }).success, false);
        assert.equal(listExperimentsSchema.safeParse({ createdFrom: '2026-10-15T00:00:00.000Z', createdBefore: '2026-10-14T00:00:00.000Z' }).success, false);
        const filtered = await ExperimentService.list(user, listExperimentsSchema.parse({ search: 'EXAMPLE', taskKind: 'train', status: 'prepared' }));
        assert.equal(filtered.total, 1); assert.equal(filtered.items[0].id, first.experimentId);
        assert.equal((await ExperimentService.list(user, listExperimentsSchema.parse({ search: '%' }))).total, 0);
        assert.equal((await ExperimentService.list({ ...user, id: 'other-owner' }, listExperimentsSchema.parse({ search: 'Example' }))).total, 0);
        await database.undoLastMigration();
        assert.equal((await database.query("PRAGMA table_info('experiment')")).some((column: { name: string }) => column.name === 'notes'), false);
        await database.undoLastMigration();
        assert.equal((await experimentIndexes()).length, 0);
        assert.equal((await database.query('SELECT id FROM task_attempt WHERE id = ?', [first.attemptId])).length, 1);
        await database.runMigrations();
        assert.equal((await experimentIndexes()).length, 4);
        assert.equal((await ExperimentTable.findOneByOrFail({ id: first.experimentId })).notes, '');
        assert.deepEqual((await ExperimentTable.findOneByOrFail({ id: first.experimentId })).tags, []);
        const metadataSpecSnapshot = (await ExperimentTable.findOneByOrFail({ id: first.experimentId })).spec;
        const annotations = experimentMetadataSchema.parse({ experimentId: first.experimentId, notes: '第一轮基准', tags: [' baseline ', 'baseline', '医疗'] });
        await ExperimentService.updateMetadata(user, annotations);
        assert.deepEqual((await ExperimentTable.findOneByOrFail({ id: first.experimentId })).spec, metadataSpecSnapshot);
        assert.equal((await ExperimentService.list(user, listExperimentsSchema.parse({ tag: '医疗' }))).total, 1);
        assert.equal((await ExperimentService.list(user, listExperimentsSchema.parse({ tag: '医' }))).total, 0);
        await assert.rejects(ExperimentService.updateMetadata({ ...user, id: 'other' }, annotations), /unavailable/);
        assert.equal(experimentMetadataSchema.safeParse({ ...annotations, tags: ['x'.repeat(41)] }).success, false);
        assert.equal(experimentMetadataSchema.safeParse({ ...annotations, notes: 'x'.repeat(10001) }).success, false);
        assert.equal((await TaskAttemptTable.findOneByOrFail({ id: first.attemptId })).resourceReservationId, null);
        await assert.rejects(ExperimentService.prepare({ ...input, spec: { changed: true } }), /different inputs/);
        await assert.rejects(ExperimentService.prepare({ ...input, container: 'someone-elses-container' }), /not assigned/);
        await assert.rejects(ExperimentService.detail({ ...user, id: 'owner-b' }, first.experimentId), /unavailable/);
        let serial = 0;
        const event = async (attemptId: string, type: ExperimentEventInput['type'], payload: Record<string, unknown>) => {
            const attempt = await TaskAttemptTable.findOneByOrFail({ id: attemptId });
            return { eventId: `event-${++serial}`, attemptId, sequence: attempt.lastSequence + 1,
                occurredAt: new Date().toISOString(), type, payload };
        };
        const running = await event(first.attemptId, 'status', { status: 'running' });
        await ExperimentService.append('node-a', [running]);
        await ExperimentService.append('node-a', [running]);
        assert.equal(await ExperimentEventTable.count(), 1, 'Lost-ACK replay must not duplicate events');
        await assert.rejects(ExperimentService.append('node-a', [{ ...running, payload: { status: 'failed' } }]), /payload conflict/);
        await assert.rejects(ExperimentService.append('node-a', [{ ...running, eventId: 'out-of-order', sequence: 3 }]), /sequence 2/);
        await assert.rejects(ExperimentService.append('node-b', [running]), /not assigned/);
        const makeAsset = (id: string, kind: string, role: string) => {
            const files = [{ path: 'content.json', size: 4, sha256: crypto.createHash('sha256').update(id).digest('hex') }];
            return { id, kind, role, direction: 'input', logicalName: id, integrity: 'verified',
                fingerprint: crypto.createHash('sha256').update(canonicalJson(files)).digest('hex'),
                location: { container: 'trainer', path: `/versions/${id}` },
                manifest: { schemaVersion: 1, files, byteCount: 4, fileCount: 1 },
                metadata: { sourcePath: `/original/${id}`, storage: 'snapshot' } };
        };
        const model = makeAsset('model-1', 'model', 'base_model');
        const dataset = makeAsset('dataset-1', 'dataset', 'training_dataset');
        for (const asset of [model, dataset]) await ExperimentService.append('node-a', [await event(first.attemptId, 'artifact', asset)]);
        await assert.rejects(ExperimentService.append('node-a', [await event(first.attemptId, 'artifact', {
            ...model, location: { container: 'trainer', path: '/changed' },
        })]), /immutable/);
        await ExperimentService.append('node-a', [await event(first.attemptId, 'artifact', {
            ...model, metadata: { ...model.metadata, sourcePath: '/another-name/model' },
        })]);
        assert.equal((await ExperimentService.forPath(user, 'node-a', 'trainer', '/another-name/model')).length, 1);
        assert.equal((await ExperimentService.forPath({ ...user, id: 'owner-b' }, 'node-a', 'trainer', '/another-name/model')).length, 0);
        await ExperimentService.append('node-a', [await event(first.attemptId, 'status', { status: 'failed' })]);
        const log = { runtimeNodeId: 'node-a', attemptId: first.attemptId,
            source: { id: 'log-first', label: 'Execution', kind: 'execution' as const, status: 'complete' as const },
            chunks: [{ startOffset: 0, endOffset: 21, content: 'password=private\n' }] };
        await ExperimentService.appendLog(log);
        await ExperimentService.appendLog(log);
        await ExperimentService.appendLog({ ...log, chunks: [{ startOffset: 21, endOffset: 26, content: 'tail\n' }] });
        assert.equal(await ExperimentLogSourceTable.count(), 1);
        assert.equal(await ExperimentLogChunkTable.count(), 2, 'Lost-ACK replay must not duplicate logs');
        assert.equal((await ExperimentService.logSources(user, first.experimentId, first.attemptId))[0].status, 'complete');
        const logPage = await ExperimentService.logChunks(user, first.experimentId, first.attemptId, 'log-first');
        assert.equal(logPage.chunks.length, 2);
        assert.equal((await ExperimentService.logChunks(user, first.experimentId, first.attemptId,
            'log-first', undefined, 21)).chunks.length, 1);
        assert.equal((await ExperimentService.logChunks(user, first.experimentId, first.attemptId,
            'log-first', 21)).chunks.length, 1);
        assert.ok(!logPage.chunks[0].content.includes('private'), 'Raw secrets must not be stored');
        const businessLog = { ...log, source: { ...log.source, id: 'business-log', kind: 'training' as const },
            chunks: [{ startOffset: 0, endOffset: 100, content: '中文🙂 password=private\nER' },
                { startOffset: 100, endOffset: 200, content: 'ROR 出错\n' }] };
        await ExperimentService.appendLog(businessLog);
        const logSearchInput = searchExperimentLogsSchema.parse({ experimentId: first.experimentId, attemptId: first.attemptId,
            sourceId: 'business-log', keyword: 'ERROR' });
        const searched = await ExperimentService.searchLogs(user, logSearchInput);
        assert.equal(searched.matches.length, 1);
        assert.ok(!JSON.stringify(searched).includes('private'));
        await ExperimentService.appendLog({ ...businessLog, chunks: [{ startOffset: 200, endOffset: 300, content: 'ERROR later\n' }] });
        assert.equal((await ExperimentService.searchLogs(user, { ...logSearchInput, snapshotEndOffset: searched.snapshotEndOffset })).matches.length, 1);
        assert.equal((await ExperimentService.searchLogs(user, logSearchInput)).matches.length, 2);
        const context = await ExperimentService.logContext(user, { experimentId: first.experimentId, attemptId: first.attemptId,
            sourceId: 'business-log', position: searched.matches[0].position, matchLength: 5, snapshotEndOffset: searched.snapshotEndOffset });
        assert.match(context.chunks.map(chunk => chunk.content).join(''), /ERROR/);
        assert.ok(!context.chunks.some(chunk => chunk.content.includes('later')));
        await assert.rejects(ExperimentService.searchLogs({ ...user, id: 'other' }, logSearchInput), /unavailable/);
        await assert.rejects(ExperimentService.logContext({ ...user, id: 'other' }, { ...logSearchInput,
            position: searched.matches[0].position, matchLength: 5, snapshotEndOffset: searched.snapshotEndOffset }), /unavailable/);
        await assert.rejects(ExperimentService.businessLogSource(user, first.experimentId, first.attemptId, 'log-first'), /unavailable/);
        const summary = await ExperimentService.summary(user, first.experimentId);
        assert.ok(summary.attempts.every(attempt => !('actual' in attempt.configuration)));
        assert.ok(summary.assets.every(asset => !('files' in asset.manifest)));
        assert.equal(summary.lineage.length, 0);
        assert.equal(summary.experiment.notes, '第一轮基准');
        for (const method of ['summary', 'configuration', 'assets', 'evaluations'] as const)
            await assert.rejects(ExperimentService[method]({ ...user, id: 'other' }, first.experimentId), /unavailable/);
        await assert.rejects(ExperimentService.appendLog({ ...log, runtimeNodeId: 'node-b' }), /not assigned/);
        await assert.rejects(ExperimentService.appendLog({ ...log,
            chunks: [{ ...log.chunks[0], content: 'changed' }] }), /payload conflict/);
        await assert.rejects(ExperimentService.logSources({ ...user, id: 'owner-b' }, first.experimentId,
            first.attemptId), /unavailable/);
        await assert.rejects(ExperimentService.logChunks(user, first.experimentId, first.attemptId,
            'another-source'), /unavailable/);
        await assert.rejects(ExperimentService.append('node-a', [await event(first.attemptId, 'status', { status: 'running' })]), /Terminal/);
        const retry = await ExperimentService.prepare({ ...input, requestKey: 'request-retry', experimentId: first.experimentId, taskId: first.taskId });
        assert.equal(retry.attemptNumber, 2);
        assert.notEqual(retry.attemptId, first.attemptId);
        const evaluate = await ExperimentService.prepare({ ...input, kind: 'evaluate', spec: {}, requestKey: 'evaluate-1', experimentId: first.experimentId });
        const evaluation = { id: 'result-1', modelVersionId: model.id, protocol: { version: 1 }, protocolComplete: false,
            expectedCount: null, succeededCount: null, failedCount: null,
            metrics: [{ datasetVersionId: dataset.id, name: 'accuracy', value: 50, sampleCount: null, direction: 'higher' }] };
        await ExperimentService.append('node-a', [await event(evaluate.attemptId, 'evaluation', evaluation)]);
        const detail = await ExperimentService.detail(user, first.experimentId);
        assert.equal(detail.evaluations[0].expectedCount, null);
        assert.equal(detail.metrics[0].sampleCount, null);
        const compared = await ExperimentService.compare(user, [first.experimentId]);
        assert.ok(Object.keys(compared.groups)[0].startsWith('incomplete:'));
        const completeEvaluation = { ...evaluation, id: 'comparison-complete', protocolComplete: true,
            expectedCount: 4, succeededCount: 4, failedCount: 0,
            metrics: [{ ...evaluation.metrics[0], sampleCount: 4 }] };
        const comparisonAttempt = await ExperimentService.prepare({ ...input, kind: 'evaluate', spec: {},
            requestKey: 'comparison-attempt', experimentId: first.experimentId });
        await ExperimentService.append('node-a', [await event(comparisonAttempt.attemptId, 'evaluation', completeEvaluation)]);
        const assertComparisonEligibility = async (resultId: string, eligible: boolean) => {
            const comparison = await ExperimentService.compare(user, [first.experimentId]);
            const group = Object.entries(comparison.groups).find(([, rows]) => rows.some(row => row.result.id === resultId));
            assert.ok(group, 'Ineligible results must remain visible');
            assert.equal(!group[0].startsWith('incomplete:'), eligible, `Unexpected comparison eligibility for ${resultId}`);
        };
        await assertComparisonEligibility(completeEvaluation.id, false);
        for (const status of ['running', 'unknown']) {
            await ExperimentService.append('node-a', [await event(comparisonAttempt.attemptId, 'status', {
                status, runtime: { inputIntegrityAfterExecution: 'verified' },
            })]);
            await assertComparisonEligibility(completeEvaluation.id, false);
        }
        await ExperimentService.append('node-a', [await event(comparisonAttempt.attemptId, 'status', {
            status: 'succeeded', runtime: { inputIntegrityAfterExecution: 'unverified' },
        })]);
        await assertComparisonEligibility(completeEvaluation.id, false);
        await ExperimentService.append('node-a', [await event(comparisonAttempt.attemptId, 'status', {
            status: 'succeeded', runtime: { inputIntegrityAfterExecution: 'verified' },
        })]);
        await assertComparisonEligibility(completeEvaluation.id, true);
        for (const status of ['failed', 'cancelled', 'succeeded']) {
            const stopped = await ExperimentService.prepare({ ...input, kind: 'evaluate', spec: {},
                requestKey: `comparison-${status}`, experimentId: first.experimentId });
            const result = { ...completeEvaluation, id: `result-${status}` };
            await ExperimentService.append('node-a', [await event(stopped.attemptId, 'evaluation', result)]);
            await ExperimentService.append('node-a', [await event(stopped.attemptId, 'status', {
                status, runtime: status === 'succeeded' ? {} : { inputIntegrityAfterExecution: 'verified' },
            })]);
            await assertComparisonEligibility(result.id, false);
        }
        await ExperimentService.append('node-a', [await event(evaluate.attemptId, 'status', {
            status: 'succeeded', runtime: { inputIntegrityAfterExecution: 'verified' },
        })]);
        await assertComparisonEligibility(evaluation.id, false);
        await ExperimentService.append('node-a', [await event(comparisonAttempt.attemptId, 'evaluation', {
            ...completeEvaluation, id: 'comparison-partial', succeededCount: 3, failedCount: 1,
        })]);
        await assertComparisonEligibility('comparison-partial', false);
        await assertComparisonEligibility(completeEvaluation.id, true);
        const trainedModel = { ...makeAsset('trained-model-1', 'model', 'model'), direction: 'output' };
        await ExperimentService.append('node-a', [await event(retry.attemptId, 'artifact', trainedModel)]);
        await ExperimentService.append('node-a', [await event(retry.attemptId, 'status', { status: 'succeeded' })]);
        const separateEvaluation = await ExperimentService.prepare({ ...input, kind: 'evaluate', requestKey: 'separate-evaluation',
            spec: { kind: 'evaluate' }, inputs: [{ assetVersionId: trainedModel.id, role: 'evaluated_model' },
                { assetVersionId: dataset.id, role: 'evaluation_dataset' }] });
        const trainedScore = { ...completeEvaluation, id: 'trained-score', modelVersionId: trainedModel.id };
        await ExperimentService.append('node-a', [await event(separateEvaluation.attemptId, 'evaluation', trainedScore)]);
        let trainingComparison = await ExperimentService.compare(user, [first.experimentId, separateEvaluation.experimentId]);
        assert.equal(trainingComparison.trainingRuns.length, 2, 'Both original and retried training executions remain selectable');
        assert.equal(trainingComparison.linkedEvaluations.find(item => item.result.id === trainedScore.id)?.ineligibleReason,
            'execution-not-succeeded');
        await ExperimentService.append('node-a', [await event(separateEvaluation.attemptId, 'status', {
            status: 'succeeded', runtime: { inputIntegrityAfterExecution: 'verified' },
        })]);
        trainingComparison = await ExperimentService.compare(user, [first.experimentId, separateEvaluation.experimentId]);
        assert.deepEqual(trainingComparison.linkedEvaluations.filter(item => item.result.id === trainedScore.id)
            .map(item => item.trainingAttemptId), [retry.attemptId]);
        assert.equal(trainingComparison.linkedEvaluations.find(item => item.result.id === trainedScore.id)?.eligible, true);
        assert.equal((await ExperimentService.detail(user, first.experimentId)).linkedEvaluations
            .find(item => item.result.id === trainedScore.id)?.sourceExperimentId, separateEvaluation.experimentId,
            'Training detail must include a standalone evaluation without selecting it for comparison');
        assert.ok(!trainingComparison.linkedEvaluations.some(item => item.result.id === completeEvaluation.id),
            'An evaluation of the base model must not be attributed to the trained output');
        const derivedModel = { ...makeAsset('derived-model-1', 'model', 'merged_model'), direction: 'output',
            metadata: { inputVersionIds: [trainedModel.id] } };
        await ExperimentService.append('node-a', [await event(separateEvaluation.attemptId, 'artifact', derivedModel)]);
        await ExperimentService.append('node-a', [await event(separateEvaluation.attemptId, 'evaluation', {
            ...trainedScore, id: 'derived-score', modelVersionId: derivedModel.id,
        })]);
        trainingComparison = await ExperimentService.compare(user, [first.experimentId]);
        assert.equal(trainingComparison.linkedEvaluations.find(item => item.result.id === 'derived-score')?.trainingAttemptId,
            retry.attemptId, 'An explicitly derived evaluated model keeps the training provenance');
        assert.ok((await ExperimentService.forAsset(user, derivedModel.id)).some(record => record.id === first.experimentId),
            'A merged model entry must trace back to the training producing its input');
        await ExperimentTable.update(separateEvaluation.experimentId, { ownerUserId: 'other-owner' });
        trainingComparison = await ExperimentService.compare(user, [first.experimentId]);
        assert.ok(!trainingComparison.linkedEvaluations.some(item => item.result.id === trainedScore.id),
            'Evaluations owned by another user must not be exposed through model lineage');
        await ExperimentTable.update(separateEvaluation.experimentId, { ownerUserId: user.id });
        const copiedEvaluation = await ExperimentService.prepare({ ...input, kind: 'evaluate', container: 'evaluator',
            requestKey: 'copied-evaluation', spec: { kind: 'evaluate' } });
        const copiedModel = { ...trainedModel, id: 'copied-checkpoint', kind: 'checkpoint', direction: 'input',
            role: 'evaluated_model', location: { container: 'evaluator', path: '/versions/copied-checkpoint' } };
        await ExperimentService.append('node-a', [await event(copiedEvaluation.attemptId, 'artifact', copiedModel)]);
        await ExperimentService.append('node-a', [await event(copiedEvaluation.attemptId, 'evaluation', {
            ...trainedScore, id: 'copied-score', modelVersionId: copiedModel.id,
        })]);
        const copiedDetail = await ExperimentService.detail(user, first.experimentId);
        for (const records of [await ExperimentService.forAsset(user, trainedModel.id),
            await ExperimentService.forPath(user, 'node-a', 'trainer', trainedModel.location.path)]) {
            assert.ok(records.some(record => record.id === copiedEvaluation.experimentId),
                'The model Experiment entry must include evaluations of an equivalent copied version');
            assert.equal(new Set(records.map(record => record.id)).size, records.length);
        }
        assert.ok((await ExperimentService.forPath(user, 'node-a', 'evaluator', copiedModel.location.path))
            .some(record => record.id === first.experimentId), 'The copied model entry must include its training record');
        assert.equal(copiedDetail.linkedEvaluations.find(item => item.result.id === 'copied-score')?.trainingAttemptId,
            retry.attemptId, 'Verified identical model content must match across kind and container');
        assert.equal(copiedDetail.linkedEvaluations.find(item => item.result.id === 'copied-score')?.eligible, false);
        assert.ok((await ExperimentService.detail(user, copiedEvaluation.experimentId)).lineage.some(link =>
            link.assetVersionId === copiedModel.id && link.relatedAssetVersionId === trainedModel.id &&
            link.attemptId === retry.attemptId && link.direction === 'output' && link.match === 'content'),
            'Standalone evaluation must link back to the training output');
        const changedModel = { ...makeAsset('changed-copy', 'checkpoint', 'evaluated_model'),
            location: { ...copiedModel.location }, metadata: { ...copiedModel.metadata } };
        await ExperimentService.append('node-a', [await event(copiedEvaluation.attemptId, 'artifact', changedModel)]);
        await ExperimentService.append('node-a', [await event(copiedEvaluation.attemptId, 'evaluation', {
            ...trainedScore, id: 'changed-copy-score', modelVersionId: changedModel.id,
        })]);
        assert.ok(!(await ExperimentService.detail(user, first.experimentId)).linkedEvaluations.some(item =>
            item.result.id === 'changed-copy-score'), 'The same model path with different content must not match');
        await ExperimentTable.update(copiedEvaluation.experimentId, { ownerUserId: 'other-owner' });
        assert.ok(!(await ExperimentService.forAsset(user, trainedModel.id))
            .some(record => record.id === copiedEvaluation.experimentId));
        assert.ok(!(await ExperimentService.detail(user, first.experimentId)).linkedEvaluations.some(item =>
            item.result.id === 'copied-score'), 'Content equivalence must not expose another user evaluation');
        await ExperimentTable.update(copiedEvaluation.experimentId, { ownerUserId: user.id });
        const imported = await ExperimentService.prepare({ ...input, requestKey: 'import-1', kind: 'preprocess', spec: { kind: 'preprocess', script: 'dataset_upload' } });
        await ExperimentService.append('node-a', [await event(imported.attemptId, 'artifact', { ...dataset, direction: 'output', role: 'uploaded_dataset' })]);
        const lineage = (await ExperimentService.detail(user, first.experimentId)).lineage;
        assert.ok(lineage.some(link => link.assetVersionId === dataset.id && link.experimentId === imported.experimentId && link.direction === 'output'));
        const processed = await ExperimentService.prepare({ ...input, requestKey: 'processed-lineage', kind: 'preprocess',
            spec: { kind: 'preprocess' }, inputs: [{ assetVersionId: dataset.id, role: 'source_dataset' }] });
        const processedDataset = { ...makeAsset('processed-dataset', 'dataset', 'processed_dataset'), direction: 'output',
            metadata: { inputVersionIds: [dataset.id] } };
        await ExperimentService.append('node-a', [await event(processed.attemptId, 'artifact', processedDataset)]);
        assert.ok((await ExperimentService.forAsset(user, processedDataset.id)).some(record => record.id === imported.experimentId),
            'Processed data must expose its upload origin');
        assert.ok((await ExperimentService.forAsset(user, dataset.id)).some(record => record.id === processed.experimentId),
            'Uploaded data must expose downstream processing');
        const fixedSpec = { schemaVersion: 1, kind: 'train', script: 'batch_train_lora', inputVersions: [
            { assetVersionId: dataset.id, role: 'training_dataset', fingerprint: dataset.fingerprint }],
            resolved: { arguments: {}, environment: { LR: '0.001', RESUME: '' } } };
        const executableDraft = await ExperimentTable.save(ExperimentTable.create({
            id: 'existing-draft', name: 'Existing draft', project: input.project,
            ownerUserId: user.id, groupId: null, conversationRunId: null, workflowId: null,
            parentExperimentId: null, status: 'draft', spec: fixedSpec,
            createdAt: new Date().toISOString(), updatedAt: new Date().toISOString(),
        }));
        const dispatches: Array<{ nodeId: string; path: string; body: Record<string, unknown> }> = [];
        remoteResourceClient.request = async <T>(nodeId: string, path: string, options?: RequestInit): Promise<T> => {
            dispatches.push({ nodeId, path, body: JSON.parse(String(options?.body)) });
            return { data: { status: 'running' } } as T;
        };
        await ExperimentService.launchDraft(user, executableDraft.id);
        assert.equal(dispatches.length, 1);
        assert.equal(dispatches[0].nodeId, 'node-a');
        assert.equal(dispatches[0].path, 'experiments/launch');
        assert.deepEqual(dispatches[0].body.inputAssets, [{ id: dataset.id, kind: dataset.kind,
            role: 'training_dataset', fingerprint: dataset.fingerprint,
            location: dataset.location, metadata: dataset.metadata }], 'Replay must receive stored input locations, not reconstruct latest paths');
        const assignedGroup = ResourceAccessService.getGroupForUser;
        ResourceAccessService.getGroupForUser = async () => null;
        await assert.rejects(ExperimentService.launchDraft(user, executableDraft.id), /no longer assigned/);
        assert.equal(dispatches.length, 1, 'Revoked container assignment must stop before Runtime dispatch');
        ResourceAccessService.getGroupForUser = assignedGroup;
        remoteResourceClient.request = originals.request;
        const resumable = await ExperimentService.prepare({ ...input, requestKey: 'resumable', spec: fixedSpec });
        await ExperimentService.append('node-a', [await event(resumable.attemptId, 'status', { status: 'failed' })]);
        const checkpoint = makeAsset('checkpoint-1', 'checkpoint', 'checkpoint');
        await ExperimentService.append('node-a', [await event(resumable.attemptId, 'artifact', { ...checkpoint, direction: 'output' })]);
        const resumedSpec = { ...fixedSpec, inputVersions: [...fixedSpec.inputVersions,
            { assetVersionId: checkpoint.id, role: 'resume_checkpoint', fingerprint: checkpoint.fingerprint }],
            resolved: { ...fixedSpec.resolved, environment: { LR: '0.001', RESUME: checkpoint.location.path } } };
        const resumedInput = { ...input, requestKey: 'resumed', experimentId: resumable.experimentId, taskId: resumable.taskId,
            resumeFromVersionId: checkpoint.id, spec: resumedSpec };
        await assert.rejects(ExperimentService.prepare({ ...resumedInput, spec: { ...resumedSpec,
            resolved: { ...resumedSpec.resolved, environment: { LR: '0.002', RESUME: checkpoint.location.path } } } }), /preserve/);
        const resumed = await ExperimentService.prepare(resumedInput);
        assert.equal(resumed.attemptNumber, 2);
        assert.equal((await TaskAttemptTable.findOneByOrFail({ id: resumed.attemptId })).resumeFromVersionId, checkpoint.id);
        const grpoSpec = { ...fixedSpec, script: 'grpo_train', resolved: { arguments: {}, environment: {},
            positionalArguments: ['trainer.total_epochs=1', 'trainer.resume_mode=auto'] } };
        const grpo = await ExperimentService.prepare({ ...input, requestKey: 'grpo-first', spec: grpoSpec });
        await ExperimentService.append('node-a', [await event(grpo.attemptId, 'status', { status: 'failed' })]);
        const grpoResumeSpec = { ...grpoSpec, inputVersions: resumedSpec.inputVersions,
            resolved: { ...grpoSpec.resolved, resumeCheckpointName: 'global_step_30', positionalArguments: [
                'trainer.total_epochs=1', 'trainer.resume_mode=resume_path', 'trainer.resume_from_path=/versions/checkpoint-1'] } };
        const grpoRetry = { ...input, requestKey: 'grpo-retry', experimentId: grpo.experimentId, taskId: grpo.taskId,
            resumeFromVersionId: checkpoint.id, spec: grpoResumeSpec };
        await assert.rejects(ExperimentService.prepare({ ...grpoRetry, spec: { ...grpoResumeSpec,
            resolved: { ...grpoResumeSpec.resolved, positionalArguments: ['trainer.total_epochs=2'] } } }), /preserve/);
        assert.equal((await ExperimentService.prepare(grpoRetry)).attemptNumber, 2);
        const publishSpec = { schemaVersion: 1, script: 'workflow_publish', kind: 'publish', executionBackend: 'container_transfer',
            resolved: { sourceContainer: 'trainer', destinationContainer: 'evaluator', destination: '/published/model' } };
        await assert.rejects(ExperimentService.prepare({ ...input, kind: 'publish', requestKey: 'publish-denied',
            spec: { ...publishSpec, resolved: { ...publishSpec.resolved, destinationContainer: 'unassigned' } } }), /not assigned/);
        const published = await ExperimentService.prepare({ ...input, kind: 'publish', requestKey: 'publish', spec: publishSpec });
        await ExperimentService.append('node-a', [await event(published.attemptId, 'artifact', { ...model, role: 'published_source' })]);
        const publishedAsset = { ...model, id: 'published-model', direction: 'output', role: 'published_model',
            location: { container: 'evaluator', path: '/versions/published-model' }, metadata: { ...model.metadata, inputVersionIds: [model.id] } };
        await ExperimentService.append('node-a', [await event(published.attemptId, 'artifact', publishedAsset)]);
        await assert.rejects(ExperimentService.append('node-a', [await event(published.attemptId, 'artifact', {
            ...publishedAsset, id: 'wrong-target', location: { ...publishedAsset.location, container: 'unassigned' },
        })]), /container/);
        await assert.rejects(ExperimentService.append('node-a', [await event(evaluate.attemptId, 'artifact', {
            ...publishedAsset, id: 'invalid-transfer-kind',
        })]), /container/);
        const benchmark = await ExperimentService.prepare({ ...input, requestKey: 'native-benchmark', container: 'inference-worker',
            kind: 'benchmark', spec: { kind: 'benchmark', executionBackend: 'inference_worker' } });
        const deployment = await ExperimentService.prepare({ ...input, requestKey: 'native-deployment', container: 'inference-worker',
            kind: 'deploy', spec: { kind: 'deploy', executionBackend: 'inference_worker', serviceInstanceId: 'service-one' } });
        await ExperimentService.append('node-a', [await event(deployment.attemptId, 'status', {
            status: 'succeeded', runtime: { serviceRunId: 'run-one' },
        })]);
        const deployedAttempt = await TaskAttemptTable.findOneByOrFail({ id: deployment.attemptId });
        const stopRequest = serviceOperationSchema.parse({ operationId: 'stop-one', operation: 'stop_service',
            serviceInstanceId: 'service-one', serviceRunId: 'run-one', status: 'requested', actorUserId: user.id,
            trigger: 'manual', reservationId: null, releaseReady: null, result: '', returnCode: null });
        assert.equal(serviceOperationSchema.safeParse({ ...stopRequest, status: 'succeeded' }).success, false);
        assert.equal(serviceOperationSchema.safeParse({ ...stopRequest, status: 'failed', releaseReady: true }).success, false);
        const stopStarted = experimentEventSchema.parse(await event(deployment.attemptId, 'service_operation', stopRequest));
        await ExperimentService.append('node-a', [stopStarted]);
        await ExperimentService.append('node-a', [stopStarted]);
        const stopFinished = await event(deployment.attemptId, 'service_operation', {
            ...stopRequest, status: 'succeeded', releaseReady: true, result: 'confirmed stopped', returnCode: 0,
        });
        await ExperimentService.append('node-a', [stopFinished]);
        await ExperimentService.append('node-a', [stopFinished]);
        const stoppedSummary = await ExperimentService.summary(user, deployment.experimentId);
        assert.equal(stoppedSummary.serviceOperations.length, 1);
        assert.equal(stoppedSummary.serviceOperations[0].status, 'succeeded');
        assert.equal(stoppedSummary.experiment.status, 'succeeded');
        assert.equal(stoppedSummary.attempts[0].endedAt, deployedAttempt.endedAt);
        await assert.rejects(ExperimentService.append('node-a', [await event(deployment.attemptId, 'service_operation', {
            ...stopRequest, status: 'failed', releaseReady: false,
        })]), /immutable/);
        await assert.rejects(ExperimentService.summary({ ...user, id: 'other-owner' }, deployment.experimentId), /unavailable/);
        await assert.rejects(ExperimentService.prepare({ ...input, requestKey: 'invalid-native-train', container: 'inference-worker',
            kind: 'train', spec: { executionBackend: 'inference_worker' } }), /Invalid inference worker/);
        await assert.rejects(ExperimentService.prepare({ ...input, requestKey: 'wrong-native-node', runtimeNodeId: 'node-b', container: 'inference-worker',
            kind: 'benchmark', spec: { executionBackend: 'inference_worker' } }));
        for (const asset of [model, dataset]) await ExperimentService.append('node-a', [await event(benchmark.attemptId, 'artifact', {
            ...asset, id: `native-${asset.id}`, location: { ...asset.location, container: 'inference-worker' },
        })]);
        await ExperimentService.append('node-a', [await event(benchmark.attemptId, 'evaluation', {
            ...evaluation, id: 'native-result', modelVersionId: 'native-model-1', expectedCount: 4, succeededCount: 3, failedCount: 1,
            metrics: [{ datasetVersionId: 'native-dataset-1', name: 'accuracy', value: 0.5, sampleCount: 4, direction: 'higher' }],
        })]);
        assert.equal((await ExperimentService.detail(user, benchmark.experimentId)).evaluations[0].failedCount, 1);

        const benchmarkSpec = { schemaVersion: 1, kind: 'benchmark', script: 'runner.py', executionBackend: 'inference_worker',
            serviceInstanceId: 'service-first', resolved: { command: ['runner.py', '--output', '/job-first/result.json'] },
            protocol: { runnerSha256: 'a'.repeat(64), arguments: ['--split', 'test'], serviceConfiguration: { temperature: 0 } },
            inputVersions: [{ assetVersionId: model.id, role: 'evaluated_model', fingerprint: model.fingerprint },
                { assetVersionId: dataset.id, role: 'evaluation_dataset', fingerprint: dataset.fingerprint }] };
        const firstBenchmark = await ExperimentService.prepare({ ...input, requestKey: 'benchmark-retry-first', kind: 'benchmark',
            container: 'inference-worker', spec: benchmarkSpec });
        const retriedBenchmark = { ...input, requestKey: 'benchmark-retry-second', kind: 'benchmark' as const,
            container: 'inference-worker', experimentId: firstBenchmark.experimentId, taskId: firstBenchmark.taskId,
            spec: { ...benchmarkSpec, serviceInstanceId: 'service-restarted',
                resolved: { command: ['runner.py', '--output', '/job-second/result.json'] } } };
        await assert.rejects(ExperimentService.prepare(retriedBenchmark), /not terminal/);
        await ExperimentService.append('node-a', [await event(firstBenchmark.attemptId, 'status', { status: 'failed' })]);
        await assert.rejects(ExperimentService.prepare({ ...retriedBenchmark, spec: { ...retriedBenchmark.spec,
            protocol: { ...benchmarkSpec.protocol, runnerSha256: 'b'.repeat(64) } } }), /preserve/);
        await assert.rejects(ExperimentService.prepare({ ...retriedBenchmark, spec: { ...retriedBenchmark.spec,
            protocol: { ...benchmarkSpec.protocol, arguments: ['--split', 'validation'] } } }), /preserve/);
        await assert.rejects(ExperimentService.prepare({ ...retriedBenchmark, spec: { ...retriedBenchmark.spec,
            inputVersions: benchmarkSpec.inputVersions.map(version => ({ ...version, fingerprint: 'c'.repeat(64) })) } }), /preserve/);
        const secondBenchmark = await ExperimentService.prepare(retriedBenchmark);
        assert.equal(secondBenchmark.taskId, firstBenchmark.taskId);
        assert.equal(secondBenchmark.attemptNumber, 2);
        assert.notEqual(secondBenchmark.attemptId, firstBenchmark.attemptId);

        const reservation = { id: 'inference-lease', groupId: 'group-a', poolId: 'pool-a', homeNodeId: 'node-a',
            requestedByUserId: user.id, status: TrainingReservationStatus.RESERVED, requestedNodeCount: 1,
            gpusPerNode: 1, masterPort: 50000, taskCategory: 'inference', taskType: 'inference', taskTypeText: null,
            expiresAt: new Date(Date.now() + 600000).toISOString() };
        await TrainingReservationTable.create(reservation).save();
        await TrainingReservationNodeTable.create({ id: 'inference-node', reservationId: reservation.id,
            nodeId: 'node-b', gpuIndexes: [0], isMaster: true }).save();
        const remoteInput = { ...input, requestKey: 'remote-deploy', runtimeNodeId: 'node-b',
            resourceReservationId: reservation.id, container: 'inference-worker', kind: 'deploy' as const,
            spec: { kind: 'deploy', executionBackend: 'inference_worker' } };
        const remote = await ExperimentService.prepare(remoteInput);
        assert.equal((await TaskAttemptTable.findOneByOrFail({ id: remote.attemptId })).resourceReservationId, reservation.id);
        await assert.rejects(ExperimentService.prepare({ ...remoteInput, requestKey: 'unallocated', runtimeNodeId: 'node-c' }), /not allocated/);
        await assert.rejects(ExperimentService.prepare({ ...remoteInput, requestKey: 'wrong-backend', container: 'trainer',
            kind: 'train', spec: {} }), /native inference worker/);
        const invalidReservations: Array<Partial<TrainingReservationTable>> = [
            { groupId: 'other-group' }, { homeNodeId: 'other-home' }, { requestedByUserId: 'other-user' },
            { status: TrainingReservationStatus.RELEASED }, { status: TrainingReservationStatus.FAILED },
            { expiresAt: new Date(Date.now() - 1000).toISOString() }, { expiresAt: 'invalid' },
            { taskCategory: 'train', taskType: 'train', taskTypeText: 'train' },
        ];
        for (const [index, changed] of invalidReservations.entries()) {
            await TrainingReservationTable.update(reservation.id, changed);
            await assert.rejects(ExperimentService.prepare({ ...remoteInput, requestKey: `invalid-allocation-${index}` }),
                error => error instanceof TRPCError && error.code === 'FORBIDDEN');
            await TrainingReservationTable.update(reservation.id, reservation);
        }
        // Runtime-created reservations authorize the current group, without inventing an individual requester.
        await TrainingReservationTable.update(reservation.id, { requestedByUserId: null });
        await ExperimentService.prepare({ ...remoteInput, requestKey: 'group-runtime-benchmark', kind: 'benchmark',
            spec: { ...remoteInput.spec, kind: 'benchmark' } });
        const renewed = await TrainingResourceService.renewForRuntime(reservation.id, 'node-b');
        assert.equal(renewed.status, TrainingReservationStatus.RUNNING);
        assert.equal((await ResourceAuditEventTable.findOneByOrFail({ eventType: 'training_reservation_renewed' })).details?.runtimeNodeId, 'node-b');
        await assert.rejects(TrainingResourceService.renewForRuntime(reservation.id, 'node-c'), /not allocated/);
        await assert.rejects(TrainingResourceService.releaseForRuntime(reservation.id, 'node-b'), /无权释放/);
        await TrainingReservationTable.update(reservation.id, { taskCategory: 'train', taskType: 'train' });
        await assert.rejects(TrainingResourceService.renewForRuntime(reservation.id, 'node-b'), /active inference/);
        await TrainingResourceService.renewForRuntime(reservation.id, 'node-a'); // Existing home-node training renewal.
        await TrainingReservationTable.update(reservation.id, { ...reservation, expiresAt: new Date(Date.now() - 1000).toISOString() });
        await assert.rejects(TrainingResourceService.renewForRuntime(reservation.id, 'node-b'), /active inference/);
        await TrainingReservationTable.update(reservation.id, { status: TrainingReservationStatus.RELEASED });
        // Archival after lease release still belongs to the originally authorized node.
        const late = await event(remote.attemptId, 'status', { status: 'failed' });
        await assert.rejects(ExperimentService.append('node-a', [late]), /not assigned/);
        await ExperimentService.append('node-b', [late]);
        assert.equal((await TaskAttemptTable.findOneByOrFail({ id: remote.attemptId })).status, 'failed');
        const redacted = redactExperimentValue({ argv: ['tool', '--api_key', 'private'], text: '{"password":"private"}' });
        assert.ok(!JSON.stringify(redacted).includes('private'));
        console.log('Experiment service integration checks passed (in-memory SQLite).');
    } finally {
        AuthDao.getUserById = originals.getUser;
        resourceNodeRegistry.get = originals.getNode;
        ResourceAccessService.getGroupForUser = originals.getGroup;
        remoteResourceClient.request = originals.request;
        if (database.isInitialized) await database.destroy();
        if (resources.isInitialized) await resources.destroy();
    }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
