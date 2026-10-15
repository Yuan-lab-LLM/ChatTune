// Node 24+. Exercises the real read service with in-memory repository doubles.
// Does not replace the TypeORM/database integration suite in experiment-records.test.ts.
const assert = require('node:assert/strict');
const { test, after } = require('node:test');
const { registerHooks } = require('node:module');
const { readFileSync } = require('node:fs');
const { pathToFileURL, fileURLToPath } = require('node:url');
const path = require('node:path');
const { DatabaseSync } = require('node:sqlite');
const { modelExperimentPath } = require('../../client/src/pages/DashboardPage/RunPage/ModelManager/modelExperimentPath.ts');
const { associationLabel, associationLabels, associationRole } = require('../../client/src/pages/DashboardPage/RunPage/ExperimentManager/associationLabels.ts');

const serviceUrl = pathToFileURL(path.resolve(__dirname, '../src/services/experimentService.ts')).href;
const source = readFileSync(new URL(serviceUrl), 'utf8');
const entities = {};
const matches = (row, where) => Object.entries(where).every(([key, value]) =>
    value && value.values ? value.values.includes(row[key]) : value?.op === 'gte' ? row[key] >= value.value
        : value?.op === 'lte' ? row[key] <= value.value : value?.op === 'lt' ? row[key] < value.value : row[key] === value);
for (const name of ['ExperimentTable', 'ExperimentTaskTable', 'TaskAttemptTable', 'TaskAssetTable',
    'AssetVersionTable', 'EvaluationResultTable', 'EvaluationMetricTable', 'ExperimentLogSourceTable', 'ExperimentLogChunkTable',
    'ExperimentEventTable', 'TrainingReservationTable']) {
    entities[name] = class {
        static rows = [];
        static queries = [];
        static async findBy(where) { this.queries.push(where); return structuredClone(this.rows.filter(row => matches(row, where))); }
        static async find({ where, select, order, take }) {
            let rows = await this.findBy(where);
            if (order) rows.sort((a, b) => {
                for (const [key, direction] of Object.entries(order)) {
                    const delta = a[key] < b[key] ? -1 : a[key] > b[key] ? 1 : 0;
                    if (delta) return direction === 'DESC' ? -delta : delta;
                }
                return 0;
            });
            if (take) rows = rows.slice(0, take);
            return select ? rows.map(row => Object.fromEntries(select.filter(key => key in row).map(key => [key, row[key]]))) : rows;
        }
        static async findOneBy(where) { return (await this.findBy(where))[0] || null; }
        static async update(where, values) { for (const row of this.rows) if (matches(row, where)) Object.assign(row, values); }
        static createQueryBuilder(alias) {
            const table = this, selections = [], joins = [], orders = [];
            let condition;
            let parameters = {};
            const query = {
                select: (sql, as) => { selections.push(`${sql} AS ${as}`); return query; },
                addSelect: (sql, as) => { selections.push(`${sql} AS ${as}`); return query; },
                where: (sql, values) => { condition = sql; parameters = values; return query; },
                andWhere: (_sql, values) => { Object.assign(parameters, values); return query; },
                orderBy: (sql, direction) => { orders.length = 0; orders.push(`${sql} ${direction}`); return query; },
                addOrderBy: (sql, direction) => { orders.push(`${sql} ${direction}`); return query; },
                innerJoin: (target, name, on) => {
                    joins.push(`INNER JOIN ${target === entities.TaskAttemptTable ? 'attempt' : 'task'} ${name} ON ${on}`);
                    return query;
                },
                getOne: async () => {
                    if (parameters.experimentId) {
                        const db = new DatabaseSync(':memory:');
                        try {
                            db.exec('CREATE TABLE event(id TEXT, attemptId TEXT, type TEXT, payload TEXT, sequence INTEGER); CREATE TABLE attempt(id TEXT, taskId TEXT); CREATE TABLE task(id TEXT, experimentId TEXT)');
                            for (const row of table.rows) db.prepare('INSERT INTO event VALUES (?, ?, ?, ?, ?)').run(
                                row.id, row.attemptId, row.type, JSON.stringify(row.payload), row.sequence);
                            for (const row of entities.TaskAttemptTable.rows) db.prepare('INSERT INTO attempt VALUES (?, ?)').run(row.id, row.taskId);
                            for (const row of entities.ExperimentTaskTable.rows) db.prepare('INSERT INTO task VALUES (?, ?)').run(row.id, row.experimentId);
                            const row = db.prepare(`SELECT event.* FROM event ${alias} ${joins.join(' ')} WHERE ${condition} ORDER BY ${orders.join(', ')} LIMIT 1`).get(parameters);
                            return row ? { ...row, payload: JSON.parse(row.payload) } : null;
                        } finally { db.close(); }
                    }
                    return structuredClone(table.rows.filter(row => row.attemptId === parameters.attemptId &&
                    row.type === parameters.type && row.payload.operationId === parameters.operationId)
                    .sort((a, b) => b.sequence - a.sequence)[0] || null);
                },
                getRawMany: async () => {
                    const db = new DatabaseSync(':memory:');
                    try {
                        db.exec('CREATE TABLE attempt(id TEXT, taskId TEXT, configuration TEXT, runtime TEXT)');
                        for (const row of table.rows) db.prepare('INSERT INTO attempt VALUES (?, ?, ?, ?)').run(
                            row.id, row.taskId, JSON.stringify(row.configuration || {}), JSON.stringify(row.runtime || {}));
                        const placeholders = parameters.taskIds.map(() => '?').join(',');
                        return db.prepare(`SELECT ${selections.join(',')} FROM attempt ${alias} WHERE ${alias}.taskId IN (${placeholders})`).all(...parameters.taskIds);
                    } finally { db.close(); }
                },
            };
            return query;
        }
        static getRepository() { return { manager: {
            findOneBy: async (table, where) => (await table.findBy(where))[0],
            findOneByOrFail: async (table, where) => { const row = (await table.findBy(where))[0]; assert.ok(row); return row; },
            findOne: async (table, options) => (await table.find(options))[0],
            findBy: async (table, where) => table.findBy(where),
            create: (_table, values) => values,
            save: async row => {
                const table = Object.values(entities).find(candidate => candidate.rows.some(existing => existing.id === row.id));
                if (table) Object.assign(table.rows.find(existing => existing.id === row.id), structuredClone(row));
                else entities.ExperimentEventTable.rows.push(structuredClone(row));
                return row;
            },
            createQueryBuilder: (table, alias) => table.createQueryBuilder(alias),
            update: async (table, where, values) => table.update(where, values),
            transaction: async callback => callback(this.getRepository().manager),
        } }; }
    };
}
const schema = new Proxy(() => schema, { get: (_target, key) => key === 'parse' ? input => input : schema });
const mocks = { ...entities, In: values => ({ values }), z: schema,
    MoreThanOrEqual: value => ({ op: 'gte', value }), LessThanOrEqual: value => ({ op: 'lte', value }), LessThan: value => ({ op: 'lt', value }),
    experimentMetadataSchema: { parse: input => input },
    serviceOperationSchema: { parse: input => input },
    TRPCError: class extends Error { constructor({ message }) { super(message); } } };
const mockKey = '__experimentLineageTestMocks';
globalThis[mockKey] = mocks;
const imports = new Map();
for (const match of source.matchAll(/import\s*\{([^}]+)\}\s*from\s*'([^']+)';/g)) {
    if (match[2].startsWith('node:') || ['./experimentModelIdentity', './experimentListQuery', './experimentLogSearch', './experimentStatus'].includes(match[2])) continue;
    imports.set(match[2], match[1].split(',').map(name => name.trim()).filter(Boolean));
}
const hooks = registerHooks({
    resolve(specifier, context, next) {
        if (context.parentURL === serviceUrl && imports.has(specifier)) {
            return { url: `lineage-mock:${encodeURIComponent(specifier)}`, shortCircuit: true };
        }
        if (context.parentURL === serviceUrl && ['./experimentModelIdentity', './experimentListQuery', './experimentLogSearch', './experimentStatus'].includes(specifier)) {
            return next(new URL(`${specifier}.ts`, serviceUrl).href, context);
        }
        return next(specifier, context);
    },
    load(url, context, next) {
        if (url.startsWith('lineage-mock:')) {
            const names = imports.get(decodeURIComponent(url.slice('lineage-mock:'.length)));
            return { format: 'module', shortCircuit: true,
                source: names.map(name => `export const ${name} = globalThis.${mockKey}.${name};`).join('\n') };
        }
        return next(url, context);
    },
});
const { ExperimentService } = require(fileURLToPath(serviceUrl));
after(() => { hooks.deregister(); delete globalThis[mockKey]; });

const user = { id: 'owner' };
const { ListQuery, createListDatabase } = require('./experiment-list-query.test.cjs');
const add = (table, row) => { entities[table].rows.push(row); return row; };
const run = (id, kind, ownerUserId = user.id) => {
    add('ExperimentTable', { id, ownerUserId, name: id, status: 'succeeded', spec: {} });
    add('ExperimentTaskTable', { id: `task-${id}`, experimentId: id, kind });
    return add('TaskAttemptTable', { id: `attempt-${id}`, taskId: `task-${id}`, attemptNumber: 1,
        status: 'succeeded', runtime: { inputIntegrityAfterExecution: 'verified' } });
};
const asset = (id, content = id, extra = {}) => add('AssetVersionTable', { id, ownerUserId: user.id,
    kind: 'model', integrity: 'verified', fingerprint: content, logicalName: id, nodeId: 'node-a',
    manifest: { files: [{ path: 'weights', size: 4, sha256: content }] },
    location: { container: 'trainer', path: '/same-name' }, metadata: {}, ...extra });
const link = (attempt, model, direction, role = 'evaluated_model') => add('TaskAssetTable', {
    id: `${attempt.id}-${model.id}-${direction}`, taskId: attempt.taskId, attemptId: attempt.id,
    assetVersionId: model.id, direction, role, sourcePaths: [],
});
const score = (attempt, model) => add('EvaluationResultTable', { id: `score-${model.id}`, taskId: attempt.taskId,
    attemptId: attempt.id, modelVersionId: model.id, protocolComplete: true,
    expectedCount: 4, succeededCount: 4, failedCount: 0, comparisonKey: 'same-protocol' });
const fixture = () => {
    for (const table of Object.values(entities)) { table.rows = []; table.queries = []; }
    const train = run('train', 'train');
    const trained = asset('trained', 'trained-content', { kind: 'adapter' });
    link(train, trained, 'output', 'adapter');
    const evaluate = run('evaluate', 'evaluate');
    const copy = asset('copy', 'trained-content', { kind: 'checkpoint', nodeId: 'node-b',
        location: { container: 'evaluator', path: '/renamed' } });
    link(evaluate, copy, 'input');
    score(evaluate, copy);
    return { train, trained, evaluate, copy };
};

test('workflow state controls the experiment across stages, late receipts, stop and resume', async () => {
    for (const table of Object.values(entities)) { table.rows = []; table.queries = []; }
    const train = run('workflow', 'train');
    Object.assign(entities.ExperimentTable.rows[0], { workflowId: 'wf-1' });
    Object.assign(entities.ExperimentTaskTable.rows[0], { status: 'succeeded' });
    Object.assign(train, { nodeId: 'node-a', lastSequence: 0 });
    let sequence = 0;
    const append = (type, payload) => ExperimentService.append('node-a', [{ eventId: `wf-event-${++sequence}`,
        attemptId: train.id, sequence, occurredAt: new Date().toISOString(), type, payload }]);
    const state = () => entities.ExperimentTable.rows[0].status;
    const snapshot = (status, updatedAt) => ({ workflowId: 'wf-1', status, currentStage: 'evaluate', updatedAt });
    await append('workflow_status', snapshot('running', 1));
    assert.equal(state(), 'running');
    await append('status', { status: 'succeeded' });
    assert.equal(state(), 'running', 'A late train receipt must not complete the workflow');
    const evaluate = add('ExperimentTaskTable', { id: 'workflow-eval', experimentId: 'workflow', kind: 'evaluate', status: 'prepared' });
    await ExperimentService.refreshStatus(entities.ExperimentTable.getRepository().manager, entities.ExperimentTable.rows[0]);
    assert.equal(state(), 'running', 'Preparing a later stage must retain the workflow state');
    evaluate.status = 'succeeded';
    await append('workflow_status', snapshot('finished', 3));
    assert.equal(state(), 'succeeded');
    await append('workflow_status', snapshot('running', 2));
    assert.equal(state(), 'succeeded', 'Older snapshots must not overwrite newer workflow state');
    for (const [status, expected, timestamp] of [['failed', 'failed', 4], ['running', 'running', 5],
        ['stopping', 'running', 6], ['stopped', 'cancelled', 7], ['running', 'running', 8]]) {
        await append('workflow_status', snapshot(status, timestamp));
        assert.equal(state(), expected);
        assert.equal(entities.TaskAttemptTable.rows[0].status, 'succeeded');
    }
    await assert.rejects(append('workflow_status', { ...snapshot('finished', 9), workflowId: 'another' }), /does not match/);
});

test('stop history preserves deployment outcome, coalesces request/result and reads actual reservation release', async () => {
    for (const table of Object.values(entities)) { table.rows = []; table.queries = []; }
    const attempt = run('deploy', 'deploy');
    Object.assign(attempt, { nodeId: 'node-a', lastSequence: 0, resourceReservationId: 'reservation',
        startedAt: '2026-10-15T01:00:00Z', endedAt: '2026-10-15T01:01:00Z', error: null,
        runtime: { serviceRunId: 'run-1' } });
    const task = entities.ExperimentTaskTable.rows[0];
    Object.assign(task, { status: 'succeeded', spec: { serviceInstanceId: 'service-1' } });
    add('TrainingReservationTable', { id: 'reservation', status: 'running', releasedAt: null });
    const payload = { operationId: 'stop-1', operation: 'stop_service', serviceInstanceId: 'service-1', serviceRunId: 'run-1',
        status: 'requested', actorUserId: 'admin', trigger: 'admin', reservationId: 'reservation', releaseReady: null,
        result: '', returnCode: null };
    const request = { eventId: 'stop-request', attemptId: attempt.id, sequence: 1,
        occurredAt: '2026-10-15T02:00:00Z', type: 'service_operation', payload };
    const append = event => ExperimentService.append('node-a', [event]);
    await assert.rejects(append({ ...request, payload: { ...payload, status: 'succeeded' } }), /recorded request/);
    for (const changed of [{ serviceInstanceId: 'other' }, { serviceRunId: 'other' }, { reservationId: 'other' }]) {
        await assert.rejects(append({ ...request, payload: { ...payload, ...changed } }), /deployment instance/);
    }
    await assert.rejects(ExperimentService.append('other-node', [request]), /not assigned/);
    await append(request);
    const pending = await ExperimentService.summary(user, 'deploy');
    assert.equal(pending.serviceOperations[0].status, 'requested');
    assert.equal(pending.serviceOperations[0].completedAt, null);
    const completed = { ...request, eventId: 'stop-result', sequence: 2, occurredAt: '2026-10-15T02:00:04Z',
        payload: { ...payload, status: 'succeeded', result: 'PID and port released', releaseReady: true, returnCode: 0 } };
    await assert.rejects(append({ ...completed, payload: { ...completed.payload, actorUserId: 'someone-else' } }), /identity is immutable/);
    await assert.rejects(append({ ...completed, payload: { ...completed.payload, cause: 'restart' } }), /identity is immutable/);
    await append(completed);
    await append(completed);
    const summary = await ExperimentService.summary(user, 'deploy');
    assert.equal(summary.serviceOperations.length, 1);
    assert.equal(summary.serviceOperations[0].requestedAt, request.occurredAt);
    assert.equal(summary.serviceOperations[0].completedAt, completed.occurredAt);
    assert.equal(summary.serviceOperations[0].reservationStatus, 'running', 'Stopping the process is not releasing the Studio reservation');
    assert.equal(summary.experiment.status, 'succeeded');
    assert.equal(summary.attempts[0].endedAt, '2026-10-15T01:01:00Z');
    assert.equal(summary.attempts[0].error, null);
    assert.equal(entities.ExperimentEventTable.rows.length, 2, 'Lost ACK cannot duplicate an operation');
    Object.assign(entities.TrainingReservationTable.rows[0], { status: 'released', releasedAt: '2026-10-15T02:00:05Z' });
    const detail = await ExperimentService.detail(user, 'deploy');
    assert.equal(detail.serviceOperations[0].reservationStatus, 'released');
    assert.equal(detail.serviceOperations[0].releasedAt, '2026-10-15T02:00:05Z');
    await assert.rejects(ExperimentService.summary({ id: 'other-owner' }, 'deploy'), /unavailable/);
    await assert.rejects(append({ ...completed, eventId: 'changed', sequence: 3,
        payload: { ...completed.payload, status: 'failed' } }), /immutable/);
    const failedRequest = { ...request, eventId: 'stop-request-2', sequence: 3, payload: { ...payload, operationId: 'stop-2', cause: 'restart' } };
    await append(failedRequest);
    await append({ ...completed, eventId: 'stop-result-2', sequence: 4,
        payload: { ...failedRequest.payload, status: 'failed', releaseReady: false, result: 'port busy' } });
    assert.equal((await ExperimentService.summary(user, 'deploy')).experiment.status, 'succeeded');
    assert.equal((await ExperimentService.summary(user, 'deploy')).serviceOperations.length, 2);
    assert.equal((await ExperimentService.summary(user, 'deploy')).serviceOperations.find(item => item.operationId === 'stop-2').cause, 'restart');
});

test('training detail and comparison find standalone evaluation across container/type', async () => {
    fixture();
    for (const result of [await ExperimentService.detail(user, 'train'), await ExperimentService.compare(user, ['train'])]) {
        assert.deepEqual(result.linkedEvaluations.map(item => [item.trainingAttemptId, item.result.id, item.eligible]),
            [['attempt-train', 'score-copy', true]]);
    }
    const detail = await ExperimentService.detail(user, 'evaluate');
    assert.ok(detail.lineage.some(item => item.assetVersionId === 'copy' && item.relatedAssetVersionId === 'trained'
        && item.direction === 'output' && item.match === 'content'));
});
test('same path with changed content and base-model scores are excluded', async () => {
    const { train, evaluate } = fixture();
    const base = asset('base');
    link(train, base, 'input', 'base_model');
    link(evaluate, base, 'input', 'baseline_model');
    score(evaluate, base);
    const changed = asset('changed');
    link(evaluate, changed, 'input');
    score(evaluate, changed);
    assert.deepEqual((await ExperimentService.detail(user, 'train')).linkedEvaluations.map(item => item.result.id), ['score-copy']);
});
test('unverified copies and another user evaluations are excluded', async () => {
    const { evaluate, copy } = fixture();
    copy.integrity = 'incomplete';
    assert.equal((await ExperimentService.detail(user, 'train')).linkedEvaluations.length, 0);
    copy.integrity = 'verified';
    entities.ExperimentTable.rows.find(row => row.id === 'evaluate').ownerUserId = 'other';
    assert.equal((await ExperimentService.detail(user, 'train')).linkedEvaluations.length, 0);
    assert.ok(!(await ExperimentService.detail(user, 'train')).lineage.some(item => item.attemptId === evaluate.id));
    await assert.rejects(ExperimentService.detail({ id: 'other' }, 'train'), /unavailable/);
});
test('failed evaluation remains visible but cannot enter a comparable group', async () => {
    const { evaluate } = fixture();
    evaluate.status = 'failed';
    const [related] = (await ExperimentService.detail(user, 'train')).linkedEvaluations;
    assert.equal(related.eligible, false);
    assert.equal(related.ineligibleReason, 'execution-not-succeeded');
});
test('identical output from multiple training attempts preserves both associations', async () => {
    const { trained } = fixture();
    const retry = add('TaskAttemptTable', { id: 'attempt-retry', taskId: 'task-train', attemptNumber: 2 });
    link(retry, trained, 'output', 'adapter');
    const related = (await ExperimentService.detail(user, 'train')).linkedEvaluations;
    assert.deepEqual(related.map(item => item.trainingAttemptId).sort(), ['attempt-retry', 'attempt-train']);
});
test('merged then published model retains its training association', async () => {
    const { evaluate, copy } = fixture();
    const merged = asset('merged', 'merged-content', { metadata: { inputVersionIds: [copy.id] } });
    link(evaluate, merged, 'output', 'merged_model');
    const publish = run('publish', 'publish');
    link(publish, merged, 'input', 'source_model');
    const published = asset('published', 'published-content', { metadata: { inputVersionIds: [merged.id] } });
    link(publish, published, 'output', 'published_model');
    const benchmark = run('benchmark', 'benchmark');
    link(benchmark, published, 'input');
    score(benchmark, published);
    assert.ok((await ExperimentService.detail(user, 'train')).linkedEvaluations.some(item =>
        item.result.id === 'score-published' && item.trainingAttemptId === 'attempt-train'));
});
test('output metadata alone cannot invent a transformation without input links', async () => {
    const { trained, evaluate } = fixture();
    const unrelated = run('unrelated', 'merge');
    const invented = asset('invented', 'invented-content', { metadata: { inputVersionIds: [trained.id] } });
    link(unrelated, invented, 'output');
    link(evaluate, invented, 'input');
    score(evaluate, invented);
    assert.ok(!(await ExperimentService.detail(user, 'train')).linkedEvaluations.some(item => item.result.id === 'score-invented'));
});

test('model path and asset entry points include training and independent evaluation', async () => {
    const { trained, copy } = fixture();
    for (const model of [trained, copy]) {
        for (const records of [await ExperimentService.forAsset(user, model.id),
            await ExperimentService.forPath(user, model.nodeId, model.location.container, model.location.path)]) {
            assert.deepEqual(records.map(record => record.id).sort(), ['evaluate', 'train']);
        }
    }
});

test('model inventory parent path plus name reaches the training/evaluation records', async () => {
    const { trained } = fixture();
    trained.location.path = '/versions/immutable-trained';
    trained.metadata.sourcePath = '/home/workspace/models/base/Qwen';
    const inventoryModel = { path: '/home/workspace/models/base', name: 'Qwen',
        nodeId: 'node-a', containerName: 'trainer' };
    const lookup = modelPath => ExperimentService.forPath(user, inventoryModel.nodeId, inventoryModel.containerName, modelPath);
    assert.deepEqual(await lookup(inventoryModel.path), [], 'A category directory is not a specific model');
    assert.deepEqual((await lookup(modelExperimentPath(inventoryModel))).map(row => row.id).sort(), ['evaluate', 'train']);
    assert.deepEqual(await lookup(modelExperimentPath({ ...inventoryModel, name: 'AnotherModel' })), [],
        'Sibling models must not inherit the selected model records');
});
test('model path entry preserves recorded source aliases', async () => {
    const { trained } = fixture();
    trained.metadata.sourcePath = '/original';
    entities.TaskAssetTable.rows.find(row => row.assetVersionId === trained.id).sourcePaths.push('/alias');
    for (const sourcePath of ['/original', '/alias']) {
        assert.deepEqual((await ExperimentService.forPath(user, 'node-a', 'trainer', sourcePath))
            .map(record => record.id).sort(), ['evaluate', 'train']);
    }
    assert.deepEqual(await ExperimentService.forPath(user, 'node-b', 'trainer', '/original'), []);
    assert.deepEqual(await ExperimentService.forPath(user, 'node-a', 'evaluator', '/original'), []);
    assert.deepEqual(await ExperimentService.forPath(user, 'node-a', 'trainer', '/unknown'), []);
});
test('model entry excludes unverified copies, changed contents and other users', async () => {
    const { trained, copy } = fixture();
    const lookup = () => ExperimentService.forPath(user, trained.nodeId, trained.location.container, trained.location.path);
    copy.integrity = 'incomplete';
    assert.deepEqual((await lookup()).map(record => record.id), ['train']);
    copy.integrity = 'verified';
    copy.fingerprint = 'changed';
    assert.deepEqual((await lookup()).map(record => record.id), ['train']);
    copy.fingerprint = trained.fingerprint;
    const originalManifest = copy.manifest;
    copy.manifest = { files: [{ path: 'other-weights', size: 4, sha256: 'different' }] };
    assert.deepEqual((await lookup()).map(record => record.id), ['train']);
    copy.manifest = originalManifest;
    entities.ExperimentTable.rows.find(row => row.id === 'evaluate').ownerUserId = 'other';
    assert.deepEqual((await lookup()).map(record => record.id), ['train']);
    assert.deepEqual(await ExperimentService.forPath({ id: 'other' }, 'node-a', 'trainer', trained.location.path), []);
    await assert.rejects(ExperimentService.forAsset({ id: 'other' }, trained.id), /unavailable/);
});
test('shared source button matches verified dataset copies within the dataset family', async () => {
    const { trained, copy } = fixture();
    trained.kind = 'dataset';
    copy.kind = 'dataset';
    assert.deepEqual((await ExperimentService.forPath(user, trained.nodeId, trained.location.container, trained.location.path))
        .map(record => record.id).sort(), ['evaluate', 'train']);
    copy.kind = 'model';
    assert.deepEqual((await ExperimentService.forAsset(user, trained.id)).map(record => record.id), ['train']);
});

test('merged model entry traces training without including sibling runs using its base', async () => {
    const { train, trained, evaluate, copy } = fixture();
    const imported = run('base-import', 'publish');
    const base = asset('base');
    link(imported, base, 'output');
    link(train, base, 'input', 'base_model');
    trained.metadata.inputVersionIds = [base.id];
    const sibling = run('sibling', 'train');
    link(sibling, base, 'input', 'base_model');
    const unrelated = asset('sibling-output', 'sibling-content', { metadata: { inputVersionIds: [base.id] } });
    link(sibling, unrelated, 'output');
    const merged = asset('merged', 'merged-content', { metadata: { inputVersionIds: [copy.id] } });
    link(evaluate, merged, 'output');
    for (const scope of ['all', 'model']) {
        const records = await ExperimentService.forAsset(user, merged.id, scope);
        assert.deepEqual(records.map(row => row.id).sort(), ['base-import', 'evaluate', 'train']);
    }
    const detail = await ExperimentService.detail(user, 'evaluate');
    assert.ok(detail.lineage.some(row => row.assetVersionId === merged.id && row.experimentId === 'train' && row.match === 'upstream'));
});

test('upload, preprocessing, filtering and training are linked in both dataset entry directions', async () => {
    fixture();
    const upload = run('upload', 'preprocess');
    const original = asset('raw', 'raw-content', { kind: 'dataset' });
    link(upload, original, 'output', 'uploaded_dataset');
    const preprocess = run('preprocess', 'preprocess');
    const rawCopy = asset('raw-copy', 'raw-content', { kind: 'dataset', location: { container: 'processor', path: '/raw-copy' } });
    link(preprocess, rawCopy, 'input', 'source_dataset');
    const clean = asset('clean', 'clean-content', { kind: 'dataset', metadata: { inputVersionIds: [rawCopy.id] } });
    link(preprocess, clean, 'output');
    const filter = run('filter', 'preprocess');
    link(filter, clean, 'input', 'source_dataset');
    const filtered = asset('filtered', 'filtered-content', { kind: 'dataset', metadata: { inputVersionIds: [clean.id] } });
    link(filter, filtered, 'output');
    const trainer = run('dataset-training', 'train');
    link(trainer, filtered, 'input', 'training_dataset');
    for (const version of [original, filtered]) {
        for (const scope of ['all', 'dataset']) {
            assert.deepEqual((await ExperimentService.forAsset(user, version.id, scope)).map(row => row.id).sort(),
                ['dataset-training', 'filter', 'preprocess', 'upload']);
        }
    }
    const rows = (await ExperimentService.detail(user, 'upload')).lineage;
    assert.ok(rows.some(row => row.experimentId === 'dataset-training' && row.match === 'downstream'));
});

test('resume checkpoint and deployed model expose the original training record', async () => {
    const { train, trained } = fixture();
    const checkpoint = asset('checkpoint', 'checkpoint-content', { kind: 'checkpoint' });
    link(train, checkpoint, 'output', 'checkpoint');
    const resumed = run('resumed', 'train');
    resumed.resumeFromVersionId = checkpoint.id;
    link(resumed, checkpoint, 'input', 'resume_checkpoint');
    assert.ok((await ExperimentService.forAsset(user, checkpoint.id)).some(row => row.id === 'train'));
    const deploy = run('deploy', 'deploy');
    link(deploy, trained, 'input', 'served_model');
    assert.ok((await ExperimentService.detail(user, 'deploy')).lineage.some(row =>
        row.assetVersionId === trained.id && row.experimentId === 'train' && row.direction === 'output'));
});

test('report lookup follows recorded inputs to training and prevents forged/cross-owner ancestry', async () => {
    const { evaluate, copy } = fixture();
    const report = asset('report', 'report-content', { kind: 'report',
        metadata: { inputVersionIds: [copy.id, 'foreign-model', 'unlinked'] },
        location: { container: 'evaluator', path: '/reports/result.json' } });
    link(evaluate, report, 'output', 'report');
    const foreign = run('foreign', 'train', 'other');
    const foreignModel = asset('foreign-model', 'foreign-content', { ownerUserId: 'other' });
    link(foreign, foreignModel, 'output');
    link(evaluate, foreignModel, 'input');
    const stranger = run('stranger', 'train');
    link(stranger, asset('unlinked'), 'output');
    assert.deepEqual((await ExperimentService.forPath(user, 'node-a', 'evaluator', '/reports/result.json'))
        .map(row => row.id).sort(), ['evaluate', 'train']);
});

test('cyclic recorded transformations terminate and deduplicate records', async () => {
    fixture();
    const left = asset('left', 'left', { kind: 'dataset', metadata: { inputVersionIds: ['right'] } });
    const right = asset('right', 'right', { kind: 'dataset', metadata: { inputVersionIds: ['left'] } });
    for (const [id, input, output] of [['left-step', right, left], ['right-step', left, right]]) {
        const step = run(id, 'preprocess');
        link(step, input, 'input'); link(step, output, 'output');
    }
    assert.deepEqual((await ExperimentService.forAsset(user, left.id)).map(row => row.id).sort(), ['left-step', 'right-step']);
});
test('model entry queries pick up new evaluation records and deduplicate experiments', async () => {
    const { evaluate, trained } = fixture();
    const first = await ExperimentService.forAsset(user, trained.id);
    const extra = asset('another-copy', trained.fingerprint, { kind: 'checkpoint' });
    link(evaluate, extra, 'input');
    const later = run('later', 'evaluate');
    link(later, extra, 'input');
    assert.deepEqual(first.map(record => record.id).sort(), ['evaluate', 'train']);
    assert.deepEqual((await ExperimentService.forAsset(user, trained.id)).map(record => record.id).sort(),
        ['evaluate', 'later', 'train']);
});

test('model cards exclude data ancestry while experiment details retain it', async () => {
    const { train, trained } = fixture();
    const imported = run('dataset-import', 'preprocess');
    const original = asset('original-data', 'original-data', { kind: 'dataset' });
    link(imported, original, 'output', 'uploaded_dataset');
    const processing = run('processing', 'preprocess');
    link(processing, original, 'input', 'source_dataset');
    const processed = asset('processed-data', 'processed-data', { kind: 'dataset', metadata: { inputVersionIds: [original.id] } });
    link(processing, processed, 'output', 'processed_dataset');
    link(train, processed, 'input', 'training_dataset');
    trained.metadata.inputVersionIds = [processed.id];
    for (const records of [await ExperimentService.forAsset(user, trained.id, 'model'),
        await ExperimentService.forPath(user, trained.nodeId, trained.location.container, trained.location.path, 'model')]) {
        assert.deepEqual(records.map(record => record.id).sort(), ['evaluate', 'train']);
        assert.ok(records.every(record => record.assetRelations.every(relation => relation.relatedAssetKind !== 'dataset')));
    }
    const detail = await ExperimentService.detail(user, 'train');
    assert.ok(detail.lineage.some(relation => relation.assetVersionId === trained.id && relation.experimentId === 'processing'
        && relation.relatedAssetKind === 'dataset' && relation.match === 'upstream'));
    assert.deepEqual((await ExperimentService.forAsset(user, trained.id)).map(record => record.id).sort(),
        ['dataset-import', 'evaluate', 'processing', 'train']);
});

test('model cards retain real processing usage with an explicit association reason', async () => {
    const { trained } = fixture();
    const processing = run('direct-processing', 'preprocess');
    link(processing, trained, 'input', 'evaluated_model'); // Legacy records used this role.
    const records = await ExperimentService.forAsset(user, trained.id, 'model');
    const record = records.find(record => record.id === 'direct-processing');
    assert.ok(record);
    assert.deepEqual(associationLabels(record.assetRelations), ['used-for-preprocessing']);
    assert.equal(associationRole(record.assetRelations[0]), 'processing_model');
    assert.equal(entities.TaskAssetTable.rows.find(row => row.taskId === processing.taskId).role, 'evaluated_model',
        'Display interpretation must not rewrite immutable legacy records');
});

test('dataset cards do not inherit model imports used during processing', async () => {
    fixture();
    const modelImport = run('model-import', 'publish');
    const model = asset('processor-model');
    link(modelImport, model, 'output');
    const datasetImport = run('dataset-import', 'preprocess');
    const original = asset('original-data', 'original-data', { kind: 'dataset' });
    link(datasetImport, original, 'output', 'uploaded_dataset');
    const processing = run('processing', 'preprocess');
    link(processing, original, 'input', 'source_dataset');
    link(processing, model, 'input', 'processing_model');
    const processed = asset('processed-data', 'processed-data', { kind: 'dataset', metadata: { inputVersionIds: [original.id, model.id] } });
    link(processing, processed, 'output', 'processed_dataset');
    const training = run('dataset-training', 'train');
    link(training, processed, 'input', 'training_dataset');
    assert.deepEqual((await ExperimentService.forAsset(user, processed.id, 'dataset')).map(row => row.id).sort(),
        ['dataset-import', 'dataset-training', 'processing']);
    const detail = await ExperimentService.detail(user, 'processing');
    assert.ok(detail.lineage.some(relation => relation.experimentId === 'model-import' && relation.relatedAssetKind === 'model'));
});

test('report cards show report generation while model and data ancestry remain in detail', async () => {
    const { evaluate, copy } = fixture();
    const report = asset('report', 'report-content', { kind: 'report', metadata: { inputVersionIds: [copy.id] },
        location: { container: 'evaluator', path: '/reports/result.json' } });
    link(evaluate, report, 'output', 'evaluation_report');
    const records = await ExperimentService.forPath(user, 'node-a', 'evaluator', '/reports/result.json', 'report');
    assert.deepEqual(records.map(row => row.id), ['evaluate']);
    assert.deepEqual(associationLabels(records[0].assetRelations), ['produced-by']);
    assert.ok((await ExperimentService.detail(user, 'evaluate')).lineage.some(row =>
        row.assetVersionId === report.id && row.experimentId === 'train' && row.match === 'upstream'));
});

test('resource scopes cannot match a different family or other owner at the same path', async () => {
    const { trained } = fixture();
    assert.deepEqual(await ExperimentService.forPath(user, trained.nodeId, trained.location.container, trained.location.path, 'dataset'), []);
    assert.deepEqual(await ExperimentService.forPath({ id: 'other' }, trained.nodeId, trained.location.container, trained.location.path, 'model'), []);
    await assert.rejects(ExperimentService.forAsset({ id: 'other' }, trained.id, 'model'), /unavailable/);
});

test('association labels distinguish model/data sources and deduplicate repeated reasons', () => {
    assert.equal(associationLabel({ match: 'upstream', direction: 'output', relatedAssetKind: 'dataset' }), 'upstream-dataset');
    assert.equal(associationLabel({ match: 'upstream', direction: 'output', relatedAssetKind: 'adapter' }), 'upstream-model');
    assert.equal(associationLabel({ match: 'downstream', direction: 'input', relatedAssetKind: 'dataset' }), 'downstream-dataset');
    for (const [kind, label] of [['train', 'used-for-training'], ['evaluate', 'used-for-evaluation'],
        ['benchmark', 'used-for-evaluation'], ['merge', 'used-for-merge'], ['publish', 'used-for-publication'], ['deploy', 'used-for-inference']]) {
        assert.equal(associationLabel({ match: 'content', direction: 'input', taskKind: kind }), label);
    }
    assert.deepEqual(associationLabels([
        { match: 'version', direction: 'output' }, { match: 'content', direction: 'output' },
        { match: 'version', direction: 'input', taskKind: 'evaluate' },
    ]), ['produced-by', 'used-for-evaluation']);
    assert.equal(associationRole({ match: 'version', direction: 'input', taskKind: 'preprocess',
        relatedAssetKind: 'dataset', role: 'training_dataset' }), 'source_dataset');
});

test('many equivalent outputs batch fingerprint and owned-link queries within one detail request', async () => {
    const { train, trained } = fixture();
    for (let index = 0; index < 30; index++) {
        const copy = asset(`output-${index}`, trained.fingerprint, { kind: 'checkpoint' });
        link(train, copy, 'output', 'checkpoint');
    }
    const detail = await ExperimentService.detail(user, 'train');
    assert.equal(detail.assets.length, 31);
    assert.equal(entities.AssetVersionTable.queries.filter(query => query.fingerprint).length, 1,
        'Equal content must not be queried again for every root/direction or linked evaluation');
    assert.equal(entities.TaskAssetTable.queries.filter(query => query.assetVersionId && !query.direction).length, 1,
        'Owned associations for all root versions must be loaded in a single batch');
    assert.ok(detail.lineage.some(item => item.experimentId === 'evaluate'));
    const foreign = await ExperimentService.forPath({ id: 'other' }, 'node-a', 'trainer', '/same-name');
    assert.deepEqual(foreign, [], 'Request caches cannot carry another user\'s records');
});

test('standalone comparison groups include the exact reason for excluding an execution', async () => {
    const { evaluate } = fixture();
    evaluate.status = 'failed';
    const result = await ExperimentService.compare(user, ['train', 'evaluate']);
    assert.equal(result.groups['incomplete:score-copy'][0].ineligibleReason, 'execution-not-succeeded');
    evaluate.status = 'succeeded';
    assert.equal((await ExperimentService.compare(user, ['train', 'evaluate'])).groups['same-protocol'][0].ineligibleReason, null);
});

test('list service applies filters before database pagination and returns the filtered total', async () => {
    const db = createListDatabase();
    const original = entities.ExperimentTable.createQueryBuilder;
    entities.ExperimentTable.createQueryBuilder = () => new ListQuery(db);
    try {
        const result = await ExperimentService.list(user, { offset: 0, limit: 30, search: 'needle', taskKind: 'train' });
        assert.equal(result.total, 1); assert.deepEqual(result.items.map(row => row.id), ['late']);
        const page = await ExperimentService.list(user, { offset: 30, limit: 30, project: 'alpha', taskKind: 'train' });
        assert.ok(page.total > 30); assert.equal(page.items.length, page.total - 30);
    } finally { entities.ExperimentTable.createQueryBuilder = original; db.close(); }
});

test('lightweight comparison preserves associations and eligibility without lineage or manifests', async () => {
    fixture();
    const light = await ExperimentService.compare(user, ['train', 'evaluate'], true);
    assert.equal(light.linkedEvaluations.length, 1); assert.equal(light.linkedEvaluations[0].eligible, true);
    assert.ok(light.details.every(detail => detail.lineage.length === 0 && detail.assets.every(asset => !asset.manifest.files)));
    assert.ok(!light.linkedEvaluations[0].model.manifest.files);
    const full = await ExperimentService.detail(user, 'train');
    assert.ok(full.assets[0].manifest.files); assert.ok(full.lineage.length);
});

test('summary extracts only status configuration and compact assets; sections retain owned data', async () => {
    const { train } = fixture();
    train.configuration = { completeness: 'incomplete', actual: { big: 'x'.repeat(10000) } };
    train.runtime = { inputIntegrityAfterExecution: 'verified', archiveStatus: 'retrying', benchmarkReport: { big: 'x'.repeat(10000) } };
    const summary = await ExperimentService.summary(user, 'train');
    assert.deepEqual(summary.attempts[0].configuration, { completeness: 'incomplete' });
    assert.equal(summary.attempts[0].runtime.archiveStatus, 'retrying');
    assert.equal(summary.attempts[0].runtime.benchmarkReport, undefined);
    assert.deepEqual(summary.assets[0].manifest, {}); assert.equal(summary.lineage.length, 0);
    assert.ok((await ExperimentService.configuration(user, 'train')).attempts[0].configuration.actual);
    const evaluation = await ExperimentService.evaluations(user, 'train');
    assert.equal(evaluation.linkedEvaluations.length, 1); assert.equal(evaluation.linkedEvaluations[0].model.manifest.files, undefined);
    train.runtime.evaluationObservation = 'pending';
    assert.equal((await ExperimentService.assets(user, 'train')).observations[0].evaluationObservation, undefined);
    train.runtime.evaluationObservation = { source: 'medical_monitor', items: [] };
    assert.deepEqual((await ExperimentService.assets(user, 'train')).observations[0].evaluationObservation,
        { source: 'medical_monitor', items: [] });
    for (const method of ['summary', 'configuration', 'assets', 'evaluations'])
        await assert.rejects(ExperimentService[method]({ id: 'other' }, 'train'), /unavailable/);
});

test('metadata updates only annotations and rejects another owner', async () => {
    fixture();
    const row = entities.ExperimentTable.rows[0], before = structuredClone(row);
    await ExperimentService.updateMetadata(user, { experimentId: row.id, notes: '备注', tags: ['medical'] });
    assert.deepEqual(row, { ...before, notes: '备注', tags: ['medical'] });
    await assert.rejects(ExperimentService.updateMetadata({ id: 'other' }, { experimentId: row.id, notes: 'tamper', tags: [] }), /unavailable/);
    assert.equal(row.notes, '备注');
});

test('business-log search and context keep a fixed snapshot, reject foreign owners and hidden sources', async () => {
    const { train } = fixture();
    add('ExperimentLogSourceTable', { id: 'business', attemptId: train.id, kind: 'training', nextOffset: 200, status: 'streaming' });
    add('ExperimentLogSourceTable', { id: 'hidden', attemptId: train.id, kind: 'execution', nextOffset: 0 });
    add('ExperimentLogChunkTable', { sourceId: 'business', startOffset: 0, endOffset: 100, content: '中文🙂 token=[REDACTED] ER' });
    add('ExperimentLogChunkTable', { sourceId: 'business', startOffset: 100, endOffset: 200, content: 'ROR failure\n' });
    const input = { experimentId: 'train', attemptId: train.id, sourceId: 'business', keyword: 'ERROR', caseSensitive: false };
    const first = await ExperimentService.searchLogs(user, input);
    assert.equal(first.matches.length, 1); assert.equal(first.snapshotEndOffset, 200);
    entities.ExperimentLogSourceTable.rows[0].nextOffset = 300;
    add('ExperimentLogChunkTable', { sourceId: 'business', startOffset: 200, endOffset: 300, content: 'ERROR later\n' });
    assert.equal((await ExperimentService.searchLogs(user, { ...input, snapshotEndOffset: first.snapshotEndOffset })).matches.length, 1);
    assert.equal((await ExperimentService.searchLogs(user, input)).matches.length, 2);
    const contextInput = { ...input, position: first.matches[0].position, matchLength: 5, snapshotEndOffset: 200 };
    const context = await ExperimentService.logContext(user, contextInput);
    assert.match(context.chunks.map(block => block.content).join(''), /ERROR/);
    assert.ok(!context.chunks.some(block => block.content.includes('later')));
    for (const method of ['searchLogs', 'logContext'])
        await assert.rejects(ExperimentService[method]({ id: 'other' }, method === 'searchLogs' ? input : contextInput), /unavailable/);
    await assert.rejects(ExperimentService.businessLogSource(user, 'train', train.id, 'hidden'), /unavailable/);
    await assert.rejects(ExperimentService.businessLogSource(user, 'train', 'another-attempt', 'business'), /unavailable/);
    await assert.rejects(ExperimentService.searchLogs(user, { ...input, cursor: { startOffset: 0, characterOffset: 9999 } }), /Invalid log cursor/);
    const exported = [];
    for await (const chunk of ExperimentService.archivedLogChunks('business', 200)) exported.push(chunk.content);
    assert.equal(exported.join(''), '中文🙂 token=[REDACTED] ERROR failure\n');
});
