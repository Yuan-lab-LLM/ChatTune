// Node 24+, no dependencies or generated files.
const assert = require('node:assert/strict');
const { test } = require('node:test');
const { standaloneComparisonGroups } = require('../../client/src/pages/DashboardPage/RunPage/ExperimentManager/comparisonGroups.ts');
const { mergeLogWindow, LOG_WINDOW_CHUNKS, LOG_WINDOW_CHARACTERS } = require('../../client/src/pages/DashboardPage/RunPage/ExperimentManager/logWindow.ts');
const { hasEvaluationTab } = require('../../client/src/pages/DashboardPage/RunPage/ExperimentManager/evaluationVisibility.ts');
const { filterBenchmarkSamples, displayMetric, benchmarkReport } = require('../../client/src/pages/DashboardPage/RunPage/ExperimentManager/benchmarkReport.ts');
const { experimentName, executionLabel } = require('../../client/src/pages/DashboardPage/RunPage/ExperimentManager/displayLabels.ts');
const en = require('../../client/src/i18n/en.json');
const zh = require('../../client/src/i18n/zh.json');
const translate = dictionary => (key, options = {}) => {
    const value = key.split('.').reduce((item, name) => item?.[name], dictionary) ?? options.defaultValue ?? key;
    return value.replace(/\{\{(\w+)\}\}/g, (_, name) => options[name]);
};

test('workflow titles describe the pipeline in both languages while preserving custom and independent names', () => {
    const record = { name: 'batch_train_lora', workflowId: 'workflow-1', spec: { kind: 'train', script: 'batch_train_lora' } };
    assert.equal(experimentName(translate(en), record), 'One-click workflow · LoRA SFT training');
    assert.equal(experimentName(translate(zh), record), '一键工作流 · LoRA SFT 训练');
    assert.equal(experimentName(translate(en), { ...record, workflowId: null }), 'batch_train_lora');
    assert.equal(experimentName(translate(en), { ...record, name: 'My training proposal' }), 'My training proposal');
    assert.equal(record.name, 'batch_train_lora');
});

test('DPO and PT workflow titles follow the actual recorded script across languages', () => {
    const cases = [
        ['dpo_train_launcher', 'DPO 训练', 'DPO training'],
        ['train_multinode_dpo_pipeline', '多节点 DPO 训练', 'Multi-node DPO training'],
        ['batch_train_pretrain_lora', 'LoRA 预训练', 'LoRA pretraining'],
        ['batch_train_pretrain_full', '全参数预训练', 'Full-parameter pretraining'],
    ];
    for (const [script, chinese, english] of cases) {
        const record = { name: script, workflowId: 'workflow-1', spec: { kind: 'train', script } };
        assert.equal(experimentName(translate(zh), record), `一键工作流 · ${chinese}`);
        assert.equal(experimentName(translate(en), record), `One-click workflow · ${english}`);
        assert.equal(experimentName(translate(zh), { ...record, workflowId: null }), script);
        assert.equal(experimentName(translate(zh), { ...record, name: '自定义工作流' }), '自定义工作流');
    }
});

test('task-local execution numbering identifies training, evaluation and retries even before runtime metadata arrives', () => {
    const tasks = [{ id: 'train', kind: 'train' }, { id: 'eval', kind: 'evaluate' }];
    const attempt = { taskId: 'train', attemptNumber: 1, runtime: {} };
    assert.equal(executionLabel(translate(en), attempt, tasks), 'Training · Execution #1');
    assert.equal(executionLabel(translate(en), { ...attempt, taskId: 'eval' }, tasks), 'Evaluation · Execution #1');
    assert.equal(executionLabel(translate(en), { ...attempt, attemptNumber: 2 }, tasks), 'Training · Execution #2');
    assert.equal(executionLabel(translate(en), { ...attempt, runtime: { kind: 'benchmark' } }), 'Benchmark · Execution #1');
    assert.equal(executionLabel(translate(en), attempt), 'Execution #1');
});

test('repeated deployments have distinct labels without changing task-local retry numbers', () => {
    const tasks = [{ id: 'deploy-one', kind: 'deploy' }, { id: 'deploy-two', kind: 'deploy' }];
    const attempt = { taskId: 'deploy-two', attemptNumber: 1, runtime: {} };
    assert.equal(executionLabel(translate(en), attempt, tasks), 'Service deployment #2 · Execution #1');
    assert.equal(executionLabel(translate(zh), { ...attempt, taskId: 'deploy-one' }, tasks), '服务部署 #1 · 执行 #1');
    assert.equal(executionLabel(translate(en), { ...attempt, attemptNumber: 2 }, tasks), 'Service deployment #2 · Execution #2');
});

test('benchmark filters search sample contents and keep request failures distinct from wrong answers', () => {
    const sample = (index, status, question) => ({ index, status, question, id: '', subset: '', expected: '', predicted: '', response: '', error: '', metrics: {} });
    const rows = [sample(1, 'incorrect', 'question one'), { ...sample(2, 'failed', 'question two'), error: 'timeout' }];
    assert.deepEqual(filterBenchmarkSamples(rows, '', 'failed').map(row => row.index), [2]);
    assert.deepEqual(filterBenchmarkSamples(rows, 'TIMEOUT', 'all').map(row => row.index), [2]);
    assert.deepEqual(filterBenchmarkSamples(rows, 'one', 'incorrect').map(row => row.index), [1]);
    assert.equal(rows.length, 2, 'Filtering must not mutate cached report data');
});

test('ratio metrics display as percentages without changing points or already recorded percents', () => {
    assert.deepEqual(displayMetric(0.2242, 'ratio'), { value: 22.42, unit: '%' });
    assert.deepEqual(displayMetric(52.97, 'percent'), { value: 52.97, unit: 'percent' });
    assert.deepEqual(displayMetric(67, 'score'), { value: 67, unit: 'score' });
    assert.equal(benchmarkReport(null), undefined);
    assert.equal(benchmarkReport({ schemaVersion: 1, samples: [] }), undefined);
});

const experiment = (kind, overrides = {}) => ({
    experiment: { spec: { kind } }, tasks: [{ kind }], evaluations: [], linkedEvaluations: [], ...overrides,
});

test('processing, merge, publication and deployment never show an evaluation tab', () => {
    for (const kind of ['preprocess', 'merge', 'publish', 'deploy']) {
        assert.equal(hasEvaluationTab(experiment(kind)), false, kind);
        assert.equal(hasEvaluationTab(experiment(kind, { tasks: [{ kind }, { kind: 'benchmark' }] })), false, kind);
        assert.equal(hasEvaluationTab(experiment(kind, {
            evaluations: [{ id: 'result' }], linkedEvaluations: [{ result: { id: 'linked' } }],
        })), false, kind);
    }
});

test('evaluation tasks retain their tab before results arrive', () => {
    for (const kind of ['evaluate', 'benchmark']) {
        assert.equal(hasEvaluationTab(experiment(kind)), true);
        assert.equal(hasEvaluationTab(experiment(kind, { tasks: [] })), true);
        assert.equal(hasEvaluationTab(experiment(undefined, { tasks: [{ kind }] })), true);
    }
});

test('training shows evaluation only when it has evaluation tasks or results', () => {
    assert.equal(hasEvaluationTab(undefined), false);
    assert.equal(hasEvaluationTab(experiment('train')), false);
    assert.equal(hasEvaluationTab(experiment('train', { evaluations: [{ id: 'result' }] })), true);
    assert.equal(hasEvaluationTab(experiment('train', { linkedEvaluations: [{ result: { id: 'linked' } }] })), true);
    assert.equal(hasEvaluationTab(experiment('train', { tasks: [{ kind: 'train' }, { kind: 'evaluate' }] })), true);
});

test('mixed selection retains independent evaluations without duplicating linked results', () => {
    const linked = { result: { id: 'linked' } };
    const standalone = { result: { id: 'standalone' } };
    const incomplete = { result: { id: 'incomplete' } };
    const groups = { matching: [linked, standalone], 'incomplete:incomplete': [incomplete], duplicate: [linked] };
    assert.deepEqual(standaloneComparisonGroups(groups, ['linked', 'linked']), {
        matching: [standalone], 'incomplete:incomplete': [incomplete],
    });
    assert.deepEqual(standaloneComparisonGroups(groups, []), groups);
    assert.equal(groups.matching.length, 2, 'Selection must not mutate cached query data');
});

test('switching training attempts restores evaluations that are no longer displayed', () => {
    const groups = { matching: [{ result: { id: 'old-attempt' } }, { result: { id: 'new-attempt' } }] };
    assert.deepEqual(standaloneComparisonGroups(groups, ['new-attempt']).matching.map(entry => entry.result.id), ['old-attempt']);
    assert.deepEqual(standaloneComparisonGroups(groups, ['old-attempt']).matching.map(entry => entry.result.id), ['new-attempt']);
});

const chunk = (index, content = `${index}\n`) => ({ startOffset: index * 100000, endOffset: (index + 1) * 100000, content });
test('a long running stream keeps bounded state and the newest byte cursor', () => {
    let window = [];
    for (let page = 0; page < 1000; page++) {
        window = mergeLogWindow(window, Array.from({ length: 8 }, (_, index) => chunk(page * 8 + index)), 'newer');
        assert.ok(window.length <= LOG_WINDOW_CHUNKS);
    }
    assert.equal(window[0].startOffset, chunk(8000 - LOG_WINDOW_CHUNKS).startOffset);
    assert.equal(window.at(-1).endOffset, chunk(7999).endOffset);
});

test('character limit retains complete Unicode chunks with original byte offsets', () => {
    const items = Array.from({ length: 64 }, (_, index) => chunk(index, '中文🙂'.repeat(8192)));
    const window = mergeLogWindow([], items, 'newer');
    assert.equal(window.reduce((sum, item) => sum + item.content.length, 0), LOG_WINDOW_CHARACTERS);
    assert.equal(window.at(-1), items.at(-1));
    assert.ok(window.length < LOG_WINDOW_CHUNKS);
    assert.ok(window.every(item => items.includes(item)), 'No chunk content or offsets are sliced');
});

test('browsing backwards then forwards reloads evicted chunks without gaps or duplicates', () => {
    const tail = Array.from({ length: 64 }, (_, index) => chunk(64 + index));
    const older = Array.from({ length: 8 }, (_, index) => chunk(56 + index));
    const history = mergeLogWindow(tail, older.reverse(), 'older');
    assert.deepEqual(history.map(item => item.startOffset), Array.from({ length: 64 }, (_, index) => chunk(56 + index).startOffset));
    const restored = mergeLogWindow(history, tail.slice(-8), 'newer');
    assert.deepEqual(restored, tail);
    assert.equal(mergeLogWindow(restored, [], 'newer'), restored);
    assert.equal(mergeLogWindow(restored, [restored.at(-1)], 'newer'), restored);
});

test('jumping to the latest starts a fresh contiguous window instead of joining separated pages', () => {
    const latest = Array.from({ length: 8 }, (_, index) => chunk(1000 + index));
    assert.deepEqual(mergeLogWindow([], latest, 'newer'), latest);
});
