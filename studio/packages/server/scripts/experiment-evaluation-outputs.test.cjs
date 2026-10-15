// Node 24+, no dependencies or generated files.
const assert = require('node:assert/strict');
const { test } = require('node:test');
const { evaluationOutputs } = require('../../client/src/pages/DashboardPage/RunPage/ExperimentManager/evaluationOutputs.ts');

const attempt = items => ({ id: 'attempt-1', attemptNumber: 1, runtime: {
    evaluationObservation: { source: 'medical_monitor', currentStage: 'record_to_diagnose', items },
} });
const file = (name, path, exists = true) => ({ name, path, exists, status: exists ? 'finished' : 'pending', scores: {} });

test('running evaluation shows generated scores and pending report with no asset versions', () => {
    const groups = evaluationOutputs([attempt([
        { ...file('common_eval', '/common/results.log'), scores: { average: 51.97 } },
        { ...file('medical_eval', '/medical/results.log'), scores: { average: 50.39 } },
        { ...file('record_to_diagnose', '/diagnosis.json', false), status: 'running', logError: true },
        file('evaluation_report', '/report.txt', false),
    ])], [], []);
    assert.equal(groups[0].currentStage, 'record_to_diagnose');
    assert.equal(groups[0].items.filter(item => item.exists && !item.archivedVersionId).length, 2);
    assert.deepEqual(groups[0].items[0].scores, [['average', 51.97]]);
    assert.equal(groups[0].items[2].logError, true);
    assert.equal(groups[0].items[3].exists, false);
});

test('matching source path is archived only with an output link for the same attempt', () => {
    const attempts = [attempt([file('report', '/report.txt')])];
    const assets = [{ id: 'av-1', metadata: { sourcePath: '/report.txt' }, location: { container: 'trainer', path: '/snapshot' } }];
    const link = { attemptId: 'other-attempt', assetVersionId: 'av-1', direction: 'output' };
    assert.equal(evaluationOutputs(attempts, assets, [link])[0].items[0].archivedVersionId, undefined);
    assert.equal(evaluationOutputs(attempts, assets, [{ ...link, attemptId: 'attempt-1', direction: 'input' }])[0].items[0].archivedVersionId, undefined);
    assert.equal(evaluationOutputs(attempts, assets, [{ ...link, attemptId: 'attempt-1' }])[0].items[0].archivedVersionId, 'av-1');
});

test('deduplicates paths, ignores malformed telemetry, and preserves separate executions', () => {
    const items = [file('stage', '/file'), file('artifact', '/file'), null, { name: 'missing-path' }];
    const groups = evaluationOutputs([attempt(items), { ...attempt(items), id: 'attempt-2', attemptNumber: 2 }], [], []);
    assert.equal(groups.length, 2);
    assert.equal(groups[0].items.length, 1);
    assert.equal(groups[0].items[0].name, 'stage');
    assert.deepEqual(evaluationOutputs([{ id: 'legacy', attemptNumber: 1, runtime: {} }], [], []), []);
});
