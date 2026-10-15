// Node 24+, no dependencies or generated files.
const assert = require('node:assert/strict');
const { test } = require('node:test');
const { experimentStatus } = require('../src/services/experimentStatus.ts');
const tasks = (...statuses) => statuses.map(status => ({ status }));

test('workflow stays running across successful stages and gaps before the next task exists', () => {
    for (const records of [tasks('succeeded'), tasks('succeeded', 'prepared'), tasks('succeeded', 'running'),
        tasks('succeeded', 'succeeded'), tasks('succeeded', 'unknown')]) {
        assert.equal(experimentStatus(records, 'wf-1', 'running'), 'running');
    }
});

test('only the workflow lifecycle determines completion, failure, stopping and resume', () => {
    assert.equal(experimentStatus(tasks('succeeded'), 'wf-1', 'finished'), 'succeeded');
    assert.equal(experimentStatus(tasks('succeeded'), 'wf-1', 'failed'), 'failed');
    assert.equal(experimentStatus(tasks('succeeded'), 'wf-1', 'stopping'), 'running');
    assert.equal(experimentStatus(tasks('succeeded'), 'wf-1', 'stopped'), 'cancelled');
    assert.equal(experimentStatus(tasks('failed'), 'wf-1', 'running'), 'running');
});

test('legacy workflow stage success alone cannot confirm overall success; standalone aggregation is preserved', () => {
    assert.equal(experimentStatus(tasks('succeeded'), 'wf-1'), 'unknown');
    assert.equal(experimentStatus(tasks('succeeded', 'running'), 'wf-1'), 'running');
    for (const [statuses, expected] of [[['succeeded'], 'succeeded'], [['succeeded', 'running'], 'running'],
        [['failed', 'prepared'], 'prepared'], [['failed'], 'failed'], [['cancelled'], 'cancelled'], [['unknown'], 'unknown']]) {
        assert.equal(experimentStatus(tasks(...statuses), null), expected);
    }
});
