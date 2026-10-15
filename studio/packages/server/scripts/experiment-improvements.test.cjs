const assert = require('node:assert/strict');
const { test } = require('node:test');
const { searchArchivedLogs } = require('../src/services/experimentLogSearch.ts');
const { metricDelta, parameterFields, canonicalValue, latestResult } = require('../../client/src/pages/DashboardPage/RunPage/ExperimentManager/comparisonValues.ts');
const { experimentRefreshInterval, experimentQueryOptions } = require('../../client/src/pages/DashboardPage/RunPage/ExperimentManager/queryPolicy.ts');
const { modelShardSummary } = require('../src/services/experimentModelIdentity.ts');
const blocks = contents => contents.map((content, index) => ({ startOffset: index * 1000, endOffset: (index + 1) * 1000, content }));
async function* archive(chunks, cursor) { for (const chunk of chunks) if (!cursor || chunk.startOffset >= cursor.startOffset) yield chunk; }
const search = (chunks, extra = {}) => searchArchivedLogs(archive(chunks, extra.cursor), { keywords: ['ERROR'], caseSensitive: false, ...extra });

test('search crosses blocks and preserves Unicode character positions after redaction', async () => {
    const chunks = blocks(['中文🙂 token=[REDACTED]\nER', 'ROR 出错🙂\nnext']);
    const result = await search(chunks);
    assert.equal(result.matches.length, 1);
    assert.deepEqual(result.matches[0].position, { startOffset: 0, characterOffset: chunks[0].content.indexOf('ER') });
    assert.equal(result.matches[0].text, 'ERROR');
    assert.match(result.matches[0].after, /出错🙂/);
});

test('search treats regex characters literally and supports case-sensitive matching', async () => {
    const chunks = blocks(['error ERROR [a.*] エラー']);
    assert.equal((await search(chunks)).matches.length, 2);
    assert.equal((await search(chunks, { caseSensitive: true })).matches.length, 1);
    assert.equal((await search(chunks, { keywords: ['[a.*]'] })).matches.length, 1);
});

test('paged searches resume without duplicates, including cross-block results', async () => {
    const chunks = blocks(['ERROR ER', 'ROR ERROR ER', 'ROR tail']);
    const all = [];
    let cursor;
    for (let index = 0; index < 10; index++) {
        const page = await search(chunks, { limit: 1, cursor });
        all.push(...page.matches); cursor = page.nextCursor;
        if (!cursor) break;
    }
    assert.equal(all.length, 4);
    assert.equal(new Set(all.map(item => JSON.stringify(item.position))).size, 4);
});

test('bounded scans make progress through many tiny blocks and find long split keywords', async () => {
    const chunks = blocks([...('x'.repeat(200) + 'a'.repeat(200) + 'tail')]);
    const all = [];
    let cursor;
    for (let index = 0; index < 20; index++) {
        const page = await search(chunks, { keywords: ['a'.repeat(200)], maxChunks: 10, cursor });
        if (page.nextCursor) assert.notDeepEqual(page.nextCursor, cursor);
        all.push(...page.matches); cursor = page.nextCursor;
        if (!cursor) break;
    }
    assert.equal(all.length, 1);
    assert.deepEqual(all[0].position, { startOffset: 200000, characterOffset: 0 });
});

test('search scans beyond the displayed log window with bounded pages', async () => {
    const chunks = blocks(Array.from({ length: 300 }, (_, index) => 'line '.repeat(100) + (index === 260 ? 'ERROR' : '')));
    let cursor, matches = [];
    for (let index = 0; index < 10; index++) {
        const page = await search(chunks, { cursor }); matches.push(...page.matches); cursor = page.nextCursor;
        if (!cursor) break;
    }
    assert.equal(matches.length, 1);
    assert.equal(matches[0].position.startOffset, 260000);
});

test('Unicode case folding retains original indexes rather than lowercased string offsets', async () => {
    const chunks = blocks(['İ中文🙂ERROR\nerror']);
    const result = await search(chunks);
    assert.deepEqual(result.matches.map(item => item.position.characterOffset), [5, 11]);
});

test('baseline deltas respect higher/lower/neutral and ratio percentage points', () => {
    const delta = metricDelta(.75, .72, 'higher', 'ratio', true);
    assert.ok(Math.abs(delta.value - 3) < 1e-10); assert.equal(delta.percentagePoints, true); assert.equal(delta.trend, 'improved');
    assert.equal(metricDelta(.75, .72, 'higher', 'ratio', false).percentagePoints, false);
    assert.equal(metricDelta(2, 3, 'lower', 'seconds', true).trend, 'improved');
    assert.equal(metricDelta(4, 3, 'lower', 'seconds', true).trend, 'regressed');
    assert.equal(metricDelta(4, 3, 'neutral', 'count', true).trend, 'neutral');
    assert.equal(metricDelta(4, undefined, 'higher', 'ratio', true), null);
    assert.equal(metricDelta(0, 0, 'higher', 'ratio', true).trend, 'unchanged');
});

test('configuration sources remain distinct and missing actual configuration never falls back', () => {
    const spec = { requested: { lr: .1 }, resolved: { lr: .2, output_dir: '/attempt-a', nested: { work_dir: '/work' } } };
    assert.equal(parameterFields('requested', spec, {}, {}).fields.lr, .1);
    const resolved = parameterFields('resolved', spec, {}, {});
    assert.equal(resolved.fields.lr, .2); assert.equal(resolved.runtime.output_dir, '/attempt-a');
    assert.equal(resolved.runtime['nested.work_dir'], '/work'); assert.ok(!('output_dir' in resolved.fields));
    assert.deepEqual(parameterFields('actual', spec, {}, {}).fields, {});
    assert.equal(parameterFields('actual', spec, { actual: { source: 'journal', lr: .3 } }, {}).fields.lr, .3);
    assert.equal(parameterFields('environment', spec, {}, { python: '3.10' }).fields.python, '3.10');
    assert.equal(canonicalValue({ a: 1, b: 2 }), canonicalValue({ b: 2, a: 1 }));
});

test('default evaluation uses latest time and a stable result-ID tie break, never highest score', () => {
    const items = [{ result: { id: 'old', createdAt: '2026-01-01', score: 1 } },
        { result: { id: 'new-a', createdAt: '2026-02-01', score: .5 } },
        { result: { id: 'new-b', createdAt: '2026-02-01', score: .3 } }];
    assert.equal(latestResult(items).result.id, 'new-b');
});

test('history polls at 60s, active runs at 15s, unchanged uncertain states back off', () => {
    const now = Date.parse('2026-10-15T10:00:00Z');
    const old = '2026-10-15T09:00:00Z';
    assert.equal(experimentRefreshInterval([{ status: 'succeeded' }], now), 60000);
    assert.equal(experimentRefreshInterval([{ status: 'draft' }], now), 60000);
    assert.equal(experimentRefreshInterval([{ status: 'running', updatedAt: old }], now), 15000);
    assert.equal(experimentRefreshInterval([{ status: 'unknown', updatedAt: old }], now), 60000);
    assert.equal(experimentRefreshInterval([{ status: 'failed', updatedAt: old, runtime: { archiveStatus: 'retrying' } }], now), 60000);
    assert.equal(experimentRefreshInterval([{ status: 'unknown', updatedAt: new Date(now).toISOString() }], now), 15000);
    assert.equal(experimentQueryOptions.staleTime, 60000); assert.equal(experimentQueryOptions.gcTime, 600000);
});

test('compact evaluation assets retain missing-model-shard warnings', () => {
    const manifest = { files: [{ path: 'model-00001-of-00002.safetensors' }, { path: 'model.safetensors.index.json' }] };
    assert.deepEqual(modelShardSummary(manifest), { expected: 2, present: 1, complete: false });
    manifest.files.push({ path: 'model-00002-of-00002.safetensors' });
    assert.deepEqual(modelShardSummary(manifest), { expected: 2, present: 2, complete: true });
});
