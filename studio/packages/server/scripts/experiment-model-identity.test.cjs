// Node 24+; no dependencies, compilation, or files written by this test.
const assert = require('node:assert/strict');
const { test } = require('node:test');
const { equivalentModelVersions, modelContentKey } = require('../src/services/experimentModelIdentity.ts');

const model = (id, overrides = {}) => ({
    id, ownerUserId: 'owner', kind: 'adapter', integrity: 'verified', fingerprint: 'a'.repeat(64),
    manifest: { files: [{ path: 'weights.bin', size: 4, sha256: 'b'.repeat(64) }] }, ...overrides,
});
const matches = (left, right) => equivalentModelVersions([left, right]).get(left.id).map(asset => asset.id);

test('same content matches across asset kinds, names, nodes and containers', () => {
    const original = model('trained', { nodeId: 'a', location: { container: 'trainer', path: '/train' } });
    for (const kind of ['model', 'adapter', 'checkpoint', 'base_model']) {
        const copy = model('copy', { kind, nodeId: 'b', logicalName: 'renamed',
            location: { container: 'evaluator', path: '/eval' } });
        assert.deepEqual(matches(original, copy), ['trained', 'copy']);
    }
});
test('different users never share a content identity', () => {
    assert.deepEqual(matches(model('trained'), model('copy', { ownerUserId: 'other' })), ['trained']);
});
test('changed content does not match even at the same location', () => {
    assert.deepEqual(matches(model('trained'), model('copy', { fingerprint: 'c'.repeat(64) })), ['trained']);
});
test('matching fingerprint still requires the exact file manifest', () => {
    assert.deepEqual(matches(model('trained'), model('copy', {
        manifest: { files: [{ path: 'weights.bin', size: 5, sha256: 'b'.repeat(64) }] },
    })), ['trained']);
});
test('missing or unverified content never matches another version', () => {
    for (const overrides of [{ integrity: 'incomplete' }, { integrity: 'missing' }, { fingerprint: null },
        { manifest: {} }, { manifest: { files: [] } }, { manifest: { files: [null] } }]) {
        assert.equal(modelContentKey(model('copy', overrides)), null);
        assert.deepEqual(matches(model('trained'), model('copy', overrides)), ['trained']);
    }
});
test('non-model assets cannot become model aliases', () => {
    for (const kind of ['dataset', 'report', 'config']) {
        assert.deepEqual(matches(model('trained'), model('copy', { kind })), ['trained']);
    }
});
test('manifest property order and non-content metadata do not change identity', () => {
    assert.deepEqual(matches(model('trained'), model('copy', { manifest: {
        schemaVersion: 1, files: [{ sha256: 'b'.repeat(64), size: 4, path: 'weights.bin' }],
    } })), ['trained', 'copy']);
});
test('multiple identical training outputs preserve every candidate version', () => {
    const assets = [model('first'), model('second'), model('evaluation')];
    for (const equivalents of equivalentModelVersions(assets).values()) {
        assert.deepEqual(equivalents.map(asset => asset.id), ['first', 'second', 'evaluation']);
    }
});
