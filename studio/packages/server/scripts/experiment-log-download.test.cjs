const assert = require('node:assert/strict');
const { test } = require('node:test');
const { EventEmitter } = require('node:events');
const { readFileSync } = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const babelPath = process.argv[2] || process.env.MEDFLOW_TEST_BABEL;
class TRPCError extends Error { constructor({ code }) { super(code); this.code = code; } }
function harness() {
    const fixture = { user: { id: 'owner' }, snapshot: null, source: { id: 'business', nextOffset: 200 }, forbidden: false,
        chunks: [{ endOffset: 100, content: '中文🙂 [REDACTED]\n' }, { endOffset: 200, content: 'ERROR\n' }], produced: 0 };
    const imports = {
        '@trpc/server': { TRPCError },
        '../../../shared/src/experiment': { experimentLogIdentitySchema: { parse: input => input } },
        '../dao/Auth': { AuthDao: { getUserByToken: async () => fixture.user } },
        '../trpc/router': { getRequestAuthToken: () => 'cookie-token' },
        './experimentService': { ExperimentService: {
            businessLogSource: async () => { if (fixture.forbidden) throw new TRPCError({ code: 'FORBIDDEN' }); return fixture.source; },
            archivedLogChunks: async function* (_id, snapshot) {
                fixture.snapshot = snapshot;
                for (const chunk of fixture.chunks) if (chunk.endOffset <= snapshot) { fixture.produced++; yield chunk; }
            },
        } },
    };
    const module = { exports: {} }, filename = path.resolve(__dirname, '../src/services/experimentLogDownload.ts');
    vm.runInNewContext(require(babelPath).babelTransform(readFileSync(filename, 'utf8'), filename, false, [], []).code,
        { module, exports: module.exports, require: name => { assert.ok(imports[name], name); return imports[name]; } });
    const res = new EventEmitter(); Object.assign(res, { statusCode: 200, headers: {}, chunks: [], destroyed: false, headersSent: false,
        setHeader(name, value) { this.headers[name] = value; }, status(code) { this.statusCode = code; return this; },
        write(content) { this.headersSent = true; this.chunks.push(content); return true; },
        end() { this.ended = true; }, json(value) { this.body = value; this.ended = true; },
        destroy() { this.destroyed = true; this.emit('close'); } });
    return { fixture, res, run: () => module.exports.downloadExperimentLog({ params: { experimentId: 'exp', attemptId: 'attempt', sourceId: 'business' } }, res) };
}

test('download authenticates before accessing log data and rejects forbidden sources', { skip: !babelPath }, async () => {
    for (const [user, forbidden, code] of [[null, false, 401], [{ mustChangePassword: true }, false, 403], [{ id: 'other' }, true, 403]]) {
        const h = harness(); h.fixture.user = user; h.fixture.forbidden = forbidden;
        await h.run(); assert.equal(h.res.statusCode, code); assert.equal(h.fixture.produced, 0);
    }
});

test('download streams a fixed redacted snapshot and honors backpressure', { skip: !babelPath }, async () => {
    const h = harness(); let blocked = true;
    const write = h.res.write.bind(h.res);
    h.res.write = content => { write(content); if (blocked) { blocked = false; return false; } return true; };
    const running = h.run();
    for (let index = 0; index < 12; index++) await Promise.resolve();
    assert.equal(h.fixture.produced, 1, 'Do not read another block while the response is backpressured');
    h.fixture.source.nextOffset = 300; h.fixture.chunks.push({ endOffset: 300, content: 'late log\n' });
    h.res.emit('drain'); await running;
    assert.equal(h.fixture.snapshot, 200);
    assert.equal(h.res.chunks.join(''), '中文🙂 [REDACTED]\nERROR\n');
    assert.equal(h.res.headers['Cache-Control'], 'private, no-store');
    assert.equal(h.res.headers['Content-Type'], 'text/plain; charset=utf-8');
    assert.match(h.res.headers['Content-Disposition'], /attachment; filename="attempt-business.txt"/);
    assert.equal(h.res.listenerCount('close'), 0); assert.equal(h.res.ended, true);
});

test('disconnect while backpressured stops the stream and releases listeners', { skip: !babelPath }, async () => {
    const h = harness(); h.res.write = () => false;
    const running = h.run();
    for (let index = 0; index < 12; index++) await Promise.resolve();
    h.res.destroy(); await running;
    assert.equal(h.fixture.produced, 1, 'Do not read another block after close');
    assert.equal(h.res.listenerCount('drain'), 0); assert.equal(h.res.listenerCount('close'), 0);
});
