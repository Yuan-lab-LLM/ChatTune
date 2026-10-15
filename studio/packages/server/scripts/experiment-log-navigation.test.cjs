// Supply an EXISTING Playwright babelBundle.js path as argv[2] or MEDFLOW_TEST_BABEL.
// Executes the real component with deterministic hooks, queries, timers and scroll geometry.
// No packages are installed, and transpilation stays in memory. Browser QA remains separate.
const assert = require('node:assert/strict');
const { test } = require('node:test');
const { readFileSync } = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const babelPath = process.argv[2] || process.env.MEDFLOW_TEST_BABEL;
const filename = path.resolve(__dirname, '../../client/src/pages/DashboardPage/RunPage/ExperimentManager/ExperimentLogs.tsx');
const compiled = babelPath ? require(babelPath).babelTransform(readFileSync(filename, 'utf8'), filename, false, [], []).code : null;
const chunk = index => ({ startOffset: index * 100, endOffset: (index + 1) * 100, content: `line ${index}\n` });
const page = start => Array.from({ length: 8 }, (_, index) => chunk(start + index));
const deferred = () => { let resolve; const promise = new Promise(done => { resolve = done; }); return { promise, resolve }; };

function harness() {
    const slots = [], timers = new Map(), calls = [];
    let cursor = 0, dirty = false, effects = [], layouts = [], tree, timerId = 0;
    const same = (a, b) => a && b && a.length === b.length && a.every((value, index) => Object.is(value, b[index]));
    const state = initial => {
        const index = cursor++;
        slots[index] ??= { value: typeof initial === 'function' ? initial() : initial };
        return [slots[index].value, value => {
            const next = typeof value === 'function' ? value(slots[index].value) : value;
            if (!Object.is(next, slots[index].value)) { slots[index].value = next; dirty = true; }
        }];
    };
    const effect = (fn, deps, target) => {
        const index = cursor++;
        if (!same(slots[index]?.deps, deps)) target.push(() => {
            slots[index]?.cleanup?.();
            slots[index] = { deps, cleanup: fn() };
        });
    };
    const react = {
        useState: state,
        useRef: initial => { const index = cursor++; return slots[index] ??= { current: initial }; },
        useCallback: (fn, deps) => {
            const index = cursor++;
            if (!same(slots[index]?.deps, deps)) slots[index] = { deps, fn };
            return slots[index].fn;
        },
        useEffect: (fn, deps) => effect(fn, deps, effects),
        useLayoutEffect: (fn, deps) => effect(fn, deps, layouts),
    };
    const fixture = {
        active: true, sourceOptions: null,
        attempt: { id: 'attempt', attemptNumber: 1, status: 'running', runtime: { logCaptureVersion: 1 } },
        source: { id: 'source', kind: 'training', label: 'training', status: 'streaming', nextOffset: 100800 },
        latest: { chunks: page(1000), latestOffset: 100800, status: 'streaming' },
        fetch: async () => ({ chunks: [], latestOffset: fixture.source.nextOffset, status: fixture.source.status }),
    };
    const utils = { getExperimentLogChunks: { fetch: async input => { calls.push(input); return fixture.fetch(input); } } };
    const trpc = {
        useUtils: () => utils,
        listExperimentLogSources: { useQuery: (input, options) => { fixture.sourceInput = input; fixture.sourceOptions = options; return { data: [fixture.source] }; } },
        getExperimentLogChunks: { useQuery: input => ({ data: input.sourceId ? fixture.latest : undefined }) },
    };
    const jsx = (type, props) => ({ type, props });
    const module = { exports: {} };
    const components = Object.fromEntries(['Alert', 'Button', 'Empty', 'Select', 'Space', 'Spin', 'Tag', 'Typography'].map(name => [name, name]));
    components.Typography = { Text: 'Text' };
    const mocks = { react, antd: components, 'react-i18next': { useTranslation: () => ({ t: key => key }) },
        '@/api/trpc': { trpc }, './logWindow': require('../../client/src/pages/DashboardPage/RunPage/ExperimentManager/logWindow.ts'),
        './ExperimentLogSearch': { default: 'ExperimentLogSearch' },
        './queryPolicy': require('../../client/src/pages/DashboardPage/RunPage/ExperimentManager/queryPolicy.ts'),
        './displayLabels': require('../../client/src/pages/DashboardPage/RunPage/ExperimentManager/displayLabels.ts'),
        'react/jsx-runtime': { jsx, jsxs: jsx, Fragment: 'Fragment' } };
    vm.runInNewContext(compiled, { module, exports: module.exports,
        require: name => { assert.ok(mocks[name], `Unexpected import ${name}`); return mocks[name]; },
        window: { setInterval: fn => { timers.set(++timerId, fn); return timerId; }, clearInterval: id => timers.delete(id) },
    }, { filename });
    const log = { children: [], clientHeight: 600, scrollHeight: 1200, top: 0,
        getBoundingClientRect: () => ({ top: 0, bottom: 600 }) };
    Object.defineProperty(log, 'scrollTop', { get: () => log.top,
        set: value => { log.top = Math.max(0, Math.min(value, log.scrollHeight - log.clientHeight)); } });
    const nodes = node => !node || typeof node !== 'object' ? [] : Array.isArray(node)
        ? node.flatMap(nodes) : [node, ...nodes(node.props?.children)];
    const render = () => {
        for (let count = 0; count < 25; count++) {
            cursor = 0; dirty = false; effects = []; layouts = [];
            tree = module.exports.default({ experimentId: 'experiment', attempts: fixture.attempts || [fixture.attempt], tasks: fixture.tasks, active: fixture.active });
            const pre = nodes(tree).find(node => node.type === 'pre');
            if (pre) {
                pre.props.ref.current = log;
                const spans = nodes(pre.props.children).filter(node => node.type === 'span');
                log.scrollHeight = Math.max(1200, spans.length * 100);
                log.children = spans.map((node, index) => ({ dataset: { offset: String(node.props['data-offset']) },
                    getBoundingClientRect: () => ({ top: index * 100 - log.top, bottom: (index + 1) * 100 - log.top }) }));
            }
            for (const fn of [...layouts, ...effects]) fn();
            if (!dirty) return tree;
        }
        throw new Error('Component did not settle');
    };
    const scroll = top => {
        log.scrollTop = top;
        nodes(tree).find(node => node.type === 'pre').props.onScroll({ currentTarget: log });
        render();
    };
    const click = label => {
        const button = nodes(tree).find(node => node.type === 'Button' && node.props.children === label);
        assert.ok(button, `Missing button ${label}`);
        return button.props.onClick();
    };
    const flush = async () => { for (let index = 0; index < 8; index++) await Promise.resolve(); render(); };
    render();
    return { fixture, calls, render, scroll, click, flush,
        selectors: () => nodes(tree).filter(node => node.type === 'Select'),
        offsets: () => log.children.map(node => Number(node.dataset.offset)),
        tick: async () => { for (const fn of [...timers.values()]) await fn(); render(); },
        timers: () => timers.size,
        dispose: () => { for (const slot of slots) slot?.cleanup?.(); },
    };
}

test('workflow execution selector distinguishes task-local #1 entries and switches log queries back to training', { skip: !babelPath }, () => {
    const h = harness();
    try {
        h.fixture.tasks = [{ id: 'train-task', kind: 'train' }, { id: 'eval-task', kind: 'evaluate' }];
        h.fixture.attempts = [
            { ...h.fixture.attempt, id: 'training-attempt', taskId: 'train-task', status: 'succeeded' },
            { ...h.fixture.attempt, taskId: 'eval-task' },
        ];
        h.render();
        const selector = h.selectors()[0];
        assert.match(selector.props.options[0].label, /kind.train.*#1/);
        assert.match(selector.props.options[1].label, /kind.evaluate.*#1/);
        selector.props.onChange('training-attempt');
        h.render();
        assert.equal(h.fixture.sourceInput.attemptId, 'training-attempt');
        h.selectors()[0].props.onChange('attempt');
        h.render();
        assert.equal(h.fixture.sourceInput.attemptId, 'attempt');
    } finally { h.dispose(); }
});

test('reading history pauses tail polling and forward paging reloads evicted logs', { skip: !babelPath }, async () => {
    const h = harness();
    try {
        assert.equal(h.timers(), 1);
        h.fixture.fetch = async input => ({ chunks: input.beforeOffset !== undefined ? page(input.beforeOffset / 100 - 8)
            : page(input.afterOffset / 100), latestOffset: 100800, status: 'streaming' });
        h.scroll(0);
        await h.flush();
        assert.equal(h.timers(), 0);
        assert.equal(h.offsets()[0], 99200);
        for (let index = 0; index < 9; index++) { h.scroll(0); await h.flush(); }
        assert.equal(h.offsets().length, 64);
        const before = h.offsets().at(-1);
        await h.click('experimentRecords.load-later-logs');
        h.render();
        assert.equal(h.offsets().at(-1), before + 800);
        assert.equal(h.offsets().length, 64);
    } finally { h.dispose(); }
});

test('jump-to-latest rejects a delayed historical page and restarts following at the new cursor', { skip: !babelPath }, async () => {
    const h = harness();
    try {
        const delayed = deferred();
        h.fixture.source.nextOffset = 200800;
        h.fixture.fetch = input => input.beforeOffset !== undefined ? delayed.promise
            : Promise.resolve({ chunks: page(2000), latestOffset: 200800, status: 'streaming' });
        h.scroll(0);
        await h.click('experimentRecords.jump-to-latest-logs');
        h.render();
        assert.equal(h.offsets()[0], 200000);
        delayed.resolve({ chunks: page(992), latestOffset: 100800, status: 'streaming' });
        await h.flush();
        assert.equal(h.offsets()[0], 200000, 'Old history response must not contaminate the latest window');
        h.fixture.fetch = async input => ({ chunks: [], latestOffset: 200800, status: 'streaming' });
        await h.tick();
        assert.equal(h.calls.at(-1).afterOffset, 200800);
    } finally { h.dispose(); }
});

test('new archive offsets do not stop following or grow the rendered window beyond its limit', { skip: !babelPath }, async () => {
    const h = harness();
    try {
        for (let index = 0; index < 20; index++) {
            h.fixture.source.nextOffset += 800;
            h.fixture.fetch = async input => ({ chunks: page(input.afterOffset / 100),
                latestOffset: h.fixture.source.nextOffset, status: 'streaming' });
            await h.tick();
            h.scroll(1000000);
            assert.ok(h.offsets().length <= 64);
            assert.equal(h.timers(), 1);
        }
        assert.equal(h.offsets().at(-1) + 100, h.fixture.source.nextOffset);
    } finally { h.dispose(); }
});

test('returning to the live tail supersedes an outstanding backwards page', { skip: !babelPath }, async () => {
    const h = harness();
    try {
        const delayed = deferred();
        h.fixture.fetch = () => delayed.promise;
        h.scroll(0);
        assert.equal(h.timers(), 0);
        h.scroll(1000000);
        assert.equal(h.timers(), 1);
        delayed.resolve({ chunks: page(992), latestOffset: 100800, status: 'streaming' });
        await h.flush();
        assert.equal(h.offsets()[0], 100000);
        assert.equal(h.offsets().at(-1), 100700);
    } finally { h.dispose(); }
});

test('hidden log pages stop both tail polling and source queries and resume when visible', { skip: !babelPath }, () => {
    const h = harness();
    try {
        assert.equal(h.timers(), 1);
        h.fixture.active = false; h.render();
        assert.equal(h.timers(), 0); assert.equal(h.fixture.sourceOptions.enabled, false);
        assert.equal(h.fixture.sourceOptions.refetchInterval({ state: { data: [h.fixture.source] } }), false);
        h.fixture.active = true; h.render();
        assert.equal(h.timers(), 1); assert.equal(h.fixture.sourceOptions.enabled, true);
    } finally { h.dispose(); }
});

test('complete logs stop polling even when the exited execution is awaiting its report', { skip: !babelPath }, () => {
    const h = harness();
    try {
        h.fixture.attempt.status = 'unknown'; h.fixture.attempt.runtime.exitCode = 0;
        h.fixture.source.status = 'complete'; h.render();
        assert.equal(h.fixture.sourceOptions.refetchInterval({ state: { data: [h.fixture.source] } }), false);
        assert.equal(h.timers(), 0);
    } finally { h.dispose(); }
});
