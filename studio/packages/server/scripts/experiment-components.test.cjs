// Supply an existing Playwright babelBundle.js path. Transforms stay in memory.
const assert = require('node:assert/strict');
const { test } = require('node:test');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const babelPath = process.argv[2] || process.env.MEDFLOW_TEST_BABEL;
const root = path.resolve(__dirname, '../../client/src/pages/DashboardPage/RunPage/ExperimentManager');
const deferred = () => { let resolve; const promise = new Promise(done => { resolve = done; }); return { promise, resolve }; };
const nodes = value => !value || typeof value !== 'object' ? [] : Array.isArray(value) ? value.flatMap(nodes)
    : [value, ...nodes(value.props?.children)];
function harness(file, mocks, props, name = 'default', globals = {}) {
    const slots = []; let cursor = 0, dirty = false, effects = [], output;
    const same = (a, b) => a && b && a.length === b.length && a.every((value, index) => Object.is(value, b[index]));
    const react = {
        useId: () => 'test-id',
        useState: initial => {
            const index = cursor++; slots[index] ??= { value: typeof initial === 'function' ? initial() : initial };
            return [slots[index].value, next => {
                const value = typeof next === 'function' ? next(slots[index].value) : next;
                if (!Object.is(value, slots[index].value)) { slots[index].value = value; dirty = true; }
            }];
        },
        useRef: initial => { const index = cursor++; return slots[index] ??= { current: initial }; },
        useEffect: (fn, deps) => {
            const index = cursor++;
            if (!same(slots[index]?.deps, deps)) effects.push(() => {
                slots[index]?.cleanup?.(); slots[index] = { deps, cleanup: fn() };
            });
        },
    };
    const jsx = (type, props) => ({ type, props });
    const components = Object.fromEntries(['Alert', 'Button', 'Checkbox', 'Collapse', 'ConfigProvider', 'Descriptions', 'Drawer', 'Empty', 'Select', 'Space', 'Spin', 'Switch', 'Table', 'Tabs', 'Tag'].map(value => [value, value]));
    components.Typography = { Text: 'Text', Paragraph: 'Paragraph', Title: 'Title' }; components.Input = { Search: 'InputSearch' };
    const module = { exports: {} }, filename = path.join(root, file);
    const imports = { react, antd: components, 'react/jsx-runtime': { jsx, jsxs: jsx, Fragment: 'Fragment' },
        'react-i18next': { useTranslation: () => ({ t: key => key, i18n: { language: 'en' } }) }, ...mocks };
    const compiled = require(babelPath).babelTransform(fs.readFileSync(filename, 'utf8'), filename, false, [], []).code;
    vm.runInNewContext(compiled, { module, exports: module.exports, AbortController, Date,
        require: specifier => { assert.ok(imports[specifier], 'Unexpected import ' + specifier); return imports[specifier]; }, ...globals }, { filename });
    const render = () => {
        for (let iteration = 0; iteration < 30; iteration++) {
            cursor = 0; dirty = false; effects = [];
            output = module.exports[name](props);
            for (const effect of effects) effect();
            if (!dirty) return output;
        }
        throw new Error('Unsettled component');
    };
    return { props, render, get: () => output, find: predicate => nodes(output).find(predicate), all: predicate => nodes(output).filter(predicate),
        flush: async () => { for (let index = 0; index < 12; index++) await Promise.resolve(); return render(); },
        dispose: () => { for (const slot of slots) slot?.cleanup?.(); } };
}

test('service history distinguishes stopped processes from released reservations and keeps failures visible', { skip: !babelPath }, () => {
    const operation = { id: 'op', operationId: 'stop', attemptId: 'a', serviceInstanceId: 'service-1',
        status: 'succeeded', actorUserId: 'admin', trigger: 'admin', requestedAt: '2026-10-15T01:00:00Z',
        completedAt: '2026-10-15T01:00:05Z', reservationId: 'reservation', reservationStatus: 'running',
        releasedAt: null, releaseReady: true, result: 'stopped', returnCode: 0 };
    const props = { operations: [operation] };
    const h = harness('ExperimentServiceOperations.tsx', {}, props);
    try {
        h.render();
        const descriptions = () => h.find(node => node.type === 'Descriptions').props.items;
        const reservationTag = () => nodes(descriptions().find(item => item.key === 'reservation').children).find(node => node.type === 'Tag');
        assert.equal(descriptions().find(item => item.key === 'process').children, 'experimentRecords.service-release.confirmed');
        assert.equal(reservationTag().props.children, 'experimentRecords.service-reservation.running');
        assert.equal(descriptions().find(item => item.key === 'actor').children, 'admin');
        assert.equal(descriptions().find(item => item.key === 'cause').children, 'experimentRecords.service-cause.unknown');
        operation.cause = 'restart'; h.render();
        assert.ok(h.find(node => node.type === 'Text' && node.props.children === 'experimentRecords.service-restart-stop'));
        assert.equal(descriptions().find(item => item.key === 'cause').children, 'experimentRecords.service-cause.restart');
        operation.cause = 'stop'; h.render();
        assert.ok(h.find(node => node.type === 'Text' && node.props.children === 'experimentRecords.service-stop'));
        operation.reservationStatus = 'released'; h.render();
        assert.equal(reservationTag().props.children, 'experimentRecords.service-reservation.released');
        Object.assign(operation, { status: 'failed', releaseReady: false, result: 'port busy' }); h.render();
        assert.ok(h.find(node => node.type === 'Tag' && node.props.children === 'experimentRecords.service-operation-status.failed'));
        assert.equal(descriptions().find(item => item.key === 'process').children, 'experimentRecords.service-release.unconfirmed');
        Object.assign(operation, { status: 'requested', completedAt: null, releaseReady: null }); h.render();
        assert.equal(descriptions().find(item => item.key === 'completed').children, '—');
        props.operations = []; h.render();
        assert.equal(h.find(node => node.type === 'Empty').props.description, 'experimentRecords.service-history-empty');
    } finally { h.dispose(); }
});

test('experiment drawer owns one notes editor in the overview across refreshes and tab changes', { skip: !babelPath }, async () => {
    const data = { experiment: { id: 'exp', name: 'Workflow', status: 'succeeded', spec: {}, notes: '', tags: [] },
        tasks: [], attempts: [], links: [], assets: [], evaluations: [], metrics: [], lineage: [], linkedEvaluations: [] };
    const detail = { refetch: async () => {}, isFetching: false, dataUpdatedAt: 100 };
    const list = { data: { items: [] }, ...detail };
    const names = ['ExperimentFilters', 'ExperimentComparison', 'ExperimentLogs', 'ExperimentEvaluation', 'ExperimentObservedOutputs',
        'ExperimentMetadata', 'ExperimentServiceOperations'];
    const mocks = Object.fromEntries(names.map(name => ['./' + name, { __esModule: true, default: name }]));
    Object.assign(mocks, {
        './index.css': {}, 'antd/locale/en_US': {}, 'antd/locale/zh_CN': {},
        '@ant-design/icons': { CopyOutlined: 'Copy', ExportOutlined: 'Export', FilterOutlined: 'Filter', ImportOutlined: 'Import' },
        '@/api/trpc': { trpc: { launchExperimentDraft: { useMutation: () => ({}) } } },
        '@/context/SidebarContext': { useStudioSidebar: () => ({ selectedExperimentId: 'exp', setSelectedExperimentId: () => {} }) },
        './useExperimentQueries': { useExperimentQueries: () => ({ data, list, detail, comparison: detail, visible: true }) },
        './SourceButton': { default: 'SourceButton' }, './evaluationOutputs': { evaluationOutputs: () => [] },
        './evaluationVisibility': { hasEvaluationTab: () => false },
        './associationLabels': { associationLabel: () => '', associationRole: () => '' },
        './displayLabels': { experimentName: () => 'Workflow', experimentLabel: () => '', executionLabel: () => '' },
    });
    const h = harness('index.tsx', mocks, {});
    const tabs = () => h.find(node => node.type === 'Tabs');
    const editors = () => [
        ...h.all(node => node.type === 'ExperimentMetadata'),
        ...tabs().props.items.flatMap(item => nodes(item.children).filter(node => node.type === 'ExperimentMetadata')),
    ];
    try {
        h.render();
        assert.equal(editors().length, 1);
        assert.ok(nodes(tabs().props.items.find(item => item.key === 'overview').children).includes(editors()[0]));
        await editors()[0].props.onSaved(); h.render();
        assert.equal(editors().length, 1);
        tabs().props.onChange('executions'); h.render();
        tabs().props.onChange('overview'); h.render();
        assert.equal(editors().length, 1);
    } finally { h.dispose(); }
});

test('queries load only visible sections, preserve cached history and keep late evaluations polling', { skip: !babelPath }, async () => {
    const calls = new Map(), options = new Map(), listeners = new Set(), document = { visibilityState: 'visible',
        addEventListener: (_event, callback) => listeners.add(callback), removeEventListener: (_event, callback) => listeners.delete(callback) };
    const base = { experiment: { id: 'exp', spec: {} }, tasks: [], attempts: [{ id: 'attempt', lastSequence: 1, status: 'succeeded', runtime: {} }],
        links: [], assets: [], evaluations: [], metrics: [], lineage: [], linkedEvaluations: [] };
    const payloads = { listExperiments: { items: [{ status: 'succeeded' }] }, getExperimentSummary: base,
        getExperimentEvaluations: { evaluations: [], metrics: [], assets: [], attempts: [], linkedEvaluations: [] } };
    const trpc = Object.fromEntries(['listExperiments', 'getExperimentSummary', 'getExperimentConfiguration', 'getExperimentAssets', 'getExperimentEvaluations', 'compareExperiments']
        .map(name => [name, { useQuery: (input, config) => {
            options.set(name, { input, config });
            return { data: payloads[name], isFetching: false, dataUpdatedAt: 100, refetch: async () => calls.set(name, (calls.get(name) || 0) + 1) };
        } }]));
    const props = { selectedId: 'exp', tab: 'overview', comparing: false, selectedKeys: [], filters: {}, offset: 0 };
    const h = harness('useExperimentQueries.ts', { '@/api/trpc': { trpc }, './queryPolicy': require(path.join(root, 'queryPolicy.ts')) }, props, 'useExperimentQueries', { document });
    try {
        h.render();
        assert.equal(options.get('getExperimentConfiguration').config.enabled, false);
        assert.equal(options.get('getExperimentAssets').config.enabled, false);
        assert.equal(options.get('getExperimentEvaluations').config.enabled, true);
        assert.equal(options.get('getExperimentSummary').config.refetchInterval({ state: { data: base } }), 60000);
        assert.equal(options.get('getExperimentEvaluations').config.refetchInterval({ state: { data: payloads.getExperimentEvaluations } }), 60000);
        props.tab = 'configuration'; h.render();
        assert.equal(options.get('getExperimentConfiguration').config.enabled, true);
        assert.equal(options.get('getExperimentEvaluations').config.enabled, false);
        assert.equal(options.get('getExperimentConfiguration').config.refetchInterval, undefined);
        assert.equal(options.get('getExperimentConfiguration').config.gcTime, 600000);
        const oldRevision = options.get('getExperimentConfiguration').input.revision;
        base.attempts[0].lastSequence++; h.render();
        assert.notEqual(options.get('getExperimentConfiguration').input.revision, oldRevision);
        props.tab = 'assets'; payloads.getExperimentAssets = { assets: [], links: [], lineage: [], observations: [] }; h.render();
        assert.equal(calls.get('getExperimentAssets'), 1);
        document.visibilityState = 'hidden'; for (const callback of listeners) callback(); h.render();
        for (const { config } of options.values()) assert.equal(config.enabled, false);
        document.visibilityState = 'visible'; for (const callback of listeners) callback(); h.render();
        assert.equal(options.get('getExperimentAssets').config.enabled, true);
        props.selectedId = undefined; h.render();
        assert.equal(options.get('getExperimentSummary').config.enabled, false);
        assert.equal(options.get('getExperimentAssets').config.enabled, false);
    } finally { h.dispose(); }
});

test('comparison changes baseline and result without crossing protocols or forgetting valid selection', { skip: !babelPath }, () => {
    const dataset = { id: 'data', kind: 'dataset', logicalName: 'Dataset', fingerprint: 'data-hash' };
    const model = id => ({ id, fingerprint: id, kind: 'model', logicalName: id, integrity: 'verified' });
    const attempt = id => ({ id, taskId: id + '-task', attemptNumber: 1, createdAt: '2026-10-15', status: 'succeeded',
        configuration: { actual: { lr: .1 } }, environment: {}, startedAt: null, endedAt: null });
    const details = ['a', 'b'].map(id => ({ experiment: { id, name: id, spec: {} },
        tasks: [{ id: id + '-task', kind: 'train', spec: { resolved: { lr: id === 'a' ? .1 : .2 } } }],
        attempts: [attempt(id)], links: [], assets: [model(id), dataset] }));
    const entry = (id, resultId, value, time) => ({ trainingAttemptId: id, trainingExperimentId: id,
        sourceExperimentId: 'eval-' + id, sourceExperimentName: 'Eval ' + id, evaluationAttempt: attempt('eval-' + id),
        result: { id: resultId, modelVersionId: id, createdAt: time, comparisonKey: 'protocol', protocol: {} }, model: model(id), datasets: [dataset],
        metrics: [{ datasetVersionId: 'data', subset: 'all', name: 'accuracy', unit: 'ratio', direction: 'higher', value, sampleCount: 10 }],
        eligible: true, ineligibleReason: null });
    const data = { details, groups: {}, trainingRuns: details.map(detail => ({ experimentId: detail.experiment.id,
        attempt: detail.attempts[0], inputs: [], outputs: [] })), linkedEvaluations: [entry('a', 'a-old', .72, '2026-10-13'),
        entry('a', 'a-new', .74, '2026-10-14'), entry('b', 'b-new', .75, '2026-10-14')] };
    const h = harness('ExperimentComparison.tsx', {
        './comparisonGroups': require(path.join(root, 'comparisonGroups.ts')), './comparisonValues': require(path.join(root, 'comparisonValues.ts')),
        './displayLabels': require(path.join(root, 'displayLabels.ts')),
    }, { data });
    const table = () => h.all(node => node.type === 'Table').find(node => node.props.dataSource[0]?.values && node.props.dataSource[0]?.unit);
    const baseline = () => h.find(node => node.type === 'Select' && node.props['aria-label'] === 'experimentRecords.baseline-result');
    try {
        h.render();
        assert.equal(baseline().props.value, 'a:a-new');
        const row = table().props.dataSource[0];
        assert.ok(Math.abs(row.values['b:b-new'].value - row.values['a:a-new'].value - .01) < 1e-10);
        baseline().props.onChange('b:b-new'); h.render();
        assert.equal(baseline().props.value, 'b:b-new');
        const resultSelector = h.find(node => node.type === 'Select' && node.props['aria-label'] === 'experimentRecords.select-evaluation-result');
        resultSelector.props.onChange('a:a-old'); h.render();
        assert.equal(baseline().props.value, 'b:b-new');
        assert.equal(table().props.dataSource[0].values['a:a-old'].value, .72);
        const valueCell = table().props.columns.at(-1).render(null, table().props.dataSource[0]);
        assert.ok(nodes(valueCell).some(node => node.type === 'Tag' && node.props.color === undefined) === false,
            'The baseline column must have no delta badge');
        data.linkedEvaluations[2].eligible = false; data.linkedEvaluations[2].ineligibleReason = 'inputs-not-verified'; h.render();
        assert.equal(baseline().props.value, 'a:a-old');
        assert.ok(h.all(node => node.type === 'Table').some(node => node.props.dataSource.some(row => row.reason === 'inputs-not-verified')));
    } finally { h.dispose(); }
});

test('log search freezes snapshot, pages with fixed terms, and cancels stale search/context requests', { skip: !babelPath }, async () => {
    const requests = [], navigations = [], pending = deferred();
    let searchHandler = async () => ({ matches: [{ position: { startOffset: 100, characterOffset: 4 }, length: 5, before: '', text: 'ERROR', after: '' }],
        nextCursor: { startOffset: 100, characterOffset: 9 }, snapshotEndOffset: 200 });
    const trpcClient = { searchExperimentLogs: { query: (input, options) => { requests.push({ input, options }); return searchHandler(); } },
        getExperimentLogContext: { query: (input, options) => { requests.push({ input, options }); return pending.promise; } } };
    const props = { experimentId: 'exp', attemptId: 'a', sourceId: 'source', active: true, onNavigate: value => navigations.push(value) };
    const h = harness('ExperimentLogSearch.tsx', { '@/api/trpc': { trpcClient } }, props);
    const button = label => h.find(node => node.type === 'Button' && node.props.children === label);
    try {
        h.render();
        const input = h.find(node => node.type === 'InputSearch'); input.props.onChange({ target: { value: 'ERROR' } }); h.render();
        h.find(node => node.type === 'InputSearch').props.onSearch(); await h.flush();
        assert.equal(requests[0].input.snapshotEndOffset, undefined);
        h.find(node => node.type === 'InputSearch').props.onChange({ target: { value: 'other' } }); h.render();
        button('experimentRecords.search-more').props.onClick(); await h.flush();
        assert.equal(requests[1].input.snapshotEndOffset, 200); assert.equal(requests[1].input.keyword, 'ERROR');
        button('experimentRecords.next-match').props.onClick(); h.render();
        const contextRequest = requests.at(-1);
        props.sourceId = 'new-source'; h.render();
        assert.equal(contextRequest.options.signal.aborted, true);
        pending.resolve({ chunks: [] }); await h.flush(); assert.equal(navigations.length, 0);
        const delayed = deferred(); searchHandler = () => delayed.promise;
        h.find(node => node.type === 'InputSearch').props.onSearch(); h.render();
        const oldSearch = requests.at(-1); props.active = false; h.render();
        assert.equal(oldSearch.options.signal.aborted, true);
        delayed.resolve({ matches: [], nextCursor: null, snapshotEndOffset: 300 }); await h.flush();
        assert.equal(button('experimentRecords.search-more'), undefined);
    } finally { h.dispose(); }
});

test('filter execution distinguishes empty quality results from scoring API failures', { skip: !babelPath }, () => {
    const data = { experiment: { id: 'exp', name: 'Workflow', status: 'succeeded', spec: {}, notes: '', tags: [] },
        tasks: [], attempts: [], links: [], assets: [], evaluations: [], metrics: [], lineage: [], linkedEvaluations: [] };
    const detail = { refetch: async () => {}, isFetching: false, dataUpdatedAt: 100 };
    const list = { data: { items: [] }, ...detail };
    const names = ['ExperimentFilters', 'ExperimentComparison', 'ExperimentLogs', 'ExperimentEvaluation', 'ExperimentObservedOutputs',
        'ExperimentMetadata', 'ExperimentServiceOperations'];
    const mocks = Object.fromEntries(names.map(name => ['./' + name, { __esModule: true, default: name }]));
    Object.assign(mocks, {
        './index.css': {}, 'antd/locale/en_US': {}, 'antd/locale/zh_CN': {},
        '@ant-design/icons': { CopyOutlined: 'Copy', ExportOutlined: 'Export', FilterOutlined: 'Filter', ImportOutlined: 'Import' },
        '@/api/trpc': { trpc: { launchExperimentDraft: { useMutation: () => ({}) } } },
        '@/context/SidebarContext': { useStudioSidebar: () => ({ selectedExperimentId: 'exp', setSelectedExperimentId: () => {} }) },
        './useExperimentQueries': { useExperimentQueries: () => ({ data, list, detail, comparison: detail, visible: true }) },
        './SourceButton': { default: 'SourceButton' }, './evaluationOutputs': { evaluationOutputs: () => [] },
        './evaluationVisibility': { hasEvaluationTab: () => false },
        './associationLabels': { associationLabel: () => '', associationRole: () => '' },
        './displayLabels': { experimentName: () => 'Workflow', experimentLabel: () => '', executionLabel: () => '' },
    });
    for (const [outcome, status, label, alertType, messageKey] of [
        ['no_qualifying_data', 'succeeded', 'filter-empty-status', 'info', 'filter-empty-description'],
        ['scoring_failed', 'failed', 'filter-scoring-failed-status', 'error', 'filter-scoring-failed-description'],
    ]) {
        data.attempts = [{ id: 'filter', status, configuration: {}, runtime: { archiveStatus: 'complete',
            dataProcessing: { outcome, total_items: 150, api_failed_items: outcome === 'scoring_failed' ? 150 : 0 } },
            error: status === 'failed' ? 'Scoring failed' : null }];
        const h = harness('index.tsx', mocks, {});
        try {
            h.render();
            const execution = h.find(node => node.type === 'Tabs').props.items.find(item => item.key === 'executions');
            const children = nodes(execution.children);
            const statusTag = children.find(node => node.props?.filterOutcome === outcome);
            assert.equal(statusTag.type(statusTag.props).props.children, 'experimentRecords.' + label);
            const alert = children.find(node => node.type === 'Alert');
            assert.equal(alert.props.type, alertType);
            assert.equal(alert.props.message, 'experimentRecords.' + messageKey);
            assert.equal(children.some(node => node.type === 'Alert' && node.props.type === 'warning'), false);
        } finally { h.dispose(); }
    }
});
