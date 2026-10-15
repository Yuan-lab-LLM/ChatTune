import { useState } from 'react';
import { Alert, Collapse, Descriptions, Empty, Select, Space, Switch, Table, Tag, Typography } from 'antd';
import type { inferRouterOutputs } from '@trpc/server';
import { useTranslation } from 'react-i18next';
import { standaloneComparisonGroups } from './comparisonGroups';
import { experimentLabel, parameterLabel } from './displayLabels';
import { canonicalValue, latestResult, metricDelta, parameterFields, type ParameterSource } from './comparisonValues';
import type { AppRouter } from '../../../../../../server/src/trpc/router';

type ComparisonData = inferRouterOutputs<AppRouter>['compareExperiments'];
type TrainingRun = ComparisonData['trainingRuns'][number];
type LinkedEvaluation = ComparisonData['linkedEvaluations'][number];
type Asset = ComparisonData['details'][number]['assets'][number];
type Candidate = { key: string; participant: string; experimentName: string; experimentId: string;
    attempt: { id: string; attemptNumber: number; status: string } | undefined;
    result: LinkedEvaluation['result']; metrics: LinkedEvaluation['metrics']; model: Asset | undefined; datasets: Asset[];
    source: string; eligible: boolean; reason: string | null };
type MetricRow = { key: string; dataset: string; subset: string; name: string; unit: string; direction: string;
    values: Record<string, { value: number; sampleCount: number | null }> };
const shortId = (value: string) => value.length > 18 ? `${value.slice(0, 18)}…` : value;
const display = (value: unknown) => value === undefined ? '—' : typeof value === 'string' ? value : JSON.stringify(value);

export default function ExperimentComparison({ data }: { data: ComparisonData }) {
    const { t, i18n } = useTranslation();
    const [attemptSelection, setAttemptSelection] = useState<Record<string, string>>({});
    const [resultSelection, setResultSelection] = useState<Record<string, string>>({});
    const [baselines, setBaselines] = useState<Record<string, string>>({});
    const [onlyDifferences, setOnlyDifferences] = useState(true);
    const [parameterSource, setParameterSource] = useState<ParameterSource>('resolved');
    const [percentage, setPercentage] = useState(true);
    const stamp = (value: string | null) => value ? new Date(value).toLocaleString(i18n.language) : '—';
    const number = (value: number) => value.toLocaleString(i18n.language, { maximumFractionDigits: 4 });
    const selected = data.details.map(detail => {
        const training = data.trainingRuns.filter(run => run.experimentId === detail.experiment.id);
        const evaluationTasks = new Set(detail.tasks.filter(task => ['evaluate', 'benchmark'].includes(task.kind)).map(task => task.id));
        const runs = (training.length ? training : detail.attempts.filter(attempt => evaluationTasks.has(attempt.taskId)).map(attempt => ({
            experimentId: detail.experiment.id, attempt,
            inputs: detail.links.filter(link => link.attemptId === attempt.id && link.direction === 'input')
                .flatMap(link => { const asset = detail.assets.find(asset => asset.id === link.assetVersionId); return asset ? [{ link, asset }] : []; }),
            outputs: detail.links.filter(link => link.attemptId === attempt.id && link.direction === 'output')
                .flatMap(link => { const asset = detail.assets.find(asset => asset.id === link.assetVersionId); return asset ? [{ link, asset }] : []; }),
        }))).sort((a, b) => b.attempt.attemptNumber - a.attempt.attemptNumber || b.attempt.createdAt.localeCompare(a.attempt.createdAt));
        const eligibleIds = new Set(Object.values(data.groups).flat().filter(item => !item.ineligibleReason).map(item => item.result.attemptId));
        return { detail, runs, training: training.length > 0, run: runs.find(run => run.attempt.id === attemptSelection[detail.experiment.id])
            || (!training.length ? runs.find(run => eligibleIds.has(run.attempt.id)) : undefined) || runs[0] };
    });
    const selectedRuns = selected.filter((item): item is typeof item & { run: TrainingRun } => Boolean(item.run));
    const selectedIds = new Set(selectedRuns.map(item => item.run.attempt.id));
    const linked = data.linkedEvaluations.filter(item => selectedIds.has(item.trainingAttemptId));
    const standalone = standaloneComparisonGroups(data.groups, linked.map(item => item.result.id));
    const candidates: Candidate[] = linked.map(item => {
        const selection = selectedRuns.find(run => run.run.attempt.id === item.trainingAttemptId)!;
        return { key: `${item.trainingAttemptId}:${item.result.id}`,
            participant: `${item.trainingAttemptId}:${item.model.fingerprint || item.model.id}`,
            experimentName: selection.detail.experiment.name, experimentId: item.trainingExperimentId,
            attempt: selection.run.attempt, result: item.result, metrics: item.metrics, model: item.model, datasets: item.datasets,
            source: `${item.sourceExperimentName} · #${item.evaluationAttempt.attemptNumber}`,
            eligible: item.eligible, reason: item.ineligibleReason };
    });
    for (const group of Object.values(standalone)) for (const entry of group) {
        const detail = data.details.find(item => item.experiment.id === entry.experimentId)!;
        const selection = selected.find(item => item.detail.experiment.id === entry.experimentId);
        if (selection && !selection.training && selection.run && entry.result.attemptId !== selection.run.attempt.id) continue;
        const model = detail.assets.find(asset => asset.id === entry.result.modelVersionId);
        candidates.push({ key: `${entry.experimentId}:${entry.result.id}`,
            participant: `${entry.experimentId}:${entry.result.attemptId}:${model?.fingerprint || entry.result.modelVersionId}`,
            experimentName: detail.experiment.name, experimentId: entry.experimentId,
            attempt: detail.attempts.find(attempt => attempt.id === entry.result.attemptId), result: entry.result,
            metrics: entry.metrics, model, datasets: detail.assets.filter(asset => asset.kind === 'dataset'),
            source: detail.experiment.name, eligible: !entry.ineligibleReason, reason: entry.ineligibleReason });
    }
    const groups = new Map<string, Candidate[]>();
    for (const entry of candidates.filter(item => item.eligible)) {
        const items = groups.get(entry.result.comparisonKey) || [];
        items.push(entry); groups.set(entry.result.comparisonKey, items);
    }
    const fields = new Map(selectedRuns.map(({ detail, run }) => [run.attempt.id, parameterFields(parameterSource,
        detail.tasks.find(task => task.id === run.attempt.taskId)?.spec || detail.experiment.spec,
        run.attempt.configuration, run.attempt.environment)]));
    const paths = [...new Set([...fields.values()].flatMap(item => Object.keys(item.fields)))].sort();
    const parameters = paths.map(path => {
        const values = Object.fromEntries(selectedRuns.map(({ run }) => [run.attempt.id, fields.get(run.attempt.id)?.fields[path]]));
        return { path, values, different: new Set(selectedRuns.map(({ run }) => {
            const values = fields.get(run.attempt.id)!.fields;
            return Object.prototype.hasOwnProperty.call(values, path) ? `value:${canonicalValue(values[path])}` : 'missing';
        })).size > 1 };
    });
    const candidateLabel = (entry: Candidate) => `${entry.experimentName} · #${entry.attempt?.attemptNumber ?? '—'} · ${entry.model?.logicalName || entry.result.modelVersionId} · ${shortId(entry.result.modelVersionId)}`;
    const versions = (items: TrainingRun['inputs']) => <Space direction="vertical" size={2}>{items.map(({ link, asset }) =>
        <span key={link.id}>{asset.logicalName} <Typography.Text copyable={{ text: asset.id }} type="secondary">{shortId(asset.id)}</Typography.Text>
            {asset.integrity !== 'verified' && <Tag color="orange">{t('experimentRecords.unverified-version')}</Tag>}</span>)}</Space>;

    return <div className="space-y-5">
        <Alert type="info" message={t('experimentRecords.only-successful-input-verified-executions-with-complete-results-under-identical-evaluation-conditions-share-a-comparison-group')} />
        <div className="grid gap-3 md:grid-cols-2">{selected.map(({ detail, runs, run }) =>
            <div key={detail.experiment.id} className="rounded-lg border p-3">
                <Typography.Text strong>{detail.experiment.name}</Typography.Text>
                <div><Typography.Text copyable={{ text: detail.experiment.id }} type="secondary">{shortId(detail.experiment.id)}</Typography.Text></div>
                {run ? <>
                    <Space wrap className="my-2"><Select size="small" value={run.attempt.id}
                        aria-label={t('experimentRecords.execution')} style={{ minWidth: 180 }}
                        onChange={value => setAttemptSelection(previous => ({ ...previous, [detail.experiment.id]: value }))}
                        options={runs.map(item => ({ value: item.attempt.id,
                            label: `#${item.attempt.attemptNumber} · ${t(`experimentRecords.status.${item.attempt.status}`)} · ${shortId(item.attempt.id)}` }))} />
                    </Space>
                    <Descriptions size="small" column={1} items={[
                        { key: 'time', label: t('experimentRecords.execution-time'), children: `${stamp(run.attempt.startedAt)} → ${stamp(run.attempt.endedAt)}` },
                        { key: 'model', label: t('experimentRecords.base-model'), children: versions(run.inputs.filter(item => item.link.role === 'base_model')) },
                        { key: 'dataset', label: t('experimentRecords.training-dataset'), children: versions(run.inputs.filter(item => item.asset.kind === 'dataset')) },
                        { key: 'outputs', label: t('experimentRecords.produced-model'), children: versions(run.outputs.filter(item => ['model', 'adapter', 'checkpoint'].includes(item.asset.kind))) },
                    ]} />
                </> : <Tag>{t('experimentRecords.no-training-execution')}</Tag>}
            </div>)}</div>

        <section>
            <Space wrap><Typography.Title level={5} className="!mb-0">{t('experimentRecords.evaluation-comparison')}</Typography.Title>
                <Switch size="small" checked={percentage} onChange={setPercentage} aria-label={t('experimentRecords.show-percent')} />
                <Typography.Text>{t('experimentRecords.show-percent')}</Typography.Text></Space>
            {selectedRuns.filter(({ run, training }) => training && !linked.some(item => item.trainingAttemptId === run.attempt.id)).map(({ detail, run }) =>
                <Alert key={run.attempt.id} type="info" className="mt-2" message={`${detail.experiment.name} · #${run.attempt.attemptNumber}: ${t('experimentRecords.no-evaluation-for-training')}`} />)}
            {[...groups.entries()].map(([groupKey, entries]) => {
                const participants = new Map<string, Candidate[]>();
                for (const entry of entries) {
                    const items = participants.get(entry.participant) || [];
                    items.push(entry); participants.set(entry.participant, items);
                }
                const results = [...participants.entries()].map(([participant, items]) =>
                    items.find(item => item.key === resultSelection[`${groupKey}:${participant}`]) || latestResult(items)!);
                const baseline = results.find(item => item.participant === baselines[groupKey]) || results[0];
                const rows = new Map<string, MetricRow>();
                for (const entry of results) for (const metric of entry.metrics) {
                    const dataset = entry.datasets.find(item => item.id === metric.datasetVersionId);
                    const key = JSON.stringify([dataset?.fingerprint || metric.datasetVersionId, metric.subset, metric.name, metric.unit, metric.direction]);
                    const row: MetricRow = rows.get(key) || { key, dataset: dataset ? `${dataset.logicalName} · ${shortId(dataset.fingerprint || dataset.id)}` : metric.datasetVersionId,
                        subset: metric.subset, name: metric.name, unit: metric.unit, direction: metric.direction, values: {} };
                    row.values[entry.key] = { value: metric.value, sampleCount: metric.sampleCount }; rows.set(key, row);
                }
                return <div key={groupKey} className="mt-3 rounded-lg border p-3">
                    <Typography.Text strong>{t('experimentRecords.matching-evaluation-conditions')}</Typography.Text>
                    <Space wrap className="my-2"><Typography.Text>{t('experimentRecords.baseline-result')}</Typography.Text>
                        <Select style={{ minWidth: 240, maxWidth: '100%' }} value={baseline?.key}
                            aria-label={t('experimentRecords.baseline-result')}
                            onChange={value => setBaselines(previous => ({ ...previous, [groupKey]: results.find(item => item.key === value)!.participant }))}
                            options={results.map(entry => ({ value: entry.key, label: candidateLabel(entry) }))} />
                    </Space>
                    {[...participants.entries()].filter(([, items]) => items.length > 1).map(([participant, items]) =>
                        <div key={participant} className="mb-2"><Typography.Text>{candidateLabel(items[0])} · {t('experimentRecords.evaluation')}: </Typography.Text>
                            <Select style={{ minWidth: 250, maxWidth: '100%' }} value={results.find(item => item.participant === participant)?.key}
                                aria-label={t('experimentRecords.select-evaluation-result')}
                                onChange={value => setResultSelection(previous => ({ ...previous, [`${groupKey}:${participant}`]: value }))}
                                options={[...items].sort((a, b) => b.result.createdAt.localeCompare(a.result.createdAt)).map(item => ({
                                    value: item.key, label: `${item.source} · ${stamp(item.result.createdAt)} · ${shortId(item.result.id)}` }))} />
                        </div>)}
                    {results.length < 2 && <Alert type="info" message={t('experimentRecords.only-one-training-evaluated')} />}
                    <Table<MetricRow> className="mt-2" size="small" rowKey="key" pagination={false} dataSource={[...rows.values()]}
                        scroll={{ x: Math.max(800, 360 + results.length * 250) }} columns={[
                            { title: t('experimentRecords.dataset-version'), dataIndex: 'dataset', width: 150 },
                            { title: t('experimentRecords.subset'), dataIndex: 'subset', width: 100 },
                            { title: t('experimentRecords.metric'), dataIndex: 'name', width: 140, render: value => experimentLabel(t, 'metric-labels', value) },
                            ...results.map(entry => ({ key: entry.key, width: 250, title: <div className="whitespace-normal">
                                {candidateLabel(entry)} {entry.key === baseline.key && <Tag>{t('experimentRecords.baseline-result')}</Tag>}
                                <div className="text-xs font-normal">{entry.source} · {stamp(entry.result.createdAt)}</div>
                                <Typography.Text copyable={{ text: entry.result.id }} type="secondary">{shortId(entry.result.id)}</Typography.Text>
                            </div>, render: (_: unknown, row: MetricRow) => {
                                const metric = row.values[entry.key];
                                if (!metric) return '—';
                                const asPercent = percentage && row.unit === 'ratio';
                                const delta = entry.key === baseline.key ? null : metricDelta(metric.value,
                                    row.values[baseline.key]?.value, row.direction, row.unit, percentage);
                                return <div>{number(metric.value * (asPercent ? 100 : 1))}{asPercent ? '%' : ` ${experimentLabel(t, 'unit-labels', row.unit)}`}
                                    {delta && <div><Tag color={delta.trend === 'improved' ? 'green' : delta.trend === 'regressed' ? 'red' : undefined}>
                                        {delta.value > 0 ? '+' : ''}{number(delta.value)}{delta.percentagePoints ? ` ${t('experimentRecords.percentage-points')}` : ''}
                                        {' · '}{t(`experimentRecords.delta-${delta.trend}`)}</Tag></div>}
                                    <div className="text-xs text-gray-500">{t('experimentRecords.samples')}: {metric.sampleCount ?? '—'}</div>
                                </div>;
                            } })),
                        ]} />
                    <Collapse className="mt-2" size="small" items={[{ key: 'protocol', label: t('experimentRecords.evaluation-protocol'),
                        children: <pre className="max-h-[400px] overflow-auto text-xs">{JSON.stringify(results[0]?.result.protocol, null, 2)}</pre> }]} />
                </div>;
            })}
            {candidates.some(item => !item.eligible) && <div className="mt-3">
                <Typography.Text strong>{t('experimentRecords.not-eligible-for-direct-comparison')}</Typography.Text>
                <Table<Candidate> size="small" rowKey="key" pagination={false} scroll={{ x: 850 }} dataSource={candidates.filter(item => !item.eligible)} columns={[
                    { title: t('experimentRecords.experiment'), render: (_, item) => candidateLabel(item) },
                    { title: t('experimentRecords.evaluation-source'), dataIndex: 'source' },
                    { title: t('experimentRecords.reason'), render: (_, item) => t(`experimentRecords.comparison-reason.${item.reason}`) },
                    { title: t('experimentRecords.value'), render: (_, item) => item.metrics.map(metric => {
                        const asPercent = percentage && metric.unit === 'ratio';
                        return `${experimentLabel(t, 'metric-labels', metric.name)}: ${number(metric.value * (asPercent ? 100 : 1))}${asPercent ? '%' : ` ${experimentLabel(t, 'unit-labels', metric.unit)}`}`;
                    }).join(', ') || '—' },
                ]} />
            </div>}
            {!candidates.length && <Empty description={t('experimentRecords.no-evaluation-results')} />}
        </section>

        <section>
            <Space wrap className="mb-2"><Typography.Title level={5} className="!mb-0">{t('experimentRecords.parameter-differences')}</Typography.Title>
                <Select size="small" value={parameterSource} onChange={setParameterSource} aria-label={t('experimentRecords.parameter-source')}
                    options={(['requested', 'resolved', 'actual', 'environment'] as const).map(value => ({ value, label: t(`experimentRecords.parameter-${value}`) }))} />
                <Switch size="small" checked={onlyDifferences} onChange={setOnlyDifferences} aria-label={t('experimentRecords.only-differences')} />
                <Typography.Text>{t('experimentRecords.only-differences')}</Typography.Text></Space>
            {parameterSource === 'actual' && selectedRuns.map(({ detail, run }) => run.attempt.configuration.completeness !== 'complete' &&
                <Alert key={run.attempt.id} className="mb-2" type="warning" message={`${detail.experiment.name}: ${t('experimentRecords.configuration-incomplete')}`}
                    description={<pre className="whitespace-pre-wrap">{JSON.stringify(run.attempt.configuration.missing || [], null, 2)}</pre>} />)}
            <Table size="small" rowKey="path" pagination={false} scroll={{ x: 680 }} dataSource={parameters.filter(row => !onlyDifferences || row.different)}
                locale={{ emptyText: <Empty description={t(parameters.length ? 'experimentRecords.no-parameter-differences' : 'experimentRecords.not-collected')} /> }}
                rowClassName={row => row.different ? 'bg-amber-50/50' : ''} columns={[
                    { title: t('experimentRecords.parameter'), dataIndex: 'path', width: 260, render: path => parameterLabel(t, path) },
                    ...selectedRuns.map(({ detail, run }) => ({ key: run.attempt.id, title: `${detail.experiment.name} · #${run.attempt.attemptNumber}`,
                        render: (_: unknown, row: typeof parameters[number]) => row.values[run.attempt.id] === undefined
                            ? t('experimentRecords.not-collected') : <Typography.Text code className="break-all">{display(row.values[run.attempt.id])}</Typography.Text> })),
                ]} />
            <Collapse className="mt-2" size="small" items={[{ key: 'runtime', label: t('experimentRecords.runtime-information'),
                children: selectedRuns.map(({ detail, run }) => <div key={run.attempt.id}><Typography.Text strong>{detail.experiment.name}</Typography.Text>
                    <pre className="max-h-[320px] overflow-auto whitespace-pre-wrap text-xs">{JSON.stringify({ attemptId: run.attempt.id,
                        ...fields.get(run.attempt.id)?.runtime }, null, 2)}</pre></div>) }]} />
        </section>
        <Collapse size="small" items={[{ key: 'raw', label: t('experimentRecords.raw-records'), children:
            <Collapse size="small" items={selectedRuns.map(({ detail, run }) => ({ key: run.attempt.id,
                label: `${detail.experiment.name} · #${run.attempt.attemptNumber}`, children:
                    <pre className="max-h-[400px] overflow-auto whitespace-pre-wrap break-all text-xs">{JSON.stringify({
                        requested: detail.tasks.find(task => task.id === run.attempt.taskId)?.spec.requested,
                        configuration: run.attempt.configuration, environment: run.attempt.environment }, null, 2)}</pre> }))} /> }]} />
    </div>;
}

