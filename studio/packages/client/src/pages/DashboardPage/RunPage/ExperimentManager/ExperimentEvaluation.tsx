import { useState, type Key, type ReactNode } from 'react';
import { SearchOutlined } from '@ant-design/icons';
import { Alert, Collapse, Empty, Input, Table, Tag, Typography } from 'antd';
import type { ColumnsType } from 'antd/es/table';
import type { inferRouterOutputs } from '@trpc/server';
import { useTranslation } from 'react-i18next';
import type { AppRouter } from '../../../../../../server/src/trpc/router';
import { experimentLabel } from './displayLabels';
import BenchmarkDetails from './BenchmarkDetails';
import { benchmarkReport, displayMetric } from './benchmarkReport';
import { ExperimentLink } from './SourceButton';

type ExperimentDetail = inferRouterOutputs<AppRouter>['getExperiment'];
type EvaluationResult = ExperimentDetail['evaluations'][number];
type Metric = ExperimentDetail['metrics'][number];
type Props = {
    data: ExperimentDetail;
    result: EvaluationResult;
    children: ReactNode;
};

export default function ExperimentEvaluation({ data, result, children }: Props) {
    const { t, i18n } = useTranslation();
    const [search, setSearch] = useState('');
    const [page, setPage] = useState(1);
    const [pageSize, setPageSize] = useState(10);
    const formatNumber = (value: number) => value.toLocaleString(i18n.language, { maximumFractionDigits: 20 });
    const formatCount = (value: number | null) => value === null ? '—' : formatNumber(value);
    const formatValue = (metric: Metric) => {
        const displayed = displayMetric(metric.value, metric.unit);
        return `${displayed.value.toLocaleString(i18n.language, { maximumFractionDigits: 4 })}${displayed.unit === 'percent' || displayed.unit === '%'
            ? '%' : displayed.unit ? ` ${experimentLabel(t, 'unit-labels', displayed.unit)}` : ''}`;
    };
    const metrics = data.metrics.filter(metric => metric.resultId === result.id);
    // Only promote recorded overall scores; never average subset scores.
    const overall = metrics.filter(metric => metric.subset === 'all');
    const subsets = metrics.filter(metric => metric.subset !== 'all');
    const names = [...new Set(subsets.map(metric => metric.name))];
    const datasetIds = [...new Set(subsets.map(metric => metric.datasetVersionId))];
    const datasetName = (id: string) => data.assets.find(asset => asset.id === id)?.logicalName || id;
    const datasetLabel = (id: string) => {
        const name = datasetName(id);
        return name === id ? id : `${name} · ${id.slice(0, 16)}…`;
    };
    const model = data.assets.find(asset => asset.id === result.modelVersionId);
    const attempt = data.attempts.find(item => item.id === result.attemptId);
    const taskKind = data.tasks.find(task => task.id === result.taskId)?.kind
        || data.linkedEvaluations.find(item => item.result.id === result.id)?.evaluationTaskKind;
    const resultTitle = taskKind === 'benchmark' ? 'benchmark-result'
        : taskKind === 'evaluate' ? 'model-evaluation-result' : 'evaluation-result';
    const report = benchmarkReport(attempt?.runtime.benchmarkReport);
    const query = search.trim().toLocaleLowerCase(i18n.language);
    const filteredSubsets = subsets.filter(metric => metric.subset.toLocaleLowerCase(i18n.language).includes(query));
    const columns: ColumnsType<Metric> = [
        { title: t('experimentRecords.subset'), dataIndex: 'subset', render: value => <span className="evaluation-subset-name">{value || '—'}</span> },
        ...(datasetIds.length > 1 ? [{ title: t('experimentRecords.dataset-version'), dataIndex: 'datasetVersionId',
            filters: datasetIds.map(id => ({ text: datasetLabel(id), value: id })),
            onFilter: (value: Key | boolean, metric: Metric) => metric.datasetVersionId === value,
            render: (id: string) => <span className="evaluation-dataset-name" title={id}>{datasetLabel(id)}</span> }] : []),
        ...(names.length > 1 ? [{ title: t('experimentRecords.metric'), dataIndex: 'name',
            filters: names.map(name => ({ text: experimentLabel(t, 'metric-labels', name), value: name })),
            onFilter: (value: Key | boolean, metric: Metric) => metric.name === value,
            render: (name: string) => experimentLabel(t, 'metric-labels', name) }] : []),
        { title: t('experimentRecords.value'), key: 'value', align: 'right', width: 160,
            sorter: new Set(subsets.map(metric => `${metric.name}:${metric.unit}`)).size === 1
                ? (left, right) => left.value - right.value : undefined,
            render: (_, metric) => <span className="evaluation-table-value">{formatValue(metric)}</span> },
        { title: t('experimentRecords.samples'), dataIndex: 'sampleCount', align: 'right', width: 100,
            render: (value: number | null) => <span title={value === null ? t('experimentRecords.unknown') : undefined}>{formatCount(value)}</span> },
    ];

    return <section className="experiment-evaluation">
        <header className="evaluation-header">
            <div className="flex flex-wrap items-center gap-2">
                <h3 className="m-0 text-base font-semibold">{t(`experimentRecords.${resultTitle}`)}</h3>
                {attempt && <Tag>{t('experimentRecords.execution')} #{attempt.attemptNumber}</Tag>}
            </div>
            <Typography.Text type="secondary" className="evaluation-model" title={result.modelVersionId}>
                {t('experimentRecords.model-version')}: {model ? `${model.logicalName} · ${result.modelVersionId.slice(0, 16)}…` : result.modelVersionId}
            </Typography.Text>
            {typeof attempt?.runtime.deploymentExperimentId === 'string' && <div className="mt-1 flex items-center gap-2">
                <Typography.Text type="secondary">{t('experimentRecords.service-deployment-record')}</Typography.Text>
                <ExperimentLink experimentId={attempt.runtime.deploymentExperimentId} />
            </div>}
        </header>
        <dl className="evaluation-stats">
            <div>
                <dt>{t(report ? 'experimentRecords.successful-samples' : 'experimentRecords.completed-samples')}</dt>
                <dd>{formatCount(result.succeededCount)} <span>/ {formatCount(result.expectedCount)}</span></dd>
            </div>
            <div>
                <dt>{t('experimentRecords.failed')}</dt>
                <dd className={result.failedCount !== null && result.failedCount > 0 ? 'evaluation-failed' : undefined}>{formatCount(result.failedCount)}</dd>
            </div>
            <div>
                <dt>{t('experimentRecords.metric-count')}</dt>
                <dd>{formatNumber(metrics.length)}</dd>
            </div>
        </dl>
        {!result.protocolComplete && <Alert className="mb-4" type="warning" showIcon
            message={t('experimentRecords.incomplete-protocol-score-improvements-are-not-directly-comparable')} />}
        {overall.length > 0 && <section className="evaluation-section" aria-label={t('experimentRecords.overall-metrics')}>
            <h4 className="evaluation-section-title">{t('experimentRecords.overall-metrics')}</h4>
            <div className="evaluation-score-grid">
                {overall.map(metric => <article key={metric.id} className="evaluation-score">
                    <div className="evaluation-score-label" title={metric.name}>{experimentLabel(t, 'metric-labels', metric.name)}</div>
                    <div className="evaluation-score-value">{formatValue(metric)}</div>
                    <div className="evaluation-score-dataset" title={metric.datasetVersionId}>{datasetLabel(metric.datasetVersionId)}</div>
                    <div className="evaluation-score-samples">{t('experimentRecords.samples')}: {formatCount(metric.sampleCount)}</div>
                </article>)}
            </div>
        </section>}
        {subsets.length > 0 && <section className="evaluation-section" aria-label={t('experimentRecords.subset-details')}>
            <div className="evaluation-table-heading">
                <div className="flex flex-wrap items-center gap-2">
                    <h4 className="evaluation-section-title">{t('experimentRecords.subset-details')}</h4>
                    {names.length === 1 && <Tag>{experimentLabel(t, 'metric-labels', names[0])}</Tag>}
                </div>
                <Input className="evaluation-search" prefix={<SearchOutlined />} allowClear value={search}
                    aria-label={t('experimentRecords.search-subsets')} placeholder={t('experimentRecords.search-subsets')}
                    onChange={event => { setSearch(event.target.value); setPage(1); }} />
            </div>
            {datasetIds.length === 1 && <p className="evaluation-table-dataset" title={datasetIds[0]}>{datasetLabel(datasetIds[0])}</p>}
            <Table<Metric> className="evaluation-table" size="small" rowKey="id" scroll={{ x: datasetIds.length > 1 ? 700 : 480 }}
                dataSource={filteredSubsets} columns={columns}
                locale={{ emptyText: <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={t('experimentRecords.no-matching-subsets')} /> }}
                pagination={{ current: page, pageSize, showSizeChanger: true, pageSizeOptions: [10, 20, 50],
                    showTotal: total => t('experimentRecords.metric-rows', { count: total }) }}
                onChange={pagination => { setPage(pagination.current || 1); setPageSize(pagination.pageSize || 10); }} />
        </section>}
        {!metrics.length && <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={t('experimentRecords.no-metrics-recorded')} />}
        {report && <BenchmarkDetails key={result.id} report={report} />}
        {Boolean(attempt?.runtime.benchmarkDetailsError) && <Alert className="mb-3" type="warning"
            message={t('experimentRecords.sample-details-unavailable')} />}
        <Collapse className="evaluation-technical-details" ghost size="small" items={[
            { key: 'technical', label: t('experimentRecords.technical-details'), children },
        ]} />
    </section>;
}
