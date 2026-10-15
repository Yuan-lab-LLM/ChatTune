import { useEffect, useId, useState } from 'react';
import { CopyOutlined, ExportOutlined, FilterOutlined, ImportOutlined } from '@ant-design/icons';
import { Alert, Button, Collapse, ConfigProvider, Descriptions, Drawer, Empty, Space, Spin, Table, Tabs, Tag, Typography, message } from 'antd';
import enUS from 'antd/locale/en_US';
import zhCN from 'antd/locale/zh_CN';
import { useTranslation } from 'react-i18next';
import { trpc } from '@/api/trpc';
import { useStudioSidebar } from '@/context/SidebarContext';
import type { ExperimentListFilters } from '@shared/experiment';
import ExperimentFilters from './ExperimentFilters';
import ExperimentComparison from './ExperimentComparison';
import ExperimentLogs from './ExperimentLogs';
import ExperimentEvaluation from './ExperimentEvaluation';
import ExperimentObservedOutputs from './ExperimentObservedOutputs';
import { evaluationOutputs } from './evaluationOutputs';
import { hasEvaluationTab } from './evaluationVisibility';
import ExperimentSourceButton from './SourceButton';
import { associationLabel, associationRole } from './associationLabels';
import { executionLabel, experimentLabel, experimentName } from './displayLabels';
import { useExperimentQueries } from './useExperimentQueries';
import ExperimentMetadata from './ExperimentMetadata';
import ExperimentServiceOperations from './ExperimentServiceOperations';
import './index.css';

const JsonView = ({ value }: { value: unknown }) => {
    const { t } = useTranslation();
    const json = JSON.stringify(value, null, 2) ?? 'null';
    const copyJson = async () => {
        try {
            if (navigator.clipboard?.writeText) {
                await navigator.clipboard.writeText(json);
            } else {
                const textarea = document.createElement('textarea');
                textarea.value = json;
                textarea.style.position = 'fixed';
                textarea.style.opacity = '0';
                document.body.appendChild(textarea);
                try {
                    textarea.select();
                    if (!document.execCommand('copy')) throw new Error('Copy failed');
                } finally {
                    textarea.remove();
                }
            }
            message.success(t('experimentRecords.json-copied'));
        } catch {
            message.error(t('experimentRecords.copy-json-failed'));
        }
    };
    return <div>
        <div className="mb-2 flex justify-end">
            <Button size="small" type="text" icon={<CopyOutlined />} onClick={copyJson}>
                {t('experimentRecords.copy-json')}
            </Button>
        </div>
        <pre className="max-h-[480px] overflow-auto rounded-lg bg-muted/50 p-3 text-xs whitespace-pre-wrap break-all">
            {json}
        </pre>
    </div>;
};
const asRecord = (value: unknown): Record<string, unknown> => value && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown> : {};
const textField = (value: unknown): string | undefined => typeof value === 'string' && value ? value : undefined;
const incompleteModelShards = (manifest: unknown): string | undefined => {
    const files = asRecord(manifest).files;
    if (!Array.isArray(files)) {
        const summary = asRecord(asRecord(manifest).shardSummary);
        return summary.complete === false ? `${summary.present}/${summary.expected}` : undefined;
    }
    const names = files.map(file => textField(asRecord(file).path)).filter((name): name is string => Boolean(name));
    const shards = names.map(name => /^model-(\d+)-of-(\d+)\.safetensors$/.exec(name)).filter((match): match is RegExpExecArray => Boolean(match));
    if (!shards.length) return undefined;
    const expected = Number(shards[0][2]);
    const numbers = new Set(shards.map(match => Number(match[1])));
    return numbers.size !== expected || shards.some(match => Number(match[2]) !== expected)
        || !names.includes('model.safetensors.index.json')
        ? `${numbers.size}/${expected}` : undefined;
};
const StatusTag = ({ status, filterOutcome }: { status: string; filterOutcome?: unknown }) => {
    const { t } = useTranslation();
    return (
        <Tag color={status === 'succeeded' || status === 'verified' ? 'green'
        : status === 'failed' || status === 'missing' ? 'red' : status === 'running' ? 'blue' : 'default'}>{t(filterOutcome === 'no_qualifying_data' && status === 'succeeded' ? 'experimentRecords.filter-empty-status'
            : filterOutcome === 'scoring_failed' && status === 'failed' ? 'experimentRecords.filter-scoring-failed-status'
            : `experimentRecords.status.${status}`, { defaultValue: status })}</Tag>
    );
};

export default function ExperimentManager() {
    const { t, i18n } = useTranslation();
    const formatTime = (value: string | null) => value && !Number.isNaN(Date.parse(value))
        ? new Date(value).toLocaleString(i18n.language.startsWith('zh') ? 'zh-CN' : 'en-US') : '—';
    const [offset, setOffset] = useState(0);
    const { selectedExperimentId: selectedId, setSelectedExperimentId: setSelectedId } = useStudioSidebar();
    const [selectedKeys, setSelectedKeys] = useState<string[]>([]);
    const [comparing, setComparing] = useState(false);
    const [filters, setFilters] = useState<ExperimentListFilters>({});
    const [filtersOpen, setFiltersOpen] = useState(false);
    const [tab, setTab] = useState('overview');
    useEffect(() => { setTab('overview'); }, [selectedId]);
    const filtersPanelId = useId();
    const filterCount = [filters.search, filters.status, filters.taskKind, filters.project, filters.tag,
        filters.createdFrom || filters.createdBefore].filter(Boolean).length;
    const applyFilters = (next: ExperimentListFilters) => {
        setFilters(next); setOffset(0); setSelectedKeys([]); setComparing(false);
    };
    const { list, detail, comparison, data, visible } = useExperimentQueries({ selectedId, tab, comparing, selectedKeys, filters, offset });
    // Combine only the evaluation view, keeping other tabs scoped to this experiment.
    const evaluationData = data && { ...data,
        evaluations: [...new Map([...data.evaluations, ...data.linkedEvaluations.map(item => item.result)]
            .map(result => [result.id, result])).values()],
        attempts: [...new Map([...data.attempts, ...data.linkedEvaluations.map(item => item.evaluationAttempt)]
            .map(attempt => [attempt.id, attempt])).values()],
        assets: [...new Map([...data.assets, ...data.linkedEvaluations.flatMap(item => [item.model, ...item.datasets])]
            .map(asset => [asset.id, asset])).values()],
        metrics: [...new Map([...data.metrics, ...data.linkedEvaluations.flatMap(item => item.metrics)]
            .map(metric => [metric.id, metric])).values()],
    };
    const showEvaluation = hasEvaluationTab(data);
    const inputAssetIds = new Set(data?.links.filter(link => link.direction === 'input').map(link => link.assetVersionId));
    const outputAssetIds = new Set(data?.links.filter(link => link.direction === 'output').map(link => link.assetVersionId));
    const observedOutputs = data ? evaluationOutputs(data.attempts, data.assets, data.links) : [];
    const unarchivedOutputCount = observedOutputs.reduce((count, group) => count
        + group.items.filter(item => item.exists && !item.archivedVersionId).length, 0);
    const assetGroups = [
        { key: 'input', title: t('experimentRecords.input-assets'), description: t('experimentRecords.input-assets-description'),
            icon: <ImportOutlined />, assets: data?.assets.filter(asset => inputAssetIds.has(asset.id)) || [] },
        { key: 'output', title: t('experimentRecords.output-assets'), description: t('experimentRecords.output-assets-description'),
            icon: <ExportOutlined />, assets: data?.assets.filter(asset => outputAssetIds.has(asset.id)) || [] },
        { key: 'other', title: t('experimentRecords.other-assets'), description: t('experimentRecords.other-assets-description'),
            icon: null, assets: data?.assets.filter(asset => !inputAssetIds.has(asset.id) && !outputAssetIds.has(asset.id)) || [] },
    ].filter(group => group.key !== 'other' || group.assets.length > 0);
    const cancelledWithOutputs = Boolean(data?.attempts.some(attempt => attempt.status === 'cancelled'
        && data.links.some(link => link.attemptId === attempt.id && link.direction === 'output')));
    const archiveIssue = data?.attempts.find(attempt => attempt.runtime.archiveStatus === 'retrying'
        || attempt.runtime.archiveWarning);
    const incompleteOutputs = data?.assets.filter(asset => asset.kind === 'model'
        && data.links.some(link => link.assetVersionId === asset.id && link.direction === 'output')
        && incompleteModelShards(asset.manifest));
    const isDatasetUpload = data?.experiment.spec.operation === 'import' && data.experiment.spec.script === 'dataset_upload';
    const uploadAttempt = isDatasetUpload ? data?.attempts[data.attempts.length - 1] : undefined;
    const uploadRuntime = asRecord(uploadAttempt?.runtime);
    const uploadInfo = asRecord(uploadRuntime.upload);
    const uploadedVersionId = textField(uploadRuntime.uploadedVersionId);
    const uploadAsset = isDatasetUpload ? data?.assets.find(asset => asset.id === uploadedVersionId)
        || data?.assets.find(asset => data?.links.some(link => link.assetVersionId === asset.id && link.role === 'uploaded_dataset'
            && link.attemptId === uploadAttempt?.id)) : undefined;
    const uploadVerified = uploadAttempt?.status === 'succeeded' && uploadRuntime.archiveStatus === 'complete'
        && uploadAsset?.integrity === 'verified';
    const uploadSummary = uploadVerified ? 'upload-complete-and-verified'
        : uploadAttempt?.status === 'failed' ? 'upload-failed'
        : uploadRuntime.archiveStatus === 'retrying' ? 'upload-archive-retrying'
        : uploadAttempt?.status === 'succeeded' ? 'upload-version-not-verified' : 'upload-archive-pending';
    const launch = trpc.launchExperimentDraft.useMutation({
        onSuccess: async () => { message.success(t('experimentRecords.execution-submitted-refresh-to-view-status')); await list.refetch(); await detail.refetch(); },
        onError: error => message.error(error.message),
    });
    const items = list.data?.items || [];
    return <ConfigProvider locale={i18n.language.startsWith('zh') ? zhCN : enUS}><section className="flex h-full min-h-0 flex-col gap-4 overflow-auto p-4">
        <div>
            <Typography.Title level={4}>{t('experimentRecords.experiments')}</Typography.Title>
            <Typography.Text type="secondary">{t('experimentRecords.trace-data-configuration-executions-models-and-evaluations')}</Typography.Text>
        </div>
        <Space wrap size={8}>
            <Button size="small" icon={<FilterOutlined />} type={filterCount ? 'primary' : 'default'}
                aria-expanded={filtersOpen} aria-controls={filtersPanelId} onClick={() => setFiltersOpen(open => !open)}>
                {t(filtersOpen ? 'experimentRecords.hide-filters' : 'experimentRecords.show-filters')}{filterCount ? ` (${filterCount})` : ''}
            </Button>
            <Button size="small" onClick={() => list.refetch()} loading={list.isFetching}>{t('experimentRecords.refresh')}</Button>
            <Typography.Text type="secondary">{t('experimentRecords.last-refreshed')}: {list.dataUpdatedAt ? formatTime(new Date(list.dataUpdatedAt).toISOString()) : '—'}</Typography.Text>
            <Button size="small" disabled={selectedKeys.length < 2 || selectedKeys.length > 5} onClick={() => setComparing(true)}>{t('experimentRecords.compare-selected')}</Button>
        </Space>
        <div id={filtersPanelId} hidden={!filtersOpen}>
            <ExperimentFilters onApply={applyFilters} />
        </div>
        {list.error && <Alert type="error" message={list.error.message} />}
        <Table size="small" rowKey="id" loading={list.isFetching} dataSource={items} locale={{ emptyText: <Empty description={t('experimentRecords.no-matching-experiments')} /> }}
            rowSelection={{ selectedRowKeys: selectedKeys, preserveSelectedRowKeys: true, onChange: keys => { setSelectedKeys(keys.map(String)); setComparing(false); } }}
            pagination={{ current: offset / 30 + 1, pageSize: 30, total: list.data?.total || 0, showSizeChanger: false,
                showTotal: total => t('experimentRecords.filtered-total', { count: total }),
                onChange: page => setOffset((page - 1) * 30) }}
            columns={[
                { title: t('experimentRecords.experiment'), key: 'name', render: (_, row) => <Button type="link" className="!h-auto !whitespace-normal !p-0 text-left" onClick={() => setSelectedId(row.id)}>{experimentName(t, row)}</Button> },
                { title: t('experimentRecords.status-label'), dataIndex: 'status', render: status => <StatusTag status={status} /> },
                { title: t('experimentRecords.tags'), dataIndex: 'tags', render: tags => (tags || []).map((tag: string) => <Tag key={tag}>{tag}</Tag>) },
                { title: t('experimentRecords.created'), dataIndex: 'createdAt', render: value => new Date(value).toLocaleString(i18n.language.startsWith('zh') ? 'zh-CN' : 'en-US') },
            ]} />
        <Drawer open={Boolean(selectedId)} onClose={() => setSelectedId(undefined)} width="min(100vw, 960px)"
            title={data ? experimentName(t, data.experiment) : t('experimentRecords.experiment-details')}
            extra={data?.experiment.status === 'draft' && ['train', 'evaluate'].includes(String(data.experiment.spec.kind)) &&
                <Button type="primary" loading={launch.isPending}
                    onClick={() => selectedId && launch.mutate({ experimentId: selectedId })}>{t('experimentRecords.launch-draft')}</Button>}>
            {detail.error && <Alert type="error" message={detail.error.message} />}
            {!data ? <Spin /> : <>
                <Space wrap className="mb-3"><Button size="small" onClick={() => detail.refetch()} loading={detail.isFetching}>{t('experimentRecords.refresh')}</Button>
                    <Typography.Text type="secondary">{t('experimentRecords.last-refreshed')}: {detail.dataUpdatedAt ? formatTime(new Date(detail.dataUpdatedAt).toISOString()) : '—'}</Typography.Text>
                    {detail.isLoading && <Spin size="small" />}
                </Space>
                <div className="mb-3"><Typography.Text type="secondary" copyable={{ text: data.experiment.id }}>
                    {t('experimentRecords.record-id')}: {data.experiment.id.slice(0, 20)}…
                </Typography.Text></div>
                <Tabs key={data.experiment.id} activeKey={tab} onChange={setTab} items={[
                    { key: 'overview', label: t('experimentRecords.overview'), children: <>
                        <ExperimentMetadata key={data.experiment.id} experimentId={data.experiment.id}
                            notes={data.experiment.notes || ''} tags={data.experiment.tags || []}
                            onSaved={async () => { await detail.refetch(); await list.refetch(); }} />
                        {isDatasetUpload ? <>
                        <Alert type={uploadVerified ? 'success' : uploadAttempt?.status === 'failed' ? 'error' : 'warning'} showIcon
                            message={t(`experimentRecords.${uploadSummary}`)} className="mb-4" />
                        <Descriptions column={1} size="small" bordered items={[
                            { key: 'file', label: t('experimentRecords.upload-file'), children: textField(uploadInfo.filename) || textField(data.experiment.spec.filename) || '—' },
                            { key: 'dataset', label: t('experimentRecords.upload-dataset'), children: uploadAsset?.logicalName || '—' },
                            { key: 'status', label: t('experimentRecords.status-label'), children: <StatusTag status={uploadAttempt?.status || data.experiment.status} /> },
                            { key: 'archive', label: t('experimentRecords.archive-status'), children: t(`experimentRecords.archive.${String(uploadRuntime.archiveStatus || 'pending')}`, { defaultValue: String(uploadRuntime.archiveStatus || 'pending') }) },
                            { key: 'integrity', label: t('experimentRecords.content-verification'), children: uploadAsset ? <StatusTag status={uploadAsset.integrity} /> : t('experimentRecords.version-pending') },
                            { key: 'version', label: t('experimentRecords.uploaded-version'), children: uploadAsset
                                ? <Typography.Text copyable={{ text: uploadAsset.id }}>{uploadAsset.id.slice(0, 20)}…</Typography.Text>
                                : t('experimentRecords.version-pending') },
                        ]} />
                    </> : <>
                        <Descriptions column={1} size="small" bordered items={[
                            { key: 'status', label: t('experimentRecords.status-label'), children: <StatusTag status={data.experiment.status} /> },
                            { key: 'project', label: t('experimentRecords.project'), children: data.experiment.project },
                            { key: 'workflow', label: t('experimentRecords.workflow'), children: data.experiment.workflowId || '—' },
                            { key: 'parent', label: t('experimentRecords.source-experiment'), children: data.experiment.parentExperimentId ? <Button type="link" onClick={() => setSelectedId(data.experiment.parentExperimentId!)}>{data.experiment.parentExperimentId}</Button> : '—' },
                            { key: 'counts', label: t(showEvaluation ? 'experimentRecords.executions-and-evaluations' : 'experimentRecords.execution'),
                                children: showEvaluation ? `${data.attempts.length} / ${evaluationData?.evaluations.length || 0}` : data.attempts.length },
                            { key: 'versions', label: t('experimentRecords.input-and-output-versions'), children: `${new Set(data.links.filter(link => link.direction === 'input').map(link => link.assetVersionId)).size} / ${new Set(data.links.filter(link => link.direction === 'output').map(link => link.assetVersionId)).size}` },
                        ]} />
                        {data.attempts.some(a => a.configuration.completeness !== 'complete') && <Alert className="mt-4" type="warning" showIcon
                            message={t('experimentRecords.some-executions-lack-a-complete-configuration-from-the-trainer')} />}
                        {cancelledWithOutputs && <Alert className="mt-4" type="info" showIcon
                            message={t('experimentRecords.cancelled-execution-has-output-artifacts')} />}
                        {Boolean(incompleteOutputs?.length) && <Alert className="mt-4" type="error" showIcon
                            message={t('experimentRecords.incomplete-model-shards')} />}
                        {archiveIssue && <Alert className="mt-4" type="warning" showIcon
                            message={t('experimentRecords.some-artifacts-are-pending-or-missing-inspect-the-execution-history')}
                            description={textField(archiveIssue.runtime.archiveError) || textField(archiveIssue.runtime.archiveWarning)} />}
                        {data.attempts.some(a => a.runtime.inputIntegrityAfterExecution === 'unverified') && <Alert className="mt-4" type="error" showIcon
                            message={t('experimentRecords.some-inputs-failed-verification-after-execution-affected-results-are-excluded-from-direct-comparison')} />}
                    </>}
                    </> },
                    { key: 'configuration', label: t('experimentRecords.configuration'), children: <>
                        <Descriptions column={1} size="small" bordered items={[
                            { key: 'kind', label: t('experimentRecords.experiment-type'), children: textField(data.experiment.spec.kind)
                                ? t(`experimentRecords.kind.${data.experiment.spec.kind}`, { defaultValue: String(data.experiment.spec.kind) }) : '—' },
                            { key: 'script', label: t('experimentRecords.operation'), children: experimentLabel(t, 'operation-labels', textField(data.experiment.spec.operation) || textField(data.experiment.spec.script) || '—') },
                        ]} />
                        <Collapse className="mt-4" items={[
                            { key: 'spec', label: t('experimentRecords.specification'), children: <JsonView value={data.experiment.spec} /> },
                            ...data.attempts.map(a => ({ key: a.id, label: `${executionLabel(t, a, data.tasks)} · ${t('experimentRecords.configuration-details')}`,
                                children: <JsonView value={{ configuration: a.configuration, environment: a.environment }} /> })),
                        ]} />
                    </> },
                    { key: 'executions', label: t('experimentRecords.executions'), children: isDatasetUpload ? data.attempts.length ? data.attempts.map(a => {
                        const runtime = asRecord(a.runtime);
                        const upload = asRecord(runtime.upload);
                        return <div key={a.id} className="mb-4 rounded-lg border p-4">
                            <Space wrap><Typography.Text strong>{t('experimentRecords.dataset-upload-record')}</Typography.Text><StatusTag status={a.status} /></Space>
                            <Descriptions className="mt-3" column={1} size="small" items={[
                                { key: 'file', label: t('experimentRecords.upload-file'), children: textField(upload.filename) || '—' },
                                { key: 'archive', label: t('experimentRecords.archive-status'), children: <Tag color={runtime.archiveStatus === 'complete' ? 'green' : 'default'}>
                                    {t(`experimentRecords.archive.${String(runtime.archiveStatus || 'pending')}`, { defaultValue: String(runtime.archiveStatus || 'pending') })}</Tag> },
                                { key: 'time', label: t('experimentRecords.execution-time'), children: `${formatTime(a.startedAt)} → ${formatTime(a.endedAt)}` },
                            ]} />
                            {a.error && <Alert type="error" message={a.error} className="mb-3" />}
                            <Collapse size="small" items={[{ key: 'technical', label: t('experimentRecords.technical-details'), children: <>
                                <Typography.Paragraph copyable>{a.id}</Typography.Paragraph>
                                <p>{a.nodeId} / {a.container}</p>
                                <JsonView value={a.runtime} />
                            </> }]} />
                        </div>;
                    }) : <Empty description={t('experimentRecords.no-data')} /> : data.attempts.length ? data.attempts.map(a => <div key={a.id} className="mb-4 rounded-lg border p-4">
                        <Space wrap><Typography.Text strong>{executionLabel(t, a, data.tasks)}</Typography.Text><StatusTag status={a.status} filterOutcome={asRecord(a.runtime.dataProcessing).outcome} /></Space>
                        <Descriptions className="mt-3" column={1} size="small" items={[
                            { key: 'time', label: t('experimentRecords.execution-time'), children: `${formatTime(a.startedAt)} → ${formatTime(a.endedAt)}` },
                            { key: 'location', label: t('experimentRecords.execution-location'), children: `${a.nodeId} / ${a.container}` },
                            ...(textField(a.runtime.serviceInstanceId) ? [{ key: 'instance', label: t('experimentRecords.service-instance'),
                                children: <Typography.Text copyable>{String(a.runtime.serviceInstanceId)}</Typography.Text> }] : []),
                            { key: 'archive', label: t('experimentRecords.archive-status'), children: t(`experimentRecords.archive.${String(a.runtime.archiveStatus || 'pending')}`, { defaultValue: String(a.runtime.archiveStatus || 'pending') }) },
                            ...(typeof a.runtime.exitCode === 'number' ? [{ key: 'exit-code', label: t('experimentRecords.exit-code'), children: String(a.runtime.exitCode) }] : []),
                            ...(Array.isArray(a.runtime.signals) && a.runtime.signals.length ? [{ key: 'stop-signals', label: t('experimentRecords.stop-signals'), children: a.runtime.signals.map(String).join(', ') }] : []),
                        ]} />
                        {a.status === 'succeeded' && asRecord(a.runtime.dataProcessing).outcome === 'no_qualifying_data'
                            && <Alert type="info" message={t('experimentRecords.filter-empty-description')} className="mb-3" />}
                        {a.error && <Alert type="error" message={asRecord(a.runtime.dataProcessing).outcome === 'scoring_failed'
                            ? t('experimentRecords.filter-scoring-failed-description', {
                                failed: asRecord(a.runtime.dataProcessing).api_failed_items,
                                total: asRecord(a.runtime.dataProcessing).total_items }) : a.error} className="mb-3" />}
                        {(textField(a.runtime.archiveError) || textField(a.runtime.archiveWarning)) && <Alert type="warning"
                            message={textField(a.runtime.archiveError) || textField(a.runtime.archiveWarning)} className="mb-3" />}
                        <Collapse size="small" items={[{ key: 'technical', label: t('experimentRecords.technical-details'), children: <>
                            <Typography.Paragraph copyable>{a.id}</Typography.Paragraph>
                            {a.resourceReservationId && <p>{t('experimentRecords.resource-reservation')}: <Typography.Text copyable>{a.resourceReservationId}</Typography.Text></p>}
                            {a.resumeFromVersionId && <p>{t('experimentRecords.resume-checkpoint')}: <Typography.Text copyable>{a.resumeFromVersionId}</Typography.Text>
                                <ExperimentSourceButton resourceKind="model" assetVersionId={a.resumeFromVersionId} /></p>}
                            <JsonView value={a.runtime} />
                        </> }]} />
                    </div>) : <Empty description={t('experimentRecords.no-data')} /> },
                    ...(data.tasks.some(task => task.kind === 'deploy') ? [{ key: 'operations', label: t('experimentRecords.service-history'),
                        children: <ExperimentServiceOperations operations={data.serviceOperations || []} /> }] : []),
                    { key: 'logs', label: t('experimentRecords.logs'), children:
                        <ExperimentLogs experimentId={data.experiment.id} attempts={data.attempts} tasks={data.tasks} active={visible && Boolean(selectedId) && tab === 'logs'} /> },
                    { key: 'assets', label: t('experimentRecords.data-and-artifacts'), children: isDatasetUpload ? uploadAsset ? <div className="rounded-lg border p-4">
                        <Space wrap><Typography.Text strong>{uploadAsset.logicalName}</Typography.Text><Tag>{t('experimentRecords.uploaded-dataset')}</Tag><StatusTag status={uploadAsset.integrity} /></Space>
                        <Descriptions className="mt-3" column={1} size="small" items={[
                            { key: 'version', label: t('experimentRecords.uploaded-version'), children: <Typography.Text copyable={{ text: uploadAsset.id }}>{uploadAsset.id.slice(0, 20)}…</Typography.Text> },
                            { key: 'source', label: t('experimentRecords.upload-file'), children: textField(uploadAsset.metadata.originalFilename) || '—' },
                            { key: 'integrity', label: t('experimentRecords.content-verification'), children: <StatusTag status={uploadAsset.integrity} /> },
                        ]} />
                        {data.lineage.filter(link => link.assetVersionId === uploadAsset.id && link.experimentId !== data.experiment.id).map(link =>
                            <Space key={`${link.attemptId}:${link.relatedAssetVersionId}:${link.direction}:${link.role}`} wrap className="mb-2">
                                <Tag>{t(`experimentRecords.${associationLabel(link)}`)}</Tag>
                                <Button type="link" onClick={() => setSelectedId(link.experimentId)}>{link.name}</Button>
                            </Space>)}
                        <Collapse size="small" items={[{ key: 'technical', label: t('experimentRecords.technical-details'), children: <>
                            <Typography.Paragraph copyable>{uploadAsset.fingerprint || t('experimentRecords.content-unverified')}</Typography.Paragraph>
                            <JsonView value={{ location: uploadAsset.location, metadata: uploadAsset.metadata,
                                relationships: data.links.filter(link => link.assetVersionId === uploadAsset.id), manifest: uploadAsset.manifest }} />
                        </> }]} />
                    </div> : <Empty description={t('experimentRecords.version-pending')} /> : data.assets.length || observedOutputs.length ? <div className="experiment-assets">
                        {assetGroups.map(group => <section key={group.key} className={`experiment-assets-group experiment-assets-group--${group.key}`}
                            aria-labelledby={`experiment-assets-${group.key}`}>
                            <header className="experiment-assets-heading">
                                <div className="flex items-center gap-2">
                                    {group.icon}
                                    <h3 id={`experiment-assets-${group.key}`} className="m-0 text-base font-semibold">{group.title}</h3>
                                    {group.key === 'output' && observedOutputs.length ? <Space size={8} wrap>
                                        <span className="experiment-assets-count">{t('experimentRecords.archived-output-count', { count: group.assets.length })}</span>
                                        <span className="experiment-assets-count">{t('experimentRecords.unarchived-output-count', { count: unarchivedOutputCount })}</span>
                                    </Space> : <span className="experiment-assets-count">{group.assets.length}</span>}
                                </div>
                                <p className="mt-1 mb-0 text-sm text-muted-foreground">{group.description}</p>
                            </header>
                            {group.key === 'output' && <ExperimentObservedOutputs groups={observedOutputs} />}
                            <div className="experiment-assets-list">
                                {group.assets.length ? group.assets.map(asset => {
                                    const lineage = data.lineage.filter(link => link.assetVersionId === asset.id && link.experimentId !== data.experiment.id);
                                    const incomplete = asset.kind === 'model' && incompleteModelShards(asset.manifest);
                                    return <article key={asset.id} className="experiment-asset-card">
                                        <div className="flex flex-wrap items-start justify-between gap-2">
                                            <div className="min-w-0 flex-1">
                                                <Typography.Text strong className="experiment-asset-name">{asset.logicalName}</Typography.Text>
                                            </div>
                                            <Space size={4} wrap>
                                                <Tag>{t(`experimentRecords.asset-kind.${asset.kind}`, { defaultValue: asset.kind })}</Tag>
                                                <StatusTag status={incomplete ? 'incomplete' : asset.integrity} />
                                            </Space>
                                        </div>
                                        <div className="mt-2 flex flex-wrap items-center gap-x-2 gap-y-1 text-sm">
                                            <Typography.Text type="secondary">{t('experimentRecords.asset-version')}:</Typography.Text>
                                            <Typography.Text copyable={{ text: asset.id }} title={asset.id}>{asset.id.slice(0, 20)}…</Typography.Text>
                                        </div>
                                        {lineage.length > 0 && <div className="mt-3 flex flex-col gap-2">
                                            {lineage.map(link => <div key={`${link.attemptId}:${link.relatedAssetVersionId}:${link.direction}:${link.role}`} className="flex flex-wrap items-center gap-2">
                                                <Typography.Text type="secondary">{t(`experimentRecords.${associationLabel(link)}`)}:</Typography.Text>
                                                <Button type="link" className="!h-auto !whitespace-normal !p-0 text-left break-all" onClick={() => setSelectedId(link.experimentId)}>{link.name}</Button>
                                                <Typography.Text type="secondary">{t(`experimentRecords.asset-role.${associationRole(link)}`, { defaultValue: link.role })}</Typography.Text>
                                                {link.match === 'content' && <Tag>{t('experimentRecords.same-asset-content')}</Tag>}
                                            </div>)}
                                        </div>}
                                        <Collapse className="experiment-asset-details" ghost size="small" items={[{ key: 'technical', label: t('experimentRecords.technical-details'), children: <>
                                            <div className="mb-3 flex flex-wrap items-center gap-2">
                                                <Typography.Text type="secondary">{t('experimentRecords.content-verification')}:</Typography.Text>
                                                <StatusTag status={asset.integrity} />
                                            </div>
                                            <Typography.Paragraph className="break-all" copyable>{asset.fingerprint || t('experimentRecords.content-unverified')}</Typography.Paragraph>
                                            <JsonView value={{ location: asset.location, metadata: asset.metadata,
                                                relationships: data.links.filter(link => link.assetVersionId === asset.id), manifest: asset.manifest }} />
                                        </> }]} />
                                    </article>;
                                }) : group.key === 'output' && observedOutputs.length ? null
                                    : <div className="experiment-assets-empty">{t(`experimentRecords.no-${group.key}-assets`)}</div>}
                            </div>
                        </section>)}
                    </div> : <Empty description={t('experimentRecords.no-asset-versions-registered')} /> },
                    { key: 'evaluation', label: t('experimentRecords.evaluation'), children: evaluationData?.evaluations.length ? evaluationData.evaluations.map(result => {
                        const related = data.linkedEvaluations.filter(item => item.result.id === result.id);
                        const source = related[0];
                        return <div key={result.id}>
                        {source && <Space wrap className="mb-2">
                            <Typography.Text type="secondary">{t('experimentRecords.evaluation-source')}:</Typography.Text>
                            <Button type="link" onClick={() => setSelectedId(source.sourceExperimentId)}>{source.sourceExperimentName}</Button>
                            <StatusTag status={source.evaluationAttempt.status} />
                            <Typography.Text type="secondary">{t('experimentRecords.related-training-executions')}:</Typography.Text>
                            {related.map(item => <Typography.Text key={item.trainingAttemptId} copyable={{ text: item.trainingAttemptId }}
                                title={item.trainingAttemptId}>#{data.attempts.find(attempt => attempt.id === item.trainingAttemptId)?.attemptNumber}
                                {' · '}{item.trainingAttemptId.slice(0, 16)}…</Typography.Text>)}
                        </Space>}
                        <ExperimentEvaluation data={evaluationData} result={result}>
                            <Typography.Paragraph className="break-all" copyable>{result.id}</Typography.Paragraph>
                            <JsonView value={{ modelVersionId: result.modelVersionId, protocol: result.protocol,
                                metrics: evaluationData.metrics.filter(metric => metric.resultId === result.id), reportVersionId: result.reportVersionId }} />
                        </ExperimentEvaluation></div>;
                    }) : <Empty description={t('experimentRecords.no-evaluation-results')} /> },
                ].filter(item => item.key !== 'evaluation' || showEvaluation)} />
            </>}
        </Drawer>
        <Drawer open={comparing} onClose={() => setComparing(false)} width="min(100vw, 1400px)" title={t('experimentRecords.compare-experiments')}>
            <Space wrap className="mb-3"><Button size="small" onClick={() => comparison.refetch()} loading={comparison.isFetching}>{t('experimentRecords.refresh')}</Button>
                <Typography.Text type="secondary">{t('experimentRecords.last-refreshed')}: {comparison.dataUpdatedAt ? formatTime(new Date(comparison.dataUpdatedAt).toISOString()) : '—'}</Typography.Text>
            </Space>
            {comparison.error && <Alert type="error" message={comparison.error.message} />}
            {comparison.isLoading && <Spin />}
            {comparison.data && <ExperimentComparison data={comparison.data} />}
        </Drawer>
    </section></ConfigProvider>;
}
