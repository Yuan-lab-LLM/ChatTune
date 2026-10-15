import { useState } from 'react';
import { Alert, Button, Empty, Modal, Spin, Tag } from 'antd';
import { useTranslation } from 'react-i18next';
import { trpc } from '@/api/trpc';
import { useStudioSidebar } from '@/context/SidebarContext';
import { associationLabels } from './associationLabels';
import { experimentQueryOptions, experimentRefreshInterval } from './queryPolicy';
import { useExperimentVisibility } from './useExperimentQueries';

export function ExperimentLink({ experimentId }: { experimentId?: string }) {
    const sidebar = useStudioSidebar();
    const { t } = useTranslation();
    if (!experimentId) return null;
    return <Button type="link" size="small" onClick={event => {
        event.stopPropagation();
        sidebar.setSelectedExperimentId(experimentId);
        sidebar.setRunPageSection('experiments');
        sidebar.setRunPagePanelOpen(true);
    }}>{t('experimentRecords.view-experiment')}</Button>;
}

export default function ExperimentSourceButton({ nodeId, container, path, assetVersionId, resourceKind = 'all' }: {
    nodeId?: string; container?: string; path?: string; assetVersionId?: string;
    resourceKind?: 'all' | 'model' | 'dataset' | 'report';
}) {
    const [open, setOpen] = useState(false);
    const visible = useExperimentVisibility();
    const { t } = useTranslation();
    const sidebar = useStudioSidebar();
    const pathQuery = trpc.getPathExperiments.useQuery({ nodeId: nodeId || '', container: container || '', path: path || '', scope: resourceKind },
        { ...experimentQueryOptions, enabled: visible && open && !assetVersionId && Boolean(nodeId && container && path),
            refetchInterval: query => visible && open ? experimentRefreshInterval(query.state.data || []) : false });
    const assetQuery = trpc.getAssetExperiments.useQuery({ assetVersionId: assetVersionId || '', scope: resourceKind },
        { ...experimentQueryOptions, enabled: visible && open && Boolean(assetVersionId),
            refetchInterval: query => visible && open ? experimentRefreshInterval(query.state.data || []) : false });
    const query = assetVersionId ? assetQuery : pathQuery;
    if (!assetVersionId && (!nodeId || !container || !path)) return null;
    return <>
        <Button type="link" size="small" className="!px-2 !text-primary hover:!text-primary/80" onClick={event => { event.stopPropagation(); setOpen(true); }}>{t('experimentRecords.related-experiments')}</Button>
        <Modal open={open} title={t('experimentRecords.related-experiments')} footer={null} onCancel={() => setOpen(false)}>
            {resourceKind !== 'all' && <p className="mb-3 text-sm text-muted-foreground">{t(`experimentRecords.association-scope.${resourceKind}`)}</p>}
            <div className="mb-3 flex justify-end">
                <span className="mr-2 text-xs text-muted-foreground">{t('experimentRecords.last-refreshed')}: {query.dataUpdatedAt ? new Date(query.dataUpdatedAt).toLocaleString() : '—'}</span>
                <Button size="small" loading={query.isFetching} onClick={() => query.refetch()}>{t('experimentRecords.refresh')}</Button>
            </div>
            {query.isLoading ? <Spin /> : query.error ? <Alert type="error" message={query.error.message} /> : query.data?.length ?
                <div className="flex flex-col gap-2">{query.data.map(experiment => <button key={experiment.id} type="button"
                    className="flex w-full items-center justify-between gap-3 rounded-lg border border-border bg-background px-4 py-3 text-left transition-colors hover:border-primary hover:bg-muted/50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-primary"
                    onClick={() => {
                        sidebar.setSelectedExperimentId(experiment.id);
                        sidebar.setRunPageSection('experiments');
                        sidebar.setRunPagePanelOpen(true);
                        setOpen(false);
                    }}>
                    <span className="min-w-0">
                        <span className="block truncate font-medium">{experiment.name}</span>
                        <span className="block text-xs text-muted-foreground">{experiment.spec.operation === 'import' && experiment.spec.script === 'dataset_upload'
                            ? t('experimentRecords.dataset-upload-record') : t('experimentRecords.experiment-record')}</span>
                        <span className="mt-1 flex flex-wrap gap-1">
                            {associationLabels(experiment.assetRelations).map(label => <Tag key={label} className="!m-0">{t(`experimentRecords.${label}`)}</Tag>)}
                            {experiment.assetRelations.some(relation => relation.match === 'content') &&
                                <Tag className="!m-0">{t('experimentRecords.same-asset-content')}</Tag>}
                        </span>
                    </span>
                    <span className="flex shrink-0 items-center gap-2"><Tag className="!m-0" color={experiment.status === 'succeeded' ? 'green' : experiment.status === 'failed' ? 'red' : 'default'}>
                        {t(`experimentRecords.status.${experiment.status}`, { defaultValue: experiment.status })}</Tag>
                        <span className="text-sm font-medium text-primary">{t('experimentRecords.view-details')} →</span></span>
                </button>)}</div> : <Empty description={t('experimentRecords.no-recorded-experiment-lineage')} />}
        </Modal>
    </>;
}
