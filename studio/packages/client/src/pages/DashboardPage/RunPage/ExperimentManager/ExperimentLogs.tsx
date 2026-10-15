import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { Alert, Button, Empty, Select, Space, Spin, Tag, Typography } from 'antd';
import { useTranslation } from 'react-i18next';
import { trpc } from '@/api/trpc';
import { mergeLogWindow, LOG_WINDOW_CHUNKS, type LogChunk as Chunk } from './logWindow';
import { executionLabel, logSourceLabel } from './displayLabels';
import ExperimentLogSearch from './ExperimentLogSearch';
import { experimentQueryOptions } from './queryPolicy';

type Attempt = { id: string; taskId?: string; attemptNumber: number; status: string; error?: string | null; runtime: Record<string, unknown> };

export default function ExperimentLogs({ experimentId, attempts, tasks = [], active = true }: {
    experimentId: string; attempts: Attempt[]; tasks?: Array<{ id: string; kind: string }>; active?: boolean;
}) {
    const { t } = useTranslation();
    const utils = trpc.useUtils();
    const [attemptId, setAttemptId] = useState(attempts[attempts.length - 1]?.id || '');
    const [sourceId, setSourceId] = useState('');
    const [sourceSelectedByUser, setSourceSelectedByUser] = useState(false);
    const [chunks, setChunks] = useState<Chunk[]>([]);
    const [loadingOlder, setLoadingOlder] = useState(false);
    const [loadingNew, setLoadingNew] = useState(false);
    const [followingLatest, setFollowingLatest] = useState(true);
    const [readError, setReadError] = useState('');
    const [highlight, setHighlight] = useState<{ startOffset: number; characterOffset: number; length: number } | null>(null);
    const [refreshedAt, setRefreshedAt] = useState<number | null>(null);
    const cursorRef = useRef<number | null>(null);
    const logRef = useRef<HTMLPreElement>(null);
    const loadingOlderRef = useRef(false);
    const loadingNewRef = useRef(false);
    const windowGenerationRef = useRef(0);
    const followTailRef = useRef(true);
    const restoreScrollRef = useRef<{ selection: string; offset: string; relativeTop: number } | null>(null);
    const selectionKey = `${experimentId}:${attemptId}:${sourceId}`;
    const selectionKeyRef = useRef(selectionKey);
    selectionKeyRef.current = selectionKey;
    const currentAttempt = attempts.find(item => item.id === attemptId);
    const attemptFinished = ['succeeded', 'failed', 'cancelled'].includes(currentAttempt?.status || '');
    const executionExited = attemptFinished || typeof currentAttempt?.runtime.exitCode === 'number';
    const sources = trpc.listExperimentLogSources.useQuery({ experimentId, attemptId }, {
        ...experimentQueryOptions,
        staleTime: 0,
        enabled: active && Boolean(attemptId) && currentAttempt?.runtime.logCaptureVersion === 1,
        select: data => data.filter(item => item.kind !== 'execution' && item.kind !== 'operation'),
        refetchInterval: query => {
            const data = query.state.data as Array<{ status: string }> | undefined;
            if (!active || (executionExited && data?.length && data.every(item => item.status === 'complete'))) return false;
            return executionExited && !data?.length ? 60000 : 3000;
        },
    });
    const latest = trpc.getExperimentLogChunks.useQuery({ experimentId, attemptId, sourceId }, {
        ...experimentQueryOptions,
        staleTime: 0,
        enabled: active && Boolean(attemptId && sourceId),
        refetchOnWindowFocus: false,
    });
    const selected = sources.data?.find(item => item.id === sourceId);
    const hasSelected = Boolean(selected);
    const sourceStreaming = selected?.status === 'streaming';
    const completedOffset = sourceStreaming ? null : selected?.nextOffset;
    const sourceNextOffsetRef = useRef(0);
    sourceNextOffsetRef.current = selected?.nextOffset ?? 0;
    const merge = useCallback((incoming: Chunk[], direction: 'older' | 'newer' = 'newer') => {
        const log = logRef.current;
        if (incoming.length && log && !followTailRef.current) {
            const top = log.getBoundingClientRect().top;
            const anchor = Array.from(log.children).find(child => child.getBoundingClientRect().bottom > top) as HTMLElement | undefined;
            if (anchor?.dataset.offset) restoreScrollRef.current = { selection: selectionKeyRef.current,
                offset: anchor.dataset.offset, relativeTop: anchor.getBoundingClientRect().top - top };
        }
        setChunks(previous => mergeLogWindow(previous, incoming, direction));
    }, []);
    useLayoutEffect(() => {
        const log = logRef.current;
        if (!log) return;
        const previousPosition = restoreScrollRef.current;
        if (previousPosition?.selection === selectionKey) {
            const anchor = Array.from(log.children).find(child => (child as HTMLElement).dataset.offset === previousPosition.offset);
            if (anchor) log.scrollTop += anchor.getBoundingClientRect().top - log.getBoundingClientRect().top - previousPosition.relativeTop;
            restoreScrollRef.current = null;
        } else if (followTailRef.current) {
            log.scrollTop = log.scrollHeight;
        }
    }, [chunks, selectionKey]);
    useEffect(() => {
        setAttemptId(attempts[attempts.length - 1]?.id || '');
        setSourceId('');
        setSourceSelectedByUser(false);
        setChunks([]);
    }, [experimentId]);
    useEffect(() => {
        const available = sources.data || [];
        if (!available.length) {
            setSourceId('');
            return;
        }
        const priorities = ['training', 'training_phase', 'deployment', 'service', 'data_processing', 'evaluation_phase', 'evaluation_result'];
        const priority = (kind: string) => {
            const index = priorities.indexOf(kind);
            return index < 0 ? priorities.length : index;
        };
        const preferred = priorities.map(kind => available.find(item => item.kind === kind)).find(Boolean) || available[0];
        const selected = available.find(item => item.id === sourceId);
        if (!selected || (!sourceSelectedByUser && priority(preferred.kind) < priority(selected.kind))) {
            setSourceId(preferred.id);
        }
    }, [sources.data, sourceId, sourceSelectedByUser]);
    useEffect(() => {
        setChunks([]);
        windowGenerationRef.current++;
        cursorRef.current = null;
        loadingOlderRef.current = false;
        loadingNewRef.current = false;
        followTailRef.current = true;
        restoreScrollRef.current = null;
        setFollowingLatest(true);
        setLoadingOlder(false);
        setLoadingNew(false);
        setReadError('');
        setHighlight(null); setRefreshedAt(null);
    }, [attemptId, sourceId]);
    useEffect(() => {
        if (!latest.data || cursorRef.current !== null) return;
        merge(latest.data.chunks);
        setRefreshedAt(Date.now());
        cursorRef.current ??= latest.data.chunks[latest.data.chunks.length - 1]?.endOffset ?? latest.data.latestOffset;
    }, [latest.data, merge]);
    useEffect(() => {
        if (!active || !followingLatest || !attemptId || !sourceId || !latest.data || !hasSelected ||
            (!sourceStreaming && (cursorRef.current ?? 0) >= sourceNextOffsetRef.current)) return;
        let polling = true;
        let busy = false;
        let timer: number;
        const requestedSelection = selectionKey;
        const generation = windowGenerationRef.current;
        const poll = async () => {
            if (!polling || busy || !followTailRef.current || selectionKeyRef.current !== requestedSelection) return;
            busy = true;
            setLoadingNew(true);
            try {
                for (let page = 0; page < 8; page++) {
                    const cursor = cursorRef.current ?? 0;
                    const result = await utils.getExperimentLogChunks.fetch({ experimentId, attemptId, sourceId,
                        afterOffset: cursor });
                    if (!polling || !followTailRef.current || generation !== windowGenerationRef.current || selectionKeyRef.current !== requestedSelection) return;
                    if (result.chunks.length) {
                        merge(result.chunks);
                        cursorRef.current = result.chunks[result.chunks.length - 1].endOffset;
                    }
                    if (result.status !== 'streaming' && (cursorRef.current ?? 0) >= result.latestOffset)
                        window.clearInterval(timer);
                    if (result.chunks.length < 8 || (cursorRef.current ?? 0) >= result.latestOffset) break;
                }
                if (polling && selectionKeyRef.current === requestedSelection) { setReadError(''); setRefreshedAt(Date.now()); }
            } catch (error) {
                if (polling && selectionKeyRef.current === requestedSelection)
                    setReadError(error instanceof Error ? error.message : String(error));
            } finally {
                busy = false;
                if (polling && selectionKeyRef.current === requestedSelection) setLoadingNew(false);
            }
        };
        timer = window.setInterval(poll, 3000);
        return () => { polling = false; window.clearInterval(timer); setLoadingNew(false); };
    }, [active, experimentId, attemptId, sourceId, latest.data, hasSelected, sourceStreaming, completedOffset,
        selectionKey, followingLatest, merge, utils]);
    const loadOlder = useCallback(async (preservePosition = true) => {
        if (loadingOlderRef.current || loadingNewRef.current || !chunks.length || chunks[0].startOffset === 0) return;
        const requestedSelection = selectionKey;
        const generation = windowGenerationRef.current;
        loadingOlderRef.current = true;
        if (preservePosition) { followTailRef.current = false; setFollowingLatest(false); }
        setLoadingOlder(true);
        try {
            const result = await utils.getExperimentLogChunks.fetch({ experimentId, attemptId, sourceId,
                beforeOffset: chunks[0].startOffset });
            if (selectionKeyRef.current !== requestedSelection || generation !== windowGenerationRef.current) return;
            // Scrolling back to the live tail supersedes an outstanding history request.
            if (preservePosition && followTailRef.current) return;
            if (result.chunks.length) {
                merge(result.chunks, 'older');
            }
            setReadError('');
        } catch (error) {
            if (selectionKeyRef.current === requestedSelection && generation === windowGenerationRef.current)
                setReadError(error instanceof Error ? error.message : String(error));
        } finally {
            if (selectionKeyRef.current === requestedSelection && generation === windowGenerationRef.current) {
                loadingOlderRef.current = false;
                setLoadingOlder(false);
            }
        }
    }, [chunks, selectionKey, experimentId, attemptId, sourceId, merge, utils]);
    const loadNewer = useCallback(async () => {
        if (loadingNewRef.current || loadingOlderRef.current || !chunks.length || followTailRef.current) return;
        const endOffset = chunks[chunks.length - 1].endOffset;
        if (endOffset >= sourceNextOffsetRef.current) return;
        const requestedSelection = selectionKey;
        const generation = windowGenerationRef.current;
        loadingNewRef.current = true;
        setLoadingNew(true);
        try {
            const result = await utils.getExperimentLogChunks.fetch({ experimentId, attemptId, sourceId, afterOffset: endOffset });
            if (selectionKeyRef.current !== requestedSelection || generation !== windowGenerationRef.current) return;
            merge(result.chunks);
            setReadError('');
        } catch (error) {
            if (selectionKeyRef.current === requestedSelection && generation === windowGenerationRef.current)
                setReadError(error instanceof Error ? error.message : String(error));
        } finally {
            if (selectionKeyRef.current === requestedSelection && generation === windowGenerationRef.current) {
                loadingNewRef.current = false;
                setLoadingNew(false);
            }
        }
    }, [chunks, selectionKey, experimentId, attemptId, sourceId, merge, utils]);
    useEffect(() => {
        const log = logRef.current;
        if (chunks.length < LOG_WINDOW_CHUNKS && chunks[0]?.startOffset > 0 && log && log.scrollHeight <= log.clientHeight + 160)
            void loadOlder(false);
    }, [chunks, loadOlder]);
    const jumpToLatest = async () => {
        setHighlight(null);
        const requestedSelection = selectionKey;
        const generation = ++windowGenerationRef.current;
        followTailRef.current = false;
        setFollowingLatest(false);
        loadingOlderRef.current = false;
        setLoadingOlder(false);
        loadingNewRef.current = true;
        setLoadingNew(true);
        try {
            const result = await utils.getExperimentLogChunks.fetch({ experimentId, attemptId, sourceId }, { staleTime: 0 });
            if (selectionKeyRef.current !== requestedSelection || generation !== windowGenerationRef.current) return;
            cursorRef.current = result.chunks[result.chunks.length - 1]?.endOffset ?? result.latestOffset;
            restoreScrollRef.current = null;
            followTailRef.current = true;
            setFollowingLatest(true);
            setChunks(mergeLogWindow([], result.chunks, 'newer'));
            setReadError('');
            setRefreshedAt(Date.now());
        } catch (error) {
            if (selectionKeyRef.current === requestedSelection && generation === windowGenerationRef.current)
                setReadError(error instanceof Error ? error.message : String(error));
        } finally {
            if (selectionKeyRef.current === requestedSelection && generation === windowGenerationRef.current) {
                loadingNewRef.current = false;
                setLoadingNew(false);
            }
        }
    };
    const legacyExternalLogs = ['dpo_train_launcher', 'single_model_evaluation_vpn',
        'compare_between_models_vpn', 'ckpt_eval'].includes(String(currentAttempt?.runtime.scriptName || ''));
    if (!attempts.length) return <Empty description={t('experimentRecords.no-data')} />;
    return <div className="flex flex-col gap-3">
        {tasks.length > 1 && <Typography.Text type="secondary">{t('experimentRecords.logs-per-execution')}</Typography.Text>}
        <Space wrap>
            <Select style={{ minWidth: 180 }} value={attemptId} onChange={value => { setAttemptId(value); setSourceId(''); setSourceSelectedByUser(false); }}
                options={attempts.map(item => ({ value: item.id,
                    label: `${executionLabel(t, item, tasks)} · ${t(`experimentRecords.status.${item.status}`, { defaultValue: item.status })}` }))} />
            <Select style={{ minWidth: 220 }} value={sourceId || undefined} onChange={value => { setSourceId(value); setSourceSelectedByUser(true); }}
                disabled={!sources.data?.length}
                placeholder={t('experimentRecords.select-log-source')}
                options={(sources.data || []).map(item => ({ value: item.id, label: logSourceLabel(t, item) }))} />
            {selected && <Tag color={selected.status === 'complete' ? 'green' : selected.status === 'error' ? 'red' : 'blue'}>
                {t(`experimentRecords.log-status.${selected.status}`)}</Tag>}
            <Button size="small" disabled={!active} onClick={() => { void sources.refetch(); if (sourceId) void jumpToLatest(); }}>{t('experimentRecords.refresh')}</Button>
            <Typography.Text type="secondary">{t('experimentRecords.last-refreshed')}: {refreshedAt ? new Date(refreshedAt).toLocaleString() : '—'}</Typography.Text>
        </Space>
        {sourceId && <ExperimentLogSearch experimentId={experimentId} attemptId={attemptId} sourceId={sourceId} active={active}
            onNavigate={context => {
                windowGenerationRef.current++; followTailRef.current = false; setFollowingLatest(false);
                loadingNewRef.current = false; loadingOlderRef.current = false;
                setLoadingNew(false); setLoadingOlder(false); restoreScrollRef.current = null;
                setChunks(context.chunks); setHighlight({ ...context.position, length: context.matchLength });
                setRefreshedAt(Date.now());
                window.requestAnimationFrame(() => {
                    const element = logRef.current?.querySelector('mark');
                    element?.scrollIntoView({ block: 'center' });
                });
            }} />}
        {sources.error && <Alert type="error" message={sources.error.message} />}
        {currentAttempt?.status === 'failed' && currentAttempt.error && <Alert type="error" showIcon
            message={t('experimentRecords.failure-reason')}
            description={<div className="whitespace-pre-wrap break-all">{currentAttempt.error}</div>} />}
        {legacyExternalLogs && <Alert type="warning" showIcon message={t('experimentRecords.legacy-external-logs')} />}
        {(latest.error || readError || selected?.error) && <Alert type="error"
            message={latest.error?.message || readError || selected?.error} />}
        {currentAttempt?.runtime.logCaptureVersion !== 1
            ? <Empty description={t('experimentRecords.legacy-logs-unavailable')} />
            : !sources.data && sources.isLoading ? <Spin /> : !sources.data?.length
            ? <Empty description={<>
                <div>{t(attemptFinished ? 'experimentRecords.no-archived-business-logs' : 'experimentRecords.log-pending')}</div>
                {attemptFinished && legacyExternalLogs && <Typography.Text type="secondary">
                    {t('experimentRecords.historical-business-logs-not-backfilled')}
                </Typography.Text>}
            </>} /> : <>
                {(loadingOlder || !followingLatest) && <Space size="small">
                    {loadingOlder && <Typography.Text type="secondary">{t('experimentRecords.loading-earlier-logs')}</Typography.Text>}
                    {!followingLatest && <Button size="small" loading={loadingNew} onClick={jumpToLatest}>
                        {t('experimentRecords.jump-to-latest-logs')}</Button>}
                </Space>}
                <pre ref={logRef} tabIndex={0} onScroll={event => {
                    const log = event.currentTarget;
                    const loadingEarlier = log.scrollTop < 80 && (chunks[0]?.startOffset ?? 0) > 0;
                    const atWindowBottom = !loadingEarlier && log.scrollHeight - log.scrollTop - log.clientHeight < 48;
                    const endOffset = chunks[chunks.length - 1]?.endOffset ?? 0;
                    const atBottom = atWindowBottom && (followTailRef.current || endOffset >= sourceNextOffsetRef.current);
                    if (followTailRef.current !== atBottom) {
                        followTailRef.current = atBottom;
                        setFollowingLatest(atBottom);
                        if (atBottom) cursorRef.current = endOffset;
                    }
                    if (loadingEarlier) void loadOlder();
                    else if (atWindowBottom && !atBottom) void loadNewer();
                }} className="min-h-[180px] max-h-[600px] overflow-auto rounded-lg bg-muted/50 p-3 text-xs whitespace-pre-wrap break-all"
                    aria-label={t('experimentRecords.logs')}>{chunks.some(chunk => chunk.content.length) ? chunks.map((chunk, index) => {
                        const targetIndex = highlight ? chunks.findIndex(item => item.startOffset === highlight.startOffset) : -1;
                        const distance = targetIndex < 0 || index < targetIndex ? -1 : chunks.slice(targetIndex, index).reduce((sum, item) => sum + item.content.length, 0);
                        const start = highlight && distance >= 0 ? Math.max(0, highlight.characterOffset - distance) : 0;
                        const end = highlight && distance >= 0 ? Math.min(chunk.content.length, highlight.characterOffset + highlight.length - distance) : 0;
                        return <span key={chunk.startOffset} data-offset={chunk.startOffset}>{end > start
                            ? <>{chunk.content.slice(0, start)}<mark>{chunk.content.slice(start, end)}</mark>{chunk.content.slice(end)}</>
                            : chunk.content}</span>;
                    }) : t('experimentRecords.log-pending')}</pre>
                <Typography.Text type="secondary">{t('experimentRecords.log-window-help')}</Typography.Text>
                {!followingLatest && (chunks[chunks.length - 1]?.endOffset ?? 0) < sourceNextOffsetRef.current &&
                    <Button size="small" loading={loadingNew} onClick={loadNewer}>{t('experimentRecords.load-later-logs')}</Button>}
                {chunks[0]?.startOffset > 0 && <Typography.Text type="secondary">
                    {t('experimentRecords.scroll-up-for-earlier-logs')}</Typography.Text>}
                <Typography.Text type="secondary">{loadingNew ? t('experimentRecords.loading-new-logs')
                    : selected?.status === 'complete' ? t('experimentRecords.logs-collection-complete')
                    : selected?.status === 'error' ? t('experimentRecords.log-status.error')
                    : t('experimentRecords.logs-refresh-automatically')}</Typography.Text>
            </>}
    </div>;
}
