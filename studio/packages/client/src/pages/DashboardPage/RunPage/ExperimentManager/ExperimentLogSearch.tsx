import { useEffect, useRef, useState } from 'react';
import { Alert, Button, Checkbox, Input, Space, Typography } from 'antd';
import type { inferRouterOutputs } from '@trpc/server';
import { useTranslation } from 'react-i18next';
import { trpcClient } from '@/api/trpc';
import type { AppRouter } from '../../../../../../server/src/trpc/router';

type Page = inferRouterOutputs<AppRouter>['searchExperimentLogs'];
type Context = inferRouterOutputs<AppRouter>['getExperimentLogContext'];
export default function ExperimentLogSearch({ experimentId, attemptId, sourceId, active, onNavigate }: {
    experimentId: string; attemptId: string; sourceId: string; active: boolean; onNavigate: (context: Context) => void;
}) {
    const { t } = useTranslation();
    const [keyword, setKeyword] = useState('');
    const [caseSensitive, setCaseSensitive] = useState(false);
    const [page, setPage] = useState<Page>();
    const [index, setIndex] = useState(-1);
    const [busy, setBusy] = useState(false);
    const [error, setError] = useState('');
    const [searchInput, setSearchInput] = useState<{ keyword: string; caseSensitive: boolean; suspectedErrors: boolean }>();
    const controller = useRef<AbortController | null>(null);
    const generation = useRef(0);
    const identity = { experimentId, attemptId, sourceId };
    const key = `${experimentId}:${attemptId}:${sourceId}`;
    const identityRef = useRef(key);
    identityRef.current = key;
    useEffect(() => {
        setPage(undefined); setIndex(-1); setSearchInput(undefined); setError('');
        generation.current++; controller.current?.abort(); setBusy(false);
    }, [key]);
    useEffect(() => {
        if (!active) { generation.current++; controller.current?.abort(); setBusy(false); }
        return () => { generation.current++; controller.current?.abort(); };
    }, [active, key]);
    const startRequest = () => {
        controller.current?.abort();
        const request = new AbortController(); controller.current = request;
        const version = ++generation.current;
        setBusy(true); setError('');
        return { request, current: () => generation.current === version && identityRef.current === key && !request.signal.aborted };
    };
    const navigate = async (target: number) => {
        const match = page?.matches[target];
        if (!match || !page) return;
        const pending = startRequest();
        try {
            const context = await trpcClient.getExperimentLogContext.query({ ...identity, position: match.position,
                matchLength: match.length, snapshotEndOffset: page.snapshotEndOffset }, { signal: pending.request.signal });
            if (pending.current()) { setIndex(target); onNavigate(context); }
        } catch (failure) {
            if (pending.current()) setError(failure instanceof Error ? failure.message : String(failure));
        } finally { if (pending.current()) setBusy(false); }
    };
    const search = async (suspectedErrors = false, more = false) => {
        const terms = more ? searchInput : { keyword: keyword || 'ERROR', caseSensitive, suspectedErrors };
        if (!terms || (!suspectedErrors && !more && !keyword.trim())) return;
        if (!more) { setPage(undefined); setIndex(-1); setSearchInput(terms); }
        const pending = startRequest();
        try {
            const result = await trpcClient.searchExperimentLogs.query({ ...identity, ...terms,
                ...(more && page ? { cursor: page.nextCursor || undefined, snapshotEndOffset: page.snapshotEndOffset } : {}) },
            { signal: pending.request.signal });
            if (pending.current()) setPage(previous => ({ ...result,
                matches: more ? [...(previous?.matches || []), ...result.matches] : result.matches }));
        } catch (failure) {
            if (pending.current()) setError(failure instanceof Error ? failure.message : String(failure));
        } finally { if (pending.current()) setBusy(false); }
    };
    return <div className="space-y-2">
        <Space wrap>
            <Input.Search style={{ width: 260 }} value={keyword} maxLength={200} allowClear
                placeholder={t('experimentRecords.search-archived-logs')} aria-label={t('experimentRecords.search-archived-logs')}
                onChange={event => setKeyword(event.target.value)} onSearch={() => void search()} loading={busy} disabled={!active} />
            <Checkbox checked={caseSensitive} onChange={event => setCaseSensitive(event.target.checked)}>{t('experimentRecords.case-sensitive')}</Checkbox>
            <Button size="small" disabled={busy || !active} onClick={() => void search(true)}>{t('experimentRecords.find-suspected-errors')}</Button>
            <Button size="small" disabled={busy || !active || !searchInput} onClick={() => void search(searchInput?.suspectedErrors)}>{t('experimentRecords.search-again')}</Button>
            <Button size="small" disabled={!active} href={`/api/experiments/${encodeURIComponent(experimentId)}/attempts/${encodeURIComponent(attemptId)}/logs/${encodeURIComponent(sourceId)}/download`}
                target="_blank" rel="noopener noreferrer">{t('experimentRecords.download-redacted-log')}</Button>
        </Space>
        <Typography.Paragraph type="secondary" className="!mb-0">{t('experimentRecords.log-search-help')}</Typography.Paragraph>
        {error && <Alert type="error" message={error} />}
        {page && <>
            <Space wrap><Typography.Text>{t('experimentRecords.log-match-count', { count: page.matches.length })}</Typography.Text>
                <Button size="small" disabled={busy || index <= 0} onClick={() => void navigate(index - 1)}>{t('experimentRecords.previous-match')}</Button>
                <Button size="small" disabled={busy || index + 1 >= page.matches.length} onClick={() => void navigate(index + 1)}>{t('experimentRecords.next-match')}</Button>
                {page.nextCursor && <Button size="small" loading={busy} onClick={() => void search(false, true)}>{t('experimentRecords.search-more')}</Button>}
            </Space>
            <div className="max-h-[220px] overflow-auto rounded border p-2" role="list" aria-label={t('experimentRecords.log-search-results')}>
                {page.matches.map((match, matchIndex) => <button key={`${match.position.startOffset}:${match.position.characterOffset}`}
                    type="button" role="listitem" aria-current={index === matchIndex ? 'true' : undefined} disabled={busy}
                    className="block w-full whitespace-pre-wrap break-all rounded p-1 text-left text-xs hover:bg-muted"
                    onClick={() => void navigate(matchIndex)}>{match.before}<mark>{match.text}</mark>{match.after}</button>)}
                {!page.matches.length && <Typography.Text type="secondary">{t(page.nextCursor ? 'experimentRecords.search-continue' : 'experimentRecords.no-log-matches')}</Typography.Text>}
            </div>
        </>}
    </div>;
}
