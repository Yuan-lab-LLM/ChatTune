import { useEffect, useState } from 'react';
import { trpc } from '@/api/trpc';
import type { ExperimentListFilters } from '@shared/experiment';
import { experimentQueryOptions, experimentRefreshInterval } from './queryPolicy';

export function useExperimentVisibility() {
    const [visible, setVisible] = useState(() => document.visibilityState !== 'hidden');
    useEffect(() => {
        const update = () => setVisible(document.visibilityState !== 'hidden');
        document.addEventListener('visibilitychange', update);
        return () => document.removeEventListener('visibilitychange', update);
    }, []);
    return visible;
}

export function useExperimentQueries(input: { selectedId?: string; tab: string; comparing: boolean;
    selectedKeys: string[]; filters: ExperimentListFilters; offset: number }) {
    const { selectedId, tab, comparing, selectedKeys, filters, offset } = input;
    const visible = useExperimentVisibility();
    const enabled = visible && Boolean(selectedId);
    const identity = { experimentId: selectedId || '' };
    const list = trpc.listExperiments.useQuery({ ...filters, offset, limit: 30 }, {
        ...experimentQueryOptions, enabled: visible,
        refetchInterval: query => visible ? experimentRefreshInterval(query.state.data?.items || []) : false,
    });
    const summary = trpc.getExperimentSummary.useQuery(identity, {
        ...experimentQueryOptions, enabled,
        refetchInterval: query => enabled ? experimentRefreshInterval(query.state.data?.attempts || []) : false,
    });
    const revision = summary.data?.attempts.map(attempt => `${attempt.id}:${attempt.lastSequence}`).join('|') || '';
    const configuration = trpc.getExperimentConfiguration.useQuery({ ...identity, revision }, {
        ...experimentQueryOptions, enabled: enabled && ['configuration', 'executions'].includes(tab),
    });
    const assets = trpc.getExperimentAssets.useQuery({ ...identity, revision }, {
        ...experimentQueryOptions, enabled: enabled && tab === 'assets',
    });
    const evaluations = trpc.getExperimentEvaluations.useQuery(identity, {
        ...experimentQueryOptions, enabled: enabled && ['overview', 'evaluation'].includes(tab),
        refetchInterval: query => enabled && ['overview', 'evaluation'].includes(tab)
            ? experimentRefreshInterval([...(summary.data?.attempts || []), ...(query.state.data?.attempts || []),
                ...(query.state.data?.linkedEvaluations.map(item => item.evaluationAttempt) || [])]) : false,
    });
    const comparison = trpc.compareExperiments.useQuery({ experimentIds: selectedKeys, lightweight: true }, {
        ...experimentQueryOptions, enabled: visible && comparing && selectedKeys.length >= 2,
        refetchInterval: query => visible && comparing ? experimentRefreshInterval([
            ...(query.state.data?.details.flatMap(detail => detail.attempts) || []),
            ...(query.state.data?.linkedEvaluations.map(item => item.evaluationAttempt) || []),
        ]) : false,
    });
    // A freshly opened asset tab rechecks relations created in other experiments.
    useEffect(() => {
        if (enabled && tab === 'assets' && assets.data && !assets.isFetching) void assets.refetch();
    }, [enabled, selectedId, tab]); // eslint-disable-line react-hooks/exhaustive-deps

    const base = summary.data;
    const config = ['configuration', 'executions'].includes(tab) ? configuration.data : undefined;
    const assetData = tab === 'assets' ? assets.data : undefined;
    const evalData = ['overview', 'evaluation'].includes(tab) ? evaluations.data : undefined;
    const data = base && {
        ...base,
        experiment: { ...base.experiment, spec: config?.spec || base.experiment.spec },
        tasks: config?.tasks || base.tasks,
        attempts: (config?.attempts || base.attempts).map(attempt => ({ ...attempt,
            runtime: { ...attempt.runtime, ...evalData?.attempts.find(item => item.id === attempt.id)?.runtime,
                ...(assetData?.observations.find(item => item.id === attempt.id)
                ? { evaluationObservation: assetData.observations.find(item => item.id === attempt.id)!.evaluationObservation } : {}) } })),
        assets: assetData?.assets || evalData?.assets || base.assets,
        links: assetData?.links || base.links,
        lineage: assetData?.lineage || base.lineage,
        evaluations: evalData?.evaluations || base.evaluations,
        metrics: evalData?.metrics || base.metrics,
        linkedEvaluations: evalData?.linkedEvaluations || base.linkedEvaluations,
    };
    const currentSection = tab === 'assets' ? assets : ['configuration', 'executions'].includes(tab) ? configuration
        : ['overview', 'evaluation'].includes(tab) ? evaluations : summary;
    const refresh = async () => {
        await summary.refetch();
        if (currentSection !== summary) await currentSection.refetch();
    };
    return { list, comparison, data, visible,
        detail: { error: summary.error || currentSection.error, refetch: refresh,
            isFetching: summary.isFetching || currentSection.isFetching,
            isLoading: summary.isLoading || currentSection.isLoading,
            dataUpdatedAt: currentSection.dataUpdatedAt || summary.dataUpdatedAt } };
}
