export const experimentQueryOptions = {
    staleTime: 60_000, gcTime: 10 * 60_000, refetchOnWindowFocus: 'always' as const,
    refetchIntervalInBackground: false,
    trpc: { abortOnUnmount: true },
};
type RecordStatus = { status: string; updatedAt?: string; runtime?: Record<string, unknown> };
export function experimentRefreshInterval(records: readonly RecordStatus[], now = Date.now()): number {
    return records.some(item => ['prepared', 'starting', 'running'].includes(item.status)
        || ((item.status === 'unknown' || ['retrying', 'awaiting_outputs', 'pending'].includes(String(item.runtime?.archiveStatus)))
            && (!item.updatedAt || now - Date.parse(item.updatedAt) < 5 * 60_000))) ? 15_000 : 60_000;
}
