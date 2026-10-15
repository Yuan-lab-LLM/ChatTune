type Attempt = { id: string; attemptNumber: number; runtime: Record<string, unknown> };
type Asset = { id: string; metadata: Record<string, unknown>; location: { container: string; path: string } };
type Link = { attemptId: string; assetVersionId: string; direction: string };
export type ObservedOutput = {
    name: string; path: string; status: string; exists: boolean;
    scores: [string, number][]; logPath?: string; logError: boolean; archivedVersionId?: string;
};
const record = (value: unknown): Record<string, unknown> => value && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown> : {};
const text = (value: unknown): string | undefined => typeof value === 'string' && value ? value : undefined;

/** Monitor discoveries are telemetry; only an output link proves version registration. */
export function evaluationOutputs(attempts: Attempt[], assets: Asset[], links: Link[]) {
    return attempts.flatMap(attempt => {
        const observation = record(attempt.runtime.evaluationObservation);
        if (observation.source !== 'medical_monitor' || !Array.isArray(observation.items)) return [];
        const outputIds = new Set(links.filter(link => link.attemptId === attempt.id && link.direction === 'output')
            .map(link => link.assetVersionId));
        const outputs = assets.filter(asset => outputIds.has(asset.id));
        const rows = new Map<string, ObservedOutput>();
        for (const value of observation.items) {
            const item = record(value);
            const path = text(item.path);
            const name = text(item.name);
            if (!path || !name) continue;
            const archived = outputs.find(asset => asset.metadata.sourcePath === path || asset.location.path === path);
            const row: ObservedOutput = {
                name, path, status: text(item.status) || 'unknown', exists: item.exists === true,
                scores: Object.entries(record(item.scores)).filter((entry): entry is [string, number] =>
                    ['average', 'accuracy', 'mean_fuzziness'].includes(entry[0])
                    && typeof entry[1] === 'number' && Number.isFinite(entry[1])),
                logPath: text(item.logPath), logError: item.logError === true, archivedVersionId: archived?.id,
            };
            // A stage and an auxiliary artifact can refer to the same file.
            if (!rows.has(path)) rows.set(path, row);
        }
        return rows.size ? [{ attemptId: attempt.id, attemptNumber: attempt.attemptNumber,
            currentStage: text(observation.currentStage), items: [...rows.values()] }] : [];
    });
}
