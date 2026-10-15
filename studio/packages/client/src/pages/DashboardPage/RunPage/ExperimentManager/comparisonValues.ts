export type ParameterSource = 'requested' | 'resolved' | 'actual' | 'environment';
const record = (value: unknown): Record<string, unknown> => value && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown> : {};
export const canonicalValue = (value: unknown): string => {
    if (Array.isArray(value)) return `[${value.map(canonicalValue).join(',')}]`;
    if (value && typeof value === 'object') return `{${Object.keys(value).sort()
        .map(key => `${JSON.stringify(key)}:${canonicalValue(record(value)[key])}`).join(',')}}`;
    return JSON.stringify(value) ?? 'undefined';
};
const runtimeKey = /^(?:working_?directory|work_?dir|cwd|output_?(?:dir|directory|folder|path)|log_?(?:dir|path|file)|experimentDirectory|executionLogPath|attempt_?id|experiment_?id|task_?id|request_?key)$/i;
export function parameterFields(source: ParameterSource, spec: Record<string, unknown>,
    configuration: Record<string, unknown>, environment: Record<string, unknown>) {
    const fields: Record<string, unknown> = {}, runtime: Record<string, unknown> = {};
    const flatten = (value: unknown, path: string, target: Record<string, unknown>) => {
        if (value && typeof value === 'object' && !Array.isArray(value) && Object.keys(value).length) {
            for (const [key, item] of Object.entries(value)) flatten(item, path ? `${path}.${key}` : key,
                runtimeKey.test(key) ? runtime : target);
        } else if (path) target[path] = value;
    };
    const value = source === 'environment' ? environment : source === 'actual' ? configuration.actual : spec[source];
    if (value !== undefined && value !== null) flatten(value, '', fields);
    if (['requested', 'resolved'].includes(source) && spec.script !== undefined) fields.script = spec.script;
    return { fields, runtime };
}
export function metricDelta(value: number, baseline: number | undefined, direction: string, unit: string, percentage: boolean) {
    if (baseline === undefined || !Number.isFinite(value) || !Number.isFinite(baseline)) return null;
    const delta = value - baseline;
    const tolerance = Number.EPSILON * Math.max(1, Math.abs(value), Math.abs(baseline)) * 4;
    const sign = Math.abs(delta) <= tolerance ? 0 : Math.sign(delta);
    return { value: sign === 0 ? 0 : delta * (percentage && unit === 'ratio' ? 100 : 1),
        percentagePoints: percentage && unit === 'ratio',
        trend: sign === 0 ? 'unchanged' : direction === 'neutral' ? 'neutral'
            : (direction === 'higher' && sign > 0) || (direction === 'lower' && sign < 0) ? 'improved' : 'regressed' };
}
export function latestResult<T extends { result: { id: string; createdAt: string } }>(items: T[]): T | undefined {
    return [...items].sort((a, b) => b.result.createdAt.localeCompare(a.result.createdAt)
        || b.result.id.localeCompare(a.result.id))[0];
}
