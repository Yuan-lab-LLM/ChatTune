export type SampleStatus = 'correct' | 'incorrect' | 'failed' | 'invalid' | 'unscored';
export type BenchmarkSample = {
    index: number; id: string; subset: string; question: string; expected: string;
    predicted: string; response: string; error: string; status: SampleStatus; metrics: Record<string, number>;
};
export type BenchmarkReport = {
    schemaVersion: 1; total: number; previewLimit: number; truncated: boolean; groupsTruncated?: boolean;
    counts: Record<SampleStatus, number>;
    groups: Array<{ name: string; samples: number; correct: number; scored: number; failed: number }>;
    samples: BenchmarkSample[];
};
export function benchmarkReport(value: unknown): BenchmarkReport | undefined {
    if (!value || typeof value !== 'object') return undefined;
    const report = value as BenchmarkReport;
    return report.schemaVersion === 1 && Array.isArray(report.samples) && Array.isArray(report.groups)
        && report.counts && typeof report.total === 'number' ? report : undefined;
}

export function filterBenchmarkSamples(samples: BenchmarkSample[], search: string, status: string) {
    const query = search.trim().toLocaleLowerCase();
    return samples.filter(item => (status === 'all' || item.status === status) && (!query ||
        [item.id, String(item.index), item.subset, item.question, item.expected, item.predicted, item.response, item.error]
            .some(value => value.toLocaleLowerCase().includes(query))));
}

/** Preserve stored metrics and only change their presentation. */
export function displayMetric(value: number, unit: string) {
    return unit === 'ratio' ? { value: value * 100, unit: '%' } : { value, unit };
}
