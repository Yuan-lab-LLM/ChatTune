type EvaluationContext = {
    experiment: { spec: { kind?: unknown } };
    tasks: readonly { kind: string }[];
    evaluations: readonly unknown[];
    linkedEvaluations: readonly unknown[];
};

export function hasEvaluationTab(data: EvaluationContext | undefined): boolean {
    if (!data) return false;
    // Stage takes precedence over tasks/results attached to the same record.
    const kind = String(data.experiment.spec.kind ?? '');
    if (['preprocess', 'merge', 'publish', 'deploy'].includes(kind)) return false;
    return ['evaluate', 'benchmark'].includes(kind)
        || data.tasks.some(task => ['evaluate', 'benchmark'].includes(task.kind))
        || data.evaluations.length > 0 || data.linkedEvaluations.length > 0;
}
