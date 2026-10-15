import type { TFunction } from 'i18next';

type LabelCategory = 'parameter-labels' | 'operation-labels' | 'metric-labels' | 'unit-labels' | 'subset-labels' | 'stage-labels';

/** Translate known system names while preserving user-defined names and identifiers. */
export function experimentLabel(t: TFunction, category: LabelCategory, value: string): string {
    return t(`experimentRecords.${category}.${value}`, { defaultValue: value });
}

export function parameterLabel(t: TFunction, path: string): string {
    const name = path.replace(/^resolved\.(?:environment|arguments)\./, '').replace(/^resolved\./, '');
    return t(`experimentRecords.parameter-labels.${name.replace(/_/g, '-').toLowerCase()}`, { defaultValue: name });
}

export function experimentName(t: TFunction, experiment: {
    name: string; workflowId?: string | null; spec: Record<string, unknown>;
}): string {
    // Older workflow records were named after the first training script.
    // Preserve custom names and the stored proposal; this is a display label only.
    if (experiment.workflowId && experiment.spec.kind === 'train' && experiment.name === experiment.spec.script) {
        return t('experimentRecords.workflow-record-name', {
            training: experimentLabel(t, 'operation-labels', experiment.name),
        });
    }
    return experiment.name;
}

export function executionLabel(t: TFunction, attempt: {
    taskId?: string; attemptNumber: number; runtime: Record<string, unknown>;
}, tasks: Array<{ id: string; kind: string }> = []): string {
    const kind = tasks.find(task => task.id === attempt.taskId)?.kind || attempt.runtime.kind;
    const execution = `${t('experimentRecords.execution')} #${attempt.attemptNumber}`;
    const deployments = tasks.filter(task => task.kind === 'deploy');
    const deploymentIndex = deployments.findIndex(task => task.id === attempt.taskId);
    const label = kind === 'deploy' && deployments.length > 1 && deploymentIndex >= 0
        ? t('experimentRecords.deployment-number', { number: deploymentIndex + 1 })
        : typeof kind === 'string' && kind ? t(`experimentRecords.kind.${kind}`, { defaultValue: kind }) : '';
    return label ? `${label} · ${execution}` : execution;
}

export function logSourceLabel(t: TFunction, source: { label: string; kind: string }): string {
    const suffix = source.label.match(/ \(\d+\)$/)?.[0] || '';
    const label = source.label.slice(0, source.label.length - suffix.length);
    const knownLabels: Record<string, string> = {
        '执行日志': 'log-execution-output', 'Execution output': 'log-execution-output',
        '训练日志': 'log-training-output', 'Training log': 'log-training-output',
        '评测日志': 'log-evaluation-output', 'Evaluation log': 'log-evaluation-output',
        '数据处理日志': 'log-data-processing', '数据统计日志': 'log-data-analysis',
    };
    const key = Object.prototype.hasOwnProperty.call(knownLabels, label) ? knownLabels[label] : undefined;
    return (key ? t(`experimentRecords.${key}`) : label) + suffix;
}
