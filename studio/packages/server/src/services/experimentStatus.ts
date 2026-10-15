type TaskStatus = { status: string };
const terminal = new Set(['succeeded', 'failed', 'cancelled']);

/** A workflow can still have stages which have not created their tasks yet. */
export function experimentStatus(tasks: TaskStatus[], workflowId: string | null, workflowStatus?: string): string {
    if (workflowId && workflowStatus) {
        const statuses: Record<string, string> = {
            running: 'running', stopping: 'running', finished: 'succeeded', failed: 'failed', stopped: 'cancelled',
        };
        return statuses[workflowStatus] ?? 'unknown';
    }
    if (tasks.some(t => t.status === 'unknown')) return 'unknown';
    if (tasks.some(t => t.status === 'running' || t.status === 'starting')) return 'running';
    if (tasks.some(t => !terminal.has(t.status)) || !tasks.length) return 'prepared';
    if (tasks.some(t => t.status === 'failed')) return 'failed';
    if (tasks.some(t => t.status === 'cancelled')) return 'cancelled';
    // Old runtimes cannot confirm completion of the whole workflow.
    return workflowId ? 'unknown' : 'succeeded';
}
