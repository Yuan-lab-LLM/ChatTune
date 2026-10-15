const record = (value: unknown): Record<string, unknown> =>
    value !== null && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {};

/** Instance identity comes from the structured service response, never model names or ports. */
export function serviceExperimentReferences(value: unknown) {
    return (Array.isArray(value) ? value : []).flatMap(raw => {
        const item = record(raw);
        if (typeof item.instance_id !== 'string' || !item.instance_id) return [];
        const experimentId = record(item.experiment).experimentId;
        const ports = record(item.ports);
        const port = Number(ports.vllm ?? ports.VLLM_OPENAI_PORT);
        return [{ instanceId: item.instance_id, status: item.status,
            port: Number.isInteger(port) && port > 0 && port <= 65535 ? port : 0,
            model: typeof item.model === 'string' ? item.model : undefined,
            experimentId: typeof experimentId === 'string' && /^[A-Za-z0-9_-]{1,120}$/.test(experimentId)
                ? experimentId : undefined }];
    });
}
