/** Model inventory stores the parent directory in path and its child in name. */
export function modelExperimentPath(model: { path?: string; name?: string }): string | undefined {
    if (!model.path || !model.name) return undefined;
    return `${model.path.replace(/\/+$/, '')}/${model.name}`;
}
