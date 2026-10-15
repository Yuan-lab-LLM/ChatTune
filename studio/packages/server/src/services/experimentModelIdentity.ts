/** Content equivalence preserves each version's immutable location and kind. */
export type ModelIdentity = {
    id: string;
    ownerUserId: string;
    kind: string;
    integrity: string;
    fingerprint: string | null;
    manifest: Record<string, unknown>;
};

export const modelAssetKinds = ['model', 'adapter', 'checkpoint', 'base_model'];

/** Preserve the existing missing-shard warning without returning the full file manifest. */
export function modelShardSummary(manifest: Record<string, unknown>) {
    if (!Array.isArray(manifest.files)) return undefined;
    const names = manifest.files.map(file => file?.path).filter((name): name is string => typeof name === 'string');
    const shards = names.map(name => /^model-(\d+)-of-(\d+)\.safetensors$/.exec(name)).filter((item): item is RegExpExecArray => Boolean(item));
    if (!shards.length) return undefined;
    const expected = Number(shards[0][2]), present = new Set(shards.map(item => Number(item[1]))).size;
    return { expected, present, complete: present === expected && shards.every(item => Number(item[2]) === expected)
        && names.includes('model.safetensors.index.json') };
}

export function assetContentKey(asset: ModelIdentity): string | null {
    const family = modelAssetKinds.includes(asset.kind) ? 'model' : asset.kind === 'dataset' ? 'dataset' : null;
    if (!family || asset.integrity !== 'verified' ||
        !asset.fingerprint || !Array.isArray(asset.manifest.files) || !asset.manifest.files.length) return null;
    const files: Array<[string, number, string]> = [];
    for (const file of asset.manifest.files) {
        if (!file || typeof file.path !== 'string' || typeof file.size !== 'number' ||
            typeof file.sha256 !== 'string') return null;
        files.push([file.path, file.size, file.sha256]);
    }
    return JSON.stringify([asset.ownerUserId, family, asset.fingerprint, files]);
}

export const modelContentKey = (asset: ModelIdentity) => modelAssetKinds.includes(asset.kind) ? assetContentKey(asset) : null;

export function equivalentModelVersions<T extends ModelIdentity>(assets: T[]): Map<string, T[]> {
    const groups = new Map<string, T[]>();
    for (const asset of assets) {
        const key = assetContentKey(asset);
        if (key) groups.set(key, [...(groups.get(key) || []), asset]);
    }
    return new Map(assets.map(asset => {
        const key = assetContentKey(asset);
        return [asset.id, key ? groups.get(key)! : [asset]];
    }));
}
