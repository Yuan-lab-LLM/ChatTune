type Association = { match: string; direction: string; taskKind?: string; relatedAssetKind?: string; role?: string };
const modelKinds = new Set(['model', 'adapter', 'checkpoint', 'base_model']);

/** Labels describe the relationship, independently of the experiment's title. */
export function associationLabel(relation: Association): string {
    if (relation.match === 'upstream' || relation.match === 'downstream') {
        const family = relation.relatedAssetKind === 'dataset' ? 'dataset'
            : modelKinds.has(relation.relatedAssetKind || '') ? 'model' : null;
        return family ? `${relation.match}-${family}` : `${relation.match}-record`;
    }
    if (relation.direction === 'output') return 'produced-by';
    const uses: Record<string, string> = {
        train: 'used-for-training', preprocess: 'used-for-preprocessing', evaluate: 'used-for-evaluation',
        benchmark: 'used-for-evaluation', merge: 'used-for-merge', publish: 'used-for-publication', deploy: 'used-for-inference',
    };
    return uses[relation.taskKind || ''] || 'used-by';
}

export function associationLabels(relations: Association[]): string[] {
    return [...new Set(relations.map(associationLabel))];
}

/** Interpret legacy processing records without rewriting immutable events. */
export function associationRole(relation: Association): string {
    if (relation.taskKind === 'preprocess' && relation.direction === 'input') {
        if (relation.role === 'evaluated_model' && modelKinds.has(relation.relatedAssetKind || '')) return 'processing_model';
        if (relation.role === 'training_dataset' && relation.relatedAssetKind === 'dataset') return 'source_dataset';
    }
    return relation.role || '';
}
