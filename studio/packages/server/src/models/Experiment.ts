import { BaseEntity, Column, Entity, Index, PrimaryColumn } from 'typeorm';

export abstract class ExperimentRecord extends BaseEntity {
    @PrimaryColumn() id: string;
    @Column() createdAt: string;
}

@Entity('experiment')
@Index('IDX_experiment_owner_created', ['ownerUserId', 'createdAt'])
export class ExperimentTable extends ExperimentRecord {
    @Column() name: string;
    @Column({ type: 'text', default: '' }) notes: string;
    @Column({ type: 'simple-json', default: '[]' }) tags: string[];
    @Column() project: string;
    @Column() ownerUserId: string;
    @Column({ type: 'text', nullable: true }) groupId: string | null;
    @Column({ type: 'text', nullable: true }) conversationRunId: string | null;
    @Column({ type: 'text', nullable: true }) workflowId: string | null;
    @Column({ type: 'text', nullable: true }) parentExperimentId: string | null;
    @Column({ default: 'draft' }) status: string;
    @Column({ type: 'simple-json' }) spec: Record<string, unknown>;
    @Column() updatedAt: string;
}

@Entity('experiment_task')
@Index('IDX_experiment_task_experiment', ['experimentId'])
export class ExperimentTaskTable extends ExperimentRecord {
    @Column() experimentId: string;
    @Column() kind: string;
    @Column() status: string;
    @Column({ type: 'simple-json' }) spec: Record<string, unknown>;
    @Column() updatedAt: string;
}

@Entity('task_attempt')
@Index('IDX_attempt_task_number', ['taskId', 'attemptNumber'], { unique: true })
@Index('IDX_attempt_node_request', ['nodeId', 'requestKey'], { unique: true })
export class TaskAttemptTable extends ExperimentRecord {
    @Column() taskId: string;
    @Column() nodeId: string;
    @Column() container: string;
    @Column() requestKey: string;
    @Column() requestHash: string;
    @Column({ type: 'integer' }) attemptNumber: number;
    @Column() status: string;
    @Column({ type: 'integer', default: 0 }) lastSequence: number;
    @Column({ type: 'text', nullable: true }) resumeFromVersionId: string | null;
    @Column({ type: 'text', nullable: true }) resourceReservationId: string | null;
    @Column({ type: 'simple-json' }) configuration: Record<string, unknown>;
    @Column({ type: 'simple-json' }) environment: Record<string, unknown>;
    @Column({ type: 'simple-json' }) runtime: Record<string, unknown>;
    @Column({ type: 'text', nullable: true }) error: string | null;
    @Column({ type: 'text', nullable: true }) startedAt: string | null;
    @Column({ type: 'text', nullable: true }) endedAt: string | null;
    @Column() updatedAt: string;
}

@Entity('asset_version')
@Index('IDX_asset_owner_fingerprint', ['ownerUserId', 'fingerprint'])
@Index('IDX_asset_owner_node', ['ownerUserId', 'nodeId'])
export class AssetVersionTable extends ExperimentRecord {
    @Column() ownerUserId: string;
    @Column() nodeId: string;
    @Column({ type: 'text', nullable: true }) resourceId: string | null;
    @Column() kind: string;
    @Column() logicalName: string;
    @Column({ type: 'text', nullable: true }) fingerprint: string | null;
    @Column() integrity: string;
    @Column({ type: 'simple-json' }) location: { container: string; path: string };
    @Column({ type: 'simple-json' }) manifest: Record<string, unknown>;
    @Column({ type: 'simple-json' }) metadata: Record<string, unknown>;
}

@Entity('task_asset')
@Index('IDX_task_asset_relation', ['attemptId', 'assetVersionId', 'direction', 'role'], { unique: true })
@Index('IDX_task_asset_version', ['assetVersionId'])
@Index('IDX_task_asset_task', ['taskId'])
export class TaskAssetTable extends ExperimentRecord {
    @Column() taskId: string;
    @Column() attemptId: string;
    @Column() assetVersionId: string;
    @Column() direction: string;
    @Column() role: string;
    @Column({ type: 'simple-json' }) sourcePaths: string[];
}

@Entity('evaluation_result')
@Index('IDX_evaluation_task', ['taskId'])
@Index('IDX_evaluation_model', ['modelVersionId'])
@Index('IDX_evaluation_attempt', ['attemptId'])
export class EvaluationResultTable extends ExperimentRecord {
    @Column() taskId: string;
    @Column() attemptId: string;
    @Column() modelVersionId: string;
    @Column({ type: 'text', nullable: true }) reportVersionId: string | null;
    @Column({ type: 'simple-json' }) protocol: Record<string, unknown>;
    @Column() protocolComplete: boolean;
    @Column() comparisonKey: string;
    @Column({ type: 'integer', nullable: true }) expectedCount: number | null;
    @Column({ type: 'integer', nullable: true }) succeededCount: number | null;
    @Column({ type: 'integer', nullable: true }) failedCount: number | null;
}

@Entity('evaluation_metric')
@Index('IDX_metric_result_key', ['resultId', 'datasetVersionId', 'subset', 'name'], { unique: true })
export class EvaluationMetricTable extends ExperimentRecord {
    @Column() resultId: string;
    @Column() datasetVersionId: string;
    @Column() subset: string;
    @Column() name: string;
    @Column({ type: 'double precision' }) value: number;
    @Column({ type: 'integer', nullable: true }) sampleCount: number | null;
    @Column() unit: string;
    @Column() direction: string;
}

@Entity('experiment_event')
@Index('IDX_experiment_event_sequence', ['attemptId', 'sequence'], { unique: true })
export class ExperimentEventTable extends ExperimentRecord {
    @Column() attemptId: string;
    @Column() nodeId: string;
    @Column({ type: 'integer' }) sequence: number;
    @Column() type: string;
    @Column() occurredAt: string;
    @Column() payloadHash: string;
    @Column({ type: 'simple-json' }) payload: Record<string, unknown>;
}

@Entity('experiment_log_source')
@Index('IDX_experiment_log_source_attempt', ['attemptId'])
export class ExperimentLogSourceTable extends ExperimentRecord {
    @Column() attemptId: string;
    @Column() nodeId: string;
    @Column() label: string;
    @Column() kind: string;
    @Column() status: string;
    @Column({ type: 'text', nullable: true }) error: string | null;
    @Column({ type: 'integer', default: 0 }) nextOffset: number;
    @Column() updatedAt: string;
}

@Entity('experiment_log_chunk')
@Index('IDX_experiment_log_chunk_offset', ['sourceId', 'startOffset'], { unique: true })
export class ExperimentLogChunkTable extends ExperimentRecord {
    @Column() sourceId: string;
    @Column() attemptId: string;
    @Column({ type: 'integer' }) startOffset: number;
    @Column({ type: 'integer' }) endOffset: number;
    @Column() contentHash: string;
    @Column({ type: 'text' }) content: string;
}

export const experimentEntities = [ExperimentTable, ExperimentTaskTable, TaskAttemptTable,
    AssetVersionTable, TaskAssetTable, EvaluationResultTable, EvaluationMetricTable, ExperimentEventTable,
    ExperimentLogSourceTable, ExperimentLogChunkTable];
