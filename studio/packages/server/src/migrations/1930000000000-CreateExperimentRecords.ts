import { MigrationInterface, QueryRunner, Table, TableColumnOptions, TableIndexOptions } from 'typeorm';

// Frozen migration definitions. Do not derive historical schemas from live entities.
const column = (name: string, type = 'varchar', nullable = false): TableColumnOptions =>
    ({ name, type, isNullable: nullable });
const strings = (...names: string[]) => names.map(name => column(name));
const nullable = (...names: string[]) => names.map(name => column(name, 'text', true));
const json = (...names: string[]) => names.map(name => column(name, 'text'));
const integers = (...names: string[]) => names.map(name => column(name, 'integer'));
const index = (name: string, columnNames: string[], isUnique = false): TableIndexOptions => ({ name, columnNames, isUnique });
const tables = [
    {
        name: 'experiment', columns: [...strings('name', 'project', 'ownerUserId', 'updatedAt'),
            column('status'), ...nullable('groupId', 'conversationRunId', 'workflowId', 'parentExperimentId'), ...json('spec')],
        indices: [index('IDX_experiment_owner_created', ['ownerUserId', 'createdAt'])],
    },
    {
        name: 'experiment_task', columns: [...strings('experimentId', 'kind', 'status', 'updatedAt'), ...json('spec')],
        indices: [index('IDX_experiment_task_experiment', ['experimentId'])],
    },
    {
        name: 'task_attempt', columns: [...strings('taskId', 'nodeId', 'container', 'requestKey', 'requestHash', 'status', 'updatedAt'),
            ...integers('attemptNumber', 'lastSequence'), ...nullable('resumeFromVersionId', 'error', 'startedAt', 'endedAt'),
            ...json('configuration', 'environment', 'runtime')],
        indices: [index('IDX_attempt_task_number', ['taskId', 'attemptNumber'], true),
            index('IDX_attempt_node_request', ['nodeId', 'requestKey'], true)],
    },
    {
        name: 'asset_version', columns: [...strings('ownerUserId', 'nodeId', 'kind', 'logicalName', 'integrity'),
            ...nullable('resourceId', 'fingerprint'), ...json('location', 'manifest', 'metadata')],
        indices: [index('IDX_asset_owner_fingerprint', ['ownerUserId', 'fingerprint'])],
    },
    {
        name: 'task_asset', columns: [...strings('taskId', 'attemptId', 'assetVersionId', 'direction', 'role'), ...json('sourcePaths')],
        indices: [index('IDX_task_asset_relation', ['attemptId', 'assetVersionId', 'direction', 'role'], true),
            index('IDX_task_asset_version', ['assetVersionId'])],
    },
    {
        name: 'evaluation_result', columns: [...strings('taskId', 'attemptId', 'modelVersionId', 'comparisonKey'),
            ...nullable('reportVersionId'), ...json('protocol'), column('protocolComplete', 'boolean'),
            ...['expectedCount', 'succeededCount', 'failedCount'].map(name => column(name, 'integer', true))],
        indices: [index('IDX_evaluation_task', ['taskId'])],
    },
    {
        name: 'evaluation_metric', columns: [...strings('resultId', 'datasetVersionId', 'subset', 'name', 'unit', 'direction'),
            column('value', 'double precision'), column('sampleCount', 'integer', true)],
        indices: [index('IDX_metric_result_key', ['resultId', 'datasetVersionId', 'subset', 'name'], true)],
    },
    {
        name: 'experiment_event', columns: [...strings('attemptId', 'nodeId', 'type', 'occurredAt', 'payloadHash'),
            ...integers('sequence'), ...json('payload')],
        indices: [index('IDX_experiment_event_sequence', ['attemptId', 'sequence'], true)],
    },
];

export class CreateExperimentRecords1930000000000 implements MigrationInterface {
    name = 'CreateExperimentRecords1930000000000';
    async up(queryRunner: QueryRunner): Promise<void> {
        for (const definition of tables) {
            await queryRunner.createTable(new Table({ ...definition, columns: [
                { name: 'id', type: 'varchar', isPrimary: true }, column('createdAt'), ...definition.columns,
            ] }), true);
        }
    }
    async down(queryRunner: QueryRunner): Promise<void> {
        for (const { name } of [...tables].reverse()) await queryRunner.dropTable(name, true);
    }
}
