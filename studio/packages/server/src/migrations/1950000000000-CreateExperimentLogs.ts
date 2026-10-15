import { MigrationInterface, QueryRunner, Table, TableIndex } from 'typeorm';

export class CreateExperimentLogs1950000000000 implements MigrationInterface {
    name = 'CreateExperimentLogs1950000000000';
    async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.createTable(new Table({ name: 'experiment_log_source', columns: [
            { name: 'id', type: 'varchar', isPrimary: true },
            ...['attemptId', 'nodeId', 'label', 'kind', 'status', 'createdAt', 'updatedAt'].map(name => ({ name, type: 'varchar' })),
            { name: 'error', type: 'text', isNullable: true },
            { name: 'nextOffset', type: 'integer', default: 0 },
        ] }), true);
        await queryRunner.createIndex('experiment_log_source', new TableIndex({
            name: 'IDX_experiment_log_source_attempt', columnNames: ['attemptId'],
        }));
        await queryRunner.createTable(new Table({ name: 'experiment_log_chunk', columns: [
            { name: 'id', type: 'varchar', isPrimary: true },
            { name: 'sourceId', type: 'varchar' }, { name: 'attemptId', type: 'varchar' },
            { name: 'startOffset', type: 'integer' }, { name: 'endOffset', type: 'integer' },
            { name: 'contentHash', type: 'varchar' }, { name: 'content', type: 'text' },
            { name: 'createdAt', type: 'varchar' },
        ] }), true);
        await queryRunner.createIndex('experiment_log_chunk', new TableIndex({
            name: 'IDX_experiment_log_chunk_offset', columnNames: ['sourceId', 'startOffset'], isUnique: true,
        }));
    }
    async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.dropTable('experiment_log_chunk', true);
        await queryRunner.dropTable('experiment_log_source', true);
    }
}
