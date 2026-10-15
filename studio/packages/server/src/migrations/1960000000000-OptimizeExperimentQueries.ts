import { TableIndex } from 'typeorm';
import type { MigrationInterface, QueryRunner } from 'typeorm';

const indices = [
    { table: 'asset_version', name: 'IDX_asset_owner_node', columnNames: ['ownerUserId', 'nodeId'] },
    { table: 'task_asset', name: 'IDX_task_asset_task', columnNames: ['taskId'] },
    { table: 'evaluation_result', name: 'IDX_evaluation_model', columnNames: ['modelVersionId'] },
    { table: 'evaluation_result', name: 'IDX_evaluation_attempt', columnNames: ['attemptId'] },
];

export class OptimizeExperimentQueries1960000000000 implements MigrationInterface {
    name = 'OptimizeExperimentQueries1960000000000';
    async up(queryRunner: QueryRunner): Promise<void> {
        for (const { table, ...index } of indices) await queryRunner.createIndex(table, new TableIndex(index));
    }
    async down(queryRunner: QueryRunner): Promise<void> {
        for (const { table, name } of [...indices].reverse()) await queryRunner.dropIndex(table, name);
    }
}
