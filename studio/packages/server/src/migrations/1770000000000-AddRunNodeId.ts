import { MigrationInterface, QueryRunner } from 'typeorm';

export class AddRunNodeId1770000000000 implements MigrationInterface {
    name = 'AddRunNodeId1770000000000';

    async up(queryRunner: QueryRunner): Promise<void> {
        const tableExists = await queryRunner.hasTable('run_table');
        if (!tableExists) {
            console.log(
                'run_table does not exist. Skipping migration (first-time installation).',
            );
            console.log('TypeORM will create run_table from entity definitions.');
            return;
        }

        const hasColumn = await queryRunner.hasColumn('run_table', 'nodeId');
        if (!hasColumn) {
            // SQLite 原生 ALTER TABLE ADD COLUMN 不需要重建表，避免 view 依赖问题
            await queryRunner.query(
                `ALTER TABLE run_table ADD COLUMN nodeId VARCHAR DEFAULT 'unknown' NULL`,
            );
        }
        await queryRunner.query(
            `CREATE INDEX IF NOT EXISTS IDX_run_table_node_id ON run_table(nodeId)`,
        );
    }

    async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query(
            `DROP INDEX IF EXISTS IDX_run_table_node_id`,
        );

        const tableExists = await queryRunner.hasTable('run_table');
        if (!tableExists) {
            return;
        }

        const hasColumn = await queryRunner.hasColumn('run_table', 'nodeId');
        if (hasColumn) {
            await queryRunner.dropColumn('run_table', 'nodeId');
        }
    }
}
