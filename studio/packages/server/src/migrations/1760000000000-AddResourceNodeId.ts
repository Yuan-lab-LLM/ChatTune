import { MigrationInterface, QueryRunner, TableColumn, TableIndex } from 'typeorm';

export class AddResourceNodeId1760000000000 implements MigrationInterface {
    name = 'AddResourceNodeId1760000000000';

    async up(queryRunner: QueryRunner): Promise<void> {
        for (const tableName of ['management_cache', 'refresh_job']) {
            if (!(await queryRunner.hasColumn(tableName, 'nodeId'))) {
                await queryRunner.addColumn(tableName, new TableColumn({
                    name: 'nodeId',
                    type: 'varchar',
                    default: "'local'",
                    isNullable: false,
                }));
            }
        }
        await queryRunner.createIndex('management_cache', new TableIndex({
            name: 'IDX_management_cache_node_id',
            columnNames: ['nodeId'],
        }));
        await queryRunner.createIndex('refresh_job', new TableIndex({
            name: 'IDX_refresh_job_node_id',
            columnNames: ['nodeId'],
        }));
    }

    async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.dropColumn('refresh_job', 'nodeId');
        await queryRunner.dropColumn('management_cache', 'nodeId');
    }
}
