import { MigrationInterface, QueryRunner, TableColumn } from 'typeorm';

export class AddResourcePoolNodeAllowedGpuIndexes1840000000000 implements MigrationInterface {
    name = 'AddResourcePoolNodeAllowedGpuIndexes1840000000000';

    public async up(queryRunner: QueryRunner): Promise<void> {
        if (!(await queryRunner.hasTable('resource_pool_node'))) return;
        if (!(await queryRunner.hasColumn('resource_pool_node', 'allowedGpuIndexes'))) {
            await queryRunner.addColumn('resource_pool_node', new TableColumn({
                name: 'allowedGpuIndexes',
                type: 'text',
                isNullable: true,
            }));
        }
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        if (
            await queryRunner.hasTable('resource_pool_node')
            && await queryRunner.hasColumn('resource_pool_node', 'allowedGpuIndexes')
        ) {
            await queryRunner.dropColumn('resource_pool_node', 'allowedGpuIndexes');
        }
    }
}
