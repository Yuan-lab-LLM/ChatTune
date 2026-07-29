import { MigrationInterface, QueryRunner, TableColumn } from 'typeorm';

export class AddResourcePoolEnabled1860000000000 implements MigrationInterface {
    name = 'AddResourcePoolEnabled1860000000000';

    public async up(queryRunner: QueryRunner): Promise<void> {
        if (!(await queryRunner.hasTable('resource_pool'))) return;
        if (!(await queryRunner.hasColumn('resource_pool', 'enabled'))) {
            await queryRunner.addColumn('resource_pool', new TableColumn({
                name: 'enabled', type: 'boolean', default: '1',
            }));
        }
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        if (!(await queryRunner.hasTable('resource_pool'))) return;
        if (await queryRunner.hasColumn('resource_pool', 'enabled')) {
            await queryRunner.dropColumn('resource_pool', 'enabled');
        }
    }
}
