import { MigrationInterface, QueryRunner, TableColumn, TableIndex } from 'typeorm';

export class AddDatasetStorageState1920000000000 implements MigrationInterface {
    name = 'AddDatasetStorageState1920000000000';

    public async up(queryRunner: QueryRunner): Promise<void> {
        if (!(await queryRunner.hasTable('resource_catalog'))) return;
        if (!(await queryRunner.hasColumn('resource_catalog', 'storageState'))) {
            await queryRunner.addColumn('resource_catalog', new TableColumn({
                name: 'storageState',
                type: 'varchar',
                isNullable: false,
                default: "'ready'",
            }));
        }
        await queryRunner.query(
            "UPDATE resource_catalog SET storageState = 'ready' WHERE storageState IS NULL OR TRIM(storageState) = ''",
        );
        const table = await queryRunner.getTable('resource_catalog');
        const lookup = table?.indices.find((index) => index.name === 'IDX_resource_catalog_lookup');
        if (lookup?.isUnique) {
            await queryRunner.dropIndex('resource_catalog', lookup);
            await queryRunner.createIndex('resource_catalog', new TableIndex({
                name: 'IDX_resource_catalog_lookup',
                columnNames: ['bizType', 'nodeId', 'containerName', 'itemKey'],
                isUnique: false,
            }));
        }
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        if (await queryRunner.hasColumn('resource_catalog', 'storageState')) {
            await queryRunner.dropColumn('resource_catalog', 'storageState');
        }
    }
}
