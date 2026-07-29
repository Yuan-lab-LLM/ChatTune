import { MigrationInterface, QueryRunner, Table } from 'typeorm';

export class CreateResourceShareScopeTable1820000000000 implements MigrationInterface {
    name = 'CreateResourceShareScopeTable1820000000000';

    public async up(queryRunner: QueryRunner): Promise<void> {
        if (await queryRunner.hasTable('resource_share_scope')) return;
        await queryRunner.createTable(new Table({
            name: 'resource_share_scope',
            columns: [
                { name: 'id', type: 'varchar', isPrimary: true },
                { name: 'resourceId', type: 'varchar' },
                { name: 'scopeKey', type: 'varchar' },
                { name: 'groupId', type: 'text', isNullable: true },
                { name: 'createdAt', type: 'datetime', default: "datetime('now')" },
            ],
            indices: [{
                name: 'IDX_resource_share_scope_unique',
                columnNames: ['resourceId', 'scopeKey'],
                isUnique: true,
            }],
        }), true);
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        if (await queryRunner.hasTable('resource_share_scope')) {
            await queryRunner.dropTable('resource_share_scope', true);
        }
    }
}
