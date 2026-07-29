import { MigrationInterface, QueryRunner, Table } from 'typeorm';

export class CreateResourceAuditEventTable1830000000000 implements MigrationInterface {
    name = 'CreateResourceAuditEventTable1830000000000';

    public async up(queryRunner: QueryRunner): Promise<void> {
        if (await queryRunner.hasTable('resource_audit_event')) return;
        await queryRunner.createTable(new Table({
            name: 'resource_audit_event',
            columns: [
                { name: 'id', type: 'varchar', isPrimary: true },
                { name: 'eventType', type: 'varchar' },
                { name: 'resourceId', type: 'text', isNullable: true },
                { name: 'actorUserId', type: 'text', isNullable: true },
                { name: 'details', type: 'text', isNullable: true },
                { name: 'createdAt', type: 'datetime', default: "datetime('now')" },
            ],
            indices: [{
                name: 'IDX_resource_audit_event_resource_created',
                columnNames: ['resourceId', 'createdAt'],
            }],
        }), true);
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        if (await queryRunner.hasTable('resource_audit_event')) {
            await queryRunner.dropTable('resource_audit_event', true);
        }
    }
}
