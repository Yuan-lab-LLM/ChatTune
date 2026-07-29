import { MigrationInterface, QueryRunner, Table } from 'typeorm';

export class CreateTrainingResourceTables1810000000000 implements MigrationInterface {
    name = 'CreateTrainingResourceTables1810000000000';

    public async up(queryRunner: QueryRunner): Promise<void> {
        const tables = [
            new Table({
                name: 'resource_pool',
                columns: [
                    { name: 'id', type: 'varchar', isPrimary: true },
                    { name: 'name', type: 'varchar', isUnique: true },
                    { name: 'description', type: 'text', isNullable: true },
                    { name: 'createdAt', type: 'datetime', default: "datetime('now')" },
                    { name: 'updatedAt', type: 'datetime', default: "datetime('now')" },
                ],
            }),
            new Table({
                name: 'resource_pool_node',
                columns: [
                    { name: 'id', type: 'varchar', isPrimary: true },
                    { name: 'poolId', type: 'varchar' },
                    { name: 'nodeId', type: 'varchar' },
                    { name: 'sshAlias', type: 'varchar' },
                    { name: 'trainAddress', type: 'varchar' },
                    { name: 'ncclSocketIfname', type: 'text', isNullable: true },
                    { name: 'enabled', type: 'boolean', default: '1' },
                ],
                indices: [{
                    name: 'IDX_resource_pool_node_unique',
                    columnNames: ['poolId', 'nodeId'],
                    isUnique: true,
                }],
            }),
            new Table({
                name: 'group_resource_quota',
                columns: [
                    { name: 'id', type: 'varchar', isPrimary: true },
                    { name: 'groupId', type: 'varchar' },
                    { name: 'poolId', type: 'varchar' },
                    { name: 'homeNodeId', type: 'varchar' },
                    { name: 'guaranteedGpuCount', type: 'integer', default: '0' },
                    { name: 'maxGpuCount', type: 'integer', default: '1' },
                    { name: 'maxConcurrentJobs', type: 'integer', default: '1' },
                    { name: 'maxNodesPerJob', type: 'integer', default: '1' },
                    { name: 'updatedAt', type: 'datetime', default: "datetime('now')" },
                ],
                indices: [{
                    name: 'IDX_group_resource_quota_unique',
                    columnNames: ['groupId', 'poolId'],
                    isUnique: true,
                }],
            }),
            new Table({
                name: 'training_reservation',
                columns: [
                    { name: 'id', type: 'varchar', isPrimary: true },
                    { name: 'groupId', type: 'varchar' },
                    { name: 'poolId', type: 'varchar' },
                    { name: 'homeNodeId', type: 'varchar' },
                    { name: 'requestedByUserId', type: 'text', isNullable: true },
                    { name: 'status', type: 'varchar' },
                    { name: 'requestedNodeCount', type: 'integer' },
                    { name: 'gpusPerNode', type: 'integer' },
                    { name: 'masterPort', type: 'integer' },
                    { name: 'expiresAt', type: 'varchar' },
                    { name: 'errorMessage', type: 'text', isNullable: true },
                    { name: 'createdAt', type: 'datetime', default: "datetime('now')" },
                    { name: 'updatedAt', type: 'datetime', default: "datetime('now')" },
                ],
                indices: [{
                    name: 'IDX_training_reservation_group_status',
                    columnNames: ['groupId', 'status'],
                }],
            }),
            new Table({
                name: 'training_reservation_node',
                columns: [
                    { name: 'id', type: 'varchar', isPrimary: true },
                    { name: 'reservationId', type: 'varchar' },
                    { name: 'nodeId', type: 'varchar' },
                    { name: 'gpuIndexes', type: 'text' },
                    { name: 'isMaster', type: 'boolean', default: '0' },
                ],
                indices: [{
                    name: 'IDX_training_reservation_node_unique',
                    columnNames: ['reservationId', 'nodeId'],
                    isUnique: true,
                }],
            }),
        ];

        for (const table of tables) {
            if (!(await queryRunner.hasTable(table.name))) {
                await queryRunner.createTable(table, true);
            }
        }
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        for (const table of [
            'training_reservation_node',
            'training_reservation',
            'group_resource_quota',
            'resource_pool_node',
            'resource_pool',
        ]) {
            if (await queryRunner.hasTable(table)) {
                await queryRunner.dropTable(table, true);
            }
        }
    }
}
