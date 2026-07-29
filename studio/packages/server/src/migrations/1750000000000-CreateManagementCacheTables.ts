import { MigrationInterface, QueryRunner, Table } from 'typeorm';

export class CreateManagementCacheTables1750000000000
    implements MigrationInterface
{
    name = 'CreateManagementCacheTables1750000000000';

    public async up(queryRunner: QueryRunner): Promise<void> {
        const hasManagementCache = await queryRunner.hasTable('management_cache');
        if (!hasManagementCache) {
            await queryRunner.createTable(
                new Table({
                    name: 'management_cache',
                    columns: [
                        {
                            name: 'id',
                            type: 'varchar',
                            isPrimary: true,
                        },
                        {
                            name: 'bizType',
                            type: 'varchar',
                        },
                        {
                            name: 'containerName',
                            type: 'varchar',
                        },
                        {
                            name: 'itemKey',
                            type: 'varchar',
                        },
                        {
                            name: 'payload',
                            type: 'text',
                        },
                        {
                            name: 'sourcePath',
                            type: 'varchar',
                            isNullable: true,
                        },
                        {
                            name: 'createdAt',
                            type: 'datetime',
                            default: "datetime('now')",
                        },
                        {
                            name: 'updatedAt',
                            type: 'datetime',
                            default: "datetime('now')",
                        },
                    ],
                    indices: [
                        {
                            name: 'IDX_management_cache_biz_type',
                            columnNames: ['bizType'],
                        },
                        {
                            name: 'IDX_management_cache_container_name',
                            columnNames: ['containerName'],
                        },
                        {
                            name: 'IDX_management_cache_biz_container',
                            columnNames: ['bizType', 'containerName'],
                        },
                        {
                            name: 'IDX_management_cache_updated_at',
                            columnNames: ['updatedAt'],
                        },
                    ],
                }),
                true,
            );
        }

        const hasRefreshJob = await queryRunner.hasTable('refresh_job');
        if (!hasRefreshJob) {
            await queryRunner.createTable(
                new Table({
                    name: 'refresh_job',
                    columns: [
                        {
                            name: 'id',
                            type: 'varchar',
                            isPrimary: true,
                        },
                        {
                            name: 'bizType',
                            type: 'varchar',
                        },
                        {
                            name: 'containerName',
                            type: 'varchar',
                        },
                        {
                            name: 'triggerType',
                            type: 'varchar',
                        },
                        {
                            name: 'status',
                            type: 'varchar',
                        },
                        {
                            name: 'errorMessage',
                            type: 'text',
                            isNullable: true,
                        },
                        {
                            name: 'startedAt',
                            type: 'datetime',
                            default: "datetime('now')",
                        },
                        {
                            name: 'finishedAt',
                            type: 'datetime',
                            isNullable: true,
                        },
                        {
                            name: 'updatedAt',
                            type: 'datetime',
                            default: "datetime('now')",
                        },
                    ],
                    indices: [
                        {
                            name: 'IDX_refresh_job_biz_container',
                            columnNames: ['bizType', 'containerName'],
                        },
                        {
                            name: 'IDX_refresh_job_status',
                            columnNames: ['status'],
                        },
                    ],
                }),
                true,
            );
        }
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        if (await queryRunner.hasTable('refresh_job')) {
            await queryRunner.dropTable('refresh_job', true);
        }

        if (await queryRunner.hasTable('management_cache')) {
            await queryRunner.dropTable('management_cache', true);
        }
    }
}
