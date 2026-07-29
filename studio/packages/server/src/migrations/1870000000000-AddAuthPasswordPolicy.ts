import { MigrationInterface, QueryRunner, Table, TableColumn } from 'typeorm';

export class AddAuthPasswordPolicy1870000000000 implements MigrationInterface {
    async up(queryRunner: QueryRunner): Promise<void> {
        if (
            (await queryRunner.hasTable('auth_user_table')) &&
            !(await queryRunner.hasColumn('auth_user_table', 'mustChangePassword'))
        ) {
            await queryRunner.addColumn(
                'auth_user_table',
                new TableColumn({
                    name: 'mustChangePassword',
                    type: 'boolean',
                    default: false,
                }),
            );
        }

        if (!(await queryRunner.hasTable('auth_login_failure_table'))) {
            await queryRunner.createTable(
                new Table({
                    name: 'auth_login_failure_table',
                    columns: [
                        {
                            name: 'username',
                            type: 'varchar',
                            isPrimary: true,
                        },
                        {
                            name: 'failureCount',
                            type: 'integer',
                            default: 0,
                        },
                        {
                            name: 'firstFailedAt',
                            type: 'text',
                        },
                        {
                            name: 'lockedUntil',
                            type: 'text',
                            isNullable: true,
                        },
                        {
                            name: 'updatedAt',
                            type: 'text',
                        },
                    ],
                }),
            );
        }
    }

    async down(queryRunner: QueryRunner): Promise<void> {
        if (await queryRunner.hasTable('auth_login_failure_table')) {
            await queryRunner.dropTable('auth_login_failure_table');
        }

        if (
            (await queryRunner.hasTable('auth_user_table')) &&
            (await queryRunner.hasColumn('auth_user_table', 'mustChangePassword'))
        ) {
            await queryRunner.dropColumn('auth_user_table', 'mustChangePassword');
        }
    }
}
