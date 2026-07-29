import { MigrationInterface, QueryRunner, TableColumn } from 'typeorm';

export class AddAuthSessionLastSeenAt1850000000000 implements MigrationInterface {
    async up(queryRunner: QueryRunner): Promise<void> {
        if (
            !(await queryRunner.hasTable('auth_session_table')) ||
            (await queryRunner.hasColumn('auth_session_table', 'lastSeenAt'))
        ) {
            return;
        }

        await queryRunner.addColumn(
            'auth_session_table',
            new TableColumn({
                name: 'lastSeenAt',
                type: 'text',
                isNullable: true,
            }),
        );
    }

    async down(queryRunner: QueryRunner): Promise<void> {
        if (
            (await queryRunner.hasTable('auth_session_table')) &&
            (await queryRunner.hasColumn('auth_session_table', 'lastSeenAt'))
        ) {
            await queryRunner.dropColumn('auth_session_table', 'lastSeenAt');
        }
    }
}
