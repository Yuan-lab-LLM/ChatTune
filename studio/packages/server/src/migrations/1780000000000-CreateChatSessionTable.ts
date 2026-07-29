import { MigrationInterface, QueryRunner, Table } from 'typeorm';

export class CreateChatSessionTable1780000000000 implements MigrationInterface {
    name = 'CreateChatSessionTable1780000000000';

    public async up(queryRunner: QueryRunner): Promise<void> {
        if (await queryRunner.hasTable('chat_session')) {
            return;
        }

        await queryRunner.createTable(
            new Table({
                name: 'chat_session',
                columns: [
                    { name: 'id', type: 'varchar', isPrimary: true },
                    { name: 'userId', type: 'varchar' },
                    { name: 'runId', type: 'varchar' },
                    { name: 'sessionId', type: 'varchar' },
                    { name: 'clearedAt', type: 'text', isNullable: true },
                    {
                        name: 'updatedAt',
                        type: 'datetime',
                        default: "datetime('now')",
                    },
                ],
                uniques: [
                    {
                        name: 'UQ_chat_session_user_run',
                        columnNames: ['userId', 'runId'],
                    },
                ],
            }),
            true,
        );
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        if (await queryRunner.hasTable('chat_session')) {
            await queryRunner.dropTable('chat_session', true);
        }
    }
}
