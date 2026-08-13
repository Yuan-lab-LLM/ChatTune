import { MigrationInterface, QueryRunner, TableColumn } from 'typeorm';

export class AddUserGroupDefaultContainer1790000000000 implements MigrationInterface {
    name = 'AddUserGroupDefaultContainer1790000000000';

    public async up(queryRunner: QueryRunner): Promise<void> {
        if (!(await queryRunner.hasTable('user_group'))) return;
        if (!(await queryRunner.hasColumn('user_group', 'defaultContainerName'))) {
            await queryRunner.addColumn(
                'user_group',
                new TableColumn({
                    name: 'defaultContainerName',
                    type: 'varchar',
                    default: "'training_container'",
                }),
            );
        }
        if (!(await queryRunner.hasColumn('user_group', 'defaultEvaluateContainerName'))) {
            await queryRunner.addColumn(
                'user_group',
                new TableColumn({
                    name: 'defaultEvaluateContainerName',
                    type: 'varchar',
                    default: "''",
                }),
            );
        }
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        if (!(await queryRunner.hasTable('user_group'))) return;
        if (await queryRunner.hasColumn('user_group', 'defaultContainerName')) {
            await queryRunner.dropColumn('user_group', 'defaultContainerName');
        }
        if (await queryRunner.hasColumn('user_group', 'defaultEvaluateContainerName')) {
            await queryRunner.dropColumn('user_group', 'defaultEvaluateContainerName');
        }
    }
}
