import { MigrationInterface, QueryRunner, TableColumn } from 'typeorm';

export class AddUserGroupGrpoContainer1810000000000 implements MigrationInterface {
    name = 'AddUserGroupGrpoContainer1810000000000';

    public async up(queryRunner: QueryRunner): Promise<void> {
        if (await queryRunner.hasTable('user_group')) {
            if (!(await queryRunner.hasColumn('user_group', 'defaultGrpoContainerName'))) {
                await queryRunner.addColumn(
                    'user_group',
                    new TableColumn({
                        name: 'defaultGrpoContainerName',
                        type: 'varchar',
                        default: "''",
                    }),
                );
            }
        }

        if (await queryRunner.hasTable('group_node_assignment')) {
            if (!(await queryRunner.hasColumn('group_node_assignment', 'grpoContainerStatus'))) {
                await queryRunner.addColumn(
                    'group_node_assignment',
                    new TableColumn({
                        name: 'grpoContainerStatus',
                        type: 'varchar',
                        default: "'pending'",
                    }),
                );
            }
            if (!(await queryRunner.hasColumn('group_node_assignment', 'grpoContainerError'))) {
                await queryRunner.addColumn(
                    'group_node_assignment',
                    new TableColumn({
                        name: 'grpoContainerError',
                        type: 'text',
                        isNullable: true,
                    }),
                );
            }
        }
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        if (await queryRunner.hasTable('group_node_assignment')) {
            if (await queryRunner.hasColumn('group_node_assignment', 'grpoContainerError')) {
                await queryRunner.dropColumn('group_node_assignment', 'grpoContainerError');
            }
            if (await queryRunner.hasColumn('group_node_assignment', 'grpoContainerStatus')) {
                await queryRunner.dropColumn('group_node_assignment', 'grpoContainerStatus');
            }
        }

        if (await queryRunner.hasTable('user_group')) {
            if (await queryRunner.hasColumn('user_group', 'defaultGrpoContainerName')) {
                await queryRunner.dropColumn('user_group', 'defaultGrpoContainerName');
            }
        }
    }
}