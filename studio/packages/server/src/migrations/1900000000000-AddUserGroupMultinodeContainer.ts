import { MigrationInterface, QueryRunner, TableColumn } from 'typeorm';

export class AddUserGroupMultinodeContainer1900000000000 implements MigrationInterface {
    name = 'AddUserGroupMultinodeContainer1900000000000';

    public async up(queryRunner: QueryRunner): Promise<void> {
        const defaultContainer =
            process.env.MULTINODE_DOCKER_CONTAINER?.trim() ||
            process.env.MEDFLOW_MULTINODE_DOCKER_CONTAINER?.trim() ||
            process.env.MEDFLOW_LOCAL_MULTINODE_CONTAINER?.trim() ||
            'qingnang_train_multi';

        if (await queryRunner.hasTable('user_group')) {
            if (!(await queryRunner.hasColumn('user_group', 'defaultMultinodeContainerName'))) {
                await queryRunner.addColumn(
                    'user_group',
                    new TableColumn({
                        name: 'defaultMultinodeContainerName',
                        type: 'varchar',
                        default: `'${defaultContainer.replace(/'/g, "''")}'`,
                    }),
                );
            }
        }

        if (await queryRunner.hasTable('group_node_assignment')) {
            if (!(await queryRunner.hasColumn('group_node_assignment', 'multinodeContainerStatus'))) {
                await queryRunner.addColumn(
                    'group_node_assignment',
                    new TableColumn({
                        name: 'multinodeContainerStatus',
                        type: 'varchar',
                        default: "'pending'",
                    }),
                );
            }
            if (!(await queryRunner.hasColumn('group_node_assignment', 'multinodeContainerError'))) {
                await queryRunner.addColumn(
                    'group_node_assignment',
                    new TableColumn({
                        name: 'multinodeContainerError',
                        type: 'text',
                        isNullable: true,
                    }),
                );
            }
        }
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        if (await queryRunner.hasTable('group_node_assignment')) {
            if (await queryRunner.hasColumn('group_node_assignment', 'multinodeContainerError')) {
                await queryRunner.dropColumn('group_node_assignment', 'multinodeContainerError');
            }
            if (await queryRunner.hasColumn('group_node_assignment', 'multinodeContainerStatus')) {
                await queryRunner.dropColumn('group_node_assignment', 'multinodeContainerStatus');
            }
        }

        if (await queryRunner.hasTable('user_group')) {
            if (await queryRunner.hasColumn('user_group', 'defaultMultinodeContainerName')) {
                await queryRunner.dropColumn('user_group', 'defaultMultinodeContainerName');
            }
        }
    }
}
