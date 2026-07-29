import { MigrationInterface, QueryRunner, TableColumn } from 'typeorm';

export class AddGroupContainerStatus1800000000000 implements MigrationInterface {
    name = 'AddGroupContainerStatus1800000000000';

    public async up(queryRunner: QueryRunner): Promise<void> {
        if (!(await queryRunner.hasTable('group_node_assignment'))) return;
        if (!(await queryRunner.hasColumn('group_node_assignment', 'trainingContainerStatus'))) {
            await queryRunner.addColumn('group_node_assignment', new TableColumn({
                name: 'trainingContainerStatus',
                type: 'varchar',
                default: "'pending'",
            }));
        }
        if (!(await queryRunner.hasColumn('group_node_assignment', 'trainingContainerError'))) {
            await queryRunner.addColumn('group_node_assignment', new TableColumn({
                name: 'trainingContainerError',
                type: 'text',
                isNullable: true,
            }));
        }
        if (!(await queryRunner.hasColumn('group_node_assignment', 'evaluationContainerStatus'))) {
            await queryRunner.addColumn('group_node_assignment', new TableColumn({
                name: 'evaluationContainerStatus',
                type: 'varchar',
                default: "'pending'",
            }));
        }
        if (!(await queryRunner.hasColumn('group_node_assignment', 'evaluationContainerError'))) {
            await queryRunner.addColumn('group_node_assignment', new TableColumn({
                name: 'evaluationContainerError',
                type: 'text',
                isNullable: true,
            }));
        }
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        if (!(await queryRunner.hasTable('group_node_assignment'))) return;
        for (const column of [
            'evaluationContainerError',
            'evaluationContainerStatus',
            'trainingContainerError',
            'trainingContainerStatus',
        ]) {
            if (await queryRunner.hasColumn('group_node_assignment', column)) {
                await queryRunner.dropColumn('group_node_assignment', column);
            }
        }
    }
}
