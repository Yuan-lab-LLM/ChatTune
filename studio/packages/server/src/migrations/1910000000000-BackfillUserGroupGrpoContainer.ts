import { MigrationInterface, QueryRunner } from 'typeorm';

export class BackfillUserGroupGrpoContainer1910000000000 implements MigrationInterface {
    name = 'BackfillUserGroupGrpoContainer1910000000000';

    public async up(queryRunner: QueryRunner): Promise<void> {
        if (!(await queryRunner.hasTable('user_group'))) return;
        if (!(await queryRunner.hasColumn('user_group', 'defaultGrpoContainerName'))) return;

        const defaultContainer =
            process.env.MEDFLOW_LOCAL_GRPO_CONTAINER?.trim() ||
            process.env.MEDFLOW_GRPO_DOCKER_CONTAINER?.trim() ||
            process.env.MEDFLOW_GRPO_CONTAINER?.trim() ||
            process.env.AGENT3_DEFAULT_GRPO_DOCKER_CONTAINER?.trim() ||
            'qingnang_grpo';
        const escapedDefault = defaultContainer.replace(/'/g, "''");

        await queryRunner.query(
            `UPDATE user_group
             SET defaultGrpoContainerName = '${escapedDefault}'
             WHERE defaultGrpoContainerName IS NULL
                OR TRIM(defaultGrpoContainerName) = ''
                OR defaultGrpoContainerName = 'grpo_container'`,
        );
    }

    public async down(_queryRunner: QueryRunner): Promise<void> {
        // Keep corrected container names on rollback.
    }
}
