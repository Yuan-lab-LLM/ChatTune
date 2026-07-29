import { MigrationInterface, QueryRunner, TableColumn } from 'typeorm';

export class AddTrainingReservationTaskType1890000000000 implements MigrationInterface {
    name = 'AddTrainingReservationTaskType1890000000000';

    public async up(queryRunner: QueryRunner): Promise<void> {
        if (!(await queryRunner.hasTable('training_reservation'))) return;
        for (const columnName of ['taskCategory', 'taskType', 'taskTypeText']) {
            if (!(await queryRunner.hasColumn('training_reservation', columnName))) {
                await queryRunner.addColumn('training_reservation', new TableColumn({
                    name: columnName,
                    type: 'text',
                    isNullable: true,
                }));
            }
        }
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        if (!(await queryRunner.hasTable('training_reservation'))) return;
        for (const columnName of ['taskTypeText', 'taskType', 'taskCategory']) {
            if (await queryRunner.hasColumn('training_reservation', columnName)) {
                await queryRunner.dropColumn('training_reservation', columnName);
            }
        }
    }
}