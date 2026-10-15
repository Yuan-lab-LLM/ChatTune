import { MigrationInterface, QueryRunner, TableColumn } from 'typeorm';

export class AddExperimentReservation1940000000000 implements MigrationInterface {
    name = 'AddExperimentReservation1940000000000';
    async up(queryRunner: QueryRunner): Promise<void> {
        if (!await queryRunner.hasColumn('task_attempt', 'resourceReservationId')) {
            await queryRunner.addColumn('task_attempt', new TableColumn({
                name: 'resourceReservationId', type: 'text', isNullable: true,
            }));
        }
    }
    async down(queryRunner: QueryRunner): Promise<void> {
        if (await queryRunner.hasColumn('task_attempt', 'resourceReservationId')) {
            await queryRunner.dropColumn('task_attempt', 'resourceReservationId');
        }
    }
}
