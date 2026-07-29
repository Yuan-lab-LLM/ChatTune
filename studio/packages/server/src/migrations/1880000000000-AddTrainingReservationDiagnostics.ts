import { MigrationInterface, QueryRunner, TableColumn } from 'typeorm';

export class AddTrainingReservationDiagnostics1880000000000 implements MigrationInterface {
    name = 'AddTrainingReservationDiagnostics1880000000000';

    public async up(queryRunner: QueryRunner): Promise<void> {
        if (!(await queryRunner.hasTable('training_reservation'))) return;
        for (const columnName of ['expiredReason', 'lastRenewedAt', 'releaseResult', 'releasedAt']) {
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
        for (const columnName of ['releasedAt', 'releaseResult', 'lastRenewedAt', 'expiredReason']) {
            if (await queryRunner.hasColumn('training_reservation', columnName)) {
                await queryRunner.dropColumn('training_reservation', columnName);
            }
        }
    }
}