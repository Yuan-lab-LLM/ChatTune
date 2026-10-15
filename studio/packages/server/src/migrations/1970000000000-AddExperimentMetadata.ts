import { TableColumn } from 'typeorm';
import type { MigrationInterface, QueryRunner } from 'typeorm';

export class AddExperimentMetadata1970000000000 implements MigrationInterface {
    name = 'AddExperimentMetadata1970000000000';
    async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.addColumn('experiment', new TableColumn({ name: 'notes', type: 'text', default: "''" }));
        await queryRunner.addColumn('experiment', new TableColumn({ name: 'tags', type: 'text', default: "'[]'" }));
    }
    async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.dropColumn('experiment', 'tags');
        await queryRunner.dropColumn('experiment', 'notes');
    }
}
