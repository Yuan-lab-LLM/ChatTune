import {
    BaseEntity,
    Column,
    Entity,
    JoinColumn,
    ManyToOne,
    PrimaryColumn,
} from 'typeorm';
import { RunTable } from './Run';

@Entity()
export class InputRequestTable extends BaseEntity {
    @PrimaryColumn({ nullable: false })
    requestId: string;

    @Column()
    agentId: string;

    @Column()
    agentName: string;

    @Column('json', { nullable: true })
    structuredInput: Record<string, unknown> | null;

    @ManyToOne(() => RunTable, { nullable: false, onDelete: 'CASCADE' })
    @JoinColumn({ name: 'run_id' })
    runId: string;
}
