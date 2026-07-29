import {
    BaseEntity,
    Column,
    CreateDateColumn,
    Entity,
    Index,
    PrimaryColumn,
    UpdateDateColumn,
} from 'typeorm';
import {
    ManagementBizType,
    RefreshJobStatus,
    RefreshTriggerType,
} from '../../../shared/src';

@Entity('refresh_job')
@Index('IDX_refresh_job_biz_node_container', ['bizType', 'nodeId', 'containerName'])
@Index('IDX_refresh_job_status', ['status'])
export class RefreshJobTable extends BaseEntity {
    @PrimaryColumn()
    id: string;

    @Column()
    bizType: ManagementBizType;

    @Column({ default: 'local' })
    @Index('IDX_refresh_job_node_id')
    nodeId: string;

    @Column()
    containerName: string;

    @Column()
    triggerType: RefreshTriggerType;

    @Column()
    status: RefreshJobStatus;

    @Column({ type: 'text', nullable: true })
    errorMessage?: string | null;

    @CreateDateColumn({ type: 'datetime' })
    startedAt: Date;

    @Column({ type: 'datetime', nullable: true })
    finishedAt?: Date | null;

    @UpdateDateColumn({ type: 'datetime' })
    updatedAt: Date;
}
