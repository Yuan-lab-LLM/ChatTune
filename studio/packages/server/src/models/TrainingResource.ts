import {
    BaseEntity,
    Column,
    CreateDateColumn,
    Entity,
    Index,
    PrimaryColumn,
    UpdateDateColumn,
} from 'typeorm';

export enum TrainingReservationStatus {
    PREPARING = 'preparing',
    RESERVED = 'reserved',
    RUNNING = 'running',
    RELEASED = 'released',
    FAILED = 'failed',
}

@Entity('resource_pool')
export class ResourcePoolTable extends BaseEntity {
    @PrimaryColumn()
    id: string;

    @Column({ unique: true })
    name: string;

    @Column({ type: 'text', nullable: true })
    description?: string | null;

    @Column({ default: true })
    enabled: boolean;

    @CreateDateColumn({ type: 'datetime' })
    createdAt: Date;

    @UpdateDateColumn({ type: 'datetime' })
    updatedAt: Date;
}

@Entity('resource_pool_node')
@Index('IDX_resource_pool_node_unique', ['poolId', 'nodeId'], { unique: true })
export class ResourcePoolNodeTable extends BaseEntity {
    @PrimaryColumn()
    id: string;

    @Column()
    poolId: string;

    @Column()
    nodeId: string;

    @Column()
    sshAlias: string;

    @Column()
    trainAddress: string;

    @Column({ type: 'text', nullable: true })
    ncclSocketIfname?: string | null;

    @Column({ type: 'simple-json', nullable: true })
    allowedGpuIndexes?: number[] | null;

    @Column({ default: true })
    enabled: boolean;
}

@Entity('group_resource_quota')
@Index('IDX_group_resource_quota_unique', ['groupId', 'poolId'], { unique: true })
export class GroupResourceQuotaTable extends BaseEntity {
    @PrimaryColumn()
    id: string;

    @Column()
    groupId: string;

    @Column()
    poolId: string;

    @Column()
    homeNodeId: string;

    @Column({ default: 0 })
    guaranteedGpuCount: number;

    @Column({ default: 1 })
    maxGpuCount: number;

    @Column({ default: 1 })
    maxConcurrentJobs: number;

    @Column({ default: 1 })
    maxNodesPerJob: number;

    @UpdateDateColumn({ type: 'datetime' })
    updatedAt: Date;
}

@Entity('training_reservation')
@Index('IDX_training_reservation_group_status', ['groupId', 'status'])
export class TrainingReservationTable extends BaseEntity {
    @PrimaryColumn()
    id: string;

    @Column()
    groupId: string;

    @Column()
    poolId: string;

    @Column()
    homeNodeId: string;

    @Column({ type: 'text', nullable: true })
    requestedByUserId?: string | null;

    @Column({ type: 'varchar', enum: TrainingReservationStatus })
    status: TrainingReservationStatus;

    @Column()
    requestedNodeCount: number;

    @Column()
    gpusPerNode: number;

    @Column()
    masterPort: number;

    @Column({ type: 'text', nullable: true })
    taskCategory?: string | null;

    @Column({ type: 'text', nullable: true })
    taskType?: string | null;

    @Column({ type: 'text', nullable: true })
    taskTypeText?: string | null;

    @Column()
    expiresAt: string;

    @Column({ type: 'text', nullable: true })
    errorMessage?: string | null;

    @Column({ type: 'text', nullable: true })
    expiredReason?: string | null;

    @Column({ type: 'text', nullable: true })
    lastRenewedAt?: string | null;

    @Column({ type: 'text', nullable: true })
    releaseResult?: string | null;

    @Column({ type: 'text', nullable: true })
    releasedAt?: string | null;

    @CreateDateColumn({ type: 'datetime' })
    createdAt: Date;

    @UpdateDateColumn({ type: 'datetime' })
    updatedAt: Date;
}

@Entity('training_reservation_node')
@Index('IDX_training_reservation_node_unique', ['reservationId', 'nodeId'], { unique: true })
export class TrainingReservationNodeTable extends BaseEntity {
    @PrimaryColumn()
    id: string;

    @Column()
    reservationId: string;

    @Column()
    nodeId: string;

    @Column({ type: 'simple-json' })
    gpuIndexes: number[];

    @Column({ default: false })
    isMaster: boolean;
}

@Entity('training_resource_lock')
export class TrainingResourceLockTable extends BaseEntity {
    @PrimaryColumn()
    lockKey: string;

    @Column()
    ownerId: string;

    @Column()
    expiresAt: string;

    @UpdateDateColumn({ type: 'datetime' })
    updatedAt: Date;
}
