import {
    BaseEntity,
    Column,
    CreateDateColumn,
    Entity,
    Index,
    PrimaryColumn,
    UpdateDateColumn,
} from 'typeorm';
import { ManagementBizType } from '../../../shared/src';

export enum ResourceVisibility {
    PUBLIC = 'public',
    PRIVATE = 'private',
}

export enum PublicationStatus {
    PENDING = 'pending',
    APPROVED = 'approved',
    REJECTED = 'rejected',
}

@Entity('user_group')
export class UserGroupTable extends BaseEntity {
    @PrimaryColumn()
    id: string;

    @Column({ unique: true })
    name: string;

    @Column({ type: 'text', nullable: true })
    description?: string | null;

    @Column({ default: 'training_container' })
    defaultContainerName: string;

    @Column({ default: 'evaluation_container' })
    defaultEvaluateContainerName: string;

    @Column({ default: 'qingnang_grpo' })
    defaultGrpoContainerName: string;

    @Column({ default: 'qingnang_train_multi' })
    defaultMultinodeContainerName: string;

    @CreateDateColumn({ type: 'datetime' })
    createdAt: Date;

    @UpdateDateColumn({ type: 'datetime' })
    updatedAt: Date;
}

@Entity('user_group_member')
@Index('IDX_user_group_member_user', ['userId'], { unique: true })
export class UserGroupMemberTable extends BaseEntity {
    @PrimaryColumn()
    id: string;

    @Column()
    groupId: string;

    @Column()
    userId: string;

    @CreateDateColumn({ type: 'datetime' })
    createdAt: Date;
}

@Entity('group_node_assignment')
@Index('IDX_group_node_assignment_group', ['groupId'], { unique: true })
@Index('IDX_group_node_assignment_node', ['nodeId'])
export class GroupNodeAssignmentTable extends BaseEntity {
    @PrimaryColumn()
    id: string;

    @Column()
    groupId: string;

    @Column()
    nodeId: string;

    @Column({ default: 'pending' })
    trainingContainerStatus: string;

    @Column({ type: 'text', nullable: true })
    trainingContainerError?: string | null;

    @Column({ default: 'pending' })
    evaluationContainerStatus: string;

    @Column({ type: 'text', nullable: true })
    evaluationContainerError?: string | null;

    @Column({ default: 'pending' })
    grpoContainerStatus: string;

    @Column({ type: 'text', nullable: true })
    grpoContainerError?: string | null;

    @Column({ default: 'pending' })
    multinodeContainerStatus: string;

    @Column({ type: 'text', nullable: true })
    multinodeContainerError?: string | null;

    @UpdateDateColumn({ type: 'datetime' })
    updatedAt: Date;
}

@Entity('resource_catalog')
@Index('IDX_resource_catalog_lookup', ['bizType', 'nodeId', 'containerName', 'itemKey'], { unique: true })
export class ResourceCatalogTable extends BaseEntity {
    @PrimaryColumn()
    id: string;

    @Column()
    bizType: ManagementBizType;

    @Column()
    nodeId: string;

    @Column()
    containerName: string;

    @Column()
    itemKey: string;

    @Column({ type: 'text', nullable: true })
    ownerUserId?: string | null;

    @Column({ type: 'text', nullable: true })
    groupId?: string | null;

    @Column({ type: 'varchar', enum: ResourceVisibility, default: ResourceVisibility.PUBLIC })
    visibility: ResourceVisibility;

    @Column({ type: 'text', nullable: true })
    sourcePath?: string | null;

    @Column({ type: 'simple-json', nullable: true })
    payload?: Record<string, unknown> | null;

    @CreateDateColumn({ type: 'datetime' })
    createdAt: Date;

    @UpdateDateColumn({ type: 'datetime' })
    updatedAt: Date;
}

@Entity('resource_share_scope')
@Index('IDX_resource_share_scope_unique', ['resourceId', 'scopeKey'], { unique: true })
export class ResourceShareScopeTable extends BaseEntity {
    @PrimaryColumn()
    id: string;

    @Column()
    resourceId: string;

    @Column()
    scopeKey: string;

    @Column({ type: 'text', nullable: true })
    groupId?: string | null;

    @CreateDateColumn({ type: 'datetime' })
    createdAt: Date;
}

@Entity('resource_audit_event')
@Index('IDX_resource_audit_event_resource_created', ['resourceId', 'createdAt'])
export class ResourceAuditEventTable extends BaseEntity {
    @PrimaryColumn()
    id: string;

    @Column()
    eventType: string;

    @Column({ type: 'text', nullable: true })
    resourceId?: string | null;

    @Column({ type: 'text', nullable: true })
    actorUserId?: string | null;

    @Column({ type: 'simple-json', nullable: true })
    details?: Record<string, unknown> | null;

    @CreateDateColumn({ type: 'datetime' })
    createdAt: Date;
}

@Entity('resource_publication_request')
export class ResourcePublicationRequestTable extends BaseEntity {
    @PrimaryColumn()
    id: string;

    @Column()
    resourceId: string;

    @Column()
    requesterUserId: string;

    @Column({ type: 'varchar', enum: PublicationStatus, default: PublicationStatus.PENDING })
    status: PublicationStatus;

    @Column({ type: 'text', nullable: true })
    reviewedBy?: string | null;

    @Column({ type: 'text', nullable: true })
    reviewNote?: string | null;

    @CreateDateColumn({ type: 'datetime' })
    createdAt: Date;

    @UpdateDateColumn({ type: 'datetime' })
    updatedAt: Date;
}
