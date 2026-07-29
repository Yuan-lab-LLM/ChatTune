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

@Entity('management_cache')
@Index('IDX_management_cache_biz_node_container', ['bizType', 'nodeId', 'containerName'])
@Index('IDX_management_cache_updated_at', ['updatedAt'])
export class ManagementCacheTable extends BaseEntity {
    @PrimaryColumn()
    id: string;

    @Column()
    @Index('IDX_management_cache_biz_type')
    bizType: ManagementBizType;

    @Column({ default: 'local' })
    @Index('IDX_management_cache_node_id')
    nodeId: string;

    @Column()
    @Index('IDX_management_cache_container_name')
    containerName: string;

    @Column()
    itemKey: string;

    @Column({ type: 'simple-json' })
    payload: Record<string, unknown>;

    @Column({ nullable: true })
    sourcePath?: string;

    @CreateDateColumn({ type: 'datetime' })
    createdAt: Date;

    @UpdateDateColumn({ type: 'datetime' })
    updatedAt: Date;
}
