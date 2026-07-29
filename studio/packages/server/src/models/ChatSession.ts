import {
    BaseEntity,
    Column,
    Entity,
    Index,
    PrimaryColumn,
    UpdateDateColumn,
} from 'typeorm';

@Entity('chat_session')
@Index(['userId', 'runId'], { unique: true })
export class ChatSessionTable extends BaseEntity {
    @PrimaryColumn()
    id: string;

    @Column()
    userId: string;

    @Column()
    runId: string;

    @Column()
    sessionId: string;

    @Column({ type: 'text', nullable: true })
    clearedAt?: string | null;

    @UpdateDateColumn()
    updatedAt: Date;
}
