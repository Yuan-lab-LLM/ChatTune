import { BaseEntity, Column, Entity, PrimaryColumn } from 'typeorm';

export enum UserRole {
    ADMIN = 'admin',
    USER = 'user',
}

@Entity()
export class AuthUserTable extends BaseEntity {
    @PrimaryColumn()
    id: string;

    @Column({ unique: true })
    username: string;

    @Column()
    passwordHash: string;

    @Column()
    passwordSalt: string;

    @Column({ type: 'varchar', enum: UserRole, default: UserRole.USER })
    role: UserRole;

    @Column({ default: false })
    disabled: boolean;

    @Column({ default: false })
    mustChangePassword: boolean;

    @Column()
    createdAt: string;

    @Column({ type: 'text', nullable: true })
    createdBy?: string;
}

@Entity()
export class AuthSessionTable extends BaseEntity {
    @PrimaryColumn()
    token: string;

    @Column()
    userId: string;

    @Column()
    createdAt: string;

    @Column()
    expiresAt: string;

    @Column({ type: 'text', nullable: true })
    lastSeenAt?: string | null;
}

@Entity()
export class AuthLoginFailureTable extends BaseEntity {
    @PrimaryColumn()
    username: string;

    @Column({ default: 0 })
    failureCount: number;

    @Column()
    firstFailedAt: string;

    @Column({ type: 'text', nullable: true })
    lockedUntil?: string | null;

    @Column()
    updatedAt: string;
}

