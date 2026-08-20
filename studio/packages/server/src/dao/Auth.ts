import crypto from 'crypto';
import { MoreThan } from 'typeorm';
import {
    AuthLoginFailureTable,
    AuthSessionTable,
    AuthUserTable,
    UserRole,
} from '../models/Auth';
import {
    GroupNodeAssignmentTable,
    UserGroupMemberTable,
    UserGroupTable,
} from '../models/ResourceAccess';
import { ResourceAccessService } from '../services/resourceAccessService';

export interface SafeAuthUser {
    id: string;
    username: string;
    role: UserRole;
    disabled: boolean;
    mustChangePassword: boolean;
    createdAt: string;
    createdBy?: string | null;
    assignedNodeId?: string | null;
    isOnline?: boolean;
    group?: {
        id: string;
        name: string;
        defaultContainerName: string;
        defaultEvaluateContainerName: string;
        defaultGrpoContainerName: string;
        defaultMultinodeContainerName: string;
    } | null;
}

const SESSION_DAYS = Math.max(
    1,
    Number(process.env.MEDFLOW_SESSION_DAYS || 7) || 7,
);
const SESSION_MAX_AGE_SECONDS = Math.floor(SESSION_DAYS * 24 * 60 * 60);
const sessionExpiresAt = (fromMs = Date.now()) =>
    new Date(fromMs + SESSION_MAX_AGE_SECONDS * 1000).toISOString();
const ONLINE_WINDOW_MS = 90_000;
const LAST_SEEN_WRITE_INTERVAL_MS = 15_000;
const LOGIN_FAILURE_WINDOW_MS = Number(
    process.env.MEDFLOW_LOGIN_FAILURE_WINDOW_MS || 15 * 60 * 1000,
);
const LOGIN_LOCKOUT_MS = Number(
    process.env.MEDFLOW_LOGIN_LOCKOUT_MS || 15 * 60 * 1000,
);
const LOGIN_MAX_FAILURES = Number(process.env.MEDFLOW_LOGIN_MAX_FAILURES || 5);

const toSafeUser = (user: AuthUserTable): SafeAuthUser => ({
    id: user.id,
    username: user.username,
    role: user.role,
    disabled: user.disabled ?? false,
    mustChangePassword: user.mustChangePassword ?? false,
    createdAt: user.createdAt,
    createdBy: user.createdBy ?? null,
});

const hashPassword = (password: string, salt: string) =>
    crypto.scryptSync(password, salt, 64).toString('hex');

const passwordMatches = (password: string, user: AuthUserTable) => {
    const candidateHash = hashPassword(password, user.passwordSalt);
    return crypto.timingSafeEqual(
        Buffer.from(candidateHash, 'hex'),
        Buffer.from(user.passwordHash, 'hex'),
    );
};

export class AuthUserDisabledError extends Error {
    constructor() {
        super('auth.error.accountDisabled');
    }
}

export class AuthDao {
    static sessionMaxAgeSeconds() {
        return SESSION_MAX_AGE_SECONDS;
    }

    private static loginFailureKey(username: string) {
        return username.trim().toLowerCase();
    }

    private static async assertLoginNotLocked(username: string) {
        const key = this.loginFailureKey(username);
        const state = await AuthLoginFailureTable.findOne({
            where: { username: key },
        });
        if (!state?.lockedUntil) return;

        if (new Date(state.lockedUntil).getTime() <= Date.now()) {
            await AuthLoginFailureTable.delete({ username: key });
            return;
        }

        throw new Error('auth.error.tooManyLoginAttempts');
    }

    private static async recordLoginFailure(username: string) {
        const key = this.loginFailureKey(username);
        const now = Date.now();
        const nowIso = new Date(now).toISOString();
        const current = await AuthLoginFailureTable.findOne({
            where: { username: key },
        });
        const firstFailedAtMs = current?.firstFailedAt
            ? new Date(current.firstFailedAt).getTime()
            : 0;
        const withinWindow = Boolean(
            current && now - firstFailedAtMs <= LOGIN_FAILURE_WINDOW_MS,
        );
        const failureCount = current && withinWindow ? current.failureCount + 1 : 1;
        const lockedUntil =
            failureCount >= LOGIN_MAX_FAILURES
                ? new Date(now + LOGIN_LOCKOUT_MS).toISOString()
                : null;

        const state = current ?? AuthLoginFailureTable.create({ username: key });
        state.failureCount = failureCount;
        state.firstFailedAt = withinWindow && current ? current.firstFailedAt : nowIso;
        state.lockedUntil = lockedUntil;
        state.updatedAt = nowIso;
        await state.save();
    }

    private static async clearLoginFailures(username: string) {
        await AuthLoginFailureTable.delete({ username: this.loginFailureKey(username) });
    }

    private static async withResourceAccess(user: SafeAuthUser): Promise<SafeAuthUser> {
        if (user.role === UserRole.ADMIN) {
            return { ...user, assignedNodeId: null, group: null };
        }
        const group = await ResourceAccessService.getGroupForUser(user.id);
        return {
            ...user,
            assignedNodeId: await ResourceAccessService.getAssignedNodeId(user.id),
            group: group ? {
                id: group.id,
                name: group.name,
                defaultContainerName: group.defaultContainerName,
                defaultEvaluateContainerName: group.defaultEvaluateContainerName,
                defaultGrpoContainerName: group.defaultGrpoContainerName,
                defaultMultinodeContainerName: group.defaultMultinodeContainerName,
            } : null,
        };
    }

    static async ensureDefaultAdmin() {
        const adminCount = await AuthUserTable.count({
            where: { role: UserRole.ADMIN },
        });

        const existingDefaultAdmin = await AuthUserTable.findOne({
            where: { username: 'admin', role: UserRole.ADMIN },
        });
        if (
            existingDefaultAdmin &&
            !existingDefaultAdmin.mustChangePassword &&
            passwordMatches('admin123', existingDefaultAdmin)
        ) {
            existingDefaultAdmin.mustChangePassword = true;
            await existingDefaultAdmin.save();
            console.warn(
                'Default admin password detected. The admin account must change its password on next login.',
            );
        }

        if (adminCount > 0) {
            return;
        }

        const username = process.env.MEDFLOW_ADMIN_USERNAME?.trim() || 'admin';
        const password = process.env.MEDFLOW_ADMIN_PASSWORD || 'admin123';
        const mustChangePassword = password === 'admin123';

        await this.createUser(
            username,
            password,
            UserRole.ADMIN,
            undefined,
            mustChangePassword,
        );
        console.warn(
            mustChangePassword
                ? `Default admin user created. Username: ${username}. Password must be changed on first login.`
                : `Default admin user created. Username: ${username}.`,
        );
    }

    static async createUser(
        username: string,
        password: string,
        role: UserRole,
        createdBy?: string,
        mustChangePassword = false,
    ): Promise<SafeAuthUser> {
        const normalizedUsername = username.trim();
        const exists = await AuthUserTable.findOne({
            where: { username: normalizedUsername },
        });

        if (exists) {
            throw new Error('auth.error.usernameExists');
        }

        const salt = crypto.randomBytes(16).toString('hex');
        const user = AuthUserTable.create({
            id: crypto.randomUUID(),
            username: normalizedUsername,
            passwordHash: hashPassword(password, salt),
            passwordSalt: salt,
            role,
            mustChangePassword,
            createdAt: new Date().toISOString(),
            createdBy,
        });
        await user.save();
        return toSafeUser(user);
    }

    static async login(username: string, password: string) {
        await this.assertLoginNotLocked(username);
        const user = await AuthUserTable.findOne({
            where: { username: username.trim() },
        });

        if (!user) {
            await this.recordLoginFailure(username);
            return null;
        }

        if (user.disabled) {
            throw new AuthUserDisabledError();
        }

        const passwordOk = passwordMatches(password, user);

        if (!passwordOk) {
            await this.recordLoginFailure(username);
            return null;
        }
        await this.clearLoginFailures(username);

        const now = new Date();
        const expiresAt = sessionExpiresAt(now.getTime());

        const session = AuthSessionTable.create({
            token: crypto.randomBytes(32).toString('hex'),
            userId: user.id,
            createdAt: now.toISOString(),
            expiresAt,
            lastSeenAt: now.toISOString(),
        });
        await session.save();

        return {
            token: session.token,
            user: await this.withResourceAccess(toSafeUser(user)),
        };
    }

    static async getUserByToken(token?: string | null) {
        if (!token) {
            return null;
        }

        const session = await AuthSessionTable.findOne({
            where: {
                token,
                expiresAt: MoreThan(new Date().toISOString()),
            },
        });

        if (!session) {
            return null;
        }

        const nowMs = Date.now();
        const lastSeenAt = session.lastSeenAt
            ? new Date(session.lastSeenAt).getTime()
            : 0;
        if (nowMs - lastSeenAt >= LAST_SEEN_WRITE_INTERVAL_MS) {
            session.lastSeenAt = new Date(nowMs).toISOString();
            session.expiresAt = sessionExpiresAt(nowMs);
            await session.save();
        }

        const user = await AuthUserTable.findOne({
            where: { id: session.userId },
        });

        if (!user || user.disabled) return null;
        return this.withResourceAccess(toSafeUser(user));
    }

    static async getUserByUsername(username?: string | null) {
        const normalizedUsername = username?.trim();
        if (!normalizedUsername) {
            return null;
        }

        const user = await AuthUserTable.findOne({
            where: { username: normalizedUsername },
        });

        return user && !user.disabled ? toSafeUser(user) : null;
    }

    static async getUserById(userId?: string | null) {
        const normalizedUserId = userId?.trim();
        if (!normalizedUserId) {
            return null;
        }

        const user = await AuthUserTable.findOne({
            where: { id: normalizedUserId },
        });

        return user && !user.disabled
            ? this.withResourceAccess(toSafeUser(user))
            : null;
    }

    static async logout(token?: string | null) {
        if (!token) {
            return;
        }

        await AuthSessionTable.delete({ token });
    }

    static async changePassword(
        userId: string,
        currentPassword: string,
        newPassword: string,
    ) {
        const user = await AuthUserTable.findOne({
            where: { id: userId },
        });

        if (!user) {
            throw new Error('auth.error.userNotFound');
        }

        const passwordOk = passwordMatches(currentPassword, user);

        if (!passwordOk) {
            throw new Error('auth.error.currentPasswordIncorrect');
        }

        const salt = crypto.randomBytes(16).toString('hex');
        user.passwordSalt = salt;
        user.passwordHash = hashPassword(newPassword, salt);
        user.mustChangePassword = false;
        await user.save();

        await AuthSessionTable.delete({ userId });
    }

    private static async assertCanModifyUser(
        targetUserId: string,
        actorUserId: string,
        options: { allowSelf?: boolean; changingRoleTo?: UserRole | null } = {},
    ) {
        if (!options.allowSelf && targetUserId === actorUserId) {
            throw new Error('auth.error.cannotModifySelf');
        }

        const targetUser = await AuthUserTable.findOne({
            where: { id: targetUserId },
        });

        if (!targetUser) {
            throw new Error('auth.error.userNotFound');
        }

        const wouldRemoveAdmin =
            targetUser.role === UserRole.ADMIN &&
            (options.changingRoleTo === UserRole.USER ||
                options.changingRoleTo === null);

        if (wouldRemoveAdmin) {
            const adminCount = await AuthUserTable.count({
                where: { role: UserRole.ADMIN },
            });
            if (adminCount <= 1) {
                throw new Error('auth.error.cannotRemoveLastAdmin');
            }
        }

        return targetUser;
    }

    static async updateUserRole(
        userId: string,
        role: UserRole,
        actorUserId: string,
    ): Promise<SafeAuthUser> {
        const user = await this.assertCanModifyUser(userId, actorUserId, {
            changingRoleTo: role,
        });

        if (role === UserRole.ADMIN) {
            await ResourceAccessService.removeUserAccess(userId);
        }

        user.role = role;
        await user.save();

        if (role !== UserRole.ADMIN) {
            await AuthSessionTable.delete({ userId });
            await ResourceAccessService.ensureDefaultGroup();
        }

        return toSafeUser(user);
    }

    static async setUserDisabled(
        userId: string,
        disabled: boolean,
        actorUserId: string,
    ): Promise<SafeAuthUser> {
        const user = await this.assertCanModifyUser(userId, actorUserId, {
            changingRoleTo: disabled ? UserRole.USER : undefined,
        });

        user.disabled = disabled;
        await user.save();

        if (disabled) {
            await AuthSessionTable.delete({ userId });
        }

        return toSafeUser(user);
    }

    static async resetUserPassword(userId: string, newPassword: string) {
        const user = await AuthUserTable.findOne({
            where: { id: userId },
        });

        if (!user) {
            throw new Error('auth.error.userNotFound');
        }

        const salt = crypto.randomBytes(16).toString('hex');
        user.passwordSalt = salt;
        user.passwordHash = hashPassword(newPassword, salt);
        user.mustChangePassword = false;
        await user.save();

        await AuthSessionTable.delete({ userId });
    }

    static async revokeUserSessions(userId: string, actorUserId: string) {
        if (userId === actorUserId) {
            throw new Error('auth.error.useLogoutForSelf');
        }

        const user = await AuthUserTable.findOne({
            where: { id: userId },
        });

        if (!user) {
            throw new Error('auth.error.userNotFound');
        }

        await AuthSessionTable.delete({ userId });
    }

    static async deleteUser(userId: string, actorUserId: string) {
        await this.assertCanModifyUser(userId, actorUserId, {
            changingRoleTo: null,
        });

        await ResourceAccessService.removeUserAccess(userId);
        await AuthSessionTable.delete({ userId });
        await AuthUserTable.delete({ id: userId });
    }

    static async listUsers(): Promise<SafeAuthUser[]> {
        const users = await AuthUserTable.find({
            order: { createdAt: 'ASC' },
        });
        const [groups, members, nodeAssignments] = await Promise.all([
            UserGroupTable.find(),
            UserGroupMemberTable.find(),
            GroupNodeAssignmentTable.find(),
        ]);
        const groupsById = new Map(groups.map((group) => [group.id, group]));
        const nodeByGroupId = new Map(
            nodeAssignments.map((assignment) => [
                assignment.groupId,
                assignment.nodeId,
            ]),
        );
        const groupByUserId = new Map<
            string,
            {
                id: string;
                name: string;
                defaultContainerName: string;
                defaultEvaluateContainerName: string;
                defaultGrpoContainerName: string;
                defaultMultinodeContainerName: string;
                nodeId?: string | null;
            }
        >();
        members.forEach((member) => {
            const group = groupsById.get(member.groupId);
            if (!group) return;
            groupByUserId.set(member.userId, {
                id: group.id,
                name: group.name,
                defaultContainerName: group.defaultContainerName,
                defaultEvaluateContainerName:
                    group.defaultEvaluateContainerName,
                defaultGrpoContainerName: group.defaultGrpoContainerName,
                defaultMultinodeContainerName: group.defaultMultinodeContainerName,
                nodeId: nodeByGroupId.get(group.id) ?? null,
            });
        });
        const onlineSince = new Date(Date.now() - ONLINE_WINDOW_MS).toISOString();
        const onlineSessions = await AuthSessionTable.find({
            where: {
                expiresAt: MoreThan(new Date().toISOString()),
                lastSeenAt: MoreThan(onlineSince),
            },
        });
        const onlineUserIds = new Set(
            onlineSessions.map((session) => session.userId),
        );

        return users.map((user) => ({
            ...toSafeUser(user),
            isOnline: !user.disabled && onlineUserIds.has(user.id),
            assignedNodeId:
                user.role === UserRole.USER
                    ? groupByUserId.get(user.id)?.nodeId ?? null
                    : null,
            group:
                user.role === UserRole.USER
                    ? groupByUserId.get(user.id) ?? null
                    : null,
        }));
    }
}

