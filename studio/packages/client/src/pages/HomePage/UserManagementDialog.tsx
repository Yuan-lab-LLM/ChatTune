import { FormEvent, useEffect, useMemo, useState } from 'react';
import {
    KeyRoundIcon,
    LogOutIcon,
    MoreHorizontalIcon,
    SearchIcon,
    ShieldCheckIcon,
    Trash2Icon,
    UserIcon,
    UserPlusIcon,
} from 'lucide-react';
import { useTranslation } from 'react-i18next';

import { trpc } from '@/api/trpc';
import { Button } from '@/components/ui/button';
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogHeader,
    DialogTitle,
} from '@/components/ui/dialog';
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuSeparator,
    DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Input } from '@/components/ui/input';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { useAuth } from '@/context/AuthContext';
import { translateAuthError } from '@/utils/authErrors';
import AdminResourcePanel from './AdminResourcePanel';
import ResourceAuditPanel from './ResourceAuditPanel';
import ResourceSharingPanel from './ResourceSharingPanel';
import TrainingResourcePanel from './TrainingResourcePanel';

const secondaryActionClassName =
    'border-sky-200 bg-sky-50 text-sky-700 shadow-xs hover:border-sky-300 hover:bg-sky-100 hover:text-sky-800 dark:border-sky-400/25 dark:bg-sky-400/10 dark:text-sky-300 dark:hover:bg-sky-400/15';

type RoleFilter = 'all' | 'user' | 'admin';

interface UserManagementDialogProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
}

const generatePassword = () => {
    const alphabet =
        'ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789_-';
    const bytes = new Uint8Array(14);
    window.crypto.getRandomValues(bytes);
    return Array.from(bytes, (byte) => alphabet[byte % alphabet.length]).join(
        '',
    );
};

const formatDateTime = (value?: string | null) => {
    if (!value) return '-';
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return '-';
    return date.toLocaleString();
};

const UserManagementDialog = ({
    open,
    onOpenChange,
}: UserManagementDialogProps) => {
    const { t } = useTranslation();
    const { user: currentUser } = useAuth();
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [role, setRole] = useState<'user' | 'admin'>('user');
    const [groupId, setGroupId] = useState('');
    const [keyword, setKeyword] = useState('');
    const [roleFilter, setRoleFilter] = useState<RoleFilter>('all');
    const [groupFilter, setGroupFilter] = useState('all');
    const [error, setError] = useState('');
    const [success, setSuccess] = useState('');
    const [resetPasswordUser, setResetPasswordUser] = useState<{
        id: string;
        username: string;
    } | null>(null);
    const [deleteUser, setDeleteUser] = useState<{
        id: string;
        username: string;
        role: 'user' | 'admin';
        groupName: string;
        isOnline?: boolean;
    } | null>(null);
    const [resetPasswordValue, setResetPasswordValue] = useState('');
    const [resetPasswordError, setResetPasswordError] = useState('');
    const [roleDrafts, setRoleDrafts] = useState<
        Record<string, 'user' | 'admin'>
    >({});
    const [actionKey, setActionKey] = useState<string | null>(null);

    const usersQuery = trpc.listUsers.useQuery(undefined, {
        enabled: open,
        refetchInterval: open ? 30_000 : false,
        refetchIntervalInBackground: true,
    });
    const groupsQuery = trpc.listResourceGroups.useQuery(undefined, {
        enabled: open,
    });
    const createUserMutation = trpc.createUser.useMutation({
        onSuccess: async () => {
            setUsername('');
            setPassword('');
            setRole('user');
            setGroupId('');
            setSuccess(t('auth.createUserSuccess'));
            await Promise.all([usersQuery.refetch(), groupsQuery.refetch()]);
        },
    });
    const updateUserRoleMutation = trpc.updateUserRole.useMutation({
        onSuccess: () => Promise.all([usersQuery.refetch(), groupsQuery.refetch()]),
    });
    const resetUserPasswordMutation = trpc.resetUserPassword.useMutation({
        onSuccess: () => usersQuery.refetch(),
    });
    const setUserDisabledMutation = trpc.setUserDisabled.useMutation({
        onSuccess: () => usersQuery.refetch(),
    });
    const revokeUserSessionsMutation = trpc.revokeUserSessions.useMutation({
        onSuccess: () => usersQuery.refetch(),
    });
    const deleteUserMutation = trpc.deleteUser.useMutation({
        onSuccess: () => Promise.all([usersQuery.refetch(), groupsQuery.refetch()]),
    });
    const moveUserMutation = trpc.moveUserToResourceGroup.useMutation({
        onSuccess: async () => {
            await Promise.all([usersQuery.refetch(), groupsQuery.refetch()]);
        },
    });

    useEffect(() => {
        if (open) {
            setError('');
            setSuccess('');
        }
    }, [open]);

    const users = useMemo(
        () => usersQuery.data?.data ?? [],
        [usersQuery.data?.data],
    );
    const groups = groupsQuery.data?.data || [];
    const getGroupDisplayName = (group: { id: string; name: string }) =>
        group.id === 'default-users'
            ? t('resourceAccess.defaultGroupName')
            : group.name;
    const userGroupMap = useMemo(() => {
        const map = new Map<string, { id: string; name: string }>();
        users.forEach((authUser) => {
            if (authUser.group) {
                map.set(authUser.id, {
                    id: authUser.group.id,
                    name:
                        authUser.group.id === 'default-users'
                            ? t('resourceAccess.defaultGroupName')
                            : authUser.group.name,
                });
            }
        });
        groups.forEach((group) => {
            group.members.forEach((member) => {
                map.set(member.userId, {
                    id: group.id,
                    name: getGroupDisplayName(group),
                });
            });
        });
        return map;
    }, [groups, t, users]);

    const filteredUsers = users.filter((authUser) => {
        const query = keyword.trim().toLowerCase();
        const assignedGroup = userGroupMap.get(authUser.id);
        const matchesKeyword =
            !query ||
            authUser.username.toLowerCase().includes(query) ||
            assignedGroup?.name.toLowerCase().includes(query);
        const matchesRole =
            roleFilter === 'all' || authUser.role === roleFilter;
        const matchesGroup =
            groupFilter === 'all' || assignedGroup?.id === groupFilter;
        return matchesKeyword && matchesRole && matchesGroup;
    });

    useEffect(() => {
        setRoleDrafts((currentDrafts) => {
            const nextDrafts: Record<string, 'user' | 'admin'> = {};
            users.forEach((authUser) => {
                nextDrafts[authUser.id] =
                    currentDrafts[authUser.id] ?? authUser.role;
            });
            return nextDrafts;
        });
    }, [users]);

    useEffect(() => {
        if (open && role === 'user' && !groupId && groups[0]) {
            setGroupId(groups[0].id);
        }
    }, [groups, groupId, open, role]);

    const runAdminAction = async (
        key: string,
        action: () => Promise<unknown>,
        successMessage: string,
    ) => {
        setError('');
        setSuccess('');
        setActionKey(key);
        try {
            await action();
            setSuccess(successMessage);
            return true;
        } catch (actionError) {
            setError(
                translateAuthError(actionError, t, 'auth.adminActionFailed'),
            );
            return false;
        } finally {
            setActionKey(null);
        }
    };

    const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        setError('');
        setSuccess('');

        try {
            await createUserMutation.mutateAsync({
                username,
                password,
                role,
                groupId: role === 'user' ? groupId : undefined,
            });
        } catch (createError) {
            setError(
                translateAuthError(createError, t, 'auth.createUserFailed'),
            );
        }
    };

    const submitPasswordReset = async (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        if (!resetPasswordUser) return;
        setResetPasswordError('');
        setSuccess('');
        try {
            await resetUserPasswordMutation.mutateAsync({
                userId: resetPasswordUser.id,
                newPassword: resetPasswordValue,
            });
            setSuccess(t('auth.userPasswordReset'));
            setResetPasswordUser(null);
            setResetPasswordValue('');
        } catch (resetError) {
            setResetPasswordError(
                translateAuthError(resetError, t, 'auth.adminActionFailed'),
            );
        }
    };

    const confirmDeleteUser = async () => {
        if (!deleteUser) return;
        const deleted = await runAdminAction(
            `delete:${deleteUser.id}`,
            () => deleteUserMutation.mutateAsync({ userId: deleteUser.id }),
            t('auth.userDeleted'),
        );
        if (deleted) setDeleteUser(null);
    };

    const normalUserGroupUnavailable =
        role === 'user' &&
        (groupsQuery.isLoading || groups.length === 0 || !groupId);

    return (
        <Dialog open={open} onOpenChange={onOpenChange}>
            <DialogContent className="max-h-[90vh] overflow-y-auto bg-slate-50/95 sm:max-w-[1180px]">
                <DialogHeader>
                    <DialogTitle className="flex items-center gap-2">
                        <UserPlusIcon className="size-5" />
                        {t('auth.userManagement')}
                    </DialogTitle>
                    <DialogDescription>
                        {t('auth.userManagementDescription')}
                    </DialogDescription>
                </DialogHeader>

                <Tabs defaultValue="users" className="space-y-4">
                    <TabsList className="grid h-auto w-full grid-cols-5">
                        <TabsTrigger value="users">
                            {t('auth.usersTab')}
                        </TabsTrigger>
                        <TabsTrigger value="groups">
                            {t('resourceAccess.groupsTab')}
                        </TabsTrigger>
                        <TabsTrigger value="sharing">
                            {t('resourceAccess.sharingTab')}
                        </TabsTrigger>
                        <TabsTrigger value="audit">
                            {t('resourceAccess.auditTab')}
                        </TabsTrigger>
                        <TabsTrigger value="training">
                            {t('trainingResource.trainingTab')}
                        </TabsTrigger>
                    </TabsList>
                    <TabsContent value="users" className="space-y-4">
                        {(error || success) && (
                            <div
                                className={error ? 'auth-error' : 'auth-success'}
                                role="status"
                            >
                                {error || success}
                            </div>
                        )}

                        <section className="rounded-lg border border-border/50 bg-card/95 p-4 shadow-sm">
                            <div className="mb-4 flex items-center gap-2">
                                <UserPlusIcon className="size-4 text-primary" />
                                <h3 className="text-base font-semibold">
                                    {t('auth.createUser')}
                                </h3>
                            </div>
                            <form
                                className="grid gap-3 rounded-lg border border-border/40 bg-muted/20 p-3 md:grid-cols-2 lg:grid-cols-[1fr_1fr_1fr_auto]"
                                onSubmit={handleSubmit}
                            >
                                <label className="space-y-1.5">
                                    <span className="text-xs font-medium text-muted-foreground">
                                        {t('auth.username')}
                                    </span>
                                    <Input
                                        value={username}
                                        onChange={(event) =>
                                            setUsername(event.target.value)
                                        }
                                        required
                                        minLength={3}
                                        maxLength={32}
                                    />
                                </label>
                                <label className="space-y-1.5">
                                    <span className="text-xs font-medium text-muted-foreground">
                                        {t('auth.initialPassword')}
                                    </span>
                                    <div className="flex gap-2">
                                        <Input
                                            value={password}
                                            onChange={(event) =>
                                                setPassword(event.target.value)
                                            }
                                            type="password"
                                            required
                                            minLength={6}
                                        />
                                        <Button
                                            type="button"
                                            variant="outline"
                                            className="shrink-0"
                                            onClick={() =>
                                                setPassword(generatePassword())
                                            }
                                        >
                                            {t('auth.generatePassword')}
                                        </Button>
                                    </div>
                                </label>
                                <label className="space-y-1.5">
                                    <span className="text-xs font-medium text-muted-foreground">
                                        {t('auth.role')}
                                    </span>
                                    <select
                                        className="h-9 w-full rounded-md border border-input bg-background px-3 text-sm outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
                                        value={role}
                                        onChange={(event) =>
                                            setRole(
                                                event.target.value as
                                                    | 'user'
                                                    | 'admin',
                                            )
                                        }
                                    >
                                        <option value="user">
                                            {t('auth.normalUser')}
                                        </option>
                                        <option value="admin">
                                            {t('auth.adminUser')}
                                        </option>
                                    </select>
                                </label>
                                {role === 'user' && (
                                    <label className="space-y-1.5 md:col-span-2 lg:col-span-1">
                                        <span className="text-xs font-medium text-muted-foreground">
                                            {t('resourceAccess.userGroup')}
                                        </span>
                                        <select
                                            className="h-9 w-full rounded-md border border-input bg-background px-3 text-sm outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
                                            value={groupId}
                                            onChange={(event) =>
                                                setGroupId(event.target.value)
                                            }
                                            required
                                            disabled={groupsQuery.isLoading}
                                        >
                                            <option value="">
                                                {groupsQuery.isLoading
                                                    ? t('auth.loadingGroups')
                                                    : t(
                                                          'resourceAccess.selectGroup',
                                                      )}
                                            </option>
                                            {groups.map((group) => (
                                                <option
                                                    key={group.id}
                                                    value={group.id}
                                                >
                                                    {getGroupDisplayName(group)}
                                                </option>
                                            ))}
                                        </select>
                                    </label>
                                )}
                                <div className="self-end md:col-span-2 lg:col-span-1">
                                    <Button
                                        type="submit"
                                        disabled={
                                            createUserMutation.isPending ||
                                            normalUserGroupUnavailable
                                        }
                                    >
                                        <UserPlusIcon className="size-4" />
                                        {createUserMutation.isPending
                                            ? t('auth.creatingUser')
                                            : t('auth.createUser')}
                                    </Button>
                                </div>
                            </form>
                            {normalUserGroupUnavailable && (
                                <p className="mt-2 text-xs text-muted-foreground">
                                    {t('auth.createUserGroupHint')}
                                </p>
                            )}
                        </section>

                        <section className="rounded-lg border border-border/50 bg-card/95 p-4 shadow-sm">
                            <div className="mb-4 flex flex-col gap-3 xl:flex-row xl:items-center xl:justify-between">
                                <div className="flex items-center gap-2">
                                    <UserIcon className="size-4 text-primary" />
                                    <h3 className="text-base font-semibold">
                                        {t('auth.currentUsers')}
                                    </h3>
                                    <span className="text-xs text-muted-foreground">
                                        {t('auth.userCount', {
                                            count: users.length,
                                        })}
                                    </span>
                                </div>
                                <div className="user-management-filters">
                                    <div className="auth-input user-management-search rounded-lg border border-border/40 bg-background">
                                        <SearchIcon className="size-4" />
                                        <Input
                                            value={keyword}
                                            onChange={(event) =>
                                                setKeyword(event.target.value)
                                            }
                                            placeholder={t('auth.searchUsers')}
                                        />
                                    </div>
                                    <select
                                        value={roleFilter}
                                        onChange={(event) =>
                                            setRoleFilter(
                                                event.target
                                                    .value as RoleFilter,
                                            )
                                        }
                                    >
                                        <option value="all">
                                            {t('auth.allRoles')}
                                        </option>
                                        <option value="user">
                                            {t('auth.normalUser')}
                                        </option>
                                        <option value="admin">
                                            {t('auth.adminUser')}
                                        </option>
                                    </select>
                                    <select
                                        value={groupFilter}
                                        onChange={(event) =>
                                            setGroupFilter(event.target.value)
                                        }
                                    >
                                        <option value="all">
                                            {t('auth.allGroups')}
                                        </option>
                                        {groups.map((group) => (
                                            <option
                                                key={group.id}
                                                value={group.id}
                                            >
                                                {getGroupDisplayName(group)}
                                            </option>
                                        ))}
                                    </select>
                                </div>
                            </div>

                            <div className="user-management-table">
                                <div className="user-management-table-head">
                                    <span>{t('auth.username')}</span>
                                    <span>{t('auth.role')}</span>
                                    <span>{t('resourceAccess.userGroup')}</span>
                                    <span>{t('auth.createdAt')}</span>
                                    <span>{t('auth.createdBy')}</span>
                                    <span>{t('auth.actions')}</span>
                                </div>
                                {usersQuery.isLoading && (
                                    <div className="user-management-empty">
                                        {t('auth.loadingUsers')}
                                    </div>
                                )}
                                {!usersQuery.isLoading && users.length === 0 && (
                                    <div className="user-management-empty">
                                        {t('auth.noUsers')}
                                    </div>
                                )}
                                {!usersQuery.isLoading &&
                                    users.length > 0 &&
                                    filteredUsers.length === 0 && (
                                        <div className="user-management-empty">
                                            {t('auth.noMatchedUsers')}
                                        </div>
                                    )}
                                {filteredUsers.map((authUser) => {
                                    const currentGroup =
                                        userGroupMap.get(authUser.id);
                                    const isSelf =
                                        authUser.id === currentUser?.id;
                                    const roleDraft =
                                        roleDrafts[authUser.id] ??
                                        authUser.role;

                                    return (
                                        <div
                                            className="user-management-table-row"
                                            key={authUser.id}
                                        >
                                            <div className="user-cell-identity">
                                                <div className="user-row-avatar">
                                                    {authUser.role ===
                                                    'admin' ? (
                                                        <ShieldCheckIcon className="size-4" />
                                                    ) : (
                                                        <UserIcon className="size-4" />
                                                    )}
                                                </div>
                                                <div className="min-w-0">
                                                    <div className="user-row-name">
                                                        {authUser.username}
                                                        <span
                                                            aria-label={t(
                                                                authUser.isOnline
                                                                    ? 'auth.onlineStatus'
                                                                    : 'auth.offlineStatus',
                                                            )}
                                                            className={`user-online-dot ${
                                                                authUser.isOnline
                                                                    ? 'is-online'
                                                                    : 'is-offline'
                                                            }`}
                                                            role="status"
                                                            title={t(
                                                                authUser.isOnline
                                                                    ? 'auth.onlineStatus'
                                                                    : 'auth.offlineStatus',
                                                            )}
                                                        />
                                                    </div>
                                                    {isSelf && (
                                                        <div className="text-xs text-muted-foreground">
                                                            {t(
                                                                'auth.currentAccount',
                                                            )}
                                                        </div>
                                                    )}
                                                </div>
                                            </div>
                                            <div className="user-role-editor">
                                                <ShieldCheckIcon className="size-4" />
                                                <select
                                                    aria-label={t(
                                                        'auth.roleForUser',
                                                        {
                                                            username:
                                                                authUser.username,
                                                        },
                                                    )}
                                                    value={roleDraft}
                                                    disabled={isSelf}
                                                    onChange={(event) =>
                                                        setRoleDrafts(
                                                            (currentDrafts) => ({
                                                                ...currentDrafts,
                                                                [authUser.id]:
                                                                    event.target
                                                                        .value as
                                                                        | 'user'
                                                                        | 'admin',
                                                            }),
                                                        )
                                                    }
                                                >
                                                    <option value="user">
                                                        {t('auth.normalUser')}
                                                    </option>
                                                    <option value="admin">
                                                        {t('auth.adminUser')}
                                                    </option>
                                                </select>
                                            </div>
                                            <div>
                                                {authUser.role === 'user' ? (
                                                    <select
                                                        aria-label={t(
                                                            'resourceAccess.userGroup',
                                                        )}
                                                        value={
                                                            currentGroup?.id ||
                                                            ''
                                                        }
                                                        disabled={
                                                            actionKey ===
                                                            `group:${authUser.id}`
                                                        }
                                                        onChange={(event) =>
                                                            void runAdminAction(
                                                                `group:${authUser.id}`,
                                                                () =>
                                                                    moveUserMutation.mutateAsync(
                                                                        {
                                                                            userId:
                                                                                authUser.id,
                                                                            groupId:
                                                                                event
                                                                                    .target
                                                                                    .value,
                                                                        },
                                                                    ),
                                                                t(
                                                                    'resourceAccess.userGroupUpdated',
                                                                ),
                                                            )
                                                        }
                                                    >
                                                        <option value="">
                                                            {t(
                                                                'resourceAccess.selectGroup',
                                                            )}
                                                        </option>
                                                        {groups.map((group) => (
                                                            <option
                                                                key={group.id}
                                                                value={group.id}
                                                            >
                                                                {getGroupDisplayName(
                                                                    group,
                                                                )}
                                                            </option>
                                                        ))}
                                                    </select>
                                                ) : (
                                                    <span className="text-xs text-muted-foreground">
                                                        {t('auth.noGroupForAdmin')}
                                                    </span>
                                                )}
                                            </div>
                                            <div className="text-xs text-muted-foreground">
                                                {formatDateTime(
                                                    authUser.createdAt,
                                                )}
                                            </div>
                                            <div className="truncate text-xs text-muted-foreground">
                                                {authUser.createdBy || '-'}
                                            </div>
                                            <div className="user-row-actions">
                                                <Button
                                                    type="button"
                                                    variant="outline"
                                                    className={
                                                        secondaryActionClassName
                                                    }
                                                    disabled={
                                                        isSelf ||
                                                        roleDraft ===
                                                            authUser.role ||
                                                        actionKey ===
                                                            `role:${authUser.id}`
                                                    }
                                                    onClick={() =>
                                                        runAdminAction(
                                                            `role:${authUser.id}`,
                                                            () =>
                                                                updateUserRoleMutation.mutateAsync(
                                                                    {
                                                                        userId:
                                                                            authUser.id,
                                                                        role: roleDraft,
                                                                    },
                                                                ),
                                                            t(
                                                                'auth.userRoleUpdated',
                                                            ),
                                                        )
                                                    }
                                                >
                                                    {t('auth.saveRole')}
                                                </Button>
                                                <DropdownMenu>
                                                    <DropdownMenuTrigger
                                                        asChild
                                                    >
                                                        <Button
                                                            type="button"
                                                            variant="outline"
                                                            size="icon"
                                                        >
                                                            <MoreHorizontalIcon className="size-4" />
                                                            <span className="sr-only">
                                                                {t(
                                                                    'auth.moreActions',
                                                                )}
                                                            </span>
                                                        </Button>
                                                    </DropdownMenuTrigger>
                                                    <DropdownMenuContent align="end">
                                                        {!isSelf && (
                                                            <DropdownMenuItem
                                                                onSelect={() => {
                                                                    setResetPasswordUser(
                                                                        {
                                                                            id: authUser.id,
                                                                            username:
                                                                                authUser.username,
                                                                        },
                                                                    );
                                                                    setResetPasswordValue(
                                                                        '',
                                                                    );
                                                                    setResetPasswordError(
                                                                        '',
                                                                    );
                                                                }}
                                                            >
                                                                <KeyRoundIcon className="size-4" />
                                                                {t(
                                                                    'auth.resetPassword',
                                                                )}
                                                            </DropdownMenuItem>
                                                        )}
                                                        <DropdownMenuItem
                                                            disabled={isSelf}
                                                            onSelect={() =>
                                                                void runAdminAction(
                                                                    `disabled:${authUser.id}`,
                                                                    () =>
                                                                        setUserDisabledMutation.mutateAsync(
                                                                            {
                                                                                userId:
                                                                                    authUser.id,
                                                                                disabled:
                                                                                    !authUser.disabled,
                                                                            },
                                                                        ),
                                                                    authUser.disabled
                                                                        ? t(
                                                                              'auth.userEnabled',
                                                                          )
                                                                        : t(
                                                                              'auth.userDisabled',
                                                                          ),
                                                                )
                                                            }
                                                        >
                                                            <UserIcon className="size-4" />
                                                            {authUser.disabled
                                                                ? t(
                                                                      'auth.enableUser',
                                                                  )
                                                                : t(
                                                                      'auth.disableUser',
                                                                  )}
                                                        </DropdownMenuItem>
                                                        <DropdownMenuItem
                                                            disabled={isSelf}
                                                            onSelect={() =>
                                                                void runAdminAction(
                                                                    `sessions:${authUser.id}`,
                                                                    () =>
                                                                        revokeUserSessionsMutation.mutateAsync(
                                                                            {
                                                                                userId:
                                                                                    authUser.id,
                                                                            },
                                                                        ),
                                                                    t(
                                                                        'auth.userSessionsRevoked',
                                                                    ),
                                                                )
                                                            }
                                                        >
                                                            <LogOutIcon className="size-4" />
                                                            {t(
                                                                'auth.revokeSessions',
                                                            )}
                                                        </DropdownMenuItem>
                                                        <DropdownMenuSeparator />
                                                        <DropdownMenuItem
                                                            disabled={isSelf}
                                                            variant="destructive"
                                                            onSelect={() =>
                                                                setDeleteUser({
                                                                    id: authUser.id,
                                                                    username:
                                                                        authUser.username,
                                                                    role: authUser.role,
                                                                    groupName:
                                                                        currentGroup?.name ||
                                                                        t(
                                                                            'auth.noGroupForAdmin',
                                                                        ),
                                                                    isOnline:
                                                                        authUser.isOnline,
                                                                })
                                                            }
                                                        >
                                                            <Trash2Icon className="size-4" />
                                                            {t(
                                                                'auth.deleteUser',
                                                            )}
                                                        </DropdownMenuItem>
                                                    </DropdownMenuContent>
                                                </DropdownMenu>
                                            </div>
                                        </div>
                                    );
                                })}
                            </div>
                        </section>
                    </TabsContent>
                    <TabsContent value="groups">
                        <AdminResourcePanel />
                    </TabsContent>
                    <TabsContent value="sharing">
                        <ResourceSharingPanel />
                    </TabsContent>
                    <TabsContent value="audit">
                        <ResourceAuditPanel />
                    </TabsContent>
                    <TabsContent value="training">
                        <TrainingResourcePanel />
                    </TabsContent>
                </Tabs>

                {resetPasswordUser && (
                    <Dialog
                        open={Boolean(resetPasswordUser)}
                        onOpenChange={(nextOpen) => {
                            if (!nextOpen) {
                                setResetPasswordUser(null);
                                setResetPasswordValue('');
                                setResetPasswordError('');
                            }
                        }}
                    >
                        <DialogContent className="sm:max-w-md">
                            <DialogHeader>
                                <DialogTitle>
                                    {t('auth.resetPasswordTitle')}
                                </DialogTitle>
                                <DialogDescription>
                                    {t('auth.resetPasswordDescription', {
                                        username: resetPasswordUser.username,
                                    })}
                                </DialogDescription>
                            </DialogHeader>
                            <form
                                className="space-y-3"
                                onSubmit={submitPasswordReset}
                            >
                                <label className="grid gap-1 text-xs font-medium text-muted-foreground">
                                    {t('auth.newPassword')}
                                    <div className="flex gap-2">
                                        <Input
                                            type="password"
                                            value={resetPasswordValue}
                                            onChange={(event) =>
                                                setResetPasswordValue(
                                                    event.target.value,
                                                )
                                            }
                                            minLength={6}
                                            required
                                        />
                                        <Button
                                            type="button"
                                            variant="outline"
                                            onClick={() =>
                                                setResetPasswordValue(
                                                    generatePassword(),
                                                )
                                            }
                                        >
                                            {t('auth.generatePassword')}
                                        </Button>
                                    </div>
                                </label>
                                {resetPasswordError && (
                                    <div className="auth-error">
                                        {resetPasswordError}
                                    </div>
                                )}
                                <div className="flex justify-end gap-2">
                                    <Button
                                        type="button"
                                        variant="outline"
                                        onClick={() => {
                                            setResetPasswordUser(null);
                                            setResetPasswordValue('');
                                            setResetPasswordError('');
                                        }}
                                    >
                                        {t('common-cancel')}
                                    </Button>
                                    <Button
                                        type="submit"
                                        disabled={
                                            resetUserPasswordMutation.isPending
                                        }
                                    >
                                        {t('auth.resetPassword')}
                                    </Button>
                                </div>
                            </form>
                        </DialogContent>
                    </Dialog>
                )}

                {deleteUser && (
                    <Dialog
                        open={Boolean(deleteUser)}
                        onOpenChange={(nextOpen) => {
                            if (!nextOpen) setDeleteUser(null);
                        }}
                    >
                        <DialogContent className="sm:max-w-md">
                            <DialogHeader>
                                <DialogTitle>
                                    {t('auth.deleteUserTitle')}
                                </DialogTitle>
                                <DialogDescription>
                                    {t('auth.deleteUserDescription', {
                                        username: deleteUser.username,
                                    })}
                                </DialogDescription>
                            </DialogHeader>
                            <div className="rounded-lg border border-border/50 bg-muted/20 p-3 text-sm">
                                <div className="grid grid-cols-2 gap-2">
                                    <span className="text-muted-foreground">
                                        {t('auth.role')}
                                    </span>
                                    <span>
                                        {deleteUser.role === 'admin'
                                            ? t('auth.adminUser')
                                            : t('auth.normalUser')}
                                    </span>
                                    <span className="text-muted-foreground">
                                        {t('resourceAccess.userGroup')}
                                    </span>
                                    <span>{deleteUser.groupName}</span>
                                    <span className="text-muted-foreground">
                                        {t('auth.onlineStatus')}
                                    </span>
                                    <span>
                                        {deleteUser.isOnline
                                            ? t('auth.onlineStatus')
                                            : t('auth.offlineStatus')}
                                    </span>
                                </div>
                            </div>
                            <div className="flex justify-end gap-2">
                                <Button
                                    type="button"
                                    variant="outline"
                                    onClick={() => setDeleteUser(null)}
                                >
                                    {t('common-cancel')}
                                </Button>
                                <Button
                                    type="button"
                                    variant="destructive"
                                    disabled={
                                        actionKey === `delete:${deleteUser.id}`
                                    }
                                    onClick={() => void confirmDeleteUser()}
                                >
                                    <Trash2Icon className="size-4" />
                                    {t('auth.deleteUser')}
                                </Button>
                            </div>
                        </DialogContent>
                    </Dialog>
                )}
            </DialogContent>
        </Dialog>
    );
};

export default UserManagementDialog;
