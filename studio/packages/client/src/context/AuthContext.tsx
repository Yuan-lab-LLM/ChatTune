import {
    createContext,
    useCallback,
    useContext,
    useEffect,
    useMemo,
    useState,
    type ReactNode,
} from 'react';
import { trpc, queryClient } from '@/api/trpc';

export type AuthRole = 'admin' | 'user';

export interface AuthUser {
    id: string;
    username: string;
    role: AuthRole;
    disabled?: boolean;
    mustChangePassword?: boolean;
    createdAt: string;
    createdBy?: string | null;
    assignedNodeId?: string | null;
    group?: {
        id: string;
        name: string;
        defaultContainerName: string;
        defaultEvaluateContainerName: string;
        defaultGrpoContainerName: string;
        defaultMultinodeContainerName: string;
    } | null;
}

interface AuthContextValue {
    user: AuthUser | null;
    isAuthenticated: boolean;
    isAdmin: boolean;
    isChecking: boolean;
    login: (username: string, password: string) => Promise<void>;
    logout: () => Promise<void>;
    forceLogout: () => void;
    changePassword: (
        currentPassword: string,
        newPassword: string,
    ) => Promise<void>;
}

const ENVIRONMENT_CONFIG_CACHE_KEY = 'medflow_environment_config';
const AuthContext = createContext<AuthContextValue | null>(null);

export const AuthProvider = ({ children }: { children: ReactNode }) => {
    const [user, setUser] = useState<AuthUser | null>(null);
    const [hasCheckedSession, setHasCheckedSession] = useState(false);

    const currentUserQuery = trpc.getCurrentUser.useQuery(undefined, {
        retry: false,
        refetchInterval: user ? 30_000 : false,
        refetchIntervalInBackground: true,
    });
    const loginMutation = trpc.login.useMutation();
    const logoutMutation = trpc.logout.useMutation();
    const changePasswordMutation = trpc.changePassword.useMutation();

    useEffect(() => {
        if (currentUserQuery.data?.data) {
            setUser(currentUserQuery.data.data as AuthUser);
            setHasCheckedSession(true);
            return;
        }

        if (currentUserQuery.isError) {
            setUser(null);
            setHasCheckedSession(true);
        }
    }, [currentUserQuery.data, currentUserQuery.isError]);

    useEffect(() => {
        if (!currentUserQuery.isLoading && !currentUserQuery.data) {
            setHasCheckedSession(true);
        }
    }, [currentUserQuery.data, currentUserQuery.isLoading]);

    const login = useCallback(
        async (username: string, password: string) => {
            const result = await loginMutation.mutateAsync({
                username,
                password,
            });
            const authData = result.data;

            localStorage.removeItem(ENVIRONMENT_CONFIG_CACHE_KEY);
            setUser(authData.user as AuthUser);
            setHasCheckedSession(true);
            await queryClient.invalidateQueries();
        },
        [loginMutation],
    );

    const logout = useCallback(async () => {
        try {
            await logoutMutation.mutateAsync();
        } finally {
            localStorage.removeItem(ENVIRONMENT_CONFIG_CACHE_KEY);
            setUser(null);
            setHasCheckedSession(true);
            queryClient.clear();
        }
    }, [logoutMutation]);

    const forceLogout = useCallback(() => {
        localStorage.removeItem(ENVIRONMENT_CONFIG_CACHE_KEY);
        setUser(null);
        setHasCheckedSession(true);
        queryClient.clear();
    }, []);

    const changePassword = useCallback(
        async (currentPassword: string, newPassword: string) => {
            await changePasswordMutation.mutateAsync({
                currentPassword,
                newPassword,
            });
            localStorage.removeItem(ENVIRONMENT_CONFIG_CACHE_KEY);
            setUser(null);
            setHasCheckedSession(true);
            queryClient.clear();
        },
        [changePasswordMutation],
    );

    const value = useMemo<AuthContextValue>(
        () => ({
            user,
            isAuthenticated: Boolean(user),
            isAdmin: user?.role === 'admin',
            isChecking: !hasCheckedSession && !user,
            login,
            logout,
            forceLogout,
            changePassword,
        }),
        [changePassword, forceLogout, hasCheckedSession, login, logout, user],
    );

    return (
        <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
    );
};

export const useAuth = () => {
    const context = useContext(AuthContext);

    if (!context) {
        throw new Error('useAuth must be used within AuthProvider');
    }

    return context;
};

