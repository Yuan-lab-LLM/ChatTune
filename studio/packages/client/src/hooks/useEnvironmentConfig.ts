import { useEffect, useMemo } from 'react';

import { trpc } from '@/api/trpc';
import { DEFAULT_ENVIRONMENT_CONFIG } from '@shared/config/environment';

const ENVIRONMENT_CONFIG_CACHE_KEY = 'medflow_environment_config';

interface CachedEnvironmentConfig {
    defaultContainerName?: string;
    defaultEvaluateContainerName?: string;
    defaultGrpoContainerName?: string;
    defaultMultinodeContainerName?: string;
}

const readCachedEnvironmentConfig = (): CachedEnvironmentConfig | null => {
    if (typeof window === 'undefined') {
        return null;
    }

    try {
        const raw = localStorage.getItem(ENVIRONMENT_CONFIG_CACHE_KEY);
        return raw ? JSON.parse(raw) : null;
    } catch {
        return null;
    }
};

export const getCachedDefaultContainerName = (): string => {
    return (
        readCachedEnvironmentConfig()?.defaultContainerName?.trim() ||
        DEFAULT_ENVIRONMENT_CONFIG.defaultContainerName
    );
};

export const getCachedDefaultEvaluateContainerName = (): string => {
    return (
        readCachedEnvironmentConfig()?.defaultEvaluateContainerName?.trim() ||
        readCachedEnvironmentConfig()?.defaultContainerName?.trim() ||
        DEFAULT_ENVIRONMENT_CONFIG.defaultEvaluateContainerName
    );
};

export const getCachedDefaultGrpoContainerName = (): string => {
    return (
        readCachedEnvironmentConfig()?.defaultGrpoContainerName?.trim() ||
        DEFAULT_ENVIRONMENT_CONFIG.defaultGrpoContainerName
    );
};

export const getCachedDefaultMultinodeContainerName = (): string => {
    return (
        readCachedEnvironmentConfig()?.defaultMultinodeContainerName?.trim() ||
        DEFAULT_ENVIRONMENT_CONFIG.defaultMultinodeContainerName
    );
};

export const useEnvironmentConfig = () => {
    const query = trpc.getEnvironmentConfig.useQuery(undefined, {
        staleTime: 0,
        gcTime: 5 * 60 * 1000,
        refetchOnMount: 'always',
        refetchOnWindowFocus: true,
        refetchInterval: 10 * 1000,
        retry: 1,
    });

    const defaultContainerName = useMemo(
        () =>
            query.data?.data?.defaultContainerName?.trim() ||
            getCachedDefaultContainerName(),
        [query.data?.data?.defaultContainerName],
    );

    const defaultEvaluateContainerName = useMemo(
        () =>
            query.data?.data?.defaultEvaluateContainerName?.trim() ||
            query.data?.data?.defaultContainerName?.trim() ||
            getCachedDefaultEvaluateContainerName(),
        [
            query.data?.data?.defaultEvaluateContainerName,
            query.data?.data?.defaultContainerName,
        ],
    );

    const defaultGrpoContainerName = useMemo(
        () =>
            query.data?.data?.defaultGrpoContainerName?.trim() ||
            getCachedDefaultGrpoContainerName(),
        [query.data?.data?.defaultGrpoContainerName],
    );

    const defaultMultinodeContainerName = useMemo(
        () =>
            query.data?.data?.defaultMultinodeContainerName?.trim() ||
            getCachedDefaultMultinodeContainerName(),
        [query.data?.data?.defaultMultinodeContainerName],
    );

    useEffect(() => {
        if (!query.data?.data?.defaultContainerName) {
            return;
        }

        localStorage.setItem(
            ENVIRONMENT_CONFIG_CACHE_KEY,
            JSON.stringify({
                defaultContainerName: query.data.data.defaultContainerName,
                defaultEvaluateContainerName:
                    query.data.data.defaultEvaluateContainerName,
                defaultGrpoContainerName:
                    query.data.data.defaultGrpoContainerName,
                defaultMultinodeContainerName:
                    query.data.data.defaultMultinodeContainerName,
            }),
        );
    }, [
        query.data?.data?.defaultContainerName,
        query.data?.data?.defaultEvaluateContainerName,
        query.data?.data?.defaultGrpoContainerName,
        query.data?.data?.defaultMultinodeContainerName,
    ]);

    return {
        defaultContainerName,
        defaultEvaluateContainerName,
        defaultGrpoContainerName,
        defaultMultinodeContainerName,
        isLoading: query.isLoading,
        error: query.error,
    };
};
