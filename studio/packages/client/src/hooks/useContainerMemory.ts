import { useState, useCallback, useEffect } from 'react';
import {
    getCachedDefaultContainerName,
    useEnvironmentConfig,
} from './useEnvironmentConfig';
import { DEFAULT_ENVIRONMENT_CONFIG } from '@shared/config/environment';

const MAX_HISTORY = 5;

export interface UseContainerMemoryReturn {
    /** 当前选中的容器名 */
    containerName: string;
    /** 设置容器名（会自动保存到历史记录） */
    setContainerName: (name: string) => void;
    /** 历史记录列表（最多5个，按最近使用排序） */
    history: string[];
    /** 清除历史记录 */
    clearHistory: () => void;
}

/**
 * 管理容器名称的自定义Hook
 * - 自动从localStorage读取上次使用的容器名
 * - 保存最近使用的5个容器名到历史记录
 * - 默认值来自统一环境配置
 * - 支持 namespace 参数，不同管理互不干扰
 * 
 * @param namespace - 命名空间，如 'dataset', 'model', 'evaluation'
 */
export function useContainerMemory(
    namespace: string,
    defaultContainerOverride?: string,
): UseContainerMemoryReturn {
    const storageKey = `${namespace}-containers`;
    const { defaultContainerName } = useEnvironmentConfig();
    const effectiveDefaultContainerName =
        defaultContainerOverride?.trim() || defaultContainerName;

    // 读取历史记录
    const [history, setHistory] = useState<string[]>(() => {
        try {
            const stored = localStorage.getItem(storageKey);
            if (stored) {
                const parsed = JSON.parse(stored);
                if (Array.isArray(parsed) && parsed.length > 0) {
                    return parsed;
                }
            }
        } catch (e) {
            console.error(`Failed to parse ${namespace} container history:`, e);
        }
        return [defaultContainerOverride?.trim() || getCachedDefaultContainerName()];
    });

    // 当前选中的容器名（默认使用历史记录的第一个）
    const [containerName, setContainerNameState] = useState<string>(() => {
        return history[0] || defaultContainerOverride?.trim() || getCachedDefaultContainerName();
    });

    useEffect(() => {
        const fallbackDefault = getCachedDefaultContainerName();
        setHistory((prev) => {
            if (prev.length === 0) {
                return [effectiveDefaultContainerName];
            }
            if (
                prev.length === 1 &&
                (prev[0] === fallbackDefault ||
                    prev[0] === DEFAULT_ENVIRONMENT_CONFIG.defaultContainerName) &&
                prev[0] !== effectiveDefaultContainerName
            ) {
                return [effectiveDefaultContainerName];
            }
            return prev;
        });
        setContainerNameState((prev) =>
            !prev ||
            prev === fallbackDefault ||
            prev === DEFAULT_ENVIRONMENT_CONFIG.defaultContainerName
                ? effectiveDefaultContainerName
                : prev,
        );
    }, [effectiveDefaultContainerName]);

    // 保存历史记录到localStorage
    useEffect(() => {
        try {
            localStorage.setItem(storageKey, JSON.stringify(history));
        } catch (e) {
            console.error(`Failed to save ${namespace} container history:`, e);
        }
    }, [history, storageKey]);

    // 设置容器名并更新历史记录
    const setContainerName = useCallback((name: string) => {
        const trimmedName = name.trim();
        if (!trimmedName) return;

        setContainerNameState(trimmedName);

        // 更新历史记录：将新名称放到最前面，去重，限制数量
        setHistory(prev => {
            const filtered = prev.filter(h => h !== trimmedName);
            return [trimmedName, ...filtered].slice(0, MAX_HISTORY);
        });
    }, []);

    // 清除历史记录
    const clearHistory = useCallback(() => {
        setHistory([effectiveDefaultContainerName]);
        setContainerNameState(effectiveDefaultContainerName);
        try {
            localStorage.removeItem(storageKey);
        } catch (e) {
            console.error(`Failed to clear ${namespace} container history:`, e);
        }
    }, [effectiveDefaultContainerName, namespace, storageKey]);

    return {
        containerName,
        setContainerName,
        history,
        clearHistory,
    };
}
