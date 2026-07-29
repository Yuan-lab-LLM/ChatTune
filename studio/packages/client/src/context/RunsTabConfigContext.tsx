import { createContext, useCallback, useContext, useEffect, useState } from 'react';
import { RunsTabConfig, defaultRunsTabConfig } from '@shared/config/client';

const RUNS_TAB_CONFIG_STORAGE_KEY = 'runs-tab-config';

interface RunsTabConfigContextType {
    config: RunsTabConfig;
    updateConfig: (updates: Partial<RunsTabConfig>) => void;
    setConfig: (config: RunsTabConfig) => void;
}

const RunsTabConfigContext = createContext<RunsTabConfigContextType>({
    config: defaultRunsTabConfig,
    updateConfig: () => {},
    setConfig: () => {},
});

export function RunsTabConfigProvider({
    children,
}: {
    children: React.ReactNode;
}) {
    const [config, setConfigState] = useState<RunsTabConfig>(() => {
        // 从 localStorage 读取保存的配置
        if (typeof window !== 'undefined') {
            try {
                const savedConfig = localStorage.getItem(RUNS_TAB_CONFIG_STORAGE_KEY);
                if (savedConfig) {
                    const parsed = JSON.parse(savedConfig);
                    // 合并默认配置，确保新添加的字段有默认值
                    return { ...defaultRunsTabConfig, ...parsed };
                }
            } catch (e) {
                console.error('Failed to parse runs tab config:', e);
            }
        }
        return defaultRunsTabConfig;
    });

    // 保存配置到 localStorage
    useEffect(() => {
        if (typeof window !== 'undefined') {
            localStorage.setItem(RUNS_TAB_CONFIG_STORAGE_KEY, JSON.stringify(config));
        }
    }, [config]);

    const setConfig = useCallback((newConfig: RunsTabConfig) => {
        setConfigState(newConfig);
    }, []);

    const updateConfig = useCallback((updates: Partial<RunsTabConfig>) => {
        setConfigState((prev) => ({ ...prev, ...updates }));
    }, []);

    return (
        <RunsTabConfigContext.Provider value={{ config, updateConfig, setConfig }}>
            {children}
        </RunsTabConfigContext.Provider>
    );
}

export const useRunsTabConfig = () => useContext(RunsTabConfigContext);
