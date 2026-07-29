export const ClientConfig = {
    socketPath: '/client',
} as const;

/**
 * Configuration for Runs Tab display options
 */
export interface RunsTabConfig {
    showProjectName: boolean;        // 显示项目名称
    showActiveRunsToggle: boolean;   // 显示"仅显示活跃运行"切换按钮
    showFocusLatestToggle: boolean;  // 显示"自动切换到最新运行实例"切换按钮
}

/**
 * Default Runs Tab configuration
 * All options are hidden by default
 */
export const defaultRunsTabConfig: RunsTabConfig = {
    showProjectName: false,        // 默认隐藏
    showActiveRunsToggle: false,   // 默认隐藏
    showFocusLatestToggle: false,  // 默认隐藏
};
