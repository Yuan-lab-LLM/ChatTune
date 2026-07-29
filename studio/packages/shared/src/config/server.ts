import dotenv from 'dotenv';
import fs from 'fs';
import path from 'path';
import { APP_INFO, DEFAULT_CONFIG } from './common';
import { DEFAULT_ENVIRONMENT_CONFIG } from './environment';

// Load environment variables
dotenv.config();

export const ENV = {
    isDevelopment: process.env.NODE_ENV === 'development',
    isProduction: process.env.NODE_ENV === 'production',
    platform: process.platform,
    homeDir: process.env.HOME || process.env.USERPROFILE || '',
} as const;

const resolveAgentConfigPath = (): string => {
    if (process.env.MEDFLOW_AGENT_CONFIG_PATH?.trim()) {
        return process.env.MEDFLOW_AGENT_CONFIG_PATH.trim();
    }

    const relativePath = path.join(
        'agent',
        'config',
        'config.yaml',
    );
    const candidates = [
        path.resolve(process.cwd(), '..', relativePath),
        path.resolve(process.cwd(), '..', '..', '..', relativePath),
        path.resolve(__dirname, '..', '..', '..', '..', '..', relativePath),
    ];

    return (
        candidates.find((candidate) => fs.existsSync(candidate)) ||
        candidates[0]
    );
};

export const PATHS = {
    getAppDataDir: () => {
        switch (process.platform) {
            case 'win32':
                return path.join(process.env.APPDATA || '', APP_INFO.name);
            case 'linux':
                return path.join(process.env.HOME || '', APP_INFO.name);
            case 'darwin':
                return path.join(
                    process.env.HOME || '',
                    'Library',
                    'Application Support',
                    APP_INFO.name,
                );
            default:
                throw new Error(`Unsupported platform: ${process.platform}`);
        }
    },
    getLogsDir: () => path.join(PATHS.getAppDataDir(), 'logs'),
    getUserConfigPath: () => path.join(PATHS.getAppDataDir(), 'config.json'),
    getAgentConfigPath: () => resolveAgentConfigPath(),
} as const;

const getEnvironmentDefaultContainerName = () =>
    process.env.MEDFLOW_DEFAULT_CONTAINER?.trim() ||
    process.env.DEFAULT_DOCKER_CONTAINER?.trim() ||
    process.env.AGENT3_DEFAULT_DOCKER_CONTAINER?.trim() ||
    '';

const getEnvironmentDefaultEvaluateContainerName = () =>
    process.env.MEDFLOW_DEFAULT_EVALUATE_CONTAINER?.trim() ||
    process.env.DEFAULT_EVALUATE_DOCKER_CONTAINER?.trim() ||
    process.env.AGENT3_DEFAULT_EVALUATE_DOCKER_CONTAINER?.trim() ||
    '';

const getEnvironmentDefaultGrpoContainerName = () =>
    process.env.MEDFLOW_LOCAL_GRPO_CONTAINER?.trim() ||
    process.env.MEDFLOW_GRPO_DOCKER_CONTAINER?.trim() ||
    process.env.MEDFLOW_GRPO_CONTAINER?.trim() ||
    '';

const resolveConfigValue = (value: string): string => {
    const trimmedValue = value.trim().replace(/^['"]|['"]$/g, '');
    const envPattern = /^\$\{([A-Z0-9_]+)(?::([^}]*))?\}$/;
    const match = trimmedValue.match(envPattern);

    if (!match) {
        return trimmedValue;
    }

    const [, envName, fallback = ''] = match;
    return process.env[envName]?.trim() || fallback.trim();
};

const readDockerContainerFromYaml = (configPath: string, key: string): string => {
    try {
        if (!fs.existsSync(configPath)) {
            return '';
        }

        const content = fs.readFileSync(configPath, 'utf8');
        const environmentBlock = content.match(
            /^environment:\s*\r?\n((?:[ \t]+.*(?:\r?\n|$))*)/m,
        )?.[1];
        const targetContent = environmentBlock || content;
        const match = targetContent.match(new RegExp(`^[ \\t]*${key}:\\s*(.+?)\\s*$`, 'm'));

        return match ? resolveConfigValue(match[1]) : '';
    } catch (error) {
        console.error('Failed to load agent environment config:', error);
        return '';
    }
};

export const ServerConfig = {
    port: parseInt(process.env.PORT || DEFAULT_CONFIG.server.port.toString()),
    host:
        process.env.STUDIO_HOST?.trim() ||
        process.env.HOST?.trim() ||
        DEFAULT_CONFIG.server.host,
    publicUrl:
        process.env.STUDIO_PUBLIC_URL?.trim() ||
        `http://localhost:${process.env.PORT || DEFAULT_CONFIG.server.port}`,
    otelGrpcPort: parseInt(
        process.env.OTEL_GRPC_PORT ||
            DEFAULT_CONFIG.server.otelGrpcPort.toString(),
    ),
    database: {
        type: 'better-sqlite3' as const,
        database:
            process.env.MEDFLOW_DATABASE_PATH?.trim() ||
            path.join(PATHS.getAppDataDir(), 'database.sqlite'),
    },
    environment: {
        defaultContainerName:
            getEnvironmentDefaultContainerName() ||
            DEFAULT_ENVIRONMENT_CONFIG.defaultContainerName,
        defaultEvaluateContainerName:
            getEnvironmentDefaultEvaluateContainerName() ||
            DEFAULT_ENVIRONMENT_CONFIG.defaultEvaluateContainerName,
        defaultGrpoContainerName:
            getEnvironmentDefaultGrpoContainerName() ||
            DEFAULT_ENVIRONMENT_CONFIG.defaultGrpoContainerName,
    },
} as const;

// 服务器端的配置管理
export class ConfigManager {
    private static instance: ConfigManager;
    private config: typeof ServerConfig;

    private constructor() {
        this.config = ServerConfig;
        this.loadUserConfig();
    }

    static getInstance() {
        if (!ConfigManager.instance) {
            ConfigManager.instance = new ConfigManager();
        }
        return ConfigManager.instance;
    }

    private loadUserConfig() {
        const userConfigPath = PATHS.getUserConfigPath();
        try {
            if (fs.existsSync(userConfigPath)) {
                const userConfig = JSON.parse(
                    fs.readFileSync(userConfigPath, 'utf8'),
                );
                this.config = {
                    ...this.config,
                    ...userConfig,
                    environment: {
                        ...this.config.environment,
                        ...(userConfig.environment || {}),
                    },
                } as typeof ServerConfig;
            }
        } catch (error) {
            console.error('Failed to load user config:', error);
        }
    }

    reloadUserConfig() {
        this.config = ServerConfig;
        this.loadUserConfig();
    }

    getConfig() {
        return this.config;
    }

    private getUserDefaultContainerName(): string {
        const value = this.config.environment?.defaultContainerName?.trim();
        if (!value || value === DEFAULT_ENVIRONMENT_CONFIG.defaultContainerName) {
            return '';
        }
        return value;
    }

    private getAgentDefaultContainerName(): string {
        return readDockerContainerFromYaml(PATHS.getAgentConfigPath(), 'default_docker_container');
    }

    private getUserDefaultEvaluateContainerName(): string {
        const value = this.config.environment?.defaultEvaluateContainerName?.trim();
        if (!value || value === DEFAULT_ENVIRONMENT_CONFIG.defaultEvaluateContainerName) {
            return '';
        }
        return value;
    }

    private getAgentDefaultEvaluateContainerName(): string {
        return readDockerContainerFromYaml(PATHS.getAgentConfigPath(), 'default_evaluate_docker_container');
    }

    private getUserDefaultGrpoContainerName(): string {
        const value = this.config.environment?.defaultGrpoContainerName?.trim();
        if (!value || value === DEFAULT_ENVIRONMENT_CONFIG.defaultGrpoContainerName) {
            return '';
        }
        return value;
    }

    getDefaultContainerName(): string {
        this.reloadUserConfig();
        return (
            getEnvironmentDefaultContainerName() ||
            this.getUserDefaultContainerName() ||
            this.getAgentDefaultContainerName() ||
            DEFAULT_ENVIRONMENT_CONFIG.defaultContainerName
        );
    }

    getDefaultEvaluateContainerName(): string {
        this.reloadUserConfig();
        return (
            getEnvironmentDefaultEvaluateContainerName() ||
            this.getUserDefaultEvaluateContainerName() ||
            this.getAgentDefaultEvaluateContainerName() ||
            this.getDefaultContainerName() ||
            DEFAULT_ENVIRONMENT_CONFIG.defaultEvaluateContainerName
        );
    }

    getDefaultGrpoContainerName(): string {
        this.reloadUserConfig();
        return (
            getEnvironmentDefaultGrpoContainerName() ||
            this.getUserDefaultGrpoContainerName() ||
            DEFAULT_ENVIRONMENT_CONFIG.defaultGrpoContainerName
        );
    }

    saveConfig() {
        try {
            const userConfigPath = PATHS.getUserConfigPath();
            fs.mkdirSync(path.dirname(userConfigPath), { recursive: true });
            fs.writeFileSync(
                userConfigPath,
                JSON.stringify(this.config, null, 2),
            );
        } catch (error) {
            console.error('Failed to save config:', error);
        }
    }

    async setPort(port: number) {
        this.config = {
            ...this.config,
            port: port,
        };
    }
    async setOtelGrpcPort(otelGrpcPort: number) {
        this.config = {
            ...this.config,
            otelGrpcPort: otelGrpcPort,
        };
    }

    getDatabasePath(): string {
        return this.config.database.database;
    }

    getDatabaseSize(): number {
        try {
            const dbPath = this.getDatabasePath();
            if (fs.existsSync(dbPath)) {
                const stats = fs.statSync(dbPath);
                return stats.size;
            } else {
                return 0;
            }
        } catch (error) {
            console.error('Failed to get database size:', error);
            return 0;
        }
    }

    getFormattedDatabaseSize(): string {
        const sizeInBytes = this.getDatabaseSize();
        if (sizeInBytes === 0) {
            return '0 B';
        }
        const units = ['B', 'KB', 'MB', 'GB', 'TB'];
        let size = sizeInBytes;
        let unitIndex = 0;

        while (size >= 1024 && unitIndex < units.length - 1) {
            size /= 1024;
            unitIndex++;
        }
        const decimalPlaces = unitIndex === 0 ? 0 : 2;
        const formattedSize = size.toFixed(decimalPlaces);

        return `${formattedSize} ${units[unitIndex]}`;
    }

    getDataStats(): {
        path: string;
        size: number; // in bytes
        formattedSize: string; // formatted size with appropriate unit
    } {
        const dbPath = this.getDatabasePath();
        const size = this.getDatabaseSize();
        const formattedSize = this.getFormattedDatabaseSize();
        return {
            path: dbPath,
            size,
            formattedSize,
        };
    }
}
