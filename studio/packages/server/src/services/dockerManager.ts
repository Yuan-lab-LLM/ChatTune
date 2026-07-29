import { exec } from 'child_process';
import { promisify } from 'util';
import * as path from 'path';
import { ConfigManager } from '../../../shared/src/config/server';

const execAsync = promisify(exec);

// 数据集路径映射
const DATASET_PATHS = {
    raw: '/home/workspace/dataset',
    sft: '/home/workspace/dataset_batch_train',
    dpo: '/home/workspace/dataset_daily_train',
};

// 模型路径映射
const MODEL_PATHS = {
    base_train: '/home/workspace/models/base',
    batch_trained: '/home/workspace/models/batch_train',
    daily_trained: '/home/workspace/models/dpo_train/internal/saves',
    inference: '/home/workspace/medical_models',
};
const DPO_MODEL_PATHS = {
    saves: '/home/workspace/models/dpo_train/internal/saves',
    export: '/home/workspace/models/dpo_train/internal/export',
} as const;

// 评测文件路径
const TESTS_PATHS = {
    medical: '/home/workspace/tests/medical/choice',
    general: '/home/workspace/tests/general',
} as const;
const DEFAULT_TESTS_PATH = TESTS_PATHS.medical;
const EVALUATION_LOGS_PATH = '/home/workspace/tests/logs/benchmark';
const GRPO_MODEL_PATH = '/home/workspace/models/grpo_train';
const GRPO_DATA_ROOT = '/home/workspace/verl/examples/data_preprocess/data';
const HIDDEN_DATASET_FILES = new Set([
    'dataset_info.json',
    'preprocessing_audit.json',
    'preprocessing_summary.json',
    'score_audit.json',
    'score_summary.json',
]);

const isVisibleDatasetFile = (filename: string): boolean =>
    !HIDDEN_DATASET_FILES.has(filename.toLowerCase());

export interface DatasetFilePreview {
    filename: string;
    preview: string;
}

export interface DatasetInfo {
    name: string;
    type: string;
    path: string;
    files: string[];
    filePreviews: DatasetFilePreview[];
    size?: string;
    createdAt?: string;
}

interface DatasetDirectorySummary {
    name: string;
    files: string[];
    size?: string;
    modifiedTime?: string;
}

interface ModelDirectorySummary {
    name: string;
    size?: string;
    modifiedTime?: string;
    merged?: boolean;
    checkpoints?: string[];
}

interface MedicalTestSummary {
    filename: string;
    size?: string;
    metaContent?: string;
    category: keyof typeof TESTS_PATHS;
    basePath: string;
}

export interface ModelInfo {
    name: string;
    type: string;
    path: string;
    merged?: boolean;
    checkpoints?: string[];
    size?: string;
    createdAt?: string;
}

export interface GrpoFileInfo {
    name: string;
    path: string;
    directory: string;
    type: 'parquet';
}

export interface GrpoResourceInfo {
    containerName: string;
    models: ModelInfo[];
    trainFiles: GrpoFileInfo[];
    valFiles: GrpoFileInfo[];
}

export interface MedicalTestFile {
    filename: string;
    type: string;
    size: string;
    description: string;
    category?: keyof typeof TESTS_PATHS;
}

export interface EvaluationResult {
    jobId: string;
    model: string;
    dataset: string;
    status: string;
    accuracy: number;
    avgF1: number;
    totalScore?: number;
    startTime: string;
    endTime?: string;
    folderPath: string;
}

interface EnvironmentResourceSummary {
    container: {
        exists: boolean;
        running: boolean;
        message: string;
    };
    datasets: number;
    models: number;
    medicalTests: number;
    evaluationResults: number;
}

/**
 * Docker 管理器 - 用于跨容器查询数据集和模型
 */
export class DockerManager {
    private get defaultContainer() {
        return ConfigManager.getInstance().getDefaultContainerName();
    }
    private readonly previewByteLimit = 128 * 1024;
    private readonly maxPreviewByteLimit = 2 * 1024 * 1024;
    private readonly previewLineLimit = 80;
    private readonly largeFileThreshold = 2 * 1024 * 1024;

    private isValidContainerName(container: string): boolean {
        return /^[a-zA-Z0-9_.-]+$/.test(container.trim());
    }

    /**
     * 在 Docker 容器中执行命令
     */
    private async runDockerCommand(
        container: string,
        command: string,
        timeout = 30000
    ): Promise<{ stdout: string; stderr: string; success: boolean }> {
        if (!this.isValidContainerName(container)) {
            return {
                stdout: '',
                stderr: 'Invalid Docker container name',
                success: false,
            };
        }

        const dockerCmd = `docker exec ${container} ${command}`;
        console.log(`[DockerManager] Executing: ${dockerCmd}`);
        
        try {
            const { stdout, stderr } = await execAsync(dockerCmd, {
                timeout,
                maxBuffer: 10 * 1024 * 1024,
            });
            console.log(`[DockerManager] Command succeeded. stdout length: ${stdout.length}`);
            return { stdout, stderr, success: true };
        } catch (error: any) {
            console.error(`[DockerManager] Command failed: ${dockerCmd}`);
            console.error(`[DockerManager] Error: ${error.message || String(error)}`);
            return {
                stdout: '',
                stderr: error.message || String(error),
                success: false,
            };
        }
    }

    /**
     * 检查路径是否存在
     */
    private async checkPathExists(
        container: string,
        path: string
    ): Promise<boolean> {
        const { stdout, success } = await this.runDockerCommand(
            container,
            `test -d ${path} && echo 'exists' || echo 'not_exists'`
        );
        const exists = success && stdout.trim() === 'exists';
        console.log(`[DockerManager] checkPathExists: path=${path}, success=${success}, stdout="${stdout.trim()}", exists=${exists}`);
        return exists;
    }

    /**
     * 列出目录内容
     */
    private async listDirectory(
        container: string,
        path: string
    ): Promise<string[]> {
        if (!(await this.checkPathExists(container, path))) {
            return [];
        }

        const { stdout, success } = await this.runDockerCommand(
            container,
            `ls -1 ${path}`
        );

        if (!success) {
            return [];
        }

        return stdout
            .split('\n')
            .map((line) => line.trim())
            .filter((line) => line.length > 0);
    }

    /**
     * 检查 Docker 容器是否存在且正在运行。
     */
    async checkContainerStatus(
        container: string
    ): Promise<{ exists: boolean; running: boolean; message: string }> {
        const trimmedContainer = container.trim();

        if (!trimmedContainer) {
            return {
                exists: false,
                running: false,
                message: '容器名称为空',
            };
        }

        if (!/^[a-zA-Z0-9_.-]+$/.test(trimmedContainer)) {
            return {
                exists: false,
                running: false,
                message: '容器名称只能包含字母、数字、下划线、点和短横线',
            };
        }

        try {
            const { stdout } = await execAsync(
                `docker inspect -f "{{.State.Running}}" ${trimmedContainer}`,
                { timeout: 10000 }
            );
            const running = stdout.trim() === 'true';

            return {
                exists: true,
                running,
                message: running ? '容器正在运行' : '容器存在，但当前未运行',
            };
        } catch (error: any) {
            const message = error?.message || String(error);
            const notFound =
                message.includes('No such object') ||
                message.includes('No such container');

            return {
                exists: false,
                running: false,
                message: notFound
                    ? `未找到容器 "${trimmedContainer}"`
                    : `无法检查 Docker 容器：${message}`,
            };
        }
    }

    private async countDatasetFolders(container: string): Promise<number> {
        const datasetPaths = [
            DATASET_PATHS.raw,
            DATASET_PATHS.sft,
            DATASET_PATHS.dpo,
        ];
        const counts = await Promise.all(
            datasetPaths.map(async (datasetPath: string) => {
                const datasetNames = await this.listDirectory(
                    container,
                    datasetPath
                );

                if (datasetNames.length === 0) {
                    return 0;
                }

                const validFlags = await Promise.all(
                    datasetNames.map(async (datasetName) => {
                        const files = await this.listDirectory(
                            container,
                            `${datasetPath}/${datasetName}`
                        );
                        return files.some(
                            (file) =>
                                file.endsWith('.json')
                        );
                    })
                );

                return validFlags.filter(Boolean).length;
            })
        );

        return counts.reduce((total, count) => total + count, 0);
    }

    private async countModelFolders(container: string): Promise<number> {
        const modelPaths = [
            MODEL_PATHS.base_train,
            MODEL_PATHS.batch_trained,
            DPO_MODEL_PATHS.saves,
            DPO_MODEL_PATHS.export,
            MODEL_PATHS.inference,
        ];
        const counts = await Promise.all(
            modelPaths.map(async (modelPath: string) => {
                const modelNames = await this.listDirectory(container, modelPath);
                return modelNames.length;
            })
        );

        return counts.reduce((total, count) => total + count, 0);
    }

    private async countMedicalTestFiles(container: string): Promise<number> {
        return (await this.listMedicalTestSummaries(container)).length;
    }

    private async countEvaluationResultFolders(container: string): Promise<number> {
        const logsPath = '/home/workspace/tests/logs/benchmark';
        const entries = await this.listDirectory(container, logsPath);
        return entries.filter((entry) => /^\d+_[a-f0-9]+$/.test(entry)).length;
    }

    /**
     * 轻量级环境体检：只统计目录和关键资源，不读取数据集预览。
     */
    async getEnvironmentResourceSummary(
        container: string = this.defaultContainer
    ): Promise<EnvironmentResourceSummary> {
        const containerStatus = await this.checkContainerStatus(container);

        if (!containerStatus.running) {
            return {
                container: containerStatus,
                datasets: 0,
                models: 0,
                medicalTests: 0,
                evaluationResults: 0,
            };
        }

        const [datasets, models, medicalTests, evaluationResults] =
            await Promise.all([
                this.countDatasetFolders(container),
                this.countModelFolders(container),
                this.countMedicalTestFiles(container),
                this.countEvaluationResultFolders(container),
            ]);

        return {
            container: containerStatus,
            datasets,
            models,
            medicalTests,
            evaluationResults,
        };
    }

    /**
     * 检查文件夹是否包含 _merged 后缀
     */
    private async checkMergedSuffix(
        container: string,
        basePath: string,
        folderName: string
    ): Promise<boolean> {
        if (folderName.endsWith('_merged')) {
            return true;
        }

        const fullPath = `${basePath}/${folderName}`;
        const { stdout, success } = await this.runDockerCommand(
            container,
            `ls -1 ${fullPath} | grep '_merged' || true`
        );
        return success && stdout.trim().length > 0;
    }

    /**
     * 获取文件/目录的统计信息（大小和修改时间）
     */
    private async getFileStats(
        container: string,
        filePath: string
    ): Promise<{ size: string; modifiedTime: string } | null> {
        try {
            // 方法1: 使用 du 命令获取实际占用空间（包括目录内所有文件）
            const duResult = await this.runDockerCommand(
                container,
                `du -sb "${filePath}" 2>/dev/null | cut -f1`
            );
            
            // 方法2: 使用 stat 获取修改时间
            const statResult = await this.runDockerCommand(
                container,
                `stat -c "%y" "${filePath}" 2>/dev/null`
            );
            
            // 解析大小
            let size: string | null = null;
            if (duResult.success && duResult.stdout) {
                const sizeBytes = parseInt(duResult.stdout.trim(), 10);
                if (!isNaN(sizeBytes)) {
                    size = this.formatBytes(sizeBytes);
                }
            }
            
            // 解析修改时间
            let modifiedTime: string | null = null;
            if (statResult.success && statResult.stdout) {
                // 时间格式: 2024-01-15 14:30:00.000000000 +0800
                const parts = statResult.stdout.trim().split(' ');
                modifiedTime = parts.slice(0, 2).join(' ');
            }
            
            // 如果 du 失败，尝试使用 stat 获取大小（对文件有效，对目录不准确）
            if (!size) {
                const statSizeResult = await this.runDockerCommand(
                    container,
                    `stat -c "%s" "${filePath}" 2>/dev/null`
                );
                if (statSizeResult.success && statSizeResult.stdout) {
                    const sizeBytes = parseInt(statSizeResult.stdout.trim(), 10);
                    if (!isNaN(sizeBytes)) {
                        size = this.formatBytes(sizeBytes);
                    }
                }
            }
            
            // 如果 stat 获取时间失败，尝试使用 ls
            if (!modifiedTime) {
                const lsResult = await this.runDockerCommand(
                    container,
                    `ls -ld --time-style=+"%Y-%m-%d %H:%M:%S" "${filePath}" 2>/dev/null`
                );
                if (lsResult.success && lsResult.stdout) {
                    const parts = lsResult.stdout.trim().split(/\s+/);
                    modifiedTime = parts.slice(5, 7).join(' ');
                }
            }
            
            if (size || modifiedTime) {
                return {
                    size: size || undefined,
                    modifiedTime: modifiedTime || undefined,
                } as { size: string; modifiedTime: string };
            }
            
            return null;
        } catch (error) {
            console.error(`[DockerManager] Failed to get stats for ${filePath}:`, error);
            return null;
        }
    }

    /**
     * 格式化字节大小为人类可读格式
     */
    private formatBytes(bytes: number): string {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    private escapeForDoubleQuotes(value: string): string {
        return value.replace(/(["\\$`])/g, '\\$1');
    }

    private extractCompleteJsonArrayItems(
        content: string,
        count: number,
    ): string[] {
        const arrayStart = content.indexOf('[');
        if (arrayStart === -1) {
            return [];
        }

        const items: string[] = [];
        let start = -1;
        let depth = 0;
        let inString = false;
        let escaped = false;

        for (let i = arrayStart + 1; i < content.length; i++) {
            const char = content[i];

            if (inString) {
                if (escaped) {
                    escaped = false;
                    continue;
                }

                if (char === '\\') {
                    escaped = true;
                    continue;
                }

                if (char === '"') {
                    inString = false;
                }
                continue;
            }

            if (char === '"') {
                inString = true;
                continue;
            }

            if (start === -1) {
                if (/\s/.test(char) || char === ',') {
                    continue;
                }
                start = i;
                depth = char === '{' || char === '[' ? 1 : 0;
                continue;
            }

            if (char === '{' || char === '[') {
                depth += 1;
                continue;
            }

            if (char === '}' || char === ']') {
                if (depth > 0) {
                    depth -= 1;
                }

                if (depth === 0) {
                    const rawItem = content.slice(start, i + 1).trim();
                    if (rawItem) {
                        items.push(rawItem);
                    }
                    start = -1;

                    if (items.length >= count) {
                        break;
                    }
                }
            }
        }

        return items;
    }

    private async readPreviewSnippet(
        container: string,
        filePath: string,
        isJsonl: boolean,
        byteLimit: number,
    ): Promise<{ stdout: string; stderr: string; success: boolean }> {
        const safeFilePath = this.escapeForDoubleQuotes(filePath);
        const readCommand = isJsonl
            ? `head -n ${this.previewLineLimit} "${safeFilePath}" 2>&1`
            : `head -c ${byteLimit} "${safeFilePath}" 2>&1`;

        return this.runDockerCommand(container, readCommand);
    }

    /**
     * 获取文件预览（前3条 JSON 数据）
     */
    private async getFilePreview(
        container: string,
        filePath: string,
        count = 3
    ): Promise<string> {
        console.log(`[DockerManager] Reading file for preview: ${filePath}`);

        const isJsonl = filePath.endsWith('.jsonl');
        let currentByteLimit = this.previewByteLimit;
        let isLargeFile = false;
        let stdout = '';
        let stderr = '';
        let success = false;

        while (true) {
            const result = await this.readPreviewSnippet(
                container,
                filePath,
                isJsonl,
                currentByteLimit,
            );
            stdout = result.stdout;
            stderr = result.stderr;
            success = result.success;

            console.log(
                `[DockerManager] Read file result: success=${success}, stdout length=${stdout.length}, stderr=${stderr}, byteLimit=${currentByteLimit}`,
            );

            if (!success || isJsonl) {
                break;
            }

            const trimmed = stdout.trim();
            if (!trimmed) {
                break;
            }

            try {
                const parsed = JSON.parse(trimmed);
                if (Array.isArray(parsed) || (typeof parsed === 'object' && parsed !== null)) {
                    break;
                }
            } catch {
                const partialItems = this.extractCompleteJsonArrayItems(trimmed, count);
                const reachedEof = stdout.length < currentByteLimit;
                const reachedCap = currentByteLimit >= this.maxPreviewByteLimit;

                if (
                    partialItems.length >= count ||
                    reachedEof ||
                    reachedCap
                ) {
                    isLargeFile = !reachedEof && currentByteLimit >= this.largeFileThreshold;
                    break;
                }

                currentByteLimit = Math.min(
                    currentByteLimit * 2,
                    this.maxPreviewByteLimit,
                );
                continue;
            }
        }

        if (!success) {
            console.error(`[DockerManager] Failed to read file: ${filePath}, error: ${stderr}`);
            return `(无法读取文件: ${stderr || '未知错误'})`;
        }

        if (!stdout.trim()) {
            console.log(`[DockerManager] File is empty: ${filePath}`);
            return '(文件为空)';
        }

        const content = stdout.trim();
        console.log(`[DockerManager] File content preview (first 200 chars): ${content.substring(0, 200)}`);

        // 尝试解析为 JSON 数组
        try {
            const data = JSON.parse(content);
            if (Array.isArray(data)) {
                const items = data.slice(0, count);
                const parsedPreview = items
                    .map((item) => JSON.stringify(item, null, 2))
                    .join('\n---\n');
                return isLargeFile
                    ? `(大文件预览，仅展示前 ${count} 条可解析内容)\n${parsedPreview}`
                    : parsedPreview;
            } else if (typeof data === 'object' && data !== null) {
                // 支持单个 JSON 对象（如医疗病历格式）
                const parsedObject = JSON.stringify(data, null, 2);
                return isLargeFile
                    ? `(大文件预览，仅展示截取片段解析结果)\n${parsedObject}`
                    : parsedObject;
            }
        } catch {
            // 不是 JSON 数组或对象，尝试 JSONL 格式
        }

        if (!isJsonl) {
            const partialItems = this.extractCompleteJsonArrayItems(content, count);
            if (partialItems.length > 0) {
                const parsedItems = partialItems
                    .map((item) => {
                        try {
                            return JSON.stringify(JSON.parse(item), null, 2);
                        } catch {
                            return item;
                        }
                    })
                    .join('\n---\n');

                return isLargeFile
                    ? `(大文件预览，仅展示前 ${partialItems.length} 条完整记录)\n${parsedItems}`
                    : parsedItems;
            }
        }

        // 尝试按行解析（JSONL 格式）
        const lines = content.split('\n');
        const jsonObjects: string[] = [];

        for (const line of lines) {
            const trimmedLine = line.trim();
            if (!trimmedLine || ['[', ']', '{', '}'].includes(trimmedLine)) {
                continue;
            }

            // 去除行尾逗号
            const cleanLine = trimmedLine.endsWith(',')
                ? trimmedLine.slice(0, -1)
                : trimmedLine;

            try {
                const obj = JSON.parse(cleanLine);
                jsonObjects.push(JSON.stringify(obj, null, 2));
                if (jsonObjects.length >= count) {
                    break;
                }
            } catch {
                continue;
            }
        }

        if (jsonObjects.length > 0) {
            const jsonlPreview = jsonObjects.join('\n---\n');
            return isLargeFile
                ? `(大文件预览，仅展示前 ${jsonObjects.length} 条记录)\n${jsonlPreview}`
                : jsonlPreview;
        }

        // 无法解析，返回原始内容前 10 行
        const rawPreview = (
            lines
                .slice(0, 10)
                .filter((l) => l.trim())
                .join('\n')
        );
        return isLargeFile
            ? `(大文件预览，仅截取文件开头片段)\n${rawPreview}`
            : rawPreview;
    }

    private async listDatasetSummaries(
        container: string,
        datasetPath: string,
    ): Promise<DatasetDirectorySummary[]> {
        const safeDatasetPath = this.escapeForDoubleQuotes(datasetPath);
        const command = [
            'sh -c ',
            `'if [ ! -d "${safeDatasetPath}" ]; then exit 0; fi; `,
            `for d in "${safeDatasetPath}"/*; do `,
            `[ -d "$d" ] || continue; `,
            `name=$(basename "$d"); `,
            `size=$(du -sb "$d" 2>/dev/null | cut -f1); `,
            `mtime=$(stat -c "%y" "$d" 2>/dev/null | cut -d "." -f1); `,
            `files=$(find "$d" -maxdepth 1 -type f -name "*.json" ! -name "dataset_info.json" ! -name "preprocessing_audit.json" ! -name "preprocessing_summary.json" ! -name "score_audit.json" ! -name "score_summary.json" -exec basename {} \\; 2>/dev/null | tr "\\n" "\\t"); `,
            `printf "%s\\t%s\\t%s\\t%s\\n" "$name" "$size" "$mtime" "$files"; `,
            `done'`,
        ].join('');

        const { stdout, success } = await this.runDockerCommand(
            container,
            command,
            120000,
        );

        if (!success || !stdout.trim()) {
            return [];
        }

        return stdout
            .split('\n')
            .map((line) => line.trimEnd())
            .filter((line) => line.length > 0)
            .map((line) => {
                const [name, sizeBytes, modifiedTime, ...files] = line.split('\t');
                const parsedSize = Number.parseInt(sizeBytes, 10);
                return {
                    name,
                    files: files.filter((file) => file && isVisibleDatasetFile(file)),
                    size: Number.isFinite(parsedSize)
                        ? this.formatBytes(parsedSize)
                        : undefined,
                    modifiedTime: modifiedTime || undefined,
                };
            })
            .filter((summary) => summary.name);
    }

    private async listModelSummaries(
        container: string,
        modelPath: string,
        includeMerged: boolean,
    ): Promise<ModelDirectorySummary[]> {
        const safeModelPath = this.escapeForDoubleQuotes(modelPath);
        const mergedCommand = includeMerged
            ? `merged=$(find "$d" -maxdepth 1 -name "*_merged*" -print -quit 2>/dev/null); `
            : `merged=""; `;
        const command = [
            'sh -c ',
            `'if [ ! -d "${safeModelPath}" ]; then exit 0; fi; `,
            `for d in "${safeModelPath}"/*; do `,
            `[ -d "$d" ] || continue; `,
            `name=$(basename "$d"); `,
            `size=$(du -sb "$d" 2>/dev/null | cut -f1); `,
            `mtime=$(stat -c "%y" "$d" 2>/dev/null | cut -d "." -f1); `,
            mergedCommand,
            `checkpoints=$(find "$d" -maxdepth 1 -type d -name "checkpoint-*" -printf "%f\\n" 2>/dev/null | sort -V | awk 'BEGIN { sep="" } { printf "%s%s", sep, $0; sep="," }'); `,
            `if [ -n "$merged" ]; then merged_flag=1; else merged_flag=0; fi; `,
            `printf "%s\\t%s\\t%s\\t%s\\t%s\\n" "$name" "$size" "$mtime" "$merged_flag" "$checkpoints"; `,
            `done'`,
        ].join('');

        const { stdout, success } = await this.runDockerCommand(
            container,
            command,
            120000,
        );

        if (!success || !stdout.trim()) {
            return [];
        }

        return stdout
            .split('\n')
            .map((line) => line.trimEnd())
            .filter((line) => line.length > 0)
            .map((line) => {
                const [name, sizeBytes, modifiedTime, mergedFlag, checkpointsText] = line.split('\t');
                const parsedSize = Number.parseInt(sizeBytes, 10);
                return {
                    name,
                    size: Number.isFinite(parsedSize)
                        ? this.formatBytes(parsedSize)
                        : undefined,
                    modifiedTime: modifiedTime || undefined,
                    merged: mergedFlag === '1',
                    checkpoints: (checkpointsText || '').split(',').filter(Boolean),
                };
            })
            .filter((summary) => summary.name);
    }

    private async listGrpoModelSummaries(
        container: string,
    ): Promise<ModelDirectorySummary[]> {
        return (await this.listModelSummaries(container, GRPO_MODEL_PATH, false))
            .filter((summary) => summary.size !== '0 B');
    }

    private async listMedicalTestSummariesFromPath(
        container: string,
        category: keyof typeof TESTS_PATHS,
        testsPath: string,
    ): Promise<MedicalTestSummary[]> {
        const safeTestsPath = this.escapeForDoubleQuotes(testsPath);
        const command =
            category === 'general'
                ? [
                    'sh -c ',
                    `'if [ ! -d "${safeTestsPath}" ]; then exit 0; fi; `,
                    `for d in "${safeTestsPath}"/*; do `,
                    `[ -d "$d" ] || continue; `,
                    `name=$(basename "$d"); `,
                    `size=$(du -sb "$d" 2>/dev/null | cut -f1); `,
                    `printf "%s\\t%s\\t\\n" "$name" "$size"; `,
                    `done'`,
                ].join('')
                : [
                    'sh -c ',
                    `'if [ ! -d "${safeTestsPath}" ]; then exit 0; fi; `,
                    `for f in "${safeTestsPath}"/*.json "${safeTestsPath}"/*.jsonl; do `,
                    `[ -f "$f" ] || continue; `,
                    `name=$(basename "$f"); `,
                    `size=$(stat -c "%s" "$f" 2>/dev/null); `,
                    `meta=""; `,
                    `if [ -f "$f.meta" ]; then meta=$(cat "$f.meta" 2>/dev/null | tr "\\n\\t" "  "); fi; `,
                    `printf "%s\\t%s\\t%s\\n" "$name" "$size" "$meta"; `,
                    `done'`,
                ].join('');

        const { stdout, success } = await this.runDockerCommand(
            container,
            command,
            120000,
        );

        if (!success || !stdout.trim()) {
            return [];
        }

        return stdout
            .split('\n')
            .map((line) => line.trimEnd())
            .filter((line) => line.length > 0)
            .map((line) => {
                const [filename, sizeBytes, ...metaParts] = line.split('\t');
                const parsedSize = Number.parseInt(sizeBytes, 10);
                return {
                    filename,
                    size: Number.isFinite(parsedSize)
                        ? this.formatBytes(parsedSize)
                        : undefined,
                    metaContent: metaParts.join('\t') || undefined,
                    category,
                    basePath: testsPath,
                };
            })
            .filter((summary) => summary.filename);
    }

    private async listMedicalTestSummaries(
        container: string,
    ): Promise<MedicalTestSummary[]> {
        const summaryGroups = await Promise.all(
            Object.entries(TESTS_PATHS).map(([category, testsPath]) =>
                this.listMedicalTestSummariesFromPath(
                    container,
                    category as keyof typeof TESTS_PATHS,
                    testsPath,
                )
            )
        );
        return summaryGroups.flat();
    }

    private getTestsPathForType(testType: string): string {
        const normalized = testType.trim().toLowerCase();
        const generalSignals = [
            'general',
            '通用',
            'mmlu',
            'cmmlu',
            'ceval',
            'c-eval',
            'gsm8k',
            'math',
            'squad',
            'drop',
            'truthfulqa',
            'humaneval',
            'livecodebench',
            'ifeval',
            'bbh',
        ];

        return generalSignals.some((signal) => normalized.includes(signal))
            ? TESTS_PATHS.general
            : TESTS_PATHS.medical;
    }

    private async resolveTestBasePath(
        container: string,
        filename: string,
    ): Promise<string> {
        const summaries = await this.listMedicalTestSummaries(container);
        return summaries.find((summary) => summary.filename === filename)?.basePath
            || DEFAULT_TESTS_PATH;
    }

    private async listEvaluationResultSummaries(
        container: string,
        logsPath: string,
    ): Promise<EvaluationResult[]> {
        const safeLogsPath = this.escapeForDoubleQuotes(logsPath);
        const pythonScript = [
            'import json, os, sys',
            'folder, folder_path = sys.argv[1], sys.argv[2]',
            'load_json = lambda path: json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}',
            'meta = load_json(os.path.join(folder_path, "meta.json"))',
            'result = load_json(os.path.join(folder_path, "result.json"))',
            'summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}',
            'total_score = result.get("total_score", summary.get("total_score"))',
            'payload = {"jobId": meta.get("job_id") or folder, "model": meta.get("model") or "unknown", "dataset": meta.get("dataset") or "unknown", "status": meta.get("status") or "unknown", "accuracy": summary.get("accuracy") or 0, "avgF1": summary.get("avg_f1") or 0, "startTime": meta.get("start_time") or "", "endTime": meta.get("end_time"), "folderPath": folder_path}',
            'payload.update({"totalScore": total_score} if isinstance(total_score, (int, float)) else {})',
            'print(json.dumps(payload, ensure_ascii=False))',
        ].join('; ');
        const safePythonScript = this.escapeForDoubleQuotes(pythonScript);
        const command = [
            'sh -c ',
            `'if [ ! -d "${safeLogsPath}" ]; then exit 0; fi; `,
            `for d in "${safeLogsPath}"/*; do `,
            `[ -d "$d" ] || continue; `,
            `folder=$(basename "$d"); `,
            `case "$folder" in [0-9]*_[a-f0-9]*) ;; *) continue ;; esac; `,
            `python3 -c "${safePythonScript}" "$folder" "$d" 2>/dev/null || `,
            `python -c "${safePythonScript}" "$folder" "$d" 2>/dev/null; `,
            `done'`,
        ].join('');

        const { stdout, success } = await this.runDockerCommand(
            container,
            command,
            120000,
        );

        if (!success || !stdout.trim()) {
            return [];
        }

        return stdout
            .split('\n')
            .map((line) => line.trim())
            .filter((line) => line.length > 0)
            .map((line) => {
                try {
                    return JSON.parse(line) as EvaluationResult;
                } catch {
                    return null;
                }
            })
            .filter((summary): summary is EvaluationResult => Boolean(summary));
    }

    /**
     * 获取数据集详细信息
     */
    private async getDatasetDetails(
        datasetPath: string,
        datasetType: string,
        summary: DatasetDirectorySummary
    ): Promise<DatasetInfo> {
        const files = summary.files;
        return {
            name: summary.name,
            type: datasetType,
            path: datasetPath,
            files,
            filePreviews: [],
            size: summary.size,
            createdAt: summary.modifiedTime,
        };
    }

    /**
     * 获取所有数据集（并行优化版）
     */
    async getDatasets(
        container: string = this.defaultContainer
    ): Promise<Record<string, DatasetInfo[]>> {
        console.log(`[DockerManager] Getting datasets from container: ${container} (并行模式)`);
        
        // 并行检查所有路径是否存在
        const pathChecks = await Promise.all(
            Object.entries(DATASET_PATHS).map(async ([dataType, path]) => {
                const exists = await this.checkPathExists(container, path);
                console.log(`[DockerManager] Path ${path} exists: ${exists}`);
                return { dataType, path, exists };
            })
        );

        // 并行获取所有数据集的详情
        const results = await Promise.all(
            pathChecks.map(async ({ dataType, path, exists }) => {
                if (!exists) {
                    return { dataType, datasets: [] };
                }
                
                const datasetSummaries = await this.listDatasetSummaries(container, path);
                console.log(`[DockerManager] Found ${datasetSummaries.length} datasets in ${path}`);
                
                // 并行获取每个数据集的详情
                const datasets = await Promise.all(
                    datasetSummaries.map(summary =>
                        this.getDatasetDetails(path, dataType, summary)
                    )
                );
                
                // 过滤掉没有 json 文件的数据集
                const validDatasets = datasets.filter(details => {
                    const jsonFileCount = details.files.filter(
                        (file) => file.endsWith('.json'),
                    ).length;

                    if (jsonFileCount > 0) {
                        console.log(`[DockerManager] Added dataset: ${details.name} (${jsonFileCount} json files)`);
                        return true;
                    } else {
                        console.log(`[DockerManager] Skipped dataset: ${details.name} (no json files)`);
                        return false;
                    }
                });
                
                return { dataType, datasets: validDatasets };
            })
        );

        // 组装结果
        const result: Record<string, DatasetInfo[]> = {};
        results.forEach(({ dataType, datasets }) => {
            result[dataType] = datasets;
        });

        console.log(`[DockerManager] Returning ${Object.values(result).flat().length} total datasets`);
        return result;
    }

    async getDatasetFilePreviews(
        container: string,
        datasetType: 'raw' | 'sft' | 'dpo',
        datasetName: string
    ): Promise<DatasetFilePreview[]> {
        const datasetPath = DATASET_PATHS[datasetType];
        if (!datasetPath) {
            throw new Error(`无效的数据集类型: ${datasetType}`);
        }

        const summaries = await this.listDatasetSummaries(container, datasetPath);
        const summary = summaries.find((item) => item.name === datasetName);
        if (!summary) {
            throw new Error(`数据集 "${datasetName}" 不存在`);
        }

        const previewFiles = summary.files.filter(
            (file) => isVisibleDatasetFile(file) && (file.endsWith('.json')),
        );

        return Promise.all(
            previewFiles.map(async (file) => ({
                filename: file,
                preview: await this.getFilePreview(
                    container,
                    `${datasetPath}/${datasetName}/${file}`,
                    3,
                ),
            })),
        );
    }

    /**
     * 获取所有模型（批量优化版）
     */
    async getModels(
        container: string = this.defaultContainer
    ): Promise<Record<string, ModelInfo[]>> {
        console.log(`[DockerManager] Getting models from container: ${container} (批量模式)`);

        const toModelInfo = (
            type: string,
            modelPath: string,
            summary: ModelDirectorySummary,
        ): ModelInfo => ({
            name: summary.name,
            type,
            path: modelPath,
            merged: summary.merged,
            checkpoints: summary.checkpoints,
            size: summary.size,
            createdAt: summary.modifiedTime,
        });

        const [
            baseSummaries,
            batchSummaries,
            dpoSaveSummaries,
            dpoExportSummaries,
            inferenceSummaries,
        ] = await Promise.all([
            this.listModelSummaries(container, MODEL_PATHS.base_train, false),
            this.listModelSummaries(container, MODEL_PATHS.batch_trained, true),
            this.listModelSummaries(container, DPO_MODEL_PATHS.saves, false),
            this.listModelSummaries(container, DPO_MODEL_PATHS.export, false),
            this.listModelSummaries(container, MODEL_PATHS.inference, false),
        ]);

        return {
            base_train: baseSummaries.map((summary) =>
                toModelInfo('base_train', MODEL_PATHS.base_train, summary),
            ),
            batch_trained: batchSummaries.map((summary) =>
                toModelInfo('batch_trained', MODEL_PATHS.batch_trained, summary),
            ),
            daily_trained: [
                ...dpoSaveSummaries.map((summary) =>
                    toModelInfo('daily_trained', DPO_MODEL_PATHS.saves, {
                        ...summary,
                        merged: false,
                    }),
                ),
                ...dpoExportSummaries.map((summary) =>
                    toModelInfo('daily_trained', DPO_MODEL_PATHS.export, {
                        ...summary,
                        merged: true,
                    }),
                ),
            ],
            inference: inferenceSummaries.map((summary) =>
                toModelInfo('inference', MODEL_PATHS.inference, summary),
            ),
        };
    }

    async getGrpoResources(
        container: string
    ): Promise<GrpoResourceInfo> {
        const modelSummaries = await this.listGrpoModelSummaries(container);
        const models = modelSummaries.map((summary) => ({
            name: summary.name,
            type: 'grpo_base',
            path: GRPO_MODEL_PATH,
            size: summary.size,
            createdAt: summary.modifiedTime,
        }));

        const parseParquetFiles = (stdout: string): GrpoFileInfo[] =>
            stdout
                .split('\n')
                .map((line) => line.trim())
                .filter((line) => line.endsWith('.parquet'))
                .map((filePath) => {
                    const normalized = filePath.replace(/\\/g, '/');
                    const lastSlash = normalized.lastIndexOf('/');
                    return {
                        name: lastSlash >= 0 ? normalized.slice(lastSlash + 1) : normalized,
                        path: normalized,
                        directory: lastSlash >= 0 ? normalized.slice(0, lastSlash) : '',
                        type: 'parquet' as const,
                    };
                });

        const listGrpoParquetFiles = async (directory: string): Promise<GrpoFileInfo[]> => {
            const safeDirectory = this.escapeForDoubleQuotes(directory);
            const { stdout, success } = await this.runDockerCommand(
                container,
                `sh -c 'if [ -d "${safeDirectory}" ]; then find "${safeDirectory}" -type f -name "*.parquet" | sort | head -200; fi'`,
                30000
            );
            return success ? parseParquetFiles(stdout) : [];
        };

        let [trainFiles, valFiles] = await Promise.all([
            listGrpoParquetFiles(`${GRPO_DATA_ROOT}/train`),
            listGrpoParquetFiles(`${GRPO_DATA_ROOT}/val`),
        ]);

        if (trainFiles.length === 0 && valFiles.length === 0) {
            const parquetFiles = await listGrpoParquetFiles(GRPO_DATA_ROOT);
            trainFiles = parquetFiles.filter((file) => /train/i.test(file.name));
            valFiles = parquetFiles.filter((file) => /val|valid|dev/i.test(file.name));
            trainFiles = trainFiles.length > 0 ? trainFiles : parquetFiles;
            valFiles = valFiles.length > 0 ? valFiles : parquetFiles;
        }

        return {
            containerName: container,
            models,
            trainFiles,
            valFiles,
        };
    }

    /**
     * 检查数据集是否已存在
     */
    async checkDatasetExists(
        container: string,
        datasetType: 'raw' | 'sft' | 'dpo',
        datasetName: string
    ): Promise<boolean> {
        const basePath = DATASET_PATHS[datasetType];
        const fullPath = `${basePath}/${datasetName}`;
        const exists = await this.checkPathExists(container, fullPath);
        console.log(`[DockerManager] checkDatasetExists: type=${datasetType}, path=${fullPath}, exists=${exists}`);
        return exists;
    }

    /**
     * 复制文件到容器
     */
    private async copyToContainer(
        container: string,
        localPath: string,
        containerPath: string
    ): Promise<boolean> {
        const dockerCmd = `docker cp "${localPath}" "${container}:${containerPath}"`;
        console.log(`[DockerManager] Copying to container: ${dockerCmd}`);
        
        try {
            await execAsync(dockerCmd, { timeout: 60000 });
            console.log(`[DockerManager] File copied successfully to ${containerPath}`);
            return true;
        } catch (error: any) {
            console.error(`[DockerManager] Failed to copy to container: ${error.message}`);
            return false;
        }
    }

    /**
     * 从容器复制文件
     */
    private async copyFromContainer(
        container: string,
        containerPath: string,
        localPath: string
    ): Promise<boolean> {
        const dockerCmd = `docker cp "${container}:${containerPath}" "${localPath}"`;
        console.log(`[DockerManager] Copying from container: ${dockerCmd}`);
        
        try {
            await execAsync(dockerCmd, { timeout: 60000 });
            console.log(`[DockerManager] File copied successfully from container`);
            return true;
        } catch (error: any) {
            console.error(`[DockerManager] Failed to copy from container: ${error.message}`);
            return false;
        }
    }

    /**
     * 从容器读取文件内容
     */
    private async readFileFromContainer(
        container: string,
        containerPath: string
    ): Promise<string | null> {
        const tempPath = `/tmp/docker_read_${Date.now()}_${path.basename(containerPath)}`;
        
        try {
            const copied = await this.copyFromContainer(container, containerPath, tempPath);
            if (!copied) {
                return null;
            }
            
            const fs = await import('fs');
            const content = fs.readFileSync(tempPath, 'utf-8');
            fs.unlinkSync(tempPath);
            
            return content;
        } catch (error: any) {
            console.error(`[DockerManager] Failed to read file from container: ${error.message}`);
            return null;
        }
    }

    /**
     * 在容器中解压压缩包
     */
    private async extractArchive(
        container: string,
        archivePath: string,
        extractTo: string
    ): Promise<boolean> {
        let innerCommand: string;
        
        if (archivePath.endsWith('.zip')) {
            innerCommand = `unzip -o "${archivePath}" -d "${extractTo}"`;
        } else if (archivePath.endsWith('.tar.gz') || archivePath.endsWith('.tgz')) {
            innerCommand = `tar -xzf "${archivePath}" -C "${extractTo}"`;
        } else if (archivePath.endsWith('.tar')) {
            innerCommand = `tar -xf "${archivePath}" -C "${extractTo}"`;
        } else {
            console.error(`[DockerManager] Unsupported archive format: ${archivePath}`);
            return false;
        }
        
        // 使用 sh -c 包装整个命令，确保 mkdir 和 tar 都在容器内执行
        const command = `sh -c 'mkdir -p "${extractTo}" && ${innerCommand}'`;
        
        console.log(`[DockerManager] Extracting archive: ${command}`);
        console.log(`[DockerManager] Extract target: ${extractTo}`);
        
        const { success } = await this.runDockerCommand(container, command, 120000);
        if (success) {
            console.log(`[DockerManager] Archive extracted successfully to ${extractTo}`);
        } else {
            console.error(`[DockerManager] Failed to extract archive`);
        }
        return success;
    }

    /**
     * 在容器中创建 zip 文件
     */
    private async createTar(
        container: string,
        sourcePath: string,
        outputPath: string
    ): Promise<boolean> {
        // 使用 tar -czf 创建 .tar.gz 压缩包
        const command = `tar -czf "${outputPath}" -C "$(dirname "${sourcePath}")" "$(basename "${sourcePath}")"`;
        console.log(`[DockerManager] Creating tar.gz: ${command}`);
        
        const { success } = await this.runDockerCommand(container, command, 120000);
        if (success) {
            console.log(`[DockerManager] Tar.gz created successfully: ${outputPath}`);
        } else {
            console.error(`[DockerManager] Failed to create tar.gz`);
        }
        return success;
    }

    /**
     * 检查路径是否为目录
     */
    private async isDirectory(
        container: string,
        path: string
    ): Promise<boolean> {
        const { stdout, success } = await this.runDockerCommand(
            container,
            `[ -d "${path}" ] && echo 'is_dir' || echo 'not_dir'`
        );
        return success && stdout.trim() === 'is_dir';
    }

    /**
     * 移动文件
     */
    private async moveFile(
        container: string,
        sourcePath: string,
        targetPath: string
    ): Promise<boolean> {
        const command = `mv "${sourcePath}" "${targetPath}"`;
        const { success } = await this.runDockerCommand(container, command);
        if (success) {
            console.log(`[DockerManager] Moved: ${sourcePath} -> ${targetPath}`);
        } else {
            console.error(`[DockerManager] Failed to move: ${sourcePath} -> ${targetPath}`);
        }
        return success;
    }

    /**
     * 删除目录
     */
    private async removeDirectory(
        container: string,
        path: string
    ): Promise<boolean> {
        const command = `rm -rf "${path}"`;
        const { success } = await this.runDockerCommand(container, command);
        if (success) {
            console.log(`[DockerManager] Removed directory: ${path}`);
        } else {
            console.error(`[DockerManager] Failed to remove directory: ${path}`);
        }
        return success;
    }

    private getSafeManagedName(name: string, label: string): string {
        const normalized = name.trim();
        if (
            !normalized ||
            normalized !== name ||
            normalized === '.' ||
            normalized.includes('/') ||
            normalized.includes('\\') ||
            normalized.includes('..') ||
            normalized.startsWith('-') ||
            normalized.includes('\0') ||
            /['"$`;&|<>()[\]{}!*?\r\n\t]/.test(normalized)
        ) {
            throw new Error(`${label} 名称不合法`);
        }
        return normalized;
    }

    private buildManagedItemPath(basePath: string, name: string, label: string): string {
        const safeName = this.getSafeManagedName(name, label);
        return `${basePath}/${safeName}`;
    }

    private getSafeEvaluationResultFolder(folderPath: string): string {
        const normalized = folderPath.trim().replace(/\/+$/, '');
        const prefix = `${EVALUATION_LOGS_PATH}/`;
        if (!normalized.startsWith(prefix)) {
            throw new Error('评测结果路径不合法');
        }

        const folderName = normalized.slice(prefix.length);
        if (!/^[0-9]+_[a-f0-9]+$/.test(folderName)) {
            throw new Error('评测结果目录名称不合法');
        }

        return folderName;
    }

    private async removeManagedPath(
        container: string,
        targetPath: string,
        missingMessage: string,
        successMessage: string,
    ): Promise<{ success: boolean; message: string }> {
        const safeTargetPath = this.escapeForDoubleQuotes(targetPath);
        const command = `sh -c 'if [ ! -e "${safeTargetPath}" ]; then echo not_exists; exit 0; fi; rm -rf -- "${safeTargetPath}" && echo deleted'`;
        const { stdout, success, stderr } = await this.runDockerCommand(container, command, 120000);

        if (!success) {
            return {
                success: false,
                message: stderr || '删除失败',
            };
        }

        if (stdout.includes('not_exists')) {
            return {
                success: false,
                message: missingMessage,
            };
        }

        return {
            success: true,
            message: successMessage,
        };
    }

    /**
     * 扁平化目录：如果目录下只有一个子目录，将子目录内容提升到父目录
     * 例如：dataset/20260325/20260325/ -> dataset/20260325/
     */
    private async flattenDirectory(
        container: string,
        path: string
    ): Promise<{ flattened: boolean; message: string }> {
        console.log(`[DockerManager] Checking if directory needs flattening: ${path}`);
        
        // 列出目录内容
        const entries = await this.listDirectory(container, path);
        console.log(`[DockerManager] Directory entries: ${entries.join(', ')}`);
        
        // 如果目录为空，返回错误
        if (entries.length === 0) {
            return {
                flattened: false,
                message: '解压后的文件夹为空',
            };
        }
        
        // 如果只有一个条目且是目录，进行扁平化
        if (entries.length === 1) {
            const singleEntry = entries[0];
            const fullPath = `${path}/${singleEntry}`;
            
            if (await this.isDirectory(container, fullPath)) {
                console.log(`[DockerManager] Single subdirectory detected: ${singleEntry}`);
                
                // 列出子目录内容
                const subEntries = await this.listDirectory(container, fullPath);
                console.log(`[DockerManager] Subdirectory entries: ${subEntries.join(', ')}`);
                
                // 将子目录内容移动到父目录
                let moveSuccess = true;
                for (const entry of subEntries) {
                    const sourcePath = `${fullPath}/${entry}`;
                    const targetPath = `${path}/${entry}`;
                    
                    if (!(await this.moveFile(container, sourcePath, targetPath))) {
                        moveSuccess = false;
                        break;
                    }
                }
                
                if (!moveSuccess) {
                    return {
                        flattened: false,
                        message: '扁平化目录时移动文件失败',
                    };
                }
                
                // 删除空子目录
                if (!(await this.removeDirectory(container, fullPath))) {
                    console.warn(`[DockerManager] Failed to remove empty subdirectory: ${fullPath}`);
                    // 不移除成功，只是警告
                }
                
                return {
                    flattened: true,
                    message: `已自动扁平化目录结构：${singleEntry}/ -> /`,
                };
            }
        }
        
        // 不需要扁平化
        return {
            flattened: false,
            message: '目录结构正常，无需扁平化',
        };
    }

    /**
     * 检查数据集格式
     * - 检查是否有子目录（要求扁平结构）
     * - 检查是否只包含 .json 文件
     * - SFT/DPO 检查是否包含 dataset_info.json（仅警告）
     * @returns { valid: boolean, message?: string, warning?: string }
     */
    private async checkDatasetFormat(
        container: string,
        path: string,
        datasetType: 'raw' | 'sft' | 'dpo'
    ): Promise<{ valid: boolean; message?: string; warning?: string }> {
        console.log(`[DockerManager] Checking dataset format: ${path} (type: ${datasetType})`);
        
        const errors: string[] = [];
        let warning = '';
        
        // 获取目录内容
        const entries = await this.listDirectory(container, path);
        console.log(`[DockerManager] Format check - entries: ${entries.join(', ')}`);
        
        if (entries.length === 0) {
            return {
                valid: false,
                message: '数据集文件夹为空',
            };
        }
        
        // 检查是否有子目录（要求扁平结构）
        const subDirectories: string[] = [];
        for (const entry of entries) {
            const fullPath = `${path}/${entry}`;
            if (await this.isDirectory(container, fullPath)) {
                subDirectories.push(entry);
            }
        }
        
        if (subDirectories.length > 0) {
            errors.push(`检测到子目录：${subDirectories.join(', ')}\n请将所有文件直接放在压缩包根目录，不要包含子文件夹`);
        }
        
        // 检查是否只包含 .json 文件
        const invalidFiles: string[] = [];
        let hasDatasetInfo = false;
        
        for (const entry of entries) {
            // 跳过子目录（已经在上面检查过）
            const fullPath = `${path}/${entry}`;
            if (await this.isDirectory(container, fullPath)) {
                continue;
            }
            
            // 检查是否是 dataset_info.json
            if (entry === 'dataset_info.json') {
                hasDatasetInfo = true;
                continue;
            }
            
            // 检查文件扩展名
            const lowerName = entry.toLowerCase();
            if (!lowerName.endsWith('.json')) {
                invalidFiles.push(entry);
            }
        }
        
        if (invalidFiles.length > 0) {
            // 只显示前 10 个不符合要求的文件
            const displayFiles = invalidFiles.slice(0, 10);
            const moreCount = invalidFiles.length > 10 ? ` 等共${invalidFiles.length}个文件` : '';
            errors.push(`检测到非 JSON 文件：${displayFiles.join(', ')}${moreCount}\n数据集只能包含 .json 格式的数据文件`);
        }
        
        // raw 原始数据不要求 dataset_info.json；SFT/DPO 缺失时仅提示警告。
        if (datasetType !== 'raw' && !hasDatasetInfo) {
            warning = '（注意：缺少 dataset_info.json 文件，建议添加）';
            console.log(`[DockerManager] Warning: dataset_info.json not found in ${path}`);
        }
        
        // 如果有错误，返回失败
        if (errors.length > 0) {
            return {
                valid: false,
                message: errors.join('；'),
            };
        }
        
        // 检查通过
        return {
            valid: true,
            warning,
        };
    }

    /**
     * 上传数据集
     * @returns 成功返回 true，失败返回错误信息
     */
    async uploadDataset(
        container: string,
        datasetType: 'raw' | 'sft' | 'dpo',
        datasetName: string,
        fileBuffer: Buffer,
        filename: string
    ): Promise<{ success: boolean; message: string; warning?: string }> {
        console.log(`[DockerManager] Uploading dataset: ${datasetName} (type: ${datasetType}) to ${container}`);
        
        // 1. 检查数据集是否已存在
        const exists = await this.checkDatasetExists(container, datasetType, datasetName);
        console.log(`[DockerManager] Dataset exists check: ${exists}`);
        if (exists) {
            return {
                success: false,
                message: `数据集 "${datasetName}" 已存在，请先删除或使用其他名称`,
            };
        }
        
        const basePath = DATASET_PATHS[datasetType];
        console.log(`[DockerManager] Base path: ${basePath}`);
        if (!basePath) {
            return {
                success: false,
                message: `无效的数据集类型: ${datasetType}`,
            };
        }
        
        // 使用相同的时间戳生成临时文件路径
        const timestamp = Date.now();
        const safeFilename = filename.replace(/[^\w.-]/g, '_');
        const tempContainerPath = `/tmp/upload_${timestamp}_${safeFilename}`;
        const targetPath = `${basePath}/${datasetName}`;
        console.log(`[DockerManager] Target path: ${targetPath}`);
        let tempLocalPath = '';
        
        try {
            // 2. 验证文件扩展名（只允许 tar 格式）
            const lowerFilename = filename.toLowerCase();
            const allowedExtensions = ['.tar', '.tar.gz', '.tgz'];
            const isValidExtension = allowedExtensions.some(ext => lowerFilename.endsWith(ext));
            
            console.log(`[DockerManager] Filename: ${filename}, Valid extension: ${isValidExtension}`);
            
            if (!isValidExtension) {
                return {
                    success: false,
                    message: `不支持的文件格式 "${filename}"。请上传 .tar、.tar.gz 或 .tgz 格式的压缩包`,
                };
            }
            
            // 3. 保存到本地临时文件（使用相同的时间戳）
            const fs = await import('fs');
            const os = await import('os');
            tempLocalPath = path.join(os.tmpdir(), `upload_${timestamp}_${safeFilename}`);
            fs.writeFileSync(tempLocalPath, fileBuffer);
            console.log(`[DockerManager] Saved to temp file: ${tempLocalPath}, size: ${fileBuffer.length} bytes`);
            
            // 4. 复制到容器并解压
            console.log(`[DockerManager] Copying to container: ${tempLocalPath} -> ${tempContainerPath}`);
            
            // 复制到容器
            const copied = await this.copyToContainer(container, tempLocalPath, tempContainerPath);
            console.log(`[DockerManager] Copy result: ${copied}`);
            if (!copied) {
                throw new Error('Failed to copy file to container');
            }
            
            // 解压
            console.log(`[DockerManager] Extracting: ${tempContainerPath} -> ${targetPath}`);
            const extracted = await this.extractArchive(container, tempContainerPath, targetPath);
            console.log(`[DockerManager] Extract result: ${extracted}`);
            if (!extracted) {
                throw new Error('Failed to extract archive');
            }
            
            // 删除容器中的临时压缩包
            await this.runDockerCommand(container, `rm -f "${tempContainerPath}"`);
            
            // 6. 扁平化目录（处理嵌套目录结构）
            const { flattened, message: flattenMessage } = await this.flattenDirectory(container, targetPath);
            console.log(`[DockerManager] Flatten result: ${flattened}, ${flattenMessage}`);
            
            // 如果扁平化检测到空目录，删除上传内容并返回错误
            if (flattenMessage.includes('为空')) {
                await this.removeDirectory(container, targetPath);
                return {
                    success: false,
                    message: `上传失败：${flattenMessage}`,
                };
            }
            
            // 7. 文件格式检查
            const formatCheck = await this.checkDatasetFormat(container, targetPath, datasetType);
            if (!formatCheck.valid) {
                // 格式检查失败，删除上传内容
                await this.removeDirectory(container, targetPath);
                return {
                    success: false,
                    message: `上传失败：${formatCheck.message}`,
                };
            }
            
            console.log(`[DockerManager] Dataset uploaded successfully: ${datasetName}`);
            
            // 构建成功消息
            let message = `数据集 "${datasetName}" 上传成功`;
            if (flattened) {
                message += `（${flattenMessage}）`;
            }
            
            return {
                success: true,
                message,
                warning: formatCheck.warning || undefined,
            };
            
        } catch (error: any) {
            console.error(`[DockerManager] Upload failed: ${error.message}`);
            await this.runDockerCommand(container, `rm -f "${tempContainerPath}"`);
            await this.removeDirectory(container, targetPath);
            return {
                success: false,
                message: `上传失败: ${error.message}`,
            };
        } finally {
            if (tempLocalPath) {
                const fs = await import('fs');
                try {
                    if (fs.existsSync(tempLocalPath)) {
                        fs.unlinkSync(tempLocalPath);
                    }
                } catch (cleanupError: any) {
                    console.warn(`[DockerManager] Failed to remove temp file ${tempLocalPath}: ${cleanupError.message}`);
                }
            }
        }
    }

    /**
     * 下载数据集为 zip 文件
     * @returns zip 文件的 buffer，失败返回 null
     */
    async downloadDatasetAsTar(
        container: string,
        datasetType: 'raw' | 'sft' | 'dpo',
        datasetName: string
    ): Promise<{ buffer: Buffer | null; filename: string; message: string }> {
        console.log(`[DockerManager] Downloading dataset: ${datasetName} from ${container}`);
        
        const basePath = DATASET_PATHS[datasetType];
        const sourcePath = `${basePath}/${datasetName}`;
        const tarFilename = `${datasetName}.tar.gz`;
        const containerTarPath = `/tmp/${tarFilename}_${Date.now()}`;
        const fs = await import('fs');
        
        try {
            // 1. 检查数据集是否存在
            const exists = await this.checkPathExists(container, sourcePath);
            if (!exists) {
                return {
                    buffer: null,
                    filename: '',
                    message: `数据集 "${datasetName}" 不存在`,
                };
            }
            
            // 2. 在容器中创建 tar.gz 文件
            const created = await this.createTar(container, sourcePath, containerTarPath);
            if (!created) {
                throw new Error('Failed to create tar.gz file in container');
            }
            
            // 3. 复制 tar.gz 文件到本地
            const tempLocalPath = `/tmp/download_${Date.now()}_${tarFilename}`;
            const copied = await this.copyFromContainer(container, containerTarPath, tempLocalPath);
            if (!copied) {
                throw new Error('Failed to copy tar.gz file from container');
            }
            
            // 4. 读取 tar.gz 文件
            const buffer = fs.readFileSync(tempLocalPath);
            
            // 5. 清理临时文件
            fs.unlinkSync(tempLocalPath);
            await this.runDockerCommand(container, `rm -f "${containerTarPath}"`);
            
            console.log(`[DockerManager] Dataset downloaded successfully: ${datasetName}, size: ${buffer.length} bytes`);
            return {
                buffer,
                filename: tarFilename,
                message: '下载成功',
            };
            
        } catch (error: any) {
            console.error(`[DockerManager] Download failed: ${error.message}`);
            return {
                buffer: null,
                filename: '',
                message: `下载失败: ${error.message}`,
            };
        }
    }

    /**
     * 安全删除数据集：只允许删除固定数据集根目录下的单个目录。
     */
    async deleteDataset(
        container: string,
        datasetType: 'raw' | 'sft' | 'dpo',
        datasetName: string,
    ): Promise<{ success: boolean; message: string }> {
        try {
            const basePath = DATASET_PATHS[datasetType];
            const targetPath = this.buildManagedItemPath(basePath, datasetName, '数据集');
            console.log(`[DockerManager] Deleting dataset: ${container}:${targetPath}`);

            return await this.removeManagedPath(
                container,
                targetPath,
                `数据集 "${datasetName}" 不存在`,
                `数据集 "${datasetName}" 已删除`,
            );
        } catch (error: any) {
            return {
                success: false,
                message: error.message || '删除数据集失败',
            };
        }
    }

    /**
     * 安全删除模型：只允许删除固定模型根目录下的单个目录。
     */
    async deleteModel(
        container: string,
        modelType: 'base_train' | 'batch_trained' | 'daily_trained' | 'inference',
        modelName: string,
        modelPath?: string,
    ): Promise<{ success: boolean; message: string }> {
        try {
            const allowedBasePaths = modelType === 'daily_trained'
                ? Object.values(DPO_MODEL_PATHS)
                : [MODEL_PATHS[modelType]];
            const requestedPath = modelPath?.trim();
            const basePath = requestedPath && allowedBasePaths.includes(requestedPath as typeof allowedBasePaths[number])
                ? requestedPath
                : allowedBasePaths[0];
            const targetPath = this.buildManagedItemPath(basePath, modelName, '模型');
            console.log(`[DockerManager] Deleting model: ${container}:${targetPath}`);

            return await this.removeManagedPath(
                container,
                targetPath,
                `模型 "${modelName}" 不存在`,
                `模型 "${modelName}" 已删除`,
            );
        } catch (error: any) {
            return {
                success: false,
                message: error.message || '删除模型失败',
            };
        }
    }

    /**
     * 获取医疗评测文件列表（并行优化版）
     */
    async getMedicalTests(container: string): Promise<MedicalTestFile[]> {
        console.log(`[DockerManager] Getting evaluation tests from ${container}:${Object.values(TESTS_PATHS).join(', ')} (批量模式)`);
        
        try {
            const summaries = await this.listMedicalTestSummaries(container);
            const files = summaries.map((summary) => {
                let type: MedicalTestFile['type'] = 'other';
                let description = '';

                if (summary.metaContent) {
                    try {
                        const meta = JSON.parse(summary.metaContent);
                        description = meta.testType || '';
                    } catch {
                        // 元数据解析失败
                    }
                }

                if (summary.category === 'general') {
                    type = 'general';
                } else if (!description) {
                    if (summary.filename.startsWith('2021')) {
                        type = 'exam2021';
                    } else if (summary.filename.startsWith('2024')) {
                        type = 'exam2024';
                    } else if (summary.filename.startsWith('step')) {
                        type = 'usmle';
                    } else if (summary.filename.toLowerCase().includes('medbench')) {
                        type = 'medbench';
                    }
                }

                return {
                    filename: summary.filename,
                    type,
                    size: summary.size || 'unknown',
                    description,
                    category: summary.category,
                };
            });
            
            console.log(`[DockerManager] Found ${files.length} evaluation test files`);
            return files;
            
        } catch (error: any) {
            console.error(`[DockerManager] Error getting medical tests: ${error.message}`);
            return [];
        }
    }
    
    /**
     * 上传医疗评测文件
     */
    async uploadMedicalTest(
        container: string,
        testType: string,
        filename: string,
        fileBuffer: Buffer
    ): Promise<{ success: boolean; message: string }> {
        const testsPath = this.getTestsPathForType(testType);
        console.log(`[DockerManager] Uploading evaluation test: ${filename} to ${container}:${testsPath}`);
        
        // 检查文件扩展名
        const lowerFilename = filename.toLowerCase();
        if (!lowerFilename.endsWith('.json') && !lowerFilename.endsWith('.jsonl')) {
            return {
                success: false,
                message: '只支持上传 .json 或 .jsonl 格式的文件',
            };
        }
        
        // 检查文件大小（20MB）
        const MAX_SIZE = 20 * 1024 * 1024;
        if (fileBuffer.length > MAX_SIZE) {
            return {
                success: false,
                message: `文件大小超过 20MB 限制（当前: ${(fileBuffer.length / 1024 / 1024).toFixed(2)}MB）`,
            };
        }
        
        try {
            // 检查目录是否存在，不存在则创建
            const exists = await this.checkPathExists(container, testsPath);
            if (!exists) {
                const { success } = await this.runDockerCommand(
                    container,
                    `mkdir -p ${testsPath}`
                );
                if (!success) {
                    return {
                        success: false,
                        message: '无法创建评测目录',
                    };
                }
            }
            
            // 保存到本地临时文件
            const fs = await import('fs');
            const tempLocalPath = `/tmp/medical_test_${Date.now()}_${filename}`;
            fs.writeFileSync(tempLocalPath, fileBuffer);
            
            // 复制到容器
            const targetPath = `${testsPath}/${filename}`;
            const copied = await this.copyToContainer(container, tempLocalPath, targetPath);
            
            // 清理本地临时文件
            fs.unlinkSync(tempLocalPath);
            
            if (!copied) {
                return {
                    success: false,
                    message: '复制文件到容器失败',
                };
            }
            
            // 保存评测类型元数据
            const metaFilename = `${filename}.meta`;
            const metaContent = JSON.stringify({ testType });
            const tempMetaPath = `/tmp/medical_test_meta_${Date.now()}.json`;
            fs.writeFileSync(tempMetaPath, metaContent);
            const metaTargetPath = `${testsPath}/${metaFilename}`;
            await this.copyToContainer(container, tempMetaPath, metaTargetPath);
            fs.unlinkSync(tempMetaPath);
            
            console.log(`[DockerManager] Medical test uploaded successfully: ${filename}`);
            return {
                success: true,
                message: `文件 "${filename}" 上传成功`,
            };
            
        } catch (error: any) {
            console.error(`[DockerManager] Upload failed: ${error.message}`);
            return {
                success: false,
                message: `上传失败: ${error.message}`,
            };
        }
    }
    
    /**
     * 下载医疗评测文件
     */
    async downloadMedicalTest(
        container: string,
        filename: string
    ): Promise<{ buffer: Buffer | null; filename: string; message: string }> {
        try {
            const safeFilename = this.getSafeManagedName(filename, '评测文件');
            const summaries = await this.listMedicalTestSummaries(container);
            const summary = summaries.find((item) => item.filename === safeFilename);

            if (!summary) {
                return {
                    buffer: null,
                    filename: '',
                    message: `评测文件 "${safeFilename}" 不存在`,
                };
            }

            const sourcePath = `${summary.basePath}/${safeFilename}`;
            console.log(`[DockerManager] Downloading evaluation test: ${safeFilename} from ${container}:${summary.basePath}`);

            const fs = await import('fs');

            if (summary.category === 'general') {
                const tarFilename = `${safeFilename}.tar.gz`;
                const containerTarPath = `/tmp/${tarFilename}_${Date.now()}`;
                const created = await this.createTar(container, sourcePath, containerTarPath);
                if (!created) {
                    return {
                        buffer: null,
                        filename: '',
                        message: `评测目录 "${safeFilename}" 打包失败`,
                    };
                }

                const tempLocalPath = `/tmp/download_medical_test_${Date.now()}_${tarFilename}`;
                const copied = await this.copyFromContainer(container, containerTarPath, tempLocalPath);
                await this.runDockerCommand(container, `rm -f "${containerTarPath}"`);

                if (!copied) {
                    return {
                        buffer: null,
                        filename: '',
                        message: '从容器复制评测目录压缩包失败',
                    };
                }

                const buffer = fs.readFileSync(tempLocalPath);
                fs.unlinkSync(tempLocalPath);

                console.log(`[DockerManager] Evaluation test directory downloaded successfully: ${safeFilename}, size: ${buffer.length} bytes`);
                return {
                    buffer,
                    filename: tarFilename,
                    message: '下载成功',
                };
            }

            const safeSourcePath = this.escapeForDoubleQuotes(sourcePath);
            const { stdout, success } = await this.runDockerCommand(
                container,
                `sh -c 'test -f "${safeSourcePath}" && echo exists || echo not_exists'`
            );

            if (!success || stdout.includes('not_exists')) {
                return {
                    buffer: null,
                    filename: '',
                    message: `文件 "${safeFilename}" 不存在`,
                };
            }

            // 复制到本地临时文件
            const tempLocalPath = `/tmp/download_medical_test_${Date.now()}_${safeFilename}`;
            const copied = await this.copyFromContainer(container, sourcePath, tempLocalPath);
            
            if (!copied) {
                return {
                    buffer: null,
                    filename: '',
                    message: '从容器复制文件失败',
                };
            }
            
            // 读取文件
            const buffer = fs.readFileSync(tempLocalPath);
            
            // 清理临时文件
            fs.unlinkSync(tempLocalPath);
            
            console.log(`[DockerManager] Medical test downloaded successfully: ${safeFilename}, size: ${buffer.length} bytes`);
            return {
                buffer,
                filename: safeFilename,
                message: '下载成功',
            };
            
        } catch (error: any) {
            console.error(`[DockerManager] Download failed: ${error.message}`);
            return {
                buffer: null,
                filename: '',
                message: `下载失败: ${error.message}`,
            };
        }
    }

    /**
     * 安全删除医疗评测文件，同时清理对应的 .meta 文件。
     */
    async deleteMedicalTest(
        container: string,
        filename: string,
    ): Promise<{ success: boolean; message: string }> {
        try {
            const safeFilename = this.getSafeManagedName(filename, '评测文件');
            const summaries = await this.listMedicalTestSummaries(container);
            const summary = summaries.find((item) => item.filename === safeFilename);

            if (!summary) {
                return {
                    success: false,
                    message: `评测文件 "${safeFilename}" 不存在`,
                };
            }

            if (summary.category === 'general') {
                const targetPath = `${summary.basePath}/${safeFilename}`;
                console.log(`[DockerManager] Deleting general evaluation test directory: ${container}:${targetPath}`);

                return await this.removeManagedPath(
                    container,
                    targetPath,
                    `评测目录 "${safeFilename}" 不存在`,
                    `评测目录 "${safeFilename}" 已删除`,
                );
            }

            const lowerFilename = safeFilename.toLowerCase();
            if (!lowerFilename.endsWith('.json') && !lowerFilename.endsWith('.jsonl')) {
                return {
                    success: false,
                    message: '只允许删除 .json 或 .jsonl 评测文件',
                };
            }

            const basePath = summary.basePath;
            const targetPath = `${basePath}/${safeFilename}`;
            const metaPath = `${targetPath}.meta`;
            const safeTargetPath = this.escapeForDoubleQuotes(targetPath);
            const safeMetaPath = this.escapeForDoubleQuotes(metaPath);
            const command = `sh -c 'if [ ! -f "${safeTargetPath}" ]; then echo not_exists; exit 0; fi; rm -f -- "${safeTargetPath}" "${safeMetaPath}" && echo deleted'`;
            console.log(`[DockerManager] Deleting medical test: ${container}:${targetPath}`);

            const { stdout, success, stderr } = await this.runDockerCommand(container, command);
            if (!success) {
                return {
                    success: false,
                    message: stderr || '删除评测文件失败',
                };
            }
            if (stdout.includes('not_exists')) {
                return {
                    success: false,
                    message: `评测文件 "${safeFilename}" 不存在`,
                };
            }
            return {
                success: true,
                message: `评测文件 "${safeFilename}" 已删除`,
            };
        } catch (error: any) {
            return {
                success: false,
                message: error.message || '删除评测文件失败',
            };
        }
    }

    /**
     * 获取评测结果列表
     */
    async getEvaluationResults(container: string): Promise<EvaluationResult[]> {
        console.log(`[DockerManager] Getting evaluation results from ${container}:${EVALUATION_LOGS_PATH} (批量模式)`);
        
        try {
            const summaries = await this.listEvaluationResultSummaries(container, EVALUATION_LOGS_PATH);
            console.log(`[DockerManager] Found ${summaries.length} evaluation result folders`);

            // 过滤掉没有有效数据的结果，并按开始时间倒序排列
            const validResults = summaries
                .filter(r => r.startTime)
                .sort((a, b) => new Date(b.startTime).getTime() - new Date(a.startTime).getTime());
            
            console.log(`[DockerManager] Found ${validResults.length} valid evaluation results`);
            return validResults;
            
        } catch (error: any) {
            console.error(`[DockerManager] Error getting evaluation results: ${error.message}`);
            return [];
        }
    }

    /**
     * 安全删除评测结果目录。
     */
    async deleteEvaluationResult(
        container: string,
        folderPath: string,
    ): Promise<{ success: boolean; message: string }> {
        try {
            const folderName = this.getSafeEvaluationResultFolder(folderPath);
            const targetPath = `${EVALUATION_LOGS_PATH}/${folderName}`;
            console.log(`[DockerManager] Deleting evaluation result: ${container}:${targetPath}`);

            return await this.removeManagedPath(
                container,
                targetPath,
                `评测结果 "${folderName}" 不存在`,
                `评测结果 "${folderName}" 已删除`,
            );
        } catch (error: any) {
            return {
                success: false,
                message: error.message || '删除评测结果失败',
            };
        }
    }
    
    /**
     * 下载评测结果文件
     */
    async downloadEvaluationResult(
        container: string,
        folderPath: string,
        filename: string
    ): Promise<{ buffer: Buffer | null; filename: string; message: string }> {
        const folderName = this.getSafeEvaluationResultFolder(folderPath);
        const safeFilename = this.getSafeManagedName(filename, '评测结果文件');
        const sourcePath = `${EVALUATION_LOGS_PATH}/${folderName}/${safeFilename}`;
        const safeSourcePath = this.escapeForDoubleQuotes(sourcePath);
        console.log(`[DockerManager] Downloading evaluation result: ${sourcePath}`);
        
        try {
            // 检查文件是否存在
            const { success } = await this.runDockerCommand(
                container,
                `test -f "${safeSourcePath}" && echo 'exists' || echo 'not_exists'`
            );
            
            if (!success) {
                return {
                    buffer: null,
                    filename: '',
                    message: `文件 "${filename}" 不存在`,
                };
            }
            
            // 复制到本地临时文件
            const fs = await import('fs');
            const tempLocalPath = `/tmp/download_eval_result_${Date.now()}_${filename}`;
            const copied = await this.copyFromContainer(container, sourcePath, tempLocalPath);
            
            if (!copied) {
                return {
                    buffer: null,
                    filename: '',
                    message: '从容器复制文件失败',
                };
            }
            
            // 读取文件
            const buffer = fs.readFileSync(tempLocalPath);
            
            // 清理临时文件
            fs.unlinkSync(tempLocalPath);
            
            console.log(`[DockerManager] Evaluation result downloaded successfully: ${filename}`);
            return {
                buffer,
                filename,
                message: '下载成功',
            };
            
        } catch (error: any) {
            console.error(`[DockerManager] Download failed: ${error.message}`);
            return {
                buffer: null,
                filename: '',
                message: `下载失败: ${error.message}`,
            };
        }
    }
}

// 导出单例实例
export const dockerManager = new DockerManager();
