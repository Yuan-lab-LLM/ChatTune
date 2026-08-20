import { useEffect, useMemo, useRef, useState } from 'react';
import { ArrowRight, CheckCircle2, ClipboardList, HelpCircle, Sparkles } from 'lucide-react';
import { Button } from '@/components/ui/button.tsx';
import { Input } from '@/components/ui/input.tsx';
import { trpc } from '@/api/trpc';
import { useEnvironmentConfig } from '@/hooks/useEnvironmentConfig';

export interface AgentDatasetOption {
    nodeId?: string;
    nodeName?: string;
    name: string;
    type?: string;
    path?: string;
    size?: string;
    description?: string;
}

export interface AgentModelOption {
    nodeId?: string;
    nodeName?: string;
    name: string;
    type?: string;
    path?: string;
    size?: string;
    description?: string;
    merged?: boolean;
    checkpoints?: string[];
}

interface AgentFileOption {
    name: string;
    path: string;
    directory?: string;
    type?: string;
}

export interface AgentWaitingPrompt {
    title: string;
    body: string;
    kind: 'type' | 'param' | 'choice' | 'waiting';
    quickReplies: string[];
    fields: string[];
    resourceContainer?: string;
    options?: string[];
    knownParams?: Record<string, string>;
    scriptName?: string;
    detectedFormat?: string;
    excludedInputFolders?: string[];
}

interface AgentUiProtocol {
    version?: string;
    type?: string;
    agent?: string;
    message?: string;
    currentStage?: string;
    stages?: Record<string, {
        status?: string;
        container?: string;
        pid?: string | number;
        message?: string;
        model_path?: string;
        progress_percent?: number | string;
    }>;
    kind?: string;
    title?: string;
    options?: string[];
    fields?: string[];
    requiredParams?: string[];
    missingParams?: string[];
    container?: string;
    launchMode?: string;
    isMultinode?: boolean;
    knownParams?: Record<string, string>;
    script?: string;
    scriptName?: string;
    detectedFormat?: string;
    errorReason?: string;
    containerPath?: string;
    input_folder?: string;
    inputFolder?: string;
    selectedInputFolder?: string;
    currentArgs?: Record<string, unknown>;
}

export type AgentProtocol = AgentUiProtocol;

const KNOWN_FIELDS = [
    'model_path',
    'dataset_dir',
    'dataset_name',
    'input_folder',
    'schedule_time',
    'data_type',
    'strategy',
    'model_fir',
    'model_sec',
    'CKPT_PATH',
    'train_files',
    'val_files',
];

const DATA_TYPE_OPTIONS = [
    { value: 'sft', label: 'SFT' },
    { value: 'dpo', label: 'DPO' },
];

const STRATEGY_OPTIONS = [
    { value: 'inspection', label: '检查' },
    { value: 'diagnosis', label: '诊断' },
    { value: 'prescription', label: '处方' },
];

const GENERAL_PREPROCESS_FORMATS = new Set(['openai', 'sharegpt', 'sft', 'dpo', 'text']);

const normalizeScriptName = (value?: string): string => (value || '').trim().toLowerCase();

const asRecord = (value: unknown): Record<string, unknown> =>
    value && typeof value === 'object' && !Array.isArray(value)
        ? value as Record<string, unknown>
        : {};

const stringValue = (value: unknown): string =>
    typeof value === 'string' ? value.trim() : '';

const normalizeDatasetPath = (path: string) =>
    path.replaceAll(String.fromCharCode(92), '/').replace(/\/+$/, '');

const firstString = (...values: unknown[]): string => {
    for (const value of values) {
        const text = stringValue(value);
        if (text) {
            return text;
        }
    }
    return '';
};

const knownParamsFromProtocol = (protocol: AgentUiProtocol): Record<string, string> | undefined => {
    const knownParams = { ...(protocol.knownParams || {}) };
    const currentArgs = asRecord(protocol.currentArgs);
    const cliParams = asRecord(currentArgs.cli_params_to_update);
    const additionalArgs = asRecord(currentArgs.additional_args);
    const inputFolder = firstString(
        knownParams.input_folder,
        protocol.inputFolder,
        protocol.selectedInputFolder,
        currentArgs.inputFolder,
        currentArgs.selectedInputFolder,
        cliParams.input_folder,
        additionalArgs.input_folder,
    );
    if (inputFolder) {
        knownParams.input_folder = inputFolder;
    }
    return Object.keys(knownParams).length > 0 ? knownParams : undefined;
};

const excludedInputFoldersFromProtocol = (protocol: AgentUiProtocol): string[] => {
    if (protocol.errorReason !== 'unknown_data_format') {
        return [];
    }

    const currentArgs = asRecord(protocol.currentArgs);
    return unique([
        protocol.containerPath,
        protocol.input_folder,
        protocol.inputFolder,
        protocol.selectedInputFolder,
        currentArgs.input_folder,
        currentArgs.inputFolder,
        currentArgs.selected_input_folder,
        currentArgs.selectedInputFolder,
    ].map((value) => normalizeDatasetPath(stringValue(value))));
};

const INTERACTIVE_AGENT_NAMES = [
    'evaluator',
    'dataprocessor',
    'trainer',
    'inference',
    'monitor',
    'analysis',
];

const isInteractiveAgent = (agent?: string): boolean => {
    if (!agent) {
        return true;
    }

    const normalizedAgent = agent.toLowerCase();
    return INTERACTIVE_AGENT_NAMES.some((name) => normalizedAgent.includes(name));
};

const unique = (items: string[]) =>
    items
        .map((item) => item.trim())
        .filter(Boolean)
        .filter((item, index, list) => list.indexOf(item) === index);

const isPlaceholderParamReply = (value: string): boolean => {
    const normalized = value.trim().replace(/：/g, ':');
    const match = normalized.match(/^([A-Za-z_][\w.-]*)\s*=\s*(.+)$/);
    if (!match) {
        return false;
    }

    const rawValue = match[2].trim().replace(/^['"`]|['"`]$/g, '');
    return !rawValue ||
        /[<>]/.test(rawValue) ||
        /^(?:xxx|xx|x|待提供|未提供|未知|数据集名称|数据目录路径|模型路径)$/i.test(rawValue);
};

const isDirectReplyOption = (value: string): boolean => {
    if (/[<>]/.test(value) || isPlaceholderParamReply(value)) {
        return false;
    }

    return [
        'lora批量训练',
        '全参批量训练',
        '增强训练',
        'grpo训练',
        '单模型评估',
        '双模型评估',
        'ckpt评估',
        '是',
        '否',
    ].includes(value) ||
        value.includes('=') ||
        value.endsWith('评估');
};

const extractAgentProtocol = (text: string): AgentUiProtocol | null => {
    const markerIndex = text.lastIndexOf('[协议]');
    if (markerIndex < 0) {
        return null;
    }

    const afterMarker = text.slice(markerIndex + '[协议]'.length).trim();
    const jsonStart = afterMarker.indexOf('{');
    if (jsonStart < 0) {
        return null;
    }

    const jsonSource = afterMarker.slice(jsonStart);
    let depth = 0;
    let inString = false;
    let escaped = false;
    let jsonEnd = -1;

    for (let index = 0; index < jsonSource.length; index += 1) {
        const char = jsonSource[index];
        if (escaped) {
            escaped = false;
            continue;
        }
        if (char === '\\') {
            escaped = true;
            continue;
        }
        if (char === '"') {
            inString = !inString;
            continue;
        }
        if (inString) {
            continue;
        }
        if (char === '{') {
            depth += 1;
        }
        if (char === '}') {
            depth -= 1;
            if (depth === 0) {
                jsonEnd = index + 1;
                break;
            }
        }
    }

    if (jsonEnd < 0) {
        return null;
    }

    const jsonText = jsonSource.slice(0, jsonEnd);
    try {
        return JSON.parse(jsonText) as AgentUiProtocol;
    } catch (error) {
        console.warn('Failed to parse agent UI protocol:', error);
        return null;
    }
};

const fieldsFromProtocol = (protocol: AgentUiProtocol, message: string): string[] => {
    const normalizeTrainingFields = (fields: string[]) => {
        const hasEnhancedTrainingFields =
            fields.includes('dataset_dir') ||
            fields.includes('dataset_name') ||
            /(?:增强训练|dpo_train_launcher)/i.test(`${protocol.title || ''}\n${message}`);
        if (!hasEnhancedTrainingFields) {
            return fields;
        }
        return fields.filter((field) => !['train_files', 'val_files'].includes(field));
    };

    if (protocol.fields?.length) {
        return normalizeTrainingFields(protocol.fields.filter((field) => KNOWN_FIELDS.includes(field)));
    }

    if (protocol.missingParams?.length) {
        return normalizeTrainingFields(protocol.missingParams.filter((field) => KNOWN_FIELDS.includes(field)));
    }

    if (protocol.requiredParams?.length) {
        return normalizeTrainingFields(protocol.requiredParams.filter((field) => KNOWN_FIELDS.includes(field)));
    }

    if (protocol.kind === 'data_preprocess_params') {
        return ['data_type', 'strategy'];
    }


    if (protocol.kind?.includes('choice') || protocol.options?.length) {
        return [];
    }

    return extractFields(message);
};

const normalizePercent = (value: unknown): number | undefined => {
    if (value === undefined || value === null || value === '') {
        return undefined;
    }

    const parsed =
        typeof value === 'number'
            ? value
            : Number(String(value).replace(/%$/, '').trim());
    if (!Number.isFinite(parsed)) {
        return undefined;
    }

    return Math.max(0, Math.min(100, Math.round(parsed)));
};

const displayWorkflowStatusMessage = (protocol: AgentUiProtocol): string | null => {
    if (protocol.type !== 'workflow_status' || !protocol.message?.trim()) {
        return null;
    }

    let message = protocol.message.trim();
    const currentStage = protocol.currentStage;
    const stage = currentStage ? protocol.stages?.[currentStage] : undefined;
    const progressPercent = normalizePercent(stage?.progress_percent);

    if (progressPercent !== undefined) {
        const progressLinePattern = /(-\s*进度\s*[:：]\s*)`?[^`\n]*%`?/;
        if (progressLinePattern.test(message)) {
            message = message.replace(progressLinePattern, `$1\`${progressPercent}%\``);
        } else if (/当前阶段详情[:：]/.test(message)) {
            message = message.replace(
                /(当前阶段详情[:：]\s*)/,
                `$1\n- 进度：\`${progressPercent}%\``,
            );
        }
    }

    if (stage?.message && !message.includes(stage.message)) {
        message = message.replace(
            /(当前阶段详情[:：]\s*)/,
            `$1\n- 阶段说明：\`${stage.message}\``,
        );
    }

    return message;
};

export const cleanupAgentWaitingText = (text: string): string => {
    let content = text.trim();
    const protocol = extractAgentProtocol(content);

    const workflowStatusMessage = protocol ? displayWorkflowStatusMessage(protocol) : null;
    if (workflowStatusMessage) {
        content = workflowStatusMessage;
    }

    content = content.replace(/^\[[^\]]+\]\s*[^：:\n]+执行结果[:：]\s*/m, '');
    content = content.replace(/^[a-zA-Z_]+_\[User-[A-Z0-9]+\][:：]\s*/m, '');
    content = content.replace(/^\[等待用户回复\|[^\]]+\]\s*/i, '');
    content = content.replace(
        /^以下内容是发给用户的补充信息提示，请原样展示给用户，不要代替用户回答。\s*/m,
        '',
    );
    content = content.replace(/\n?\[协议\]\s*\{[\s\S]*$/m, '');
    content = content.replace(/\n?\[耗时\][\s\S]*$/m, '');
    content = content.replace(/\[(?:类型请求|参数请求|等待用户选择)\]\s*/g, '');
    content = content.replace(/你可以直接回复，例如[:：][\s\S]*$/m, '');
    content = content.replace(/你可以直接回复[:：][\s\S]*$/m, '');
    content = content.replace(/请直接回复[:：][\s\S]*$/m, '');
    content = content.replace(/当前参数[:：][\s\S]*$/m, '');

    return content.trim();
};


const extractOptionsFromParentheses = (text: string): string[] => {

    const match = text.match(/（([^）]+)）/);
    if (!match) return [];
    return unique(match[1].split(/[、,，]|或/g));
};

const extractBacktickOptions = (text: string): string[] => {
    const values = [...text.matchAll(/`([^`]+)`/g)].map((match) => match[1]);
    return unique(values.filter(isDirectReplyOption));
};

const extractFields = (text: string): string[] =>
    KNOWN_FIELDS.filter((field) => new RegExp(`\\b${field}\\b`, 'i').test(text));

const resourceContainerFromProtocol = (protocol: AgentUiProtocol): string | undefined => {
    const container = protocol.container?.trim();
    if (container) {
        return container;
    }
    return undefined;
};

const inferQuickReplies = (text: string, fields: string[]): string[] => {
    const replies: string[] = [];
    if (text.includes('训练类型') || text.includes('训练方式')) {
        replies.push('lora批量训练', '全参批量训练', '增强训练', 'grpo训练', '双机 LoRA SFT', '双机增强训练');
    }

    if (text.includes('评估方式') || text.includes('评估类型') || text.includes('评测方式')) {
        replies.push('单模型评估', '双模型评估', 'ckpt评估');
    }

    if (
        text.includes('是否') ||
        text.includes('继续执行') ||
        text.includes('[等待用户选择]')
    ) {
        replies.push('是', '否');
    }

    return unique([...extractOptionsFromParentheses(text), ...extractBacktickOptions(text), ...replies])
        .filter(isDirectReplyOption)
        .filter((reply) => {
            if (fields.includes('data_type') && fields.includes('strategy')) {
                return !/(?:data_type|strategy)\s*=/.test(reply);
            }
            if (fields.includes('schedule_time')) {
                return !/^schedule_time\s*=/.test(reply);
            }
            if (fields.includes('dataset_name')) {
                return !/^dataset_name\s*=/.test(reply);
            }
            return true;
        });
};

export const parseAgentWaitingPrompt = (
    text: string,
    metadataProtocol?: AgentProtocol | null,
): AgentWaitingPrompt | null => {
    const rawText = text.trim();
    const protocol = metadataProtocol || extractAgentProtocol(rawText);
    if (protocol) {
        if (protocol.type !== 'need_input' || !isInteractiveAgent(protocol.agent)) {
            return null;
        }

        const protocolMessage = protocol.message || cleanupAgentWaitingText(rawText);
        const fields = fieldsFromProtocol(protocol, protocolMessage);
        const scriptName = normalizeScriptName(protocol.scriptName || protocol.script);
        const detectedFormat = (protocol.detectedFormat || '').trim().toLowerCase();
        const knownParams = knownParamsFromProtocol(protocol);
        const isGeneralPreprocessDataTypePrompt =
            scriptName === 'data_preprocessing' &&
            fields.length === 1 &&
            fields.includes('data_type') &&
            GENERAL_PREPROCESS_FORMATS.has(detectedFormat);
        const isPreprocessDataTypeStrategyPrompt =
            scriptName === 'data_preprocessing' &&
            fields.includes('data_type') &&
            fields.includes('strategy');
        const explicitOptions = (protocol.options || []).filter(isDirectReplyOption);
        const quickReplies = unique(
            explicitOptions.length > 0
                ? explicitOptions
                : inferQuickReplies(protocolMessage, fields),
        );
        if (protocol.kind === 'monitor_params' && fields.length === 0 && quickReplies.length === 0) {
            return null;
        }
        const kind: AgentWaitingPrompt['kind'] =
            protocol.kind?.includes('choice')
                ? 'choice'
                : fields.length > 0
                  ? 'param'
                  : 'waiting';

        return {
            title:
                protocol.title ||
                (kind === 'choice'
                    ? '确认下一步'
                    : kind === 'param'
                      ? '补充必要信息'
                      : '需要你回复'),
            body: isGeneralPreprocessDataTypePrompt
                ? `已识别输入格式为 ${detectedFormat}，请选择预处理输出数据类型。`
                : isPreprocessDataTypeStrategyPrompt
                  ? '请选择预处理输出数据类型和医疗数据处理方向。'
                  : cleanupAgentWaitingText(protocolMessage),
            kind,
            quickReplies,
            fields,
            resourceContainer: resourceContainerFromProtocol(protocol),
            options: protocol.options || [],
            knownParams,
            scriptName,
            detectedFormat,
            excludedInputFolders: excludedInputFoldersFromProtocol(protocol),
        };
    }

    const wrapperMatch = rawText.match(/^\[等待用户回复\|([^\]]+)\]/);
    const hasWaitingMarker =
        wrapperMatch ||
        /\[(?:类型请求|参数请求|等待用户选择)\]/.test(rawText);

    if (!hasWaitingMarker) {
        return null;
    }

    const body = cleanupAgentWaitingText(rawText);
    const fields = extractFields(rawText);
    const markerKind = rawText.includes('[类型请求]')
        ? 'type'
        : rawText.includes('[参数请求]')
          ? 'param'
          : rawText.includes('[等待用户选择]')
            ? 'choice'
            : 'waiting';

    const category = wrapperMatch?.[1]?.trim();
    const title =
        category ||
        (markerKind === 'type'
            ? '请选择类型'
            : markerKind === 'param'
              ? '补充必要信息'
              : markerKind === 'choice'
                ? '确认下一步'
                : '需要你回复');

    return {
        title,
        body,
        kind: markerKind,
        quickReplies: inferQuickReplies(rawText, fields),
        fields,
    };
};

interface AgentWaitingCardProps {
    prompt: AgentWaitingPrompt;
    onReply?: (text: string) => void;
    persistenceKey?: string;
    datasets?: AgentDatasetOption[];
    models?: AgentModelOption[];
    onRefreshDatasets?: (containerName?: string) => Promise<AgentDatasetOption[]>;
    onRefreshModels?: (containerName?: string) => Promise<AgentModelOption[]>;
    resourceGroupId?: string;
}

const readSubmittedState = (persistenceKey?: string): boolean => {
    if (!persistenceKey || typeof window === 'undefined') {
        return false;
    }

    try {
        return window.localStorage.getItem(persistenceKey) === '1';
    } catch {
        return false;
    }
};

const writeSubmittedState = (persistenceKey?: string) => {
    if (!persistenceKey || typeof window === 'undefined') {
        return;
    }

    try {
        window.localStorage.setItem(persistenceKey, '1');
    } catch {
        // Ignore storage failures so the card still works in memory-only mode.
    }
};

const fieldLabel = (field: string): string => {
    const labels: Record<string, string> = {
        model_path: '模型路径',
        dataset_dir: '数据集目录',
        dataset_name: '数据集名称',
        input_folder: '数据路径',
        schedule_time: '训练时间',
        data_type: '数据类型',
        strategy: '任务方向',
        model_fir: '第一个模型路径',
        model_sec: '第二个模型路径',
        CKPT_PATH: 'Checkpoint 路径',
        train_files: '训练文件',
        val_files: '验证文件',
    };
    return labels[field] || field;
};

const fieldPlaceholder = (field: string): string => {
    const placeholders: Record<string, string> = {
        model_path: '/home/workspace/models/base/example-model-name',
        dataset_dir: '/home/workspace/dataset_daily_train/xxx',
        dataset_name: 'for_train_xxx',
        input_folder: '/home/workspace/dataset/xxx',
        schedule_time: '30 或 14:00:00',
        data_type: 'sft 或 dpo',
        strategy: 'diagnosis / inspection / prescription',
        model_fir: '/home/workspace/models/xxx',
        model_sec: '/home/workspace/models/yyy',
        CKPT_PATH: '/home/workspace/checkpoints/xxx',
        train_files: '/home/workspace/data/train.parquet',
        val_files: '/home/workspace/data/val.parquet',
    };
    return placeholders[field] || `${field}=...`;
};

const joinPathAndNameValue = (path: string, name: string) => {
    const normalizedPath = path.replace(/\/+$/, '');
    const normalizedName = name.replace(/^\/+/, '');
    if (normalizedPath.endsWith(`/${normalizedName}`)) {
        return normalizedPath;
    }
    return `${normalizedPath}/${normalizedName}`;
};

const datasetOptionPath = (dataset: AgentDatasetOption) => {
    if (!dataset.path) {
        return `/home/workspace/dataset/${dataset.name}`;
    }
    return joinPathAndNameValue(dataset.path, dataset.name);
};

const isPreprocessInputDataset = (dataset: AgentDatasetOption, excludedInputFolders: Set<string>): boolean => {
    const type = (dataset.type || '').toLowerCase();
    const path = normalizeDatasetPath(datasetOptionPath(dataset));
    if (excludedInputFolders.has(path)) {
        return false;
    }
    return (
        type === 'raw' ||
        (/^\/home\/workspace\/dataset(?:\/|$)/.test(path) &&
            !/^\/home\/workspace\/dataset_(?:batch|daily)_train(?:\/|$)/.test(path))
    );
};

export function AgentWaitingCard({
    prompt,
    onReply,
    persistenceKey,
    datasets = [],
    models = [],
    onRefreshDatasets,
    onRefreshModels,
    resourceGroupId,
}: AgentWaitingCardProps) {
    const promptScriptName = normalizeScriptName(prompt.scriptName);
    const isDataPreprocessingPrompt = promptScriptName === 'data_preprocessing';
    const hasDataTypePicker = isDataPreprocessingPrompt && prompt.fields.includes('data_type');
    const hasDataTypeStrategyPicker =
        hasDataTypePicker && prompt.fields.includes('strategy');
    const hasDualModelPicker =
        prompt.fields.includes('model_fir') && prompt.fields.includes('model_sec');
    const hasEvaluationModelPicker =
        prompt.fields.includes('model_fir') || prompt.fields.includes('model_sec');
    const hasCheckpointPicker = prompt.fields.includes('CKPT_PATH');
    const hasModelPicker = prompt.fields.includes('model_path');
    const hasGrpoFilePicker =
        prompt.fields.includes('train_files') || prompt.fields.includes('val_files');
    const trainingPromptText = `${prompt.title}\n${prompt.body}`;
    const isEnhancedTrainingPrompt =
        prompt.fields.includes('dataset_dir') ||
        prompt.fields.includes('dataset_name') ||
        /(?:增强训练|dpo_train_launcher)/i.test(trainingPromptText);
    const isGrpoTrainingPrompt =
        !isEnhancedTrainingPrompt && (hasGrpoFilePicker || /grpo/i.test(trainingPromptText));
    const hasDatasetPicker =
        prompt.fields.includes('dataset_dir') ||
        prompt.fields.includes('dataset_name') ||
        prompt.fields.includes('input_folder');
    const shouldSubmitGrpoParamsTogether = isGrpoTrainingPrompt && hasModelPicker && hasGrpoFilePicker;
    const shouldSubmitModelAndDatasetTogether = hasModelPicker && hasDatasetPicker && !shouldSubmitGrpoParamsTogether;
    const shouldSubmitTrainingParamsTogether =
        shouldSubmitModelAndDatasetTogether && prompt.fields.includes('schedule_time');
    const requiresDpoDataset = /增强训练/.test(`${prompt.title}\n${prompt.body}`);
    const hasDatasetNameFileOptions =
        prompt.fields.length === 1 &&
        prompt.fields.includes('dataset_name') &&
        (prompt.options || []).length > 0;
    const datasetNameOptions = useMemo(
        () =>
            hasDatasetNameFileOptions
                ? (prompt.options || []).filter(Boolean)
                : [],
        [hasDatasetNameFileOptions, prompt.options],
    );
    const editableFields = useMemo(
        () =>
            prompt.fields.filter(
                (field) =>
                    !(
                        (hasDataTypePicker && field === 'data_type') ||
                        (hasDataTypeStrategyPicker && field === 'strategy') ||
                        (hasEvaluationModelPicker && ['model_fir', 'model_sec'].includes(field)) ||
                        (hasCheckpointPicker && field === 'CKPT_PATH') ||
                        (hasModelPicker && field === 'model_path') ||
                        (hasGrpoFilePicker && ['train_files', 'val_files'].includes(field)) ||
                        (hasDatasetPicker && ['dataset_dir', 'dataset_name', 'input_folder'].includes(field))
                    ),
            ),
        [hasCheckpointPicker, hasDataTypePicker, hasDataTypeStrategyPicker, hasDatasetPicker, hasEvaluationModelPicker, hasGrpoFilePicker, hasModelPicker, prompt.fields],
    );
    const [values, setValues] = useState<Record<string, string>>({});
    const [dataType, setDataType] = useState('');
    const [strategy, setStrategy] = useState('');
    const [selectedModel, setSelectedModel] = useState<AgentModelOption | null>(null);
    const [selectedModelFir, setSelectedModelFir] = useState<AgentModelOption | null>(null);
    const [selectedModelSec, setSelectedModelSec] = useState<AgentModelOption | null>(null);
    const [selectedDataset, setSelectedDataset] = useState<AgentDatasetOption | null>(null);
    const [selectedTrainFile, setSelectedTrainFile] = useState<AgentFileOption | null>(null);
    const [selectedValFile, setSelectedValFile] = useState<AgentFileOption | null>(null);
    const { defaultGrpoContainerName } = useEnvironmentConfig();
    const [grpoContainerName, setGrpoContainerName] = useState(defaultGrpoContainerName);
    const [grpoModels, setGrpoModels] = useState<AgentModelOption[]>([]);
    const [grpoTrainFiles, setGrpoTrainFiles] = useState<AgentFileOption[]>([]);
    const [grpoValFiles, setGrpoValFiles] = useState<AgentFileOption[]>([]);
    const [isLoadingGrpoResources, setIsLoadingGrpoResources] = useState(false);
    const grpoResourceRequestKeyRef = useRef('');
    const [resourceDatasets, setResourceDatasets] = useState<AgentDatasetOption[] | null>(null);
    const [resourceModels, setResourceModels] = useState<AgentModelOption[] | null>(null);
    const [loadedResourceKey, setLoadedResourceKey] = useState('');
    const [hasSubmitted, setHasSubmitted] = useState(() => readSubmittedState(persistenceKey));
    const [isRefreshingDatasets, setIsRefreshingDatasets] = useState(false);
    const [isRefreshingModels, setIsRefreshingModels] = useState(false);
    const queryGrpoResourcesMutation = trpc.queryGrpoResources.useMutation();
    const usesProtocolResources = Boolean(prompt.resourceContainer && !isGrpoTrainingPrompt);
    const effectiveDatasets = resourceDatasets ?? (usesProtocolResources ? [] : datasets);
    const effectivePromptModels = resourceModels ?? (usesProtocolResources ? [] : models);
    const excludedInputFolders = new Set((prompt.excludedInputFolders || []).map(normalizeDatasetPath));
    const inputFolderDatasets = prompt.fields.includes('input_folder')
        ? isDataPreprocessingPrompt
          ? effectiveDatasets.filter((dataset) => isPreprocessInputDataset(dataset, excludedInputFolders))
          : effectiveDatasets.filter((dataset) => (dataset.type || '').toLowerCase() !== 'raw')
        : effectiveDatasets;
    const selectableDatasets = datasetNameOptions.length > 0
        ? datasetNameOptions.map((name) => ({
            name,
            type: 'dpo',
            path: prompt.knownParams?.dataset_dir,
        }))
        : requiresDpoDataset
          ? inputFolderDatasets.filter((dataset) => (dataset.type || '').toLowerCase() === 'dpo')
          : inputFolderDatasets;
    const effectiveModels = isGrpoTrainingPrompt ? grpoModels : effectivePromptModels;

    const filledFields = editableFields.filter((field) => (values[field] || '').trim().length > 0);
    const canSubmitFields = shouldSubmitTrainingParamsTogether
        ? Boolean(selectedModel && selectedDataset && (values.schedule_time || '').trim())
        : shouldSubmitGrpoParamsTogether
          ? Boolean(selectedModel && selectedTrainFile && selectedValFile)
        : filledFields.length > 0;

    const loadGrpoResources = async () => {
        if (!isGrpoTrainingPrompt) return;
        const requestKey = (resourceGroupId || '') + ':' + defaultGrpoContainerName;
        grpoResourceRequestKeyRef.current = requestKey;
        setGrpoContainerName(defaultGrpoContainerName);
        setGrpoModels([]);
        setGrpoTrainFiles([]);
        setGrpoValFiles([]);
        setIsLoadingGrpoResources(true);
        try {
            const response = await queryGrpoResourcesMutation.mutateAsync({
                groupId: resourceGroupId,
            });
            if (grpoResourceRequestKeyRef.current !== requestKey) return;
            if (response.success && response.data) {
                setGrpoContainerName(response.data.containerName || defaultGrpoContainerName);
                setGrpoModels(response.data.models || []);
                setGrpoTrainFiles(response.data.trainFiles || []);
                setGrpoValFiles(response.data.valFiles || []);
            }
        } catch (error) {
            if (grpoResourceRequestKeyRef.current === requestKey) {
                console.warn('Failed to load GRPO resources:', error);
            }
        } finally {
            if (grpoResourceRequestKeyRef.current === requestKey) {
                setIsLoadingGrpoResources(false);
            }
        }
    };

    useEffect(() => {
        if (isGrpoTrainingPrompt) {
            loadGrpoResources();
        }
    }, [isGrpoTrainingPrompt, resourceGroupId, defaultGrpoContainerName]);

    useEffect(() => {
        setHasSubmitted(readSubmittedState(persistenceKey));
    }, [persistenceKey]);

    useEffect(() => {
        const container = prompt.resourceContainer?.trim();
        if (!container || isGrpoTrainingPrompt || hasSubmitted) {
            return;
        }

        const needsDatasets =
            hasDatasetPicker &&
            !hasDatasetNameFileOptions &&
            Boolean(onRefreshDatasets);
        const needsModels =
            (hasModelPicker || hasEvaluationModelPicker || hasCheckpointPicker) &&
            Boolean(onRefreshModels);
        if (!needsDatasets && !needsModels) {
            return;
        }

        const key = `${container}:${needsDatasets ? 'datasets' : ''}:${needsModels ? 'models' : ''}`;
        if (loadedResourceKey === key) {
            return;
        }

        setLoadedResourceKey(key);
        let cancelled = false;
        const loadResources = async () => {
            try {
                const [nextDatasets, nextModels] = await Promise.all([
                    needsDatasets ? onRefreshDatasets?.(container) : Promise.resolve(undefined),
                    needsModels ? onRefreshModels?.(container) : Promise.resolve(undefined),
                ]);
                if (cancelled) {
                    return;
                }
                if (nextDatasets) {
                    setResourceDatasets(nextDatasets);
                }
                if (nextModels) {
                    setResourceModels(nextModels);
                }
            } catch (error) {
                console.warn('Failed to load agent resources from protocol container:', error);
            }
        };

        loadResources();
        return () => {
            cancelled = true;
        };
    }, [
        hasCheckpointPicker,
        hasDatasetPicker,
        hasEvaluationModelPicker,
        hasModelPicker,
        hasSubmitted,
        isGrpoTrainingPrompt,
        hasDatasetNameFileOptions,
        loadedResourceKey,
        onRefreshDatasets,
        onRefreshModels,
        prompt.resourceContainer,
    ]);

    const submitReply = (text: string) => {
        const trimmedText = text.trim();
        if (!trimmedText || hasSubmitted) {
            return;
        }

        writeSubmittedState(persistenceKey);
        setHasSubmitted(true);
        onReply?.(trimmedText);
    };

    const knownPreprocessInputFolder =
        isDataPreprocessingPrompt ? (prompt.knownParams?.input_folder || '').trim() : '';

    const submitDataTypeStrategy = (nextDataType: string, nextStrategy: string) => {
        if (!nextDataType) {
            return;
        }
        if (hasDataTypeStrategyPicker && !nextStrategy) {
            return;
        }
        const parts: string[] = [];
        if (knownPreprocessInputFolder) {
            parts.push(`input_folder=${knownPreprocessInputFolder}`);
        }
        parts.push(`data_type=${nextDataType}`);
        if (hasDataTypeStrategyPicker) {
            parts.push(`strategy=${nextStrategy}`);
        }
        submitReply(parts.join('，'));
    };

    const chooseDataType = (value: string) => {
        setDataType(value);
        submitDataTypeStrategy(value, strategy);
    };

    const chooseStrategy = (value: string) => {
        setStrategy(value);
        submitDataTypeStrategy(dataType, value);
    };

    const joinPathAndName = joinPathAndNameValue;

    const datasetPath = datasetOptionPath;

    const shouldSendDatasetName = (dataset: AgentDatasetOption) => {
        if (!prompt.fields.includes('dataset_name')) {
            return false;
        }
        return !(requiresDpoDataset && /^20\d{6}$/.test(dataset.name));
    };

    const modelPath = (model: AgentModelOption) => {
        if (!model.path) {
            return model.name;
        }
        return joinPathAndName(model.path, model.name);
    };

    const isMergedModel = (model: AgentModelOption, normalizedPath: string, baseName: string) => {
        const modelType = (model.type || '').toLowerCase();
        const isDpoExportModel =
            ['dpo', 'daily_trained'].includes(modelType) &&
            /(?:^|\/)dpo_train\/internal\/export(?:\/|$)/.test(normalizedPath);
        return model.merged === true || /(?:^|[_-])merged$/i.test(baseName) || isDpoExportModel;
    };

    const isAllowedEnhancedTrainingModel = (model: AgentModelOption) => {
        const fullPath = modelPath(model);
        const normalizedPath = fullPath.replace(/\\/g, '/').replace(/\/+$/, '');
        const baseName = normalizedPath.split('/').pop() || model.name;
        const modelType = (model.type || '').toLowerCase();
        return (
            modelType === 'base_train' ||
            /(?:^|\/)base_train(?:\/|$)/.test(normalizedPath) ||
            isMergedModel(model, normalizedPath, baseName)
        );
    };

    const isAllowedEvaluationModel = (model: AgentModelOption) => {
        const fullPath = modelPath(model);
        const normalizedPath = fullPath.replace(/\\/g, '/').replace(/\/+$/, '');
        const baseName = normalizedPath.split('/').pop() || model.name;
        const modelType = (model.type || '').toLowerCase();
        const isBaseModel =
            modelType === 'base_train' ||
            /(?:^|\/)base_train(?:\/|$)/.test(normalizedPath) ||
            /(?:^|\/)base(?:\/|$)/.test(normalizedPath);
        const isSftOrDpoModel =
            ['sft', 'dpo', 'batch_trained', 'daily_trained'].includes(modelType) ||
            /(?:^|\/)(?:batch_train|daily_train|dpo_train)(?:\/|$)/.test(normalizedPath);
        return isBaseModel || (isSftOrDpoModel && isMergedModel(model, normalizedPath, baseName));
    };

    const isCheckpointSourceModel = (model: AgentModelOption) => {
        const checkpoints = model.checkpoints || [];
        if (checkpoints.length === 0) {
            return false;
        }

        const fullPath = modelPath(model);
        const normalizedPath = fullPath.replace(/\\/g, '/').replace(/\/+$/, '');
        const baseName = normalizedPath.split('/').pop() || model.name;
        const modelType = (model.type || '').toLowerCase();
        const isSftOrDpoModel =
            ['sft', 'dpo', 'batch_trained', 'daily_trained'].includes(modelType) ||
            /(?:^|\/)(?:batch_train|daily_train|dpo_train)(?:\/|$)/.test(normalizedPath);

        return isSftOrDpoModel && !isMergedModel(model, normalizedPath, baseName);
    };

    const selectableModels = requiresDpoDataset && !isGrpoTrainingPrompt
        ? effectiveModels.filter(isAllowedEnhancedTrainingModel)
        : effectiveModels;
    const selectableEvaluationModels = hasEvaluationModelPicker
        ? effectiveModels.filter(isAllowedEvaluationModel)
        : effectiveModels;
    const selectableCheckpointModels = hasCheckpointPicker
        ? effectiveModels.filter(isCheckpointSourceModel)
        : effectiveModels;

    const datasetParts = (dataset: AgentDatasetOption) => {
        const parts: string[] = [];
        if (prompt.fields.includes('dataset_dir')) {
            parts.push(`dataset_dir=${datasetPath(dataset)}`);
        }
        if (shouldSendDatasetName(dataset)) {
            parts.push(`dataset_name=${dataset.name}`);
        }
        if (prompt.fields.includes('input_folder')) {
            parts.push(`input_folder=${datasetPath(dataset)}`);
        }
        return parts;
    };

    const trainingParamParts = (
        model: AgentModelOption | null,
        dataset: AgentDatasetOption | null,
    ) => {
        if (!model || !dataset) {
            return null;
        }

        const parts = [`model_path=${modelPath(model)}`, ...datasetParts(dataset)];
        if (prompt.fields.includes('schedule_time')) {
            const scheduleTime = (values.schedule_time || '').trim();
            if (!scheduleTime) {
                return null;
            }
            parts.push(`schedule_time=${scheduleTime}`);
        }
        return parts;
    };

    const grpoParamParts = (
        model: AgentModelOption | null,
        trainFile: AgentFileOption | null,
        valFile: AgentFileOption | null,
    ) => {
        if (!model || !trainFile || !valFile) {
            return null;
        }
        return [
            `model_path=${modelPath(model)}`,
            `train_files=${trainFile.path}`,
            `val_files=${valFile.path}`,
        ];
    };

    const submitGrpoParams = (
        model: AgentModelOption | null,
        trainFile: AgentFileOption | null,
        valFile: AgentFileOption | null,
    ) => {
        const parts = grpoParamParts(model, trainFile, valFile);
        if (parts) {
            submitReply(parts.join('，'));
        }
    };

    const submitModelAndDataset = (
        model: AgentModelOption | null,
        dataset: AgentDatasetOption | null,
    ) => {
        const parts = trainingParamParts(model, dataset);
        if (!parts) {
            return;
        }

        submitReply(parts.join('，'));
    };

    const chooseDataset = (dataset: AgentDatasetOption) => {
        if (shouldSubmitModelAndDatasetTogether) {
            setSelectedDataset(dataset);
            submitModelAndDataset(selectedModel, dataset);
            return;
        }

        const parts = datasetParts(dataset);
        if (parts.length > 0) {
            submitReply(parts.join('，'));
        }
    };

    const chooseModel = (model: AgentModelOption) => {
        if (shouldSubmitGrpoParamsTogether) {
            setSelectedModel(model);
            submitGrpoParams(model, selectedTrainFile, selectedValFile);
            return;
        }

        if (shouldSubmitModelAndDatasetTogether) {
            setSelectedModel(model);
            submitModelAndDataset(model, selectedDataset);
            return;
        }

        submitReply(`model_path=${modelPath(model)}`);
    };

    const chooseGrpoFile = (field: 'train_files' | 'val_files', file: AgentFileOption) => {
        if (field === 'train_files') {
            setSelectedTrainFile(file);
            submitGrpoParams(selectedModel, file, selectedValFile);
            return;
        }

        setSelectedValFile(file);
        submitGrpoParams(selectedModel, selectedTrainFile, file);
    };

    const chooseEvaluationModel = (model: AgentModelOption) => {
        if (!hasDualModelPicker) {
            const field = prompt.fields.includes('model_sec') ? 'model_sec' : 'model_fir';
            submitReply(`${field}=${modelPath(model)}`);
            return;
        }

        if (!selectedModelFir) {
            setSelectedModelFir(model);
            return;
        }

        if (!selectedModelSec) {
            setSelectedModelSec(model);
            submitReply(`model_fir=${modelPath(selectedModelFir)}，model_sec=${modelPath(model)}`);
            return;
        }

        setSelectedModelFir(model);
        setSelectedModelSec(null);
    };

    const chooseCheckpoint = (model: AgentModelOption, checkpoint: string) => {
        submitReply(`CKPT_PATH=${modelPath(model)}/${checkpoint}`);
    };

    const refreshDatasets = async () => {
        if (!onRefreshDatasets) return;
        setIsRefreshingDatasets(true);
        try {
            const nextDatasets = await onRefreshDatasets(prompt.resourceContainer);
            setResourceDatasets(nextDatasets);
        } finally {
            setIsRefreshingDatasets(false);
        }
    };

    const refreshModels = async () => {
        if (!onRefreshModels) return;
        setIsRefreshingModels(true);
        try {
            const nextModels = await onRefreshModels(prompt.resourceContainer);
            setResourceModels(nextModels);
        } finally {
            setIsRefreshingModels(false);
        }
    };

    const submitFields = () => {
        if (shouldSubmitTrainingParamsTogether) {
            submitModelAndDataset(selectedModel, selectedDataset);
            return;
        }

        if (shouldSubmitGrpoParamsTogether) {
            submitGrpoParams(selectedModel, selectedTrainFile, selectedValFile);
            return;
        }

        const text = filledFields
            .map((field) => `${field}=${(values[field] || '').trim()}`)
            .join('，');
        if (text) {
            submitReply(text);
        }
    };

    return (
        <div className={`mb-2 w-full rounded-[24px] border border-slate-200/80 bg-white/88 p-4 text-slate-800 shadow-[0_18px_44px_-34px_rgba(15,23,42,0.34)] backdrop-blur-sm dark:border-white/10 dark:bg-slate-900/72 dark:text-slate-100 ${hasSubmitted ? 'opacity-75' : ''}`}>
            <div className="flex items-start gap-3.5">
                <div className="mt-0.5 flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl border border-sky-200/80 bg-sky-50 text-sky-700 shadow-[0_12px_22px_-18px_rgba(2,132,199,0.55)] dark:border-sky-400/20 dark:bg-sky-500/12 dark:text-sky-200">
                    {prompt.kind === 'choice' ? (
                        <CheckCircle2 className="h-5 w-5" />
                    ) : prompt.kind === 'param' ? (
                        <ClipboardList className="h-5 w-5" />
                    ) : (
                        <HelpCircle className="h-5 w-5" />
                    )}
                </div>
                <div className="min-w-0 flex-1">
                    <div className="text-[15px] font-semibold leading-6 text-slate-900 dark:text-slate-100">{prompt.title}</div>
                    <p className="mt-1.5 whitespace-pre-wrap text-[13px] leading-6 text-slate-600 dark:text-slate-300">
                        {prompt.body}
                    </p>
                    {hasSubmitted && (
                        <div className="mt-3 inline-flex rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1 text-xs font-medium text-emerald-700 dark:border-emerald-400/20 dark:bg-emerald-500/10 dark:text-emerald-200">
                            已发送，等待后台继续处理
                        </div>
                    )}

                    {prompt.quickReplies.length > 0 && (
                        <div className="mt-4 flex flex-wrap gap-2.5">
                            {prompt.quickReplies.slice(0, 6).map((reply) => (
                                <Button
                                    key={reply}
                                    size="sm"
                                    variant="outline"
                                    className="h-9 rounded-full border-slate-200/90 bg-white px-4 text-[13px] font-semibold text-slate-700 shadow-[0_10px_22px_-18px_rgba(15,23,42,0.35)] transition-all hover:border-sky-200 hover:bg-sky-50/70 hover:text-sky-700 dark:border-white/10 dark:bg-slate-950/50 dark:text-slate-200 dark:hover:border-sky-500/40 dark:hover:bg-sky-500/10 dark:hover:text-sky-200"
                                    disabled={hasSubmitted}
                                    onClick={(event) => {
                                        event.stopPropagation();
                                        submitReply(reply);
                                    }}
                                >
                                    {reply}
                                </Button>
                            ))}
                        </div>
                    )}

                    {hasDataTypePicker && (
                        <div className="mt-4 space-y-4 rounded-[20px] border border-slate-200/80 bg-slate-50/62 p-3.5 dark:border-white/10 dark:bg-slate-950/34">
                            <div>
                                <div className="mb-2 text-xs font-semibold text-slate-500 dark:text-slate-400">
                                    数据类型
                                </div>
                                <div className="grid gap-2 sm:grid-cols-2">
                                    {DATA_TYPE_OPTIONS.map((option) => {
                                        const selected = dataType === option.value;
                                        return (
                                            <button
                                                key={option.value}
                                                type="button"
                                                disabled={hasSubmitted}
                                                className={`rounded-2xl border px-4 py-3 text-left transition-all ${
                                                    selected
                                                        ? 'border-sky-300 bg-white text-sky-800 shadow-[0_14px_28px_-22px_rgba(2,132,199,0.55)] ring-1 ring-sky-100 dark:border-sky-400/40 dark:bg-sky-500/12 dark:text-sky-100 dark:ring-sky-400/10'
                                                        : 'border-slate-200/90 bg-white/78 text-slate-700 hover:border-sky-200 hover:bg-white dark:border-white/10 dark:bg-slate-950/38 dark:text-slate-200 dark:hover:border-sky-500/30 dark:hover:bg-sky-500/10'
                                                }`}
                                                onClick={(event) => {
                                                    event.stopPropagation();
                                                    chooseDataType(option.value);
                                                }}
                                            >
                                                <span className="block text-[13px] font-semibold">{option.label}</span>
                                            </button>
                                        );
                                    })}
                                </div>
                            </div>

                            {hasDataTypeStrategyPicker && (
                                <div>
                                    <div className="mb-2 text-xs font-semibold text-slate-500 dark:text-slate-400">
                                        数据策略
                                    </div>
                                    <div className="grid gap-2 sm:grid-cols-3">
                                        {STRATEGY_OPTIONS.map((option) => {
                                            const selected = strategy === option.value;
                                            return (
                                                <button
                                                    key={option.value}
                                                    type="button"
                                                    disabled={hasSubmitted}
                                                    className={`rounded-2xl border px-4 py-3 text-left transition-all ${
                                                        selected
                                                            ? 'border-sky-300 bg-white text-sky-800 shadow-[0_14px_28px_-22px_rgba(2,132,199,0.55)] ring-1 ring-sky-100 dark:border-sky-400/40 dark:bg-sky-500/12 dark:text-sky-100 dark:ring-sky-400/10'
                                                            : 'border-slate-200/90 bg-white/78 text-slate-700 hover:border-sky-200 hover:bg-white dark:border-white/10 dark:bg-slate-950/38 dark:text-slate-200 dark:hover:border-sky-500/30 dark:hover:bg-sky-500/10'
                                                    }`}
                                                    onClick={(event) => {
                                                        event.stopPropagation();
                                                        chooseStrategy(option.value);
                                                    }}
                                                >
                                                    <span className="block text-[13px] font-semibold">{option.label}</span>
                                                </button>
                                            );
                                        })}
                                    </div>
                                </div>
                            )}
                        </div>
                    )}

                    {hasCheckpointPicker && (
                        <div className="mt-4 rounded-[20px] border border-slate-200/80 bg-slate-50/62 p-3.5 dark:border-white/10 dark:bg-slate-950/34">
                            <div className="mb-3 flex items-center justify-between gap-3">
                                <div className="text-xs font-semibold text-slate-500 dark:text-slate-400">
                                    选择 Checkpoint
                                </div>
                                {onRefreshModels && (
                                    <Button
                                        size="sm"
                                        variant="outline"
                                        className="h-8 rounded-full border-slate-200 bg-white px-3 text-xs"
                                        disabled={isRefreshingModels || hasSubmitted}
                                        onClick={(event) => {
                                            event.stopPropagation();
                                            refreshModels();
                                        }}
                                    >
                                        {isRefreshingModels ? '刷新中' : '刷新'}
                                    </Button>
                                )}
                            </div>
                            {selectableCheckpointModels.length > 0 ? (
                                <div className="grid max-h-64 gap-2 overflow-auto pr-1 sm:grid-cols-2">
                                    {selectableCheckpointModels.slice(0, 10).map((model) => {
                                        const path = modelPath(model);
                                        return (
                                            <div
                                                key={`${model.type || 'model'}:${model.name}:${model.path || ''}`}
                                                className="rounded-2xl border border-slate-200/90 bg-white/78 px-4 py-3 text-left text-slate-700 transition-all dark:border-white/10 dark:bg-slate-950/38 dark:text-slate-200"
                                            >
                                                <span className="block truncate text-sm font-semibold">{model.name}</span>
                                                <span className="mt-1 block truncate text-xs text-slate-500 dark:text-slate-400">
                                                    {path}
                                                </span>
                                                <div className="mt-3 flex flex-wrap gap-2">
                                                    {(model.checkpoints || []).map((checkpoint) => (
                                                        <Button
                                                            key={checkpoint}
                                                            size="sm"
                                                            variant="outline"
                                                            className="h-7 rounded-full px-3 text-xs"
                                                            disabled={hasSubmitted}
                                                            onClick={(event) => {
                                                                event.stopPropagation();
                                                                chooseCheckpoint(model, checkpoint);
                                                            }}
                                                        >
                                                            {checkpoint}
                                                        </Button>
                                                    ))}
                                                </div>
                                            </div>
                                        );
                                    })}
                                </div>
                            ) : (
                                <div className="rounded-2xl border border-dashed border-slate-200 bg-white/70 px-4 py-5 text-sm text-slate-500 dark:border-white/10 dark:bg-slate-950/30 dark:text-slate-400">
                                    暂无 Checkpoint 列表，请先刷新。
                                </div>
                            )}
                        </div>
                    )}

                    {hasEvaluationModelPicker && (
                        <div className="mt-4 rounded-[20px] border border-slate-200/80 bg-slate-50/62 p-3.5 dark:border-white/10 dark:bg-slate-950/34">
                            <div className="mb-3 flex items-center justify-between gap-3">
                                <div className="text-xs font-semibold text-slate-500 dark:text-slate-400">
                                    选择评估模型
                                </div>
                                {onRefreshModels && (
                                    <Button
                                        size="sm"
                                        variant="outline"
                                        className="h-8 rounded-full border-slate-200 bg-white px-3 text-xs"
                                        disabled={isRefreshingModels || hasSubmitted}
                                        onClick={(event) => {
                                            event.stopPropagation();
                                            refreshModels();
                                        }}
                                    >
                                        {isRefreshingModels ? '刷新中' : '刷新'}
                                    </Button>
                                )}
                            </div>
                            {selectableEvaluationModels.length > 0 ? (
                                <div className="grid max-h-64 gap-2 overflow-auto pr-1 sm:grid-cols-2">
                                    {selectableEvaluationModels.slice(0, 10).map((model) => {
                                        const path = modelPath(model);
                                        const isFirst = selectedModelFir ? modelPath(selectedModelFir) === path : false;
                                        const isSecond = selectedModelSec ? modelPath(selectedModelSec) === path : false;
                                        return (
                                            <button
                                                key={`${model.type || 'model'}:${model.name}:${model.path || ''}`}
                                                type="button"
                                                disabled={hasSubmitted}
                                                className={`rounded-2xl border px-4 py-3 text-left text-slate-700 transition-all dark:text-slate-200 ${
                                                    isFirst || isSecond
                                                        ? 'border-sky-300 bg-white shadow-[0_14px_28px_-22px_rgba(2,132,199,0.55)] ring-1 ring-sky-100 dark:border-sky-400/40 dark:bg-sky-500/12 dark:ring-sky-400/10'
                                                        : 'border-slate-200/90 bg-white/78 hover:border-sky-200 hover:bg-white dark:border-white/10 dark:bg-slate-950/38 dark:hover:border-sky-500/30 dark:hover:bg-sky-500/10'
                                                }`}
                                                onClick={(event) => {
                                                    event.stopPropagation();
                                                    chooseEvaluationModel(model);
                                                }}
                                            >
                                                <span className="block truncate text-sm font-semibold">{model.name}</span>
                                                <span className="mt-1 block truncate text-xs text-slate-500 dark:text-slate-400">
                                                    {path}
                                                </span>
                                            </button>
                                        );
                                    })}
                                </div>
                            ) : (
                                <div className="rounded-2xl border border-dashed border-slate-200 bg-white/70 px-4 py-5 text-sm text-slate-500 dark:border-white/10 dark:bg-slate-950/30 dark:text-slate-400">
                                    暂无模型列表，请先刷新。
                                </div>
                            )}
                        </div>
                    )}

                    {hasModelPicker && (
                        <div className="mt-4 rounded-[20px] border border-slate-200/80 bg-slate-50/62 p-3.5 dark:border-white/10 dark:bg-slate-950/34">
                            <div className="mb-3 flex items-center justify-between gap-3">
                                <div>
                                    <div className="text-xs font-semibold text-slate-500 dark:text-slate-400">
                                        选择模型
                                    </div>
                                    {isGrpoTrainingPrompt && (
                                        <div className="mt-1 text-[11px] text-slate-400 dark:text-slate-500">
                                            GRPO Docker：{grpoContainerName}
                                        </div>
                                    )}
                                </div>
                                {(onRefreshModels || isGrpoTrainingPrompt) && (
                                    <Button
                                        size="sm"
                                        variant="outline"
                                        className="h-8 rounded-full border-slate-200 bg-white px-3 text-xs"
                                        disabled={isRefreshingModels || isLoadingGrpoResources || hasSubmitted}
                                        onClick={(event) => {
                                            event.stopPropagation();
                                            if (isGrpoTrainingPrompt) {
                                                loadGrpoResources();
                                            } else {
                                                refreshModels();
                                            }
                                        }}
                                    >
                                        {isRefreshingModels || isLoadingGrpoResources ? '刷新中' : '刷新'}
                                    </Button>
                                )}
                            </div>
                            {selectableModels.length > 0 ? (
                                <div className="grid max-h-64 gap-2 overflow-auto pr-1 sm:grid-cols-2">
                                    {selectableModels.slice(0, 8).map((model) => {
                                        const selected = selectedModel
                                            ? modelPath(selectedModel) === modelPath(model)
                                            : false;
                                        return (
                                            <button
                                                key={`${model.type || 'model'}:${model.name}:${model.path || ''}`}
                                                type="button"
                                                disabled={hasSubmitted}
                                                className={`rounded-2xl border px-4 py-3 text-left text-slate-700 transition-all dark:text-slate-200 ${
                                                    selected
                                                        ? 'border-sky-300 bg-white shadow-[0_14px_28px_-22px_rgba(2,132,199,0.55)] ring-1 ring-sky-100 dark:border-sky-400/40 dark:bg-sky-500/12 dark:ring-sky-400/10'
                                                        : 'border-slate-200/90 bg-white/78 hover:border-sky-200 hover:bg-white dark:border-white/10 dark:bg-slate-950/38 dark:hover:border-sky-500/30 dark:hover:bg-sky-500/10'
                                                }`}
                                                onClick={(event) => {
                                                    event.stopPropagation();
                                                    chooseModel(model);
                                                }}
                                            >
                                                <span className="block truncate text-sm font-semibold">{model.name}</span>
                                                <span className="mt-1 block truncate text-xs text-slate-500 dark:text-slate-400">
                                                    {model.path ? modelPath(model) : model.type || '模型路径待确认'}
                                                </span>
                                                {(model.size || model.type) && (
                                                    <span className="mt-2 inline-flex rounded-full bg-slate-100 px-2 py-0.5 text-[11px] text-slate-500 dark:bg-slate-800 dark:text-slate-400">
                                                        {model.type || model.size}
                                                    </span>
                                                )}
                                            </button>
                                        );
                                    })}
                                </div>
                            ) : (
                                <div className="rounded-2xl border border-dashed border-slate-200 bg-white/70 px-4 py-5 text-sm text-slate-500 dark:border-white/10 dark:bg-slate-950/30 dark:text-slate-400">
                                    {isGrpoTrainingPrompt
                                        ? `暂无 GRPO 模型列表，请确认 ${grpoContainerName} 容器已启动后刷新。`
                                        : '暂无模型列表，请先刷新。'}
                                </div>
                            )}
                        </div>
                    )}

                    {hasGrpoFilePicker && (
                        <div className="mt-4 rounded-[20px] border border-slate-200/80 bg-slate-50/62 p-3.5 dark:border-white/10 dark:bg-slate-950/34">
                            <div className="mb-3 flex items-center justify-between gap-3">
                                <div>
                                    <div className="text-xs font-semibold text-slate-500 dark:text-slate-400">
                                        选择 GRPO 数据文件
                                    </div>
                                    <div className="mt-1 text-[11px] text-slate-400 dark:text-slate-500">
                                        来源：{grpoContainerName}:/home/workspace/verl/examples/data_preprocess/data
                                    </div>
                                </div>
                                <Button
                                    size="sm"
                                    variant="outline"
                                    className="h-8 rounded-full border-slate-200 bg-white px-3 text-xs"
                                    disabled={isLoadingGrpoResources || hasSubmitted}
                                    onClick={(event) => {
                                        event.stopPropagation();
                                        loadGrpoResources();
                                    }}
                                >
                                    {isLoadingGrpoResources ? '刷新中' : '刷新'}
                                </Button>
                            </div>

                            <div className="grid gap-3 lg:grid-cols-2">
                                {prompt.fields.includes('train_files') && (
                                    <div>
                                        <div className="mb-2 text-xs font-semibold text-slate-500 dark:text-slate-400">
                                            训练文件
                                        </div>
                                        {grpoTrainFiles.length > 0 ? (
                                            <div className="max-h-56 space-y-2 overflow-auto pr-1">
                                                {grpoTrainFiles.slice(0, 30).map((file) => {
                                                    const selected = selectedTrainFile?.path === file.path;
                                                    return (
                                                        <button
                                                            key={`train:${file.path}`}
                                                            type="button"
                                                            disabled={hasSubmitted}
                                                            className={`w-full rounded-2xl border px-4 py-3 text-left text-slate-700 transition-all dark:text-slate-200 ${
                                                                selected
                                                                    ? 'border-sky-300 bg-white shadow-[0_14px_28px_-22px_rgba(2,132,199,0.55)] ring-1 ring-sky-100 dark:border-sky-400/40 dark:bg-sky-500/12 dark:ring-sky-400/10'
                                                                    : 'border-slate-200/90 bg-white/78 hover:border-sky-200 hover:bg-white dark:border-white/10 dark:bg-slate-950/38 dark:hover:border-sky-500/30 dark:hover:bg-sky-500/10'
                                                            }`}
                                                            onClick={(event) => {
                                                                event.stopPropagation();
                                                                chooseGrpoFile('train_files', file);
                                                            }}
                                                        >
                                                            <span className="block truncate text-sm font-semibold">{file.name}</span>
                                                            <span className="mt-1 block truncate text-xs text-slate-500 dark:text-slate-400">
                                                                {file.path}
                                                            </span>
                                                        </button>
                                                    );
                                                })}
                                            </div>
                                        ) : (
                                            <div className="rounded-2xl border border-dashed border-slate-200 bg-white/70 px-4 py-5 text-sm text-slate-500 dark:border-white/10 dark:bg-slate-950/30 dark:text-slate-400">
                                                暂无 parquet 训练文件，请刷新。
                                            </div>
                                        )}
                                    </div>
                                )}

                                {prompt.fields.includes('val_files') && (
                                    <div>
                                        <div className="mb-2 text-xs font-semibold text-slate-500 dark:text-slate-400">
                                            验证文件
                                        </div>
                                        {grpoValFiles.length > 0 ? (
                                            <div className="max-h-56 space-y-2 overflow-auto pr-1">
                                                {grpoValFiles.slice(0, 30).map((file) => {
                                                    const selected = selectedValFile?.path === file.path;
                                                    return (
                                                        <button
                                                            key={`val:${file.path}`}
                                                            type="button"
                                                            disabled={hasSubmitted}
                                                            className={`w-full rounded-2xl border px-4 py-3 text-left text-slate-700 transition-all dark:text-slate-200 ${
                                                                selected
                                                                    ? 'border-sky-300 bg-white shadow-[0_14px_28px_-22px_rgba(2,132,199,0.55)] ring-1 ring-sky-100 dark:border-sky-400/40 dark:bg-sky-500/12 dark:ring-sky-400/10'
                                                                    : 'border-slate-200/90 bg-white/78 hover:border-sky-200 hover:bg-white dark:border-white/10 dark:bg-slate-950/38 dark:hover:border-sky-500/30 dark:hover:bg-sky-500/10'
                                                            }`}
                                                            onClick={(event) => {
                                                                event.stopPropagation();
                                                                chooseGrpoFile('val_files', file);
                                                            }}
                                                        >
                                                            <span className="block truncate text-sm font-semibold">{file.name}</span>
                                                            <span className="mt-1 block truncate text-xs text-slate-500 dark:text-slate-400">
                                                                {file.path}
                                                            </span>
                                                        </button>
                                                    );
                                                })}
                                            </div>
                                        ) : (
                                            <div className="rounded-2xl border border-dashed border-slate-200 bg-white/70 px-4 py-5 text-sm text-slate-500 dark:border-white/10 dark:bg-slate-950/30 dark:text-slate-400">
                                                暂无 parquet 验证文件，请刷新。
                                            </div>
                                        )}
                                    </div>
                                )}
                            </div>
                        </div>
                    )}

                    {hasDatasetPicker && (
                        <div className="mt-4 rounded-[20px] border border-slate-200/80 bg-slate-50/62 p-3.5 dark:border-white/10 dark:bg-slate-950/34">
                            <div className="mb-3 flex items-center justify-between gap-3">
                                <div>
                                    <div className="text-xs font-semibold text-slate-500 dark:text-slate-400">
                                        {datasetNameOptions.length > 0 ? '选择数据集文件' : '选择数据集'}
                                    </div>
                                    {datasetNameOptions.length > 0 && prompt.knownParams?.dataset_dir && (
                                        <div className="mt-1 text-[11px] text-slate-400 dark:text-slate-500">
                                            来源：{prompt.knownParams.dataset_dir}
                                        </div>
                                    )}
                                </div>
                                {onRefreshDatasets && !hasDatasetNameFileOptions && (
                                    <Button
                                        size="sm"
                                        variant="outline"
                                        className="h-8 rounded-full border-slate-200 bg-white px-3 text-xs"
                                        disabled={isRefreshingDatasets || hasSubmitted}
                                        onClick={(event) => {
                                            event.stopPropagation();
                                            refreshDatasets();
                                        }}
                                    >
                                        {isRefreshingDatasets ? '刷新中' : '刷新'}
                                    </Button>
                                )}
                            </div>
                            {selectableDatasets.length > 0 ? (
                                <div className="grid max-h-64 gap-2 overflow-auto pr-1 sm:grid-cols-2">
                                    {selectableDatasets.slice(0, 10).map((dataset) => {
                                        const selected = selectedDataset
                                            ? datasetPath(selectedDataset) === datasetPath(dataset) &&
                                              selectedDataset.name === dataset.name
                                            : false;
                                        return (
                                            <button
                                                key={`${dataset.type || 'dataset'}:${dataset.name}:${dataset.path || ''}`}
                                                type="button"
                                                disabled={hasSubmitted}
                                                className={`rounded-2xl border px-4 py-3 text-left text-slate-700 transition-all dark:text-slate-200 ${
                                                    selected
                                                        ? 'border-sky-300 bg-white shadow-[0_14px_28px_-22px_rgba(2,132,199,0.55)] ring-1 ring-sky-100 dark:border-sky-400/40 dark:bg-sky-500/12 dark:ring-sky-400/10'
                                                        : 'border-slate-200/90 bg-white/78 hover:border-sky-200 hover:bg-white dark:border-white/10 dark:bg-slate-950/38 dark:hover:border-sky-500/30 dark:hover:bg-sky-500/10'
                                                }`}
                                                onClick={(event) => {
                                                    event.stopPropagation();
                                                    chooseDataset(dataset);
                                                }}
                                            >
                                                <span className="block truncate text-sm font-semibold">{dataset.name}</span>
                                                <span className="mt-1 block truncate text-xs text-slate-500 dark:text-slate-400">
                                                    {datasetPath(dataset)}
                                                </span>
                                                {(dataset.type || dataset.size) && (
                                                    <span className="mt-2 inline-flex rounded-full bg-slate-100 px-2 py-0.5 text-[11px] text-slate-500 dark:bg-slate-800 dark:text-slate-400">
                                                        {dataset.type || dataset.size}
                                                    </span>
                                                )}
                                            </button>
                                        );
                                    })}
                                </div>
                            ) : (
                                <div className="rounded-2xl border border-dashed border-slate-200 bg-white/70 px-4 py-5 text-sm text-slate-500 dark:border-white/10 dark:bg-slate-950/30 dark:text-slate-400">
                                    {requiresDpoDataset
                                        ? '暂无数据集列表，请先刷新。'
                                        : '暂无数据集列表，请先刷新。'}
                                </div>
                            )}
                        </div>
                    )}

                    {editableFields.length > 0 && (
                        <div className="mt-4 space-y-3 rounded-[20px] border border-slate-200/80 bg-slate-50/62 p-3.5 dark:border-white/10 dark:bg-slate-950/34">
                            {editableFields.map((field) => (
                                <label key={field} className="block">
                                    <span className="mb-1.5 block text-xs font-semibold text-slate-500 dark:text-slate-400">
                                        {fieldLabel(field)}
                                    </span>
                                    <Input
                                        value={values[field] || ''}
                                        placeholder={fieldPlaceholder(field)}
                                        className="h-10 rounded-2xl border-slate-200 bg-white font-mono text-xs dark:border-white/10 dark:bg-slate-950"
                                        disabled={hasSubmitted}
                                        onClick={(event) => event.stopPropagation()}
                                        onChange={(event) =>
                                            setValues((prev) => ({
                                                ...prev,
                                                [field]: event.target.value,
                                            }))
                                        }
                                    />
                                </label>
                            ))}
                            <Button
                                size="sm"
                                className="h-9 gap-1.5 rounded-full px-4 text-[13px]"
                                disabled={!canSubmitFields || hasSubmitted}
                                onClick={(event) => {
                                    event.stopPropagation();
                                    submitFields();
                                }}
                            >
                                <Sparkles className="h-3.5 w-3.5" />
                                发送已填信息
                                <ArrowRight className="h-3.5 w-3.5" />
                            </Button>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
