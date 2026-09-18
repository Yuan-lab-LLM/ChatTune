import type { DemoDataset } from './types';
import type { DatasetInfo } from '@shared/types/trpc';

export const DEFAULT_PREFERRED_DATASET_ID = 'general-conversation-example';
export const LEGACY_MEDICAL_DATASET_ID = 'medical-example';

export const DEMO_DATASETS: DemoDataset[] = [
  {
    id: 'general_conversation_raw',
    name: '通用对话示例数据',
    type: 'raw',
    description: '通用指令与多轮对话数据，用于演示通用 LLM SFT 流程',
    path: 'general-conversation-example',
    format: 'JSON',
    recordCount: 500,
    recommended: true,
  },
  {
    id: 'medical_records_raw',
    name: '示例病历数据',
    type: 'raw',
    description: '1000条病历记录，包含主诉、现病史、诊断等字段，适合医疗场景模型训练',
    path: 'medical_records_raw',
    format: 'JSON',
    recordCount: 1000,
    recommended: false,
  },
  {
    id: 'dialogue_sample_raw',
    name: '示例对话数据',
    type: 'raw',
    description: '500条医患对话记录，适合对话模型微调',
    path: 'dialogue_sample_raw',
    format: 'JSON',
    recordCount: 500,
    recommended: false,
  },
];

export const PREPROCESS_CONFIG = {
  targetType: 'sft',
  outputSuffix: '_sft',
};

export const isRawDataset = (dataset: DatasetInfo): boolean =>
  String(dataset.type || '').trim().toLowerCase() === 'raw';

export const isMedicalDemoDataset = (dataset: DatasetInfo): boolean => {
  const identity = `${dataset.name} ${dataset.path || ''}`.toLowerCase();
  return identity.includes('medical') || identity.includes('病历');
};

export const datasetInputPath = (dataset: DatasetInfo): string => {
  const configuredPath = String(dataset.path || '').trim();
  if (configuredPath.startsWith('/')) {
    const normalizedPath = configuredPath.replace(/\/+$/, '');
    return normalizedPath.endsWith(`/${dataset.name}`)
      ? normalizedPath
      : `${normalizedPath}/${dataset.name}`;
  }
  return `/home/workspace/dataset/${configuredPath || dataset.name}`;
};

export const buildSftPreprocessCommand = (dataset: DatasetInfo): string => {
  const fields = [
    '执行数据预处理',
    `data_type=${PREPROCESS_CONFIG.targetType}`,
    `input_folder=${datasetInputPath(dataset)}`,
  ];
  if (isMedicalDemoDataset(dataset)) {
    fields.push('strategy=diagnosis');
  }
  return fields.join('，');
};

export const DEFAULT_TRAINING_COMMAND = '运行lora批量训练，TEM=qwen3';

export const TRAINING_CONFIG = {
  method: 'LoRA',
  description: '高效参数微调，适合在有限算力下进行模型训练',
  estimatedTime: '5-10分钟',
};
