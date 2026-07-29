import { DemoDataset } from './types';

export const DEMO_DATASETS: DemoDataset[] = [
  {
    id: 'medical_records_raw',
    name: '示例病历数据',
    type: 'raw',
    description: '1000条病历记录，包含主诉、现病史、诊断等字段，适合医疗场景模型训练',
    path: 'medical_records_raw',
    format: 'JSON',
    recordCount: 1000,
    recommended: true,
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
  format: 'diagnosis',
  outputSuffix: '_sft',
};

export const TRAINING_CONFIG = {
  method: 'LoRA',
  description: '高效参数微调，适合在有限算力下进行模型训练',
  estimatedTime: '5-10分钟',
};
