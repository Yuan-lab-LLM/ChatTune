/**
 * Beginner Guide Wizard - Types
 */

import type { DatasetInfo, InputRequestData } from '@shared/types/trpc';

export interface DemoDataset {
  id: string;
  name: string;
  type: 'raw' | 'sft' | 'dpo' | 'pt';
  description: string;
  path: string;
  format: string;
  recordCount: number;
  recommended?: boolean;
}

export interface WizardState {
  currentStep: number;
  selectedDataset: DemoDataset | null;
  isExecuting: boolean;
  executionStatus: 'idle' | 'running' | 'success' | 'error';
  logs: string[];
  canProceed: boolean;
}

// 简化的运行数据类型
export interface WizardRunData {
  id: string;
  name: string;
  status: 'RUNNING' | 'PENDING' | 'COMPLETED' | 'FAILED';
  timestamp: number;
}

export interface QuickStartWizardProps {
  open: boolean;
  onClose: () => void;
  onTabChange: (tab: string) => void;
  onSendCommand: (command: string) => void;
  onOpenMetrics: () => void;
  currentTab: string;
  // 新增：向导查询相关状态
  queryState: 'idle' | 'querying' | 'completed';
  datasets: DatasetInfo[];
  selectedDataset: DatasetInfo | null;
  onSelectDataset: (dataset: DatasetInfo | null) => void;
  requiredDatasetId?: string;
  // 新增：运行实例输入请求
  inputRequests: InputRequestData[];
  // 新增：运行列表
  runs: WizardRunData[];
  // 新增：focusOnLatestRun 机制
  focusOnLatestRun: boolean;
  setFocusOnLatestRun: (focus: boolean) => void;
  // 新增：AsChat 发送完成回调（用于判定第二步完成）
  onAsChatSendComplete?: () => void;
  onStepChange?: (step: number, title: string) => void;
}

// 暴露给父组件的方法
export interface QuickStartWizardRef {
  handleStepComplete: (sentCommand: string) => void;
  restart: () => void;
}
