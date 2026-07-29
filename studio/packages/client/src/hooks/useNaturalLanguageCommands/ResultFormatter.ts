/**
 * Result Formatter
 * 将命令执行结果格式化为Markdown格式的AI消息
 */

import { DatasetInfo, ModelInfo, MedicalTestFile, SystemOverviewData, GPUInfo } from '@shared/types';

export interface FormattedResult {
  content: string;
  type: 'success' | 'error' | 'info' | 'warning';
}

const getGeneralEvaluationTypeName = (testName: string): string => {
  const normalizedName = testName.toLowerCase();
  const categoryMap: Record<string, string> = {
    mmlu: '通用知识',
    'mmlu-pro': '通用知识',
    'c-eval': '中文专项',
    cmmlu: '中文专项',
    gpqa: '高阶推理',
    arc: '高阶推理',
    bbh: '复杂任务',
    gsm8k: '数学推理',
    math: '数学推理',
    humaneval: '代码生成',
    livecodebench: '代码生成',
    squad: '阅读理解',
    drop: '阅读理解',
    ifeval: '程序控制',
    truthfulqa: '模型安全',
  };

  return categoryMap[normalizedName] || '通用评测';
};

const getEvaluationTypeName = (test: MedicalTestFile): string => {
  if (test.category === 'general' || test.type === 'general') {
    return getGeneralEvaluationTypeName(test.filename || '');
  }

  const testTypeMap: Record<string, string> = {
    exam2021: '中国执业医师资格考试',
    exam2024: '临床医学综合能力(西医)',
    usmle: '美国执业医师考试',
    medbench: 'MedBench评测',
    other: '其他',
  };

  return testTypeMap[test.type] || test.type || '未知';
};

/**
 * 格式化数据集列表
 */
export function formatDatasetsResult(datasets: DatasetInfo[]): FormattedResult {
  if (datasets.length === 0) {
    return {
      content: '📊 **数据集查询结果**\n\n暂无数据集。\n\n💡 提示：使用"上传数据集"命令或点击左侧"数据管理"Tab上传数据。',
      type: 'info',
    };
  }

  const rows = datasets.map(ds => {
    const size = formatFileSize(ds.size);
    const type = ds.type || '未知';
    return `| ${ds.name} | ${type} | ${size} | ${ds.createdAt || '-'} |`;
  });

  const content = `📊 **数据集查询结果**\n\n共找到 **${datasets.length}** 个数据集：\n\n| 名称 | 类型 | 大小 | 创建时间 |\n|------|------|------|----------|\n${rows.join('\n')}\n\n💡 提示：使用"下载数据集 [名称]"命令下载指定数据集。`;

  return { content, type: 'success' };
}

/**
 * 格式化模型列表
 */
export function formatModelsResult(models: ModelInfo[]): FormattedResult {
  if (models.length === 0) {
    return {
      content: '🤖 **模型查询结果**\n\n暂无模型。',
      type: 'info',
    };
  }

  const rows = models.map(model => {
    const size = formatFileSize(model.size);
    const merged = model.merged ? '✓' : '✗';
    const type = model.type || '未知';
    return `| ${model.name} | ${type} | ${merged} | ${size} | ${model.createdAt || '-'} |`;
  });

  const content = `🤖 **模型查询结果**\n\n共找到 **${models.length}** 个模型：\n\n| 名称 | 类型 | 训练完成 | 大小 | 创建时间 |\n|------|------|--------|------|----------|\n${rows.join('\n')}`;

  return { content, type: 'success' };
}

/**
 * 格式化评测文件列表
 */
export function formatEvaluationResult(tests: MedicalTestFile[]): FormattedResult {
  if (tests.length === 0) {
    return {
      content: '🧪 **评测查询结果**\n\n暂无评测文件。\n\n💡 提示：使用"上传评测集"命令或点击左侧"评测管理"Tab上传评测文件。',
      type: 'info',
    };
  }

  const rows = tests.map(test => {
    const size = formatFileSize(test.size);
    return `| ${test.filename} | ${getEvaluationTypeName(test)} | ${size} | ${test.createdAt || '-'} |`;
  });

  const content = `🧪 **评测查询结果**\n\n共找到 **${tests.length}** 个评测文件：\n\n| 文件名 | 类型 | 大小 | 创建时间 |\n|--------|------|------|----------|\n${rows.join('\n')}\n\n💡 提示：使用"下载评测 [文件名]"命令下载指定评测文件。`;

  return { content, type: 'success' };
}

/**
 * 格式化系统概览
 */
export function formatSystemOverview(data: SystemOverviewData | null): FormattedResult {
  if (!data) {
    return {
      content: '⚠️ **系统概览**\n\n暂无系统信息。',
      type: 'warning',
    };
  }

  const content = `🖥️ **系统概览**\n\n| 指标 | 数值 |\n|------|------|\n| 在线用户 | ${data.onlineUsers || 0} |\n| 总内存 | ${formatFileSize(data.totalMemory)} |\n| 已用内存 | ${formatFileSize(data.usedMemory)} |\n| 内存使用率 | ${data.memoryUsage}% |\n| CPU 使用率 | ${data.cpuUsage}% |\n| 磁盘使用率 | ${data.diskUsage}% |\n| 运行时间 | ${data.uptime || '-'} |`;

  return { content, type: 'success' };
}

/**
 * 格式化GPU信息
 */
export function formatGPUInfo(gpuInfo: GPUInfo[] | null): FormattedResult {
  if (!gpuInfo || gpuInfo.length === 0) {
    return {
      content: '⚠️ **GPU 信息**\n\n暂无 GPU 信息。\n\n💡 提示：点击左侧"系统概览"Tab可查看更多系统信息。',
      type: 'warning',
    };
  }

  const gpuCards = gpuInfo.map((gpu, index) => {
    const memoryUsed = formatFileSize(gpu.memoryUsed);
    const memoryTotal = formatFileSize(gpu.memoryTotal);
    const utilization = gpu.utilization !== undefined ? `${gpu.utilization}%` : '-';
    const temp = gpu.temperature !== undefined ? `${gpu.temperature}°C` : '-';
    
    return `**GPU ${index}: ${gpu.name}**\n\n| 指标 | 数值 |\n|------|------|\n| 显存使用 | ${memoryUsed} / ${memoryTotal} |\n| 利用率 | ${utilization} |\n| 温度 | ${temp} |\n| 功耗 | ${gpu.powerDraw || '-'} W |`;
  });

  const content = `🎮 **GPU 状态**\n\n共检测到 **${gpuInfo.length}** 个 GPU：\n\n${gpuCards.join('\n\n---\n\n')}`;

  return { content, type: 'success' };
}

/**
 * 格式化操作成功信息
 */
export function formatSuccess(message: string): FormattedResult {
  return {
    content: `✅ ${message}`,
    type: 'success',
  };
}

/**
 * 格式化操作错误信息
 */
export function formatError(message: string): FormattedResult {
  return {
    content: `❌ ${message}`,
    type: 'error',
  };
}

/**
 * 格式化帮助信息
 */
export function formatHelpMessage(isAdmin = true): FormattedResult {
  const systemHelp = isAdmin
    ? `

**系统状态**
• \`/studio system\`
• \`/studio gpu\``
    : '';

  return {
    content: `🎯 **Studio 前端管理命令指南**

**Tab 切换**
• \`/studio tab datasets\`
• \`/studio tab models\`

**数据集管理**
• \`/studio datasets\`
• \`/studio datasets download xxx\`
• \`/studio datasets upload\`

**模型管理**
• \`/studio models\`
• \`/studio models refresh\`

**评测管理**
• \`/studio evaluation\`
• \`/studio evaluation refresh\`${systemHelp}

**帮助**
• \`/studio help\``,
    type: 'info',
  };
}

/**
 * 格式化文件大小
 */
function formatFileSize(bytes: number | undefined): string {
  if (bytes === undefined || bytes === null) return '-';
  
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let size = bytes;
  let unitIndex = 0;
  
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex++;
  }
  
  return `${size.toFixed(2)} ${units[unitIndex]}`;
}
