/**
 * useNaturalLanguageCommands Hook
 * 自然语言命令处理的主要Hook
 */

import { useCallback, useState, useRef } from 'react';
import { ParsedCommand, getCommandHelp, parseSlashCommand } from './CommandParser';
import { FormattedResult } from './ResultFormatter';

export interface CommandHandler {
  onTabChange?: (tab: 'runs' | 'overview' | 'datasets' | 'models' | 'evaluation') => void;
  // Query handlers - now return data directly
  onQueryDatasets?: () => Promise<any[]>;
  onRefreshDatasets?: () => Promise<any[]>;
  onQueryModels?: () => Promise<any[]>;
  onRefreshModels?: () => Promise<any[]>;
  onQueryEvaluation?: () => Promise<any[]>;
  onRefreshEvaluation?: () => Promise<any[]>;
  onQueryEvaluationResults?: () => Promise<any[]>;
  onRefreshEvaluationResults?: () => Promise<any[]>;
  // Download handlers - now return Promise
  onDownloadDataset?: (name: string) => Promise<void>;
  onDownloadEvaluation?: (name: string) => Promise<void>;
  // Upload handlers
  onUploadDataset?: () => void;
  onUploadEvaluation?: () => void;
  // Data getters (fallback for cached data)
  getDatasets?: () => any[];
  getModels?: () => any[];
  getEvaluation?: () => any[];
  // Cached data getters (for query command with cache priority)
  getCachedDatasets?: () => any[] | null;
  getCachedModels?: () => any[] | null;
  getCachedTests?: () => any[] | null;
  getCachedEvaluationResults?: () => any[] | null;
  getDatasetCacheMeta?: () => { updatedAt?: string | null } | null;
  getModelCacheMeta?: () => { updatedAt?: string | null } | null;
  getTestCacheMeta?: () => { updatedAt?: string | null } | null;
  getEvaluationResultCacheMeta?: () => { updatedAt?: string | null } | null;
  // Query status
  hasQueriedDatasets?: boolean;
  hasQueriedModels?: boolean;
  hasQueriedTests?: boolean;
  // System info
  getSystemOverview?: () => any;
  getGPUInfo?: () => any[];
  requestGPUInfo?: () => Promise<any[]>;
  // Result display
  onShowResult?: (result: FormattedResult) => void;
}

export interface UseNaturalLanguageCommandsReturn {
  isProcessing: boolean;
  parsedCommand: ParsedCommand | null;
  processInput: (input: string) => Promise<boolean>;
  checkIsCommand: (input: string) => boolean;
}

export interface UseNaturalLanguageCommandsOptions {
  isAdmin?: boolean;
}

const ADMIN_ONLY_COMMAND_TYPES = new Set<ParsedCommand['type']>([
  'system_view',
  'system_query',
  'system_gpu',
]);
export function useNaturalLanguageCommands(
  handlers: CommandHandler,
  options: UseNaturalLanguageCommandsOptions = {},
): UseNaturalLanguageCommandsReturn {
  const [isProcessing, setIsProcessing] = useState(false);
  const [parsedCommand, setParsedCommand] = useState<ParsedCommand | null>(null);
  const handlersRef = useRef(handlers);
  const optionsRef = useRef(options);
  
  // Keep handlers ref up to date
  handlersRef.current = handlers;
  optionsRef.current = options;

  /**
   * 格式化数据集列表为表格
   */
  const formatDatasetsTable = (datasets: any[]): string => {
    if (datasets.length === 0) {
      return '📊 **数据集查询结果**\n\n暂无数据集。\n\n💡 提示：使用"上传数据集"命令或点击左侧"数据管理"入口上传数据。';
    }
    
    const rows = datasets.map(ds => {
      const size = ds.size ? formatBytes(ds.size) : '-';
      const type = ds.type || '-';
      return `| ${ds.name} | ${type} | ${size} |`;
    });
    
    return `📊 **数据集查询结果**\n\n共找到 **${datasets.length}** 个数据集：\n\n| 名称 | 类型 | 大小 |\n|------|------|------|\n${rows.join('\n')}`;
  };

  /**
   * 格式化模型列表为表格
   */
  const formatModelsTable = (models: any[]): string => {
    if (models.length === 0) {
      return '🤖 **模型查询结果**\n\n暂无模型。';
    }
    
    const getMergedSymbol = (merged: boolean | undefined, type: string): string => {
      if (merged) return '✓';
      // base_train 和 inference 类型的模型，未合并不影响使用，显示 —
      if (type === 'base_train' || type === 'inference') return '—';
      return '✗';
    };
    
    const rows = models.map(model => {
      const size = model.size ? formatBytes(model.size) : '-';
      const type = model.type || '-';
      const merged = getMergedSymbol(model.merged, type);
      return `| ${model.name} | ${type} | ${merged} | ${size} |`;
    });
    
    return `🤖 **模型查询结果**\n\n共找到 **${models.length}** 个模型：\n\n| 名称 | 类型 | 训练完成 | 大小 |\n|------|------|--------|------|\n${rows.join('\n')}`;
  };

  /**
   * 格式化评测列表为表格
   */
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

  const getEvaluationTypeName = (test: any): string => {
    if (test.category === 'general' || test.type === 'general') {
      return getGeneralEvaluationTypeName(test.filename || test.name || '');
    }

    const testTypeMap: Record<string, string> = {
      'exam2021': '中国执业医师资格考试',
      'exam2024': '临床医学综合能力(西医)',
      'usmle': '美国执业医师考试',
      'medbench': 'MedBench评测',
      'other': '其他',
    };

    return testTypeMap[test.type] || test.type || '其他';
  };

  const formatEvaluationTable = (tests: any[]): string => {
    // 构建开源评测部分
    let openSourceSection = '';
    if (tests.length === 0) {
      openSourceSection = `**开源评测**

暂无评测文件。

💡 提示：使用"上传评测集"命令或点击左侧"评测管理"入口上传评测文件。`;
    } else {
      const rows = tests.map(test => {
        const size = test.size ? formatBytes(test.size) : '-';
        const typeName = getEvaluationTypeName(test);
        return `| ${test.filename || test.name} | ${typeName} | ${size} |`;
      });
      
      openSourceSection = `**开源评测**

共找到 **${tests.length}** 个评测文件：

| 文件名 | 类型 | 大小 |
|--------|------|------|
${rows.join('\n')}`;
    }
    
    return `🧪 **评测查询结果**\n\n${openSourceSection}`;
  };

  /**
   * 格式化评测结果列表为表格
   */
  const formatEvaluationResultsTable = (results: any[]): string => {
    if (!results || results.length === 0) {
      return '';
    }
    
    const statusMap: Record<string, string> = {
      'finished': '已完成',
      'running': '运行中',
      'failed': '失败',
      'unknown': '未知',
    };
    
    const rows = results.map(result => {
      const status = statusMap[result.status] || result.status || '未知';
      const accuracy = result.accuracy !== undefined ? `${(result.accuracy * 100).toFixed(2)}%` : '-';
      const model = result.model ? (result.model.length > 15 ? result.model.substring(0, 15) + '...' : result.model) : '-';
      const dataset = result.dataset ? (result.dataset.length > 15 ? result.dataset.substring(0, 15) + '...' : result.dataset) : '-';
      return `| ${model} | ${dataset} | ${status} | ${accuracy} |`;
    });
    
    return `**三、评测结果列表**

共找到 **${results.length}** 条评测结果：

| 模型 | 数据集 | 状态 | 正确率 |
|------|--------|------|--------|
${rows.join('\n')}`;
  };

  const formatUpdatedAt = (updatedAt?: string | null): string => {
    if (!updatedAt) {
      return '更新时间：暂无';
    }

    const date = new Date(updatedAt);
    if (Number.isNaN(date.getTime())) {
      return '更新时间：暂无';
    }

    return `更新时间：${new Intl.DateTimeFormat('zh-CN', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      hour12: false,
    }).format(date)}`;
  };

  const appendMetaLine = (
    content: string,
    meta?: { updatedAt?: string | null } | null,
    label?: string,
  ): string => {
    const prefix = label ? `${label}` : '更新时间';
    if (!meta?.updatedAt) {
      return `${content}\n\n${prefix}：暂无`;
    }

    const date = new Date(meta.updatedAt);
    if (Number.isNaN(date.getTime())) {
      return `${content}\n\n${prefix}：暂无`;
    }

    return `${content}\n\n${prefix}：${new Intl.DateTimeFormat('zh-CN', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      hour12: false,
    }).format(date)}`;
  };

  /**
   * 处理命令
   */
  const processCommand = useCallback(async (command: ParsedCommand): Promise<FormattedResult | null> => {
    const handlers = handlersRef.current;
    const targetTab = command.params?.targetTab as 'runs' | 'overview' | 'datasets' | 'models' | 'evaluation';
    const isAdmin = optionsRef.current.isAdmin === true;

    if (!isAdmin && ADMIN_ONLY_COMMAND_TYPES.has(command.type)) {
      return {
        content: '⚠️ 该 Studio 命令仅管理员可用',
        type: 'warning',
      };
    }
    
    // 辅助函数：动态获取最新数据 - 每次调用都重新读取 handlersRef.current
    const getLatestData = () => {
      const currentHandlers = handlersRef.current;
      return {
        datasets: currentHandlers.getDatasets?.() || [],
        models: currentHandlers.getModels?.() || [],
        tests: currentHandlers.getEvaluation?.() || [],
        gpuInfo: currentHandlers.getGPUInfo?.() || [],
        systemOverview: currentHandlers.getSystemOverview?.(),
      };
    };
    
    switch (command.type) {
      // ========== 数据集 - 仅查看 ==========
      case 'datasets_view': {
        if (targetTab && handlers.onTabChange) {
          handlers.onTabChange(targetTab);
          return {
            content: '✅ 已打开 **数据集列表**',
            type: 'success',
          };
        }
        return { content: '❌ 无法打开数据集列表', type: 'error' };
      }

      // ========== 数据集 - 条件查询 ==========
      case 'datasets_query': {
        if (targetTab && handlers.onTabChange) {
          handlers.onTabChange(targetTab);
        }

        if (handlers.onQueryDatasets) {
          try {
            const datasets = await handlers.onQueryDatasets();
            return {
              content: appendMetaLine(
                formatDatasetsTable(datasets),
                handlers.getDatasetCacheMeta?.(),
              ),
              type: 'success',
            };
          } catch (error) {
            console.error('Failed to query datasets via command:', error);
          }
        }

        const cached = handlers.getCachedDatasets?.();
        if (cached !== null && cached !== undefined) {
          return {
            content: appendMetaLine(
              formatDatasetsTable(cached),
              handlers.getDatasetCacheMeta?.(),
            ),
            type: 'success',
          };
        }
        
        return { content: '❌ 无法查询数据集', type: 'error' };
      }

      // ========== 数据集 - 强制刷新 ==========
      case 'datasets_refresh': {
        if (targetTab && handlers.onTabChange) {
          handlers.onTabChange(targetTab);
        }
        
        if (handlers.onRefreshDatasets) {
          const datasets = await handlers.onRefreshDatasets();
          return {
            content: appendMetaLine(
              '🔄 已完成刷新，同步最新结果：\n\n' + formatDatasetsTable(datasets),
              handlers.getDatasetCacheMeta?.(),
            ),
            type: 'success',
          };
        }
        
        return { content: '❌ 无法刷新数据集', type: 'error' };
      }

      // ========== 数据集 - 下载 ==========
      case 'datasets_download': {
        const datasetName = command.params?.datasetName;
        
        if (targetTab && handlers.onTabChange) {
          handlers.onTabChange(targetTab);
        }
        
        // 检查是否指定了数据集名称
        if (!datasetName || datasetName.trim() === '') {
          return {
            content: '❌ 请指定要下载的数据集名称，例如："下载数据集 xxx"',
            type: 'error',
          };
        }
        
        if (handlers.onDownloadDataset) {
          const datasets = handlers.getDatasets?.() || [];
          const exists = datasets.some((ds: any) => ds.name === datasetName);
          
          if (!exists) {
            return {
              content: `⚠️ 未找到数据集 **${datasetName}**\n\n请使用 \`/studio datasets refresh\` 刷新列表，或检查数据集名称是否正确。`,
              type: 'warning',
            };
          }
          
          // 执行下载（loading 由 handleCommand 统一显示）
          await handlers.onDownloadDataset(datasetName);
          
          // 添加延迟，确保 loading 至少显示 500ms
          await new Promise(resolve => setTimeout(resolve, 500));
          
          // 下载完成后的消息
          return {
            content: `✅ **下载完成**

| 数据集名称 | 状态 |
|------------|------|
| ${datasetName} | 已下载完成 ✓ |

💡 提示：请在浏览器下载页面查看`,
            type: 'success',
          };
        }
        
        return {
          content: '❌ 请指定要下载的数据集名称，例如："下载数据集 xxx"',
          type: 'error',
        };
      }

      // ========== 数据集 - 上传 ==========
      case 'datasets_upload': {
        if (handlers.onUploadDataset) {
          handlers.onUploadDataset();
          return {
            content: '📤 已打开数据集上传对话框',
            type: 'success',
          };
        }
        return { content: '❌ 无法打开上传对话框', type: 'error' };
      }

      // ========== 模型 - 仅查看 ==========
      case 'models_view': {
        if (targetTab && handlers.onTabChange) {
          handlers.onTabChange(targetTab);
          return {
            content: '✅ 已打开 **模型列表**',
            type: 'success',
          };
        }
        return { content: '❌ 无法打开模型列表', type: 'error' };
      }

      // ========== 模型 - 条件查询 ==========
      case 'models_query': {
        if (targetTab && handlers.onTabChange) {
          handlers.onTabChange(targetTab);
        }

        if (handlers.onQueryModels) {
          try {
            const models = await handlers.onQueryModels();
            return {
              content: appendMetaLine(
                formatModelsTable(models),
                handlers.getModelCacheMeta?.(),
              ),
              type: 'success',
            };
          } catch (error) {
            console.error('Failed to query models via command:', error);
          }
        }

        const cached = handlers.getCachedModels?.();
        if (cached !== null && cached !== undefined) {
          return {
            content: appendMetaLine(
              formatModelsTable(cached),
              handlers.getModelCacheMeta?.(),
            ),
            type: 'success',
          };
        }
        
        return { content: '❌ 无法查询模型', type: 'error' };
      }

      // ========== 模型 - 强制刷新 ==========
      case 'models_refresh': {
        if (targetTab && handlers.onTabChange) {
          handlers.onTabChange(targetTab);
        }
        
        if (handlers.onRefreshModels) {
          const models = await handlers.onRefreshModels();
          return {
            content: appendMetaLine(
              '🔄 已完成刷新，同步最新结果：\n\n' + formatModelsTable(models),
              handlers.getModelCacheMeta?.(),
            ),
            type: 'success',
          };
        }
        
        return { content: '❌ 无法刷新模型', type: 'error' };
      }

      // ========== 评测 - 仅查看 ==========
      case 'evaluation_view': {
        if (targetTab && handlers.onTabChange) {
          handlers.onTabChange(targetTab);
          return {
            content: '✅ 已打开 **评测列表**',
            type: 'success',
          };
        }
        return { content: '❌ 无法打开评测列表', type: 'error' };
      }

      // ========== 评测 - 条件查询 ==========
      case 'evaluation_query': {
        if (targetTab && handlers.onTabChange) {
          handlers.onTabChange(targetTab);
        }

        if (handlers.onQueryEvaluation && handlers.onQueryEvaluationResults) {
          try {
            const [tests, results] = await Promise.all([
              handlers.onQueryEvaluation(),
              handlers.onQueryEvaluationResults(),
            ]);
            
            const testsTable = formatEvaluationTable(tests);
            const resultsTable = results && results.length > 0
              ? '\n\n' + formatEvaluationResultsTable(results)
              : '';
            
            return {
              content:
                appendMetaLine(
                  testsTable + resultsTable,
                  handlers.getTestCacheMeta?.(),
                  '评测集更新时间',
                ) +
                '\n' +
                formatUpdatedAt(handlers.getEvaluationResultCacheMeta?.()?.updatedAt).replace(
                  '更新时间',
                  '评测结果更新时间',
                ),
              type: 'success',
            };
          } catch (error) {
            console.error('Failed to query evaluation via command:', error);
          }
        }

        if (handlers.onQueryEvaluation) {
          try {
            const tests = await handlers.onQueryEvaluation();
            return {
              content: appendMetaLine(
                formatEvaluationTable(tests),
                handlers.getTestCacheMeta?.(),
                '评测集更新时间',
              ),
              type: 'success',
            };
          } catch (error) {
            console.error('Failed to query evaluation tests via command:', error);
          }
        }

        const cachedTests = handlers.getCachedTests?.();
        const cachedResults = handlers.getCachedEvaluationResults?.();
        if (cachedTests !== null && cachedTests !== undefined) {
          const testsTable = formatEvaluationTable(cachedTests);
          const resultsTable = cachedResults && cachedResults.length > 0
            ? '\n\n' + formatEvaluationResultsTable(cachedResults)
            : '';
          return {
            content:
              appendMetaLine(
                testsTable + resultsTable,
                handlers.getTestCacheMeta?.(),
                '评测集更新时间',
              ) +
              '\n' +
              formatUpdatedAt(handlers.getEvaluationResultCacheMeta?.()?.updatedAt).replace(
                '更新时间',
                '评测结果更新时间',
              ),
            type: 'success',
          };
        }
        
        return { content: '❌ 无法查询评测文件', type: 'error' };
      }

      // ========== 评测 - 强制刷新 ==========
      case 'evaluation_refresh': {
        if (targetTab && handlers.onTabChange) {
          handlers.onTabChange(targetTab);
        }
        
        // 强制刷新时同时查询评测数据集和评测结果
        if (handlers.onRefreshEvaluation && handlers.onRefreshEvaluationResults) {
          const [tests, results] = await Promise.all([
            handlers.onRefreshEvaluation(),
            handlers.onRefreshEvaluationResults(),
          ]);
          
          const testsTable = formatEvaluationTable(tests);
          const resultsTable = results && results.length > 0
            ? '\n\n' + formatEvaluationResultsTable(results)
            : '';
          
          return {
            content:
              appendMetaLine(
                '🔄 已完成刷新，同步最新结果：\n\n' + testsTable + resultsTable,
                handlers.getTestCacheMeta?.(),
                '评测集更新时间',
              ) +
              '\n' +
              formatUpdatedAt(handlers.getEvaluationResultCacheMeta?.()?.updatedAt).replace(
                '更新时间',
                '评测结果更新时间',
              ),
            type: 'success',
          };
        }
        
        // 如果没有评测结果查询处理器，只查询评测数据集
        if (handlers.onRefreshEvaluation) {
          const tests = await handlers.onRefreshEvaluation();
          return {
            content: appendMetaLine(
              '🔄 已完成刷新，同步最新结果：\n\n' + formatEvaluationTable(tests),
              handlers.getTestCacheMeta?.(),
              '评测集更新时间',
            ),
            type: 'success',
          };
        }
        
        return { content: '❌ 无法刷新评测列表', type: 'error' };
      }

      // ========== 评测 - 下载 ==========
      case 'evaluation_download': {
        const testName = command.params?.testName;
        
        if (targetTab && handlers.onTabChange) {
          handlers.onTabChange(targetTab);
        }
        
        // 检查是否指定了文件名
        if (!testName || testName.trim() === '') {
          return {
            content: '❌ 请指定要下载的评测文件名称，例如："下载评测 2021.json"',
            type: 'error',
          };
        }
        
        if (handlers.onDownloadEvaluation) {
          const tests = handlers.getEvaluation?.() || [];
          
          // 评测类型映射表（中文）
          const testTypeMap: Record<string, string> = {
            'exam2021': '中国执业医师资格考试',
            'exam2024': '临床医学综合能力(西医)',
            'usmle': '美国执业医师考试',
            'medbench': 'MedBench评测',
            'other': '其他',
          };
          
          // 评测类型映射表（英文）
          const testTypeMapEn: Record<string, string> = {
            'exam2021': 'National Medical Licensing Examination (NMLE)',
            'exam2024': 'Clinical Medicine Comprehensive Ability (Western Medicine)',
            'usmle': 'United States Medical Licensing Examination (USMLE)',
            'medbench': 'MedBench Evaluation',
            'other': 'Other',
          };
          
          // 多维度匹配函数
          const findMatchingTests = (input: string) => {
            const normalizedInput = input.toLowerCase().trim();
            
            return tests.filter((t: any) => {
              const filename = (t.filename || t.name || '').toLowerCase();
              const type = (t.type || 'other').toLowerCase();
              const description = (t.description || '').toLowerCase();
              
              // 1. 完全匹配 filename
              if (filename === normalizedInput) return true;
              
              // 2. 忽略扩展名匹配 filename
              const filenameNoExt = filename.replace(/\.[^/.]+$/, '');
              const inputNoExt = normalizedInput.replace(/\.[^/.]+$/, '');
              if (filenameNoExt === inputNoExt) return true;
              
              // 3. 匹配 type 对应的中文名称
              const typeName = (testTypeMap[t.type] || getEvaluationTypeName(t) || '').toLowerCase();
              if (typeName.includes(normalizedInput)) return true;
              
              // 3.1 匹配 type 对应的英文名称
              const typeNameEn = (testTypeMapEn[t.type] || '').toLowerCase();
              if (typeNameEn.includes(normalizedInput)) return true;
              
              // 4. 匹配 description 字段
              if (description.includes(normalizedInput)) return true;
              
              // 5. 部分包含匹配 filename
              if (filename.includes(normalizedInput)) return true;
              if (normalizedInput.includes(filenameNoExt)) return true;
              
              return false;
            });
          };
          
          const matchingTests = findMatchingTests(testName);
          
          // 根据匹配结果数量处理
          if (matchingTests.length === 0) {
            return {
              content: `⚠️ 未找到评测文件 **${testName}**\n\n请使用"刷新评测列表"命令刷新列表，或检查文件名/评测名称是否正确。`,
              type: 'warning',
            };
          } else if (matchingTests.length === 1) {
            // 单个匹配，直接下载
            const matchedTest = matchingTests[0];
            const actualFilename = matchedTest.filename || matchedTest.name;
            
            // 执行下载（loading 由 handleCommand 统一显示）
            await handlers.onDownloadEvaluation(actualFilename);
            
            // 添加延迟，确保 loading 至少显示 500ms
            await new Promise(resolve => setTimeout(resolve, 500));
            
            // 下载完成后的消息
            return {
              content: `✅ **下载完成**

| 评测文件名称 | 状态 |
|--------------|------|
| ${actualFilename} | 已下载完成 ✓ |

💡 提示：请在浏览器下载页面查看`,
              type: 'success',
            };
          } else {
            // 多个匹配，提示用户选择
            const testTypeMapForDisplay: Record<string, string> = {
              'exam2021': '中国执业医师资格考试',
              'exam2024': '临床医学综合能力(西医)',
              'usmle': '美国执业医师考试',
              'medbench': 'MedBench评测',
              'other': '其他',
            };
            
            const matchesList = matchingTests.map((test: any, index: number) => {
              const filename = test.filename || test.name || 'unknown';
              const typeName = getEvaluationTypeName(test);
              return `${index + 1}. ${filename} (${typeName})`;
            }).join('\n');
            
            return {
              content: `⚠️ 找到 **${matchingTests.length}** 个匹配的评测文件\n\n请明确指定要下载的文件名称：\n\n${matchesList}\n\n💡 提示：请使用完整文件名，例如"下载评测 ${matchingTests[0].filename || matchingTests[0].name}"`,
              type: 'warning',
            };
          }
        }
        
        return {
          content: '❌ 请指定要下载的评测文件名称，例如："下载评测 xxx"',
          type: 'error',
        };
      }

      // ========== 评测 - 上传 ==========
      case 'evaluation_upload': {
        if (handlers.onUploadEvaluation) {
          handlers.onUploadEvaluation();
          return {
            content: '📤 已打开评测集上传对话框',
            type: 'success',
          };
        }
        return { content: '❌ 无法打开上传对话框', type: 'error' };
      }

      // ========== 系统 - 仅查看 ==========
      case 'system_view': {
        if (targetTab && handlers.onTabChange) {
          handlers.onTabChange(targetTab);
          return {
            content: '✅ 已打开 **系统概览**',
            type: 'success',
          };
        }
        return { content: '❌ 无法打开系统概览', type: 'error' };
      }

      // ========== 系统 - 查询 ==========
      case 'system_query': {
        if (targetTab && handlers.onTabChange) {
          handlers.onTabChange(targetTab);
        }
        
        const overview = handlers.getSystemOverview?.();
        if (overview) {
          // 构建基础信息表格
          let content = `🖥️ **系统状态**

| 指标 | 数值 |
|------|------|
| 在线用户 | ${overview.onlineUsers || 0} |
| 服务运行时间 | ${overview.serverUptime || '-'} |
| 今日消息 | ${overview.messageStats?.today || 0} |
| 本周消息 | ${overview.messageStats?.thisWeek || 0} |
| 本月消息 | ${overview.messageStats?.thisMonth || 0} |`;

          // 添加 GPU 信息（如果有）
          if (overview.gpuInfo && overview.gpuInfo.length > 0) {
            const gpuCards = overview.gpuInfo.map((gpu: any, index: number) => {
              // 修复显存单位：后端返回的是 MB，需要转换为 GB/MB
              const formatMemory = (value: number | string | undefined): string => {
                if (value === undefined || value === null) return '-';
                
                // 转换为数字（MB）
                let mbValue: number;
                if (typeof value === 'string') {
                  mbValue = parseFloat(value);
                } else {
                  mbValue = value;
                }
                
                if (isNaN(mbValue)) return '-';
                
                // 如果 >= 1000 MB，转换为 GB
                if (mbValue >= 1000) {
                  return (mbValue / 1024).toFixed(2) + ' GB';
                }
                
                // 否则显示 MB
                return mbValue.toFixed(0) + ' MB';
              };
              
              const memUsed = formatMemory(gpu.memoryUsed);
              const memTotal = formatMemory(gpu.memoryTotal);
              return `**GPU ${index}: ${gpu.name || 'Unknown'}**
• 显存: ${memUsed} / ${memTotal}
• 利用率: ${gpu.utilization !== undefined ? gpu.utilization + '%' : '-'}`;
            });
            
            content += `\n\n🎮 **GPU 状态**\n\n共检测到 **${overview.gpuInfo.length}** 个 GPU：\n\n${gpuCards.join('\n\n')}`;
          }
          
          return {
            content,
            type: 'success',
          };
        }
        return {
          content: '⚠️ 暂无系统信息',
          type: 'warning',
        };
      }

      // ========== GPU ==========
      case 'system_gpu': {
        if (targetTab && handlers.onTabChange) {
          handlers.onTabChange(targetTab);
        }
        
        let gpuInfo: any[] = [];
        
        // 优先使用 requestGPUInfo 主动请求并等待响应（最多等待 8 秒）
        if (handlers.requestGPUInfo) {
          try {
            gpuInfo = await Promise.race([
              handlers.requestGPUInfo(),
              new Promise<any[]>((_, reject) => 
                setTimeout(() => reject(new Error('GPU info request timeout')), 8000)
              )
            ]);
          } catch (error) {
            console.warn('GPU info request failed or timeout, falling back to cached data');
            gpuInfo = [];
          }
        }
        
        // 如果主动请求失败或不可用，回退到轮询机制
        if (gpuInfo.length === 0) {
          let attempts = 0;
          const maxAttempts = 80; // 80 * 100ms = 8秒
          gpuInfo = getLatestData().gpuInfo;
          
          while (gpuInfo.length === 0 && attempts < maxAttempts) {
            await new Promise(resolve => setTimeout(resolve, 100));
            gpuInfo = getLatestData().gpuInfo;
            attempts++;
          }
        }
        
        if (gpuInfo.length > 0) {
          const gpuCards = gpuInfo.map((gpu: any, index: number) => {
            // 修复显存单位：后端返回的是 MB，需要转换为 GB/MB
            const formatMemory = (value: number | string | undefined): string => {
              if (value === undefined || value === null) return '-';
              
              // 转换为数字（MB）
              let mbValue: number;
              if (typeof value === 'string') {
                mbValue = parseFloat(value);
              } else {
                mbValue = value;
              }
              
              if (isNaN(mbValue)) return '-';
              
              // 如果 >= 1000 MB，转换为 GB
              if (mbValue >= 1000) {
                return (mbValue / 1024).toFixed(2) + ' GB';
              }
              
              // 否则显示 MB
              return mbValue.toFixed(0) + ' MB';
            };
            
            const memUsed = formatMemory(gpu.memoryUsed);
            const memTotal = formatMemory(gpu.memoryTotal);
            return `**GPU ${index}: ${gpu.name || 'Unknown'}**
• 显存: ${memUsed} / ${memTotal}
• 利用率: ${gpu.utilization !== undefined ? gpu.utilization + '%' : '-'}`;
          });
          return {
            content: `🎮 **GPU 状态**

共检测到 **${gpuInfo.length}** 个 GPU：

${gpuCards.join('\n\n')}`,
            type: 'success',
          };
        }
        return {
          content: '⚠️ 暂无 GPU 信息\n\n💡 提示：点击左侧"系统概览"入口可查看更多系统信息',
          type: 'warning',
        };
      }

      // ========== 运行列表 ==========
      case 'tab_switch': {
        if (targetTab && handlers.onTabChange) {
          handlers.onTabChange(targetTab);
          const tabNames: Record<string, string> = {
            'runs': '运行列表',
            'overview': '系统概览',
            'datasets': '数据管理',
            'models': '模型管理',
            'evaluation': '评测管理',
          };
          return {
            content: `✅ 已切换到 **${tabNames[targetTab] || targetTab}** 页面`,
            type: 'success',
          };
        }
        return {
          content: '❌ 无法切换功能区域，请指定有效的页面名称',
          type: 'error',
        };
      }

      // ========== 帮助 ==========
      case 'help': {
        return {
          content: getCommandHelp(isAdmin),
          type: 'info',
        };
      }

      case 'unknown': {
        return {
          content: `❌ ${
            command.params?.reason || '无法识别这个 Studio 命令'
          }\n\n请输入 \`/studio help\` 查看可用命令。`,
          type: 'error',
        };
      }

      default:
        return null;
    }
  }, []);

  /**
   * 检查输入是否为命令
   */
  const checkIsCommand = useCallback((input: string): boolean => {
    return parseSlashCommand(input) !== null;
  }, []);

  /**
   * 处理输入
   * @returns 是否为命令（true: 已处理为命令，false: 普通消息）
   */
  const processInput = useCallback(async (input: string): Promise<boolean> => {
    const trimmed = input.trim();
    
    // 只处理带 /studio 前缀的前端管理命令；普通自然语言交给后台 agent。
    const command = parseSlashCommand(trimmed);
    if (!command) {
      return false;
    }

    setParsedCommand(command);
    setIsProcessing(true);

    try {
      const result = await processCommand(command);
      if (result && handlersRef.current.onShowResult) {
        handlersRef.current.onShowResult(result);
      }
      return true;
    } catch (error) {
      console.error('Command execution error:', error);
      if (handlersRef.current.onShowResult) {
        handlersRef.current.onShowResult({
          content: '❌ 命令执行失败，请重试',
          type: 'error',
        });
      }
      return true;
    } finally {
      setIsProcessing(false);
    }
  }, [processCommand]);

  return {
    isProcessing,
    parsedCommand,
    processInput,
    checkIsCommand,
  };
}

/**
 * 格式化字节大小
 */
function formatBytes(bytes: number | string | undefined): string {
  // 处理 undefined 或 null
  if (bytes === undefined || bytes === null) {
    return '-';
  }
  
  // 如果已经是字符串，直接返回
  if (typeof bytes === 'string') {
    // 如果字符串已经包含单位（如 "412 B"），直接返回
    if (/\s*(B|KB|MB|GB|TB)\s*$/i.test(bytes)) {
      return bytes;
    }
    // 否则尝试转换为数字
    const num = parseFloat(bytes);
    if (isNaN(num)) {
      return bytes || '-';
    }
    bytes = num;
  }
  
  // 处理数字
  if (typeof bytes === 'number') {
    if (bytes === 0) return '0 B';
    if (isNaN(bytes)) return '-';
    
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }
  
  return '-';
}

// Re-export types
export * from './CommandParser';
export * from './ResultFormatter';
