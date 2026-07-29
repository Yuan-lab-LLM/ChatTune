/**
 * Studio Command Parser
 * 只解析带 /studio 前缀的前端管理命令；普通自然语言交给后台 agent。
 */

export type CommandType = 
  | 'tab_switch'
  | 'datasets_view'      // 仅打开Tab，不查询
  | 'datasets_query'     // 条件查询：未查询过则查询，已查询过则显示结果
  | 'datasets_refresh'   // 强制刷新
  | 'datasets_upload'
  | 'datasets_download'
  | 'models_view'        // 仅打开Tab，不查询
  | 'models_query'       // 条件查询
  | 'models_refresh'     // 强制刷新
  | 'evaluation_view'    // 仅打开Tab，不查询
  | 'evaluation_query'   // 条件查询
  | 'evaluation_refresh' // 强制刷新
  | 'evaluation_upload'
  | 'evaluation_download'
  | 'system_view'        // 仅打开overview Tab
  | 'system_query'       // 查看系统状态并输出
  | 'system_gpu'
  | 'help'
  | 'unknown';

export interface ParsedCommand {
  type: CommandType;
  action?: string;
  params?: Record<string, string>;
  rawInput: string;
  confidence: number;
}

const TAB_NAME_MAP: Record<string, string> = {
  '运行': 'runs',
  '运行列表': 'runs',
  'runs': 'runs',
  'run': 'runs',
  '系统概览': 'overview',
  '概览': 'overview',
  '系统': 'overview',
  'overview': 'overview',
  '数据': 'datasets',
  '数据集': 'datasets',
  '数据集列表': 'datasets',
  '数据管理': 'datasets',
  'datasets': 'datasets',
  'dataset': 'datasets',
  '模型': 'models',
  '模型列表': 'models',
  '模型管理': 'models',
  'models': 'models',
  'model': 'models',
  '评测': 'evaluation',
  '评测列表': 'evaluation',
  '评测管理': 'evaluation',
  'evaluation': 'evaluation',
};

const AGENT_PROMPT_BYPASS_PATTERNS: RegExp[] = [
  /(?:^|\s)(?:model_path|dataset_path|data_path|ckpt_path|checkpoint_path|output_dir)\s*[:=]/i,
  /(?:模型|数据|评测|checkpoint|ckpt)\s*路径\s*(?:[:=]|是)/i,
  /\/home\/workspace\/(?:models|data|datasets|output|checkpoints)\//i,
  /\b\d{8}\b/,
];

export function shouldBypassCommandDetection(input: string): boolean {
  const trimmedInput = input.trim();

  if (!trimmedInput || trimmedInput.startsWith('/')) {
    return false;
  }

  const matchedPatterns = AGENT_PROMPT_BYPASS_PATTERNS.filter((pattern) =>
    pattern.test(trimmedInput),
  );

  // 包含路径/参数描述的长句优先交给后台 agent，而不是前端管理命令。
  return matchedPatterns.length >= 2;
}

export function parseCommand(input: string): ParsedCommand {
  const trimmedInput = input.trim();

  const slashCommand = parseSlashCommand(trimmedInput);
  if (slashCommand) return slashCommand;
  return { type: 'unknown', rawInput: input, confidence: 0 };
}

export function isPotentialCommand(input: string): boolean {
  const parsed = parseCommand(input);
  return parsed.confidence > 0.5;
}

function command(
  type: CommandType,
  rawInput: string,
  params: Record<string, string> = {},
): ParsedCommand {
  return { type, params, rawInput, confidence: 1 };
}

function invalidStudioCommand(rawInput: string, reason?: string): ParsedCommand {
  return command('unknown', rawInput, reason ? { reason } : {});
}

function parseStudioNaturalAlias(
  body: string,
  rawInput: string,
): ParsedCommand | null {
  const normalized = body.trim().toLowerCase();
  const compact = normalized.replace(/\s+/g, '');

  if (/^(帮助|help|commands?|命令|怎么用)$/.test(compact)) {
    return command('help', rawInput);
  }

  if (/(gpu|显卡|显存)/i.test(body)) {
    if (/(状态|信息|status|info)/i.test(body) || compact === 'gpu') {
      return command('system_gpu', rawInput, { targetTab: 'overview' });
    }
  }

  if (/(系统|概览|system|overview)/i.test(body)) {
    if (/(打开|显示|切换|view|tab|overview)/i.test(body)) {
      return command('system_view', rawInput, { targetTab: 'overview' });
    }
    if (/(查询|获取|状态|信息|query|status|info)/i.test(body)) {
      return command('system_query', rawInput, { targetTab: 'overview' });
    }
  }

  if (/(数据集|数据|datasets?|dataset)/i.test(body)) {
    const datasetName =
      body.match(/(?:下载|download)\s*(?:数据集|数据|datasets?|dataset)?\s+(.+)$/i)?.[1]?.trim() ||
      body.match(/(?:数据集|数据|datasets?|dataset)\s+(?:下载|download)\s+(.+)$/i)?.[1]?.trim();

    if (datasetName) {
      return command('datasets_download', rawInput, {
        datasetName,
        targetTab: 'datasets',
      });
    }
    if (/(上传|upload)/i.test(body)) {
      return command('datasets_upload', rawInput, { targetTab: 'datasets' });
    }
    if (/(刷新|重新加载|refresh|reload)/i.test(body)) {
      return command('datasets_refresh', rawInput, { targetTab: 'datasets' });
    }
    if (/(切换|打开|显示|查看|view|tab)/i.test(body)) {
      return command('datasets_view', rawInput, { targetTab: 'datasets' });
    }
    if (/(查询|有哪些|列出|列表|获取|query|list|show)/i.test(body)) {
      return command('datasets_query', rawInput, { targetTab: 'datasets' });
    }
  }

  if (/(模型|models?|model)/i.test(body)) {
    if (/(刷新|重新加载|refresh|reload)/i.test(body)) {
      return command('models_refresh', rawInput, { targetTab: 'models' });
    }
    if (/(切换|打开|显示|查看|view|tab)/i.test(body)) {
      return command('models_view', rawInput, { targetTab: 'models' });
    }
    if (/(查询|有哪些|列出|列表|获取|query|list|show)/i.test(body)) {
      return command('models_query', rawInput, { targetTab: 'models' });
    }
  }

  if (/(评测|评估|evaluation|eval|tests?|test)/i.test(body)) {
    const testName =
      body.match(/(?:下载|download)\s*(?:评测|评估|evaluation|eval|tests?|test)?\s+(.+)$/i)?.[1]?.trim() ||
      body.match(/(?:评测|评估|evaluation|eval|tests?|test)\s+(?:下载|download)\s+(.+)$/i)?.[1]?.trim();

    if (testName) {
      return command('evaluation_download', rawInput, {
        testName,
        targetTab: 'evaluation',
      });
    }
    if (/(上传|upload)/i.test(body)) {
      return command('evaluation_upload', rawInput, { targetTab: 'evaluation' });
    }
    if (/(刷新|重新加载|refresh|reload)/i.test(body)) {
      return command('evaluation_refresh', rawInput, { targetTab: 'evaluation' });
    }
    if (/(切换|打开|显示|查看|view|tab)/i.test(body)) {
      return command('evaluation_view', rawInput, { targetTab: 'evaluation' });
    }
    if (/(查询|有哪些|列出|列表|获取|query|list|show)/i.test(body)) {
      return command('evaluation_query', rawInput, { targetTab: 'evaluation' });
    }
  }

  return null;
}

export function parseSlashCommand(input: string): ParsedCommand | null {
  const trimmed = input.trim();
  
  if (!/^\/studio(?:\s|$)/i.test(trimmed)) {
    return null;
  }

  const body = trimmed.replace(/^\/studio\s*/i, '').trim();
  const aliasCommand = parseStudioNaturalAlias(body, input);
  if (aliasCommand) return aliasCommand;

  const parts = body.split(/\s+/).filter(Boolean);
  if (parts.length === 0) {
    return command('help', input);
  }

  const cmd = parts[0].toLowerCase();
  const args = parts.slice(1);
  const subCmd = args[0]?.toLowerCase();

  // Tab 切换
  if (cmd === 'tab' && args.length > 0) {
    const targetTab = TAB_NAME_MAP[args[0].toLowerCase()] || args[0];
    if (!['runs', 'overview', 'datasets', 'models', 'evaluation'].includes(targetTab)) {
      return invalidStudioCommand(input, `未知页面：${args[0]}`);
    }
    return command('tab_switch', input, { targetTab });
  }

  // 系统相关
  if (cmd === 'system') {
    if (subCmd === 'view' || subCmd === 'overview' || subCmd === 'tab') {
      return command('system_view', input, { targetTab: 'overview' });
    }
    return command('system_query', input, { targetTab: 'overview' });
  }

  if (cmd === 'gpu') {
    return command('system_gpu', input, { targetTab: 'overview' });
  }

  // 数据集相关
  if (cmd === 'datasets' || cmd === 'dataset') {
    if (subCmd === 'view' || subCmd === 'tab') {
      return command('datasets_view', input, { targetTab: 'datasets' });
    }
    if (subCmd === 'refresh' || subCmd === 'reload') {
      return command('datasets_refresh', input, { targetTab: 'datasets' });
    }
    if (subCmd === 'upload') {
      return command('datasets_upload', input, { targetTab: 'datasets' });
    }
    if (subCmd === 'download') {
      const datasetName = args.slice(1).join(' ').trim();
      if (!datasetName) {
        return invalidStudioCommand(input, '请指定要下载的数据集名称');
      }
      return command('datasets_download', input, {
        datasetName,
        targetTab: 'datasets',
      });
    }
    if (subCmd && !['list', 'query', 'show'].includes(subCmd)) {
      return invalidStudioCommand(input, `未知数据集命令：${subCmd}`);
    }
    // 默认查询
    return command('datasets_query', input, { targetTab: 'datasets' });
  }

  // 模型相关
  if (cmd === 'models' || cmd === 'model') {
    if (subCmd === 'view' || subCmd === 'tab') {
      return command('models_view', input, { targetTab: 'models' });
    }
    if (subCmd === 'refresh' || subCmd === 'reload') {
      return command('models_refresh', input, { targetTab: 'models' });
    }
    if (subCmd && !['list', 'query', 'show'].includes(subCmd)) {
      return invalidStudioCommand(input, `未知模型命令：${subCmd}`);
    }
    // 默认查询
    return command('models_query', input, { targetTab: 'models' });
  }

  // 评测相关
  if (cmd === 'evaluation' || cmd === 'eval') {
    if (subCmd === 'view' || subCmd === 'tab') {
      return command('evaluation_view', input, { targetTab: 'evaluation' });
    }
    if (subCmd === 'refresh' || subCmd === 'reload') {
      return command('evaluation_refresh', input, { targetTab: 'evaluation' });
    }
    if (subCmd === 'upload') {
      return command('evaluation_upload', input, { targetTab: 'evaluation' });
    }
    if (subCmd === 'download') {
      const testName = args.slice(1).join(' ').trim();
      if (!testName) {
        return invalidStudioCommand(input, '请指定要下载的评测文件名称');
      }
      return command('evaluation_download', input, {
        testName,
        targetTab: 'evaluation',
      });
    }
    if (subCmd && !['list', 'query', 'show'].includes(subCmd)) {
      return invalidStudioCommand(input, `未知评测命令：${subCmd}`);
    }
    // 默认查询
    return command('evaluation_query', input, { targetTab: 'evaluation' });
  }

  // 帮助
  if (cmd === 'help') {
    return command('help', input);
  }

  return invalidStudioCommand(input, `未知命令：${parts[0]}`);
}

export function getCommandHelp(isAdmin = true): string {
  const systemHelp = isAdmin
    ? `

**🖥️ 系统状态**

• \`/studio system view\` - 打开系统概览Tab

• \`/studio system\` - 查询系统状态

• \`/studio gpu\` - 显示GPU信息

• 也支持：\`/studio gpu status\`、\`/studio 查看显卡状态\`
`
    : '';

  return `🎯 **Studio 前端管理命令指南**

**📊 数据集管理**

• \`/studio datasets view\` - 仅打开Tab

• \`/studio datasets\` - 优先使用缓存

• \`/studio datasets refresh\` - 强制刷新

• \`/studio datasets download xxx\` - 下载（显示进度）

• \`/studio datasets upload\` - 上传

• 也支持：\`/studio 查询数据集列表\`、\`/studio dataset list\`

**🤖 模型管理**

• \`/studio models view\` - 仅打开Tab

• \`/studio models\` - 优先使用缓存

• \`/studio models refresh\` - 强制刷新

• 也支持：\`/studio 查询模型列表\`、\`/studio model list\`

**🧪 评测管理**

• \`/studio evaluation view\` - 仅打开Tab

• \`/studio evaluation\` - 优先使用缓存

• \`/studio evaluation refresh\` - 强制刷新

• \`/studio evaluation download xxx\` - 下载（显示进度）

• \`/studio evaluation upload\` - 上传

• 也支持：\`/studio 查询评测列表\`、\`/studio eval list\`
${systemHelp}

💡 **提示**: 所有命令都可以在模板库的「管理」分类中找到

💡 **查询 vs 刷新**: "查询"优先使用本地缓存，"刷新"总是从服务器获取最新数据`;
}
