import { execFile } from 'child_process';
import { promisify } from 'util';
import * as path from 'path';
import { existsSync } from 'fs';

const execFileAsync = promisify(execFile);

const findAgentRoot = (): string => {
    const configured = process.env.MEDFLOW_AGENT_ROOT?.trim();
    if (configured) {
        return configured;
    }

    const candidates = [
        path.resolve(process.cwd(), '..', 'agent'),
        path.resolve(process.cwd(), '..', '..', 'agent'),
        path.resolve(process.cwd(), '..', '..', '..', 'agent'),
        path.resolve(process.cwd(), 'agent'),
        '/home/workspace/agent',
    ];

    return candidates.find((candidate) => existsSync(path.join(candidate, "medflow_agent_tools"))) || candidates[0];
};

const getPythonExecutables = (): string[] => {
    const configured = process.env.MEDFLOW_PYTHON_EXECUTABLE?.trim();
    const candidates = process.platform === 'win32'
        ? ['py', 'python', 'python3']
        : [
              process.env.PYTHON?.trim() || '',
              'python3',
              'python',
              '/bin/python3',
              '/bin/python',
              '/usr/bin/python3',
              '/usr/local/bin/python3',
              '/usr/local/bin/python',
              '/opt/conda/bin/python',
              '/root/miniconda3/bin/python',
              '/home/workspace/miniconda3/bin/python',
              '/usr/bin/python',
          ].filter(Boolean);

    return configured ? [configured, ...candidates.filter((item) => item !== configured)] : candidates;
};

export interface AgentMonitorParams {
    container?: string;
    pid?: string;
    trainType?: string;
    historyLimit?: number;
    timeWindowMinutes?: number;
}

export interface AgentWorkflowStatusParams {
    workflowId: string;
}

const normalizeTrainType = (trainType?: string): string | undefined => {
    const value = trainType?.trim().toLowerCase();
    if (!value) {
        return undefined;
    }

    if (value.includes('grpo')) {
        return 'grpo';
    }
    if (value.includes('lora')) {
        return 'lora';
    }
    if (value.includes('全参') || value.includes('full')) {
        return 'full';
    }
    if (value.includes('增强') || value.includes('dpo') || value.includes('enhanced')) {
        return 'dpo';
    }

    return value;
};

export async function queryAgentTrainingMetrics(params: AgentMonitorParams): Promise<Record<string, unknown>> {
    const agentRoot = findAgentRoot();
    const agentPackageRoot = agentRoot;
    if (!existsSync(agentRoot)) {
        throw new Error(
            `Agent root not found for silent training monitor: ${agentRoot}. ` +
                'Set MEDFLOW_AGENT_ROOT to the agent directory used by the runtime.',
        );
    }
    if (!existsSync(path.join(agentPackageRoot, "medflow_agent_tools"))) {
        throw new Error(
            `MedFlow agent tools not found for silent training monitor under: ${agentPackageRoot}. ` +
                "Set MEDFLOW_AGENT_ROOT to the agent directory that contains medflow_agent_tools.",
        );
    }

    const requestPayload = {
        container_name: params.container,
        pid: params.pid,
        train_type: normalizeTrainType(params.trainType),
        history_limit: params.historyLimit ?? 200,
        time_window_minutes: params.timeWindowMinutes ?? 120,
    };
    const payload = JSON.stringify(requestPayload);

    console.log('[AgentMonitorService] calling monitor_training', {
        agentRoot,
        agentPackageRoot,
        args: requestPayload,
    });

    const script = [
        'import importlib, json, os, sys',
        `agent_package_root = ${JSON.stringify(agentPackageRoot)}`,
        'agent_package_root_abs = os.path.abspath(agent_package_root)',
        'sys.path = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != agent_package_root_abs]',
        'try:',
        '    monitor_module = importlib.import_module("medflow_agent_tools.runlocal_monitor")',
        '    monitor_import_source = "python_environment"',
        'except Exception:',
        '    sys.path.insert(0, agent_package_root)',
        '    monitor_module = importlib.import_module("medflow_agent_tools.runlocal_monitor")',
        '    monitor_import_source = "agent_package_root_fallback"',
        'monitor_training = monitor_module.monitor_training',
        'def _response_text(response):',
        '    if not getattr(response, "content", None):',
        '        return ""',
        '    first = response.content[0]',
        '    return first.get("text", "") if isinstance(first, dict) else getattr(first, "text", "")',
        'def _load_payload(text):',
        '    try:',
        '        return json.loads(text) if text else {}',
        '    except Exception:',
        '        return {}',
        'def _has_metrics(data):',
        '    metrics = data.get("metrics", {}) if isinstance(data, dict) else {}',
        '    history = metrics.get("history") if isinstance(metrics, dict) else None',
        '    return bool(metrics.get("wandb_run_dir") or metrics.get("latest_loss") is not None or (isinstance(history, list) and len(history) > 0))',
        'args = json.loads(sys.argv[1])',
        'response = monitor_training(**args)',
        'text = _response_text(response)',
        'data = _load_payload(text)',
        'metrics = data.get("metrics", {}) if isinstance(data, dict) else {}',
        'needs_fallback = bool(args.get("pid")) and not _has_metrics(data) and metrics.get("wandb_select_reason") in {"pid_no_wandb_match", "metadata_pid_not_found", "no_runs"}',
        'if needs_fallback:',
        '    fallback_args = dict(args)',
        '    fallback_args.pop("pid", None)',
        '    fallback_args["time_window_minutes"] = max(int(args.get("time_window_minutes") or 0), 1440)',
        '    fallback_response = monitor_training(**fallback_args)',
        '    fallback_text = _response_text(fallback_response)',
        '    fallback_data = _load_payload(fallback_text)',
        '    fallback_metrics_preview = fallback_data.get("metrics", {}) if isinstance(fallback_data, dict) else {}',
        '    fallback_status_preview = fallback_data.get("status") if isinstance(fallback_data, dict) else None',
        '    if _has_metrics(fallback_data):',
        '        fallback_metrics = fallback_data.setdefault("metrics", {})',
        '        fallback_debug = fallback_metrics.setdefault("debug", {})',
        '        if isinstance(fallback_debug, dict):',
        '            fallback_debug["silent_monitor_pid_fallback_used"] = True',
        '            fallback_debug["silent_monitor_original_pid"] = args.get("pid")',
        '            fallback_debug["silent_monitor_original_wandb_select_reason"] = metrics.get("wandb_select_reason")',
        '            fallback_debug["silent_monitor_original_error_reason"] = metrics.get("error_reason")',
        '        text = json.dumps(fallback_data, ensure_ascii=False)',
        '    else:',
        '        debug = metrics.setdefault("debug", {}) if isinstance(metrics, dict) else {}',
        '        if isinstance(debug, dict):',
        '            debug["silent_monitor_pid_fallback_attempted"] = True',
        '            debug["silent_monitor_fallback_status"] = fallback_status_preview',
        '            debug["silent_monitor_fallback_wandb_select_reason"] = fallback_metrics_preview.get("wandb_select_reason") if isinstance(fallback_metrics_preview, dict) else None',
        '            debug["silent_monitor_fallback_error_reason"] = fallback_metrics_preview.get("error_reason") if isinstance(fallback_metrics_preview, dict) else None',
        '            debug["silent_monitor_fallback_wandb_run_dir"] = fallback_metrics_preview.get("wandb_run_dir") if isinstance(fallback_metrics_preview, dict) else None',
        '            debug["silent_monitor_fallback_latest_loss"] = fallback_metrics_preview.get("latest_loss") if isinstance(fallback_metrics_preview, dict) else None',
        '            debug["silent_monitor_fallback_history_count"] = fallback_metrics_preview.get("history_count") if isinstance(fallback_metrics_preview, dict) else None',
        '            text = json.dumps(data, ensure_ascii=False)',
        'data = _load_payload(text)',
        'if isinstance(data, dict):',
        '    metrics = data.setdefault("metrics", {})',
        '    debug = metrics.setdefault("debug", {}) if isinstance(metrics, dict) else {}',
        '    if isinstance(debug, dict):',
        '        debug["silent_monitor_import_source"] = monitor_import_source',
        '        debug["silent_monitor_module_file"] = getattr(monitor_module, "__file__", None)',
        '        text = json.dumps(data, ensure_ascii=False)',
        'print(text)',
    ].join('\n');

    let stdout = '';
    let lastError: unknown = null;
    const triedExecutables: string[] = [];

    for (const pythonExecutable of getPythonExecutables()) {
        triedExecutables.push(pythonExecutable);
        try {
            console.log('[AgentMonitorService] trying Python executable', {
                pythonExecutable,
                agentRoot,
            });
            const result = await execFileAsync(pythonExecutable, ['-c', script, payload], {
                timeout: 90000,
                maxBuffer: 10 * 1024 * 1024,
                cwd: agentRoot,
                encoding: 'utf8',
                env: {
                    ...process.env,
                    PYTHONPATH: (process.env.PYTHONPATH || '')
                        .split(path.delimiter)
                        .filter((item) => item && path.resolve(item) !== path.resolve(agentPackageRoot))
                        .join(path.delimiter),
                },
            });
            stdout = result.stdout;
            if (result.stderr?.trim()) {
                console.log('[AgentMonitorService] monitor_training stderr', result.stderr.trim());
            }
            console.log('[AgentMonitorService] monitor_training subprocess completed', {
                pythonExecutable,
                stdoutLength: stdout.length,
            });
            lastError = null;
            break;
        } catch (error: any) {
            lastError = error;
            if (error?.code !== 'ENOENT') {
                throw error;
            }
        }
    }

    if (lastError) {
        throw new Error(
            `No Python executable found for silent training monitor. Tried: ${triedExecutables.join(', ')}. ` +
                `Agent root: ${agentRoot}. Set MEDFLOW_PYTHON_EXECUTABLE to the Python used by the agent runtime.`,
        );
    }

    const text = stdout
        .split(/\r?\n/)
        .map((line) => line.trim())
        .filter(Boolean)
        .reverse()
        .find((line) => line.startsWith('{')) || '';
    if (!text) {
        throw new Error('Agent monitor returned empty output');
    }

    const data = JSON.parse(text) as Record<string, unknown>;
    const metrics = (data.metrics && typeof data.metrics === 'object' ? data.metrics : {}) as Record<string, unknown>;
    const debug = (metrics.debug && typeof metrics.debug === 'object' ? metrics.debug : {}) as Record<string, unknown>;
    const hasReturnedMetrics =
        metrics.wandb_run_dir != null ||
        metrics.latest_loss != null ||
        (Array.isArray(metrics.history) && metrics.history.length > 0);
    const fallbackDecision = debug.silent_monitor_pid_fallback_used
        ? 'used'
        : debug.silent_monitor_pid_fallback_attempted
          ? 'attempted_no_metrics'
          : 'not_triggered';
    const fallbackNotTriggeredReason = fallbackDecision === 'not_triggered'
        ? !requestPayload.pid
            ? 'no_pid'
            : hasReturnedMetrics
              ? 'has_metrics'
              : `select_reason:${String(metrics.wandb_select_reason ?? 'missing')}`
        : undefined;
    console.log('[AgentMonitorService] monitor_training fallback decision', {
        fallbackDecision,
        fallbackNotTriggeredReason,
        status: data.status,
        wandbSelectReason: metrics.wandb_select_reason,
        wandbUrlPending: metrics.wandb_url_pending,
        wandbRunDir: metrics.wandb_run_dir,
        latestLoss: metrics.latest_loss,
        historyCount: metrics.history_count,
        errorReason: metrics.error_reason,
    });
    console.log('[AgentMonitorService] monitor_training result', {
        status: data.status,
        container: metrics.container_name,
        pid: metrics.pid,
        pidAlive: metrics.pid_alive,
        pidSource: metrics.pid_source,
        outputDir: metrics.output_dir,
        outputDirSource: metrics.output_dir_source,
        wandbRoot: metrics.wandb_root,
        wandbRootSource: metrics.wandb_root_source,
        wandbRunDir: metrics.wandb_run_dir,
        wandbRunName: metrics.wandb_run_name,
        wandbSelectReason: metrics.wandb_select_reason,
        wandbUrl: metrics.wandb_url,
        trainingLogPath: metrics.training_log_path,
        metricsLogPath: metrics.metrics_log_path,
        latestLoss: metrics.latest_loss,
        latestStep: metrics.latest_step,
        currentStep: metrics.current_step,
        historyCount: metrics.history_count,
        lossSource: metrics.loss_source,
        errorReason: metrics.error_reason,
        selectedCmd: debug.selected_cmd,
        processCount: debug.process_count,
        outputLogLatest: debug.output_log_latest,
        monitorCodeMarker: debug.monitor_code_marker,
        importSource: debug.silent_monitor_import_source,
        moduleFile: debug.silent_monitor_module_file,
        pidFallbackUsed: debug.silent_monitor_pid_fallback_used,
        pidFallbackAttempted: debug.silent_monitor_pid_fallback_attempted,
        originalPid: debug.silent_monitor_original_pid,
        originalWandbSelectReason: debug.silent_monitor_original_wandb_select_reason,
        fallbackStatus: debug.silent_monitor_fallback_status,
        fallbackWandbSelectReason: debug.silent_monitor_fallback_wandb_select_reason,
        fallbackWandbRunDir: debug.silent_monitor_fallback_wandb_run_dir,
        fallbackLatestLoss: debug.silent_monitor_fallback_latest_loss,
        fallbackHistoryCount: debug.silent_monitor_fallback_history_count,
        llmCalled: debug.llm_called,
        llmError: debug.llm_error,
    });

    return data;
}

export async function queryAgentWorkflowStatus(params: AgentWorkflowStatusParams): Promise<Record<string, unknown>> {
    const agentRoot = findAgentRoot();
    if (!existsSync(agentRoot)) {
        throw new Error(
            `Agent root not found for workflow status query: ${agentRoot}. ` +
                'Set MEDFLOW_AGENT_ROOT to the agent directory used by the runtime.',
        );
    }

    // This endpoint intentionally reads the workflow DB only. Runtime side effects
    // such as benchmark reconciliation and inference-service shutdown are owned by
    // the Agent workflow manager, not Studio's auto-refresh loop.
    const payload = JSON.stringify({
        workflow_id: params.workflowId,
        db_path: process.env.AGENT3_WORKFLOW_DB_PATH?.trim() || './data/workflows/workflows.db',
    });
    const script = [
        'import json, os, sqlite3, sys',
        'args = json.loads(sys.argv[1])',
        'db_path = os.path.abspath(args["db_path"])',
        'connection = sqlite3.connect(db_path, timeout=30)',
        'connection.row_factory = sqlite3.Row',
        'row = connection.execute("SELECT * FROM workflows WHERE workflow_id=?", (args["workflow_id"],)).fetchone()',
        'connection.close()',
        'if row is None:',
        '    raise SystemExit("workflow_not_found")',
        'workflow = dict(row)',
        'stages = json.loads(workflow["stages_json"])',
        'context = json.loads(workflow["context_json"])',
        'metric_keys = {"container_name", "train_type", "output_dir", "latest_loss", "latest_epoch", "latest_step", "progress_percent", "current_step", "total_steps", "elapsed_time", "remaining_time", "pid_alive", "training_process_exists", "error_reason"}',
        'public_stages = {}',
        'for name, stage in stages.items():',
        '    public = {key: value for key, value in stage.items() if key not in {"metrics", "debug", "history"}}',
        '    metrics = stage.get("metrics")',
        '    if isinstance(metrics, dict):',
        '        public_metrics = {key: value for key, value in metrics.items() if key in metric_keys and value is not None}',
        '        if public_metrics:',
        '            public["metrics"] = public_metrics',
        '    public_stages[name] = public',
        'train_type = context.get("train_type")',
        'train_type_text = {"lora": "lora批量训练", "full": "全参批量训练", "enhanced": "增强训练", "grpo": "grpo训练"}.get(train_type, train_type)',
        'benchmark_entry = context.get("benchmark") or "2024"',
        'protocol = {',
        '    "version": "1.0",',
        '    "type": "workflow_status",',
        '    "agent": "workflow_monitor",',
        '    "workflowId": workflow["workflow_id"],',
        '    "workflowStatus": workflow["status"],',
        '    "workflowUpdatedAt": workflow.get("updated_at"),',
        '    "workflowDbPath": db_path,',
        '    "currentStage": workflow["current_stage"],',
        '    "datasetRef": workflow["dataset_ref"],',
        '    "trainType": train_type,',
        '    "trainTypeText": train_type_text,',
        '    "stages": public_stages,',
        '    "benchmark": context.get("benchmark"),',
        '    "evaluationDatasetName": context.get("evaluation_dataset_name"),',
        '    "benchmarkResultEntry": ("查看2024基准评测结果" if benchmark_entry in ("2024", "2024.json") else "查看推理基准测试" + str(benchmark_entry) + "结果"),',
        '    "error": workflow.get("error"),',
        '    "source": "workflow_db",',
        '    "confidence": 1.0,',
        '    "valid": True,',
        '}',
        'print(json.dumps(protocol, ensure_ascii=False))',
    ].join('\n');

    let stdout = '';
    let lastError: unknown = null;
    const triedExecutables: string[] = [];

    for (const pythonExecutable of getPythonExecutables()) {
        triedExecutables.push(pythonExecutable);
        try {
            const result = await execFileAsync(pythonExecutable, ['-c', script, payload], {
                timeout: 10000,
                maxBuffer: 1024 * 1024,
                cwd: agentRoot,
                encoding: 'utf8',
                env: process.env,
            });
            stdout = result.stdout;
            lastError = null;
            break;
        } catch (error: any) {
            lastError = error;
            if (error?.code !== 'ENOENT') {
                throw error;
            }
        }
    }

    if (lastError) {
        throw new Error(
            `No Python executable found for workflow status query. Tried: ${triedExecutables.join(', ')}.`,
        );
    }

    const text = stdout
        .split(/\r?\n/)
        .map((line) => line.trim())
        .filter(Boolean)
        .reverse()
        .find((line) => line.startsWith('{')) || '';
    if (!text) {
        throw new Error('Workflow status query returned empty output');
    }

    const protocol = JSON.parse(text) as Record<string, unknown>;
    console.info('[workflow-debug] studio workflow status query', {
        workflowId: protocol.workflowId,
        workflowStatus: protocol.workflowStatus,
        currentStage: protocol.currentStage,
        workflowUpdatedAt: protocol.workflowUpdatedAt,
        workflowDbPath: protocol.workflowDbPath,
    });
    return protocol;
}

