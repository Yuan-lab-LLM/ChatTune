"""Prepare isolated execution paths and reconcile durable container journals."""
from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
import subprocess
import json
import time

from .experiment_capture import container_json, container_python, container_asset, asset_registration
from .experiment_tracker import canonical, redact, TERMINAL
from .dataset_naming import build_dataset_execution_name


def prepare_container_execution(run_kwargs, *, handle, actor_user_id, script_name, assets):
    container, attempt = run_kwargs['docker_container'], handle['attemptId']
    inspected = subprocess.run(['docker', 'inspect', '--format', '{{json .Image}}', container],
                               capture_output=True, text=True, timeout=15)
    image_id = json.loads(inspected.stdout) if inspected.returncode == 0 else None
    directory = f'/home/workspace/medflow/users/{actor_user_id}/experiments/{attempt}'
    env = dict(run_kwargs.get('env_vars') or {})
    args = dict(run_kwargs.get('script_args') or {})
    aliases, outputs, reports = {}, [], []
    dataset_output = None
    checkpoint_input = None
    if script_name in {'data_preprocessing', 'score_based_filtering'}:
        requested_output = str(args.get('output_folder') or '').rstrip('/')
        if not requested_output.startswith('/') or '..' in PurePosixPath(requested_output).parts:
            raise ValueError('Preprocessing requires an explicit absolute output directory')
        # Keep the public/private parent and give each attempt a compact product name.
        requested_path = PurePosixPath(requested_output)
        actual_output = str(requested_path.with_name(
            build_dataset_execution_name(requested_path.name, attempt)))
        args['output_folder'] = actual_output
        run_kwargs['experiment_output_assets'] = [{'path': actual_output, 'kind': 'dataset', 'role': 'processed_dataset'}]
        seed = next((asset for asset in assets if asset['role'] == 'preprocessing_resume'), None)
        dataset_output = {'path': actual_output, 'seed': seed}
        hint = dict(run_kwargs.get('protocol_hint_extra') or {})
        hint.update(outputFolder=actual_output, outputDatasetName=PurePosixPath(actual_output).name)
        run_kwargs['protocol_hint_extra'] = hint
    batch_prefixes = {'batch_train_lora': 'model_lora_', 'batch_train_full': 'model_full_',
                      'batch_train_pretrain_lora': 'model_pretrain_lora_', 'batch_train_pretrain_full': 'model_pretrain_full_'}
    if script_name in batch_prefixes:
        source = env.get('DATASET_DIR')
        if not source:
            raise ValueError('Batch training requires a resolved, versioned dataset directory')
        alias = f'data-{attempt}'
        aliases[alias] = source
        env['DATASET_DIR'] = f'{directory}/inputs/{alias}'
        model_root = 'pretrain' if script_name.startswith('batch_train_pretrain_') else 'batch_train'
        output = f'/home/workspace/models/{model_root}/{batch_prefixes[script_name]}{alias}'
        outputs.extend([output, f'{output}_merged'])
    if script_name.startswith('train_multinode_') or 'output-dir' in args:
        # A user-specified directory becomes the parent; each attempt gets a distinct child.
        args['output-dir'] = f"{str(args.get('output-dir') or directory).rstrip('/')}/{attempt}"
        outputs.append(args['output-dir'])
        if script_name in {'train_multinode_sft_pipeline', 'train_multinode_sft_full_pipeline'}:
            outputs.append(f"{args['output-dir']}_merged")
    if script_name in {'train_multinode_sft_pipeline', 'train_multinode_sft_full_pipeline',
                       'train_multinode_dpo_pipeline'}:
        dpo = script_name == 'train_multinode_dpo_pipeline'
        args['log-dir'] = '/home/workspace/log/dpo_train' if dpo else '/home/workspace/log/batch_train'
        args['log-file'] = f'{attempt}-dpo-train.log' if dpo else f'{attempt}-train-log.log'
        if dpo:
            args['run-id'] = attempt
    if script_name == 'dpo_train_launcher' or 'output_dir' in args:
        args['output_dir'] = f"{str(args.get('output_dir') or directory).rstrip('/')}/{attempt}"
        outputs.append(args['output_dir'])
    if script_name == 'train_multinode_dpo_pipeline' and not args.get('export-dir'):
        args['export-dir'] = '/home/workspace/models/dpo_train/internal/export'
    for key in ('export-dir', 'export_dir'):
        if args.get(key):
            args[key] = f"{str(args[key]).rstrip('/')}/{attempt}"
            outputs.append(args[key])
    positional = list(run_kwargs.get('positional_args') or [])
    if 'grpo' in script_name:
        positional = [arg for arg in positional if not str(arg).startswith('trainer.default_local_dir=')]
        positional.append(f'trainer.default_local_dir={directory}/output')
        outputs.append(f'{directory}/output')
        for asset in assets:
            if asset['role'] == 'resume_checkpoint' and asset.get('metadata', {}).get('resumeDirectoryName'):
                alias = asset['metadata']['resumeDirectoryName']
                aliases[alias] = asset['location']['path']
                positional = [arg for arg in positional if not str(arg).startswith(
                    ('trainer.resume_from_path=', 'trainer.del_local_ckpt_after_load='))]
                positional.extend([f'trainer.resume_from_path={directory}/inputs/{alias}',
                                   'trainer.del_local_ckpt_after_load=False'])
    if script_name == 'ckpt_eval':
        checkpoint = next((asset for asset in assets if asset['kind'] == 'checkpoint' and
                           asset['role'] == 'evaluated_model'), None)
        if not checkpoint:
            raise ValueError('Medical checkpoint evaluation requires a recorded checkpoint input')
        base = next((asset for asset in assets if asset['kind'] == 'model' and asset['role'] == 'base_model'), None)
        if not base:
            raise ValueError('Medical checkpoint evaluation requires a recorded base model input')
        # The installed legacy exporter requires a check* basename and derives
        # its export directory from the checkpoint parent/name. Keep both in
        # this attempt while preserving the content-addressed source snapshot.
        alias = f'checkpoint-{attempt}'
        env['CKPT_PATH'] = f'{directory}/inputs/{alias}'
        checkpoint_input = {'path': env['CKPT_PATH'], 'seed': checkpoint, 'baseModel': base,
                            'baseModelPath': base['location']['path']}
        for key in ('CKPT_PATH', 'ckpt-path'):
            if key in args:
                args[key] = env['CKPT_PATH']
        outputs.append(f'{directory}/inputs-{alias}_merged')
    if script_name.startswith('generic_'):
        args.pop('run-id', None)
        args['run_id'] = attempt
        result_root = f'/home/workspace/eval/llamafactory/{attempt}'
        reports = [f'{result_root}/{name}' for name in ('manifest.json', 'summary.json', 'summary.txt')]
        for asset in assets:
            if asset['role'] == 'evaluation_dataset':
                aliases[asset['metadata']['datasetName']] = asset['location']['path']
        env['MEDFLOW_EVAL_TASK_DIR'] = f'{directory}/inputs'
        if image_id:
            env['MEDFLOW_EVAL_IMAGE_ID'] = image_id
        modes = {'generic_single_model_eval': 'single_model', 'generic_compare_models_eval': 'compare_between_models',
                 'generic_ckpt_eval': 'ckpt'}
        args['mode'] = modes[script_name]
        if script_name == 'generic_ckpt_eval':
            outputs.append(f'{result_root}/merged_checkpoint')
            env['MEDFLOW_EVAL_MODEL_VERSION_ROOT'] = f'/home/workspace/medflow/users/{actor_user_id}/experiment_assets/model'
    metadata = {'scriptName': script_name, 'outputPaths': outputs, 'reportPaths': reports,
                'workingDirectory': run_kwargs.get('docker_working_dir'),
                'modelVersionRoot': env.get('MEDFLOW_EVAL_MODEL_VERSION_ROOT'),
                'inputAssets': assets, 'declaredOutputs': run_kwargs.get('experiment_output_assets') or []}
    if dataset_output:
        metadata['datasetOutputPath'] = dataset_output['path']
    if script_name == 'grpo_train':
        metadata['businessStdout'] = {'label': '训练日志', 'kind': 'training'}
    elif script_name in {'data_preprocessing', 'score_based_filtering'}:
        metadata['businessStdout'] = {'label': '数据处理日志', 'kind': 'data_processing'}
    installed = container_json(container, 'experiment_process.py', {
        'ownerId': actor_user_id, 'attemptId': attempt, 'metadata': metadata, 'aliases': aliases,
        'runnerSource': Path(__file__).with_name('experiment_process.py').read_text(encoding='utf-8'),
        'evaluationSource': Path(__file__).with_name('experiment_evaluation.py').read_text(encoding='utf-8') if script_name.startswith('generic_') else None,
        'scoringCompatSource': Path(__file__).with_name('scoring_api_compat.py').read_text(encoding='utf-8')
            if script_name == 'score_based_filtering' else None,
        'allocationFile': args.get('allocation-file'),
        'datasetOutput': dataset_output,
        'checkpointInput': checkpoint_input,
        'tokenizerCompatSource': Path(__file__).with_name('vllm_tokenizer_compat.py').read_text(encoding='utf-8')
            if script_name in {'single_model_evaluation_vpn', 'compare_between_models_vpn', 'ckpt_eval'} else None,
        'assetSource': Path(__file__).with_name('experiment_assets.py').read_text(encoding='utf-8') if args.get('allocation-file') or dataset_output or checkpoint_input or script_name.startswith('generic_') else None,
    }, 'setup', timeout=7200)
    compatibility = installed.get('vllmTokenizerCompat')
    if compatibility:
        env['PATH'] = compatibility['binDirectory'] + ':' + (env.get('PATH') or compatibility['containerPath'])
        env['MEDFLOW_VLLM_TOKENIZER_COMPAT'] = '1'
    executable = container_python(container)
    if script_name.startswith('generic_'):
        run_kwargs.update(script_path=f'{directory}/evaluation.py', docker_executable=executable)
    run_kwargs.update(env_vars=env, script_args=args, positional_args=positional,
                     experiment_runner=[executable, installed['runner'], 'run', directory])
    # The launcher must never append another attempt's output to a shared log.
    prior_log = run_kwargs.get('log_file')
    run_kwargs['log_file'] = f'{directory}/execution.log'
    if prior_log and run_kwargs.get('shell_prefix'):
        run_kwargs['shell_prefix'] = run_kwargs['shell_prefix'].replace(prior_log, run_kwargs['log_file'])
    return {'experimentDirectory': directory, 'executionLogPath': run_kwargs['log_file'],
            'outputPaths': outputs, 'reportPaths': reports,
            'inputAssets': assets, 'inputTransforms': installed.get('inputTransforms', []),
            'replicas': installed.get('replicas', []), 'containerImageId': image_id, 'archiveStatus': 'pending',
            'runtimeCompatibility': ([compatibility] if compatibility else [])
                + ([installed['scoringApiCompat']] if installed.get('scoringApiCompat') else [])}


def observe_execution_status(tracker, execution, *, probe_fn=None):
    """Refresh execution state from its own process journal, without archiving assets."""
    runtime = execution['runtime']
    directory = runtime.get('experimentDirectory')
    if not directory:
        return
    probe_fn = probe_fn or (lambda container, path: container_json(container, 'experiment_process.py', {'directory': path,
        'hints': {key: runtime[key] for key in ('output_dir', 'report_path') if runtime.get(key)}}, 'probe'))
    result = probe_fn(execution['container'], directory)
    attempt = execution['attempt_id']
    if result.get('attemptId') != attempt:
        raise ValueError('Container journal execution identity mismatch')
    state, process = result['state'], result.get('process') or {}
    # A cancelled job need not produce a model/report. Still require the
    # supervisor's exit receipt; a request alone never proves termination.
    current = tracker.execution(attempt)
    if current['runtime'].get('cancelRequested') and process.get('exitCode') is not None:
        state = 'cancelled' if current['state'] not in TERMINAL else current['state']
    if state == 'succeeded' and runtime.get('kind') in {'train', 'evaluate', 'preprocess'}:
        expected = result.get('reports') if runtime['kind'] == 'evaluate' else result.get('outputs')
        empty_filter = (runtime['kind'] == 'preprocess' and runtime.get('scriptName') == 'score_based_filtering'
                        and (result.get('dataProcessing') or {}).get('outcome') == 'no_qualifying_data')
        if not expected and not empty_filter:
            state = 'unknown'  # A launcher exiting zero alone does not prove the requested job produced its result.
    observation = {**runtime, 'pid': str(process.get('pid') or runtime.get('pid') or ''),
                   'processIdentity': process.get('identity'), 'exitCode': process.get('exitCode'),
                   'signals': process.get('signals') or []}
    if result.get('dataProcessing'):
        observation['dataProcessing'] = result['dataProcessing']
    error = process.get('error') or f"Process exited with code {process.get('exitCode')}"
    if (result.get('dataProcessing') or {}).get('outcome') == 'scoring_failed':
        counts = result['dataProcessing']
        error = f"Filtering failed: {counts['api_failed_items']} of {counts['total_items']} records were reported as scoring failures (api_failed_items)"
    # For training, a zero supervisor exit and a recognized output establish
    # completion even while the potentially long model archive is pending.
    # Evaluation and preprocessing retain their existing archive gate.
    interim_state = 'unknown' if state == 'succeeded' and runtime.get('kind') != 'train' else state
    # Terminal states cannot be revised by an intermittent probe failure.
    if execution['state'] not in TERMINAL and (interim_state != execution['state'] or observation != runtime):
        tracker.emit(attempt, 'status', {'status': interim_state, 'runtime': observation,
            **({'error': error} if state == 'failed' else {})}, preserve_terminal=True)
    return result, state


def collect_execution(tracker, execution, *, probe_fn=None, capture_fn=container_asset):
    runtime = execution['runtime']
    if not runtime.get('experimentDirectory') or runtime.get('archiveStatus') == 'complete':
        return
    result, state = observe_execution_status(tracker, execution, probe_fn=probe_fn)
    attempt = execution['attempt_id']
    process = result.get('process') or {}
    process_finished = result['state'] in TERMINAL or (state == 'cancelled' and process.get('exitCode') is not None)
    if not process_finished:
        pid = str(process.get('pid') or runtime.get('pid') or '')
        if state == 'running' and pid and runtime.get('reservationId'):
            # Recover the existing resource heartbeat after a lost launch response or Runtime restart.
            from types import SimpleNamespace
            from medflow_agent_tools.runlocal_train import _start_resource_reservation_heartbeat
            _start_resource_reservation_heartbeat(runtime['reservationId'], {'MEDFLOW_RESOURCE_NODE_ID': tracker.node_id},
                SimpleNamespace(metadata={'protocol_hint': {'container': execution['container'], 'pid': pid}}))
        return
    if runtime.get('reservationId') and not runtime.get('reservationReleased'):
        # A short job may finish before its launch response starts a heartbeat.
        # Release only with authoritative terminal journal evidence, and retry
        # after disconnection before marking reconciliation complete.
        tracker.send('releaseTrainingResourcesForRuntime', {
            'runtimeNodeId': tracker.node_id, 'reservationId': runtime['reservationId']})
        tracker.emit(attempt, 'status', {'status': tracker.execution(attempt)['state'],
            'runtime': {'reservationReleased': True}}, preserve_terminal=True)
    if state == 'unknown':
        pending = {'archiveStatus': 'awaiting_outputs', 'archiveWarning':
            'Process exited without a recognized dataset or completed filtering receipt' if runtime.get('kind') == 'preprocess'
            else 'Process exited without a recognized model or report'}
        if result.get('unattributedReports'):
            pending.update(unattributedReportPaths=result['unattributedReports'],
                archiveWarning='Discovered reports are not bound to this execution; shared reports cannot confirm success')
        if any(tracker.execution(attempt)['runtime'].get(key) != value for key, value in pending.items()):
            tracker.emit(attempt, 'status', {'status': tracker.execution(attempt)['state'], 'runtime': pending})
        return
    prior_check = tracker.event_payload(attempt, 'input-verification')
    checks = prior_check['runtime']['inputVerification'] if prior_check else []
    if not prior_check:
        verification_inputs = list(runtime.get('inputAssets', []))
        verification_inputs.extend({'id': item['checkpointVersionId'], 'kind': 'checkpoint',
            'location': {'path': item['executionPath']}, 'fingerprint': item['executionFingerprint']}
            for item in runtime.get('inputTransforms', []))
        for asset in verification_inputs:
            try:
                captured = capture_fn(execution['container'], asset['location']['path'], execution['actor_user_id'],
                                      asset['kind'], operation='manifest')
                checks.append({'assetVersionId': asset['id'], 'matches': captured['fingerprint'] == asset['fingerprint'],
                               'path': asset['location']['path'], 'expected': asset['fingerprint'], 'observed': captured['fingerprint']})
            except Exception as exc:
                checks.append({'assetVersionId': asset['id'], 'matches': False, 'error': str(exc)[:1000]})
    input_verified = bool(checks) and all(check['matches'] for check in checks)
    if not prior_check:
        tracker.emit_once(attempt, 'status', {'status': tracker.execution(attempt)['state'], 'runtime': {
            'inputIntegrityAfterExecution': 'verified' if input_verified else 'unverified', 'inputVerification': checks}}, 'input-verification')
    config = tracker.latest_payload(attempt, 'configuration') or {'configuration': {}}
    configuration = {**config['configuration'], 'actual': {
        'observedCommands': process.get('commands', []), 'configurationFiles': result.get('configurationFiles', {}),
        'entrypointFingerprints': process.get('entrypointFingerprints', {}),
        'inputTransforms': runtime.get('inputTransforms', []),
        'phaseExecutions': (result.get('evaluationManifest') or {}).get('executions', []),
        'source': 'container_process_journal_and_adapter_manifest' if (result.get('evaluationManifest') or {}).get('executions')
                  else 'container_process_journal'}, 'completeness': 'incomplete',
        'missing': ['trainer_internal_defaults_and_random_state']}
    if not tracker.event_payload(attempt, 'final-configuration'):
        tracker.emit_once(attempt, 'configuration', {'configuration': configuration,
            'environment': {**config.get('environment', {}), 'observerEnvironment': {'python': process.get('python'),
                            'platform': process.get('platform'), 'packages': process.get('packages', {})},
                            'containerImageId': runtime.get('containerImageId')}}, 'final-configuration')
    registered = []
    descriptors = list(result.get('outputs') or []) + [
        {'path': path, 'kind': 'report',
         'role': 'data_processing_report' if runtime.get('kind') == 'preprocess' else 'evaluation_report'}
        for path in result.get('reports', [])]
    for descriptor in descriptors:
        key = f"output:{descriptor['kind']}:{descriptor['path']}"
        previous = tracker.event_payload(attempt, key)
        if previous:
            registered.append(previous)
            continue
        captured = capture_fn(execution['container'], descriptor['path'], execution['actor_user_id'], descriptor['kind'])
        if descriptor.get('fingerprint') and captured['fingerprint'] != descriptor['fingerprint']:
            raise ValueError('Exported model snapshot no longer matches the model used for evaluation')
        registration = asset_registration(captured, node_id=tracker.node_id, container=execution['container'],
            owner_id=execution['actor_user_id'], source=descriptor.get('sourcePath') or descriptor['path'], kind=descriptor['kind'],
            role=descriptor['role'], direction='output')
        sources = runtime.get('inputAssets', [])
        if descriptor.get('inputPaths'):
            wanted = set(descriptor['inputPaths'])
            sources = [asset for asset in sources if asset['location']['path'] in wanted]
            if {asset['location']['path'] for asset in sources} != wanted:
                raise ValueError('Export receipt references an unrecorded input model')
        registration['metadata']['inputVersionIds'] = sorted({asset['id'] for asset in sources})
        tracker.emit_once(attempt, 'artifact', registration, key)
        registered.append(registration)
    manifest = result.get('evaluationManifest')
    if manifest:
        register_evaluation(tracker, execution, {**manifest,
            'experimentProtocolComplete': bool(manifest.get('experimentProtocolComplete')) and input_verified}, registered)
    tracker.emit(attempt, 'status', {'status': state, 'runtime': {
        'archiveStatus': 'awaiting_outputs' if state == 'unknown' else 'complete', 'archivedAt': time.time(),
        'archiveWarning': None if descriptors else 'No recognized model or report files were produced',
        'archiveError': None}}, preserve_terminal=True)


def register_evaluation(tracker, execution, manifest, outputs):
    """Legacy reports lack sample counts/protocol versions; preserve that uncertainty."""
    attempt = execution['attempt_id']
    inputs = execution['runtime'].get('inputAssets', [])
    report = next((asset for asset in outputs if asset['kind'] == 'report' and asset['logicalName'] == 'manifest.json'), None)
    model_roles = {'model_fir': 'evaluated_model', 'model_sec': 'baseline_model', 'checkpoint': 'evaluated_model', 'base': 'base_model'}
    datasets = {asset['metadata'].get('datasetName'): asset for asset in inputs if asset['role'] == 'evaluation_dataset'}
    for role, scores in (manifest.get('results') or {}).items():
        model = next((asset for asset in inputs if asset['role'] == model_roles.get(role)), None)
        if role == 'checkpoint':
            # The scored weights are the exported model; the checkpoint remains its input lineage.
            model = next((asset for asset in outputs if asset['kind'] == 'model' and asset['logicalName'] == 'merged_checkpoint'), None)
        if not model:
            continue
        metrics = []
        for dataset_name, values in scores.items():
            dataset = datasets.get(dataset_name)
            if not dataset:
                continue  # Never fabricate a dataset version from its name.
            for subset, value in [('all', values.get('average')), *(values.get('subjectScores') or {}).items()]:
                if not isinstance(value, (int, float)):
                    continue
                metrics.append({'datasetVersionId': dataset['id'], 'subset': subset, 'name': 'accuracy',
                    'value': value, 'sampleCount': values.get('sampleCount') if subset == 'all' else (values.get('subjectSampleCounts') or {}).get(subset),
                    'unit': 'percent', 'direction': 'higher'})
        if not metrics:
            continue
        result_id = 'eval-' + hashlib.sha256(canonical([attempt, role]).encode()).hexdigest()
        protocol = manifest.get('experimentProtocol') or {
            'evaluationScope': manifest.get('evaluationScope'), 'datasets': manifest.get('datasets'),
            'aggregation': manifest.get('aggregation'), 'completeness': 'incomplete',
            'missing': ['evaluator_version', 'seed', 'sample_counts', 'actual_dataset_snapshot_binding']}
        model_bound = role != 'checkpoint' or any(
            item.get('path') == model.get('location', {}).get('path') and item.get('fingerprint') == model.get('fingerprint')
            for item in manifest.get('modelArtifacts', []))
        if not model_bound:
            protocol = {**protocol, 'missing': [*protocol.get('missing', []), 'merged_model_snapshot_binding']}
        counts = [values.get('sampleCount') for values in scores.values()]
        completed = sum(counts) if counts and all(isinstance(count, int) and count >= 0 for count in counts) else None
        finished = manifest.get('status') == 'finished' and set(scores) == set(manifest.get('datasets') or [])
        payload = {'id': result_id, 'modelVersionId': model['id'], 'protocol': protocol,
            'protocolComplete': bool(manifest.get('experimentProtocolComplete')) and model_bound,
            'expectedCount': completed if finished else None, 'succeededCount': completed,
            'failedCount': 0 if finished and completed is not None else None, 'metrics': metrics}
        if report:
            payload['reportVersionId'] = report['id']
        tracker.emit_once(attempt, 'evaluation', payload, f'evaluation:{role}')


def reconcile_all(tracker):
    for execution in tracker.collectable_executions():
        try:
            if execution['runtime'].get('operation') == 'dataset_upload':
                from .experiment_upload import collect_dataset_upload
                collect_dataset_upload(tracker, execution)
            elif execution['runtime'].get('operation') == 'model_publish':
                from .experiment_publish import collect_publish
                collect_publish(tracker, execution)
            else:
                collect_execution(tracker, execution)
        except Exception as exc:
            message = str(redact(str(exc)))[:2000]
            if execution['runtime'].get('archiveError') != message:
                tracker.emit(execution['attempt_id'], 'status', {'status': tracker.execution(execution['attempt_id'])['state'],
                    'runtime': {'archiveStatus': 'retrying', 'archiveError': message}})
