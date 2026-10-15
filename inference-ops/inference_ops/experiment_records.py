"""Native worker adapter for the shared durable experiment protocol.

No inference dependencies are imported here. The Agent and inference worker use
the same record/outbox code, but each worker owns its local SQLite journal.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import types
import time
from urllib.request import Request, build_opener, ProxyHandler

_trackers = {}
_lock = threading.Lock()
BACKEND = 'inference-worker'


def shared(name):
    package = '_medflow_experiment_runtime'
    if package not in sys.modules:
        directory = Path(os.getenv('MEDFLOW_EXPERIMENT_UTILS_DIR') or Path(__file__).resolve().parent / 'experiment_runtime')
        if not (directory / 'experiment_tracker.py').is_file():
            raise ValueError('Experiment runtime is unavailable; deploy inference_ops/experiment_runtime or configure MEDFLOW_EXPERIMENT_UTILS_DIR')
        module = types.ModuleType(package)
        module.__path__ = [str(directory)]
        sys.modules[package] = module
    return importlib.import_module(f'{package}.{name}')


def identity(value):
    value = str(value or '')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', value):
        raise ValueError('Experiment records require an authenticated owner and Runtime node')
    return value


def actor_user_id(owner):
    """Translate the inference ownership key to Studio's user ID.

    Instance ownership keeps its auth: namespace; Studio experiment APIs and
    asset directories use the underlying user ID. Never infer it from aliases
    or conversation IDs.
    """
    value = str(owner or '')
    if value.startswith('auth:'):
        value = value[len('auth:'):]
    return identity(value)


def experiment_configuration_error():
    """Check worker credentials without creating journals or starting threads."""
    try:
        shared('experiment_tracker').studio_connection()
    except ValueError as exc:
        return str(exc)
    node_id = shared('runtime_credentials').current_node_id()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', node_id):
        return '实验记录需要有效的 MEDFLOW_RESOURCE_NODE_ID（当前推理 worker 的 Runtime 节点 ID）'
    return ''


def get_tracker(state_dir, node_id):
    node_id = identity(node_id)
    path = Path(state_dir).resolve() / 'experiments' / node_id / 'records.sqlite3'
    with _lock:
        if str(path) not in _trackers:
            tracker = shared('experiment_tracker').ExperimentTracker(path, node_id, collector_fn=reconcile_worker)
            tracker.start()
            _trackers[str(path)] = tracker
        return _trackers[str(path)]


def resume_workers(state_dir):
    for database in (Path(state_dir) / 'experiments').glob('*/records.sqlite3'):
        get_tracker(state_dir, database.parent.name)


def close_workers():
    with _lock:
        for tracker in _trackers.values():
            tracker.close()
        _trackers.clear()


def local_asset(state_dir, owner_id, node_id, source, kind, role, direction='input'):
    owner_id, node_id = actor_user_id(owner_id), identity(node_id)
    source = Path(source).resolve()
    if kind == 'dataset':
        shared('experiment_assets').validate_dataset_references(source)
    root = Path(state_dir).resolve() / 'experiment_assets' / owner_id / kind
    captured = shared('experiment_assets').snapshot(source, root)
    registration = shared('experiment_capture').asset_registration(captured, node_id=node_id, container=BACKEND,
        owner_id=owner_id, source=str(source), kind=kind, role=role, direction=direction)
    registration['metadata']['executionBackend'] = 'inference_worker'
    argument = str(Path(captured['path']) / source.name) if captured['sourceIsFile'] else captured['path']
    return registration, argument


def freeze_service_model(cfg, *, state_dir, owner_id, node_id, reservation_id=None):
    source = Path(cfg['ENV']['MODEL_PATH']) / cfg['ENV']['MODEL_NAME']
    stopped, failures = threading.Event(), []
    tracker = get_tracker(state_dir, node_id) if reservation_id else None
    def renew():
        tracker.send('renewTrainingResourcesForRuntime', {'runtimeNodeId': node_id, 'reservationId': reservation_id})
    def keep_alive():
        while not stopped.wait(30):
            try:
                renew()
            except Exception as exc:
                failures.append(exc)
                return
    if tracker:
        renew()
        threading.Thread(target=keep_alive, name='inference-model-snapshot-lease', daemon=True).start()
    try:
        version, path = local_asset(state_dir, owner_id, node_id, source, 'model', 'evaluated_model')
        if failures:
            raise ValueError('GPU reservation renewal failed while freezing serving model') from failures[0]
        if tracker:
            renew()
    finally:
        stopped.set()
    # The API model name stays stable; only the filesystem source is pinned.
    cfg['ENV']['MODEL_SNAPSHOT_PATH'] = path
    return version


def observed_model(instance, cfg, node_id, owner_id, commands):
    owner_id = actor_user_id(owner_id)
    version = instance.get('experiment_model_version')
    pinned = str(cfg.get('ENV', {}).get('MODEL_SNAPSHOT_PATH') or '')
    confirmed = bool(pinned and any(pinned in command for command in commands))
    if not confirmed and isinstance(version, dict) and instance.get('experimentDirectory'):
        try:
            process = shared('experiment_process')
            receipt = process.read_json(Path(instance['experimentDirectory']) / 'readiness.json')
            confirmed = bool(receipt.get('attemptId') == (instance.get('experiment') or {}).get('attemptId') and
                receipt.get('instanceId') == instance.get('instance_id') and receipt.get('runId') == instance.get('run_id') and
                receipt.get('modelVersionId') == version.get('id') and receipt.get('modelFingerprint') == version.get('fingerprint') and
                receipt.get('processIdentity') and process.process_identity(receipt.get('pid')) == receipt['processIdentity'])
        except (OSError, ValueError):
            pass
    if isinstance(version, dict) and pinned and version.get('location', {}).get('path') == pinned and confirmed:
        if shared('experiment_assets').manifest(pinned)['fingerprint'] == version.get('fingerprint'):
            return {**version, 'role': 'evaluated_model', 'direction': 'input'}
    # A historical service may have loaded bytes that no longer match its path.
    # Keep its identity visible without falsely certifying today's disk content.
    source = str(instance.get('model_path') or Path(cfg['ENV']['MODEL_PATH']) / cfg['ENV']['MODEL_NAME'])
    key = [owner_id, node_id, instance.get('instance_id'), instance.get('run_id'), source]
    asset_id = 'av-' + hashlib.sha256(json.dumps(key).encode()).hexdigest()
    return {'id': asset_id, 'kind': 'model', 'role': 'evaluated_model', 'direction': 'input',
        'logicalName': cfg['ENV']['MODEL_NAME'], 'integrity': 'incomplete',
        'location': {'container': BACKEND, 'path': source}, 'manifest': {},
        'metadata': {'sourcePath': source, 'executionBackend': 'inference_worker', 'storage': 'external',
            'serviceInstanceId': instance.get('instance_id'), 'missing': ['loaded_model_content_version']}}


def inference_experiment_context(instance, context=None, kind=None):
    result = {key: value for key, value in (context or {}).items()
              if key in {'experimentId', 'workflowId', 'conversationRunId', 'project'}}
    reservation = (instance.get('resource') or {}).get('reservation_id')
    if reservation:
        result['resourceReservationId'] = reservation
    if kind == 'benchmark' and (context or {}).get('benchmarkTaskId'):
        result['taskId'] = context['benchmarkTaskId']
        if context.get('benchmarkExperimentId'):
            result['experimentId'] = context['benchmarkExperimentId']
    return result


def start_deployment(command, env, *, instance, cfg, state_dir, node_id, model, save_fn, context=None):
    """Record the finite deployment operation; the serving process remains independent."""
    owner_id = actor_user_id(instance['owner_user_id'])
    tracker = get_tracker(state_dir, node_id)
    process = shared('experiment_process')
    spec = {'schemaVersion': 1, 'kind': 'deploy', 'script': 'service_start', 'executionBackend': 'inference_worker',
        'serviceInstanceId': instance['instance_id'], 'resolved': {'command': command, 'serviceConfiguration': cfg},
        'inputVersions': [{'assetVersionId': model['id'], 'role': 'served_model', 'fingerprint': model['fingerprint']}]}
    request_key = 'deploy-' + instance['instance_id']
    handle = tracker.prepare(request_key=request_key, actor_user_id=owner_id,
        name='Deploy ' + str(instance['model']), kind='deploy', container=BACKEND, spec=spec,
        context=inference_experiment_context(instance, context))
    attempt = handle['attemptId']
    directory = Path(instance['log_dir']) / 'experiment'
    if not tracker.claim_launch(request_key):
        raise ValueError('Deployment already claimed; reconcile its original instance')
    try:
        instance.update(experiment=handle, experimentDirectory=str(directory), deployment_registration='pending')
        save_fn(instance)  # Discoverable before any process is requested.
        directory.mkdir(exist_ok=False)
        process.atomic_json(directory / 'execution.json', {'attemptId': attempt, 'outputPaths': []})
        process.atomic_json(directory / 'deployment.json', {'attemptId': attempt, 'instanceId': instance['instance_id'],
            'runId': instance['run_id'], 'nodeId': node_id, 'ownerId': instance['owner_user_id'], 'model': model,
            'modelName': cfg['ENV']['MODEL_NAME'], 'host': cfg['ENV']['HOST_IP'],
            'port': cfg['PORTS']['VLLM_OPENAI_PORT'], 'pidDirectory': instance['pid_dir']})
        tracker.emit(attempt, 'artifact', {**model, 'role': 'served_model', 'direction': 'input'})
        tracker.emit(attempt, 'configuration', {'configuration': {'schemaVersion': 1, 'resolved': spec['resolved'],
            'actual': None, 'completeness': 'incomplete', 'missing': ['service_readiness_receipt', 'serving_engine_effective_defaults']}})
        tracker.emit(attempt, 'status', {'status': 'starting', 'runtime': {'operation': 'service_deploy',
            'experimentDirectory': str(directory), 'executionLogPath': str(directory / 'execution.log'),
            'archiveStatus': 'pending', 'stateDirectory': str(Path(state_dir).resolve()),
            'instanceMetaPath': instance['meta_file'], 'serviceInstanceId': instance['instance_id'], 'serviceRunId': instance['run_id'],
            'inputAssets': [model]}})
    except Exception as exc:
        tracker.emit(attempt, 'status', {'status': 'failed', 'error': str(exc), 'runtime': {'archiveStatus': 'complete'}})
        instance.update(status='failed', error=str(exc), deployment_registration='failed')
        save_fn(instance)
        raise
    proc_env = {**env, 'MEDFLOW_DEPLOYMENT_DIRECTORY': str(directory), 'MEDFLOW_DEPLOYMENT_OBSERVER': str(Path(__file__).resolve()),
                'MEDFLOW_DEPLOYMENT_PYTHON': sys.executable}
    try:
        with (directory / 'execution.log').open('w', encoding='utf-8') as stream:
            child = subprocess.Popen([sys.executable, str(Path(process.__file__)), 'run', str(directory), *command],
                stdout=stream, stderr=subprocess.STDOUT, env=proc_env, start_new_session=True)
    except Exception as exc:
        tracker.emit(attempt, 'status', {'status': 'unknown', 'error': str(exc)}, preserve_terminal=True)
        instance.update(status='unknown', error=str(exc))
        save_fn(instance)
        raise
    instance['script_pid'] = child.pid
    save_fn(instance)
    tracker.emit(attempt, 'status', {'status': 'running', 'runtime': {'pid': str(child.pid)}}, preserve_terminal=True)
    return child


def record_service_stop(instance, *, state_dir, node_id, operation_id, status,
                        actor=None, trigger='manual', cause='stop', result='', release_ready=None, return_code=None):
    """Append an operation to the exact deployment; never change its outcome."""
    handle = instance.get('experiment') or {}
    if not handle.get('attemptId'):
        return False  # Historical instances cannot safely be attributed to a deployment.
    tracker = get_tracker(state_dir, node_id)
    execution = tracker.execution(handle['attemptId'])
    runtime = (execution or {}).get('runtime') or {}
    if (runtime.get('serviceInstanceId') != instance.get('instance_id') or
            runtime.get('serviceRunId') != instance.get('run_id') or
            runtime.get('operation') != 'service_deploy'):
        raise ValueError('Service stop does not match its deployment execution')
    tracker.emit_once(handle['attemptId'], 'service_operation', {
        'operationId': operation_id, 'operation': 'stop_service',
        'serviceInstanceId': instance['instance_id'], 'serviceRunId': instance['run_id'],
        'status': status, 'actorUserId': actor_user_id(actor) if actor else None, 'trigger': trigger,
        'cause': cause,
        'reservationId': (instance.get('resource') or {}).get('reservation_id') or None,
        'releaseReady': release_ready, 'result': str(result)[:8000], 'returnCode': return_code,
    }, f'service-stop-{operation_id}-{status}')
    return True


def process_parent(pid):
    try:
        fields = Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')', 1)[1].split()
        return int(fields[1]) if fields[0] not in {'Z', 'X'} else 0
    except (OSError, ValueError, IndexError):
        return 0


def deployment_owned_identity(instance, pid, *, process_module=None, parent_fn=process_parent):
    """Prove ownership using a live recorded root or its current descendants."""
    process = process_module or shared('experiment_process')
    try:
        directory = Path(instance['experimentDirectory'])
        request = process.read_json(directory / 'deployment.json')
        attempt = instance['experiment']['attemptId']
        if (request['attemptId'] != attempt or request['instanceId'] != instance['instance_id'] or
            request['runId'] != instance['run_id']):
            return None
        roots = {}
        for filename, identity_key in (('service-launch.json', 'processIdentity'), ('process.json', 'identity')):
            try:
                record = process.read_json(directory / filename)
                if record.get('attemptId') != attempt or not record.get(identity_key):
                    continue
                if filename == 'service-launch.json' and record.get('runId') != request['runId']:
                    continue
                if filename == 'process.json' and record.get('exitCode') is not None:
                    continue
                root_pid = int(record['pid'])
                if root_pid > 1 and process.process_identity(root_pid) == record[identity_key]:
                    roots[root_pid] = record[identity_key]
            except (OSError, ValueError, KeyError, TypeError):
                continue
        pid = int(pid)
        observed = process.process_identity(pid)
        if pid <= 1 or not observed or not roots:
            return None
        visited, current = set(), pid
        while current > 1 and current not in visited:
            visited.add(current)
            if current in roots:
                if process.process_identity(current) == roots[current] and process.process_identity(pid) == observed:
                    return observed
                return None
            current = parent_fn(current)
        return None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def cancel_starting_deployment(instance, *, process_module=None, signal_fn=None):
    """Signal the live supervisor so cancellation is recorded before shell cleanup."""
    process = process_module or shared('experiment_process')
    try:
        record = process.read_json(Path(instance['experimentDirectory']) / 'process.json')
        if record.get('exitCode') is not None or not record.get('identity'):
            return False  # A later service stop must not rewrite a completed deployment.
        pid = int(record['pid'])
        observed = deployment_owned_identity(instance, pid, process_module=process)
        if not observed or observed != record['identity'] or process.process_identity(pid) != observed:
            return False
        (signal_fn or os.kill)(pid, signal.SIGTERM)
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def deployment_listener_owners(pid, port):
    """Bind readiness to this process tree, not merely an occupied TCP port."""
    inodes = set()
    for table in (Path('/proc/net/tcp'), Path('/proc/net/tcp6')):
        try:
            for line in table.read_text().splitlines()[1:]:
                fields = line.split()
                if len(fields) > 9 and fields[3] == '0A' and int(fields[1].rsplit(':', 1)[1], 16) == int(port):
                    inodes.add('socket:[' + fields[9] + ']')
        except FileNotFoundError:
            pass
    visited, pending, owners = set(), [int(pid)], []
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        root = Path('/proc') / str(current)
        for children in (root / 'task').glob('*/children'):
            try:
                pending.extend(int(value) for value in children.read_text().split())
            except (OSError, ValueError):
                pass
        for descriptor in (root / 'fd').glob('*'):
            try:
                if os.readlink(descriptor) in inodes:
                    owners.append(current)
                    break
            except OSError:
                pass
    return sorted(owners)


def exec_deployment(directory, command):
    """Record argv before vLLM changes its process title; exec preserves PID identity."""
    process = shared('experiment_process')
    request = process.read_json(Path(directory) / 'deployment.json')
    identity = process.process_identity(os.getpid())
    if (not command or not identity or os.getenv('SERVICE_RUN_ID') != request['runId'] or
        request['model']['location']['path'] not in command):
        raise ValueError('Deployment launch does not match its fixed model and service identity')
    process.atomic_json(Path(directory) / 'service-launch.json', process.redact({
        'attemptId': request['attemptId'], 'runId': request['runId'], 'pid': os.getpid(),
        'processIdentity': identity, 'command': command, 'entrypointFingerprints': process.entrypoint_fingerprints(command)}))
    os.execvpe(command[0], command, os.environ)


def observe_deployment(directory, *, process_module=None, fetch_fn=None, environment_fn=None, listener_fn=deployment_listener_owners):
    """Called by the startup script only after its readiness checks have passed."""
    process = process_module or shared('experiment_process')
    directory = Path(directory)
    request = process.read_json(directory / 'deployment.json')
    pid = int((Path(request['pidDirectory']) / 'vllm.pid').read_text().strip())
    identity_before = process.process_identity(pid)
    launch_path = directory / 'service-launch.json'
    launch = process.read_json(launch_path) if launch_path.exists() else {}
    launched_here = bool(identity_before and launch.get('attemptId') == request['attemptId'] and
        launch.get('runId') == request['runId'] and launch.get('pid') == pid and launch.get('processIdentity') == identity_before)
    environment = b'' if launched_here else (environment_fn or (lambda value: Path(f'/proc/{value}/environ').read_bytes()))(pid)
    if not identity_before or (not launched_here and ('SERVICE_RUN_ID=' + request['runId']).encode() not in environment.split(b'\0')):
        raise ValueError('Serving process is not owned by this deployment')
    commands = process.descendant_commands(pid)
    version = request['model']
    actual_commands = [launch['command']] if launched_here else commands
    if not any(version['location']['path'] in command for command in actual_commands):
        raise ValueError('Serving process did not load the recorded model snapshot')
    if shared('experiment_assets').manifest(version['location']['path'])['fingerprint'] != version['fingerprint']:
        raise ValueError('Serving model snapshot changed')
    host = request['host']
    if host in {'0.0.0.0', '::', ''}:
        host = '127.0.0.1'
    if ':' in host and not host.startswith('['):
        host = '[' + host + ']'
    url = f"http://{host}:{int(request['port'])}/v1/models"
    if fetch_fn:
        payload = fetch_fn(url)
    else:
        with build_opener(ProxyHandler({})).open(Request(url, headers={'Authorization': 'Bearer EMPTY'}), timeout=5) as response:
            payload = json.loads(response.read(1024 * 1024))
    model_ids = [entry.get('id') for entry in payload.get('data', []) if isinstance(entry, dict)]
    listener_pids = listener_fn(pid, request['port'])
    if not listener_pids or request['modelName'] not in model_ids or process.process_identity(pid) != identity_before:
        raise ValueError('Service readiness response or process identity changed')
    receipt = {'schemaVersion': 1, 'attemptId': request['attemptId'], 'instanceId': request['instanceId'],
        'runId': request['runId'], 'modelVersionId': version['id'], 'modelFingerprint': version['fingerprint'],
        'readyAt': time.time(), 'pid': pid, 'processIdentity': identity_before, 'observedCommands': commands,
        'launchCommand': launch.get('command') if launched_here else None,
        'launchEntrypointFingerprints': launch.get('entrypointFingerprints') if launched_here else None,
        'apiModelName': request['modelName'], 'apiModelIds': model_ids, 'endpoint': url, 'listenerPids': listener_pids,
        'entrypointFingerprints': {path: digest for command in commands for path, digest in process.entrypoint_fingerprints(command).items()}}
    process.atomic_json(directory / 'readiness.json', process.redact(receipt))
    return receipt


def collect_deployment(tracker, execution):
    runtime = execution['runtime']
    if runtime.get('archiveStatus') == 'complete':
        return
    process = shared('experiment_process')
    attempt = execution['attempt_id']
    observation = process.probe(runtime['experimentDirectory'])
    if observation.get('attemptId') != attempt:
        raise ValueError('Deployment execution identity mismatch')
    state = observation['state']
    meta = process.read_json(runtime['instanceMetaPath'])
    if meta.get('instance_id') != runtime['serviceInstanceId'] or meta.get('experiment', {}).get('attemptId') != attempt:
        raise ValueError('Deployment instance metadata belongs to another execution')
    if state not in {'succeeded', 'failed', 'cancelled'}:
        if state != execution['state']:
            tracker.emit(attempt, 'status', {'status': state}, preserve_terminal=True)
        return
    if state == 'succeeded' and meta.get('deployment_registration') not in {'confirmed', 'failed'}:
        if execution['state'] != 'unknown' or runtime.get('archiveStatus') != 'awaiting_registration':
            tracker.emit(attempt, 'status', {'status': 'unknown', 'runtime': {'archiveStatus': 'awaiting_registration'}}, preserve_terminal=True)
        return
    if meta.get('deployment_registration') == 'failed':
        state = 'failed'
    interim = 'unknown' if state == 'succeeded' else state
    if execution['state'] != interim:
        tracker.emit(attempt, 'status', {'status': interim}, preserve_terminal=True)
    receipt_path = Path(runtime['experimentDirectory']) / 'readiness.json'
    report = tracker.event_payload(attempt, 'deployment-receipt')
    if report:
        receipt_path = Path(report['location']['path']) / 'readiness.json'
    receipt = process.read_json(receipt_path) if receipt_path.exists() else None
    model = runtime['inputAssets'][0]
    if receipt:
        if (receipt.get('attemptId') != attempt or receipt.get('instanceId') != runtime['serviceInstanceId'] or
            receipt.get('runId') != runtime['serviceRunId'] or receipt.get('modelVersionId') != model['id'] or
            receipt.get('modelFingerprint') != model['fingerprint']):
            raise ValueError('Service readiness receipt does not match the deployment')
        if not report:
            artifact, _ = local_asset(runtime['stateDirectory'], execution['actor_user_id'], tracker.node_id,
                receipt_path, 'report', 'deployment_receipt', 'output')
            if process.read_json(Path(artifact['location']['path']) / 'readiness.json') != receipt:
                raise ValueError('Readiness receipt changed during archival')
            artifact['metadata']['inputVersionIds'] = [model['id']]
            tracker.emit_once(attempt, 'artifact', artifact, 'deployment-receipt')
        initial = tracker.latest_payload(attempt, 'configuration') or {'configuration': {}}
        if not tracker.event_payload(attempt, 'deployment-configuration'):
            tracker.emit_once(attempt, 'configuration', {'configuration': {**initial['configuration'], 'actual': receipt,
                'completeness': 'incomplete', 'missing': ['serving_engine_effective_defaults']}}, 'deployment-configuration')
    elif state == 'succeeded':
        raise ValueError('Startup exited successfully without a verified readiness receipt; update the service startup script')
    journal = observation.get('process') or {}
    tracker.emit(attempt, 'status', {'status': state, 'runtime': {'archiveStatus': 'complete', 'archiveError': None,
        'exitCode': journal.get('exitCode'), 'signals': journal.get('signals', []),
        'processIdentity': journal.get('identity'), 'processStartedAt': journal.get('startedAt'),
        'processFinishedAt': journal.get('finishedAt'),
        'serviceInstanceId': runtime['serviceInstanceId'], 'readyAt': receipt.get('readyAt') if receipt else None},
        **({'error': meta.get('error') or f'Service startup {state}'} if state != 'succeeded' else {})}, preserve_terminal=True)


def start_benchmark(cmd, log_file, *, job, cfg, instance, state_dir, node_id, owner_id, model_commands, context=None):
    owner_id = actor_user_id(owner_id)
    tracker = get_tracker(state_dir, node_id)
    cmd = list(cmd)
    option = '--dataset-root' if '--dataset-root' in cmd else '--dataset'
    index = cmd.index(option) + 1
    dataset, cmd[index] = local_asset(state_dir, owner_id, node_id, cmd[index], 'dataset', 'evaluation_dataset')
    model = observed_model(instance, cfg, node_id, owner_id, model_commands)
    runner = Path(cmd[2])
    protocol_args = []
    ignored = {'--base-url', '--model', '--output', '--dataset-root', '--api-key'}
    if option == '--dataset':
        ignored.add('--dataset')
    cursor = 3
    while cursor < len(cmd):
        argument = cmd[cursor]
        if argument in ignored:
            cursor += 2
            continue
        protocol_args.append(argument)
        cursor += 1
    protocol = {'schemaVersion': 1, 'runnerSha256': hashlib.sha256(runner.read_bytes()).hexdigest(),
        'arguments': protocol_args, 'serviceConfiguration': shared('experiment_tracker').redact(cfg.get('RUNTIME', {})),
        'missing': ['serving_engine_dependency_identity', 'effective_generation_defaults']}
    spec = {'schemaVersion': 1, 'kind': 'benchmark', 'script': runner.name, 'executionBackend': 'inference_worker',
        'serviceInstanceId': instance.get('instance_id'), 'serviceRunId': instance.get('run_id'),
        'resolved': {'command': cmd}, 'protocol': protocol,
        'inputVersions': [{'assetVersionId': item['id'], 'role': item['role'], 'fingerprint': item.get('fingerprint')} for item in (dataset, model)]}
    handle = tracker.prepare(request_key='benchmark-' + job['job_id'], actor_user_id=owner_id,
        name='Benchmark ' + str(job['job_id']), kind='benchmark', container=BACKEND, spec=spec,
        context=inference_experiment_context(instance, context, kind='benchmark'))
    if not tracker.claim_launch('benchmark-' + job['job_id']):
        raise ValueError('Benchmark execution already claimed; inspect its original record')
    attempt = handle['attemptId']
    directory = Path(job['job_dir']) / 'experiment'
    try:
        directory.mkdir(exist_ok=False)
        process = shared('experiment_process')
        process.atomic_json(directory / 'execution.json', {'attemptId': attempt, 'outputPaths': [],
            'kind': 'benchmark', 'businessStdout': {'label': '评测日志', 'kind': 'evaluation_phase'}})
        job['experiment'] = handle
        job['experimentDirectory'] = str(directory)
        if job.get('meta_file'):
            process.atomic_json(job['meta_file'], {**job.get('submission', {}), 'job_id': job['job_id'],
                'experiment': handle, 'experimentDirectory': str(directory), 'status': 'submitting'})
        for item in (dataset, model):
            tracker.emit(attempt, 'artifact', item)
        tracker.emit(attempt, 'configuration', {'configuration': {'schemaVersion': 1,
            'resolved': spec['resolved'], 'actual': None, 'completeness': 'incomplete', 'missing': protocol['missing']}})
        tracker.emit(attempt, 'status', {'status': 'starting', 'runtime': {
            'executionBackend': 'inference_worker', 'experimentDirectory': str(directory), 'archiveStatus': 'pending',
            'executionLogPath': str(Path(log_file).resolve()),
            'benchmarkLogPath': str(directory / 'task-output.log'),
            'stateDirectory': str(Path(state_dir).resolve()), 'inputAssets': [dataset, model],
            'protocol': protocol, 'resultPath': str(Path(job['output']).resolve()),
            'deploymentExperimentId': (instance.get('experiment') or {}).get('experimentId'),
            'serviceInstanceId': instance.get('instance_id'), 'serviceRunId': instance.get('run_id')}})
    except Exception as exc:
        tracker.emit(attempt, 'status', {'status': 'failed', 'error': str(exc), 'runtime': {'archiveStatus': 'complete'}})
        if job.get('meta_file') and Path(job['meta_file']).is_file():
            receipt = shared('experiment_process').read_json(job['meta_file'])
            if receipt.get('experiment', {}).get('attemptId') == attempt:
                shared('experiment_process').atomic_json(job['meta_file'], {**receipt,
                    'status': 'failed', 'return_code': 127, 'error': str(shared('experiment_tracker').redact(str(exc)))})
        raise
    try:
        with open(log_file, 'w', encoding='utf-8') as stream:
            proc = subprocess.Popen([sys.executable, str(Path(process.__file__)), 'run', str(directory), *cmd],
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    except Exception as exc:
        tracker.emit(attempt, 'status', {'status': 'unknown', 'error': str(exc)}, preserve_terminal=True)
        raise
    tracker.emit(attempt, 'status', {'status': 'running', 'runtime': {'pid': str(proc.pid)}}, preserve_terminal=True)
    return proc


def structured_result(data, dataset_id):
    summary, details = data.get('summary'), data.get('details')
    if not isinstance(summary, dict) or not isinstance(details, list) or not all(isinstance(item, dict) for item in details):
        return None
    total, processed = summary.get('total'), summary.get('processed')
    known = type(total) is int and type(processed) is int and 0 <= processed <= total and processed == len(details)
    failed = sum(bool(item.get('error') or item.get('evaluator_error') or
                      str(item.get('response') or item.get('model_answer') or '').startswith('ERROR:')) for item in details) if known else None
    values = summary.get('metrics') if isinstance(summary.get('metrics'), dict) else summary
    metrics = []
    counts = {'executed', 'passed', 'failed', 'timeout', 'executor_error', 'evaluator_error', 'pending', 'case_total', 'case_passed'}
    for name, value in values.items():
        if name in {'total', 'processed', 'progress', 'correct', 'invalid'} or isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and math.isfinite(value):
            count = processed if known else None
            if name == 'pass@1':
                count = values.get('executed') if type(values.get('executed')) is int else None
            if name == 'case_accuracy' or 'instruction_accuracy' in name:
                count = None  # These denominators count cases/instructions, not dataset samples.
            metrics.append({'datasetVersionId': dataset_id, 'subset': 'all', 'name': name, 'value': value,
                'sampleCount': count, 'unit': 'count' if name in counts else 'score' if name == 'total_score' else 'ratio',
                'direction': 'neutral' if name in counts else 'lower' if name.endswith(('error_rate', 'invalid_rate')) else 'higher'})
    return {'expectedCount': total if known else None, 'succeededCount': processed - failed if known else None,
            'failedCount': failed, 'metrics': metrics}


def generation_result(input_path, output_path, dataset_id):
    inputs = list(Path(input_path).glob('*.jsonl'))
    outputs = list(Path(output_path).glob('*.jsonl'))
    if not inputs or not outputs:
        return None
    def records(paths):
        return [json.loads(line) for path in paths for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    expected, rows = len(records(inputs)), records(outputs)
    if len(rows) > expected or not all(isinstance(item, dict) and isinstance(item.get('answer'), str) for item in rows):
        return None
    failed = sum(item['answer'].startswith('ERROR:') for item in rows)
    return {'expectedCount': expected, 'succeededCount': len(rows) - failed, 'failedCount': failed,
        'metrics': [{'datasetVersionId': dataset_id, 'subset': 'all', 'name': 'generated_responses',
            'value': len(rows) - failed, 'sampleCount': len(rows), 'unit': 'count', 'direction': 'neutral'}]}


def collect_benchmark(tracker, execution):
    runtime, attempt = execution['runtime'], execution['attempt_id']
    if runtime.get('archiveStatus') == 'complete':
        collect_benchmark_details(tracker, execution)
        return
    process = shared('experiment_process')
    result = process.probe(runtime['experimentDirectory'])
    if result['attemptId'] != attempt:
        raise ValueError('Benchmark journal identity mismatch')
    state = result['state']
    if state not in {'succeeded', 'failed', 'cancelled'}:
        if execution['state'] != state:
            tracker.emit(attempt, 'status', {'status': state}, preserve_terminal=True)
        return
    interim = 'unknown' if state == 'succeeded' else state
    if execution['state'] != interim:
        tracker.emit(attempt, 'status', {'status': interim, 'runtime': {'exitCode': result.get('process', {}).get('exitCode')},
            **({'error': str(result.get('process', {}).get('error') or 'Benchmark process failed')} if state == 'failed' else {})}, preserve_terminal=True)
    if not tracker.event_payload(attempt, 'benchmark-input-verification'):
        checks = []
        for item in runtime['inputAssets']:
            try:
                actual = shared('experiment_assets').manifest(item['location']['path']) if item['integrity'] == 'verified' else {}
                checks.append({'assetVersionId': item['id'], 'matches': bool(item.get('fingerprint')) and actual.get('fingerprint') == item.get('fingerprint')})
            except Exception as exc:
                checks.append({'assetVersionId': item['id'], 'matches': False, 'error': str(exc)[:1000]})
        tracker.emit_once(attempt, 'status', {'status': tracker.execution(attempt)['state'], 'runtime': {
            'inputIntegrityAfterExecution': 'verified' if all(item['matches'] for item in checks) else 'unverified',
            'inputVerification': checks}}, 'benchmark-input-verification')
    if not tracker.event_payload(attempt, 'benchmark-configuration'):
        journal = result.get('process') or {}
        initial = tracker.latest_payload(attempt, 'configuration') or {'configuration': {}}
        tracker.emit_once(attempt, 'configuration', {'configuration': {**initial['configuration'], 'schemaVersion': 1,
            'actual': {'commands': journal.get('commands'), 'entrypointFingerprints': journal.get('entrypointFingerprints')},
            'completeness': 'incomplete', 'missing': runtime['protocol']['missing']}}, 'benchmark-configuration')
    report = tracker.event_payload(attempt, 'benchmark-report')
    output = Path(runtime['resultPath'])
    if not report and output.exists() and (output.is_file() or any(output.rglob('*'))):
        report, _ = local_asset(runtime['stateDirectory'], execution['actor_user_id'], tracker.node_id,
            output, 'report', 'benchmark_report', 'output')
        tracker.emit_once(attempt, 'artifact', report, 'benchmark-report')
    if state == 'succeeded' and not report:
        raise ValueError('Benchmark exited successfully but produced no result artifact')
    if report and not tracker.event_payload(attempt, 'benchmark-result'):
        # Parse the frozen report, never a subsequently edited live result.
        dataset, model = runtime['inputAssets']
        frozen = Path(report['location']['path'])
        report_files = report['manifest']['files']
        if len(report_files) == 1 and report_files[0]['path'] == output.name and output.suffix.lower() == '.json':
            data = json.loads((frozen / output.name).read_text(encoding='utf-8'))
            payload = structured_result(data, dataset['id'])
        else:
            payload = generation_result(dataset['location']['path'], frozen, dataset['id'])
        if payload:
            payload.update({'id': 'eval-' + hashlib.sha256(attempt.encode()).hexdigest(),
                'modelVersionId': model['id'], 'reportVersionId': report['id'],
                'protocol': runtime['protocol'], 'protocolComplete': False})
            tracker.emit_once(attempt, 'evaluation', payload, 'benchmark-result')
    collect_benchmark_details(tracker, tracker.execution(attempt))
    tracker.emit(attempt, 'status', {'status': state, 'runtime': {'archiveStatus': 'complete', 'archiveError': None}}, preserve_terminal=True)


def collect_benchmark_details(tracker, execution):
    runtime, attempt = execution['runtime'], execution['attempt_id']
    if runtime.get('benchmarkDetailsVersion') == 1:
        return
    report = tracker.event_payload(attempt, 'benchmark-report')
    if not report:
        if runtime.get('archiveStatus') == 'complete':
            tracker.emit(attempt, 'status', {'status': execution['state'], 'runtime': {
                'benchmarkDetailsVersion': 1, 'benchmarkReport': None}}, preserve_terminal=True)
        return
    dataset = (runtime.get('inputAssets') or [{}])[0].get('logicalName', '')
    details = shared('experiment_benchmark').read_report(report, dataset)
    tracker.emit(attempt, 'status', {'status': execution['state'], 'runtime': {
        'benchmarkReport': details, 'benchmarkDetailsVersion': 1, 'benchmarkDetailsError': None}}, preserve_terminal=True)


def reconcile_worker(tracker):
    for execution in tracker.collectable_executions():
        try:
            if execution['runtime'].get('operation') == 'service_deploy':
                collect_deployment(tracker, execution)
            else:
                collect_benchmark(tracker, execution)
        except Exception as exc:
            message = str(shared('experiment_tracker').redact(str(exc)))[:2000]
            current = tracker.execution(execution['attempt_id'])
            if current['runtime'].get('archiveError') != message:
                tracker.emit(execution['attempt_id'], 'status', {'status': current['state'],
                    'runtime': {'archiveStatus': 'retrying', 'archiveError': message}}, preserve_terminal=True)
    # Old results stay immutable. Backfill only their display details from the frozen artifact.
    with tracker.connect() as db:
        rows = db.execute('SELECT attempt_id,runtime_json FROM experiment_execution').fetchall()
    pending = [row for row in rows if (runtime := json.loads(row['runtime_json'])).get('resultPath')
               and runtime.get('archiveStatus') == 'complete' and runtime.get('benchmarkDetailsVersion') != 1]
    for row in pending[:20]:
        execution = tracker.execution(row['attempt_id'])
        try:
            collect_benchmark_details(tracker, execution)
        except (OSError, ValueError, KeyError) as exc:
            error = str(shared('experiment_tracker').redact(str(exc)))[:1000]
            if execution['runtime'].get('benchmarkDetailsError') != error:
                tracker.emit(row['attempt_id'], 'status', {'status': execution['state'],
                    'runtime': {'benchmarkDetailsError': error}}, preserve_terminal=True)


if __name__ == '__main__':
    if sys.argv[1] == '--exec':
        exec_deployment(sys.argv[2], sys.argv[3:])
    elif sys.argv[1] == '--owns':
        try:
            request = shared('experiment_process').read_json(Path(sys.argv[2]) / 'deployment.json')
            instance = {'experimentDirectory': sys.argv[2], 'experiment': {'attemptId': request['attemptId']},
                        'instance_id': request['instanceId'], 'run_id': sys.argv[3]}
            owned = deployment_owned_identity(instance, sys.argv[4])
        except (OSError, ValueError, KeyError, TypeError):
            owned = None
        raise SystemExit(0 if owned else 1)
    else:
        observe_deployment(sys.argv[1])
