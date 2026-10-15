"""Durable workflow model transfer with verified source and destination versions."""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import os
import signal
import subprocess
import sys
import tempfile
import time

# Also run as a standalone child without importing the application's utils package.
if not __package__:
    import types
    package = types.ModuleType('_experiment_publish_runtime')
    package.__path__ = [str(Path(__file__).resolve().parent)]
    sys.modules[package.__name__] = package
    __package__ = package.__name__

from .experiment_capture import container_asset, container_json, asset_registration
from .experiment_process import atomic_json, read_json, probe, process_identity
from .experiment_tracker import TERMINAL


def recorded_source(tracker, source, container, owner, attempts):
    for attempt in attempts:
        execution = tracker.execution(attempt) if attempt else None
        if not execution or execution['actor_user_id'] != owner or execution['container'] != container:
            continue
        for kind in ('model', 'checkpoint', 'adapter'):
            asset = tracker.event_payload(attempt, f'output:{kind}:{source}')
            if asset and asset.get('integrity') == 'verified':
                return {**asset, 'direction': 'input', 'role': 'published_source'}
    captured = container_asset(container, source, owner, 'model')
    return asset_registration(captured, node_id=tracker.node_id, container=container, owner_id=owner,
                              source=source, kind='model', role='published_source')


def summary(execution):
    runtime = execution['runtime']
    state = execution['state']
    progress = {}
    directory = runtime.get('experimentDirectory')
    try:
        observed = read_json(Path(directory) / 'progress.json') if directory else {}
        if observed.get('attemptId') == execution['attempt_id']:
            progress = {key: observed[key] for key in ('progress_percent', 'current_step', 'step_name', 'message') if key in observed}
    except (OSError, ValueError):
        pass
    percent = 100 if state == 'succeeded' else min(int(progress.get('progress_percent', 0)), 99)
    return {'status': 'finished' if state == 'succeeded' else 'failed' if state in {'failed', 'cancelled'} else 'running',
        'pid': runtime.get('pid'), 'container': 'host', 'model_path': runtime.get('destination'),
        'model_name': PurePosixPath(runtime.get('destination') or '').name,
        'experiment_id': execution['experiment_id'], 'experiment_task_id': execution['task_id'],
        'experiment_attempt_id': execution['attempt_id'],
        'log_path': str(Path(directory) / 'execution.log') if directory else None,
        'progress_percent': percent,
        'error': runtime.get('publicationError'),
        'message': '模型发布已校验并归档' if state == 'succeeded' else progress.get('message', '正在复制、校验并归档模型'),
        'metrics': {**progress, 'progress_percent': percent, 'total_steps': 5, 'attemptStatus': state,
                    'source': runtime.get('source'), 'destination': runtime.get('destination'),
                    'archiveStatus': runtime.get('archiveStatus'), 'archiveError': runtime.get('archiveError'),
                    'publishedVersionId': runtime.get('publishedVersionId')}}


def start_publish(tracker, *, owner, source_container, source, target_container, destination, context, source_attempts=(), require_full_model=False):
    request_key = context['requestKey']
    with tracker.connect() as db:
        prior = db.execute('SELECT request_json,handle_json,claimed FROM experiment_intent WHERE request_key=?', (request_key,)).fetchone()
    if prior:
        request = json.loads(prior['request_json'])
        if request['actorUserId'] != owner or request['container'] != source_container:
            raise ValueError('Publication request owner or source container changed')
        spec = request['spec']
        resolved = spec['resolved']
        if resolved['destinationContainer'] != target_container or resolved['destination'] != destination or resolved['source'] != source:
            raise ValueError('Publication request destination or source changed')
        if prior['claimed']:
            handle = json.loads(prior['handle_json'])
            return summary(tracker.execution(handle['attemptId']))
        asset = resolved['sourceAsset']
    else:
        if not destination.startswith('/') or '..' in PurePosixPath(destination).parts or destination == '/':
            raise ValueError('Publication requires an absolute destination directory')
        asset = recorded_source(tracker, source, source_container, owner, source_attempts)
        spec = {'schemaVersion': 1, 'script': 'workflow_publish', 'kind': 'publish', 'executionBackend': 'container_transfer',
            'inputVersions': [{'assetVersionId': asset['id'], 'role': asset['role'], 'fingerprint': asset['fingerprint']}],
            'resolved': {'sourceContainer': source_container, 'source': source, 'sourceAsset': asset,
                         'destinationContainer': target_container, 'destination': destination}}
    if require_full_model:
        container_json(source_container, 'model_publication_files.py', {'path': asset['location']['path']})
    handle = tracker.prepare(request_key=request_key, actor_user_id=owner, name='Publish ' + PurePosixPath(destination).name,
                             kind='publish', container=source_container, spec=spec, context=context)
    attempt = handle['attemptId']
    directory = Path(tracker.db_path).resolve().parent / 'publications' / attempt
    # Persist recoverable identity before the launch claim or any copy request.
    tracker.emit_once(attempt, 'artifact', asset, 'publication-input')
    tracker.emit_once(attempt, 'configuration', {'configuration': {'schemaVersion': 1,
        'resolved': spec['resolved'], 'actual': None, 'completeness': 'incomplete', 'missing': ['transfer_verification']}}, 'publication-preparation')
    tracker.emit_once(attempt, 'status', {'status': 'starting', 'runtime': {
        'operation': 'model_publish', 'kind': 'publish', 'experimentDirectory': str(directory), 'archiveStatus': 'pending',
        'executionLogPath': str(directory / 'execution.log'),
        'source': source, 'sourceAsset': asset, 'destinationContainer': target_container, 'destination': destination}}, 'publication-intent')
    if not tracker.claim_launch(request_key):
        return summary(tracker.execution(attempt))
    try:
        directory.mkdir(parents=True, exist_ok=False)
        atomic_json(directory / 'execution.json', {'attemptId': attempt, 'outputPaths': []})
        atomic_json(directory / 'transfer.json', {'attemptId': attempt, 'owner': owner,
            'sourceContainer': source_container, 'sourceAsset': asset, 'destinationContainer': target_container,
            'destination': destination})
    except Exception as exc:
        tracker.emit(attempt, 'status', {'status': 'failed', 'error': str(exc),
            'runtime': {'archiveStatus': 'complete', 'publicationError': str(exc)}})
        return summary(tracker.execution(attempt))
    try:
        runner = Path(__file__).with_name('experiment_process.py')
        with (directory / 'execution.log').open('w', encoding='utf-8') as stream:
            child = subprocess.Popen([sys.executable, str(runner), 'run', str(directory),
                sys.executable, str(Path(__file__).resolve()), str(directory)],
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    except Exception as exc:
        tracker.emit(attempt, 'status', {'status': 'unknown', 'error': str(exc),
            'runtime': {'publicationError': str(exc)}}, preserve_terminal=True)
        # Keep the handle in the workflow stage even when the launch response is
        # uncertain; a retry must reconcile this execution instead of duplicating it.
        return summary(tracker.execution(attempt))
    tracker.emit(attempt, 'status', {'status': 'running', 'runtime': {'pid': str(child.pid)}}, preserve_terminal=True)
    return summary(tracker.execution(attempt))


def stream_copy(source_container, source, target_container, destination, *, timeout=7200, popen=subprocess.Popen):
    """Pipe binary tar data without buffering model weights on the Runtime host."""
    producer = consumer = None
    deadline = time.monotonic() + timeout
    with tempfile.TemporaryFile() as source_errors, tempfile.TemporaryFile() as target_errors:
        try:
            producer = popen(['docker', 'exec', source_container, 'tar', '-C', source, '-cf', '-', '.'],
                stdout=subprocess.PIPE, stderr=source_errors)
            consumer = popen(['docker', 'exec', '-i', target_container, 'tar', '-C', destination, '-xf', '-'],
                stdin=producer.stdout, stdout=subprocess.DEVNULL, stderr=target_errors)
            producer.stdout.close()
            target_code = consumer.wait(timeout=max(0.01, deadline - time.monotonic()))
            if target_code:
                raise ValueError('Model copy failed at destination')
            source_code = producer.wait(timeout=max(0.01, deadline - time.monotonic()))
            if source_code:
                raise ValueError('Model copy failed at source')
        except (ValueError, subprocess.TimeoutExpired) as exc:
            details = []
            for errors in (source_errors, target_errors):
                errors.seek(0, os.SEEK_END)
                errors.seek(max(0, errors.tell() - 2000))
                details.append(errors.read().decode('utf-8', errors='replace'))
            raise ValueError(f'{exc}: ' + '\n'.join(details)) from exc
        finally:
            for process in (consumer, producer):
                if process is not None:
                    if process.poll() is None:
                        process.kill()
                    process.wait()
            if producer is not None and producer.stdout is not None:
                producer.stdout.close()


def transfer(directory, *, capture_fn=container_asset, control_fn=container_json, stream_fn=stream_copy):
    directory = Path(directory)
    request = read_json(directory / 'transfer.json')
    source = request['sourceAsset']
    expected = source['fingerprint']
    target = request['destinationContainer']
    def progress(percent, step, name, message):
        atomic_json(directory / 'progress.json', {'attemptId': request['attemptId'],
            'progress_percent': percent, 'current_step': step, 'step_name': name, 'message': message})
    def verify(container, path):
        observed = capture_fn(container, path, request['owner'], 'model', operation='manifest')
        if observed['fingerprint'] != expected:
            raise ValueError('Publication contents differ from the fixed source version')
    def control(operation):
        return control_fn(target, 'experiment_publish_files.py', {**request, 'operation': operation})
    progress(0, 1, 'verify_source', '正在校验来源模型版本')
    verify(request['sourceContainer'], source['location']['path'])
    claimed = control('claim')
    try:
        if claimed['destinationExists']:
            verify(target, request['destination'])
        else:
            progress(10, 2, 'stream_to_inference', '正在流式传输模型到推理容器暂存目录')
            stream_fn(request['sourceContainer'], source['location']['path'], target, claimed['staging'])
            progress(80, 4, 'verify_destination', '正在校验目标模型内容')
            verify(target, claimed['staging'])
            control('activate')
        progress(90, 5, 'archive_model', '正在归档已发布模型版本')
        verify(target, request['destination'])
        captured = capture_fn(target, request['destination'], request['owner'], 'model')
        if captured['fingerprint'] != expected:
            raise ValueError('Published model changed while creating its version')
        atomic_json(directory / 'publication.json', {'attemptId': request['attemptId'], 'sourceVersionId': source['id'],
            'destination': request['destination'], 'destinationContainer': target, 'capture': captured})
        progress(99, 5, 'await_archive', '复制与校验完成，等待实验记录归档')
    finally:
        control('release')


def collect_publish(tracker, execution, *, probe_fn=probe, capture_fn=container_asset, control_fn=container_json):
    runtime, attempt = execution['runtime'], execution['attempt_id']
    if runtime.get('archiveStatus') == 'complete':
        return
    observed = probe_fn(runtime['experimentDirectory'])
    if observed.get('attemptId') != attempt:
        raise ValueError('Publication journal identity mismatch')
    state = observed['state']
    if state not in TERMINAL:
        if state != execution['state']:
            tracker.emit(attempt, 'status', {'status': state}, preserve_terminal=True)
        return
    # A terminated worker may not run its finally block. Release only its own
    # destination lock after the supervisor has recorded authoritative exit.
    control_fn(runtime['destinationContainer'], 'experiment_publish_files.py', {
        'operation': 'release_if_owned', 'attemptId': attempt, 'destination': runtime['destination']})
    interim_state = 'unknown' if state == 'succeeded' else state
    if execution['state'] not in TERMINAL and execution['state'] != interim_state:
        tracker.emit(attempt, 'status', {'status': interim_state, 'runtime': {
            'exitCode': observed.get('process', {}).get('exitCode')}}, preserve_terminal=True)
    receipt_path = Path(runtime['experimentDirectory']) / 'publication.json'
    if state != 'succeeded' and not receipt_path.is_file():
        message = observed.get('process', {}).get('error') or f"Publication process ended: {state}"
        tracker.emit(attempt, 'status', {'status': state, 'error': message,
            'runtime': {'archiveStatus': 'complete', 'publicationError': message}}, preserve_terminal=True)
        return
    receipt = read_json(receipt_path)
    if (receipt.get('attemptId') != attempt or receipt.get('sourceVersionId') != runtime['sourceAsset']['id'] or
        receipt.get('destination') != runtime['destination'] or receipt.get('destinationContainer') != runtime['destinationContainer'] or
        receipt['capture']['fingerprint'] != runtime['sourceAsset']['fingerprint']):
        raise ValueError('Publication receipt does not match this execution')
    if not tracker.event_payload(attempt, 'published-model'):
        # Recheck the retained version; the deployed path may since have been
        # deliberately changed. The receipt records what this execution published.
        captured = receipt['capture']
        current = capture_fn(runtime['destinationContainer'], captured['path'], execution['actor_user_id'], 'model', operation='manifest')
        if current['fingerprint'] != captured['fingerprint']:
            raise ValueError('Published immutable version changed')
        output = asset_registration(captured, node_id=tracker.node_id, container=runtime['destinationContainer'],
            owner_id=execution['actor_user_id'], source=runtime['destination'], kind='model', role='published_model', direction='output')
        output['metadata']['inputVersionIds'] = [runtime['sourceAsset']['id']]
        tracker.emit_once(attempt, 'artifact', output, 'published-model')
    output = tracker.event_payload(attempt, 'published-model')
    initial = tracker.event_payload(attempt, 'publication-preparation')['configuration']
    tracker.emit_once(attempt, 'configuration', {'configuration': {**initial, 'actual': {
        'sourceVersionId': runtime['sourceAsset']['id'], 'publishedVersionId': output['id'],
        'contentFingerprint': output['fingerprint'], 'destination': runtime['destination'],
        'observedCommands': observed.get('process', {}).get('commands', []),
        'entrypointFingerprints': observed.get('process', {}).get('entrypointFingerprints', {})},
        'completeness': 'complete', 'missing': []}, 'environment': {
            'python': observed.get('process', {}).get('python'), 'platform': observed.get('process', {}).get('platform')}}, 'publication-configuration')
    tracker.emit(attempt, 'status', {'status': state, 'runtime': {
        'archiveStatus': 'complete', 'archiveError': None, 'publishedVersionId': output['id'],
        'publicationError': None if state == 'succeeded' else f'Publication process ended: {state}'}}, preserve_terminal=True)


def stop_publish(execution):
    journal = read_json(Path(execution['runtime']['experimentDirectory']) / 'process.json')
    pid = journal.get('pid')
    if journal.get('exitCode') is not None:
        return
    if not pid or not journal.get('identity') or process_identity(pid) != journal['identity']:
        raise ValueError('Cannot stop publication: original process identity is not running')
    os.kill(int(pid), signal.SIGTERM)


if __name__ == '__main__':
    transfer(sys.argv[1])
