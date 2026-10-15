"""Asynchronous manual publication, with a durable preparation journal.

Preparation also runs outside the HTTP process: hashing a large model must not
block an API response, and closing the page must not interrupt the transfer.
"""
from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath
import subprocess
import sys
import time

if not __package__:
    import types
    package = types.ModuleType('_model_publication_runtime')
    package.__path__ = [str(Path(__file__).resolve().parent)]
    sys.modules[package.__name__] = package
    __package__ = package.__name__

from .experiment_process import atomic_json, read_json, probe
from .experiment_publish import start_publish, summary
from .experiment_capture import container_json


def job_root(tracker):
    return Path(tracker.db_path).resolve().parent / 'model-publications'


def job_directory(tracker, request_key):
    if not re.fullmatch(r'pub-[a-f0-9-]{36}', request_key):
        raise ValueError('Invalid model publication request key')
    return job_root(tracker) / request_key


def validate_request(request):
    for key in ('sourceContainer', 'destinationContainer'):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', request[key]):
            raise ValueError('Invalid publication container')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', request['owner']):
        raise ValueError('Invalid publication owner')
    for key in ('source', 'destination'):
        path = PurePosixPath(request[key])
        if not path.is_absolute() or '..' in path.parts or str(path) == '/' or '\\' in request[key]:
            raise ValueError('Invalid publication path')
    source, target = PurePosixPath(request['source']), PurePosixPath(request['destination'])
    if request['sourceContainer'] == request['destinationContainer'] and (source == target or source in target.parents):
        raise ValueError('发布目标不能位于来源模型目录内')


def start(tracker, request, *, popen=subprocess.Popen):
    validate_request(request)
    directory = job_directory(tracker, request['requestKey'])
    directory.parent.mkdir(parents=True, exist_ok=True)
    try:
        directory.mkdir()  # Exclusive durable launch claim for retries/lost HTTP responses.
    except FileExistsError:
        if read_json(directory / 'request.json') != request:
            raise ValueError('Publication request key was already used for different inputs')
        return status(tracker, request['requestKey'], request['owner'])
    try:
        atomic_json(directory / 'request.json', request)
        atomic_json(directory / 'execution.json', {'attemptId': request['requestKey'], 'outputPaths': []})
        with (directory / 'execution.log').open('wb') as stream:
            popen([sys.executable, str(Path(__file__).with_name('experiment_process.py')), 'run', str(directory),
                   sys.executable, str(Path(__file__).resolve()), str(directory), str(tracker.db_path), tracker.node_id],
                  stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    except Exception as exc:
        atomic_json(directory / 'error.json', {'error': str(exc)})
    return status(tracker, request['requestKey'], request['owner'])


def status(tracker, request_key, owner):
    directory = job_directory(tracker, request_key)
    request = read_json(directory / 'request.json')
    if request['owner'] != owner:
        raise ValueError('Publication is unavailable for this user')
    result = {'requestKey': request_key, 'source': request['source'],
              'sourceContainer': request['sourceContainer'], 'destination': request['destination'],
              'destinationContainer': request['destinationContainer'], 'status': 'preparing',
              'progress_percent': 0, 'message': '正在检查并固定来源模型版本',
              'experiment_id': None, 'error': None}
    # Recover the handle even if the preparation child exited before saving it.
    with tracker.connect() as db:
        row = db.execute('SELECT handle_json FROM experiment_intent WHERE request_key=?', (request_key,)).fetchone()
    if row and row['handle_json']:
        execution = tracker.execution(json.loads(row['handle_json'])['attemptId'])
        if execution and execution['actor_user_id'] == owner:
            result.update(summary(execution))
            result['error'] = execution['runtime'].get('publicationError') or execution['runtime'].get('archiveError') or result.get('error')
            if execution['state'] == 'unknown':
                result['status'] = 'unknown'
            if (directory / 'error.json').is_file():
                result['error'] = read_json(directory / 'error.json')['error']
            return result
    if (directory / 'error.json').is_file():
        result.update(status='failed', error=read_json(directory / 'error.json')['error'], message='模型发布准备失败')
    elif (directory / 'process.json').is_file():
        observed = probe(directory)
        if observed['state'] in ('failed', 'cancelled', 'succeeded'):
            result.update(status='failed', error='模型发布准备进程已退出，请重试')
        elif observed['state'] == 'unknown':
            result.update(status='unknown', error='发布准备进程状态待核对，请查看 Runtime 日志')
    elif time.time() - (directory / 'request.json').stat().st_mtime > 60:
        result.update(status='unknown', error='未找到发布准备进程记录，请查看 Runtime 日志')
    return result


def list_for_source(tracker, owner, container, source):
    root = job_root(tracker)
    jobs = []
    if root.exists():
        for directory in root.iterdir():
            if not directory.is_dir() or not (directory / 'request.json').is_file():
                continue
            request = read_json(directory / 'request.json')
            if request['owner'] == owner and request['sourceContainer'] == container and request['source'] == source:
                jobs.append(directory)
    # Include running jobs even when there have been many completed publications.
    results = [status(tracker, directory.name, owner) for directory in
               sorted(jobs, key=lambda item: item.stat().st_mtime, reverse=True)]
    active = [item for item in results if item['status'] not in ('finished', 'failed')]
    return active + [item for item in results if item['status'] in ('finished', 'failed')][:5]


def prepare(directory, tracker):
    directory = Path(directory)
    request = read_json(directory / 'request.json')
    try:
        validate_request(request)
        container_json(request['sourceContainer'], 'model_publication_files.py', {'path': request['source']})
        result = start_publish(tracker, owner=request['owner'], source_container=request['sourceContainer'],
            source=request['source'], target_container=request['destinationContainer'], destination=request['destination'],
            context={'requestKey': request['requestKey']}, require_full_model=True)
        atomic_json(directory / 'handle.json', result)
        tracker.flush_once()
    except Exception as exc:
        atomic_json(directory / 'error.json', {'error': str(exc)})
        with tracker.connect() as db:
            intent = db.execute('SELECT handle_json,claimed FROM experiment_intent WHERE request_key=?',
                                (request['requestKey'],)).fetchone()
        if intent and intent['handle_json'] and not intent['claimed']:
            tracker.emit(json.loads(intent['handle_json'])['attemptId'], 'status', {'status': 'failed',
                'runtime': {'archiveStatus': 'complete', 'publicationError': str(exc)}}, preserve_terminal=True)
        raise


if __name__ == '__main__':
    from .experiment_tracker import ExperimentTracker
    prepare(sys.argv[1], ExperimentTracker(sys.argv[2], sys.argv[3]))
