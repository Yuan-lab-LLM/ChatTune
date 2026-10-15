"""Durable, bounded log transport for an experiment attempt."""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path

from .experiment_process import redact

TERMINAL = {'succeeded', 'failed', 'cancelled'}
READ_SIZE = 32768
CHUNK_SIZE = 8192


def initialize(db):
    db.executescript('''
        CREATE TABLE IF NOT EXISTS experiment_log_enabled (
            attempt_id TEXT PRIMARY KEY, completed INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE IF NOT EXISTS experiment_log_local_source (
            source_id TEXT PRIMARY KEY, attempt_id TEXT NOT NULL, source_key TEXT NOT NULL,
            generation INTEGER NOT NULL, label TEXT NOT NULL, kind TEXT NOT NULL,
            backend TEXT NOT NULL, path TEXT NOT NULL, identity TEXT,
            read_offset INTEGER NOT NULL DEFAULT 0, partial BLOB NOT NULL DEFAULT X'',
            status TEXT NOT NULL DEFAULT 'streaming', error TEXT,
            last_event_sequence INTEGER NOT NULL DEFAULT 0, sent_status TEXT,
            UNIQUE(attempt_id, source_key, generation));
        CREATE TABLE IF NOT EXISTS experiment_log_pending (
            source_id TEXT NOT NULL, start_offset INTEGER NOT NULL, end_offset INTEGER NOT NULL,
            content TEXT NOT NULL, PRIMARY KEY(source_id, start_offset));
    ''')
    columns = {row[1] for row in db.execute('PRAGMA table_info(experiment_log_enabled)')}
    if 'completed' not in columns:
        db.execute('ALTER TABLE experiment_log_enabled ADD COLUMN completed INTEGER NOT NULL DEFAULT 0')


def enable(tracker, attempt_id):
    with tracker.connect() as db:
        db.execute('INSERT OR IGNORE INTO experiment_log_enabled(attempt_id) VALUES (?)', (attempt_id,))


def _source_id(attempt, key, generation):
    return 'log-' + hashlib.sha256(f'{attempt}:{key}:{generation}'.encode()).hexdigest()


def source(tracker, attempt, key, label, kind, backend, path):
    with tracker.connect() as db:
        row = db.execute('''SELECT * FROM experiment_log_local_source WHERE attempt_id=? AND source_key=?
            ORDER BY generation DESC LIMIT 1''', (attempt, key)).fetchone()
        if row:
            if row['backend'] != backend or row['path'] != path:
                raise ValueError('Experiment log source changed its declared path')
            return dict(row)
        source_id = _source_id(attempt, key, 0)
        db.execute('''INSERT INTO experiment_log_local_source
            (source_id,attempt_id,source_key,generation,label,kind,backend,path)
            VALUES (?,?,?,?,?,?,?,?)''', (source_id, attempt, key, 0, label, kind, backend, path))
        return dict(db.execute('SELECT * FROM experiment_log_local_source WHERE source_id=?', (source_id,)).fetchone())


def _rotate(tracker, row, identity):
    with tracker.connect() as db:
        _queue(db, row, b'', True)
        db.execute("UPDATE experiment_log_local_source SET status='complete' WHERE source_id=?", (row['source_id'],))
        generation = row['generation'] + 1
        source_id = _source_id(row['attempt_id'], row['source_key'], generation)
        db.execute('''INSERT OR IGNORE INTO experiment_log_local_source
            (source_id,attempt_id,source_key,generation,label,kind,backend,path,identity)
            VALUES (?,?,?,?,?,?,?,?,?)''', (source_id, row['attempt_id'], row['source_key'], generation,
                f"{row['label']} ({generation + 1})", row['kind'], row['backend'], row['path'], identity))
        return dict(db.execute('SELECT * FROM experiment_log_local_source WHERE source_id=?', (source_id,)).fetchone())


def _read_host(path, offset):
    file = Path(path)
    if not file.is_file() or file.is_symlink():
        return {'exists': False}
    with file.open('rb') as stream:
        stat = os.fstat(stream.fileno())
        stream.seek(min(offset, stat.st_size))
        data = stream.read(READ_SIZE)
    return {'exists': True, 'size': stat.st_size, 'identity': f'{stat.st_dev}:{stat.st_ino}', 'data': data}


def _read_container(container, directory, key, offset):
    from .experiment_capture import container_json
    result = container_json(container, 'experiment_process.py', {'directory': directory, 'key': key,
        'offset': offset, 'limit': READ_SIZE}, 'log-read')
    result['data'] = base64.b64decode(result.get('data') or '')
    return result


def _queue(db, row, data, terminal):
    previous = bytes(row['partial'])
    combined = previous + data
    complete = len(combined) if terminal else combined.rfind(b'\n') + 1
    if complete < 0:
        complete = 0
    start = row['read_offset'] - len(previous)
    ready = combined[:complete]
    # Keep a whole ordinary line together so credentials split across file reads
    # are redacted before any part is persisted or uploaded.
    for line in ready.splitlines(keepends=True):
        safe = redact(line.decode('utf-8', errors='replace'))
        count = max(1, (len(line) + CHUNK_SIZE - 1) // CHUNK_SIZE,
                    (len(safe) + 16000 - 1) // 16000)
        for index in range(count):
            raw_start = len(line) * index // count
            raw_end = len(line) * (index + 1) // count
            content = safe[len(safe) * index // count:len(safe) * (index + 1) // count]
            db.execute('''INSERT INTO experiment_log_pending(source_id,start_offset,end_offset,content)
                VALUES (?,?,?,?)''', (row['source_id'], start + raw_start, start + raw_end, content))
        start += len(line)
    db.execute('''UPDATE experiment_log_local_source SET read_offset=?, partial=? WHERE source_id=?''',
               (row['read_offset'] + len(data), combined[complete:], row['source_id']))


def sync_file(tracker, execution, row):
    # Output archiving can keep an exited medical evaluation in 'unknown'.
    # The supervisor's exit receipt still proves that its log tail is final.
    terminal = execution['state'] in TERMINAL or execution['runtime'].get('exitCode') is not None
    if row['status'] == 'complete':
        return
    for _ in range(32):  # Catch up at up to 1 MiB per source per poll.
        try:
            if row['backend'] == 'container':
                result = _read_container(execution['container'], execution['runtime']['experimentDirectory'],
                                         row['source_key'], row['read_offset'])
            else:
                result = _read_host(row['path'], row['read_offset'])
        except Exception as exc:
            with tracker.connect() as db:
                db.execute("UPDATE experiment_log_local_source SET status='error', error=? WHERE source_id=?",
                           (str(redact(str(exc)))[:2000], row['source_id']))
            return
        if not result['exists']:
            if terminal:
                with tracker.connect() as db:
                    _queue(db, row, b'', True)
                    db.execute("UPDATE experiment_log_local_source SET status='error', error=? WHERE source_id=?",
                               ('Log file is unavailable', row['source_id']))
            return
        identity = result['identity']
        if row['identity'] and (row['identity'] != identity or result['size'] < row['read_offset']):
            row = _rotate(tracker, row, identity)
            continue
        finished = terminal and result['size'] <= row['read_offset'] + len(result['data'])
        with tracker.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            db.execute('UPDATE experiment_log_local_source SET identity=? WHERE source_id=?', (identity, row['source_id']))
            db.execute("UPDATE experiment_log_local_source SET status='streaming', error=NULL WHERE source_id=?",
                       (row['source_id'],))
            _queue(db, row, result['data'], finished)
            if finished:
                db.execute("UPDATE experiment_log_local_source SET status='complete' WHERE source_id=?", (row['source_id'],))
            row = dict(db.execute('SELECT * FROM experiment_log_local_source WHERE source_id=?',
                                  (row['source_id'],)).fetchone())
        if len(result['data']) < READ_SIZE or finished:
            return


def sync_events(tracker, execution):
    row = source(tracker, execution['attempt_id'], 'operation', 'operation', 'operation', 'events', '')
    with tracker.connect() as db:
        events = db.execute('''SELECT sequence,event_json FROM experiment_outbox WHERE attempt_id=? AND sequence>?
            ORDER BY sequence''', (execution['attempt_id'], row['last_event_sequence'])).fetchall()
        for item in events:
            event = json.loads(item['event_json'])
            if event['type'] == 'status':
                payload = event['payload']
                line = f"{event['occurredAt']}  {payload['status']}"
                if payload.get('error'):
                    line += f"  {str(redact(payload['error']))[:2000]}"
                data = (line + '\n').encode('utf-8')
                _queue(db, row, data, True)
                row = dict(db.execute('SELECT * FROM experiment_log_local_source WHERE source_id=?',
                                      (row['source_id'],)).fetchone())
            db.execute('UPDATE experiment_log_local_source SET last_event_sequence=? WHERE source_id=?',
                       (item['sequence'], row['source_id']))
        if execution['state'] in TERMINAL:
            db.execute("UPDATE experiment_log_local_source SET status='complete' WHERE source_id=?", (row['source_id'],))


def flush(tracker, attempt_id):
    with tracker.connect() as db:
        sources = db.execute('SELECT * FROM experiment_log_local_source WHERE attempt_id=?', (attempt_id,)).fetchall()
    for row in sources:
        for _ in range(16):
            with tracker.connect() as db:
                pending = db.execute('''SELECT * FROM experiment_log_pending WHERE source_id=? ORDER BY start_offset LIMIT 8''',
                                     (row['source_id'],)).fetchall()
                more_pending = bool(pending and db.execute('''SELECT 1 FROM experiment_log_pending
                    WHERE source_id=? AND start_offset>? LIMIT 1''',
                    (row['source_id'], pending[-1]['start_offset'])).fetchone())
            sent_status = 'streaming' if more_pending else row['status']
            if not pending and row['sent_status'] == sent_status:
                break
            payload = {'runtimeNodeId': tracker.node_id, 'attemptId': attempt_id,
                'source': {'id': row['source_id'], 'label': row['label'], 'kind': row['kind'],
                           'status': sent_status,
                           **({'error': row['error']} if row['error'] and sent_status == row['status'] else {})},
                'chunks': [{'startOffset': item['start_offset'], 'endOffset': item['end_offset'],
                            'content': item['content']} for item in pending]}
            result = tracker.send('appendExperimentLogChunks', payload)
            if pending and result.get('nextOffset', -1) < pending[-1]['end_offset']:
                raise ValueError('Studio did not acknowledge the submitted log chunks')
            with tracker.connect() as db:
                for item in pending:
                    db.execute('DELETE FROM experiment_log_pending WHERE source_id=? AND start_offset=?',
                               (row['source_id'], item['start_offset']))
                db.execute('UPDATE experiment_log_local_source SET sent_status=? WHERE source_id=?',
                           (sent_status, row['source_id']))
            row = dict(row)
            row['sent_status'] = sent_status


def sync_all(tracker):
    with tracker.connect() as db:
        enabled = [row[0] for row in db.execute('SELECT attempt_id FROM experiment_log_enabled WHERE completed=0')]
    for attempt in enabled:
        execution = tracker.execution(attempt)
        if not execution:
            continue
        capture_ok = True
        upload_ok = True
        try:
            sync_events(tracker, execution)
            runtime = execution['runtime']
            directory = runtime.get('experimentDirectory')
            path = runtime.get('executionLogPath')
            if directory and path:
                backend = 'host' if execution['container'] == 'inference-worker' or runtime.get('operation') == 'model_publish' else 'container'
                row = source(tracker, attempt, 'execution', 'execution.log', 'execution', backend, path)
                sync_file(tracker, execution, row)
                if backend == 'host' and runtime.get('benchmarkLogPath'):
                    row = source(tracker, attempt, 'benchmark', '评测日志', 'evaluation_phase',
                                 'host', runtime['benchmarkLogPath'])
                    sync_file(tracker, execution, row)
                if backend == 'host' and runtime.get('operation') == 'service_deploy':
                    log_dir = Path(directory).parent
                    for key, name, kind in (('service_start', 'start-service.log', 'deployment'),
                                            ('service_engine', 'vllm.log', 'service')):
                        row = source(tracker, attempt, key, name, kind, 'host', str(log_dir / name))
                        sync_file(tracker, execution, row)
                if backend == 'container' and runtime.get('kind') in {'train', 'evaluate', 'preprocess'}:
                    from .experiment_capture import container_json
                    entries = container_json(execution['container'], 'experiment_process.py',
                        {'directory': directory}, 'log-sources')
                    for entry in entries:
                        if entry['key'] == 'execution':
                            continue
                        row = source(tracker, attempt, entry['key'], entry['label'], entry['kind'], 'container', '')
                        sync_file(tracker, execution, row)
        except Exception as exc:
            capture_ok = False
            import logging
            logging.getLogger(__name__).warning('Experiment log capture failed for %s: %s', attempt,
                str(redact(str(exc)))[:500])
        try:
            flush(tracker, attempt)
        except Exception as exc:
            upload_ok = False
            import logging
            logging.getLogger(__name__).warning('Experiment log upload failed for %s: %s', attempt,
                str(redact(str(exc)))[:500])
        if capture_ok and upload_ok and execution['state'] in TERMINAL and (not execution['runtime'].get('experimentDirectory') or
                execution['runtime'].get('archiveStatus') == 'complete'):
            with tracker.connect() as db:
                pending = db.execute('''SELECT count(*) FROM experiment_log_pending p
                    JOIN experiment_log_local_source s ON s.source_id=p.source_id WHERE s.attempt_id=?''',
                    (attempt,)).fetchone()[0]
                unfinished = db.execute('''SELECT count(*) FROM experiment_log_local_source
                    WHERE attempt_id=? AND (status='streaming' OR sent_status IS NULL OR sent_status!=status)''',
                    (attempt,)).fetchone()[0]
                if not pending and not unfinished:
                    db.execute('UPDATE experiment_log_enabled SET completed=1 WHERE attempt_id=?', (attempt,))
