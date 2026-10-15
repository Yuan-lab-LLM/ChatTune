"""Small, dependency-free benchmark progress records (no prompts or credentials)."""
import json
import re
import time
from datetime import datetime, timezone


def error_text(error):
    text = str(error)
    text = re.sub(r'(?i)(bearer\s+)\S+', r'\1[REDACTED]', text)
    text = re.sub(r'(?i)((?:api[_-]?key|token|password|authorization)\s*[=:]\s*)[^\s,;]+', r'\1[REDACTED]', text)
    return text[:1000]


def log_event(event, **fields):
    print(json.dumps({'time': datetime.now(timezone.utc).isoformat(), 'event': event, **fields},
                     ensure_ascii=False), flush=True)


def sample_error(item):
    response = item.get('response') or item.get('model_answer') or ''
    return item.get('error') or item.get('evaluator_error') or (
        response if isinstance(response, str) and response.startswith('ERROR:') else None)


def log_sample(item, completed, total):
    error = sample_error(item)
    log_event('sample_finished', sample=item.get('index', item.get('id', item.get('_index'))),
              completed=completed, total=total,
              status='failed' if error else 'invalid' if item.get('invalid') else 'completed',
              **({'correct': item['is_correct']} if 'is_correct' in item else {}),
              **({'error': error_text(error)} if error else {}))


def log_finished(started, details, metrics=None):
    failed = sum(bool(sample_error(item)) for item in details)
    log_event('benchmark_finished', processed=len(details), succeeded=len(details) - failed,
              failed=failed, elapsed_seconds=round(time.monotonic() - started, 3), metrics=metrics or {})
