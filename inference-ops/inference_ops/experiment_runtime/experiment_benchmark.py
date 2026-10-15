"""Bounded sample previews and real subset statistics from a frozen benchmark report."""
import json
import math
from pathlib import Path

PREVIEW_LIMIT = 200


def failed(item):
    response = item.get('response') or item.get('model_answer') or ''
    return bool(item.get('error') or item.get('evaluator_error') or
                isinstance(response, str) and response.startswith('ERROR:'))


def text(value, limit=2000):
    if value is None:
        return ''
    return (value if isinstance(value, str) else json.dumps(value, ensure_ascii=False))[:limit]


def judgment(item):
    if item.get('official_pending') or item.get('executed') is False and 'passed' in item:
        return None
    return next((item[key] for key in ('is_correct', 'exact_match', 'passed', 'strict_follow_all')
                 if type(item.get(key)) is bool), None)


def subset(item, dataset):
    meta = item.get('meta') if isinstance(item.get('meta'), dict) else {}
    for key in ('subset', 'subject', 'category', 'task', 'question_type'):
        value = item.get(key) or meta.get(key)
        if isinstance(value, str) and value.strip():
            return value[:200]
    # These ranges are the runner's explicit scoring sections, not inferred subjects.
    index = item.get('index')
    if dataset == '2024.json' and type(index) is int:
        for start, end, name in ((1, 40, 'A (1–40)'), (41, 115, 'A (41–115)'),
                                 (116, 135, 'B (116–135)'), (136, 165, 'X (136–165)')):
            if start <= index <= end:
                return name
    return ''


def preview(data, dataset=''):
    details = data.get('details')
    if not isinstance(details, list) or not all(isinstance(item, dict) for item in details):
        return None
    groups = {}
    groups_truncated = False
    samples, errors = [], []
    statuses = {'correct': 0, 'incorrect': 0, 'failed': 0, 'invalid': 0, 'unscored': 0}
    for position, item in enumerate(details, 1):
        correct = judgment(item)
        status = ('failed' if failed(item) else 'invalid' if item.get('invalid') else
                  'correct' if correct is True else 'incorrect' if correct is False else 'unscored')
        statuses[status] += 1
        name = subset(item, dataset)
        if name and (name in groups or len(groups) < 200):
            group = groups.setdefault(name, {'name': name, 'samples': 0, 'correct': 0, 'scored': 0, 'failed': 0})
            group['samples'] += 1
            group['failed'] += status == 'failed'
            group['correct'] += status == 'correct'
            group['scored'] += correct is not None
        elif name:
            groups_truncated = True
        numeric = {key: value for key in ('score', 'f1', 'precision', 'recall', 'latency_seconds')
                   if type(value := item.get(key)) in (int, float) and math.isfinite(value)}
        target = errors if status == 'failed' else samples
        if len(target) >= PREVIEW_LIMIT:
            continue
        index = item.get('index')
        target.append({'index': index if type(index) is int and index > 0 else position, 'id': text(item.get('id'), 200), 'subset': name,
            'question': text(item.get('question')), 'expected': text(next((item[key] for key in
                ('ground_truth', 'answer', 'references', 'reference') if item.get(key) is not None), None)),
            'predicted': text(item.get('predicted', item.get('prediction', item.get('model_answer')))),
            'response': text(item.get('response', item.get('answer')), 3000),
            'error': text(item.get('error') or ('Evaluator error' if item.get('evaluator_error') else '') or
                          (item.get('response') if failed(item) else '')), 'status': status, 'metrics': numeric})
    # Preserve failures in a bounded preview; display order remains the original sample order.
    selected = (errors + samples)[:PREVIEW_LIMIT]
    selected.sort(key=lambda item: item['index'])
    return {'schemaVersion': 1, 'total': len(details), 'previewLimit': PREVIEW_LIMIT, 'counts': statuses,
            'samples': selected, 'groups': list(groups.values()), 'groupsTruncated': groups_truncated,
            'truncated': len(details) > len(selected)}


def read_report(report, dataset=''):
    root = Path(report['location']['path'])
    files = report.get('manifest', {}).get('files', [])
    is_json = len(files) == 1 and str(files[0]['path']).endswith('.json')
    if not is_json and (not files or not all(str(file['path']).endswith('.jsonl') for file in files)):
        return None
    import hashlib
    rows = []
    for file in files:
        name = Path(file['path'])
        if name.is_absolute() or '..' in name.parts:
            raise ValueError('Invalid archived benchmark report path')
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
            raise ValueError('Archived report must stay inside its version')
        # Report artifacts are immutable; verify bytes before historical backfill too.
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != file['sha256']:
            raise ValueError('Archived benchmark report fingerprint changed')
        with path.open(encoding='utf-8') as stream:
            if is_json:
                return preview(json.load(stream), dataset)
            for line in stream:
                if line.strip():
                    item = json.loads(line)
                    rows.append({**item, 'response': item.get('answer'), 'answer': None,
                                 'subset': item.get('subset') or name.stem})
    return preview({'details': rows}, dataset)
