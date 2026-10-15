"""Versioned adapter for the existing LLaMA-Factory built-in evaluation engine.

The output manifest keeps the format consumed by runlocal_evaluate_monitor.
Only orchestration lives here; scoring remains in the installed evaluator CLI.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

DATASETS = {'ceval': ('ceval_validation', 'zh'), 'cmbsingle': ('cmbsingle_test', 'single_6'),
            'cmmlu': ('cmmlu_test', 'zh'), 'medmcqa': ('medmcqa_test', 'en'),
            'medqa': ('medqa_test', 'single_5'), 'mmlu': ('mmlu_test', 'en')}
CLI = '/usr/local/insinfersystem/llamafactory-cli'


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def freeze_export(path):
    namespace = {'__name__': '_experiment_export_assets'}
    helper = Path(__file__).with_name('experiment_assets.py')
    exec(compile(helper.read_text(encoding='utf-8'), str(helper), 'exec'), namespace)
    return namespace['snapshot'](path, os.environ['MEDFLOW_EVAL_MODEL_VERSION_ROOT'])


def parse_scores(path):
    section, subjects, summary = None, {}, {}
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        line = line.strip()
        if line.lower() in ('subject scores:', 'summary scores:'):
            section = subjects if line.lower().startswith('subject') else summary
        elif section is not None and ':' in line:
            key, raw = line.rsplit(':', 1)
            try:
                value = float(raw.strip())
                if math.isfinite(value):
                    section[key.strip()] = value
            except ValueError:
                pass
    average = next((value for name, value in summary.items() if name.lower() == 'average'), None)
    if average is None:
        raise ValueError('Evaluator result has no average score')
    return {'average': average, 'subjectScores': subjects, 'summaryScores': summary, 'resultsLog': str(path)}


def sample_counts(path):
    # LLaMA-Factory results.json: {subject: {sample_index: prediction}}.
    # A different evaluator format remains unknown rather than guessing a count.
    if not path.is_file() or path.stat().st_size > 128 * 1024 * 1024:
        return None
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (ValueError, OSError):
        return None
    if not isinstance(value, dict) or not value:
        return None
    if not all(isinstance(rows, dict) and all(str(index).isdigit() and isinstance(prediction, str)
               for index, prediction in rows.items()) for rows in value.values()):
        return None
    return {subject: len(rows) for subject, rows in value.items()}


def run(mode, args, *, result_base='/home/workspace/eval/llamafactory', log_base='/home/workspace/log/llamafactory_eval'):
    datasets = list(dict.fromkeys(name.strip().lower() for name in args.eval_datasets.split(',') if name.strip()))
    if not datasets or any(name not in DATASETS for name in datasets):
        raise ValueError('Unsupported evaluation dataset')
    task_dir = Path(os.environ['MEDFLOW_EVAL_TASK_DIR'])
    result_root = Path(result_base) / args.run_id
    log_root = Path(log_base) / args.run_id
    result_root.mkdir(parents=True, exist_ok=False)
    log_root.mkdir(parents=True, exist_ok=False)
    manifest = {'schemaVersion': 1, 'runId': args.run_id, 'evaluationScope': 'llamafactoryBuiltin',
        'evalType': mode, 'status': 'running', 'datasets': datasets, 'models': [], 'results': {},
        'currentModel': None, 'currentDataset': None, 'completedEvaluations': 0, 'totalEvaluations': 0,
        'scoreDeltas': {}, 'macroAverage': {}, 'aggregation': 'equal-weight macro average across selected datasets',
        'resultRoot': str(result_root), 'logRoot': str(log_root), 'reportPath': str(result_root / 'summary.txt'),
        'artifacts': {'manifest': str(result_root / 'manifest.json'), 'summaryJson': str(result_root / 'summary.json'),
                      'summaryText': str(result_root / 'summary.txt')},
        'executions': [], 'modelArtifacts': [],
        'error': None, 'startedAt': time.time(), 'finishedAt': None}

    def save():
        write_json(result_root / 'manifest.json', manifest)

    def execute(command, logfile):
        phase = {'operation': command[1], 'command': command, 'logPath': str(logfile),
                 'startedAt': time.time(), 'finishedAt': None, 'exitCode': None}
        manifest['executions'].append(phase)
        save()  # Durable even when the child fails before returning a result.
        try:
            with logfile.open('w', encoding='utf-8') as log:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            phase['exitCode'] = completed.returncode
            if completed.returncode:
                raise RuntimeError(f'Evaluator exited {completed.returncode}; log: {logfile}')
            return phase
        except Exception as exc:
            phase['error'] = str(exc)
            raise
        finally:
            phase['finishedAt'] = time.time()
            save()

    save()
    try:
        for name in datasets:
            for filename in (f'{name}.py', f'{name}.zip', 'mapping.json'):
                if not (task_dir / name / filename).is_file():
                    raise ValueError(f'Missing pinned evaluation input: {name}/{filename}')
        manifest['experimentProtocol'] = {'schemaVersion': 1, 'adapterSha256': checksum(__file__),
            'evaluatorSha256': checksum(CLI), 'scope': 'llamafactoryBuiltin',
            'containerImageId': os.environ.get('MEDFLOW_EVAL_IMAGE_ID'),
            'tasks': {name: {'task': DATASETS[name][0], 'language': DATASETS[name][1]} for name in sorted(datasets)},
            'template': 'fewshot', 'nShot': 5, 'batchSize': 1, 'seed': 42,
            'aggregation': manifest['aggregation'], 'inputBinding': 'immutable_snapshots'}
        with open(CLI, 'rb') as executable:
            compiled_engine = executable.read(4) == b'\x7fELF'
        # A generic Python entry script alone does not identify its imported evaluator code.
        manifest['experimentProtocolComplete'] = compiled_engine and bool(os.environ.get('MEDFLOW_EVAL_IMAGE_ID'))
        if mode == 'ckpt':
            base = os.environ['MEDFLOW_EVAL_BASE_MODEL']
            merged = str(result_root / 'merged_checkpoint')
            manifest['stage'] = 'exporting'
            save()
            exported = execute([CLI, 'export', '--model_name_or_path', base, '--adapter_name_or_path', args.CKPT_PATH,
                     '--template', args.TEM, '--finetuning_type', 'lora', '--export_dir', merged,
                     '--export_legacy_format', 'false'], log_root / 'checkpoint_export.log')
            frozen = freeze_export(merged)
            manifest['modelArtifacts'].append({'path': frozen['path'], 'sourcePath': merged,
                'fingerprint': frozen['fingerprint'], 'inputPaths': [base, args.CKPT_PATH]})
            exported['outputFingerprint'] = frozen['fingerprint']
            exported['outputSnapshotPath'] = frozen['path']
            save()
            models = [{'role': 'base', 'path': base}, {'role': 'checkpoint', 'path': frozen['path']}]
        else:
            models = [{'role': 'model_fir', 'path': args.model_fir}]
            if mode == 'compare_between_models':
                models.append({'role': 'model_sec', 'path': args.model_sec})
        manifest['models'] = models
        manifest['totalEvaluations'] = len(models) * len(datasets)
        manifest['results'] = {model['role']: {} for model in models}
        for model in models:
            role = model['role']
            for dataset in datasets:
                manifest.update(stage='evaluating', currentModel=role, currentDataset=dataset)
                save()
                output = result_root / 'models' / role / dataset
                output.parent.mkdir(parents=True, exist_ok=True)
                task, language = DATASETS[dataset]
                command = [CLI, 'eval', '--model_name_or_path', model['path'], '--task', task,
                    '--task_dir', str(task_dir), '--template', 'fewshot', '--lang', language,
                    '--n_shot', '5', '--save_dir', str(output), '--batch_size', '1', '--seed', '42',
                    '--trust_remote_code', 'true']
                logfile = log_root / f'{role}-{dataset}.log'
                execute(command, logfile)
                scores = parse_scores(output / 'results.log')
                counts = sample_counts(output / 'results.json')
                scores.update(task=task, lang=language, command=command, commandLog=str(logfile),
                              subjectSampleCounts=counts, sampleCount=sum(counts.values()) if counts is not None else None)
                manifest['results'][role][dataset] = scores
                manifest['completedEvaluations'] += 1
                save()
        manifest.update(status='finished', stage='finished')
    except Exception as exc:
        manifest.update(status='failed', stage='failed', error=str(exc))
        raise
    finally:
        manifest['finishedAt'] = time.time()
        for role, scores in manifest['results'].items():
            if scores:
                manifest['macroAverage'][role] = sum(item['average'] for item in scores.values()) / len(scores)
        roles = list(manifest['results'])
        if len(roles) == 2:
            first, second = (manifest['results'][role] for role in roles)
            manifest['scoreDeltas'] = {name: second[name]['average'] - first[name]['average'] for name in datasets if name in first and name in second}
        write_json(result_root / 'summary.json', manifest)
        lines = [f'Run ID: {args.run_id}', f"Status: {manifest['status']}"]
        for role, scores in manifest['results'].items():
            lines += [f"{role} / {name}: {item['average']:.4f}" for name, item in scores.items()]
        if manifest['error']:
            lines.append(manifest['error'])
        (result_root / 'summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=['single_model', 'compare_between_models', 'ckpt'])
    parser.add_argument('--eval_datasets', '--eval-datasets', required=True)
    parser.add_argument('--run_id', '--run-id', required=True)
    parser.add_argument('--model_fir', '--model-fir', default='')
    parser.add_argument('--model_sec', '--model-sec', default='')
    parser.add_argument('--CKPT_PATH', '--ckpt-path', default='')
    parser.add_argument('--TEM', '--tem', default='qwen3')
    args = parser.parse_args()
    run(args.mode, args)
