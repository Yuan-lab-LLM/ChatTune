"""Dependency-free execution journal deployed into an existing task container.

The supervisor records process exit independently of the Runtime service. It never
loads pickle checkpoints or treats a missing process as proof of success.
"""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
import base64
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import runpy
import signal
import shlex
import shutil
import subprocess
import sys
import time
import stat
import threading


BUSINESS_LOG_ROOTS = (
    '/home/workspace/log/daily_train', '/home/workspace/log/dpo_train',
    '/home/workspace/log/batch_train', '/home/workspace/log/pretrain',
)


def business_log_roots(metadata):
    script = metadata.get('scriptName', '')
    if script in {'single_model_evaluation_vpn', 'compare_between_models_vpn', 'ckpt_eval'}:
        return (*BUSINESS_LOG_ROOTS[:1], '/home/workspace/eval/daily_train')
    if script == 'dpo_train_launcher' or script == 'train_multinode_dpo_pipeline':
        return BUSINESS_LOG_ROOTS[1:2]
    if script.startswith('batch_train_pretrain_'):
        return BUSINESS_LOG_ROOTS[3:]
    if script.startswith(('batch_train_', 'train_multinode_sft_')):
        return BUSINESS_LOG_ROOTS[2:3]
    return ()


def business_log_kind(metadata):
    script = metadata.get('scriptName', '')
    if script in {'data_preprocessing', 'score_based_filtering'}:
        return 'data_processing'
    if script == 'dpo_train_launcher':
        return 'training'
    return 'evaluation_phase' if script in {'single_model_evaluation_vpn', 'compare_between_models_vpn', 'ckpt_eval'} or 'eval' in script or metadata.get('kind') == 'benchmark' else 'training_phase'


def business_source_kind(metadata, label):
    kind = business_log_kind(metadata)
    return 'evaluation_result' if kind == 'evaluation_phase' and label == 'results.log' else kind


def log_signature(path):
    """Remember the old tail so appended historical bytes are never copied."""
    with Path(path).open('rb') as stream:
        status = os.fstat(stream.fileno())
        size = status.st_size
        stream.seek(max(0, size - 256))
        return (status.st_dev, status.st_ino), size, stream.read(256)


class BusinessLogCapture:
    """Copy logs opened for writing by this process tree into its own directory.

    Shared filenames are not treated as ownership evidence. A writable
    descriptor or explicit write receipt qualifies; offsets exclude old appends.
    Keeping the opened inode also survives unlink/rename of the original file.
    """
    def __init__(self, root, metadata):
        self.root, self.metadata = Path(root), metadata
        self.previous, self.streams, self.sources = {}, {}, []
        self.receipts, self.receipt_offset, self.receipt_partial = {}, 0, b''
        self.log_roots = business_log_roots(metadata)
        self.declared = {str(item[3]) for item in additional_log_sources(metadata)}
        trainer = trainer_log_path(metadata)
        if trainer:
            self.declared.add(str(trainer))
        for log_root in self.log_roots:
            for path in Path(log_root).rglob('*.log'):
                try:
                    if path.is_file() and not path.is_symlink() and path.resolve() == path:
                        self.previous[str(path)] = log_signature(path)
                except OSError:
                    continue

    def register(self, target, descriptor, append):
        stream = descriptor.open('rb')
        output = None
        try:
            status = os.fstat(stream.fileno())
            identity = (status.st_dev, status.st_ino)
            if not stat.S_ISREG(status.st_mode) or identity in self.streams:
                stream.close()
                return
            offset = 0
            previous = self.previous.get(target)
            if previous and append:
                old_identity, size, tail = previous
                stream.seek(max(0, size - 256))
                if identity == old_identity and status.st_size >= size and stream.read(len(tail)) == tail:
                    offset = size
            stream.seek(offset)
            key = hashlib.sha256(f'{target}:{identity}'.encode()).hexdigest()
            destination = self.root / 'business-logs' / f'{key}.log'
            destination.parent.mkdir(exist_ok=True)
            output = destination.open('xb')
            destination.chmod(0o600)
            self.streams[identity] = (stream, output)
            self.sources.append({'key': 'business/' + key, 'label': Path(target).name[:200],
                                 'kind': business_source_kind(self.metadata, Path(target).name)})
        except Exception:
            stream.close()
            if output:
                output.close()
            raise

    def observe(self, pids):
        for pid in pids:
            for descriptor in Path(f'/proc/{pid}/fd').glob('[0-9]*'):
                try:
                    target = os.readlink(descriptor).removesuffix(' (deleted)')
                    path = Path(target)
                    if path.suffix != '.log' or target in self.declared or not any(path.is_relative_to(parent) for parent in self.log_roots):
                        continue
                    info = Path(f'/proc/{pid}/fdinfo/{descriptor.name}').read_text()
                    flags = int(re.search(r'^flags:\s*([0-7]+)', info, re.M)[1], 8)
                    if flags & 3 not in {os.O_WRONLY, os.O_RDWR}:
                        continue
                    self.register(target, descriptor, bool(flags & os.O_APPEND))
                except (OSError, ValueError, TypeError):
                    continue
        self.drain()

    def observe_receipts(self):
        """Recover short-lived writers from explicit output paths in task stdout.

        Legacy scripts log commands before running them. Reading that receipt
        also catches a stage which opens and closes its log between /proc polls.
        Input paths (cat/tail and general mentions of a log) never qualify.
        """
        launcher = self.root / 'execution.log'
        if launcher.is_file():
            with launcher.open('rb') as stream:
                stream.seek(self.receipt_offset)
                data = stream.read(1024 * 1024)
            self.receipt_offset += len(data)
            lines = (self.receipt_partial + data).split(b'\n')
            self.receipt_partial = lines.pop()
            for line in lines:
                text = line.decode('utf-8', errors='replace')
                pattern = r'(>>?\s*|--(?:output_log_file|log_file)\s+|log_dir is\s*:)\s*[\"\']?(/[^\s\"\'<>]+\.log)'
                for match in re.finditer(pattern, text):
                    path = Path(match[2])
                    if str(path) not in self.declared and '..' not in path.parts and any(path.is_relative_to(parent) for parent in self.log_roots):
                        self.receipts[str(path)] = not match[1].strip() == '>'
                for match in re.finditer(r'--save_dir\s+[\"\']?(/[^\s\"\'<>]+)', text):
                    path = Path(match[1]) / 'results.log'
                    if '..' not in path.parts and any(path.is_relative_to(parent) for parent in self.log_roots):
                        self.receipts[str(path)] = False
        for target, append in self.receipts.items():
            path = Path(target)
            try:
                if path.is_symlink() or path.resolve() != path:
                    continue
                self.register(target, path, append)
            except OSError:
                continue
        self.drain()

    def drain(self):
        for stream, output in self.streams.values():
            if os.fstat(stream.fileno()).st_size < stream.tell():
                stream.seek(0)
            # Bound each pass; retain the descriptor and resume on the next poll.
            for _ in range(32):
                data = stream.read(32768)
                if not data:
                    break
                output.write(data)
            output.flush()

    def close(self):
        for stream, output in self.streams.values():
            # Finite terminal boundary, even if another process later appends.
            remaining = max(0, os.fstat(stream.fileno()).st_size - stream.tell())
            while remaining:
                data = stream.read(min(32768, remaining))
                if not data:
                    break
                output.write(data)
                remaining -= len(data)
            stream.close()
            output.close()
        self.streams.clear()


SENSITIVE = re.compile(r"password|secret|authorization|api[_-]?key|(?:access|runtime|auth|hf(?:[_-]?hub)?|huggingface|hub)[_-]?token|^token$", re.I)


def redact(value):
    if isinstance(value, dict):
        return {str(k): "[REDACTED]" if SENSITIVE.search(str(k)) else redact(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        result, hide_next = [], False
        for item in value:
            if hide_next:
                result.append('[REDACTED]')
                hide_next = False
            else:
                result.append(redact(item))
                hide_next = isinstance(item, str) and item.startswith('--') and bool(SENSITIVE.search(item.lstrip('-'))) and '=' not in item
        return result
    if isinstance(value, str):
        value = re.sub(r"Bearer\s+[^\s\"']+", "Bearer [REDACTED]", value, flags=re.I)
        value = re.sub(r"(--(?:api[_-]?key|password|secret|authorization|(?:access|runtime|auth|hf(?:[_-]?hub)?|huggingface|hub)[_-]?token))\s+\S+",
                       r"\1 [REDACTED]", value, flags=re.I)
        value = re.sub(r"(https?://)[^\s/@]+:[^\s/@]+@", r"\1[REDACTED]@", value)
        value = re.sub(r'''(["']?(?:api[_-]?key|password|secret|authorization|(?:access|runtime|auth|hf(?:[_-]?hub)?|huggingface|hub)[_-]?token)["']?\s*[:=]\s*)(["'])(.*?)\2''',
                       r'\1"[REDACTED]"', value, flags=re.I)
        return re.sub(r"((?:api[_-]?key|password|secret|authorization|(?:access|runtime|auth|hf(?:[_-]?hub)?|huggingface|hub)[_-]?token)\s*[=:]\s*)[^\s,;\"']+",
                      r"\1[REDACTED]", value, flags=re.I)
    return value


def atomic_json(path, value):
    path = Path(path)
    if path.name == 'process.json':
        value = redact(value)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.chmod(0o600)
    os.replace(temporary, path)


def read_json(path, limit=8 * 1024 * 1024):
    path = Path(path)
    if path.stat().st_size > limit:
        raise ValueError('Experiment metadata exceeds size limit')
    return json.loads(path.read_text(encoding='utf-8'))


def process_identity(pid):
    try:
        # comm may itself contain spaces or parentheses.
        fields = Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')', 1)[1].split()
        return {'startTicks': fields[19], 'bootId': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    except (OSError, ValueError, IndexError):
        return None


def descendant_commands(pid, include_pids=False):
    processes = {}
    for directory in Path('/proc').glob('[0-9]*'):
        try:
            fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
            argv = [part.decode('utf-8', errors='replace') for part in (directory / 'cmdline').read_bytes().split(b'\0') if part]
            processes[int(directory.name)] = (int(fields[1]), argv)
        except (OSError, ValueError, IndexError):
            continue
    selected = {int(pid)}
    while True:
        children = {child for child, (parent, _) in processes.items() if parent in selected}
        if children.issubset(selected):
            break
        selected.update(children)
    if include_pids:
        return {child: processes[child][1] for child in sorted(selected) if child in processes}
    return [processes[child][1] for child in sorted(selected) if child in processes and processes[child][1]]


def output_paths(commands):
    paths = set()
    keys = {'output_dir', 'output-dir', 'export_dir', 'export-dir', 'save_dir',
            'trainer.default_local_dir', 'trainer.default_hdfs_dir'}
    for command in commands:
        for index, argument in enumerate(command):
            name, sep, value = argument.lstrip('-').partition('=')
            if name not in keys:
                continue
            if not sep:
                value = command[index + 1] if index + 1 < len(command) else ''
            if value.startswith('/') and '..' not in Path(value).parts:
                paths.add(value.rstrip('/'))
    return sorted(paths)


def entrypoint_fingerprints(argv):
    candidates = [shutil.which(argv[0]) or argv[0]] if argv else []
    candidates.extend(argument for argument in argv[1:8] if str(argument).endswith(('.py', '.sh')))
    result = {}
    for candidate in candidates:
        path = Path(candidate)
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                digest.update(chunk)
        result[str(path.resolve())] = digest.hexdigest()
    return result


def supervise(directory, argv):
    root = Path(directory)
    metadata = read_json(root / 'execution.json')
    # Also protects against accidental direct invocation after Runtime restart.
    claim = os.open(root / 'launch.claim', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(claim)
    journal = {'schemaVersion': 1, 'attemptId': metadata['attemptId'], 'pid': os.getpid(),
               'identity': process_identity(os.getpid()), 'startedAt': time.time(), 'exitCode': None,
               'commands': [argv], 'outputPaths': metadata.get('outputPaths', []),
               'entrypointFingerprints': entrypoint_fingerprints(argv),
               'python': sys.version, 'platform': platform.platform(), 'packages': {}}
    for name in ('torch', 'transformers', 'datasets', 'deepspeed', 'trl', 'llamafactory'):
        try:
            journal['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    atomic_json(root / 'process.json', journal)
    child = None
    stopped = []
    capture = BusinessLogCapture(root, metadata)
    stdout_thread = None
    compatibility_stack = ExitStack()

    def forward(signum, _frame):
        stopped.append(signum)
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, forward)
    try:
        if stopped:
            journal['exitCode'] = -stopped[-1]
            return 128 + stopped[-1]
        child_options = {}
        if metadata.get('scoringApiCompat'):
            adapter = root / 'scoring_api_compat.py'
            source = adapter.read_text(encoding='utf-8').encode('utf-8')
            if adapter.is_symlink() or hashlib.sha256(source).hexdigest() != metadata['scoringApiCompat']['sourceSha256']:
                raise ValueError('Scoring response compatibility code changed')
            argv = compatibility_stack.enter_context(runpy.run_path(str(adapter))['compatible_command'](argv))
            child_env = dict(os.environ)
            for key in ('NO_PROXY', 'no_proxy'):
                child_env[key] = ','.join(filter(None, (child_env.get(key), '127.0.0.1')))
            child_options['env'] = child_env
        business_stdout = metadata.get('businessStdout')
        if business_stdout:
            def copy_stdout():
                with (root / 'task-output.log').open('xb') as output:
                    (root / 'task-output.log').chmod(0o600)
                    while True:
                        data = child.stdout.read1(32768)
                        if not data:
                            break
                        output.write(data)
                        output.flush()
                        sys.stdout.buffer.write(data)
                        sys.stdout.buffer.flush()
                child.stdout.close()
            child = subprocess.Popen(argv, start_new_session=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **child_options)
            stdout_thread = threading.Thread(target=copy_stdout, daemon=True)
            stdout_thread.start()
        else:
            child = subprocess.Popen(argv, start_new_session=True, **child_options)
        if stopped and child.poll() is None:
            os.killpg(child.pid, stopped[-1])
        journal['childPid'] = child.pid
        seen = {json.dumps(argv)}
        while child.poll() is None:
            processes = descendant_commands(child.pid, include_pids=True)
            capture.observe(processes)
            capture.observe_receipts()
            journal['businessLogs'] = capture.sources
            for command in processes.values():
                encoded = json.dumps(command)
                if encoded not in seen and len(seen) < 256:
                    seen.add(encoded)
                    journal['commands'].append(command)
            journal['outputPaths'] = sorted(set(journal['outputPaths']) | set(output_paths(journal['commands'])))
            atomic_json(root / 'process.json', journal)
            time.sleep(0.2 if capture.log_roots else 1)
        journal['exitCode'] = child.returncode
    except Exception as exc:
        # An unexpected journal error while a child is alive must not imply termination.
        journal['error'] = str(exc)
        if child is None:
            journal['exitCode'] = 127
        elif child.poll() is not None:
            journal['exitCode'] = child.returncode
        else:
            journal['exitCode'] = child.wait()
    finally:
        if stdout_thread:
            stdout_thread.join(timeout=5)
        compatibility_stack.close()
        try:
            capture.observe_receipts()
            launcher = root / 'execution.log'
            final_size = launcher.stat().st_size if launcher.is_file() else 0
            while capture.receipt_offset < final_size:
                capture.observe_receipts()
        except OSError as exc:
            journal['logCaptureError'] = str(exc)
        finally:
            try:
                capture.close()
            except OSError as exc:
                journal['logCaptureError'] = str(exc)
        journal['businessLogs'] = capture.sources
        journal['signals'] = stopped
        journal['finishedAt'] = time.time()
        atomic_json(root / 'process.json', journal)
    return journal['exitCode'] if journal['exitCode'] >= 0 else 128 - journal['exitCode']


def prepare_dataset_output(descriptor, asset_source, purpose='Preprocessing resume'):
    """Create a new writable product; resumed bytes come only from a fixed version."""
    destination = Path(descriptor['path'])
    if not destination.is_absolute() or '..' in destination.parts or destination.resolve() != destination:
        raise ValueError('Dataset output cannot traverse symbolic links or parent directories')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()  # Never merge a new execution into an existing product.
    seed = descriptor.get('seed')
    if not seed:
        return
    namespace = {'__name__': '_experiment_asset_seed'}
    exec(compile(asset_source, 'experiment_assets.py', 'exec'), namespace)
    manifest = namespace['manifest']
    source = Path(seed['location']['path'])
    before = manifest(source)
    if not source.is_dir() or before['fingerprint'] != seed['fingerprint']:
        raise ValueError(f'{purpose} version is unavailable or changed')
    for item in before['files']:
        original = source / item['path']
        if original.is_symlink() or source.resolve() not in original.resolve().parents:
            raise ValueError(f'{purpose} version changed during copy')
        target = destination / item['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
        target.chmod(0o600)
    if manifest(source)['fingerprint'] != seed['fingerprint'] or manifest(destination)['fingerprint'] != seed['fingerprint']:
        raise ValueError(f'{purpose} copy failed content verification')


def prepare_checkpoint_input(descriptor, asset_source):
    """Make an execution copy whose adapter config binds the frozen base model."""
    base = descriptor['baseModel']
    base_path = str(descriptor['baseModelPath'])
    if base_path != base['location']['path']:
        raise ValueError('Checkpoint base model path must match its recorded version')
    namespace = {'__name__': '_experiment_checkpoint_input'}
    exec(compile(asset_source, 'experiment_assets.py', 'exec'), namespace)
    manifest = namespace['manifest']
    if manifest(base_path)['fingerprint'] != base['fingerprint']:
        raise ValueError('Checkpoint base model version is unavailable or changed')
    prepare_dataset_output(descriptor, asset_source, purpose='Checkpoint input')
    destination = Path(descriptor['path'])
    config_path = destination / 'adapter_config.json'
    original = config_path.read_bytes()
    config = json.loads(original)
    original_base = config.get('base_model_name_or_path') if isinstance(config, dict) else None
    if not isinstance(original_base, str) or not original_base.strip():
        raise ValueError('Checkpoint adapter config has no base model path')
    config['base_model_name_or_path'] = base_path
    # Write only the execution copy. atomic_json redacts secrets and is reserved
    # for audit records, not configuration actually consumed by the exporter.
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    captured = manifest(destination)
    if manifest(base_path)['fingerprint'] != base['fingerprint']:
        raise ValueError('Checkpoint base model changed during execution preparation')
    for item in captured['files']:
        target = destination / item['path']
        target.chmod(target.stat().st_mode & ~0o222)
    return {'kind': 'checkpoint_base_model_binding', 'checkpointVersionId': descriptor['seed']['id'],
            'baseModelVersionId': base['id'], 'sourceFingerprint': descriptor['seed']['fingerprint'],
            'baseModelFingerprint': base['fingerprint'], 'executionPath': str(destination),
            'executionFingerprint': captured['fingerprint'], 'baseModelPath': base_path,
            'originalBaseModelPath': original_base,
            'originalConfigSha256': hashlib.sha256(original).hexdigest(),
            'executionConfigSha256': hashlib.sha256(config_path.read_bytes()).hexdigest()}


def setup(request):
    owner, attempt = request['ownerId'], request['attemptId']
    if not all(re.fullmatch(r'[A-Za-z0-9_-]+', part) for part in (owner, attempt)):
        raise ValueError('Invalid execution identity')
    root = Path(f'/home/workspace/medflow/users/{owner}/experiments/{attempt}')
    if root.resolve() != root:
        raise ValueError('Execution directory cannot traverse symbolic links')
    root.mkdir(parents=True, exist_ok=True)
    replicas = []
    if request.get('allocationFile'):
        allocation = read_json(request['allocationFile'])
        nodes = allocation.get('nodes')
        if not isinstance(nodes, list) or not nodes or sum(bool(node.get('isMaster')) for node in nodes) != 1:
            raise ValueError('Invalid multi-node allocation')
        for node in nodes:
            if node.get('isMaster'):
                continue
            alias = str(node.get('sshAlias') or '')
            if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.@-]*', alias):
                raise ValueError('Invalid allocated SSH alias')
            for asset in request['metadata'].get('inputAssets', []):
                # The existing multi-node launch contract already requires original inputs
                # on every node. Freeze those inputs and reject differences before launch.
                operation = {'operation': 'snapshot', 'path': asset['metadata']['sourcePath'],
                             'assetKind': asset['kind'],
                             'versionRoot': str(Path(asset['location']['path']).parent)}
                completed = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', alias,
                    'python3 -c ' + shlex.quote(request['assetSource'])], input=json.dumps(operation),
                    capture_output=True, text=True, timeout=3600)
                if completed.returncode:
                    raise ValueError(f'Cannot freeze input on allocated node {node.get("nodeId")}: {completed.stderr[-1000:]}')
                captured = json.loads(completed.stdout)
                if captured['fingerprint'] != asset['fingerprint'] or captured['path'] != asset['location']['path']:
                    raise ValueError(f'Input version differs on allocated node {node.get("nodeId")}')
                replicas.append({'nodeId': node.get('nodeId'), 'assetVersionId': asset['id'],
                                 'fingerprint': captured['fingerprint'], 'path': captured['path']})
    metadata = request['metadata']
    metadata['attemptId'] = attempt
    sources = {'runner.py': request['runnerSource']}
    if request.get('evaluationSource'):
        sources['evaluation.py'] = request['evaluationSource']
        sources['experiment_assets.py'] = request['assetSource']
    scoring_compatibility = None
    if request.get('scoringCompatSource'):
        if metadata.get('scriptName') != 'score_based_filtering':
            raise ValueError('Scoring response compatibility is scoped to score-based filtering')
        sources['scoring_api_compat.py'] = request['scoringCompatSource']
        scoring_compatibility = {'kind': 'score_response_echo_compat',
            'sourceSha256': hashlib.sha256(request['scoringCompatSource'].encode('utf-8')).hexdigest()}
        metadata['scoringApiCompat'] = scoring_compatibility
    for filename, content in sources.items():
        target = root / filename
        if target.exists() and target.read_text(encoding='utf-8') != content:
            raise ValueError('Execution code is immutable')
        target.write_text(content, encoding='utf-8')
        target.chmod(0o600)
    for alias, target in request.get('aliases', {}).items():
        path = root / 'inputs' / alias
        if not re.fullmatch(r'[A-Za-z0-9_-]+', alias) or not Path(target).is_dir():
            raise ValueError('Invalid execution input alias')
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            if str(path.resolve()) != target:
                raise ValueError('Execution input alias is immutable')
        elif path.exists():
            raise ValueError('Execution input alias already exists')
        else:
            path.symlink_to(target, target_is_directory=True)
    if request.get('datasetOutput'):
        prepare_dataset_output(request['datasetOutput'], request['assetSource'])
    if request.get('checkpointInput'):
        descriptor = request['checkpointInput']
        if Path(descriptor['path']) != root / 'inputs' / f'checkpoint-{attempt}':
            raise ValueError('Checkpoint execution copy must remain in its attempt directory')
        metadata['inputTransforms'] = [prepare_checkpoint_input(descriptor, request['assetSource'])]
    target = root / 'execution.json'
    if target.exists() and read_json(target) != metadata:
        raise ValueError('Execution metadata is immutable')
    atomic_json(target, metadata)
    return {'directory': str(root), 'runner': str(root / 'runner.py'), 'replicas': replicas,
            'inputTransforms': metadata.get('inputTransforms', [])}


def complete_sharded_weights(directory):
    for prefix, extension, index_name in (
        ('model', 'safetensors', 'model.safetensors.index.json'),
        ('pytorch_model', 'bin', 'pytorch_model.bin.index.json'),
    ):
        pattern = re.compile(rf'{re.escape(prefix)}-(\d+)-of-(\d+)\.{extension}')
        shards = [path for path in directory.iterdir() if pattern.fullmatch(path.name)]
        if not shards:
            continue
        totals = {int(pattern.fullmatch(path.name).group(2)) for path in shards}
        if len(totals) != 1 or len(shards) != next(iter(totals)) or not (directory / index_name).is_file():
            return False
        total = next(iter(totals))
        if {int(pattern.fullmatch(path.name).group(1)) for path in shards} != set(range(1, total + 1)):
            return False
        try:
            index = read_json(directory / index_name)
            referenced = set(index['weight_map'].values())
        except (OSError, ValueError, KeyError, TypeError):
            return False
        if referenced != {path.name for path in shards}:
            return False
    return True


def inspect_outputs(paths):
    outputs, configs = [], {}
    seen = set()
    for raw in paths:
        root = Path(raw)
        if not root.is_dir() or root.is_symlink():
            continue
        directories = [root, *sorted(root.glob('checkpoint-*')), *sorted(root.glob('global_step_*'))]
        for directory in directories:
            if not directory.is_dir() or directory.is_symlink() or str(directory) in seen:
                continue
            seen.add(str(directory))
            adapter = (directory / 'adapter_config.json').is_file()
            weights = any(directory.glob('*.safetensors')) or any(directory.glob('pytorch_model*.bin')) or (directory / 'adapter_model.bin').is_file()
            if directory.name.startswith('global_step_'):
                weights = weights or any(directory.rglob('model_world_size*.pt')) or any(directory.rglob('model*.safetensors'))
            if weights and complete_sharded_weights(directory):
                kind = 'checkpoint' if re.fullmatch(r'(?:checkpoint-|global_step_)\d+', directory.name) else 'adapter' if adapter else 'model'
                outputs.append({'path': str(directory), 'kind': kind, 'role': kind})
            for name in ('training_args.json', 'training_args.yaml', 'trainer_state.json', '.hydra/config.yaml'):
                candidate = directory / name
                if candidate.is_file() and not candidate.is_symlink() and candidate.stat().st_size <= 1024 * 1024:
                    configs[str(candidate)] = candidate.read_text(encoding='utf-8', errors='replace')
    return outputs, configs



def inspect_filter_result(path):
    """Distinguish quality rejections from reported scoring failures; HTTP success alone is insufficient."""
    progress = path / 'score_progress.json'
    if not progress.is_file() or progress.is_symlink() or progress.stat().st_size > 1024 * 1024:
        return None
    try:
        receipt = read_json(progress)
    except (OSError, ValueError):
        return None
    if not isinstance(receipt, dict) or receipt.get('status') != 'completed' or receipt.get('error'):
        return None
    keys = ('total_items', 'processed_items', 'passed_items', 'rejected_items',
            'api_failed_items', 'invalid_items')
    counts = {key: receipt.get(key) for key in keys}
    if any(type(value) is not int or value < 0 for value in counts.values()):
        return None
    if (counts['processed_items'] != counts['total_items'] or counts['passed_items'] != 0
            or sum(counts[key] for key in ('passed_items', 'rejected_items', 'api_failed_items', 'invalid_items'))
                != counts['total_items']):
        return None
    if counts['api_failed_items'] > 0:
        return {'outcome': 'scoring_failed', **counts}
    if counts['invalid_items'] != 0:
        return None
    return {'outcome': 'no_qualifying_data', **counts}


def probe(directory, hints=None):
    root = Path(directory)
    metadata = read_json(root / 'execution.json')
    path = root / 'process.json'
    if not path.is_file():
        return {'attemptId': metadata['attemptId'], 'state': 'unknown', 'reason': 'no_process_journal'}
    process = read_json(path)
    if process.get('attemptId') != metadata['attemptId']:
        raise ValueError('Process journal belongs to another execution')
    result = {'attemptId': metadata['attemptId'], 'process': process}
    if process.get('exitCode') is None:
        identity = process_identity(process.get('pid'))
        result['state'] = 'running' if identity and identity == process.get('identity') else 'unknown'
        return result
    result['state'] = ('cancelled' if process.get('signals') else
                       'succeeded' if process['exitCode'] == 0 else 'failed')
    hints = hints or {}
    paths = list(process.get('outputPaths', []))
    if hints.get('output_dir'):
        paths.append(hints['output_dir'])
    result['outputs'], result['configurationFiles'] = inspect_outputs(paths)
    for transformation in metadata.get('inputTransforms', []):
        config = Path(transformation['executionPath']) / 'adapter_config.json'
        if config.is_file() and not config.is_symlink() and config.stat().st_size <= 1024 * 1024:
            result['configurationFiles'][str(config)] = config.read_text(encoding='utf-8')
    filter_reports = []
    for output in metadata.get('declaredOutputs', []):
        path = Path(output['path'])
        if path.exists() and not path.is_symlink():
            metadata_names = {'dataset_info.json', 'preprocessing_audit.json', 'preprocessing_summary.json',
                              'score_audit.json', 'score_progress.json', 'score_summary.json'}
            has_data = output['kind'] != 'dataset' or any(
                candidate.is_file() and not candidate.is_symlink() and candidate.name not in metadata_names and
                candidate.suffix.lower() in {'.json', '.jsonl', '.parquet', '.txt'} and candidate.stat().st_size > 0
                for candidate in (path.rglob('*') if path.is_dir() else [path]))
            if has_data:
                result['outputs'].append(output)
            elif (metadata.get('scriptName') == 'score_based_filtering' and output['kind'] == 'dataset'
                    and output.get('role') == 'processed_dataset' and result['state'] == 'succeeded'
                    and path.is_dir()):
                receipt = inspect_filter_result(path)
                if receipt:
                    result['dataProcessing'] = receipt
                    if receipt['outcome'] == 'scoring_failed':
                        result['state'] = 'failed'
                    for name in ('score_progress.json', 'score_summary.json', 'score_audit.json',
                                 'score_summary.log', 'score_audit.log', 'score_filter_process.log'):
                        report = path / name
                        if report.is_file() and not report.is_symlink():
                            filter_reports.append(str(report))
    reports = filter_reports
    report_paths = list(metadata.get('reportPaths', []))
    hinted_report = hints.get('report_path')
    if hinted_report and Path(hinted_report) not in {Path(source) for source in report_paths}:
        # A monitor can discover a shared daily report from another execution.
        # Neither existence nor a recent mtime proves that this attempt wrote it.
        candidate = Path(hinted_report)
        if candidate.is_file() and not candidate.is_symlink():
            result['unattributedReports'] = [str(candidate)]
    for source in report_paths:
        candidate = Path(source)
        if candidate.is_file() and not candidate.is_symlink():
            if candidate.name == 'manifest.json':
                result['evaluationManifest'] = read_json(candidate)
                for exported in result['evaluationManifest'].get('modelArtifacts', []):
                    fingerprint = exported.get('fingerprint', '')
                    expected_root = metadata.get('modelVersionRoot')
                    if (not expected_root or not re.fullmatch(r'[a-f0-9]{64}', fingerprint) or
                        Path(exported['path']) != Path(expected_root) / fingerprint or
                        exported.get('sourcePath') not in metadata.get('outputPaths', [])):
                        raise ValueError('Export receipt points outside the declared model output/version directories')
                    descriptor = {key: exported[key] for key in ('path', 'sourcePath', 'fingerprint', 'inputPaths')}
                    descriptor.update(kind='model', role='merged_model')
                    result['outputs'] = [item for item in result['outputs'] if item['path'] != exported['sourcePath']]
                    result['outputs'].append(descriptor)
                for phase in result['evaluationManifest'].get('executions', []):
                    log = Path(phase.get('logPath') or '')
                    if log.is_file() and not log.is_symlink():
                        reports.append(str(log))
                for scores in result['evaluationManifest'].get('results', {}).values():
                    for score in scores.values():
                        results_log = Path(score.get('resultsLog') or '')
                        for raw in (results_log, results_log.with_name('results.json') if results_log.name else results_log,
                                    Path(score.get('commandLog') or '')):
                            if raw.is_file() and not raw.is_symlink():
                                reports.append(str(raw))
            reports.append(str(candidate))
    result['reports'] = sorted(set(reports))
    return result


def trainer_log_path(metadata):
    """Resolve the trainer stdout log from this attempt's declared model output."""
    script = metadata.get('scriptName')
    layouts = {
        'batch_train_lora': ('batch_train', 'model_lora_'),
        'batch_train_full': ('batch_train', 'model_full_'),
        'batch_train_pretrain_lora': ('pretrain', 'model_pretrain_lora_'),
        'batch_train_pretrain_full': ('pretrain', 'model_pretrain_full_'),
    }
    if script not in layouts:
        return None
    root, prefix = layouts[script]
    attempt = metadata['attemptId']
    if not re.fullmatch(r'[A-Za-z0-9_-]+', attempt):
        return None
    expected = PurePosixPath('/home/workspace/models') / root / f'{prefix}data-{attempt}'
    if str(expected) not in metadata.get('outputPaths', []):
        return None
    return Path(str(PurePosixPath('/home/workspace/log') / root / f'{expected.name}-train-log.log'))


def additional_log_sources(metadata):
    """Only expose files whose names are fixed by this execution's inputs."""
    attempt = metadata.get('attemptId', '')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', attempt):
        return []
    script = metadata.get('scriptName')
    result = []
    trainer = trainer_log_path(metadata)
    if trainer is not None:
        name = trainer.name.removesuffix('-train-log.log')
        phase = 'export' if script == 'batch_train_pretrain_full' else 'merge'
        result.append((phase, f'{phase}.log', 'training_phase', trainer.parent / f'{name}-{phase}-log.log', False))
        working_directory = metadata.get('workingDirectory')
        if isinstance(working_directory, str) and working_directory.startswith('/') and '..' not in PurePosixPath(working_directory).parts:
            result.append(('dataset_analysis', '数据统计日志', 'training_phase',
                           PurePosixPath(working_directory) / f'data-{attempt}-analysis-log.log', False))
    if script in {'train_multinode_sft_pipeline', 'train_multinode_sft_full_pipeline'}:
        outputs = metadata.get('outputPaths') or []
        if any(PurePosixPath(output).name == attempt for output in outputs):
            root = PurePosixPath('/home/workspace/log/batch_train')
            result.append(('trainer', '训练日志', 'training', root / f'{attempt}-train-log.log', True))
            result.append(('merge', 'merge.log', 'training_phase', root / f'{attempt}-merge-log.log', False))
    elif script == 'train_multinode_dpo_pipeline':
        outputs = metadata.get('outputPaths') or []
        if any(PurePosixPath(output).name == attempt for output in outputs):
            result.append(('trainer', '训练日志', 'training',
                           PurePosixPath('/home/workspace/log/dpo_train') / f'{attempt}-dpo-train.log', True))
    elif script in {'data_preprocessing', 'score_based_filtering'}:
        outputs = metadata.get('declaredOutputs') or []
        for item in outputs:
            path = item.get('path') if isinstance(item, dict) else None
            if isinstance(path, str) and path.startswith('/') and '..' not in PurePosixPath(path).parts and (path.endswith('-' + attempt) or metadata.get('datasetOutputPath') == path):
                name = 'score_filter_process.log' if script == 'score_based_filtering' else 'preprocessing_audit.log'
                result.append(('processing', name, 'data_processing', PurePosixPath(path) / name, False))
                if script == 'data_preprocessing':
                    result.append(('summary', 'preprocessing_summary.log', 'data_processing',
                                   PurePosixPath(path) / 'preprocessing_summary.log', False))
                break
    return result


def log_source_path(directory, key):
    root = Path(directory)
    if root.resolve() != root or root.is_symlink():
        raise ValueError('Experiment log directory changed')
    metadata = read_json(root / 'execution.json')
    attempt = metadata['attemptId']
    if key == 'execution':
        return root / 'execution.log'
    if key == 'task-output' and metadata.get('businessStdout'):
        return root / 'task-output.log'
    if re.fullmatch(r'business/[a-f0-9]{64}', key):
        journal = read_json(root / 'process.json')
        if journal.get('attemptId') == attempt and any(item.get('key') == key for item in journal.get('businessLogs', [])):
            path = root / 'business-logs' / (key.split('/')[1] + '.log')
            if path.parent.is_symlink() or (path.parent.exists() and path.parent.resolve() != path.parent):
                raise ValueError('Business log directory changed')
            return path
    if key == 'trainer':
        path = trainer_log_path(metadata)
        if path is not None:
            if path.parent.is_symlink() or (path.parent.exists() and path.parent.resolve() != path.parent):
                raise ValueError('Training log directory changed')
            return path
    for source_key, _, _, declared, _ in additional_log_sources(metadata):
        if key == source_key:
            path = Path(str(declared))
            if path.parent.is_symlink() or (path.parent.exists() and path.parent.resolve() != path.parent):
                raise ValueError('Experiment log directory changed')
            return path
    if key.startswith('eval/') and re.fullmatch(r'[A-Za-z0-9_.-]+\.log', key[5:]):
        eval_root = Path('/home/workspace/log/llamafactory_eval') / attempt
        if eval_root.is_symlink() or (eval_root.exists() and eval_root.resolve() != eval_root):
            raise ValueError('Evaluation log directory changed')
        return eval_root / key[5:]
    match = re.fullmatch(r'eval-result/(base|checkpoint|model_fir|model_sec)/(ceval|cmbsingle|cmmlu|medmcqa|medqa|mmlu)', key)
    if match and metadata.get('scriptName') in {'generic_single_model_eval', 'generic_compare_models_eval', 'generic_ckpt_eval'}:
        result_root = Path('/home/workspace/eval/llamafactory') / attempt
        path = result_root / 'models' / match[1] / match[2] / 'results.log'
        if any(parent.is_symlink() for parent in (result_root, path.parent.parent, path.parent)):
            raise ValueError('Evaluation result log directory changed')
        return path
    raise ValueError('Unrecognized experiment log source')


def log_sources(directory):
    root = Path(directory)
    metadata = read_json(root / 'execution.json')
    result = [{'key': 'execution', 'label': '执行日志', 'kind': 'execution'}]
    if metadata.get('businessStdout'):
        result.append({'key': 'task-output', **metadata['businessStdout']})
    if (root / 'process.json').is_file():
        journal = read_json(root / 'process.json')
        if journal.get('attemptId') != metadata['attemptId']:
            raise ValueError('Business log journal belongs to another attempt')
        for item in journal.get('businessLogs', []):
            if re.fullmatch(r'business/[a-f0-9]{64}', item.get('key', '')):
                result.append({'key': item['key'], 'label': item['label'], 'kind': business_source_kind(metadata, item['label'])})
    if trainer_log_path(metadata) is not None:
        result.append({'key': 'trainer', 'label': '训练日志', 'kind': 'training'})
    for key, label, kind, declared, required in additional_log_sources(metadata):
        path = Path(str(declared))
        if required or (path.is_file() and not path.is_symlink()):
            result.append({'key': key, 'label': label, 'kind': kind})
    eval_root = Path('/home/workspace/log/llamafactory_eval') / metadata['attemptId']
    if eval_root.is_dir() and not eval_root.is_symlink() and eval_root.resolve() == eval_root:
        for path in sorted(eval_root.glob('*.log')):
            if path.is_file() and not path.is_symlink() and re.fullmatch(r'[A-Za-z0-9_.-]+\.log', path.name):
                result.append({'key': 'eval/' + path.name, 'label': path.name, 'kind': 'evaluation_phase'})
    if metadata.get('scriptName') in {'generic_single_model_eval', 'generic_compare_models_eval', 'generic_ckpt_eval'}:
        result_root = Path('/home/workspace/eval/llamafactory') / metadata['attemptId'] / 'models'
        for role in ('base', 'checkpoint', 'model_fir', 'model_sec'):
            for dataset in ('ceval', 'cmbsingle', 'cmmlu', 'medmcqa', 'medqa', 'mmlu'):
                path = result_root / role / dataset / 'results.log'
                if path.is_file() and not any(parent.is_symlink() for parent in (result_root, path.parent.parent, path.parent, path)):
                    result.append({'key': f'eval-result/{role}/{dataset}',
                                   'label': f'{role}/{dataset}/results.log', 'kind': 'evaluation_result'})
    return result


def log_read(directory, key, offset, limit):
    path = log_source_path(directory, key)
    if not path.is_file() or path.is_symlink():
        return {'exists': False}
    with path.open('rb') as stream:
        stat = os.fstat(stream.fileno())
        if offset > stat.st_size:
            return {'exists': True, 'size': stat.st_size, 'identity': f'{stat.st_dev}:{stat.st_ino}', 'data': ''}
        stream.seek(offset)
        data = stream.read(min(limit, 32768))
    return {'exists': True, 'size': stat.st_size, 'identity': f'{stat.st_dev}:{stat.st_ino}',
            'data': base64.b64encode(data).decode('ascii')}


if __name__ == '__main__':
    if sys.argv[1] == 'run':
        raise SystemExit(supervise(sys.argv[2], sys.argv[3:]))
    request = json.load(sys.stdin)
    if sys.argv[1] == 'setup':
        result = setup(request)
    elif sys.argv[1] == 'read-json':
        result = read_json(request['path'])
    elif sys.argv[1] == 'log-sources':
        result = log_sources(request['directory'])
    elif sys.argv[1] == 'log-read':
        result = log_read(request['directory'], request['key'], int(request['offset']), int(request['limit']))
    else:
        result = probe(request['directory'], request.get('hints'))
    print(json.dumps(result, ensure_ascii=False))
