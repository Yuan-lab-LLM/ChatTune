"""Run with Python's unittest; all files live in self-cleaning temporary directories."""
import ast
import concurrent.futures
import errno
from datetime import datetime
import importlib.util
import json
import os
import shutil
from pathlib import Path
import tempfile
import sys
import types
import unittest
from unittest.mock import patch


def load(name):
    package_name = "experiment_test_utils"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(Path(__file__).parents[1] / "utils")]
        sys.modules[package_name] = package
    full_name = f"{package_name}.{name}"
    spec = importlib.util.spec_from_file_location(full_name, Path(__file__).parents[1] / "utils" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


tracker_module = load("experiment_tracker")
assets = load("experiment_assets")
capture = load("experiment_capture")
launch = load("experiment_launch")
collector = sys.modules['experiment_test_utils.experiment_collector']
dataset_naming = sys.modules['experiment_test_utils.dataset_naming']
process_module = load('experiment_process')
evaluation_module = load('experiment_evaluation')
upload_module = load('experiment_upload')
publish_module = load('experiment_publish')
publish_files = load('experiment_publish_files')
workflow_module = load('workflow_manager')
native_spec = importlib.util.spec_from_file_location('experiment_native_test',
    Path(__file__).parents[2] / 'inference-ops' / 'inference_ops' / 'experiment_records.py')
native = importlib.util.module_from_spec(native_spec)
native_spec.loader.exec_module(native)


class DatasetNamingTests(unittest.TestCase):
    moment = datetime(2026, 10, 14, 14, 30, 25, 123000)

    def test_legacy_long_execution_suffix_is_removed_before_truncating_source(self):
        source = 'medical-example_sft_20261013-143025-123-01760300000000000000-attempt-12345678-abcd'
        result = dataset_naming.build_dataset_output_name(source, 'sft', filtered=True, created_at=self.moment)
        self.assertEqual(result, 'medical-example_filtered_261014-143025-123')

    def test_processing_and_filtering_do_not_accumulate_suffixes(self):
        source = 'medical-example'
        for index in range(5):
            provisional = dataset_naming.build_dataset_output_name(source, 'sft', filtered=True, created_at=self.moment)
            source = dataset_naming.build_dataset_execution_name(provisional, f'attempt-{index}', created_at=self.moment)
            self.assertRegex(source, r'^medical-example_filtered_261014-143025-[a-f0-9]{10}$')

    def test_same_time_attempts_have_distinct_short_names(self):
        provisional = dataset_naming.build_dataset_output_name('source' * 20, 'sft', created_at=self.moment)
        products = [dataset_naming.build_dataset_execution_name(provisional, attempt, created_at=self.moment)
                    for attempt in ('attempt-one', 'attempt-two')]
        self.assertNotEqual(*products)
        for product in products:
            self.assertLessEqual(len(product), 57)
            self.assertEqual(product.count('261014-143025'), 1)

    def test_shared_date_names_remain_short_and_sort_by_time(self):
        early = dataset_naming.build_dataset_execution_name('20261014', 'attempt-one', created_at=self.moment)
        late = dataset_naming.build_dataset_execution_name('20261014', 'attempt-two',
                                                         created_at=self.moment.replace(second=26))
        self.assertRegex(early, r'^20261014-143025-[a-f0-9]{10}$')
        self.assertLess(early, late)


class FakeStudio:
    def __init__(self):
        self.requests = {}
        self.events = {}
        self.offline = False
        self.lose_response = False

    def __call__(self, procedure, payload):
        if self.offline:
            raise ConnectionError("offline")
        if procedure == "prepareExperimentTask":
            key = payload["requestKey"]
            handle = self.requests.setdefault(key, {"experimentId": "exp-1", "taskId": f"task-{key}",
                "attemptId": f"attempt-{key}", "status": "prepared", "attemptNumber": 1})
            if self.lose_response:
                self.lose_response = False
                raise TimeoutError("lost prepare response")
            return handle
        for event in payload["events"]:
            self.events[event["eventId"]] = event
        if self.lose_response:
            self.lose_response = False
            raise TimeoutError("lost acknowledgment")
        return {"acknowledged": [event["eventId"] for event in payload["events"]]}


class TrackerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).parent, prefix=".experiment-test-")
        self.studio = FakeStudio()
        self.path = Path(self.temp.name) / "journal.sqlite3"
        self.tracker = tracker_module.ExperimentTracker(self.path, "node-1", self.studio)

    def tearDown(self):
        self.tracker.close()
        self.temp.cleanup()

    def prepare(self, key="request-1"):
        return self.tracker.prepare(request_key=key, actor_user_id="user-1", name="test", kind="train",
            container="trainer", spec={"learning_rate": 1e-5})

    def test_lost_prepare_response_reuses_server_intent_after_restart(self):
        self.studio.lose_response = True
        with self.assertRaises(TimeoutError):
            self.prepare()
        self.tracker = tracker_module.ExperimentTracker(self.path, "node-1", self.studio)
        self.assertEqual(self.prepare()["attemptId"], "attempt-request-1")
        self.assertEqual(len(self.studio.requests), 1)

    def test_launch_claim_is_atomic_and_survives_restart(self):
        self.prepare()
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: self.tracker.claim_launch("request-1"), range(16)))
        self.assertEqual(sum(results), 1)
        restarted = tracker_module.ExperimentTracker(self.path, "node-1", self.studio)
        self.assertFalse(restarted.claim_launch("request-1"))

    def test_events_survive_disconnect_and_lost_acknowledgment(self):
        handle = self.prepare()
        self.studio.offline = True
        self.tracker.emit(handle["attemptId"], "status", {"status": "starting"})
        self.tracker.emit(handle["attemptId"], "status", {"status": "running", "runtime": {"pid": "123"}})
        self.assertEqual(self.tracker.flush_once(), 0)
        self.tracker = tracker_module.ExperimentTracker(self.path, "node-1", self.studio)
        self.studio.offline = False
        self.studio.lose_response = True
        self.assertEqual(self.tracker.flush_once(), 0)
        self.assertEqual(self.tracker.pending_count(), 2)
        self.assertEqual(self.tracker.flush_once(), 2)
        self.assertEqual(self.tracker.pending_count(), 0)
        self.assertEqual(len(self.studio.events), 2)
        self.assertEqual([e["sequence"] for e in self.studio.events.values()], [1, 2])

    def test_concurrent_events_get_distinct_contiguous_sequences(self):
        handle = self.prepare()
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            events = list(pool.map(lambda _: self.tracker.emit(handle["attemptId"], "configuration", {"configuration": {}}), range(20)))
        self.assertEqual(sorted(e["sequence"] for e in events), list(range(1, 21)))

    def test_terminal_and_backward_transitions_are_rejected(self):
        handle = self.prepare()
        self.tracker.emit(handle["attemptId"], "status", {"status": "running"})
        with self.assertRaises(ValueError):
            self.tracker.emit(handle["attemptId"], "status", {"status": "starting"})
        self.tracker.emit(handle["attemptId"], "status", {"status": "failed"})
        with self.assertRaises(ValueError):
            self.tracker.emit(handle["attemptId"], "status", {"status": "running"})
        self.assertEqual(self.tracker.pending_count(), 2)

    def test_request_key_cannot_change_owner_or_inputs(self):
        self.prepare()
        with self.assertRaises(ValueError):
            self.tracker.prepare(request_key="request-1", actor_user_id="other-user", name="test", kind="train",
                container="trainer", spec={"learning_rate": 1e-5})

    def test_pid_reuse_does_not_bind_to_wrong_experiment(self):
        for key in ("first", "second"):
            handle = self.prepare(key)
            self.tracker.emit(handle["attemptId"], "status", {"status": "running", "runtime": {"pid": "123"}})
        self.assertIsNone(self.tracker.find_execution("trainer", "123"))

    def test_environment_allowlist_and_recursive_redaction(self):
        self.assertEqual(tracker_module.safe_environment({"LR": "1e-5", "TOKEN": "private"}), {"LR": "1e-5"})
        value = tracker_module.redact({"api_key": "private", "nested": [{"message": "Bearer private"}], "max_tokens": 100})
        self.assertNotIn("private", json.dumps(value))
        self.assertEqual(value["max_tokens"], 100)
        value = tracker_module.redact({'argv': ['trainer', '--api_key', 'private', '--lr', '0.1'],
            'config': '{"password": "private"}', 'yaml': "access_token: 'private'", 'hf_hub_token': 'private',
            'header': '{"authorization": "Basic private"}'})
        self.assertNotIn('private', json.dumps(value))


class AssetTests(unittest.TestCase):
    assets = assets

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).parent, prefix=".experiment-test-")
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source / "data.jsonl").write_text('{"question":"hello"}\n', encoding="utf-8")
        # This fixture is also reused by evaluation tests without an assets module.
        if isinstance(self, AssetTests):
            # Keep streaming coverage deterministic on reflink-capable hosts.
            self.reflink_helper = self.assets._try_reflink
            fallback = patch.object(self.assets, '_try_reflink', return_value=False)
            fallback.start()
            self.addCleanup(fallback.stop)

    def tearDown(self):
        # Immutable copied files need write permission before cleanup on Windows.
        for path in self.root.rglob("*"):
            if path.is_file() and not path.is_symlink():
                path.chmod(0o600)
        self.temp.cleanup()

    def test_snapshot_preserves_original_content_after_source_changes(self):
        snapshot = self.assets.snapshot(self.source, self.root / "versions")
        (self.source / "data.jsonl").write_text("changed", encoding="utf-8")
        self.assertEqual(self.assets.manifest(snapshot["path"])["fingerprint"], snapshot["fingerprint"])
        self.assertNotEqual(self.assets.manifest(self.source)["fingerprint"], snapshot["fingerprint"])

    def test_same_content_same_fingerprint_across_roots(self):
        snapshot = self.assets.snapshot(self.source, self.root / "versions")
        again = self.assets.snapshot(self.source, self.root / "versions")
        self.assertEqual(snapshot["fingerprint"], again["fingerprint"])
        self.assertTrue(again["cacheHit"])
        self.assertEqual(snapshot["fingerprint"], self.assets.manifest(self.source)["fingerprint"])
        self.assertFalse(list((self.root / "versions").glob(".creating-*")))

    def test_cached_snapshot_reuse_does_not_read_file_contents(self):
        first = self.assets.snapshot(self.source, self.root / "versions")
        self.assertEqual(first['cacheStatus'], 'stored')
        with patch.object(self.assets, 'file_digest', side_effect=AssertionError('unexpected full hash')):
            again = self.assets.snapshot(self.source, self.root / "versions")
        self.assertTrue(again['cacheHit'])
        self.assertEqual(again['cacheStatus'], 'hit')
        self.assertEqual(first['fingerprint'], again['fingerprint'])

    def test_new_snapshot_reads_source_once_and_verifies_copy(self):
        nested = self.source / 'nested'
        nested.mkdir()
        (nested / 'weights.bin').write_bytes(b'weights' * (1024 * 1024))
        expected = self.assets.manifest(self.source)
        original_open = Path.open
        reads = []
        def tracked_open(path, mode='r', *args, **kwargs):
            if mode == 'rb':
                reads.append(path)
            return original_open(path, mode, *args, **kwargs)
        with patch.object(Path, 'open', new=tracked_open):
            result = self.assets.snapshot(self.source, self.root / 'versions')
        self.assertEqual(result['fingerprint'], expected['fingerprint'])
        self.assertEqual(result['files'], expected['files'])
        self.assertEqual(len(reads), 2 * expected['fileCount'])
        for item in expected['files']:
            self.assertEqual(reads.count(self.source / item['path']), 1)
        self.assertFalse(result['reused'])
        self.assertEqual(result['cacheStatus'], 'stored')
        self.assertIn('copyAndHash', result['timingMs'])
        self.assertEqual(result['copyStats']['streamedFiles'], expected['fileCount'])

    def test_streamed_copy_is_verified_before_the_next_file_is_copied(self):
        (self.source / 'weights.bin').write_bytes(b'weights')
        verified = []
        original_digest = self.assets.file_digest
        def verify(path):
            # Verification is immediate: later files must not exist yet.
            if path.name == 'data.jsonl':
                self.assertFalse((path.parent / 'weights.bin').exists())
            verified.append(path.name)
            return original_digest(path)
        with patch.object(self.assets, 'file_digest', side_effect=verify):
            result = self.assets.snapshot(self.source, self.root / 'versions')
        self.assertEqual(verified, ['data.jsonl', 'weights.bin'])
        self.assertEqual(result['fileCount'], 2)

    @staticmethod
    def simulated_clone(reader, writer):
        # Portable stand-in for successful FICLONE; never alias source inodes.
        shutil.copyfileobj(reader, writer)
        writer.flush()
        return True

    def test_reflink_hashes_only_the_clone_and_reuses_the_cache(self):
        expected = self.assets.manifest(self.source)
        original_digest = self.assets.file_digest
        with patch.object(self.assets, '_try_reflink', side_effect=self.simulated_clone), \
                patch.object(self.assets, 'file_digest', wraps=original_digest) as digest:
            first = self.assets.snapshot(self.source, self.root / 'versions')
            self.assertEqual(digest.call_count, expected['fileCount'])
            for call in digest.call_args_list:
                self.assertNotEqual(call.args[0].parent, self.source)
            again = self.assets.snapshot(self.source, self.root / 'versions')
            self.assertEqual(digest.call_count, expected['fileCount'])
        self.assertEqual(first['fingerprint'], expected['fingerprint'])
        self.assertEqual(first['copyStats'], {'reflinkFiles': 1, 'reflinkBytes': expected['byteCount'],
                                              'streamedFiles': 0, 'streamedBytes': 0})
        self.assertTrue(again['cacheHit'])
        (self.source / 'data.jsonl').write_bytes(b'changed')
        self.assertEqual(self.assets.manifest(first['path'])['fingerprint'], expected['fingerprint'])

    def test_mixed_clone_and_stream_keep_the_same_manifest(self):
        (self.source / 'weights.bin').write_bytes(b'weights')
        expected = self.assets.manifest(self.source)
        def mixed_clone(reader, writer):
            return self.simulated_clone(reader, writer) if Path(reader.name).name == 'weights.bin' else False
        with patch.object(self.assets, '_try_reflink', side_effect=mixed_clone):
            result = self.assets.snapshot(self.source, self.root / 'versions')
        self.assertEqual(result['files'], expected['files'])
        self.assertEqual(result['copyStats']['reflinkFiles'], 1)
        self.assertEqual(result['copyStats']['streamedFiles'], 1)

    def test_reflink_single_file_preserves_manifest_and_file_argument_layout(self):
        source = self.source / 'data.jsonl'
        expected = self.assets.manifest(source)
        with patch.object(self.assets, '_try_reflink', side_effect=self.simulated_clone):
            result = self.assets.snapshot(source, self.root / 'versions')
        self.assertTrue(result['sourceIsFile'])
        self.assertEqual(result['files'], expected['files'])
        self.assertEqual((Path(result['path']) / source.name).read_bytes(), source.read_bytes())
        self.assertEqual(result['copyStats']['reflinkFiles'], 1)

    def test_cloned_copy_changes_during_verification_are_rejected(self):
        original_digest = self.assets.file_digest
        def corrupt(path):
            checksum = original_digest(path)
            path.write_bytes(b'corrupt')
            return checksum
        versions = self.root / 'versions'
        with patch.object(self.assets, '_try_reflink', side_effect=self.simulated_clone), \
                patch.object(self.assets, 'file_digest', side_effect=corrupt), \
                self.assertRaisesRegex(ValueError, 'copy failed content verification'):
            self.assets.snapshot(self.source, versions)
        self.assertEqual(list(versions.iterdir()), [])

    def test_linux_reflink_unsupported_errors_reset_target_and_fall_back(self):
        target = self.root / 'target.bin'
        for error in (errno.EXDEV, errno.EOPNOTSUPP, errno.ENOTTY, errno.EINVAL, errno.ENOSYS):
            with self.subTest(error=error):
                def unsupported(fd, command, source_fd):
                    self.assertEqual(command, 0x40049409)
                    os.write(fd, b'partial')
                    raise OSError(error, 'unsupported')
                fake_fcntl = types.SimpleNamespace(ioctl=unsupported)
                with patch.object(sys, 'platform', 'linux'), patch.dict(sys.modules, fcntl=fake_fcntl), \
                        (self.source / 'data.jsonl').open('rb') as reader, target.open('wb') as writer:
                    self.assertFalse(self.reflink_helper(reader, writer))
                    self.assertEqual(writer.tell(), 0)
                    self.assertEqual(os.fstat(writer.fileno()).st_size, 0)

    def test_reflink_io_failure_is_not_silently_ignored(self):
        def failed_clone(*args):
            raise OSError(errno.EIO, 'disk failure')
        versions = self.root / 'versions'
        with patch.object(sys, 'platform', 'linux'), \
                patch.dict(sys.modules, fcntl=types.SimpleNamespace(ioctl=failed_clone)), \
                patch.object(self.assets, '_try_reflink', side_effect=self.reflink_helper), \
                self.assertRaises(OSError):
            self.assets.snapshot(self.source, versions)
        self.assertEqual(list(versions.iterdir()), [])

    @unittest.skipUnless(sys.platform.startswith('linux'), 'requires Linux FICLONE')
    def test_real_reflink_preserves_independent_bytes_and_original_permissions(self):
        original = self.source / 'data.jsonl'
        with original.open('rb') as reader, (self.root / 'clone-probe').open('xb') as writer:
            if not self.reflink_helper(reader, writer):
                self.skipTest('test filesystem does not support reflink')
        mode = original.stat().st_mode
        expected = self.assets.manifest(self.source)
        with patch.object(self.assets, '_try_reflink', side_effect=self.reflink_helper):
            result = self.assets.snapshot(self.source, self.root / 'versions')
        copied = Path(result['path']) / original.name
        self.assertNotEqual(original.stat().st_ino, copied.stat().st_ino)
        self.assertEqual(original.stat().st_mode, mode)
        self.assertEqual(result['copyStats']['reflinkFiles'], 1)
        original.write_bytes(b'changed source')
        self.assertEqual(self.assets.manifest(result['path'])['fingerprint'], expected['fingerprint'])

    def test_corruption_during_streamed_verification_is_rejected(self):
        original_digest = self.assets.file_digest
        def corrupt(path):
            path.write_bytes(b'corrupt')
            return original_digest(path)
        versions = self.root / 'versions'
        with patch.object(self.assets, 'file_digest', side_effect=corrupt), \
                self.assertRaisesRegex(ValueError, 'copy failed content verification'):
            self.assets.snapshot(self.source, versions)
        self.assertEqual(list(versions.iterdir()), [])

    def test_cloned_source_changes_during_verification_are_rejected(self):
        original_digest = self.assets.file_digest
        def change_source(path):
            (self.source / 'data.jsonl').write_bytes(b'concurrent change')
            return original_digest(path)
        versions = self.root / 'versions'
        with patch.object(self.assets, '_try_reflink', side_effect=self.simulated_clone), \
                patch.object(self.assets, 'file_digest', side_effect=change_source), \
                self.assertRaisesRegex(ValueError, 'Source changed'):
            self.assets.snapshot(self.source, versions)
        self.assertEqual(list(versions.iterdir()), [])

    def test_source_changes_during_snapshot_abort_and_clean_staging(self):
        original_copy = self.assets._copy_manifest
        for change in ('rewrite', 'add', 'remove'):
            with self.subTest(change=change):
                model = self.root / change
                model.mkdir()
                weights = model / 'weights.bin'
                weights.write_bytes(b'original')
                versions = self.root / (change + '-versions')
                def changed_copy(*args, **kwargs):
                    result = original_copy(*args, **kwargs)
                    if change == 'rewrite':
                        weights.write_bytes(b'new-content')
                    elif change == 'add':
                        (model / 'extra.bin').write_bytes(b'extra')
                    else:
                        weights.unlink()
                        (model / 'replacement.bin').write_bytes(b'original')
                    return result
                with patch.object(self.assets, '_copy_manifest', side_effect=changed_copy), \
                        self.assertRaisesRegex(ValueError, 'Source changed'):
                    self.assets.snapshot(model, versions)
                self.assertEqual(list(versions.iterdir()), [])

    def test_source_replacement_before_streaming_is_rejected(self):
        original_copy = self.assets._copy_manifest
        def replaced_copy(*args, **kwargs):
            (self.source / 'data.jsonl').write_bytes(b'changed-before-open')
            return original_copy(*args, **kwargs)
        versions = self.root / 'versions'
        with patch.object(self.assets, '_copy_manifest', side_effect=replaced_copy), \
                self.assertRaisesRegex(ValueError, 'Source changed'):
            self.assets.snapshot(self.source, versions)
        self.assertEqual(list(versions.iterdir()), [])

    def test_corrupt_copy_is_rejected_and_staging_removed(self):
        original_copy = self.assets._copy_manifest
        def corrupt_copy(source, temporary, *args, **kwargs):
            result = original_copy(source, temporary, *args, **kwargs)
            (temporary / 'data.jsonl').write_bytes(b'corrupt')
            return result
        versions = self.root / 'versions'
        with patch.object(self.assets, '_copy_manifest', side_effect=corrupt_copy), \
                self.assertRaisesRegex(ValueError, 'copy failed content verification'):
            self.assets.snapshot(self.source, versions)
        self.assertEqual(list(versions.iterdir()), [])

    def test_source_modified_while_reading_is_rejected(self):
        original_open = Path.open
        source_file = self.source / 'data.jsonl'
        class ChangingReader:
            def __init__(self, stream):
                self.stream = stream
                self.changed = False
            def __enter__(self):
                return self
            def __exit__(self, *args):
                self.stream.close()
            def fileno(self):
                return self.stream.fileno()
            def read(self, size):
                chunk = self.stream.read(size)
                if chunk and not self.changed:
                    self.changed = True
                    with original_open(source_file, 'ab') as writer:
                        writer.write(b'concurrent change')
                return chunk
        def changing_open(path, mode='r', *args, **kwargs):
            stream = original_open(path, mode, *args, **kwargs)
            return ChangingReader(stream) if path == source_file and mode == 'rb' else stream
        versions = self.root / 'versions'
        with patch.object(Path, 'open', new=changing_open), self.assertRaisesRegex(ValueError, 'Source changed'):
            self.assets.snapshot(self.source, versions)
        self.assertEqual(list(versions.iterdir()), [])

    def test_concurrent_snapshot_publication_reuses_verified_destination(self):
        def competing_rename(temporary, destination):
            shutil.copytree(temporary, destination)
            raise FileExistsError(str(destination))
        versions = self.root / 'versions'
        with patch.object(Path, 'rename', new=competing_rename):
            result = self.assets.snapshot(self.source, versions)
        self.assertTrue(result['reused'])
        self.assertEqual(result['cacheStatus'], 'stored')
        self.assertEqual(result['fingerprint'], self.assets.manifest(self.source)['fingerprint'])
        self.assertFalse(list(versions.glob('.creating-*')))

    def test_uncached_corrupt_existing_version_is_rejected(self):
        versions = self.root / 'versions'
        first = self.assets.snapshot(self.source, versions)
        for path in versions.glob('.snapshot-cache-*.json'):
            path.unlink()
        copied = Path(first['path']) / 'data.jsonl'
        copied.chmod(0o600)
        copied.write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'Managed version contents were modified'):
            self.assets.snapshot(self.source, versions)
        self.assertFalse(list(versions.glob('.creating-*')))
        self.assertEqual(copied.read_bytes(), b'corrupt')

    def test_missing_cached_destination_is_recreated(self):
        first = self.assets.snapshot(self.source, self.root / 'versions')
        destination = Path(first['path'])
        for path in destination.rglob('*'):
            if path.is_file():
                path.chmod(0o600)
        shutil.rmtree(destination)
        again = self.assets.snapshot(self.source, self.root / 'versions')
        self.assertEqual(again['fingerprint'], first['fingerprint'])
        self.assertFalse(again['reused'])
        self.assertEqual(again['cacheStatus'], 'stored')

    def test_cache_falls_back_to_hash_after_source_change(self):
        first = self.assets.snapshot(self.source, self.root / "versions")
        (self.source / 'data.jsonl').write_text('{"question":"other"}\n', encoding='utf-8')
        again = self.assets.snapshot(self.source, self.root / "versions")
        self.assertFalse(again['cacheHit'])
        self.assertNotEqual(first['fingerprint'], again['fingerprint'])

    def test_missing_cache_bootstraps_from_full_verification(self):
        first = self.assets.snapshot(self.source, self.root / "versions")
        for path in (self.root / 'versions').glob('.snapshot-cache-*.json'):
            path.unlink()
        with patch.object(self.assets, 'file_digest', wraps=self.assets.file_digest) as digest:
            again = self.assets.snapshot(self.source, self.root / "versions")
        self.assertFalse(again['cacheHit'])
        self.assertEqual(again['cacheStatus'], 'stored')
        self.assertTrue(again['reused'])
        self.assertFalse(list((self.root / 'versions').glob('.creating-*')))
        self.assertGreaterEqual(digest.call_count, 2)
        self.assertEqual(first['fingerprint'], again['fingerprint'])
        with patch.object(self.assets, 'file_digest', side_effect=AssertionError('unexpected full hash')):
            cached = self.assets.snapshot(self.source, self.root / "versions")
        self.assertTrue(cached['cacheHit'])

    def test_corrupt_cache_falls_back_to_full_verification(self):
        first = self.assets.snapshot(self.source, self.root / 'versions')
        cache_file, = (self.root / 'versions').glob('.snapshot-cache-*.json')
        cache_file.write_text('{"broken": true}', encoding='utf-8')
        with patch.object(self.assets, 'file_digest', wraps=self.assets.file_digest) as digest:
            again = self.assets.snapshot(self.source, self.root / 'versions')
        self.assertFalse(again['cacheHit'])
        self.assertGreaterEqual(digest.call_count, 2)
        self.assertEqual(first['fingerprint'], again['fingerprint'])

    def test_snapshot_contains_only_original_files_for_directory_based_readers(self):
        (self.source / '.hidden.json').write_text('["source"]', encoding='utf-8')
        snapshot = self.assets.snapshot(self.source, self.root / 'versions')
        self.assertEqual(sorted(path.name for path in Path(snapshot['path']).iterdir()), ['.hidden.json', 'data.jsonl'])

    def test_detects_modified_managed_version(self):
        snapshot = self.assets.snapshot(self.source, self.root / "versions")
        copied = Path(snapshot["path"]) / "data.jsonl"
        copied.chmod(0o600)
        copied.write_text("tampered", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.assets.snapshot(self.source, self.root / "versions")

    def test_disallows_snapshot_storage_in_source(self):
        with self.assertRaises(ValueError):
            self.assets.snapshot(self.source, self.source / "versions")

    def test_empty_asset_cannot_be_verified(self):
        empty = self.root / "empty"
        empty.mkdir()
        with self.assertRaises(ValueError):
            self.assets.manifest(empty)

    def test_dataset_cannot_escape_snapshot_through_absolute_manifest_reference(self):
        info = self.source / 'dataset_info.json'
        info.write_text(json.dumps({'train': {'file_name': '/mutable/original.json'}}), encoding='utf-8')
        with self.assertRaises(ValueError):
            self.assets.validate_dataset_references(self.source)
        info.write_text(json.dumps({'train': {'file_name': 'data.jsonl'}}), encoding='utf-8')
        self.assets.validate_dataset_references(self.source)

    def test_single_file_snapshot_preserves_file_argument(self):
        source = self.source / "data.jsonl"
        captured = self.assets.snapshot(source, self.root / "versions")
        self.assertTrue(captured["sourceIsFile"])
        self.assertTrue((Path(captured["path"]) / source.name).is_file())

    def test_windows_snapshot_rename_retries_temporary_access_failure(self):
        original = Path.rename
        attempts = []
        def rename(path, destination):
            attempts.append((path, destination))
            if len(attempts) < 3:
                error = PermissionError('temporary directory access failure')
                error.winerror = 5
                raise error
            return original(path, destination)
        with patch.object(self.assets, 'os', types.SimpleNamespace(name='nt', fstat=os.fstat)), \
             patch.object(Path, 'rename', new=rename), patch.object(self.assets.time, 'sleep') as sleep:
            result = self.assets.snapshot(self.source, self.root / 'versions')
        self.assertEqual(len(attempts), 3)
        self.assertEqual(sleep.call_count, 2)
        self.assertEqual(self.assets.manifest(result['path'])['fingerprint'], self.assets.manifest(self.source)['fingerprint'])
        self.assertFalse(list((self.root / 'versions').glob('.creating-*')))

    def test_windows_snapshot_rename_permanent_failure_is_bounded_and_cleans_staging(self):
        error = PermissionError('permanent directory access failure')
        error.winerror = 5
        with patch.object(self.assets, 'os', types.SimpleNamespace(name='nt', fstat=os.fstat)), \
             patch.object(Path, 'rename', side_effect=error) as rename, patch.object(self.assets.time, 'sleep') as sleep, \
             self.assertRaises(PermissionError):
            self.assets.snapshot(self.source, self.root / 'versions')
        self.assertEqual(rename.call_count, 5)
        self.assertEqual(sleep.call_count, 4)
        self.assertEqual(list((self.root / 'versions').iterdir()), [])
        self.assertTrue((self.source / 'data.jsonl').is_file())


class NativeAssetTests(AssetTests):
    assets = native.shared("experiment_assets")


class PublicationTests(unittest.TestCase):
    setUp = TrackerTests.setUp
    tearDown = TrackerTests.tearDown

    def prepare_transfer(self):
        self.root = Path(self.temp.name)
        source = self.root / 'source'
        source.mkdir()
        (source / 'model.safetensors').write_bytes(b'model weights')
        captured = assets.snapshot(source, self.root / 'source-versions')
        self.source = capture.asset_registration(captured, node_id='node-1', container='trainer', owner_id='user-1',
            source=str(source), kind='model', role='published_source')
        self.target = self.root / 'published' / 'model'
        self.directory = self.root / 'publication'
        self.directory.mkdir()
        self.handle = self.tracker.prepare(request_key='publish', actor_user_id='user-1', name='publish', kind='publish',
            container='trainer', spec={})
        self.attempt = self.handle['attemptId']
        process_module.atomic_json(self.directory / 'execution.json', {'attemptId': self.attempt, 'outputPaths': []})
        process_module.atomic_json(self.directory / 'transfer.json', {'attemptId': self.attempt, 'owner': 'user-1',
            'sourceContainer': 'trainer', 'sourceAsset': self.source, 'destinationContainer': 'inference',
            'destination': str(self.target)})
        self.tracker.emit(self.attempt, 'status', {'status': 'running', 'runtime': {'operation': 'model_publish',
            'experimentDirectory': str(self.directory), 'archiveStatus': 'pending', 'sourceAsset': self.source,
            'source': str(source), 'destination': str(self.target), 'destinationContainer': 'inference'}})
        self.tracker.emit_once(self.attempt, 'configuration', {'configuration': {'completeness': 'incomplete'}}, 'publication-preparation')

    def control(self, container, module, request):
        self.assertEqual(container, 'inference')
        return publish_files.operate(request)

    def capture(self, container, path, owner, kind, operation='snapshot'):
        return assets.manifest(path) if operation == 'manifest' else assets.snapshot(path, self.root / 'target-versions')

    def copy(self, source_container, source, destination_container, destination):
        self.assertEqual((source_container, destination_container), ('trainer', 'inference'))
        shutil.copytree(source, destination, dirs_exist_ok=True)

    def test_publication_verifies_transfer_and_archives_linked_target_version(self):
        self.prepare_transfer()
        publish_module.transfer(self.directory, capture_fn=self.capture, control_fn=self.control, stream_fn=self.copy)
        self.assertEqual(assets.manifest(self.target)['fingerprint'], self.source['fingerprint'])
        # The retained receipt/version must survive later edits to the deployed path.
        (self.target / 'model.safetensors').chmod(0o600)
        (self.target / 'model.safetensors').write_bytes(b'changed after publication')
        process_module.atomic_json(self.directory / 'process.json', {'attemptId': self.attempt, 'exitCode': 0})
        publish_module.collect_publish(self.tracker, self.tracker.execution(self.attempt),
            capture_fn=self.capture, control_fn=self.control)
        output = self.tracker.event_payload(self.attempt, 'published-model')
        self.assertEqual(output['location']['container'], 'inference')
        self.assertEqual(output['metadata']['inputVersionIds'], [self.source['id']])
        self.assertEqual(output['fingerprint'], self.source['fingerprint'])
        self.assertEqual(self.tracker.execution(self.attempt)['state'], 'succeeded')
        pending = self.tracker.pending_count()
        publish_module.collect_publish(self.tracker, self.tracker.execution(self.attempt))
        self.assertEqual(pending, self.tracker.pending_count())
        self.assertFalse(list(self.directory.glob('transfer-*')))

    def test_publication_never_overwrites_different_existing_model(self):
        self.prepare_transfer()
        self.target.mkdir(parents=True)
        existing = self.target / 'model.safetensors'
        existing.write_bytes(b'existing model')
        with self.assertRaisesRegex(ValueError, 'differ'), patch.object(publish_module, 'stream_copy') as run:
            publish_module.transfer(self.directory, capture_fn=self.capture, control_fn=self.control, stream_fn=run)
        run.assert_not_called()
        self.assertEqual(existing.read_bytes(), b'existing model')
        self.assertFalse((self.directory / 'publication.json').exists())
        self.assertFalse((self.target.parent / '.model.publish-lock').exists())

    def test_failed_copy_cannot_activate_or_issue_publication_receipt(self):
        self.prepare_transfer()
        def interrupted(*args):
            raise ValueError('Model copy failed: interrupted')
        with self.assertRaisesRegex(ValueError, 'copy failed'):
            publish_module.transfer(self.directory, capture_fn=self.capture, control_fn=self.control,
                stream_fn=interrupted)
        self.assertFalse(self.target.exists())
        self.assertFalse((self.directory / 'publication.json').exists())
        self.assertFalse((self.target.parent / '.model.publish-lock').exists())

    def test_destination_lock_excludes_concurrent_publishers_and_foreign_cleanup(self):
        self.prepare_transfer()
        request = {'attemptId': self.attempt, 'destination': str(self.target)}
        publish_files.operate({**request, 'operation': 'claim'})
        with self.assertRaises(FileExistsError):
            publish_files.operate({**request, 'operation': 'claim', 'attemptId': 'another-attempt'})
        self.assertFalse(publish_files.operate({**request, 'operation': 'release_if_owned', 'attemptId': 'another-attempt'})['released'])
        self.assertTrue(publish_files.operate({**request, 'operation': 'release_if_owned'})['released'])

    def test_publication_repeated_submission_does_not_launch_another_process(self):
        self.prepare_transfer()
        with patch.object(publish_module, 'recorded_source', return_value=self.source), \
             patch.object(publish_module.subprocess, 'Popen', return_value=types.SimpleNamespace(pid=123)) as popen:
            options = dict(owner='user-1', source_container='trainer', source='/trained/model', target_container='inference',
                destination='/published/model', context={'requestKey': 'publish-launch'})
            first = publish_module.start_publish(self.tracker, **options)
            second = publish_module.start_publish(self.tracker, **options)
        self.assertEqual(first['experiment_attempt_id'], second['experiment_attempt_id'])
        self.assertEqual(popen.call_count, 1)
        with self.assertRaisesRegex(ValueError, 'changed'):
            publish_module.start_publish(self.tracker, **{**options, 'destination': '/another-model'})

    def test_publication_cancel_does_not_signal_a_reused_pid(self):
        self.prepare_transfer()
        process_module.atomic_json(self.directory / 'process.json', {'pid': 123, 'identity': {'startTicks': 'old'}})
        with patch.object(publish_module, 'process_identity', return_value={'startTicks': 'new'}), \
             patch.object(publish_module.os, 'kill') as kill, self.assertRaisesRegex(ValueError, 'identity'):
            publish_module.stop_publish(self.tracker.execution(self.attempt))
        kill.assert_not_called()

    def test_uncertain_publication_launch_returns_handle_and_cannot_relaunch(self):
        self.prepare_transfer()
        with patch.object(publish_module, 'recorded_source', return_value=self.source), \
             patch.object(publish_module.subprocess, 'Popen', side_effect=TimeoutError('launch response lost')) as popen:
            options = dict(owner='user-1', source_container='trainer', source='/trained/model', target_container='inference',
                destination='/published/model', context={'requestKey': 'publish-uncertain'})
            first = publish_module.start_publish(self.tracker, **options)
            repeated = publish_module.start_publish(self.tracker, **options)
        self.assertEqual(first['experiment_attempt_id'], repeated['experiment_attempt_id'])
        self.assertEqual(first['metrics']['attemptStatus'], 'unknown')
        self.assertEqual(popen.call_count, 1)

    def test_missing_publication_process_journal_never_implies_success(self):
        self.prepare_transfer()
        publish_module.collect_publish(self.tracker, self.tracker.execution(self.attempt),
            capture_fn=self.capture, control_fn=self.control)
        self.assertEqual(self.tracker.execution(self.attempt)['state'], 'unknown')
        self.assertIsNone(self.tracker.event_payload(self.attempt, 'published-model'))

    def test_zero_exit_without_receipt_waits_for_verification(self):
        self.prepare_transfer()
        process_module.atomic_json(self.directory / 'process.json', {'attemptId': self.attempt, 'exitCode': 0})
        with self.assertRaises(FileNotFoundError):
            publish_module.collect_publish(self.tracker, self.tracker.execution(self.attempt),
                capture_fn=self.capture, control_fn=self.control)
        self.assertEqual(self.tracker.execution(self.attempt)['state'], 'unknown')
        self.assertIsNone(self.tracker.event_payload(self.attempt, 'published-model'))

    def test_cancel_after_copy_retains_output_without_turning_successful(self):
        self.prepare_transfer()
        publish_module.transfer(self.directory, capture_fn=self.capture, control_fn=self.control, stream_fn=self.copy)
        process_module.atomic_json(self.directory / 'process.json', {'attemptId': self.attempt, 'exitCode': -15, 'signals': [15]})
        publish_module.collect_publish(self.tracker, self.tracker.execution(self.attempt),
            capture_fn=self.capture, control_fn=self.control)
        self.assertEqual(self.tracker.execution(self.attempt)['state'], 'cancelled')
        self.assertEqual(self.tracker.event_payload(self.attempt, 'published-model')['fingerprint'], self.source['fingerprint'])

    def test_publication_uses_recorded_training_output_even_if_original_directory_changes(self):
        self.prepare_transfer()
        source = self.source['metadata']['sourcePath']
        self.tracker.emit_once(self.attempt, 'artifact', {**self.source, 'direction': 'output', 'role': 'trained_model'},
            'output:model:' + source)
        (Path(source) / 'model.safetensors').write_bytes(b'different model')
        with patch.object(publish_module, 'container_asset') as capture_asset:
            selected = publish_module.recorded_source(self.tracker, source, 'trainer', 'user-1', [self.attempt])
        capture_asset.assert_not_called()
        self.assertEqual(selected['fingerprint'], self.source['fingerprint'])
        self.assertEqual(selected['direction'], 'input')


class InputCaptureTests(unittest.TestCase):
    @staticmethod
    def fake_capture(container, path, owner, kind):
        return {"schemaVersion": 1, "fingerprint": ("b" if kind == "model" else "a") * 64,
                "files": [{"path": "train.parquet", "size": 1, "sha256": "c" * 64}], "fileCount": 1, "byteCount": 1,
                "path": f"/versions/{kind}", "sourceIsFile": path.endswith('.parquet')}

    def test_pins_final_env_and_does_not_capture_secret_environment(self):
        kwargs = {"docker_container": "trainer", "env_vars": {"MODEL_PATH": "/base", "DATASET_DIR": "/data", "API_KEY": "private"}}
        versions = capture.pin_launch_inputs(kwargs, node_id="node", owner_id="owner", kind="train", capture_fn=self.fake_capture)
        self.assertEqual(kwargs["env_vars"]["DATASET_DIR"], "/versions/dataset")
        self.assertEqual(kwargs["env_vars"]["MODEL_PATH"], "/versions/model")
        self.assertEqual({version["role"] for version in versions}, {"base_model", "training_dataset"})
        self.assertNotIn("private", json.dumps(versions))

    def test_grpo_lists_remain_lists_of_files(self):
        kwargs = {"docker_container": "trainer", "positional_args": ['data.train_files=["/data/train.parquet"]',
            "actor_rollout_ref.model.path=/base", "trainer.max_epochs=1"]}
        capture.pin_launch_inputs(kwargs, node_id="node", owner_id="owner", kind="train", capture_fn=self.fake_capture)
        self.assertEqual(kwargs["positional_args"][0], 'data.train_files=["/versions/dataset/train.parquet"]')
        self.assertEqual(kwargs["positional_args"][1], 'actor_rollout_ref.model.path=/versions/model')
        self.assertEqual(kwargs["positional_args"][2], "trainer.max_epochs=1")

    def test_preprocessing_freezes_source_without_rewriting_destination(self):
        kwargs = {'docker_container': 'trainer', 'script_args': {
            'input_folder': '/raw/source', 'output_folder': '/processed/result', 'threshold': 90}}
        versions = capture.pin_launch_inputs(kwargs, node_id='node', owner_id='owner', kind='preprocess', capture_fn=self.fake_capture)
        self.assertEqual(kwargs['script_args']['input_folder'], '/versions/dataset')
        self.assertEqual(kwargs['script_args']['output_folder'], '/processed/result')
        self.assertEqual([(asset['kind'], asset['role']) for asset in versions], [('dataset', 'source_dataset')])

    def test_processing_model_and_dataset_are_not_labeled_as_evaluation_or_training(self):
        for module in (capture, native.shared('experiment_capture')):
            with self.subTest(module=module.__name__):
                kwargs = {'docker_container': 'trainer', 'env_vars': {'MODEL_PATH': '/processor-model', 'DATASET_DIR': '/raw-data'}}
                versions = module.pin_launch_inputs(kwargs, node_id='node', owner_id='owner', kind='preprocess', capture_fn=self.fake_capture)
                self.assertEqual({version['role'] for version in versions}, {'processing_model', 'source_dataset'})

    def test_grpo_resume_pins_checkpoint_and_retains_step_for_execution_alias(self):
        kwargs = {'docker_container': 'trainer', 'positional_args': [
            'trainer.resume_mode=resume_path', 'trainer.resume_from_path=/runs/global_step_30']}
        versions = capture.pin_launch_inputs(kwargs, node_id='node', owner_id='owner', kind='train', capture_fn=self.fake_capture)
        self.assertEqual(kwargs['positional_args'][1], 'trainer.resume_from_path=/versions/checkpoint')
        self.assertEqual(versions[0]['role'], 'resume_checkpoint')
        self.assertEqual(versions[0]['metadata']['resumeDirectoryName'], 'global_step_30')
        replayed = capture.pin_launch_inputs(kwargs, node_id='node', owner_id='owner', kind='train',
            capture_fn=self.fake_capture, resume_checkpoint_name='global_step_30')
        self.assertEqual(replayed[0]['id'], versions[0]['id'])
        self.assertEqual(replayed[0]['metadata']['resumeDirectoryName'], 'global_step_30')

    def test_preprocessing_resume_registers_prior_product_as_input(self):
        kwargs = {'docker_container': 'trainer', 'experiment_resume_output': '/prior/product',
                  'script_args': {'input_folder': '/raw/source', 'output_folder': '/new/product', 'resume': True}}
        versions = capture.pin_launch_inputs(kwargs, node_id='node', owner_id='owner', kind='preprocess', capture_fn=self.fake_capture)
        self.assertEqual({item['role'] for item in versions}, {'source_dataset', 'preprocessing_resume'})
        seed = next(item for item in versions if item['role'] == 'preprocessing_resume')
        self.assertEqual(seed['metadata']['sourcePath'], '/prior/product')
        self.assertEqual(kwargs['script_args']['output_folder'], '/new/product')

    def test_grpo_resume_rejects_ignored_missing_or_unnamed_checkpoint(self):
        for arguments in ([ 'trainer.resume_mode=resume_path' ],
                          ['trainer.resume_mode=resume_path', 'trainer.resume_from_path=null'],
                          ['trainer.resume_mode=auto', 'trainer.resume_mode=resume_path', 'trainer.resume_from_path=/runs/global_step_30'],
                          ['trainer.resume_mode=auto', 'trainer.resume_from_path=/runs/global_step_30'],
                          ['trainer.resume_mode=resume_path', 'trainer.resume_from_path=/runs/latest']):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                capture.pin_launch_inputs({'docker_container': 'trainer', 'positional_args': arguments},
                    node_id='node', owner_id='owner', kind='train', capture_fn=self.fake_capture)

    def test_grpo_resume_descriptor_is_valid_for_existing_consumers(self):
        spec = importlib.util.spec_from_file_location('training_descriptor_test',
            Path(__file__).parents[1] / 'medflow_agent_tools' / '_training_descriptors.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.validate_training_descriptors()
        descriptor = next(item for item in module.list_training_descriptors() if item['id'] == 'grpo_train')
        params = {item['name']: item for item in descriptor['params']}
        self.assertEqual(params['resume_from_path']['type'], 'checkpoint')
        self.assertIn('resume_path', params['resume_mode']['options'])

    def test_resume_underscore_argument_is_also_versioned(self):
        kwargs = {'docker_container': 'trainer', 'script_args': {'resume_from_checkpoint': '/runs/checkpoint-100'}}
        versions = capture.pin_launch_inputs(kwargs, node_id='node', owner_id='owner', kind='train', capture_fn=self.fake_capture)
        self.assertEqual(kwargs['script_args']['resume_from_checkpoint'], '/versions/checkpoint')
        self.assertEqual(versions[0]['role'], 'resume_checkpoint')

    def test_grpo_execution_alias_preserves_step_and_disables_source_deletion(self):
        kwargs = {'docker_container': 'trainer', 'positional_args': [
            'trainer.resume_mode=resume_path', 'trainer.resume_from_path=/runs/global_step_30',
            'trainer.del_local_ckpt_after_load=True']}
        versions = capture.pin_launch_inputs(kwargs, node_id='node', owner_id='owner', kind='train', capture_fn=self.fake_capture)
        with patch.object(collector, 'container_json', return_value={'runner': '/runner.py'}) as install, \
             patch.object(collector, 'container_python', return_value='python3'), \
             patch.object(collector.subprocess, 'run', return_value=types.SimpleNamespace(returncode=0, stdout='"sha256:image"')):
            collector.prepare_container_execution(kwargs, handle={'attemptId': 'attempt-1'}, actor_user_id='owner',
                script_name='grpo_train', assets=versions)
        self.assertEqual(install.call_args.args[2]['aliases'], {'global_step_30': '/versions/checkpoint'})
        self.assertIn('trainer.resume_from_path=/home/workspace/medflow/users/owner/experiments/attempt-1/inputs/global_step_30', kwargs['positional_args'])
        self.assertIn('trainer.del_local_ckpt_after_load=False', kwargs['positional_args'])
        self.assertNotIn('trainer.del_local_ckpt_after_load=True', kwargs['positional_args'])

    def test_legacy_medical_checkpoint_alias_matches_monitor_and_isolates_attempts(self):
        monitor_path = Path(__file__).parents[1] / 'medflow_agent_tools' / 'runlocal_evaluate_monitor.py'
        tree = ast.parse(monitor_path.read_text(encoding='utf-8'))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_ckpt_output_path')
        from typing import Optional
        namespace = {'Optional': Optional}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(monitor_path), 'exec'), namespace)
        outputs = []
        for attempt in ['attempt-a', 'attempt-b']:
            kwargs = {'docker_container': 'evaluator', 'script_name': 'ckpt_eval',
                      'env_vars': {'CKPT_PATH': '/runs/checkpoint-20'}}
            with patch.object(capture, 'container_json', return_value={'base_model_name_or_path': '/base'}):
                versions = capture.pin_launch_inputs(kwargs, node_id='node', owner_id='owner', kind='evaluate',
                                                     capture_fn=self.fake_capture)
            original_snapshot = versions[0]['location']['path']
            self.assertEqual(tracker_module.safe_environment(kwargs['env_vars'])['CKPT_PATH'], original_snapshot)
            with patch.object(collector, 'container_json', return_value={'runner': '/runner.py'}) as install, \
                 patch.object(collector, 'container_python', return_value='python3'), \
                 patch.object(collector.subprocess, 'run', return_value=types.SimpleNamespace(returncode=0, stdout='"image"')):
                runtime = collector.prepare_container_execution(kwargs, handle={'attemptId': attempt},
                    actor_user_id='owner', script_name='ckpt_eval', assets=versions)
            actual_path = kwargs['env_vars']['CKPT_PATH']
            self.assertTrue(Path(actual_path).name.startswith('check'))
            request = install.call_args.args[2]
            self.assertEqual(request['aliases'], {})
            self.assertEqual(request['checkpointInput']['seed']['location']['path'], original_snapshot)
            self.assertEqual(request['checkpointInput']['path'], actual_path)
            self.assertEqual(request['checkpointInput']['baseModel']['role'], 'base_model')
            self.assertEqual(request['checkpointInput']['baseModelPath'], '/versions/model')
            self.assertEqual(runtime['outputPaths'], [namespace['_ckpt_output_path'](actual_path)])
            self.assertEqual(versions[0]['location']['path'], original_snapshot)
            self.assertIn('/experiments/' + attempt + '/', runtime['outputPaths'][0])
            outputs.extend(runtime['outputPaths'])
        self.assertEqual(len(set(outputs)), 2)


class EvaluationReplayTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.locations = {}
        self.reads = []
        for key, filename in [('ceval', 'questions.json'), ('base', 'model.safetensors'),
                              ('adapter', 'adapter_model.safetensors')]:
            folder = self.root / key
            folder.mkdir()
            (folder / filename).write_text(key + '-original', encoding='utf-8')
            source = '/home/workspace/evaluation/ceval' if key == 'ceval' else '/' + key
            self.locations[source] = folder
        self.kwargs = {'docker_container': 'evaluator', 'script_name': 'generic_ckpt_eval',
                       'script_args': {'CKPT_PATH': '/adapter', 'eval_datasets': 'ceval'}}
        with patch.object(capture, 'container_json', return_value={'base_model_name_or_path': '/base'}):
            self.recorded = self.pin()
        self.reads.clear()

    def capture(self, container, source, owner, kind):
        self.reads.append(source)
        result = assets.snapshot(str(self.locations[source]), str(self.root / 'versions' / kind))
        virtual = f"/versions/{kind}/{result['fingerprint']}"
        self.locations[virtual] = Path(result['path'])
        return {**result, 'path': virtual}

    def pin(self, replay_assets=None):
        return capture.pin_launch_inputs(self.kwargs, node_id='node', owner_id='owner', kind='evaluate',
                                         capture_fn=self.capture, replay_assets=replay_assets)

    def test_replay_uses_archived_dataset_and_base_after_sources_are_removed(self):
        for key in ['/home/workspace/evaluation/ceval', '/base', '/adapter']:
            shutil.rmtree(self.locations.pop(key))
        with patch.object(capture, 'container_json') as read_config:
            replayed = self.pin(self.recorded)
        read_config.assert_not_called()
        identity = lambda items: sorted((item['id'], item['role'], item['fingerprint']) for item in items)
        self.assertEqual(identity(replayed), identity(self.recorded))
        self.assertTrue(all(path.startswith('/versions/') for path in self.reads))
        base = next(item for item in self.recorded if item['role'] == 'base_model')
        self.assertEqual(self.kwargs['env_vars']['MEDFLOW_EVAL_BASE_MODEL'], base['location']['path'])

    def test_changed_dataset_or_base_snapshot_blocks_replay(self):
        for role in ['evaluation_dataset', 'base_model']:
            item = next(item for item in self.recorded if item['role'] == role)
            folder = self.locations[item['location']['path']]
            target = next(folder.iterdir())
            target.chmod(0o600)
            original = target.read_bytes()
            target.write_bytes(b'changed')
            try:
                with self.subTest(role=role), self.assertRaisesRegex(ValueError, 'snapshot.*changed'):
                    self.pin(self.recorded)
            finally:
                target.write_bytes(original)

    def test_missing_snapshot_does_not_fall_back_to_original_directory(self):
        item = next(item for item in self.recorded if item['role'] == 'evaluation_dataset')
        folder = self.locations[item['location']['path']]
        # Remove the archived file but leave the current source dataset available.
        for target in folder.iterdir():
            target.chmod(0o600)
            target.unlink()
        folder.rmdir()
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.pin(self.recorded)
        self.assertNotIn('/home/workspace/evaluation/ceval', self.reads)

    def test_missing_ambiguous_or_wrong_container_binding_is_rejected(self):
        dataset = next(item for item in self.recorded if item['role'] == 'evaluation_dataset')
        for recorded in [[], self.recorded + [dataset],
                         [{**item, 'location': {**item['location'], 'container': 'other'}} for item in self.recorded]]:
            with self.subTest(recorded=recorded), self.assertRaisesRegex(ValueError, 'Draft'):
                self.pin(recorded)

    def test_legacy_checkpoint_replay_preserves_base_model_after_original_is_removed(self):
        self.kwargs = {'docker_container': 'evaluator', 'script_name': 'ckpt_eval',
                       'env_vars': {'CKPT_PATH': '/adapter'}}
        with patch.object(capture, 'container_json', return_value={'base_model_name_or_path': '/base'}):
            recorded = self.pin()
        self.assertEqual({item['role'] for item in recorded}, {'base_model', 'evaluated_model'})
        for key in ['/base', '/adapter']:
            shutil.rmtree(self.locations.pop(key))
        self.reads.clear()
        with patch.object(capture, 'container_json') as original_config:
            replayed = self.pin(recorded)
        original_config.assert_not_called()
        self.assertEqual({item['id'] for item in replayed}, {item['id'] for item in recorded})
        self.assertTrue(all(path.startswith('/versions/') for path in self.reads))


class EvaluationPreflightTests(unittest.TestCase):
    def setUp(self):
        import re
        import shlex
        import subprocess
        from typing import Any, Optional
        self.commands = []
        self.directories = {'/versions/model', '/versions/checkpoint'}
        self.files = {f'/versions/ceval/{name}' for name in ['ceval.py', 'mapping.json', 'ceval.zip']}
        self.files.update({'/versions/checkpoint/adapter_config.json', '/versions/checkpoint/adapter_model.safetensors'})
        self.bindings = [
            {'kind': 'model', 'role': 'base_model', 'location': {'container': 'evaluator', 'path': '/versions/model'}},
            {'kind': 'dataset', 'role': 'evaluation_dataset', 'metadata': {'datasetName': 'ceval'},
             'location': {'container': 'evaluator', 'path': '/versions/ceval'}}]
        def shell(argv, **kwargs):
            command = argv[-1]
            self.commands.append(command)
            parts = shlex.split(command)
            stdout = ''
            if parts[:2] == ['test', '-d']:
                exists = parts[2] in self.directories
            elif parts[:2] == ['test', '-f']:
                exists = parts[2] in self.files or ('||' in parts and parts[-1] in self.files)
            elif parts[:2] == ['test', '-x']:
                exists = True
            elif parts[0] == 'cat':
                exists = parts[1] in self.files
                stdout = json.dumps({'base_model_name_or_path': '/removed/base'})
            else:
                raise AssertionError(f'Unexpected preflight command: {command}')
            return types.SimpleNamespace(returncode=0 if exists else 1, stdout=stdout)
        path = Path(__file__).parents[1] / 'medflow_agent_tools' / 'runlocal_evaluate.py'
        names = {'run_evaluate_preflight', '_evaluate_preflight_issue', '_base_model_from_adapter_config',
                 '_normalize_generic_eval_datasets'}
        tree = ast.parse(path.read_text(encoding='utf-8'))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        namespace = {'Any': Any, 'Optional': Optional, 'Dict': dict, 'List': list,
            'json': json, 're': re, 'shlex': shlex, 'recorded_input': capture.recorded_input,
            'subprocess': types.SimpleNamespace(run=shell, TimeoutExpired=subprocess.TimeoutExpired),
            '_validate_evaluate_container_preflight': lambda container: None,
            'GENERIC_EVAL_DATASETS': ('ceval',),
            'GENERIC_EVAL_SCRIPTS': {'generic_single_model_eval', 'generic_compare_models_eval', 'generic_ckpt_eval'}}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
        self.preflight = namespace['run_evaluate_preflight']

    def check(self, script='generic_single_model_eval', bindings=None):
        return self.preflight(script, 'evaluator', {},
            {'model_fir': '/versions/model', 'CKPT_PATH': '/versions/checkpoint', 'eval_datasets': 'ceval'},
            {'docker_executable': '/engine'}, {}, None, replay_assets=bindings)

    def test_snapshot_dataset_preflight_does_not_require_removed_original(self):
        self.assertIsNone(self.check(bindings=self.bindings))
        self.assertTrue(any('/versions/ceval/ceval.zip' in command for command in self.commands))
        self.assertFalse(any('/home/workspace/evaluation' in command for command in self.commands))

    def test_checkpoint_preflight_uses_bound_base_instead_of_adapter_original(self):
        self.assertIsNone(self.check('generic_ckpt_eval', self.bindings))
        self.assertIn('test -d /versions/model', self.commands)
        self.assertFalse(any('/removed/base' in command for command in self.commands))

    def test_missing_snapshot_or_binding_does_not_fall_back(self):
        self.files.update({f'/home/workspace/evaluation/ceval/{name}' for name in ['ceval.py', 'mapping.json', 'ceval.zip']})
        self.files.remove('/versions/ceval/ceval.zip')
        issue = self.check(bindings=self.bindings)
        self.assertEqual(issue['error_reason'], 'evaluation_dataset_missing')
        self.assertEqual(issue['path'], '/versions/ceval/ceval.zip')
        self.assertEqual(self.check(bindings=[])['error_reason'], 'draft_input_unavailable')
        self.assertEqual(self.check('generic_ckpt_eval', self.bindings[1:])['error_reason'], 'draft_input_unavailable')

    def test_new_evaluation_still_validates_current_dataset_and_base(self):
        self.assertEqual(self.check()['error_reason'], 'evaluation_dataset_missing')
        self.assertEqual(self.check('generic_ckpt_eval')['error_reason'], 'base_model_unavailable')


class LaunchTests(unittest.TestCase):
    def setUp(self):
        TrackerTests.setUp(self)
        setup = patch.object(launch, 'prepare_container_execution', return_value={'experimentDirectory': '/attempt', 'archiveStatus': 'pending'})
        setup.start()
        self.addCleanup(setup.stop)
    tearDown = TrackerTests.tearDown

    def test_legacy_checkpoint_records_fixed_input_and_actual_execution_alias(self):
        version = capture.asset_registration(InputCaptureTests.fake_capture('trainer', '/checkpoint-20', 'user-1', 'checkpoint'),
            node_id='node-1', container='trainer', owner_id='user-1', source='/checkpoint-20',
            kind='checkpoint', role='evaluated_model')
        alias = '/experiments/attempt-legacy/inputs/checkpoint-attempt-legacy'
        def setup(kwargs, **ignored):
            kwargs['env_vars']['CKPT_PATH'] = alias
            return {'experimentDirectory': '/attempt'}
        with patch.object(launch, 'get_experiment_tracker', return_value=self.tracker), \
             patch.object(launch, 'pin_launch_inputs', return_value=[version]), \
             patch.object(launch, 'prepare_container_execution', side_effect=setup):
            response = launch.launch_with_experiment(
                lambda **kwargs: types.SimpleNamespace(metadata={'success': True, 'protocol_hint': {'pid': '123'}}),
                {'docker_container': 'trainer', 'env_vars': {'CKPT_PATH': version['location']['path']}},
                actor_user_id='user-1', script_name='ckpt_eval', kind='evaluate', context={'requestKey': 'legacy'})
        configuration = self.tracker.latest_payload('attempt-legacy', 'configuration')['configuration']
        self.assertEqual(configuration['resolved']['environment']['CKPT_PATH'], version['location']['path'])
        self.assertEqual(configuration['execution']['environment']['CKPT_PATH'], alias)
        self.assertEqual(response.metadata['protocol_hint']['evaluatedModelPath'], alias)

    def test_launch_passes_identity_and_keeps_actual_config_incomplete(self):
        calls = []
        def launcher(**kwargs):
            calls.append(kwargs)
            return types.SimpleNamespace(metadata={"success": True, "protocol_hint": {"pid": "123", "container": "trainer"}})
        with patch.object(launch, "get_experiment_tracker", return_value=self.tracker), patch.object(launch, "pin_launch_inputs", return_value=[]):
            response = launch.launch_with_experiment(launcher, {"docker_container": "trainer", "env_vars": {"API_KEY": "private"}},
                actor_user_id="user-1", script_name="lora", kind="train", context={"requestKey": "launch-1"})
        self.assertEqual(calls[0]["env_vars"]["MEDFLOW_EXPERIMENT_ATTEMPT_ID"], "attempt-launch-1")
        self.assertEqual(response.metadata["experiment"]["experimentId"], "exp-1")
        self.tracker.flush_once()
        events = list(self.studio.events.values())
        self.assertEqual(events[0]["payload"]["configuration"]["completeness"], "incomplete")
        self.assertNotIn("private", json.dumps(events))

    def test_exception_after_launch_is_unknown_and_cannot_relaunch(self):
        calls = []
        def launcher(**kwargs):
            calls.append(kwargs)
            raise TimeoutError("response lost")
        context = {"requestKey": "launch-1"}
        with patch.object(launch, "get_experiment_tracker", return_value=self.tracker), patch.object(launch, "pin_launch_inputs", return_value=[]):
            with self.assertRaises(launch.ExperimentLaunchUncertain):
                launch.launch_with_experiment(launcher, {"docker_container": "trainer"}, actor_user_id="user-1", script_name="lora", kind="train", context=context)
            with self.assertRaises(ValueError):
                launch.launch_with_experiment(launcher, {"docker_container": "trainer"}, actor_user_id="user-1", script_name="lora", kind="train", context=context)
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.tracker.execution("attempt-launch-1")["state"], "unknown")

    def test_short_job_finishes_before_launch_response_without_resurrection(self):
        def launcher(**kwargs):
            attempt = kwargs['env_vars']['MEDFLOW_EXPERIMENT_ATTEMPT_ID']
            self.tracker.emit(attempt, 'status', {'status': 'succeeded', 'runtime': {'archiveStatus': 'complete'}})
            return types.SimpleNamespace(metadata={'success': True, 'protocol_hint': {'pid': '123'}})
        with patch.object(launch, 'get_experiment_tracker', return_value=self.tracker), patch.object(launch, 'pin_launch_inputs', return_value=[]):
            response = launch.launch_with_experiment(launcher, {'docker_container': 'trainer'}, actor_user_id='user-1',
                script_name='preprocess', kind='preprocess', context={'requestKey': 'short-job'})
        self.assertEqual(self.tracker.execution('attempt-short-job')['state'], 'succeeded')
        self.assertEqual(response.metadata['protocol_hint']['status'], 'finished')
        self.assertTrue(response.metadata['success'])

    def test_lost_launch_response_does_not_overwrite_completed_job(self):
        def launcher(**kwargs):
            self.tracker.emit(kwargs['env_vars']['MEDFLOW_EXPERIMENT_ATTEMPT_ID'], 'status', {'status': 'failed'})
            raise TimeoutError('lost response')
        with patch.object(launch, 'get_experiment_tracker', return_value=self.tracker), patch.object(launch, 'pin_launch_inputs', return_value=[]):
            with self.assertRaises(launch.ExperimentLaunchUncertain):
                launch.launch_with_experiment(launcher, {'docker_container': 'trainer'}, actor_user_id='user-1',
                    script_name='preprocess', kind='preprocess', context={'requestKey': 'short-job'})
        self.assertEqual(self.tracker.execution('attempt-short-job')['state'], 'failed')


class CollectionTests(unittest.TestCase):
    setUp = TrackerTests.setUp
    tearDown = TrackerTests.tearDown
    prepare = TrackerTests.prepare

    def filter_receipt(self, **changes):
        receipt = {'status': 'completed', 'total_items': 150, 'processed_items': 150,
                   'passed_items': 0, 'rejected_items': 150, 'api_failed_items': 0, 'invalid_items': 0}
        return {**receipt, **changes}

    def filter_probe(self, attempt, receipt, module=process_module, **process):
        root = Path(self.temp.name) / attempt
        product = root / 'product'
        product.mkdir(parents=True, exist_ok=True)
        module.atomic_json(product / 'score_progress.json', receipt)
        module.atomic_json(product / 'score_summary.json', {key: value for key, value in receipt.items() if key != 'status'})
        module.atomic_json(root / 'execution.json', {'attemptId': attempt, 'scriptName': 'score_based_filtering',
            'declaredOutputs': [{'path': str(product), 'kind': 'dataset', 'role': 'processed_dataset'}]})
        module.atomic_json(root / 'process.json', {'attemptId': attempt, 'exitCode': 0, **process})
        return module.probe(root)

    def test_empty_filter_result_distinguishes_quality_rejections_from_scoring_failures(self):
        for module in (process_module, native.shared('experiment_process')):
            for outcome, rejected, api_failed, status in (
                ('no_qualifying_data', 150, 0, 'succeeded'),
                ('scoring_failed', 0, 150, 'failed'),
                ('scoring_failed', 100, 50, 'failed'),
            ):
                with self.subTest(module=module.__name__, outcome=outcome, failed=api_failed):
                    attempt = self.prepare(f'{module.__name__.split(".")[-2]}-{outcome}-{api_failed}')['attemptId']
                    self.tracker.emit(attempt, 'status', {'status': 'unknown', 'runtime': {
                        'kind': 'preprocess', 'scriptName': 'score_based_filtering', 'experimentDirectory': '/attempt',
                        'archiveStatus': 'awaiting_outputs', 'archiveWarning': 'Process exited without a recognized model or report'}})
                    result = self.filter_probe(attempt, self.filter_receipt(rejected_items=rejected,
                        api_failed_items=api_failed), module)
                    self.assertEqual(result['state'], status)
                    self.assertEqual(result['outputs'], [], 'Reports must never become a usable dataset')
                    self.assertEqual(len(result['reports']), 2)
                    self.assertEqual(result['dataProcessing']['outcome'], outcome)
                    if status == 'failed':
                        collector.observe_execution_status(self.tracker, self.tracker.execution(attempt), probe_fn=lambda *_: result)
                        self.assertIn(f'{api_failed} of 150', self.tracker.latest_payload(attempt, 'status')['error'])
                    collector.collect_execution(self.tracker, self.tracker.execution(attempt),
                        probe_fn=lambda *_: result, capture_fn=lambda *args, **kwargs: InputCaptureTests.fake_capture(*args))
                    execution = self.tracker.execution(attempt)
                    self.assertEqual(execution['state'], status)
                    self.assertEqual(execution['runtime']['dataProcessing']['outcome'], outcome)
                    self.assertEqual(execution['runtime']['archiveStatus'], 'complete')
                    self.assertIsNone(execution['runtime']['archiveWarning'])
                    for report in result['reports']:
                        asset = self.tracker.event_payload(attempt, 'output:report:' + report)
                        self.assertEqual(asset['role'], 'data_processing_report')
                    count = self.tracker.pending_count()
                    collector.collect_execution(self.tracker, execution, probe_fn=lambda *_: result)
                    self.assertEqual(self.tracker.pending_count(), count)

    def test_incomplete_or_invalid_filter_receipts_cannot_confirm_an_empty_result(self):
        changes = [{'status': 'running'}, {'status': 'failed'}, {'processed_items': 149},
                   {'api_failed_items': -1}, {'passed_items': False}, {'total_items': '150'},
                   {'error': 'service unavailable'}, {'rejected_items': 149},
                   {'rejected_items': 0, 'invalid_items': 150}]
        for module in (process_module, native.shared('experiment_process')):
            for index, change in enumerate(changes):
                with self.subTest(module=module.__name__, change=change):
                    result = self.filter_probe(f'invalid-{index}', self.filter_receipt(**change), module)
                    self.assertNotIn('dataProcessing', result)
                    self.assertEqual(result['reports'], [])
            for exit_code, signals, expected in ((1, [], 'failed'), (0, [15], 'cancelled')):
                result = self.filter_probe('not-successful', self.filter_receipt(), module,
                    exitCode=exit_code, signals=signals)
                self.assertEqual(result['state'], expected)
                self.assertNotIn('dataProcessing', result)

    def cancellation_workflow(self, attempt, stage='train'):
        execution = self.tracker.execution(attempt)
        return {'workflow_id': 'workflow-1', 'current_stage': stage,
                'context': {'actor_user_id': 'user-1', 'experiment_id': execution['experiment_id']},
                'stages': {stage: {'experiment_task_id': execution['task_id'], 'experiment_attempt_id': attempt}}}

    def test_workflow_cancellation_survives_restart_and_zero_exit_without_outputs(self):
        attempt = self.prepare()['attemptId']
        self.tracker.emit(attempt, 'status', {'status': 'running', 'runtime': {
            'experimentDirectory': '/attempt', 'kind': 'evaluate', 'archiveStatus': 'pending'}})
        with patch.object(launch, 'get_experiment_tracker', return_value=self.tracker):
            launch.request_workflow_cancellation(self.cancellation_workflow(attempt, 'evaluate'))
        self.assertEqual(self.tracker.execution(attempt)['state'], 'running', 'A stop request is not an exit receipt')
        self.tracker.close()
        self.tracker = tracker_module.ExperimentTracker(self.path, 'node-1', self.studio)
        receipt = {'attemptId': attempt, 'state': 'succeeded', 'process': {'exitCode': 0}, 'reports': []}
        collector.collect_execution(self.tracker, self.tracker.execution(attempt), probe_fn=lambda *_: receipt)
        self.assertEqual(self.tracker.execution(attempt)['state'], 'cancelled')
        self.assertEqual(self.tracker.execution(attempt)['runtime']['archiveStatus'], 'complete')
        self.tracker.flush_once()
        statuses = [event['payload']['status'] for event in self.studio.events.values() if event['type'] == 'status']
        self.assertEqual(statuses[-1], 'cancelled')
        self.assertNotIn('succeeded', statuses)

    def test_cancel_request_wins_over_stale_collector_success_but_preserves_completed_history(self):
        for key in ['cancel-before-finish', 'finish-before-cancel']:
            attempt = self.prepare(key)['attemptId']
            self.tracker.emit(attempt, 'status', {'status': 'running'})
            workflow = self.cancellation_workflow(attempt)
            with patch.object(launch, 'get_experiment_tracker', return_value=self.tracker):
                if key == 'finish-before-cancel':
                    self.tracker.emit(attempt, 'status', {'status': 'succeeded'})
                launch.request_workflow_cancellation(workflow)
                launch.request_workflow_cancellation(workflow)
            self.tracker.emit(attempt, 'status', {'status': 'succeeded', 'runtime': {'exitCode': 0}}, preserve_terminal=True)
            expected = 'succeeded' if key == 'finish-before-cancel' else 'cancelled'
            self.assertEqual(self.tracker.execution(attempt)['state'], expected)
            self.assertEqual(self.tracker.latest_payload(attempt, 'status')['status'], expected)

    def test_cancel_request_does_not_claim_a_running_or_missing_process_has_exited(self):
        attempt = self.prepare()['attemptId']
        self.tracker.emit(attempt, 'status', {'status': 'running', 'runtime': {
            'kind': 'train', 'experimentDirectory': '/attempt'}})
        with patch.object(launch, 'get_experiment_tracker', return_value=self.tracker):
            launch.request_workflow_cancellation(self.cancellation_workflow(attempt))
        for status in ['running', 'unknown']:
            collector.observe_execution_status(self.tracker, self.tracker.execution(attempt),
                probe_fn=lambda *_: {'attemptId': attempt, 'state': status, 'process': {'exitCode': None}})
            self.assertEqual(self.tracker.execution(attempt)['state'], status)

    def test_cancellation_checks_workflow_owner_and_stage_identity(self):
        attempt = self.prepare()['attemptId']
        workflow = self.cancellation_workflow(attempt)
        with patch.object(launch, 'get_experiment_tracker', return_value=self.tracker):
            workflow['context']['actor_user_id'] = 'other-user'
            with self.assertRaisesRegex(ValueError, 'workflow owner'):
                launch.request_workflow_cancellation(workflow)
            workflow['context']['actor_user_id'] = 'user-1'
            workflow['stages']['train']['experiment_task_id'] = 'other-task'
            with self.assertRaisesRegex(ValueError, 'workflow owner'):
                launch.request_workflow_cancellation(workflow)
        self.assertNotIn('cancelRequested', self.tracker.execution(attempt)['runtime'])

    def test_running_medical_monitor_records_files_and_scores_without_registering_versions(self):
        attempt = self.prepare()['attemptId']
        self.tracker.emit(attempt, 'status', {'status': 'running', 'runtime': {
            'kind': 'evaluate', 'pid': '123', 'scriptName': 'single_model_evaluation_vpn'}})
        payload = {'status': 'running', 'metrics': {'container_name': 'trainer', 'pid': '123',
            'current_stage': 'record_to_diagnose', 'stages': [
                {'name': 'common_eval', 'path': '/eval/common/results.log', 'status': 'finished',
                 'result': {'average': 51.97, 'content_tail': 'patient response'}},
                {'name': 'medical_eval', 'path': '/eval/medical/results.log', 'status': 'finished',
                 'result': {'average': 50.39}},
                {'name': 'record_to_diagnose', 'path': '/log/diagnosis.json', 'status': 'running',
                 'progress_log': '/log/diagnosis.log', 'progress_summary': {'error_detected': True,
                                                                            'tail_lines': ['patient response']}},
                {'name': 'evaluation_report', 'path': '/log/report.txt', 'status': 'pending'}],
            'artifacts': [{'name': 'responses', 'path': '/log/responses.xlsx', 'exists': True,
                           'result': 'patient response'}]}}
        with patch.object(launch, 'get_experiment_tracker', return_value=self.tracker):
            launch.record_monitor_result(payload)
            previous_sequence = self.tracker.execution(attempt)['last_sequence']
            launch.record_monitor_result(payload)
        execution = self.tracker.execution(attempt)
        self.assertEqual(execution['last_sequence'], previous_sequence)
        self.assertEqual(execution['state'], 'running')
        observation = execution['runtime']['evaluationObservation']
        self.assertEqual(observation['currentStage'], 'record_to_diagnose')
        self.assertEqual([item['scores'].get('average') for item in observation['items'][:2]], [51.97, 50.39])
        self.assertEqual(sum(item['exists'] for item in observation['items']), 3)
        self.assertTrue(observation['items'][2]['logError'])
        self.assertFalse(observation['items'][3]['exists'])
        self.assertNotIn('patient response', json.dumps(observation))
        self.assertIsNone(self.tracker.latest_payload(attempt, 'artifact'))
        self.assertIsNone(self.tracker.latest_payload(attempt, 'evaluation'))

    def test_medical_observation_ignores_invalid_values_and_unmatched_monitors(self):
        observation = launch._medical_evaluation_observation({'stages': [None, {'name': 'score',
            'result': {'average': float('nan'), 'accuracy': True, 'mean_fuzziness': 0.1}},
            {'name': 'bad-progress', 'progress_summary': 'malformed'}], 'artifacts': 'malformed'})
        self.assertEqual(observation['items'][0]['scores'], {'mean_fuzziness': 0.1})
        self.assertFalse(observation['items'][1]['logError'])
        attempt = self.prepare()['attemptId']
        self.tracker.emit(attempt, 'status', {'status': 'running', 'runtime': {
            'pid': '123', 'scriptName': 'single_model_evaluation_vpn'}})
        with patch.object(launch, 'get_experiment_tracker', return_value=self.tracker):
            launch.record_monitor_result({'status': 'running', 'metrics': {
                'container_name': 'other-container', 'pid': '123', 'stages': []}})
        self.assertNotIn('evaluationObservation', self.tracker.execution(attempt)['runtime'])

    def test_finished_training_status_does_not_wait_for_model_archive(self):
        attempt = self.prepare()['attemptId']
        self.tracker.emit(attempt, 'status', {'status': 'running', 'runtime': {
            'experimentDirectory': '/attempt', 'kind': 'train', 'pid': '123', 'archiveStatus': 'pending'}})
        finished = {'attemptId': attempt, 'state': 'succeeded',
            'process': {'pid': 123, 'exitCode': 0},
            'outputs': [{'path': '/output/model', 'kind': 'model', 'role': 'model'}]}
        collector.observe_execution_status(self.tracker, self.tracker.execution(attempt), probe_fn=lambda *_: finished)
        self.assertEqual(self.tracker.execution(attempt)['state'], 'succeeded')
        self.assertEqual(self.tracker.execution(attempt)['runtime']['archiveStatus'], 'pending')
        def failed_capture(*args):
            raise OSError('storage unavailable')
        with self.assertRaisesRegex(OSError, 'storage unavailable'):
            collector.collect_execution(self.tracker, self.tracker.execution(attempt),
                probe_fn=lambda *_: finished, capture_fn=failed_capture)
        self.assertEqual(self.tracker.execution(attempt)['state'], 'succeeded')

    def test_monitor_finish_refreshes_journal_without_trusting_monitor_status(self):
        attempt = self.prepare()['attemptId']
        self.tracker.emit(attempt, 'status', {'status': 'running', 'runtime': {
            'experimentDirectory': '/attempt', 'kind': 'train', 'pid': '123', 'archiveStatus': 'pending'}})
        result = {'attemptId': attempt, 'state': 'running', 'process': {'pid': 123, 'exitCode': None}}
        payload = {'status': 'finished', 'metrics': {'container_name': 'trainer', 'pid': '123'}}
        with patch.object(launch, 'get_experiment_tracker', return_value=self.tracker), \
             patch.object(collector, 'container_json', side_effect=lambda *args: result):
            launch.record_monitor_result(payload)
            self.assertEqual(self.tracker.execution(attempt)['state'], 'running')
            result = {'attemptId': attempt, 'state': 'succeeded', 'process': {'pid': 123, 'exitCode': 0},
                'outputs': [{'path': '/output/model', 'kind': 'model', 'role': 'model'}]}
            launch.record_monitor_result(payload)
        self.assertEqual(self.tracker.execution(attempt)['state'], 'succeeded')

    def export_result(self):
        attempt = self.prepare()['attemptId']
        inputs = [capture.asset_registration(InputCaptureTests.fake_capture('trainer', '/input/' + kind, 'user-1', kind),
            node_id='node-1', container='trainer', owner_id='user-1', source='/input/' + kind, kind=kind, role=role)
            for kind, role in [('model', 'base_model'), ('checkpoint', 'evaluated_model'), ('dataset', 'evaluation_dataset')]]
        self.tracker.emit(attempt, 'status', {'status': 'running', 'runtime': {
            'experimentDirectory': '/attempt', 'kind': 'evaluate', 'inputAssets': inputs}})
        descriptor = {'path': '/versions/exported', 'sourcePath': '/output/merged_checkpoint',
            'kind': 'model', 'role': 'merged_model', 'fingerprint': 'b' * 64,
            'inputPaths': [item['location']['path'] for item in inputs[:2]]}
        result = {'attemptId': attempt, 'state': 'succeeded', 'process': {'exitCode': 0},
            'outputs': [descriptor], 'reports': ['/output/manifest.json'],
            'evaluationManifest': {'results': {}, 'executions': [{'operation': 'export', 'exitCode': 0}]}}
        return attempt, inputs, result

    def test_export_artifact_links_only_base_and_checkpoint_versions(self):
        attempt, inputs, result = self.export_result()
        collector.collect_execution(self.tracker, self.tracker.execution(attempt), probe_fn=lambda *_: result,
            capture_fn=lambda *args, **kwargs: InputCaptureTests.fake_capture(*args))
        output = self.tracker.event_payload(attempt, 'output:model:/versions/exported')
        self.assertEqual(output['logicalName'], 'merged_checkpoint')
        self.assertEqual(output['metadata']['sourcePath'], '/output/merged_checkpoint')
        self.assertEqual(set(output['metadata']['inputVersionIds']), {item['id'] for item in inputs[:2]})
        self.assertNotIn(inputs[2]['id'], output['metadata']['inputVersionIds'])
        actual = self.tracker.latest_payload(attempt, 'configuration')['configuration']['actual']
        self.assertEqual(actual['phaseExecutions'], [{'operation': 'export', 'exitCode': 0}])

    def test_changed_export_snapshot_cannot_be_registered_as_the_evaluated_model(self):
        attempt, _, result = self.export_result()
        result['outputs'][0]['fingerprint'] = 'c' * 64
        with self.assertRaisesRegex(ValueError, 'model used for evaluation'):
            collector.collect_execution(self.tracker, self.tracker.execution(attempt), probe_fn=lambda *_: result,
                capture_fn=lambda *args, **kwargs: InputCaptureTests.fake_capture(*args))
        self.assertEqual(self.tracker.execution(attempt)['state'], 'unknown')
        self.assertIsNone(self.tracker.event_payload(attempt, 'output:model:/versions/exported'))

    def test_export_receipt_cannot_link_an_unrecorded_input(self):
        attempt, _, result = self.export_result()
        result['outputs'][0]['inputPaths'].append('/unrecorded/model')
        with self.assertRaisesRegex(ValueError, 'unrecorded input'):
            collector.collect_execution(self.tracker, self.tracker.execution(attempt), probe_fn=lambda *_: result,
                capture_fn=lambda *args, **kwargs: InputCaptureTests.fake_capture(*args))
        self.assertIsNone(self.tracker.event_payload(attempt, 'output:model:/versions/exported'))

    def test_atomic_collector_event_replay_after_restart(self):
        attempt = self.prepare()['attemptId']
        first = self.tracker.emit_once(attempt, 'artifact', {'id': 'asset-1'}, 'output:1')
        restarted = tracker_module.ExperimentTracker(self.path, 'node-1', self.studio)
        self.assertEqual(restarted.emit_once(attempt, 'artifact', {'id': 'asset-1'}, 'output:1'), first)
        with self.assertRaises(ValueError):
            restarted.emit_once(attempt, 'artifact', {'id': 'asset-2'}, 'output:1')
        self.assertEqual(restarted.pending_count(), 1)

    def test_finished_output_collection_resumes_without_duplicate_events(self):
        attempt = self.prepare()['attemptId']
        self.tracker.emit(attempt, 'status', {'status': 'running', 'runtime': {'experimentDirectory': '/attempt'}})
        probe = lambda *_: {'attemptId': attempt, 'state': 'succeeded', 'process': {'pid': 123, 'exitCode': 0, 'commands': []},
            'outputs': [{'path': '/output/model', 'kind': 'model', 'role': 'model'}], 'reports': []}
        captures = []
        def capture_fn(*args):
            captures.append(args)
            return InputCaptureTests.fake_capture(*args)
        collector.collect_execution(self.tracker, self.tracker.execution(attempt), probe_fn=probe, capture_fn=capture_fn)
        count = self.tracker.pending_count()
        restarted = tracker_module.ExperimentTracker(self.path, 'node-1', self.studio)
        collector.collect_execution(restarted, restarted.execution(attempt), probe_fn=probe, capture_fn=capture_fn)
        self.assertEqual(restarted.pending_count(), count)
        self.assertEqual(len(captures), 1)
        self.assertEqual(restarted.execution(attempt)['runtime']['archiveStatus'], 'complete')

    def test_cancelled_training_keeps_outputs_and_records_stop_signal(self):
        attempt = self.prepare()['attemptId']
        self.tracker.emit(attempt, 'status', {'status': 'running', 'runtime': {
            'experimentDirectory': '/attempt', 'kind': 'train'}})
        probe = lambda *_: {'attemptId': attempt, 'state': 'cancelled',
            'process': {'pid': 123, 'exitCode': -15, 'signals': [15], 'commands': []},
            'outputs': [{'path': '/output/model_merged', 'kind': 'model', 'role': 'model'}], 'reports': []}
        collector.collect_execution(self.tracker, self.tracker.execution(attempt), probe_fn=probe,
            capture_fn=InputCaptureTests.fake_capture)
        execution = self.tracker.execution(attempt)
        self.assertEqual(execution['state'], 'cancelled')
        self.assertEqual(execution['runtime']['signals'], [15])
        self.assertEqual(execution['runtime']['exitCode'], -15)
        self.assertEqual(execution['runtime']['archiveStatus'], 'complete')
        self.assertIsNotNone(self.tracker.event_payload(attempt, 'output:model:/output/model_merged'))

    def test_missing_supervisor_never_implies_success(self):
        attempt = self.prepare()['attemptId']
        self.tracker.emit(attempt, 'status', {'status': 'running', 'runtime': {'experimentDirectory': '/attempt'}})
        collector.collect_execution(self.tracker, self.tracker.execution(attempt),
            probe_fn=lambda *_: {'attemptId': attempt, 'state': 'unknown'})
        self.assertEqual(self.tracker.execution(attempt)['state'], 'unknown')

    def test_success_waits_for_artifact_archive_and_recovers_after_copy_failure(self):
        attempt = self.prepare()['attemptId']
        self.tracker.emit(attempt, 'status', {'status': 'running', 'runtime': {'experimentDirectory': '/attempt', 'kind': 'preprocess'}})
        probe = lambda *_: {'attemptId': attempt, 'state': 'succeeded', 'process': {'exitCode': 0},
            'outputs': [{'path': '/output/data', 'kind': 'dataset', 'role': 'processed_dataset'}], 'reports': []}
        def failed_copy(*args):
            raise OSError('storage unavailable')
        with self.assertRaises(OSError):
            collector.collect_execution(self.tracker, self.tracker.execution(attempt), probe_fn=probe, capture_fn=failed_copy)
        self.assertEqual(self.tracker.execution(attempt)['state'], 'unknown')
        collector.collect_execution(self.tracker, self.tracker.execution(attempt), probe_fn=probe,
            capture_fn=InputCaptureTests.fake_capture)
        self.assertEqual(self.tracker.execution(attempt)['state'], 'succeeded')
        self.assertEqual(self.tracker.execution(attempt)['runtime']['archiveStatus'], 'complete')

    def test_terminal_job_releases_reservation_even_without_model_output(self):
        attempt = self.prepare()['attemptId']
        self.tracker.emit(attempt, 'status', {'status': 'running', 'runtime': {
            'experimentDirectory': '/attempt', 'kind': 'train', 'reservationId': 'reservation-1'}})
        releases = []
        def send(procedure, payload):
            releases.append((procedure, payload))
            return {'status': 'released'}
        probe = lambda *_: {'attemptId': attempt, 'state': 'succeeded', 'process': {'exitCode': 0}, 'outputs': []}
        with patch.object(self.tracker, 'send', side_effect=send):
            collector.collect_execution(self.tracker, self.tracker.execution(attempt), probe_fn=probe)
            collector.collect_execution(self.tracker, self.tracker.execution(attempt), probe_fn=probe)
        self.assertEqual(releases, [('releaseTrainingResourcesForRuntime', {'runtimeNodeId': 'node-1', 'reservationId': 'reservation-1'})])
        self.assertEqual(self.tracker.execution(attempt)['state'], 'unknown')

    def test_zero_exit_without_artifacts_stays_unknown_without_event_spam(self):
        attempt = self.prepare()['attemptId']
        self.tracker.emit(attempt, 'status', {'status': 'running', 'runtime': {'experimentDirectory': '/attempt', 'kind': 'train'}})
        probe = lambda *_: {'attemptId': attempt, 'state': 'succeeded', 'process': {'exitCode': 0}, 'outputs': []}
        collector.collect_execution(self.tracker, self.tracker.execution(attempt), probe_fn=probe)
        count = self.tracker.pending_count()
        collector.collect_execution(self.tracker, self.tracker.execution(attempt), probe_fn=probe)
        self.assertEqual(self.tracker.pending_count(), count)
        self.assertEqual(self.tracker.execution(attempt)['state'], 'unknown')

    def test_wrong_execution_journal_rejected(self):
        attempt = self.prepare()['attemptId']
        self.tracker.emit(attempt, 'status', {'status': 'running', 'runtime': {'experimentDirectory': '/attempt'}})
        with self.assertRaises(ValueError):
            collector.collect_execution(self.tracker, self.tracker.execution(attempt),
                probe_fn=lambda *_: {'attemptId': 'another-attempt', 'state': 'succeeded'})
        self.assertEqual(self.tracker.execution(attempt)['state'], 'running')

    def test_unattributed_report_keeps_evaluation_pending_without_duplicate_events(self):
        attempt = self.prepare()['attemptId']
        self.tracker.emit(attempt, 'status', {'status': 'running', 'runtime': {
            'experimentDirectory': '/attempt', 'kind': 'evaluate'}})
        result = {'attemptId': attempt, 'state': 'succeeded', 'process': {'exitCode': 0},
                  'outputs': [], 'reports': [], 'unattributedReports': ['/shared/daily-report.txt']}
        with patch.object(collector, 'container_asset') as capture_report:
            collector.collect_execution(self.tracker, self.tracker.execution(attempt), probe_fn=lambda *_: result,
                capture_fn=capture_report)
            count = self.tracker.pending_count()
            collector.collect_execution(self.tracker, self.tracker.execution(attempt), probe_fn=lambda *_: result,
                capture_fn=capture_report)
            capture_report.assert_not_called()
        execution = self.tracker.execution(attempt)
        self.assertEqual(execution['state'], 'unknown')
        self.assertEqual(execution['runtime']['archiveStatus'], 'awaiting_outputs')
        self.assertEqual(execution['runtime']['unattributedReportPaths'], ['/shared/daily-report.txt'])
        self.assertIn('not bound', execution['runtime']['archiveWarning'])
        self.assertEqual(self.tracker.pending_count(), count)

    def test_changed_input_disables_direct_evaluation_comparison(self):
        attempt = self.prepare()['attemptId']
        input_assets = [dict(InputCaptureTests.fake_capture('trainer', '/model', 'user-1', 'model'))]
        version = capture.asset_registration(input_assets[0], node_id='node-1', container='trainer', owner_id='user-1',
                                             source='/model', kind='model', role='evaluated_model')
        self.tracker.emit(attempt, 'status', {'status': 'running', 'runtime': {
            'experimentDirectory': '/attempt', 'inputAssets': [version]}})
        result = {'attemptId': attempt, 'state': 'succeeded', 'process': {'exitCode': 0},
                  'outputs': [], 'reports': [], 'evaluationManifest': {'experimentProtocolComplete': True}}
        with patch.object(collector, 'register_evaluation') as register:
            collector.collect_execution(self.tracker, self.tracker.execution(attempt), probe_fn=lambda *_: result,
                capture_fn=lambda *args, **kwargs: {'fingerprint': 'changed'})
        self.assertFalse(register.call_args.args[2]['experimentProtocolComplete'])
        self.assertEqual(self.tracker.execution(attempt)['runtime']['inputIntegrityAfterExecution'], 'unverified')

    def test_modified_checkpoint_execution_copy_is_detected_even_when_original_is_unchanged(self):
        attempt = self.prepare()['attemptId']
        version = capture.asset_registration(InputCaptureTests.fake_capture('trainer', '/checkpoint', 'user-1', 'checkpoint'),
            node_id='node-1', container='trainer', owner_id='user-1', source='/checkpoint', kind='checkpoint', role='evaluated_model')
        transformation = {'checkpointVersionId': version['id'], 'executionPath': '/attempt/inputs/checkpoint',
                          'executionFingerprint': 'd' * 64}
        self.tracker.emit(attempt, 'status', {'status': 'running', 'runtime': {
            'experimentDirectory': '/attempt', 'inputAssets': [version], 'inputTransforms': [transformation]}})
        result = {'attemptId': attempt, 'state': 'failed', 'process': {'exitCode': 1}, 'outputs': [], 'reports': []}
        reads = []
        def capture_input(container, path, *args, **kwargs):
            reads.append(path)
            return {'fingerprint': version['fingerprint']}
        collector.collect_execution(self.tracker, self.tracker.execution(attempt), probe_fn=lambda *_: result, capture_fn=capture_input)
        self.assertEqual(reads, [version['location']['path'], transformation['executionPath']])
        runtime = self.tracker.execution(attempt)['runtime']
        self.assertEqual(runtime['inputIntegrityAfterExecution'], 'unverified')
        self.assertEqual([check['matches'] for check in runtime['inputVerification']], [True, False])
        self.assertEqual(self.tracker.latest_payload(attempt, 'configuration')['configuration']['actual']['inputTransforms'], [transformation])

    def test_unknown_sample_counts_are_not_fabricated(self):
        attempt = self.prepare()['attemptId']
        self.tracker.emit(attempt, 'status', {'status': 'running', 'runtime': {'inputAssets': [
            {'id': 'model-1', 'role': 'evaluated_model', 'metadata': {}},
            {'id': 'data-1', 'role': 'evaluation_dataset', 'metadata': {'datasetName': 'mmlu'}}]}})
        collector.register_evaluation(self.tracker, self.tracker.execution(attempt),
            {'status': 'finished', 'datasets': ['mmlu'], 'results': {'model_fir': {'mmlu': {'average': 60.0}}}}, [])
        payload = self.tracker.event_payload(attempt, 'evaluation:model_fir')
        self.assertIsNone(payload['expectedCount'])
        self.assertIsNone(payload['metrics'][0]['sampleCount'])
        self.assertFalse(payload['protocolComplete'])

    def test_legacy_checkpoint_report_without_frozen_model_binding_is_not_comparable(self):
        execution = {'attempt_id': 'legacy-ckpt', 'runtime': {'inputAssets': [
            {'id': 'data-1', 'role': 'evaluation_dataset', 'metadata': {'datasetName': 'mmlu'}}]}}
        report = {'status': 'finished', 'datasets': ['mmlu'], 'experimentProtocolComplete': True,
            'experimentProtocol': {'version': 1}, 'results': {'checkpoint': {'mmlu': {'average': 60.0, 'sampleCount': 1}}}}
        model = {'id': 'merged-1', 'kind': 'model', 'logicalName': 'merged_checkpoint',
                 'fingerprint': 'a' * 64, 'location': {'path': '/versions/model'}}
        with patch.object(self.tracker, 'emit_once') as emit:
            collector.register_evaluation(self.tracker, execution, report, [model])
            self.assertFalse(emit.call_args.args[2]['protocolComplete'])
            self.assertIn('merged_model_snapshot_binding', emit.call_args.args[2]['protocol']['missing'])
            report['modelArtifacts'] = [{'path': model['location']['path'], 'fingerprint': model['fingerprint']}]
            collector.register_evaluation(self.tracker, execution, report, [model])
            self.assertTrue(emit.call_args.args[2]['protocolComplete'])


class DatasetUploadTests(unittest.TestCase):
    setUp = TrackerTests.setUp
    tearDown = TrackerTests.tearDown

    def upload(self):
        source = Path(self.temp.name) / 'input'
        source.mkdir()
        (source / 'data.json').write_text('[{"text":"sample"}]', encoding='utf-8')
        handle = upload_module.prepare_dataset_upload(self.tracker, actor_user_id='user-1', container='trainer',
            source=source, destination='/datasets/my-data', filename='data.zip', archive_sha256='a' * 64, dataset_type='raw')
        self.content = assets.manifest(source)
        return handle['attemptId']

    def capture(self, *args, **kwargs):
        return {**self.content, 'path': '/versions/data'}

    def test_complete_upload_recovers_after_restart_and_uses_ordered_outbox(self):
        attempt = self.upload()
        restarted = tracker_module.ExperimentTracker(self.path, 'node-1', self.studio)
        self.assertEqual([item['attempt_id'] for item in restarted.collectable_executions()], [attempt])
        upload_module.collect_dataset_upload(restarted, restarted.execution(attempt), capture_fn=self.capture)
        self.assertEqual(restarted.execution(attempt)['state'], 'succeeded')
        version = restarted.event_payload(attempt, 'uploaded-dataset')
        self.assertEqual(version['metadata']['archiveSha256'], 'a' * 64)
        self.assertEqual(version['metadata']['sourcePath'], '/datasets/my-data')
        self.assertEqual(version['fingerprint'], self.content['fingerprint'])
        count = restarted.pending_count()
        upload_module.collect_dataset_upload(restarted, restarted.execution(attempt), capture_fn=self.capture)
        self.assertEqual(restarted.pending_count(), count)
        self.assertEqual(restarted.collectable_executions(), [])
        restarted.flush_once()
        self.assertEqual(restarted.pending_count(), 0)

    def test_partial_upload_cannot_create_an_asset_and_can_recover(self):
        attempt = self.upload()
        calls = []
        def partial(*args, **kwargs):
            calls.append(kwargs.get('operation'))
            return {'fingerprint': 'different'}
        with self.assertRaises(ValueError):
            upload_module.collect_dataset_upload(self.tracker, self.tracker.execution(attempt), capture_fn=partial)
        self.assertEqual(calls, ['manifest'])
        self.assertIsNone(self.tracker.event_payload(attempt, 'uploaded-dataset'))
        upload_module.collect_dataset_upload(self.tracker, self.tracker.execution(attempt), capture_fn=self.capture)
        self.assertEqual(self.tracker.execution(attempt)['state'], 'succeeded')

    def test_archive_retry_after_asset_event_does_not_require_mutable_source(self):
        attempt = self.upload()
        emit = self.tracker.emit_once
        def interrupted(*args):
            if args[-1] == 'upload-configuration':
                raise OSError('interrupted after immutable snapshot')
            return emit(*args)
        with patch.object(self.tracker, 'emit_once', side_effect=interrupted), self.assertRaises(OSError):
            upload_module.collect_dataset_upload(self.tracker, self.tracker.execution(attempt), capture_fn=self.capture)
        with patch.object(upload_module, 'container_asset', side_effect=AssertionError('must reuse recorded version')):
            upload_module.collect_dataset_upload(self.tracker, self.tracker.execution(attempt),
                capture_fn=lambda *args, **kwargs: self.fail('must reuse recorded version'))
        self.assertEqual(self.tracker.execution(attempt)['state'], 'succeeded')


class NativeDeploymentTests(unittest.TestCase):
    def setUp(self):
        TrackerTests.setUp(self)
        self.root = Path(self.temp.name)
        model_path = self.root / 'model'
        model_path.mkdir()
        (model_path / 'model.safetensors').write_bytes(b'weights')
        self.cfg = {'ENV': {'MODEL_PATH': str(self.root), 'MODEL_NAME': 'model', 'HOST_IP': '127.0.0.1'},
                    'PORTS': {'VLLM_OPENAI_PORT': 8100}, 'RUNTIME': {'API_KEY': 'private'}}
        self.model = native.freeze_service_model(self.cfg, state_dir=self.root, owner_id='user-1', node_id='node-1')
        self.logs = self.root / 'service'
        self.logs.mkdir()
        (self.logs / 'pids').mkdir()
        (self.logs / 'pids' / 'vllm.pid').write_text('456', encoding='utf-8')
        self.instance = {'instance_id': 'service-1', 'run_id': 'run-1', 'owner_user_id': 'user-1', 'model': 'model',
            'log_dir': str(self.logs), 'pid_dir': str(self.logs / 'pids'), 'meta_file': str(self.logs / 'meta.json'), 'status': 'starting'}

    tearDown = TrackerTests.tearDown

    def save(self, value):
        process_module.atomic_json(self.instance['meta_file'], value)

    def launch(self):
        def spawn(*args, **kwargs):
            meta = process_module.read_json(self.instance['meta_file'])
            self.assertEqual(meta['deployment_registration'], 'pending')
            self.assertTrue(meta['experiment']['attemptId'])
            self.assertEqual(kwargs['env']['MEDFLOW_DEPLOYMENT_DIRECTORY'], str(self.logs / 'experiment'))
            return types.SimpleNamespace(pid=123)
        with patch.object(native, 'get_tracker', return_value=self.tracker), patch.object(native.subprocess, 'Popen', side_effect=spawn):
            native.start_deployment(['bash', '/start.sh'], {'SERVICE_RUN_ID': 'run-1'}, instance=self.instance, cfg=self.cfg,
                state_dir=self.root, node_id='node-1', model=self.model, save_fn=self.save,
                context={'experimentId': 'training-exp', 'workflowId': 'workflow-1'})
        return self.instance['experiment']['attemptId']

    def test_authenticated_owner_snapshot_and_deployment_use_studio_actor(self):
        owner = 'auth:user-1'
        version = native.freeze_service_model(self.cfg, state_dir=self.root, owner_id=owner, node_id='node-1')
        self.assertEqual(version['id'], self.model['id'])
        self.assertEqual(version['location'], self.model['location'])
        self.assertEqual((Path(self.cfg['ENV']['MODEL_SNAPSHOT_PATH']) / 'model.safetensors').read_bytes(), b'weights')
        self.instance['owner_user_id'] = owner
        attempt = self.launch()
        with self.tracker.connect() as db:
            request = json.loads(db.execute('SELECT request_json FROM experiment_intent').fetchone()[0])
        self.assertEqual(request['actorUserId'], 'user-1')
        self.assertEqual(self.tracker.execution(attempt)['actor_user_id'], 'user-1')
        self.assertEqual(process_module.read_json(self.instance['meta_file'])['owner_user_id'], owner)
        self.observe()
        self.finish(attempt)
        native.collect_deployment(self.tracker, self.tracker.execution(attempt))
        self.assertEqual(self.tracker.execution(attempt)['state'], 'succeeded')

    def test_invalid_owner_or_node_never_reaches_snapshot(self):
        for owner, node in [('auth:', 'node-1'), ('auth:../other', 'node-1'),
                            ('auth:auth:user-1', 'node-1'), ('user:user-1:session', 'node-1'),
                            ('user-1/../../other', 'node-1'), ('auth:user-1', 'auth:node-1')]:
            with self.subTest(owner=owner, node=node), patch.object(native.shared('experiment_assets'), 'snapshot') as snapshot:
                with self.assertRaises(ValueError):
                    native.local_asset(self.root, owner, node, self.root / 'model', 'model', 'served_model')
                snapshot.assert_not_called()

    def test_deployment_does_not_create_benchmark_log_source(self):
        self.launch()
        metadata = process_module.read_json(self.logs / 'experiment' / 'execution.json')
        self.assertNotEqual(metadata.get('kind'), 'benchmark')
        self.assertNotIn('businessStdout', metadata)

    def observer(self, *, commands=None, identity_fn=None):
        return types.SimpleNamespace(read_json=process_module.read_json, atomic_json=process_module.atomic_json,
            redact=process_module.redact, entrypoint_fingerprints=lambda command: {'/vllm': 'sha'},
            process_identity=identity_fn or (lambda pid: {'bootId': 'boot', 'startTicks': '1'}),
            descendant_commands=lambda pid: commands if commands is not None else [['vllm', 'serve', self.model['location']['path']]])

    def observe(self, **kwargs):
        return native.observe_deployment(self.logs / 'experiment', process_module=kwargs.get('process_module', self.observer()),
            environment_fn=kwargs.get('environment_fn', lambda pid: b'SERVICE_RUN_ID=run-1\0'),
            listener_fn=kwargs.get('listener_fn', lambda pid, port: [pid]),
            fetch_fn=kwargs.get('fetch_fn', lambda url: {'data': [{'id': 'model'}]}))

    def finish(self, attempt, registration='confirmed'):
        self.instance['deployment_registration'] = registration
        self.save(self.instance)
        process_module.atomic_json(self.logs / 'experiment' / 'process.json', {'attemptId': attempt, 'exitCode': 0})

    def test_deployment_archives_verified_readiness_and_fixed_model(self):
        attempt = self.launch()
        receipt = self.observe()
        self.finish(attempt)
        native.collect_deployment(self.tracker, self.tracker.execution(attempt))
        self.assertEqual(self.tracker.execution(attempt)['state'], 'succeeded')
        self.assertEqual(receipt['modelVersionId'], self.model['id'])
        self.assertTrue(self.tracker.event_payload(attempt, 'deployment-receipt'))
        request = next(iter(self.studio.requests))
        self.assertEqual(request, 'deploy-service-1')
        with self.tracker.connect() as db:
            encoded = db.execute('SELECT request_json FROM experiment_intent').fetchone()[0]
        self.assertNotIn('private', encoded)
        self.assertEqual(json.loads(encoded)['experimentId'], 'training-exp')
        self.assertEqual(json.loads(encoded)['kind'], 'deploy')
        count = self.tracker.pending_count()
        native.collect_deployment(self.tracker, self.tracker.execution(attempt))
        self.assertEqual(count, self.tracker.pending_count())

    def test_deployment_binds_the_persisted_service_reservation(self):
        self.instance['resource'] = {'reservation_id': 'lease-for-service', 'runtime_node_id': 'node-1'}
        self.launch()
        with self.tracker.connect() as db:
            request = json.loads(db.execute('SELECT request_json FROM experiment_intent').fetchone()[0])
        self.assertEqual(request['resourceReservationId'], 'lease-for-service')
        context = native.inference_experiment_context(self.instance, {'resourceReservationId': 'forged-lease', 'experimentId': 'parent'})
        self.assertEqual(context, {'experimentId': 'parent', 'resourceReservationId': 'lease-for-service'})
        self.assertNotIn('resourceReservationId', native.inference_experiment_context({}, {'resourceReservationId': 'forged-lease'}))

    def test_serving_snapshot_renews_from_the_allocated_worker_before_and_after_copy(self):
        requests = []
        tracker = types.SimpleNamespace(send=lambda method, payload: requests.append((method, payload)))
        with patch.object(native, 'get_tracker', return_value=tracker):
            native.freeze_service_model(self.cfg, state_dir=self.root, owner_id='user-1', node_id='allocated-node',
                reservation_id='inference-lease')
        self.assertEqual(requests, [('renewTrainingResourcesForRuntime',
            {'runtimeNodeId': 'allocated-node', 'reservationId': 'inference-lease'})] * 2)

    def test_expired_lease_prevents_serving_snapshot_copy(self):
        tracker = types.SimpleNamespace(send=lambda *args: (_ for _ in ()).throw(ValueError('lease expired')))
        with patch.object(native, 'get_tracker', return_value=tracker), patch.object(native, 'local_asset') as capture_asset, \
             self.assertRaisesRegex(ValueError, 'lease expired'):
            native.freeze_service_model(self.cfg, state_dir=self.root, owner_id='user-1', node_id='allocated-node',
                reservation_id='inference-lease')
        capture_asset.assert_not_called()

    def test_ready_process_waits_for_resource_registration_confirmation(self):
        attempt = self.launch()
        self.observe()
        self.finish(attempt, registration='pending')
        native.collect_deployment(self.tracker, self.tracker.execution(attempt))
        self.assertEqual(self.tracker.execution(attempt)['state'], 'unknown')
        count = self.tracker.pending_count()
        native.collect_deployment(self.tracker, self.tracker.execution(attempt))
        self.assertEqual(count, self.tracker.pending_count())
        self.finish(attempt)
        native.collect_deployment(self.tracker, self.tracker.execution(attempt))
        self.assertEqual(self.tracker.execution(attempt)['state'], 'succeeded')

    def test_registration_rollback_keeps_deployment_failed_despite_readiness(self):
        attempt = self.launch()
        self.observe()
        self.finish(attempt, registration='failed')
        native.collect_deployment(self.tracker, self.tracker.execution(attempt))
        self.assertEqual(self.tracker.execution(attempt)['state'], 'failed')
        self.assertTrue(self.tracker.event_payload(attempt, 'deployment-receipt'))

    def test_startup_exit_zero_without_receipt_is_not_success(self):
        attempt = self.launch()
        self.finish(attempt)
        with self.assertRaisesRegex(ValueError, 'without a verified readiness'):
            native.collect_deployment(self.tracker, self.tracker.execution(attempt))
        self.assertEqual(self.tracker.execution(attempt)['state'], 'unknown')

    def test_readiness_rejects_wrong_owner_model_or_api(self):
        self.launch()
        for overrides in ({'environment_fn': lambda pid: b'SERVICE_RUN_ID=another-run\0'},
                          {'process_module': self.observer(commands=[['vllm', 'serve', '/mutable/model']])},
                          {'listener_fn': lambda pid, port: []},
                          {'fetch_fn': lambda url: {'data': [{'id': 'wrong-model'}]}}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.observe(**overrides)
        self.assertFalse((self.logs / 'experiment' / 'readiness.json').exists())

    def test_readiness_rejects_pid_reuse_during_api_probe(self):
        self.launch()
        identities = iter([{'startTicks': '1'}, {'startTicks': '2'}])
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            self.observe(process_module=self.observer(identity_fn=lambda pid: next(identities)))

    def test_readiness_uses_pre_exec_record_after_process_title_changes(self):
        attempt = self.launch()
        process_module.atomic_json(self.logs / 'experiment' / 'service-launch.json', {
            'attemptId': attempt, 'runId': 'run-1', 'pid': 456, 'processIdentity': {'bootId': 'boot', 'startTicks': '1'},
            'command': ['vllm', 'serve', self.model['location']['path']]})
        receipt = self.observe(process_module=self.observer(commands=[['VLLM::APIServer']]), environment_fn=lambda pid: b'')
        self.assertEqual(receipt['launchCommand'][2], self.model['location']['path'])
        self.assertEqual(receipt['observedCommands'], [['VLLM::APIServer']])
        service = {**self.instance, 'experiment_model_version': self.model}
        with patch.object(native.shared('experiment_process'), 'process_identity', return_value=receipt['processIdentity']):
            self.assertEqual(native.observed_model(service, self.cfg, 'node-1', 'user-1', [['VLLM::APIServer']])['integrity'], 'verified')

    def test_pre_exec_record_cannot_authorize_a_reused_service_pid(self):
        attempt = self.launch()
        process_module.atomic_json(self.logs / 'experiment' / 'service-launch.json', {
            'attemptId': attempt, 'runId': 'run-1', 'pid': 456, 'processIdentity': {'bootId': 'boot', 'startTicks': 'old'},
            'command': ['vllm', 'serve', self.model['location']['path']]})
        with self.assertRaisesRegex(ValueError, 'not owned'):
            self.observe(process_module=self.observer(commands=[['VLLM::APIServer']]), environment_fn=lambda pid: b'')

    def test_duplicate_deployment_cannot_overwrite_live_instance_metadata(self):
        self.launch()
        self.instance['deployment_registration'] = 'confirmed'
        self.save(self.instance)
        before = Path(self.instance['meta_file']).read_bytes()
        with patch.object(native, 'get_tracker', return_value=self.tracker), patch.object(native.subprocess, 'Popen') as spawn, \
             self.assertRaisesRegex(ValueError, 'already claimed'):
            native.start_deployment(['bash', '/start.sh'], {}, instance=self.instance, cfg=self.cfg,
                state_dir=self.root, node_id='node-1', model=self.model, save_fn=self.save,
                context={'experimentId': 'training-exp', 'workflowId': 'workflow-1'})
        spawn.assert_not_called()
        self.assertEqual(Path(self.instance['meta_file']).read_bytes(), before)

    def record_serving_identity(self, attempt):
        recorded = {'bootId': 'boot', 'startTicks': '1'}
        process_module.atomic_json(self.logs / 'experiment' / 'service-launch.json', {
            'attemptId': attempt, 'runId': 'run-1', 'pid': 456, 'processIdentity': recorded})
        return recorded

    def test_service_ownership_uses_live_identity_and_ancestry_without_environment(self):
        attempt = self.launch()
        recorded = self.record_serving_identity(attempt)
        child = {'bootId': 'boot', 'startTicks': '2'}
        identities = {456: recorded, 457: child, 789: {'bootId': 'boot', 'startTicks': '3'}}
        observer = self.observer(identity_fn=identities.get)
        def owns(pid):
            return native.deployment_owned_identity(self.instance, pid, process_module=observer,
                parent_fn=lambda value: {457: 456}.get(value, 1))
        self.assertEqual(owns(456), recorded)
        self.assertEqual(owns(457), child)
        self.assertIsNone(owns(789))
        identities[456] = {'bootId': 'boot', 'startTicks': 'new-process'}
        self.assertIsNone(owns(456))
        self.assertIsNone(owns(457))

    def test_service_ownership_rejects_foreign_attempt_run_and_instance(self):
        attempt = self.launch()
        self.record_serving_identity(attempt)
        for change in ({'run_id': 'other-run'}, {'instance_id': 'other-instance'},
                       {'experiment': {'attemptId': 'other-attempt'}}):
            with self.subTest(change=change):
                self.assertIsNone(native.deployment_owned_identity({**self.instance, **change}, 456,
                    process_module=self.observer(), parent_fn=lambda pid: 1))

    def test_service_ownership_rechecks_identity_after_traversing_ancestors(self):
        attempt = self.launch()
        recorded = self.record_serving_identity(attempt)
        values = iter([recorded, recorded, recorded, {'bootId': 'boot', 'startTicks': 'reused'}])
        self.assertIsNone(native.deployment_owned_identity(self.instance, 456,
            process_module=self.observer(identity_fn=lambda pid: next(values)), parent_fn=lambda pid: 1))

    def test_active_supervisor_proves_starting_service_ownership(self):
        attempt = self.launch()
        recorded = {'bootId': 'boot', 'startTicks': '1'}
        journal = {'attemptId': attempt, 'pid': 123, 'identity': recorded, 'exitCode': None}
        path = self.logs / 'experiment' / 'process.json'
        process_module.atomic_json(path, journal)
        self.assertEqual(native.deployment_owned_identity(self.instance, 456,
            process_module=self.observer(), parent_fn=lambda pid: 123), recorded)
        process_module.atomic_json(path, {**journal, 'exitCode': 0})
        self.assertIsNone(native.deployment_owned_identity(self.instance, 456,
            process_module=self.observer(), parent_fn=lambda pid: 123))

    def test_stop_signals_starting_supervisor_but_preserves_finished_deployment(self):
        attempt = self.launch()
        recorded = {'bootId': 'boot', 'startTicks': '1'}
        journal = {'attemptId': attempt, 'pid': 123, 'identity': recorded, 'exitCode': None}
        path = self.logs / 'experiment' / 'process.json'
        process_module.atomic_json(path, journal)
        signals = []
        send = lambda pid, signum: signals.append((pid, signum))
        self.assertTrue(native.cancel_starting_deployment(self.instance, process_module=self.observer(), signal_fn=send))
        self.assertEqual(signals, [(123, native.signal.SIGTERM)])
        process_module.atomic_json(path, {**journal, 'exitCode': 0})
        self.assertFalse(native.cancel_starting_deployment(self.instance, process_module=self.observer(), signal_fn=send))
        self.assertEqual(len(signals), 1)

    def test_stop_does_not_signal_reused_supervisor_pid(self):
        attempt = self.launch()
        process_module.atomic_json(self.logs / 'experiment' / 'process.json', {
            'attemptId': attempt, 'pid': 123, 'identity': {'bootId': 'old-boot', 'startTicks': '1'}, 'exitCode': None})
        signals = []
        self.assertFalse(native.cancel_starting_deployment(self.instance, process_module=self.observer(),
            signal_fn=lambda *args: signals.append(args)))
        self.assertEqual(signals, [])

    def test_cancelled_deployment_retains_readiness_without_becoming_successful(self):
        attempt = self.launch()
        self.observe()
        self.finish(attempt)
        process_module.atomic_json(self.logs / 'experiment' / 'process.json', {
            'attemptId': attempt, 'exitCode': -15, 'signals': [15]})
        native.collect_deployment(self.tracker, self.tracker.execution(attempt))
        self.assertEqual(self.tracker.execution(attempt)['state'], 'cancelled')
        self.assertEqual(self.tracker.execution(attempt)['runtime']['exitCode'], -15)
        self.assertEqual(self.tracker.execution(attempt)['runtime']['signals'], [15])
        self.assertTrue(self.tracker.event_payload(attempt, 'deployment-receipt'))


class NativeServiceOperationTests(unittest.TestCase):
    def setUp(self):
        TrackerTests.setUp(self)
        self.root = Path(self.temp.name)
        handle = self.tracker.prepare(request_key='deploy-service-1', actor_user_id='user-1', name='Deploy model',
            kind='deploy', container='inference-worker', spec={'kind': 'deploy', 'serviceInstanceId': 'service-1'})
        self.instance = {'instance_id': 'service-1', 'run_id': 'run-1', 'experiment': handle}
        self.attempt = handle['attemptId']
        self.tracker.emit(self.attempt, 'status', {'status': 'running', 'runtime': {
            'operation': 'service_deploy', 'serviceInstanceId': 'service-1', 'serviceRunId': 'run-1'}})

    tearDown = TrackerTests.tearDown

    def test_stop_operations_survive_retries_and_preserve_successful_deployment(self):
        self.tracker.emit(self.attempt, 'status', {'status': 'succeeded'})
        original = self.tracker.execution(self.attempt)
        values = dict(state_dir=self.root, node_id='node-1', operation_id='stop-one', actor='auth:admin-1', trigger='admin', cause='restart')
        with patch.object(native, 'get_tracker', return_value=self.tracker):
            native.record_service_stop(self.instance, status='requested', **values)
            for _ in range(2):
                native.record_service_stop(self.instance, status='succeeded', result='PID and port released', release_ready=True, **values)
        execution = self.tracker.execution(self.attempt)
        self.assertEqual(execution['state'], 'succeeded')
        self.assertEqual(execution['runtime'], original['runtime'])
        self.assertEqual(execution['last_sequence'], original['last_sequence'] + 2)
        payload = self.tracker.latest_payload(self.attempt, 'service_operation')
        self.assertEqual(payload['actorUserId'], 'admin-1')
        self.assertEqual(payload['serviceInstanceId'], 'service-1')
        self.assertEqual(payload['operationId'], 'stop-one')
        self.assertEqual(payload['cause'], 'restart')
        self.assertTrue(payload['releaseReady'])
        self.tracker.flush_once()
        self.assertEqual(self.tracker.pending_count(), 0)

    def test_failed_stop_is_independent_and_wrong_instance_cannot_borrow_deployment(self):
        values = dict(state_dir=self.root, node_id='node-1', operation_id='stop-failure')
        with patch.object(native, 'get_tracker', return_value=self.tracker):
            native.record_service_stop(self.instance, status='requested', **values)
            native.record_service_stop(self.instance, status='failed', result='port remains busy', release_ready=False, **values)
            for changed in ({'instance_id': 'other'}, {'run_id': 'other'}):
                with self.assertRaisesRegex(ValueError, 'deployment execution'):
                    native.record_service_stop({**self.instance, **changed}, status='requested', **values)
        self.assertEqual(self.tracker.execution(self.attempt)['state'], 'running')
        self.assertEqual(self.tracker.latest_payload(self.attempt, 'service_operation')['status'], 'failed')
        with patch.object(native, 'get_tracker') as get_tracker:
            self.assertFalse(native.record_service_stop({'instance_id': 'legacy'}, status='requested', **values))
            get_tracker.assert_not_called()


class WorkflowExperimentTests(unittest.TestCase):
    setUp = TrackerTests.setUp
    tearDown = TrackerTests.tearDown

    def test_lifecycle_is_queued_after_train_success_and_survives_restart(self):
        dependencies = workflow_module.WorkflowDependencies(*[lambda *args: {} for _ in range(8)])
        manager = workflow_module.WorkflowManager(str(Path(self.temp.name) / 'workflow-status.db'), dependencies,
            auto_start_worker=False, on_update=self.tracker.record_workflow)
        workflow = manager.create('user-1', 'dataset')
        handle = self.tracker.prepare(request_key='workflow-train', actor_user_id='user-1', name='train', kind='train',
            container='trainer', spec={}, context={'workflowId': workflow['workflow_id']})
        attempt = handle['attemptId']
        self.tracker.emit(attempt, 'status', {'status': 'succeeded'})
        workflow['stages']['train']['status'] = 'finished'
        workflow['current_stage'] = 'evaluate'
        manager._save(workflow)
        payload = self.tracker.latest_payload(attempt, 'workflow_status')
        self.assertEqual((payload['status'], payload['currentStage']), ('running', 'evaluate'))
        self.assertEqual(self.tracker.execution(attempt)['state'], 'succeeded')
        sequence = self.tracker.execution(attempt)['last_sequence']
        manager._save(workflow)
        self.assertEqual(self.tracker.execution(attempt)['last_sequence'], sequence, 'Unchanged polls must not add events')
        self.studio.offline = True
        self.tracker.flush_once()
        restarted = tracker_module.ExperimentTracker(self.path, 'node-1', self.studio)
        manager.on_update = restarted.record_workflow
        for status in ('failed', 'running', 'stopping', 'stopped', 'running', 'finished'):
            workflow['status'] = status
            manager._save(workflow, allow_terminal_override=True)
            self.assertEqual(restarted.latest_payload(attempt, 'workflow_status')['status'], status)
            self.assertEqual(restarted.execution(attempt)['state'], 'succeeded')
        self.studio.offline = False
        restarted.flush_once()
        self.assertEqual(restarted.pending_count(), 0)
        self.assertEqual(list(self.studio.events.values())[-1]['payload']['status'], 'finished')
        restarted.close()

    def test_status_read_repairs_missing_report_and_ignores_stale_or_unrelated_workflows(self):
        dependencies = workflow_module.WorkflowDependencies(*[lambda *args: {} for _ in range(8)])
        manager = workflow_module.WorkflowManager(str(Path(self.temp.name) / 'workflow-read.db'), dependencies,
            auto_start_worker=False, on_update=self.tracker.record_workflow)
        workflow = manager.create('user-1', 'dataset')
        self.tracker.prepare(request_key='workflow-train', actor_user_id='user-1', name='train', kind='train',
            container='trainer', spec={}, context={'workflowId': workflow['workflow_id']})
        manager.get_status_snapshot(workflow['workflow_id'])
        attempt = 'attempt-workflow-train'
        self.assertEqual(self.tracker.latest_payload(attempt, 'workflow_status')['status'], 'running')
        before = self.tracker.execution(attempt)['last_sequence']
        self.tracker.record_workflow({**workflow, 'workflow_id': 'unrelated'})
        self.tracker.record_workflow({**workflow, 'status': 'failed', 'updated_at': workflow['updated_at'] - 1})
        self.assertEqual(self.tracker.execution(attempt)['last_sequence'], before)

    def test_stopped_train_and_evaluation_resume_keep_experiment_and_task_with_a_new_request_key(self):
        path = Path(__file__).parents[1] / 'runtime_agent.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == '_workflow_experiment_context')
        for stage_name in ['train', 'evaluate']:
            dependencies = workflow_module.WorkflowDependencies(*[lambda *args: {} for _ in range(8)])
            db_path = str(Path(self.temp.name) / f'{stage_name}-workflow.db')
            manager = workflow_module.WorkflowManager(db_path, dependencies, auto_start_worker=False)
            workflow = manager.create('user-1', 'dataset', auto_start=False)
            workflow['current_stage'] = stage_name
            workflow['context']['experiment_id'] = 'exp-original'
            workflow['context']['experiment_launches'] = {stage_name: {'requestKey': 'old-request'}}
            workflow['stages'][stage_name].update(status='running', experiment_id='exp-original',
                experiment_task_id=f'{stage_name}-task', experiment_attempt_id='old-attempt')
            manager._save(workflow)
            manager.stop('user-1', invoke_stop_task=False)
            manager = workflow_module.WorkflowManager(db_path, dependencies, auto_start_worker=False)
            resumed = manager.resume('user-1')
            namespace = {'Dict': dict, 'Any': object, 'uuid': __import__('uuid'), 'get_workflow_manager': lambda: manager}
            exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
            context = namespace['_workflow_experiment_context'](resumed, stage_name)
            self.assertEqual(resumed['workflow_id'], workflow['workflow_id'])
            self.assertEqual(context['experimentId'], 'exp-original')
            self.assertEqual(context['taskId'], f'{stage_name}-task')
            self.assertNotEqual(context['requestKey'], 'old-request')
            self.assertEqual(manager._get(workflow['workflow_id'])['context']['experiment_launches'][stage_name], context)

    def test_benchmark_retry_preserves_task_identity_across_restart_and_service_redeploy(self):
        dependencies = workflow_module.WorkflowDependencies(*[lambda *args: {} for _ in range(8)])
        manager = workflow_module.WorkflowManager(str(Path(self.temp.name) / 'workflows.db'), dependencies,
                                                   auto_start_worker=False)
        workflow = manager.create('owner', 'dataset')
        workflow['current_stage'] = 'benchmark'
        workflow['stages']['benchmark']['status'] = 'running'
        dependencies.monitor_benchmark = lambda workflow: {'status': 'failed', 'error': 'failed-run', 'benchmark_job_id': 'old-job',
            'experiment_id': 'experiment-1', 'experiment_task_id': 'benchmark-task', 'experiment_attempt_id': 'attempt-old'}
        with self.assertRaisesRegex(RuntimeError, 'failed-run'):
            manager._advance_benchmark(workflow, workflow['stages']['benchmark'])
        workflow['status'] = 'failed'
        manager._save(workflow)
        resumed = manager._resume_workflow(manager._get(workflow['workflow_id']))
        self.assertEqual(resumed['current_stage'], 'deploy')
        restarted = workflow_module.WorkflowManager(manager.db_path, dependencies, auto_start_worker=False)
        persisted = restarted._get(workflow['workflow_id'])
        stage = persisted['stages']['benchmark']
        self.assertEqual(stage['experiment_task_id'], 'benchmark-task')
        self.assertNotIn('result', stage, 'Old job handle must not be mistaken for the new execution')
        context = workflow_module.workflow_inference_experiment_context(persisted,
            {'taskId': 'unrelated-task', 'conversationRunId': 'chat-1'})
        self.assertEqual(context['benchmarkTaskId'], 'benchmark-task')
        self.assertEqual(context['benchmarkExperimentId'], 'experiment-1')
        self.assertNotIn('taskId', context)
        self.assertEqual(native.inference_experiment_context({}, context, kind='benchmark')['taskId'], 'benchmark-task')
        self.assertNotIn('taskId', native.inference_experiment_context({}, context, kind='deploy'))

    def test_unrelated_inherited_benchmark_identity_does_not_leak_into_workflow(self):
        workflow = {'workflow_id': 'workflow-new', 'context': {'experiment_id': 'new-experiment'}, 'stages': {}}
        inherited = {'benchmarkTaskId': 'other-task', 'benchmarkExperimentId': 'other-experiment', 'taskId': 'train-task'}
        context = workflow_module.workflow_inference_experiment_context(workflow, inherited)
        self.assertEqual(context, {'workflowId': 'workflow-new', 'experimentId': 'new-experiment', 'operationTrigger': 'workflow'})
        self.assertEqual(inherited['benchmarkTaskId'], 'other-task')

    def test_workflow_connection_commits_or_rolls_back_then_closes(self):
        import sqlite3
        path = str(Path(self.temp.name) / 'connection.db')
        with sqlite3.connect(path, factory=workflow_module._WorkflowConnection) as connection:
            connection.execute('CREATE TABLE example (value TEXT)')
            connection.execute("INSERT INTO example VALUES ('committed')")
        with self.assertRaises(sqlite3.ProgrammingError):
            connection.execute('SELECT * FROM example')
        with self.assertRaisesRegex(ValueError, 'rollback'):
            with sqlite3.connect(path, factory=workflow_module._WorkflowConnection) as failed:
                failed.execute("INSERT INTO example VALUES ('rolled back')")
                raise ValueError('rollback')
        with self.assertRaises(sqlite3.ProgrammingError):
            failed.execute('SELECT * FROM example')
        with sqlite3.connect(path, factory=workflow_module._WorkflowConnection) as read:
            self.assertEqual(read.execute('SELECT value FROM example').fetchall(), [('committed',)])

    def test_async_workflow_inference_binds_and_restores_experiment_context_on_failure(self):
        import asyncio
        from contextvars import ContextVar
        from typing import Any, Optional
        path = Path(__file__).parents[1] / 'runtime_agent.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        function = next(node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef) and
                        node.name == '_workflow_inference_event')
        variable = ContextVar('test_experiment_context', default={'conversationRunId': 'original-chat'})
        namespace = {'Dict': dict, 'Optional': Optional, 'Any': Any,
            'REQUEST_EXPERIMENT_CONTEXT': variable,
            'workflow_inference_experiment_context': workflow_module.workflow_inference_experiment_context,
            '_workflow_inference_owner_kwargs': lambda context: {'owner_user_id': 'owner'},
            '_current_user_role': lambda: 'user', '_owner_aliases_from_any': lambda value: value or []}
        for name in ('REQUEST_RESOURCE_GROUP_ID', 'REQUEST_TRAINING_POOL_ID', 'REQUEST_USER_ROLE',
                     'REQUEST_INFERENCE_OWNER_USER_ID', 'REQUEST_INFERENCE_OWNER_ALIASES'):
            namespace[name] = ContextVar(name, default=None)
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
        workflow = {'workflow_id': 'wf-1', 'context': {'experiment_id': 'exp-1'},
                    'stages': {'benchmark': {'experiment_task_id': 'task-1', 'experiment_id': 'exp-1'}}}
        seen = []
        async def fail(*args, **kwargs):
            seen.append(variable.get())
            raise RuntimeError('transport failure')
        async def exercise():
            with self.assertRaisesRegex(RuntimeError, 'transport failure'):
                await namespace['_workflow_inference_event'](types.SimpleNamespace(_call_inference=fail), 'start benchmark', workflow, 'benchmark')
            self.assertEqual(variable.get(), {'conversationRunId': 'original-chat'})
        asyncio.run(exercise())
        self.assertEqual(seen[0]['experimentId'], 'exp-1')
        self.assertEqual(seen[0]['workflowId'], 'wf-1')
        self.assertEqual(seen[0]['benchmarkTaskId'], 'task-1')


class NativeBenchmarkTests(unittest.TestCase):
    def setUp(self):
        TrackerTests.setUp(self)
        self.root = Path(self.temp.name)
        self.data = self.root / 'questions.json'
        self.data.write_text('[{"question":"test"}]', encoding='utf-8')
        self.runner = self.root / 'eval_runner.py'
        self.runner.write_text('# runner fixture', encoding='utf-8')
        self.cfg = {'ENV': {'MODEL_PATH': str(self.root), 'MODEL_NAME': 'model'}, 'RUNTIME': {}}
        self.instance = {'instance_id': 'service-1', 'run_id': 'run-1', 'model_path': str(self.root / 'model')}
        self.job = {'job_id': 'job-1', 'job_dir': str(self.root / 'job'), 'output': str(self.root / 'job/result.json'),
                    'meta_file': str(self.root / 'job/meta.json'), 'submission': {'submit_key': 'service|dataset'}}
        Path(self.job['job_dir']).mkdir()

    def tearDown(self):
        for path in self.root.rglob('*'):
            if path.is_file() and not path.is_symlink():
                path.chmod(0o600)
        TrackerTests.tearDown(self)

    def launch(self, context=None, owner_id='user-1'):
        command = [sys.executable, '-u', str(self.runner), '--mode', 'eval', '--dataset', str(self.data),
                   '--output', self.job['output'], '--model', 'model', '--base-url', 'http://localhost/v1']
        def create_process(*args, **kwargs):
            receipt = json.loads(Path(self.job['meta_file']).read_text(encoding='utf-8'))
            self.assertEqual(receipt['status'], 'submitting')
            self.assertEqual(receipt['submit_key'], 'service|dataset')
            self.assertTrue(receipt['experiment']['attemptId'])
            return types.SimpleNamespace(pid=123)
        with patch.object(native, 'get_tracker', return_value=self.tracker), patch.object(native.subprocess, 'Popen', side_effect=create_process) as create:
            native.start_benchmark(command, str(self.root / 'job/run.log'), job=self.job, cfg=self.cfg, instance=self.instance,
                state_dir=self.root, node_id='node-1', owner_id=owner_id, model_commands=[], context=context)
        return self.job['experiment']['attemptId'], create.call_args.args[0]

    def finish(self, attempt, exit_code=0):
        directory = Path(self.tracker.execution(attempt)['runtime']['experimentDirectory'])
        process_module.atomic_json(directory / 'process.json', {'attemptId': attempt, 'exitCode': exit_code, 'commands': []})

    def test_native_launch_uses_frozen_dataset_and_supervised_child(self):
        attempt, command = self.launch()
        frozen = Path(command[command.index('--dataset') + 1])
        self.assertNotEqual(frozen, self.data)
        self.data.write_text('changed', encoding='utf-8')
        self.assertEqual(json.loads(frozen.read_text(encoding='utf-8')), [{'question': 'test'}])
        self.assertIn('run', command)
        self.assertEqual(self.tracker.execution(attempt)['runtime']['inputAssets'][1]['integrity'], 'incomplete')

    def test_benchmark_supervisor_output_reaches_experiment_logs(self):
        import subprocess
        self.runner.write_text('import sys\nprint("evaluated 3 samples password=hidden", flush=True)\n'
                               'sys.stderr.write("benchmark final partial")\nsys.exit(2)\n', encoding='utf-8')
        attempt, command = self.launch()
        result = subprocess.run(command, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 2, result.stderr)
        runtime = self.tracker.execution(attempt)['runtime']
        business_log = Path(runtime['benchmarkLogPath'])
        self.assertTrue(business_log.is_file(), 'Benchmark must create the log that its runtime declares')
        self.assertEqual(business_log.read_bytes(), result.stdout)
        self.assertIn(b'evaluated 3 samples', result.stdout)
        self.assertIn(b'benchmark final partial', result.stdout)
        Path(runtime['executionLogPath']).write_bytes(result.stdout)
        self.tracker.emit(attempt, 'status', {'status': 'failed', 'runtime': {
            'exitCode': 2, 'archiveStatus': 'complete'}})

        worker_logs = native.shared('experiment_logs')
        worker_logs.enable(self.tracker, attempt)
        uploaded = []
        def send(procedure, payload):
            self.assertEqual(procedure, 'appendExperimentLogChunks')
            uploaded.append(payload)
            return {'nextOffset': max((chunk['endOffset'] for chunk in payload['chunks']), default=0)}
        with patch.object(self.tracker, 'send', side_effect=send):
            worker_logs.sync_all(self.tracker)
        benchmark = [payload for payload in uploaded if payload['source']['kind'] == 'evaluation_phase']
        self.assertTrue(benchmark)
        content = ''.join(chunk['content'] for payload in benchmark for chunk in payload['chunks'])
        self.assertIn('evaluated 3 samples password=[REDACTED]', content)
        self.assertIn('benchmark final partial', content)
        self.assertNotIn('hidden', content)
        self.assertEqual(benchmark[-1]['source']['status'], 'complete')

    def test_authenticated_owner_benchmark_uses_studio_actor_and_same_asset(self):
        expected, _ = native.local_asset(self.root, 'user-1', 'node-1', self.data, 'dataset', 'evaluation_dataset')
        attempt, _ = self.launch(owner_id='auth:user-1')
        execution = self.tracker.execution(attempt)
        self.assertEqual(execution['actor_user_id'], 'user-1')
        self.assertEqual(execution['runtime']['inputAssets'][0]['id'], expected['id'])
        with self.tracker.connect() as db:
            request = json.loads(db.execute('SELECT request_json FROM experiment_intent').fetchone()[0])
        self.assertEqual(request['actorUserId'], 'user-1')

    def test_workflow_parent_and_conversation_are_preserved(self):
        self.instance['resource'] = {'reservation_id': 'service-lease'}
        self.instance['experiment'] = {'experimentId': 'deployment-exp'}
        self.launch({'experimentId': 'parent-exp', 'workflowId': 'workflow-1', 'conversationRunId': 'chat-1',
                     'project': 'research', 'resourceReservationId': 'forged-lease',
                     'benchmarkTaskId': 'benchmark-task', 'benchmarkExperimentId': 'parent-exp'})
        with self.tracker.connect() as db:
            request = json.loads(db.execute('SELECT request_json FROM experiment_intent').fetchone()[0])
        self.assertEqual([request[key] for key in ('experimentId', 'workflowId', 'conversationRunId', 'project')],
                         ['parent-exp', 'workflow-1', 'chat-1', 'research'])
        self.assertEqual(request['resourceReservationId'], 'service-lease')
        self.assertEqual(request['taskId'], 'benchmark-task')
        self.assertEqual(self.tracker.execution(self.job['experiment']['attemptId'])['runtime']['deploymentExperimentId'], 'deployment-exp')

    def test_service_model_requires_observed_pinned_path(self):
        model = self.root / 'model'
        model.mkdir()
        (model / 'model.safetensors').write_bytes(b'weights')
        version = native.freeze_service_model(self.cfg, state_dir=self.root, owner_id='user-1', node_id='node-1')
        self.instance['experiment_model_version'] = version
        self.assertEqual(self.cfg['ENV']['MODEL_NAME'], 'model')
        self.assertEqual(native.observed_model(self.instance, self.cfg, 'node-1', 'user-1', [])['integrity'], 'incomplete')
        self.assertEqual(native.observed_model(self.instance, self.cfg, 'node-1', 'user-1',
            [['vllm', 'serve', self.cfg['ENV']['MODEL_SNAPSHOT_PATH']]])['integrity'], 'verified')

    def test_native_collector_archives_and_replays_without_duplicate_results(self):
        attempt, _ = self.launch()
        self.finish(attempt)
        Path(self.job['output']).write_text(json.dumps({'summary': {'total': 1, 'processed': 1, 'accuracy': 1.0},
            'details': [{'response': 'A', 'is_correct': True}]}), encoding='utf-8')
        native.collect_benchmark(self.tracker, self.tracker.execution(attempt))
        payload = self.tracker.event_payload(attempt, 'benchmark-result')
        self.assertEqual(payload['expectedCount'], 1)
        self.assertFalse(payload['protocolComplete'])
        self.assertEqual(self.tracker.execution(attempt)['state'], 'succeeded')
        self.assertEqual(self.tracker.collectable_executions(), [])
        self.assertTrue(self.tracker.event_payload(attempt, 'benchmark-report'))
        count = self.tracker.pending_count()
        native.collect_benchmark(self.tracker, self.tracker.execution(attempt))
        self.assertEqual(self.tracker.pending_count(), count)

    def test_exit_zero_without_result_stays_unconfirmed(self):
        attempt, _ = self.launch()
        self.finish(attempt)
        with self.assertRaises(ValueError):
            native.collect_benchmark(self.tracker, self.tracker.execution(attempt))
        count = self.tracker.pending_count()
        with self.assertRaises(ValueError):
            native.collect_benchmark(self.tracker, self.tracker.execution(attempt))
        self.assertEqual(self.tracker.execution(attempt)['state'], 'unknown')
        self.assertEqual(self.tracker.pending_count(), count)

    def test_completed_benchmark_backfills_details_from_frozen_report_once(self):
        attempt, _ = self.launch()
        self.finish(attempt)
        Path(self.job['output']).write_text(json.dumps({'summary': {'total': 2, 'processed': 2, 'accuracy': 0.5},
            'details': [{'index': 1, 'question': 'q', 'ground_truth': 'A', 'predicted': 'A', 'is_correct': True},
                        {'index': 2, 'question': 'r', 'response': 'ERROR: timeout', 'error': 'timeout'}]}), encoding='utf-8')
        native.collect_benchmark(self.tracker, self.tracker.execution(attempt))
        original_result = self.tracker.event_payload(attempt, 'benchmark-result')
        runtime = self.tracker.execution(attempt)['runtime']
        self.assertEqual(runtime['benchmarkReport']['counts']['failed'], 1)
        for key in ('benchmarkDetailsVersion', 'benchmarkReport'):
            runtime.pop(key)
        with self.tracker.connect() as db:
            db.execute('UPDATE experiment_execution SET runtime_json=? WHERE attempt_id=?', (json.dumps(runtime), attempt))
        Path(self.job['output']).write_text('{}', encoding='utf-8')
        native.reconcile_worker(self.tracker)
        execution = self.tracker.execution(attempt)
        self.assertEqual(execution['state'], 'succeeded')
        self.assertEqual(execution['runtime']['benchmarkReport']['total'], 2)
        self.assertEqual(execution['runtime']['archiveStatus'], 'complete')
        self.assertEqual(self.tracker.event_payload(attempt, 'benchmark-result'), original_result)
        count = self.tracker.pending_count()
        native.reconcile_worker(self.tracker)
        self.assertEqual(self.tracker.pending_count(), count)

    def test_missing_journal_never_infers_success_from_result_file(self):
        attempt, _ = self.launch()
        Path(self.job['output']).write_text('{}', encoding='utf-8')
        native.collect_benchmark(self.tracker, self.tracker.execution(attempt))
        self.assertEqual(self.tracker.execution(attempt)['state'], 'unknown')

    def test_failed_requests_are_distinguished_from_wrong_answers(self):
        parsed = native.structured_result({'summary': {'total': 3, 'processed': 3, 'accuracy': 0.33},
            'details': [{'response': 'A', 'is_correct': True}, {'response': 'B', 'is_correct': False},
                        {'response': 'ERROR: connection refused', 'is_correct': False}]}, 'dataset-1')
        self.assertEqual((parsed['succeededCount'], parsed['failedCount']), (2, 1))
        self.assertEqual(parsed['metrics'][0]['sampleCount'], 3)
        unknown = native.structured_result({'summary': {'total': 100, 'processed': 100, 'accuracy': 1}, 'details': []}, 'dataset-1')
        self.assertIsNone(unknown['succeededCount'])

    def test_medbench_records_generation_counts_without_inventing_accuracy(self):
        inputs, outputs = self.root / 'inputs', self.root / 'outputs'
        inputs.mkdir()
        outputs.mkdir()
        (inputs / 'cases.jsonl').write_text('{"question":"a"}\n{"question":"b"}\n', encoding='utf-8')
        (outputs / 'cases.jsonl').write_text('{"answer":"a"}\n{"answer":"ERROR: timed out"}\n', encoding='utf-8')
        result = native.generation_result(inputs, outputs, 'dataset-1')
        self.assertEqual((result['expectedCount'], result['succeededCount'], result['failedCount']), (2, 1, 1))
        self.assertEqual([item['name'] for item in result['metrics']], ['generated_responses'])

    def test_code_evaluation_metrics_preserve_their_denominators_and_units(self):
        result = native.structured_result({'summary': {'total': 3, 'processed': 3,
            'metrics': {'executed': 2, 'pass@1': 0.5, 'case_accuracy': 0.75, 'evaluator_error': 1}},
            'details': [{}, {}, {'evaluator_error': True}]}, 'dataset-1')
        metrics = {item['name']: item for item in result['metrics']}
        self.assertEqual(metrics['executed']['unit'], 'count')
        self.assertEqual(metrics['pass@1']['sampleCount'], 2)
        self.assertIsNone(metrics['case_accuracy']['sampleCount'])
        self.assertEqual(result['failedCount'], 1)


class EvaluationAdapterTests(unittest.TestCase):
    setUp = AssetTests.setUp
    tearDown = AssetTests.tearDown

    def test_monitor_report_hint_cannot_attribute_shared_report_to_attempt(self):
        attempt_root = self.root / 'execution'
        attempt_root.mkdir()
        report = self.root / 'shared-daily-report.txt'
        report.write_text('Complete scores from another execution', encoding='utf-8')
        process_module.atomic_json(attempt_root / 'execution.json', {'attemptId': 'legacy-eval', 'reportPaths': []})
        process_module.atomic_json(attempt_root / 'process.json', {'attemptId': 'legacy-eval', 'exitCode': 0})
        for timestamp in (1, None):
            os.utime(report, (timestamp, timestamp) if timestamp is not None else None)
            result = process_module.probe(attempt_root, {'report_path': str(report)})
            self.assertEqual(result['state'], 'succeeded', 'The process exit remains authoritative')
            self.assertEqual(result['reports'], [], 'A recent shared report still lacks execution attribution')
            self.assertEqual(result['unattributedReports'], [str(report)])
            self.assertNotIn('evaluationManifest', result)

    def test_declared_report_outside_journal_directory_is_still_collected(self):
        attempt_root = self.root / 'execution'
        attempt_root.mkdir()
        report = self.root / 'isolated-results' / 'attempt-one' / 'summary.txt'
        report.parent.mkdir(parents=True)
        report.write_text('Scores for attempt-one', encoding='utf-8')
        process_module.atomic_json(attempt_root / 'execution.json', {'attemptId': 'attempt-one', 'reportPaths': [str(report)]})
        process_module.atomic_json(attempt_root / 'process.json', {'attemptId': 'attempt-one', 'exitCode': 0})
        result = process_module.probe(attempt_root, {'report_path': str(report)})
        self.assertEqual(result['reports'], [str(report)])
        self.assertNotIn('unattributedReports', result)

    def legacy_checkpoint_fixture(self):
        base = self.source / 'base'
        base.mkdir()
        (base / 'model.safetensors').write_bytes(b'base weights')
        checkpoint = self.source / 'checkpoint-20'
        checkpoint.mkdir()
        (checkpoint / 'adapter_model.safetensors').write_bytes(b'adapter weights')
        (checkpoint / 'adapter_config.json').write_text(json.dumps({'base_model_name_or_path': str(base), 'r': 16}), encoding='utf-8')
        records = []
        for source, kind, role in [(checkpoint, 'checkpoint', 'evaluated_model'), (base, 'model', 'base_model')]:
            captured = assets.snapshot(source, self.root / 'versions' / kind)
            records.append(capture.asset_registration(captured, node_id='node', container='evaluator', owner_id='owner',
                source=str(source), kind=kind, role=role))
        return {'path': str(self.root / 'execution' / 'checkpoint-attempt'), 'seed': records[0],
                'baseModel': records[1], 'baseModelPath': records[1]['location']['path']}

    def test_legacy_checkpoint_copy_binds_frozen_base_without_changing_source(self):
        descriptor = self.legacy_checkpoint_fixture()
        source = descriptor['seed']
        original = (Path(source['location']['path']) / 'adapter_config.json').read_bytes()
        shutil.rmtree(self.source)
        receipt = process_module.prepare_checkpoint_input(descriptor, Path(assets.__file__).read_text(encoding='utf-8'))
        copied = Path(descriptor['path'])
        config = json.loads((copied / 'adapter_config.json').read_text(encoding='utf-8'))
        self.assertEqual(config, {'base_model_name_or_path': descriptor['baseModelPath'], 'r': 16})
        self.assertEqual((Path(config['base_model_name_or_path']) / 'model.safetensors').read_bytes(), b'base weights')
        self.assertEqual((copied / 'adapter_model.safetensors').read_bytes(), b'adapter weights')
        self.assertEqual((Path(source['location']['path']) / 'adapter_config.json').read_bytes(), original)
        self.assertEqual(assets.manifest(source['location']['path'])['fingerprint'], source['fingerprint'])
        self.assertEqual(receipt['executionFingerprint'], assets.manifest(copied)['fingerprint'])
        self.assertNotEqual(receipt['originalConfigSha256'], receipt['executionConfigSha256'])
        self.assertEqual(receipt['baseModelVersionId'], descriptor['baseModel']['id'])
        with self.assertRaises(FileExistsError):
            process_module.prepare_checkpoint_input(descriptor, Path(assets.__file__).read_text(encoding='utf-8'))

    def test_legacy_checkpoint_copy_rejects_changed_base_or_adapter_bytes(self):
        descriptor = self.legacy_checkpoint_fixture()
        code = Path(assets.__file__).read_text(encoding='utf-8')
        for role, filename in [('baseModel', 'model.safetensors'), ('seed', 'adapter_model.safetensors')]:
            target = Path(descriptor[role]['location']['path']) / filename
            target.chmod(0o600)
            before = target.read_bytes()
            target.write_bytes(b'changed')
            try:
                with self.subTest(role=role), self.assertRaisesRegex(ValueError, 'unavailable or changed'):
                    process_module.prepare_checkpoint_input({**descriptor, 'path': str(self.root / role)}, code)
            finally:
                target.write_bytes(before)

    def checkpoint_fixture(self):
        dataset = self.source / 'mmlu'
        dataset.mkdir()
        for name in ('mmlu.py', 'mmlu.zip', 'mapping.json'):
            (dataset / name).write_text('fixture', encoding='utf-8')
        engine = self.root / 'engine'
        engine.write_bytes(b'\x7fELFfixture')
        environment = {'MEDFLOW_EVAL_TASK_DIR': str(self.source), 'MEDFLOW_EVAL_IMAGE_ID': 'sha256:fixture',
            'MEDFLOW_EVAL_BASE_MODEL': '/versions/base', 'MEDFLOW_EVAL_MODEL_VERSION_ROOT': str(self.root / 'versions')}
        arguments = types.SimpleNamespace(eval_datasets='mmlu', run_id='checkpoint-run', CKPT_PATH='/versions/checkpoint', TEM='qwen3')
        return engine, environment, arguments

    def test_checkpoint_eval_reads_frozen_export_and_probe_keeps_that_version(self):
        engine, environment, arguments = self.checkpoint_fixture()
        result_root = self.root / 'results' / arguments.run_id
        merged = result_root / 'merged_checkpoint'
        calls = []
        def execute(command, **kwargs):
            calls.append(command)
            if command[1] == 'export':
                merged.mkdir()
                (merged / 'model.safetensors').write_bytes(b'exported weights')
            else:
                model = command[command.index('--model_name_or_path') + 1]
                if model == '/versions/base':
                    (merged / 'model.safetensors').write_bytes(b'changed original export')
                else:
                    self.assertNotEqual(model, str(merged))
                    self.assertEqual((Path(model) / 'model.safetensors').read_bytes(), b'exported weights')
                output = Path(command[command.index('--save_dir') + 1])
                output.mkdir()
                (output / 'results.log').write_text('Subject scores:\nmath: 50.0\nSummary scores:\nAverage: 50.0\n', encoding='utf-8')
                (output / 'results.json').write_text(json.dumps({'math': {'0': 'A'}}), encoding='utf-8')
            return types.SimpleNamespace(returncode=0)
        with patch.object(evaluation_module, 'CLI', str(engine)), patch.object(evaluation_module.subprocess, 'run', side_effect=execute), \
             patch.dict(os.environ, environment):
            evaluation_module.run('ckpt', arguments, result_base=str(self.root / 'results'), log_base=str(self.root / 'logs'))
        manifest_path = result_root / 'manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        self.assertEqual([phase['operation'] for phase in manifest['executions']], ['export', 'eval', 'eval'])
        self.assertTrue(all(phase['exitCode'] == 0 and phase['finishedAt'] >= phase['startedAt'] for phase in manifest['executions']))
        artifact = manifest['modelArtifacts'][0]
        self.assertEqual(manifest['models'][1]['path'], artifact['path'])
        self.assertNotEqual(assets.manifest(merged)['fingerprint'], artifact['fingerprint'])
        process_module.atomic_json(self.root / 'execution.json', {'attemptId': 'eval-one', 'outputPaths': [str(merged)],
            'modelVersionRoot': environment['MEDFLOW_EVAL_MODEL_VERSION_ROOT'], 'reportPaths': [str(manifest_path)]})
        process_module.atomic_json(self.root / 'process.json', {'attemptId': 'eval-one', 'exitCode': 0, 'outputPaths': [str(merged)]})
        result = process_module.probe(self.root)
        self.assertEqual(result['outputs'], [{**artifact, 'kind': 'model', 'role': 'merged_model'}])
        self.assertIn(manifest['executions'][0]['logPath'], result['reports'])
        for changed in ({'path': str(merged)}, {'sourcePath': '/undeclared/output'}):
            invalid = {**manifest, 'modelArtifacts': [{**artifact, **changed}]}
            process_module.atomic_json(manifest_path, invalid)
            with self.assertRaisesRegex(ValueError, 'outside the declared'):
                process_module.probe(self.root)

    def test_export_failure_records_exit_and_never_starts_evaluation(self):
        engine, environment, arguments = self.checkpoint_fixture()
        with patch.object(evaluation_module, 'CLI', str(engine)), \
             patch.object(evaluation_module.subprocess, 'run', return_value=types.SimpleNamespace(returncode=42)) as execute, \
             patch.dict(os.environ, environment), self.assertRaisesRegex(RuntimeError, 'exited 42'):
            evaluation_module.run('ckpt', arguments, result_base=str(self.root / 'results'), log_base=str(self.root / 'logs'))
        execute.assert_called_once()
        manifest = json.loads((self.root / 'results' / arguments.run_id / 'manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['status'], 'failed')
        self.assertEqual(manifest['executions'][0]['exitCode'], 42)
        self.assertEqual(manifest['modelArtifacts'], [])

    def test_counts_predictions_by_subject_and_rejects_unknown_format(self):
        path = self.root / 'results.json'
        path.write_text(json.dumps({'math': {'0': 'A', '1': 'B'}, 'history': {'0': 'C'}}), encoding='utf-8')
        self.assertEqual(evaluation_module.sample_counts(path), {'math': 2, 'history': 1})
        path.write_text(json.dumps({'completedEvaluations': 2}), encoding='utf-8')
        self.assertIsNone(evaluation_module.sample_counts(path))

    def test_adapter_executes_pinned_dataset_and_records_real_sample_counts(self):
        dataset = self.source / 'mmlu'
        dataset.mkdir()
        for filename in ('mmlu.py', 'mmlu.zip', 'mapping.json'):
            (dataset / filename).write_text('fixture', encoding='utf-8')
        engine = self.root / 'engine'
        engine.write_bytes(b'\x7fELFtest-engine')
        calls = []
        def execute(command, **kwargs):
            calls.append(command)
            output = Path(command[command.index('--save_dir') + 1])
            output.mkdir()
            (output / 'results.log').write_text('Subject scores:\nmath: 50.0\nSummary scores:\nAverage: 50.0\n', encoding='utf-8')
            (output / 'results.json').write_text(json.dumps({'math': {'0': 'A', '1': 'B'}}), encoding='utf-8')
            return types.SimpleNamespace(returncode=0)
        with patch.object(evaluation_module, 'CLI', str(engine)), patch.object(evaluation_module.subprocess, 'run', side_effect=execute), patch.dict(os.environ, {
            'MEDFLOW_EVAL_TASK_DIR': str(self.source), 'MEDFLOW_EVAL_IMAGE_ID': 'sha256:fixture'}):
            evaluation_module.run('single_model', types.SimpleNamespace(eval_datasets='mmlu', run_id='run-1', model_fir='/versions/model'),
                result_base=str(self.root / 'results'), log_base=str(self.root / 'logs'))
        manifest = json.loads((self.root / 'results/run-1/manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['status'], 'finished')
        self.assertEqual(manifest['results']['model_fir']['mmlu']['sampleCount'], 2)
        self.assertEqual(calls[0][calls[0].index('--task_dir') + 1], str(self.source))
        self.assertEqual(calls[0][calls[0].index('--seed') + 1], '42')
        self.assertTrue(manifest['experimentProtocolComplete'])

    def test_attempt_setup_isolates_batch_outputs_without_changing_asset_identity(self):
        def install(container, module, request, *args, **kwargs):
            return {'runner': f"/experiments/{request['attemptId']}/runner.py"}
        outputs = []
        with patch.object(collector, 'container_json', side_effect=install), patch.object(collector, 'container_python', return_value='python3'), patch.object(
            collector.subprocess, 'run', return_value=types.SimpleNamespace(returncode=0, stdout='"sha256:image"')):
            for attempt in ('attempt-one', 'attempt-two'):
                kwargs = {'docker_container': 'trainer', 'env_vars': {'DATASET_DIR': '/versions/same-data'}}
                runtime = collector.prepare_container_execution(kwargs, handle={'attemptId': attempt}, actor_user_id='owner',
                    script_name='batch_train_lora', assets=[])
                outputs.append(runtime['outputPaths'][0])
                self.assertIn(attempt, kwargs['env_vars']['DATASET_DIR'])
        self.assertNotEqual(outputs[0], outputs[1])

    def test_training_setup_tracks_merged_siblings_and_isolates_export_directories(self):
        with patch.object(collector, 'container_json', return_value={'runner': '/runner.py'}), \
             patch.object(collector, 'container_python', return_value='python3'), \
             patch.object(collector.subprocess, 'run', return_value=types.SimpleNamespace(returncode=0, stdout='"sha256:image"')):
            for script, root in [('batch_train_lora', 'batch_train'), ('batch_train_full', 'batch_train'),
                                 ('batch_train_pretrain_lora', 'pretrain'), ('batch_train_pretrain_full', 'pretrain')]:
                kwargs = {'docker_container': 'trainer', 'env_vars': {'DATASET_DIR': '/versions/data'}}
                runtime = collector.prepare_container_execution(kwargs, handle={'attemptId': 'attempt-one'}, actor_user_id='owner',
                    script_name=script, assets=[])
                self.assertTrue(runtime['outputPaths'][0].startswith(f'/home/workspace/models/{root}/'))
                self.assertEqual(runtime['outputPaths'][1], runtime['outputPaths'][0] + '_merged')
            for script in ('train_multinode_sft_pipeline', 'train_multinode_sft_full_pipeline'):
                kwargs = {'docker_container': 'trainer', 'script_args': {'output-dir': '/runs/sft'}}
                runtime = collector.prepare_container_execution(kwargs, handle={'attemptId': 'attempt-one'}, actor_user_id='owner',
                    script_name=script, assets=[])
                self.assertEqual(runtime['outputPaths'], ['/runs/sft/attempt-one', '/runs/sft/attempt-one_merged'])
                self.assertEqual(kwargs['script_args']['log-dir'], '/home/workspace/log/batch_train')
                self.assertEqual(kwargs['script_args']['log-file'], 'attempt-one-train-log.log')
            for supplied in ({}, {'export-dir': '/exports/chosen'}):
                kwargs = {'docker_container': 'trainer', 'script_args': dict(supplied)}
                runtime = collector.prepare_container_execution(kwargs, handle={'attemptId': 'attempt-one'}, actor_user_id='owner',
                    script_name='train_multinode_dpo_pipeline', assets=[])
                export = kwargs['script_args']['export-dir']
                self.assertIn(export, runtime['outputPaths'])
                self.assertTrue(export.endswith('/attempt-one'))
                self.assertNotEqual(export, supplied.get('export-dir'))
                self.assertEqual(kwargs['script_args']['run-id'], 'attempt-one')
                self.assertEqual(kwargs['script_args']['log-dir'], '/home/workspace/log/dpo_train')
                self.assertEqual(kwargs['script_args']['log-file'], 'attempt-one-dpo-train.log')

    def test_only_finite_scores_are_accepted(self):
        path = self.root / 'results.log'
        path.write_text('Subject scores:\nmath: 50.0\nSummary scores:\nAverage: nan\n', encoding='utf-8')
        with self.assertRaises(ValueError):
            evaluation_module.parse_scores(path)

    def test_preprocessing_output_isolation_applies_to_shared_and_private_directories(self):
        with patch.object(collector, 'container_json', return_value={'runner': '/runner.py'}) as install, \
             patch.object(collector, 'container_python', return_value='python3'), \
             patch.object(collector.subprocess, 'run', return_value=types.SimpleNamespace(returncode=0, stdout='"sha256:image"')):
            for script in ('data_preprocessing', 'score_based_filtering'):
                for base in ('/home/workspace/dataset_batch_train/20260929',
                             '/home/workspace/medflow/users/owner/datasets/sft/product',
                             '/home/workspace/medflow/users/owner/datasets/sft/medical_sft_261014-143025-123'):
                    products = []
                    for attempt in ('attempt-one', 'attempt-two'):
                        seed = {'role': 'preprocessing_resume', 'location': {'path': '/versions/seed'}}
                        kwargs = {'docker_container': 'trainer', 'script_args': {'output_folder': base, 'resume': True},
                                  'protocol_hint_extra': {'threshold': 90, 'outputFolder': base}}
                        collector.prepare_container_execution(kwargs, handle={'attemptId': attempt}, actor_user_id='owner',
                            script_name=script, assets=[seed])
                        actual = kwargs['script_args']['output_folder']
                        products.append(actual)
                        self.assertEqual(str(Path(actual).parent), str(Path(base).parent))
                        self.assertLessEqual(len(Path(actual).name), 57)
                        self.assertRegex(actual, r'-[a-f0-9]{10}$')
                        self.assertNotIn(attempt, Path(actual).name)
                        self.assertEqual(kwargs['protocol_hint_extra']['outputFolder'], actual)
                        self.assertEqual(kwargs['protocol_hint_extra']['outputDatasetName'], Path(actual).name)
                        self.assertEqual(kwargs['protocol_hint_extra']['threshold'], 90)
                        request = install.call_args.args[2]
                        self.assertEqual(bool(request.get('scoringCompatSource')), script == 'score_based_filtering')
                        self.assertEqual(request['datasetOutput'], {'path': actual, 'seed': seed})
                        self.assertEqual(request['metadata']['declaredOutputs'][0]['path'], actual)
                        self.assertEqual(request['metadata']['datasetOutputPath'], actual)
                        self.assertEqual(request['metadata']['businessStdout']['kind'], 'data_processing')
                    self.assertNotEqual(*products)

    def test_preprocessing_resume_copies_verified_bytes_without_modifying_prior_version(self):
        (self.source / 'data.json').write_text('[{"text":"prior sample"}]', encoding='utf-8')
        captured = assets.snapshot(self.source, self.root / 'versions')
        seed = {'location': {'path': captured['path']}, 'fingerprint': captured['fingerprint']}
        destination = self.root / 'next-product'
        source_code = Path(assets.__file__).read_text(encoding='utf-8')
        process_module.prepare_dataset_output({'path': str(destination), 'seed': seed}, source_code)
        self.assertEqual(assets.manifest(destination)['fingerprint'], captured['fingerprint'])
        (destination / 'data.json').write_text('[{"text":"new sample"}]', encoding='utf-8')
        self.assertEqual(assets.manifest(captured['path'])['fingerprint'], captured['fingerprint'])
        with self.assertRaises(FileExistsError):
            process_module.prepare_dataset_output({'path': str(destination), 'seed': seed}, source_code)

    def test_preprocessing_resume_rejects_changed_source_and_copy(self):
        (self.source / 'data.json').write_text('[1]', encoding='utf-8')
        seed = {'location': {'path': str(self.source)}, 'fingerprint': assets.manifest(self.source)['fingerprint']}
        source_code = Path(assets.__file__).read_text(encoding='utf-8')
        (self.source / 'data.json').write_text('[2]', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'unavailable or changed'):
            process_module.prepare_dataset_output({'path': str(self.root / 'changed-source'), 'seed': seed}, source_code)
        seed['fingerprint'] = assets.manifest(self.source)['fingerprint']
        def corrupt_copy(source, target):
            Path(target).write_text('corrupt', encoding='utf-8')
        with patch.object(process_module.shutil, 'copyfile', side_effect=corrupt_copy), \
             self.assertRaisesRegex(ValueError, 'content verification'):
            process_module.prepare_dataset_output({'path': str(self.root / 'changed-copy'), 'seed': seed}, source_code)

    def test_preprocessing_fresh_output_cannot_reuse_prior_directory(self):
        destination = self.root / 'fresh-product'
        process_module.prepare_dataset_output({'path': str(destination)}, None)
        self.assertEqual(list(destination.iterdir()), [])
        (destination / 'old.json').write_text('["old sample"]', encoding='utf-8')
        with self.assertRaises(FileExistsError):
            process_module.prepare_dataset_output({'path': str(destination)}, None)
        self.assertEqual((destination / 'old.json').read_text(encoding='utf-8'), '["old sample"]')

    def test_discovers_explicit_outputs_without_confusing_model_inputs(self):
        paths = process_module.output_paths([['trainer', '--model_name_or_path', '/base', '--output_dir', '/output'],
                                             ['trainer', 'trainer.default_local_dir=/grpo']])
        self.assertEqual(paths, ['/grpo', '/output'])

    def test_output_inspection_distinguishes_adapter_checkpoint_and_merged_model(self):
        for name in ('checkpoint-10', 'checkpoint-10_merged'):
            directory = self.root / name
            directory.mkdir()
            (directory / 'model.safetensors').write_bytes(b'weights')
        outputs, _ = process_module.inspect_outputs([str(self.root)])
        self.assertEqual({Path(item['path']).name: item['kind'] for item in outputs},
                         {'checkpoint-10': 'checkpoint', 'checkpoint-10_merged': 'model'})

    def test_partial_sharded_export_is_not_registered_as_a_model(self):
        merged = self.root / 'training_merged'
        merged.mkdir()
        for number in range(1, 6):
            (merged / f'model-{number:05d}-of-00012.safetensors').write_bytes(b'weights')
        self.assertEqual(process_module.inspect_outputs([str(merged)])[0], [])
        for number in range(6, 13):
            (merged / f'model-{number:05d}-of-00012.safetensors').write_bytes(b'weights')
        index = {'weight_map': {f'layer-{number}': f'model-{number:05d}-of-00012.safetensors'
                                for number in range(1, 13)}}
        (merged / 'model.safetensors.index.json').write_text(json.dumps(index), encoding='utf-8')
        self.assertEqual(len(process_module.inspect_outputs([str(merged)])[0]), 1)

    def test_preprocessing_audit_files_do_not_count_as_a_dataset(self):
        product = self.root / 'product'
        product.mkdir()
        (product / 'score_summary.json').write_text('{}', encoding='utf-8')
        descriptor = {'path': str(product), 'kind': 'dataset', 'role': 'processed_dataset'}
        (self.root / 'execution.json').write_text(json.dumps({'attemptId': 'attempt-one', 'declaredOutputs': [descriptor]}), encoding='utf-8')
        (self.root / 'process.json').write_text(json.dumps({'attemptId': 'attempt-one', 'exitCode': 0}), encoding='utf-8')
        self.assertEqual(process_module.probe(self.root)['outputs'], [])
        (product / 'data.txt').write_text('real sample', encoding='utf-8')
        self.assertEqual(process_module.probe(self.root)['outputs'], [descriptor])

    def test_container_journal_redacts_credentials_before_writing_to_disk(self):
        command = ['preprocess', '--api_key', 'private-value', '--output_folder', '/output']
        path = self.root / 'process.json'
        process_module.atomic_json(path, {'commands': [command]})
        self.assertNotIn('private-value', path.read_text(encoding='utf-8'))
        self.assertEqual(command[2], 'private-value', 'Redaction must not mutate the actual launch arguments')
        self.assertEqual(process_module.read_json(path)['commands'][0][-1], '/output')


if __name__ == "__main__":
    unittest.main()
