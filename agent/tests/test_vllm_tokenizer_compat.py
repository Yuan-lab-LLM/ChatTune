"""No optional dependencies or GPU; reproduce the old tokenizer constructor."""
import ast
from contextlib import redirect_stderr
import hashlib
import importlib.util
import io
import os
from pathlib import Path
import shlex
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / 'utils/vllm_tokenizer_compat.py'
spec = importlib.util.spec_from_file_location('tokenizer_compat_test', MODULE_PATH)
compat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compat)


def old_tokenizer_base():
    class OldTokenizer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            # Transformers 4.51.3 takes this unconditional .keys() path.
            self.named_tokens = list(kwargs.get('extra_special_tokens', {}).keys())
            self.additional = kwargs.get('additional_special_tokens', [])
            self.decoder = kwargs.get('added_tokens_decoder')
    return OldTokenizer


class TokenizerCompatibilityTests(unittest.TestCase):
    def test_reproduces_original_failure_then_preserves_tokens_and_ids(self):
        base = old_tokenizer_base()
        decoder = {151644: {'content': '<|im_start|>', 'special': True}}
        kwargs = {'extra_special_tokens': ['<|im_start|>', '<|im_end|>'],
                  'additional_special_tokens': ['<|im_start|>', '<|vision_pad|>'],
                  'added_tokens_decoder': decoder, 'chat_template': 'original-template'}
        with self.assertRaisesRegex(AttributeError, 'keys'):
            base(**kwargs)
        self.assertTrue(compat.patch_tokenizer_base(base, '4.51.3'))
        output = io.StringIO()
        with redirect_stderr(output):
            tokenizer = base(**kwargs)
        self.assertEqual(tokenizer.additional, ['<|im_start|>', '<|vision_pad|>', '<|im_end|>'])
        self.assertIs(tokenizer.decoder, decoder)
        self.assertEqual(tokenizer.kwargs['chat_template'], 'original-template')
        self.assertEqual(kwargs['extra_special_tokens'], ['<|im_start|>', '<|im_end|>'])
        self.assertEqual(kwargs['additional_special_tokens'], ['<|im_start|>', '<|vision_pad|>'])
        self.assertIn('preserved special tokens', output.getvalue())

    def test_added_token_flags_are_preserved(self):
        token = types.SimpleNamespace(content='<|im_end|>', lstrip=True, normalized=False)
        normalized = compat.normalize_special_tokens({'extra_special_tokens': [token]})
        self.assertIs(normalized['additional_special_tokens'][0], token)

    def test_dict_and_missing_extra_tokens_are_unchanged(self):
        for kwargs in ({}, {'extra_special_tokens': {'image_token': '<image>'}}, {'extra_special_tokens': None}):
            self.assertIs(compat.normalize_special_tokens(kwargs), kwargs)

    def test_empty_list_keeps_existing_additional_tokens(self):
        result = compat.normalize_special_tokens({'extra_special_tokens': [], 'additional_special_tokens': ['<a>']})
        self.assertEqual(result, {'extra_special_tokens': {}, 'additional_special_tokens': ['<a>']})

    def test_invalid_tokens_are_not_silently_dropped(self):
        for kwargs in ({'extra_special_tokens': [123]},
                       {'extra_special_tokens': ['<a>'], 'additional_special_tokens': {'invalid': '<b>'}}):
            with self.assertRaises(TypeError):
                compat.normalize_special_tokens(kwargs)

    def test_version_gate_and_idempotence(self):
        for version in ('5.6.0', '4.50.3', 'unknown'):
            base = old_tokenizer_base()
            original = base.__init__
            self.assertFalse(compat.patch_tokenizer_base(base, version))
            self.assertIs(base.__init__, original)
        base = old_tokenizer_base()
        self.assertTrue(compat.patch_tokenizer_base(base, '4.51.3'))
        installed = base.__init__
        self.assertFalse(compat.patch_tokenizer_base(base, '4.51.3'))
        self.assertIs(base.__init__, installed)

    def test_unrelated_python_processes_do_not_inspect_packages(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(compat, 'version', side_effect=AssertionError):
            self.assertFalse(compat.install_patch())
        with patch.dict(os.environ, {'MEDFLOW_VLLM_TOKENIZER_COMPAT': '1'}), \
                patch.object(compat, 'version', return_value='5.6.0'):
            self.assertFalse(compat.install_patch())


class CollectorCompatibilityTests(unittest.TestCase):
    def setUp(self):
        path = ROOT / 'utils/experiment_collector.py'
        function = next(n for n in ast.parse(path.read_text(encoding='utf-8')).body
                        if isinstance(n, ast.FunctionDef) and n.name == 'prepare_container_execution')
        self.requests = []
        def install(container, module, request, *args, **kwargs):
            self.requests.append(request)
            return {'runner': '/runner.py', 'vllmTokenizerCompat': {
                'binDirectory': '/attempt/vllm_compat/bin', 'containerPath': '/usr/bin:/bin',
                'kind': 'vllm_tokenizer_extra_special_tokens_compat'} if request.get('tokenizerCompatSource') else None}
        self.namespace = dict(Path=Path, PurePosixPath=__import__('pathlib').PurePosixPath,
            __file__=str(path), json=__import__('json'), container_json=install,
            container_python=lambda _: 'python3',
            subprocess=types.SimpleNamespace(run=lambda *a, **k: types.SimpleNamespace(returncode=0, stdout='"image"')))
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), self.namespace)

    def prepare(self, script):
        kwargs = {'docker_container': 'trainer', 'script_args': {}, 'env_vars': {'PATH': '/custom:/usr/bin'}}
        result = self.namespace['prepare_container_execution'](kwargs, handle={'attemptId': 'attempt-1'},
            actor_user_id='user-1', script_name=script, assets=[])
        return kwargs, result

    def test_medical_serving_receives_scoped_wrapper_and_audit_record(self):
        kwargs, runtime = self.prepare('single_model_evaluation_vpn')
        self.assertIn('normalize_special_tokens', self.requests[0]['tokenizerCompatSource'])
        self.assertEqual(kwargs['env_vars']['PATH'], '/attempt/vllm_compat/bin:/custom:/usr/bin')
        self.assertNotIn('PYTHONPATH', kwargs['env_vars'])
        self.assertEqual(runtime['runtimeCompatibility'][0]['kind'], 'vllm_tokenizer_extra_special_tokens_compat')

    def test_training_launch_is_unchanged(self):
        kwargs, runtime = self.prepare('training')
        self.assertIsNone(self.requests[0]['tokenizerCompatSource'])
        self.assertEqual(kwargs['env_vars'], {'PATH': '/custom:/usr/bin'})
        self.assertEqual(runtime['runtimeCompatibility'], [])


class WrapperInstallationTests(unittest.TestCase):
    def test_wrapper_preserves_arguments_and_uses_real_executable(self):
        path = ROOT / 'utils/experiment_process.py'
        function = next(n for n in ast.parse(path.read_text(encoding='utf-8')).body
                        if isinstance(n, ast.FunctionDef) and n.name == 'prepare_vllm_tokenizer_compat')
        namespace = dict(shutil=types.SimpleNamespace(which=lambda _: '/usr/local/bin/vllm'),
                         shlex=shlex, hashlib=hashlib, os=os)
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
        # Windows mkdtemp needs normal ACLs; the test runner uses an authorized context.
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent, prefix='.tokenizer-compat-test-') as temporary:
            root = Path(temporary).resolve()
            installed = namespace['prepare_vllm_tokenizer_compat'](root, MODULE_PATH.read_text(encoding='utf-8'))
            wrapper = (root / 'vllm_compat/bin/vllm').read_text(encoding='utf-8')
            self.assertIn('exec /usr/local/bin/vllm "$@"', wrapper)
            self.assertIn('export MEDFLOW_VLLM_TOKENIZER_COMPAT=1', wrapper)
            self.assertIn('${PYTHONPATH:+:$PYTHONPATH}', wrapper)
            self.assertEqual(installed['realExecutable'], '/usr/local/bin/vllm')
            self.assertEqual(namespace['prepare_vllm_tokenizer_compat'](root, MODULE_PATH.read_text(encoding='utf-8')), installed)
            with self.assertRaisesRegex(ValueError, 'immutable'):
                namespace['prepare_vllm_tokenizer_compat'](root, '# changed code')


if __name__ == '__main__':
    unittest.main()
