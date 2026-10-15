"""Benchmark logs and frozen report previews, without installing runner dependencies."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reporting = load('benchmark_reporting_test', ROOT / 'inference_ops/experiment_runtime/experiment_benchmark.py')
logging = load('benchmark_logging', ROOT / 'benchmark/benchmark_logging.py')


class Output:
    def __init__(self, **values):
        self.__dict__.update(values)

    def model_dump(self):
        return self.__dict__


def runner(name):
    stubs = {'openai': types.SimpleNamespace(OpenAI=object),
             'pydantic': types.SimpleNamespace(BaseModel=Output),
             'pandas': types.SimpleNamespace(DataFrame=object), 'benchmark_logging': logging}
    with patch.dict(sys.modules, stubs):
        return load('test_' + name, ROOT / 'benchmark' / (name + '.py'))


class BenchmarkReportingTests(unittest.TestCase):
    def test_preview_separates_wrong_invalid_failed_and_unscored_answers(self):
        rows = [dict(index=1, is_correct=True), dict(index=2, is_correct=False),
                dict(index=3, is_correct=False, invalid=True), dict(index=4, response='ERROR: timeout'),
                dict(index=5, response='generated answer')]
        preview = reporting.preview({'details': rows})
        self.assertEqual(preview['counts'], dict(correct=1, incorrect=1, invalid=1, failed=1, unscored=1))
        self.assertEqual(preview['groups'], [], 'No invented categories for unlabelled data')

    def test_bounded_preview_prioritizes_failures_but_counts_all_samples(self):
        rows = [dict(index=i, is_correct=True, subject='math', question='q' * 4000) for i in range(1, 301)]
        rows.append(dict(index=301, error='timeout', subject='math'))
        preview = reporting.preview({'details': rows})
        self.assertTrue(preview['truncated'])
        self.assertEqual(len(preview['samples']), 200)
        self.assertEqual(preview['samples'][-1]['index'], 301)
        self.assertEqual(preview['groups'][0]['samples'], 301)
        self.assertEqual(preview['counts']['correct'], 300)
        self.assertLessEqual(len(preview['samples'][0]['question']), 2000)

    def test_exam_sections_follow_explicit_runner_ranges(self):
        preview = reporting.preview({'details': [dict(index=i, is_correct=False) for i in range(1, 166)]}, '2024.json')
        self.assertEqual([group['samples'] for group in preview['groups']], [40, 75, 20, 30])
        self.assertEqual(reporting.subset({'index': 1, 'subject': 'recorded subject'}, '2024.json'), 'recorded subject')

    def test_preview_preserves_qa_references_and_bounds_category_payloads(self):
        preview = reporting.preview({'details': [{'answer': None, 'references': ['reference one', 'reference two'],
            'exact_match': False, 'response': 'model answer'}]})
        self.assertIn('reference one', preview['samples'][0]['expected'])
        preview = reporting.preview({'details': [dict(subject=str(i), is_correct=True) for i in range(250)]})
        self.assertEqual(len(preview['groups']), 200)
        self.assertTrue(preview['groupsTruncated'])
        self.assertEqual(preview['counts']['correct'], 250)

    def test_pending_code_is_unscored_and_recorded_instruction_judgments_are_preserved(self):
        result = reporting.preview({'details': [{'executed': False, 'passed': False, 'official_pending': True},
            {'strict_follow_all': True}, {'strict_follow_all': False}]})
        self.assertEqual([item['status'] for item in result['samples']], ['unscored', 'correct', 'incorrect'])

    def test_worker_and_agent_override_use_the_same_preview_implementation(self):
        worker = ROOT / 'inference_ops/experiment_runtime/experiment_benchmark.py'
        agent = ROOT.parent / 'agent/utils/experiment_benchmark.py'
        self.assertEqual(worker.read_bytes(), agent.read_bytes())

    def test_report_read_verifies_frozen_bytes_and_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'result.json'
            path.write_text(json.dumps({'details': [{'index': 1, 'is_correct': True}]}), encoding='utf-8')
            report = {'location': {'path': directory}, 'manifest': {'files': [
                {'path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}]}}
            self.assertEqual(reporting.read_report(report)['counts']['correct'], 1)
            path.write_text('{}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                reporting.read_report(report)
            report['manifest']['files'][0]['path'] = '../private.json'
            with self.assertRaisesRegex(ValueError, 'path'):
                reporting.read_report(report)

    def test_medbench_jsonl_is_unscored_and_does_not_invent_reference_answers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'medical.jsonl'
            path.write_text('{"question":"q","answer":"generated"}\n', encoding='utf-8')
            report = {'location': {'path': directory}, 'manifest': {'files': [
                {'path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}]}}
            sample = reporting.read_report(report)['samples'][0]
            self.assertEqual((sample['status'], sample['expected'], sample['response']), ('unscored', '', 'generated'))

    def test_request_failures_are_not_scored_as_choice_answers(self):
        for name in ('eval_runner', 'general_runner'):
            module = runner(name)
            with patch.object(module, 'call_model', return_value='ERROR: failure'):
                if name == 'eval_runner':
                    result = module.process_one(None, 'model', 1, {'question': 'q', 'answer': 'E'}, '2024.json')
                    self.assertFalse(result['is_correct'])
                    self.assertIsNone(result['predicted'])
                else:
                    result = module.process_sample(None, 'model', {'id': 's1', 'task_type': 'choice', 'question': 'q'})
                    self.assertNotIn('is_correct', result)
                self.assertEqual(result['error'], 'failure')

    def test_retry_logs_are_sanitized_and_stop_sleeping_after_final_attempt(self):
        for name in ('eval_runner', 'general_runner'):
            module = runner(name)
            request = types.SimpleNamespace(parse=lambda **kwargs: (_ for _ in ()).throw(RuntimeError('password=secret')),
                                            create=lambda **kwargs: (_ for _ in ()).throw(RuntimeError('password=secret')))
            client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=request))
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream), patch.object(module.time, 'sleep') as sleep:
                if name == 'eval_runner':
                    result = module.call_model(client, 'model', 'q', 'A', ['A'])
                else:
                    with patch.object(module, 'choice_prompt', return_value='q'):
                        result = module.call_model(client, 'model', {'task_type': 'choice', 'id': 's1'})
            self.assertEqual(sleep.call_count, 2)
            self.assertNotIn('secret', stream.getvalue() + result)
            self.assertEqual(len(stream.getvalue().splitlines()), 3)

    def test_both_runners_log_start_samples_failures_and_final_summary(self):
        for name in ('eval_runner', 'general_runner'):
            module = runner(name)
            with tempfile.TemporaryDirectory() as directory:
                dataset, output = Path(directory) / 'test.json', Path(directory) / 'result.json'
                dataset.write_text('[{"question":"q","answer":"A"},{"question":"r","answer":"B"}]', encoding='utf-8')
                args = types.SimpleNamespace(api_key='private', base_url='http://localhost', model='model',
                    dataset=str(dataset), output=str(output), max_workers=1, save_every=1, sleep=0,
                    dataset_root=directory, split='test', limit=None)
                stream = io.StringIO()
                def evaluate(*values):
                    index = values[2] if name == 'eval_runner' else values[2]['_index'] + 1
                    if index == 2:
                        raise RuntimeError('test failure')
                    return dict(index=index, is_correct=True, invalid=False, f1=1, score=1)
                with contextlib.ExitStack() as stack:
                    stack.enter_context(contextlib.redirect_stdout(stream))
                    stack.enter_context(patch.object(module, 'OpenAI', return_value=object()))
                    stack.enter_context(patch.object(module, 'process_one' if name == 'eval_runner' else 'process_sample', side_effect=evaluate))
                    if name == 'general_runner':
                        for key, value in {'canonical_dataset': 'demo', 'resolve_split': 'test',
                            'dataset_meta': {'name': 'demo', 'type': 'choice'},
                            'load_samples': [{'id': '1'}, {'id': '2'}]}.items():
                            stack.enter_context(patch.object(module, key, return_value=value))
                        module.run(args)
                    else:
                        module.run_eval(args)
                events = [json.loads(line) for line in stream.getvalue().splitlines()]
                self.assertEqual(events[0]['event'], 'benchmark_started')
                self.assertEqual(events[0]['total'], 2)
                self.assertEqual(events[-1]['event'], 'benchmark_finished')
                self.assertEqual((events[-1]['succeeded'], events[-1]['failed']), (1, 1))
                self.assertEqual(len(json.loads(output.read_text(encoding='utf-8'))['details']), 2)

    def test_medbench_records_worker_exception_as_a_failed_sample_and_finishes(self):
        module = runner('eval_runner')
        with tempfile.TemporaryDirectory() as directory:
            source, destination = Path(directory) / 'medical.jsonl', Path(directory) / 'result.jsonl'
            source.write_text('{"question":"q"}\n{"question":"r"}\n', encoding='utf-8')
            def evaluate(client, model, index, item):
                if index:
                    raise RuntimeError('timeout')
                return index, {**item, 'answer': 'response'}
            stream = io.StringIO()
            with patch.object(module, 'call_one_medbench', side_effect=evaluate), contextlib.redirect_stdout(stream):
                module.process_one_medbench(source, destination, None, 'model')
            rows = [json.loads(line) for line in destination.read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[1]['error'], 'timeout')
            final = json.loads(stream.getvalue().splitlines()[-1])
            self.assertEqual((final['event'], final['succeeded'], final['failed']), ('benchmark_finished', 1, 1))


if __name__ == '__main__':
    unittest.main()
