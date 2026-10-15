"""Exercise the real stop entry point without importing Linux/service dependencies."""
import ast
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


class ServiceStopHistoryTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.meta = {'instance_id': 'service', 'run_id': 'run', 'resource': {'runtime_node_id': 'node'}}
        self.latest = {**self.meta, 'stop_return_code': 1, 'stop_result': 'script returned nonzero; ports confirmed free'}
        self.context = {}
        self.admin = False
        self.thread = ''
        self.actor = 'user'
        self.response = {'_tool_text': 'stopped', '_response_data': {'service_stop': {'stopped': True, 'release_ready': True}}}
        self.module = types.ModuleType('stop_history_test.experiment_records')
        self.module.record_service_stop = self.record
        modules = patch.dict(sys.modules, {'stop_history_test.experiment_records': self.module})
        modules.start()
        self.addCleanup(modules.stop)
        namespace = {'__package__': 'stop_history_test', 'STATE_DIR': '/state',
            'uuid': types.SimpleNamespace(uuid4=lambda: types.SimpleNamespace(hex='operation')),
            'os': types.SimpleNamespace(getenv=lambda key: None),
            'current_request_resource_context': lambda: self.context,
            'current_request_is_admin': lambda: self.admin,
            'current_request_thread_id': lambda: self.thread,
            'current_request_user_id': lambda: self.actor,
            'load_service_instance': lambda instance_id: self.latest,
            'local_tool_response_data': lambda response: response['_response_data'],
            'local_tool_text': lambda response: response['_tool_text'],
            '_stop_service_instance_runtime': self.stop,
        }
        source = Path(__file__).parents[1] / 'inference_ops/tools.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'stop_service_instance_runtime')
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
        self.run_stop = namespace['stop_service_instance_runtime']
        self.tree = tree
        self.namespace = namespace
        self.source = source

    def record(self, meta, **values):
        self.events.append(values)
        return True

    def stop(self, meta):
        self.assertEqual(self.events[0]['status'], 'requested', 'Persist request before changing the service')
        return self.response

    def test_stop_paths_record_actor_trigger_and_final_process_result(self):
        for trigger in ('manual', 'workflow', 'admin', 'system'):
            with self.subTest(trigger=trigger):
                self.events.clear()
                self.context = {'experiment_context': {'workflowId': 'workflow'}} if trigger == 'workflow' else {}
                self.thread = 'workflow-manager' if trigger == 'workflow' else ''
                self.admin = trigger == 'admin'
                self.actor = '' if trigger == 'system' else 'user'
                self.assertIs(self.run_stop(self.meta), self.response)
                self.assertEqual([event['status'] for event in self.events], ['requested', 'succeeded'])
                self.assertEqual(self.events[0]['operation_id'], self.events[1]['operation_id'])
                self.assertEqual(self.events[1]['trigger'], trigger)
                self.assertEqual(self.events[1]['actor'], self.actor)
                self.assertEqual(self.events[1]['cause'], 'stop')
                self.assertEqual(self.events[1]['return_code'], 1)
                self.assertTrue(self.events[1]['release_ready'])
                self.assertIn('ports confirmed free', self.events[1]['result'])

    def test_workflow_reservation_stop_uses_its_thread_identity(self):
        self.thread = 'workflow-manager'
        self.run_stop(self.meta)
        self.assertEqual(self.events[0]['trigger'], 'workflow')

    def test_manual_request_linked_to_workflow_is_not_an_automatic_workflow_action(self):
        self.context = {'experiment_context': {'workflowId': 'workflow'}}
        self.run_stop(self.meta)
        self.assertEqual(self.events[0]['trigger'], 'manual')

    def test_automatic_workflow_request_has_explicit_provenance_even_for_an_admin(self):
        self.context = {'experiment_context': {'workflowId': 'workflow', 'operationTrigger': 'workflow'}}
        self.admin = True
        self.run_stop(self.meta)
        self.assertEqual(self.events[0]['trigger'], 'workflow')

    def test_restart_propagates_its_cause_through_the_actual_stop_entry_points(self):
        self.admin = True
        namespace = self.namespace
        namespace.update({
            'active_service_instance_candidates': lambda **kwargs: [self.meta],
            'visible_service_instance_meta': lambda instance: self.meta,
            'running_instance_tasks_block_text': lambda *args: '',
            'running_tasks_for_service_instance': lambda instance: [],
            'time': types.SimpleNamespace(sleep=lambda seconds: None),
            'start_service': lambda: {'_tool_text': 'new instance started', '_response_data': {'service_start': {}}},
            'build_local_tool_response': lambda text, data: {'_tool_text': text, '_response_data': data},
        })
        for name in ('stop_service', 'stop_service_instance', 'restart_service_stack'):
            node = next(node for node in self.tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(self.source), 'exec'), namespace)
        response = namespace['restart_service_stack']()
        self.assertEqual([event['cause'] for event in self.events], ['restart', 'restart'])
        self.assertIn('new instance started', response['_tool_text'])

    def test_failed_stop_keeps_error_and_never_confirms_release(self):
        self.response = {'_tool_text': 'port busy', '_response_data': {'service_stop': {'stopped': False, 'release_ready': False}}}
        self.run_stop(self.meta)
        self.assertEqual(self.events[1]['status'], 'failed')
        self.assertFalse(self.events[1]['release_ready'])
        self.assertIn('port busy', self.events[1]['result'])

    def test_unexpected_error_records_failure_before_propagating(self):
        def fail(meta):
            raise RuntimeError('stop failed')
        self.run_stop.__globals__['_stop_service_instance_runtime'] = fail
        with self.assertRaisesRegex(RuntimeError, 'stop failed'):
            self.run_stop(self.meta)
        self.assertEqual(self.events[1]['status'], 'failed')

    def test_unavailable_journal_does_not_block_stop_or_queue_result_without_request(self):
        def unavailable(*args, **kwargs):
            raise OSError('journal unavailable')
        self.module.record_service_stop = unavailable
        self.run_stop.__globals__['_stop_service_instance_runtime'] = lambda meta: self.response
        self.run_stop(self.meta)
        self.assertTrue(self.response['_response_data']['service_stop']['stopped'])
        self.assertIn('experiment_recording_error', self.response['_response_data'])
        self.assertEqual(self.events, [])


if __name__ == '__main__':
    unittest.main()
