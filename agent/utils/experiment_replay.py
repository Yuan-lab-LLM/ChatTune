"""Replay a Studio-owned draft through the normal permission/resource-aware tools."""
from __future__ import annotations

import threading
from .experiment_tracker import get_experiment_tracker

_lock = threading.Lock()


def launch_draft(payload):
    tracker = get_experiment_tracker()
    if payload['nodeId'] != tracker.node_id:
        raise ValueError('Draft is assigned to another Runtime node')
    request_key = 'draft-' + payload['experimentId']
    # Serialize draft submissions through preflight, including requests whose HTTP response was lost.
    with _lock:
        with tracker.connect() as db:
            prior = db.execute('SELECT handle_json, claimed FROM experiment_intent WHERE request_key=?', (request_key,)).fetchone()
        if prior and prior['handle_json']:
            import json
            handle = json.loads(prior['handle_json'])
            if prior['claimed']:
                execution = tracker.execution(handle['attemptId'])
                if execution['actor_user_id'] != payload['actorUserId']:
                    raise ValueError('Draft execution owner mismatch')
                return {'success': True, 'experiment': handle, 'status': execution['state'], 'reused': True}
        spec = payload['spec']
        script = spec['script']
        resolved = spec.get('resolved') or {}
        # Resolved paths point to the original immutable input versions, not "latest".
        arguments = {key: value for key, value in (spec.get('requested') or {}).items()
                     if key in {'gpu-count', 'node-count', 'gpus-per-node', 'pool-id'}}
        arguments.update(resolved.get('arguments') or {})
        for argument in resolved.get('positionalArguments') or []:
            key, separator, value = str(argument).partition('=')
            if separator:
                arguments[key] = value
        env = {**(resolved.get('environment') or {}), 'container': payload['container'],
               'MEDFLOW_RESOURCE_GROUP_ID': payload.get('groupId') or ''}
        context = {'requestKey': request_key, 'experimentId': payload['experimentId'],
            'name': payload['name'], 'project': payload.get('project', 'default'),
            'expectedInputVersions': spec.get('inputVersions') or [],
            'replayInputAssets': payload.get('inputAssets') or []}
        if resolved.get('resumeCheckpointName'):
            context['resumeCheckpointName'] = resolved['resumeCheckpointName']
        if payload['kind'] == 'train':
            from medflow_agent_tools.runlocal_train import run_script_by_name_train
            tool = run_script_by_name_train
        elif payload['kind'] == 'evaluate':
            from medflow_agent_tools.runlocal_evaluate import run_script_by_name_evaluate
            tool = run_script_by_name_evaluate
        else:
            raise ValueError('Only training and evaluation drafts can be launched from experiment records')
        response = tool(script, additional_args=arguments, env_vars=env, background=True,
            actor_user_id=payload['actorUserId'], experiment_context=context)
        metadata = response.metadata or {}
        if not metadata.get('success'):
            hint = metadata.get('protocol_hint') or {}
            raise ValueError(hint.get('message') or hint.get('errorReason') or 'Draft launch preflight failed')
        return {'success': True, 'experiment': metadata.get('experiment'),
                'status': (metadata.get('protocol_hint') or {}).get('status'), 'reused': False}
