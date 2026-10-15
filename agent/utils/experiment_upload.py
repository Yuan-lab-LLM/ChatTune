"""Durable dataset-import records, reconciled independently of the HTTP request."""
from __future__ import annotations

import uuid

from .experiment_assets import manifest, validate_dataset_references
from .experiment_capture import asset_registration, container_asset


def prepare_dataset_upload(tracker, *, actor_user_id, container, source, destination,
                           filename, archive_sha256, dataset_type):
    validate_dataset_references(source)
    content = manifest(source)
    spec = {'schemaVersion': 1, 'kind': 'preprocess', 'script': 'dataset_upload',
            'operation': 'import', 'filename': filename, 'archiveSha256': archive_sha256,
            'datasetType': dataset_type, 'contentFingerprint': content['fingerprint']}
    request_key = f'upload-{uuid.uuid4()}'
    handle = tracker.prepare(request_key=request_key, actor_user_id=actor_user_id,
        name=f'Upload {destination.rsplit("/", 1)[-1]}', kind='preprocess', container=container, spec=spec)
    if not tracker.claim_launch(request_key):
        raise ValueError('Dataset upload execution was already claimed')
    tracker.emit(handle['attemptId'], 'configuration', {'configuration': {
        'schemaVersion': 1, 'requested': spec, 'resolved': {'destination': destination},
        'actual': None, 'completeness': 'incomplete', 'missing': ['uploaded_content_verification']}})
    # Persist the expected content before copying. After a crash the collector can
    # distinguish a complete upload from a partial copy without the original ZIP.
    tracker.emit(handle['attemptId'], 'status', {'status': 'running', 'runtime': {
        'operation': 'dataset_upload', 'kind': 'preprocess', 'archiveStatus': 'pending',
        'upload': {'destination': destination, 'expectedFingerprint': content['fingerprint'],
                   'filename': filename, 'archiveSha256': archive_sha256, 'datasetType': dataset_type}}})
    return handle


def collect_dataset_upload(tracker, execution, *, capture_fn=container_asset):
    runtime = execution['runtime']
    if runtime.get('archiveStatus') == 'complete':
        return
    attempt, upload = execution['attempt_id'], runtime['upload']
    key = 'uploaded-dataset'
    registration = tracker.event_payload(attempt, key)
    if not registration:
        args = (execution['container'], upload['destination'], execution['actor_user_id'], 'dataset')
        # Do not create immutable versions of partially copied data.
        observed = capture_fn(*args, operation='manifest')
        if observed['fingerprint'] != upload['expectedFingerprint']:
            raise ValueError('Uploaded data is incomplete or has changed; expected content has not been verified')
        captured = capture_fn(*args)
        if captured['fingerprint'] != upload['expectedFingerprint']:
            raise ValueError('Uploaded data changed while creating its immutable version')
        registration = asset_registration(captured, node_id=tracker.node_id, container=execution['container'],
            owner_id=execution['actor_user_id'], source=upload['destination'], kind='dataset',
            role='uploaded_dataset', direction='output')
        registration['metadata'].update({'origin': 'upload', 'archiveSha256': upload['archiveSha256'],
                                         'originalFilename': upload['filename'], 'datasetType': upload['datasetType']})
        tracker.emit_once(attempt, 'artifact', registration, key)
    if not tracker.event_payload(attempt, 'upload-configuration'):
        initial = tracker.latest_payload(attempt, 'configuration')['configuration']
        tracker.emit_once(attempt, 'configuration', {'configuration': {**initial,
            'actual': {'contentFingerprint': registration['fingerprint'], 'assetVersionId': registration['id']},
            'completeness': 'complete', 'missing': []}}, 'upload-configuration')
    tracker.emit(attempt, 'status', {'status': 'succeeded', 'runtime': {
        'archiveStatus': 'complete', 'archiveError': None, 'uploadedVersionId': registration['id']}}, preserve_terminal=True)
