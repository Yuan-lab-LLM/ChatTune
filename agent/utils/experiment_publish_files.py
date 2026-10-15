"""Small dependency-free destination operations, executed inside the target container."""
from __future__ import annotations

import json
from pathlib import Path
import re


def paths(request):
    destination = Path(request['destination'])
    attempt = request['attemptId']
    if not re.fullmatch(r'[A-Za-z0-9_-]+', attempt) or not destination.is_absolute() or '..' in destination.parts:
        raise ValueError('Invalid publication destination or attempt')
    if not destination.name or destination.parent == destination:
        raise ValueError('Cannot publish to a filesystem root')
    # Never traverse an existing symlink in a target path.
    if any(path.is_symlink() for path in (destination, *destination.parents)):
        raise ValueError('Publication destination cannot traverse symbolic links')
    lock = destination.with_name('.' + destination.name + '.publish-lock')
    staging = destination.with_name('.' + destination.name + '.publish-' + attempt)
    return destination, lock, staging


def operate(request):
    destination, lock, staging = paths(request)
    operation = request['operation']
    owner = lock / 'attempt.json'
    if operation == 'release_if_owned':
        if lock.is_symlink() or not owner.is_file() or json.loads(owner.read_text(encoding='utf-8')).get('attemptId') != request['attemptId']:
            return {'released': False}
        operation = 'release'
    if operation == 'claim':
        destination.parent.mkdir(parents=True, exist_ok=True)
        lock.mkdir()  # Exclusive; a second publication cannot race activation.
        try:
            owner.write_text(json.dumps({'attemptId': request['attemptId']}), encoding='utf-8')
            if staging.exists() or staging.is_symlink():
                raise ValueError('Publication staging directory already exists')
            staging.mkdir()
        except BaseException:
            if owner.is_file():
                owner.unlink()
            lock.rmdir()
            raise
        return {'staging': str(staging), 'destinationExists': destination.exists()}
    if lock.is_symlink() or not owner.is_file() or json.loads(owner.read_text(encoding='utf-8')).get('attemptId') != request['attemptId']:
        raise ValueError('Publication destination lock belongs to another execution')
    if operation == 'activate':
        if destination.exists() or destination.is_symlink():
            raise ValueError('Publication destination already exists; it will not be overwritten')
        if staging.is_symlink() or not staging.is_dir():
            raise ValueError('Invalid publication staging directory')
        staging.rename(destination)
        return {'path': str(destination)}
    if operation == 'release':
        # Only remove our empty control directory. Failed transfer data is retained
        # for inspection and is never treated as a published model.
        owner.unlink()
        lock.rmdir()
        if staging.is_dir() and not staging.is_symlink() and not any(staging.iterdir()):
            staging.rmdir()
        return {'released': True}
    raise ValueError('Unknown publication operation')


if __name__ == '__main__':
    import sys
    print(json.dumps(operate(json.load(sys.stdin))))
