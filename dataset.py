"""Explicit scene contract; split isolation checks never infer groups from actions."""
import hashlib
import json
from pathlib import Path

import numpy as np

from scene_data import Scene, overlap_audit
from training_loss import validate_group_labels


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def load_scenes(path):
    with np.load(path, allow_pickle=False) as archive:
        missing = {'signals', 'groups', 'activities', 'actions', 'source_ids'} - set(archive.files)
        if missing:
            raise ValueError(f'Missing scene fields: {sorted(missing)}')
        data = {name: archive[name].copy() for name in
                ('signals', 'groups', 'activities', 'actions', 'source_ids')}
    x = data['signals']
    if x.ndim != 4 or x.shape[1:] != (10, 200, 6) or not len(x):
        raise ValueError('signals must have nonempty shape [B,10,200,6]')
    if not np.issubdtype(x.dtype, np.number) or not np.isfinite(x).all():
        raise ValueError('Signals must be finite numeric values')
    for key, low, high in [('groups', 0, 5), ('activities', -1, 9), ('actions', 0, 9)]:
        value = data[key]
        if value.shape != x.shape[:2] or not np.issubdtype(value.dtype, np.integer):
            raise ValueError(f'{key} must be an integer [B,10] array')
        if ((value < low) | (value > high)).any():
            raise ValueError(f'{key} outside [{low},{high}]')
        data[key] = value.astype(np.int64)
    ids = data['source_ids']
    if ids.shape != x.shape[:2] or ids.dtype.kind not in 'US' or (ids.astype(str) == '').any():
        raise ValueError('source_ids must be nonempty string IDs with shape [B,10]')
    data['source_ids'] = ids.astype(str)
    validate_group_labels(data['groups'], data['activities'])
    for row in data['groups']:
        if not 2 <= len(np.unique(row)) <= 6:
            raise ValueError('Each training/evaluation scene needs 2-6 annotated groups')
    data['signals'] = x.astype(np.float32)
    return data


def audit_splits(parts):
    ids = {name: set(data['source_ids'].astype(str).ravel()) for name, data in parts.items()}
    names = list(parts)
    for i, left in enumerate(names):
        for right in names[i+1:]:
            if ids[left] & ids[right]:
                raise ValueError(f'Source IDs overlap: {left}/{right}')
    scenes = {name: [Scene(x, g, a, p, np.empty((10, 2), dtype=np.int64))
                     for x, g, a, p in zip(data['signals'], data['groups'],
                                           data['activities'], data['actions'])]
              for name, data in parts.items()}
    audit = overlap_audit(scenes)
    if any(any(record.values()) for record in audit.values()):
        raise ValueError(f'Exact full/half signal overlap across splits: {audit}')
    return audit


def fingerprints(data):
    keys = set()
    for person in data['signals'].reshape(-1, 200, 6):
        keys.add(hashlib.sha256(np.ascontiguousarray(person).tobytes()).hexdigest())
        for half in np.split(person, 2):
            if np.max(np.std(half, axis=0)) > 1e-6:
                keys.add(hashlib.sha256(np.ascontiguousarray(half).tobytes()).hexdigest())
    return keys
