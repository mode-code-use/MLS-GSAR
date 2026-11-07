"""Prepare UT-Group inputs with partition-local construction and overlap checks."""
import argparse
from pathlib import Path

import numpy as np

from dataset import audit_splits, load_scenes
from ut_data import prepare


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=2024)
    p.add_argument('--labeled-ratio', type=float, default=1.0)
    p.add_argument('--accept-synthetic-scenes', action='store_true')
    args = p.parse_args()
    if not args.accept_synthetic_scenes:
        p.error('Read TRAINING.md and explicitly pass --accept-synthetic-scenes')
    if args.output.exists():
        raise FileExistsError('Use a new output directory; existing data are never overwritten')
    data = prepare(args.data, args.labeled_ratio, args.seed, args.output)
    archive_id = data.manifest['sha256']
    for name, scenes in data.scenes.items():
        refs = np.stack([s.refs for s in scenes])
        source_ids = np.array([[f'{archive_id}:{row}:{person}' for row, person in scene]
                               for scene in refs])
        np.savez_compressed(args.output / f'{name}.npz',
                            signals=np.stack([s.x for s in scenes]),
                            groups=np.stack([s.groups for s in scenes]),
                            activities=np.stack([s.activities for s in scenes]),
                            actions=np.stack([s.actions for s in scenes]), source_ids=source_ids)
    audit_splits({name: load_scenes(args.output / f'{name}.npz') for name in data.scenes})
    print({name: len(values) for name, values in data.scenes.items()})
    print('Prepared train/val/test.npz and data_manifest.json; scenes are synthetic.')


if __name__ == '__main__':
    main()
