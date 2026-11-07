"""Generate random execution-test fixtures, never research benchmark results."""
import argparse
from pathlib import Path

import numpy as np


def make_split(count, split, seed):
    rng = np.random.default_rng(seed)
    groups = np.tile(np.repeat(np.arange(4), [3, 3, 3, 1]), (count, 1))
    activities = groups.copy()
    activities[groups == 3] = -1
    return {'signals': rng.normal(size=(count, 10, 200, 6)).astype(np.float32),
            'groups': groups.astype(np.int64), 'activities': activities.astype(np.int64),
            'actions': (groups % 3).astype(np.int64),
            'source_ids': np.array([[f'{split}:{i}:{j}' for j in range(10)] for i in range(count)])}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for i, (split, count) in enumerate((('train', 8), ('val', 4), ('test', 4))):
        np.savez_compressed(args.output / f'{split}.npz', **make_split(count, split, 100+i))
    print('Random fixtures only; not a benchmark dataset.')
