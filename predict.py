"""Predict unlabeled signals using a checkpoint bundle produced by train.py."""
import argparse
from pathlib import Path

import numpy as np
import torch

from best_m3_predict import BestM3


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True, help='signals.npy, shape [B,10,200,6]')
    p.add_argument('--checkpoints', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='New predictions.npz file')
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    if args.output.exists():
        p.error('Refusing to overwrite existing predictions')
    torch.set_num_threads(4)
    base = args.checkpoints
    model = BestM3.from_checkpoints(
        base / 'backbone.pt', base / 'query.pt', base / 'relation.pt',
        base / 'classifier.pt', base / 'classifier_normalization.npz', args.device)
    predictions = model.predict(np.load(args.input, allow_pickle=False))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **predictions)
    print({key: value.shape for key, value in predictions.items()})


if __name__ == '__main__':
    main()
