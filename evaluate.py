"""Evaluate a locked checkpoint bundle on a disjoint labeled scene archive."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from best_m3_predict import BestM3
from dataset import load_scenes, fingerprints, file_hash, write_json
from workflow import metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--checkpoints', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    torch.set_num_threads(4)
    if args.output.exists():
        p.error('Use a new output directory to preserve previous evaluations')
    data = load_scenes(args.data)
    base = args.checkpoints
    provenance = json.loads((base / 'selection_provenance.json').read_text())
    if set(data['source_ids'].astype(str).ravel()) & set(provenance['source_ids']):
        raise ValueError('Evaluation sources overlap training/validation')
    if fingerprints(data) & set(provenance['window_fingerprints']):
        raise ValueError('Evaluation full/half signals overlap training/validation')
    names = {'backbone': 'backbone.pt', 'query': 'query.pt', 'relation': 'relation.pt',
             'classifier': 'classifier.pt', 'normalization': 'classifier_normalization.npz'}
    paths = {k: base/v for k, v in names.items()}
    model = BestM3.from_checkpoints(**paths, device=args.device)
    prediction = model.predict(data['signals'])
    result = metrics(data, prediction)
    result['audit'] = {'data_sha256': file_hash(args.data),
                       'artifact_sha256': {k: file_hash(v) for k, v in paths.items()},
                       'source_and_exact_window_overlap': 0}
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / 'predictions.npz', **prediction)
    write_json(args.output / 'metrics.json', result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
