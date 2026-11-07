"""Train and export all checkpoints needed by BestM3, without private run paths."""
import argparse
from pathlib import Path
import platform

import numpy as np
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

from best_m3_model import M3Query, RelationConfidence, GroupClassifier
from best_m3_predict import BestM3
from dataset import load_scenes, audit_splits, fingerprints, file_hash, write_json
from workflow import (PretrainingModel, seed_all, joint_loss, query_cache,
                      relation_scores, classifier_data, balanced_bce, metrics)


def fit(module, size, loss_fn, score_fn, epochs, batch_size, lr, folder, name, rng, history):
    optimizer = torch.optim.AdamW(module.parameters(), lr=lr, weight_decay=.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(epochs, 1), eta_min=1e-5)
    best = -float('inf')
    path = folder / f'{name}.pt'
    for epoch in range(1, epochs+1):
        module.train()
        order = rng.permutation(size)
        losses = []
        for start in range(0, size, batch_size):
            ids = order[start:start+batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(ids)
            if not torch.isfinite(loss):
                raise ValueError(f'Nonfinite {name} loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        module.eval()
        score = float(score_fn())
        if not np.isfinite(score):
            raise ValueError('Nonfinite validation score')
        row = {'stage': name, 'epoch': epoch, 'train_loss': float(np.mean(losses)), 'selection_score': score}
        history.append(row)
        print(row, flush=True)
        write_json(folder / 'history.json', history)
        if score > best:
            best = score
            torch.save({'model': module.state_dict(), 'epoch': epoch, 'validation_score': score}, path)
    if epochs:
        module.load_state_dict(torch.load(path, map_location=next(module.parameters()).device, weights_only=False)['model'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train', type=Path, required=True)
    p.add_argument('--val', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--ssl-epochs', type=int, default=20)
    p.add_argument('--backbone-epochs', type=int, default=10)
    p.add_argument('--query-epochs', type=int, default=30)
    p.add_argument('--relation-epochs', type=int, default=10)
    p.add_argument('--classifier-epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--learning-rate', type=float, default=.001)
    p.add_argument('--seed', type=int, default=2024)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--device', default='cpu')
    p.add_argument('--smoke', action='store_true', help='One epoch/stage on at most eight scenes; not a result experiment')
    args = p.parse_args()
    if args.output.exists():
        p.error('Use a new output directory; existing checkpoints are never overwritten')
    if min(args.batch_size, args.threads) < 1 or args.learning_rate <= 0:
        p.error('Invalid batch size, thread count or learning rate')
    if min(args.ssl_epochs, args.backbone_epochs) < 0 or min(args.query_epochs, args.relation_epochs, args.classifier_epochs) < 1:
        p.error('Pretraining epochs must be nonnegative and downstream stage epochs positive')
    torch.set_num_threads(args.threads)
    seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    parts = {'train': load_scenes(args.train), 'val': load_scenes(args.val)}
    audit = audit_splits(parts)
    args.output.mkdir(parents=True)
    provenance = {'source_ids': sorted(set(np.concatenate([v['source_ids'].astype(str).ravel() for v in parts.values()]))),
                  'window_fingerprints': sorted(set.union(*(fingerprints(v) for v in parts.values())))}
    write_json(args.output / 'selection_provenance.json', provenance)
    if args.smoke:
        parts = {split: {k: v[:8] for k, v in data.items()} for split, data in parts.items()}
        args.ssl_epochs = args.backbone_epochs = args.query_epochs = args.relation_epochs = args.classifier_epochs = 1
        args.batch_size = 2
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({'python': platform.python_version(), 'torch': torch.__version__,
                   'train_sha256': file_hash(args.train), 'val_sha256': file_hash(args.val),
                   'split_overlap_audit': audit,
                   'training_scope': 'Public staged retraining workflow for the released architecture; not a claim to reproduce historical or paper table scores.',
                   'selection': 'train/validation only; test data not accepted by this command',
                   'source_sha256': {f.name: file_hash(f) for f in Path(__file__).parent.glob('*.py')}})
    write_json(args.output / 'config.json', config)
    train, val = parts['train'], parts['val']
    mean = train['signals'].astype(np.float64).mean((0, 1, 2)).astype(np.float32)
    std = train['signals'].astype(np.float64).std((0, 1, 2)).clip(1e-5).astype(np.float32)
    signal = np.clip((train['signals']-mean)/std, -10, 10).reshape(-1, 200, 6)
    window_count = min(len(signal), 16) if args.smoke else len(signal)
    x = torch.as_tensor(signal[:window_count], device=device)
    activities = torch.as_tensor(np.where(train['activities'].ravel()[:window_count] < 0, 10, train['activities'].ravel()[:window_count]), device=device)
    actions = torch.as_tensor(train['actions'].ravel()[:window_count], device=device)
    pre = PretrainingModel().to(device)
    history = []
    # SSL selects the last epoch, not any validation/test labels.
    fit(pre, len(x), lambda ids: pre.ssl_loss(x[ids]) if len(ids) > 1 else pre.ssl_loss(x[:2]),
        lambda: float(len([r for r in history if r['stage'] == 'ssl'])),
        args.ssl_epochs, max(2, args.batch_size), args.learning_rate, args.output, 'ssl', rng, history)

    def supervised(ids):
        features = pre.encoder(x[ids]).mean(1)
        return F.cross_entropy(pre.activity_head(features), activities[ids]) + .1*F.cross_entropy(pre.action_head(features), actions[ids])

    fit(pre, len(x), supervised,
        lambda: float(len([r for r in history if r['stage'] == 'backbone_warmup'])),
        args.backbone_epochs, args.batch_size, args.learning_rate, args.output, 'backbone_warmup', rng, history)
    torch.save({'model': {'indiv_module.'+k: v for k, v in pre.encoder.state_dict().items()},
                'normalization': {'mean': mean, 'std': std}}, args.output / 'backbone.pt')
    query, relation, classifier = M3Query().to(device), RelationConfidence().to(device), GroupClassifier().to(device)
    wrapper = BestM3(pre.encoder, query, relation, classifier, mean, std, np.zeros(771), np.ones(771), device)
    components = {name: wrapper.encode(data['signals']) for name, data in parts.items()}
    groups = torch.as_tensor(train['groups'], device=device)
    acts = torch.as_tensor(train['activities'], device=device)
    action = torch.as_tensor(train['actions'], device=device)
    features = torch.as_tensor(components['train'], device=device)

    def query_score():
        value = metrics(val, query_cache(query, components['val'], device))
        return .5*(value['membership_AP_global_off_diagonal'] + value['recognition_accuracy'])

    fit(query, len(groups), lambda ids: joint_loss(query(features[ids]), groups[ids], acts[ids], action[ids]),
        query_score, args.query_epochs, args.batch_size, args.learning_rate, args.output, 'query', rng, history)
    caches = {name: query_cache(query, value, device) for name, value in components.items()}
    upper = torch.triu(torch.ones(10, 10, dtype=torch.bool, device=device), 1)
    truth = (groups[:, :, None] == groups[:, None, :])[:, upper]
    rt = [torch.as_tensor(caches['train'][k], device=device) for k in ('features', 'affinity', 'posterior')]

    def relation_score():
        cache = dict(caches['val'], affinity=relation_scores(relation, caches['val'], device))
        return metrics(val, cache)['membership_AP_global_off_diagonal']

    fit(relation, len(groups), lambda ids: balanced_bce(relation(*(v[ids] for v in rt))[:, upper], truth[ids]),
        relation_score, args.relation_epochs, args.batch_size, args.learning_rate, args.output, 'relation', rng, history)
    train_affinity = relation_scores(relation, caches['train'], device)
    gx, prior, slots, counts = classifier_data(caches['train'], train_affinity, train['activities'])
    valid = counts.sum(-1) > 0
    cm = gx[valid].astype(np.float64).mean(0).astype(np.float32)
    cs = gx[valid].astype(np.float64).std(0).clip(1e-3).astype(np.float32)
    np.savez_compressed(args.output / 'classifier_normalization.npz', mean=cm, std=cs)
    wrapper.classifier_mean, wrapper.classifier_std = cm, cs
    cx = torch.as_tensor(np.clip((gx[valid]-cm)/cs, -10, 10).astype(np.float32), device=device)
    targets = torch.as_tensor(counts[valid], device=device)

    def classification_loss(ids):
        counts = targets[ids]
        weights = counts.sum(-1)
        smoothed = .98*counts + .02*weights[:, None]/11
        return -(smoothed*classifier(cx[ids]).log_softmax(-1)).sum()/weights.sum().clamp_min(1)

    fit(classifier, len(cx), classification_loss,
        lambda: metrics(val, wrapper.predict_components(components['val']))['recognition_accuracy'],
        args.classifier_epochs, args.batch_size, args.learning_rate, args.output, 'classifier', rng, history)
    write_json(args.output / 'validation_metrics.json', metrics(val, wrapper.predict_components(components['val'])))
    print('TRAINING_COMPLETE: exported five inference artifacts; no test evaluation was performed.', flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
