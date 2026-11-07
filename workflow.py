"""Training helpers for the public M3 architecture and its existing inference API."""
import random

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_fscore_support
import torch
from torch import nn
from torch.nn import functional as F

from augment import augment, speed_transform
from best_m3_model import SignalEncoder
from best_m3_predict import _deterministic_inference, _group_statistics
from training_loss import batched_joint_loss


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    _deterministic_inference()


def balanced_bce(logits, truth):
    loss = F.binary_cross_entropy_with_logits(logits, truth.float(), reduction='none')
    return torch.stack([loss[m].mean() for m in (truth, ~truth) if m.any()]).mean()


class PretrainingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SignalEncoder()
        self.projector = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 64))
        self.future_decoder = nn.LSTM(6, 128, batch_first=True)
        self.future_output = nn.Linear(128, 6)
        self.speed_head = nn.Linear(128, 4)
        self.activity_head = nn.Linear(128, 11)
        self.action_head = nn.Linear(128, 10)

    def ssl_loss(self, x):
        if len(x) < 2:
            raise ValueError('Contrastive learning needs at least two individual windows')
        z1 = self.projector(self.encoder(augment(x)).mean(1))
        z2 = self.projector(self.encoder(augment(x)).mean(1))
        z = F.normalize(torch.cat([z1, z2]), dim=-1)
        logits = z @ z.T / .2
        logits = logits.masked_fill(torch.eye(len(z), device=z.device, dtype=torch.bool), -1e4)
        targets = (torch.arange(len(z), device=z.device) + len(x)) % len(z)
        contrastive = F.cross_entropy(logits, targets)
        context = self.encoder(x[:, :180]).mean(1).unsqueeze(0)
        future, _ = self.future_decoder(x.new_zeros(len(x), 20, 6), (context, torch.zeros_like(context)))
        forecast = F.mse_loss(self.future_output(future), x[:, 180:])
        view, speed = speed_transform(x, (.5, 1., 1.5, 2.))
        speed_loss = F.cross_entropy(self.speed_head(self.encoder(view).mean(1)), speed)
        return contrastive + forecast + speed_loss


def joint_loss(out, groups, activities, actions):
    base = batched_joint_loss(out[:4], groups, activities, actions)
    upper = torch.triu(torch.ones(10, 10, dtype=torch.bool, device=groups.device), 1)
    truth = (groups[:, :, None] == groups[:, None, :])[:, upper]
    pair = balanced_bce(out[4][:, upper], truth)
    return base + .25 * pair


@torch.no_grad()
def query_cache(model, components, device):
    model.eval()
    parts = [[] for _ in range(5)]
    for start in range(0, len(components), 64):
        out = model(torch.as_tensor(components[start:start+64], device=device))
        values = [x.softmax(-1).cpu().numpy() for x in out[:3]]
        values += [out[4].sigmoid().cpu().numpy(), out[5].cpu().numpy()]
        for dest, value in zip(parts, values):
            dest.append(value)
    members, classes, person, pair, features = map(np.concatenate, parts)
    active = classes.argmax(-1) != 11
    for i in range(len(active)):
        if not active[i].any():
            active[i, classes[i, :, :11].max(-1).argmax()] = True
    assignment = members * active[:, None, :]
    assignment /= assignment.sum(-1, keepdims=True).clip(1e-7)
    groups = assignment.argmax(-1)
    posterior = assignment @ classes[:, :, :11]
    posterior /= posterior.sum(-1, keepdims=True).clip(1e-7)
    posterior = .5 * (posterior + person)
    affinity = .5 * (assignment @ assignment.transpose(0, 2, 1)) + .5 * pair
    labels = posterior.argmax(-1)
    return {'groups': groups, 'activities': np.where(labels == 10, -1, labels),
            'affinity': affinity, 'features': features, 'posterior': posterior}


@torch.no_grad()
def relation_scores(model, cache, device):
    model.eval()
    values = []
    for start in range(0, len(cache['groups']), 64):
        inputs = [torch.as_tensor(cache[k][start:start+64], device=device)
                  for k in ('features', 'affinity', 'posterior')]
        values.append(model(*inputs).sigmoid().cpu().numpy())
    return np.concatenate(values)


def classifier_data(cache, affinity, activities):
    x, prior, slots = _group_statistics(cache['features'], cache['posterior'], cache['groups'], affinity)
    counts = np.zeros_like(prior)
    truth = np.where(activities < 0, 10, activities)
    for i in range(len(x)):
        np.add.at(counts[i], (slots[i], truth[i]), 1)
    return x, prior, slots, counts


def metrics(truth, prediction):
    mask = ~np.eye(10, dtype=bool)
    same = (truth['groups'][:, :, None] == truth['groups'][:, None, :])[:, mask].ravel()
    partition = (prediction['groups'][:, :, None] == prediction['groups'][:, None, :])[:, mask].ravel()
    scores = prediction['affinity'][:, mask].ravel()
    p, r, f, _ = precision_recall_fscore_support(same, partition, average='binary', zero_division=0)
    valid = truth['activities'] >= 0
    if not valid.any():
        raise ValueError('No non-noise behavior targets to evaluate')
    ap = float(average_precision_score(same, scores)) if same.any() else 0.
    cp, cr, cf, _ = precision_recall_fscore_support(
        truth['activities'][valid], prediction['activities'][valid], labels=np.arange(10),
        average='macro', zero_division=0)
    return {'membership_AP_global_off_diagonal': ap, 'segmentation_pair_precision': float(p),
            'segmentation_pair_recall': float(r), 'segmentation_pair_f1': float(f),
            'recognition_macro_precision': float(cp), 'recognition_macro_recall': float(cr),
            'recognition_macro_f1': float(cf),
            'recognition_accuracy': float(accuracy_score(truth['activities'][valid], prediction['activities'][valid])),
            'scenes': len(truth['groups']), 'recognition_non_noise_people': int(valid.sum()),
            'metric_notes': 'Scores are fractions. AP pools directed off-diagonal membership scores; it is not group-set IoU mAP. Segmentation P/R/F1 evaluate person pairs, not detected group sets. Recognition excludes noise ground truth and macro-averages over fixed classes 0-9.'}
