"""M3-only inference. Inputs are sensor signals, never labels or source IDs.

Supply the four trained checkpoints and classifier normalization separately.
All paths are explicit; no historical experiment scripts are imported.
"""
import os
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
import torch

from best_m3_model import GroupClassifier, M3Query, RelationConfidence, SignalEncoder


def _deterministic_inference():
    # Match the numerical backend settings used for the selected experiment.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def _state(path, device):
    # Load only checkpoint files from a trusted source.
    return torch.load(Path(path), map_location=device, weights_only=False)


def _group_statistics(features, prior, groups, affinity):
    b, n, _ = features.shape
    x = np.zeros((b, n, 771), dtype=np.float32)
    q = np.zeros((b, n, 11), dtype=np.float32)
    slots = np.zeros((b, n), dtype=np.int64)
    for i in range(b):
        for slot, group in enumerate(np.unique(groups[i])):
            ids = np.flatnonzero(groups[i] == group)
            other = np.flatnonzero(groups[i] != group)
            z, p = features[i, ids], prior[i, ids]
            sub = affinity[i][np.ix_(ids, ids)]
            cohesion = float(sub[~np.eye(len(ids), dtype=bool)].mean()) if len(ids) > 1 else 1.
            external = float(affinity[i][np.ix_(ids, other)].mean()) if len(other) else 0.
            x[i, slot] = np.concatenate([z.mean(0), z.max(0), z.std(0), [len(ids)/n, cohesion, external]])
            q[i, slot] = p.mean(0)
            q[i, slot] /= max(float(q[i, slot].sum()), 1e-7)
            slots[i, ids] = slot
    return x, q, slots


class BestM3:
    """Selected optimized M3: fixed marginal decoder, confidence and .25 mixture."""
    def __init__(self, encoder, query, relation, classifier, signal_mean, signal_std,
                 classifier_mean, classifier_std, device="cpu"):
        _deterministic_inference()
        self.device = torch.device(device)
        self.encoder = encoder.to(self.device).eval()
        self.query = query.to(self.device).eval()
        self.relation = relation.to(self.device).eval()
        self.classifier = classifier.to(self.device).eval()
        self.signal_mean = np.asarray(signal_mean, dtype=np.float32)
        self.signal_std = np.asarray(signal_std, dtype=np.float32)
        self.classifier_mean = np.asarray(classifier_mean, dtype=np.float32)
        self.classifier_std = np.asarray(classifier_std, dtype=np.float32)
        if self.signal_mean.shape != (6,) or self.signal_std.shape != (6,):
            raise ValueError("Expected six-channel signal normalization")
        if self.classifier_mean.shape != (771,) or self.classifier_std.shape != (771,):
            raise ValueError("Expected 771-dimensional classifier normalization")
        if not all(np.isfinite(a).all() for a in (self.signal_mean, self.signal_std, self.classifier_mean, self.classifier_std)):
            raise ValueError("Nonfinite normalization")
        if (self.signal_std <= 0).any() or (self.classifier_std <= 0).any():
            raise ValueError("Standard deviations must be positive")

    @classmethod
    def from_checkpoints(cls, backbone, query, relation, classifier, normalization, device="cpu"):
        _deterministic_inference()
        base = _state(backbone, device)
        encoder = SignalEncoder()
        encoder.load_state_dict({k.removeprefix("indiv_module."): v for k, v in base["model"].items()
                                 if k.startswith("indiv_module.")})
        modules = [M3Query(), RelationConfidence(), GroupClassifier()]
        for module, path in zip(modules, (query, relation, classifier)):
            module.load_state_dict(_state(path, device)["model"])
        with np.load(normalization, allow_pickle=False) as norm:
            mean, std = norm["mean"].copy(), norm["std"].copy()
        return cls(encoder, *modules, base["normalization"]["mean"], base["normalization"]["std"],
                   mean, std, device)

    @torch.no_grad()
    def encode(self, signals):
        signals = np.asarray(signals)
        if signals.ndim != 4 or signals.shape[1:] != (10, 200, 6) or not len(signals):
            raise ValueError("Expected nonempty [batch, 10, 200, 6] sensor signals")
        if not np.isfinite(signals).all():
            raise ValueError("Nonfinite sensor signals")
        windows = signals.reshape(-1, 200, 6)
        cnn, temporal = [], []
        for start in range(0, len(windows), 128):
            x = torch.as_tensor((windows[start:start+128]-self.signal_mean)/self.signal_std,
                                dtype=torch.float32, device=self.device).clamp(-10, 10)
            tokens = self.encoder.features(x)
            cnn.append(tokens.mean(1).cpu().numpy())
            temporal.append(self.encoder.encode(tokens).mean(1).cpu().numpy())
        cnn = np.concatenate(cnn).reshape(len(signals), 10, 128)
        temporal = np.concatenate(temporal).reshape(len(signals), 10, 128)
        labels = []
        with threadpool_limits(limits=4):
            for scene in cnn:
                count = min(4, len(np.unique(scene, axis=0)))
                labels.append(np.zeros(10, dtype=np.int64) if count == 1 else
                              KMeans(count, n_init=1, random_state=2024).fit_predict(scene))
        return np.concatenate([temporal, cnn, np.asarray(labels, dtype=np.float32)[..., None]], -1)

    @torch.no_grad()
    def predict_components(self, components):
        """Run M3 on its 257D sensor components; useful for feature-cache reuse."""
        components = np.asarray(components, dtype=np.float32)
        if components.ndim != 3 or components.shape[1:] != (10, 257) or not len(components):
            raise ValueError("Expected nonempty [batch, 10, 257] sensor components")
        if not np.isfinite(components).all():
            raise ValueError("Nonfinite sensor components")
        collected = [[] for _ in range(5)]
        for start in range(0, len(components), 64):
            outputs = self.query(torch.as_tensor(components[start:start+64], device=self.device))
            values = [v.softmax(-1).cpu().numpy() for v in outputs[:3]]
            values += [outputs[4].sigmoid().cpu().numpy(), outputs[5].cpu().numpy()]
            for dest, value in zip(collected, values):
                dest.append(value)
        members, classes, person, pair, features = [np.concatenate(v) for v in collected]
        active = classes.argmax(-1) != 11
        for i in range(len(active)):
            if not active[i].any():
                active[i, classes[i, :, :11].max(-1).argmax()] = True
        assignment = members*active[:, None, :]
        assignment /= assignment.sum(-1, keepdims=True).clip(1e-7)
        groups = assignment.argmax(-1)
        posterior = assignment @ classes[:, :, :11]
        posterior /= posterior.sum(-1, keepdims=True).clip(1e-7)
        posterior = .5*(posterior+person)
        anchor_affinity = .5*(assignment @ assignment.transpose(0, 2, 1))+.5*pair
        affinities = []
        for start in range(0, len(features), 128):
            inputs = [torch.as_tensor(v[start:start+128], dtype=torch.float32, device=self.device)
                      for v in (features, anchor_affinity, posterior)]
            affinities.append(self.relation(*inputs).sigmoid().cpu().numpy())
        affinity = np.concatenate(affinities)
        x, prior, slots = _group_statistics(features, posterior, groups, affinity)
        x = np.clip((x-self.classifier_mean)/self.classifier_std, -10, 10).astype(np.float32)
        probability = []
        for start in range(0, len(x), 128):
            probability.append(self.classifier(torch.as_tensor(x[start:start+128], device=self.device)).softmax(-1).cpu().numpy())
        probability = .75*prior+.25*np.concatenate(probability)
        activity = np.take_along_axis(probability.argmax(-1), slots, axis=1)
        return {"groups": groups, "activities": np.where(activity == 10, -1, activity), "affinity": affinity}

    def predict(self, signals):
        return self.predict_components(self.encode(signals))
