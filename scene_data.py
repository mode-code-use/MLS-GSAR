"""Partition-local UT scene construction and observable overlap checks."""
from dataclasses import dataclass
import hashlib
import numpy as np


@dataclass
class Scene:
    x: np.ndarray                 # [people, time, six sensor channels]
    groups: np.ndarray            # distinct IDs, including a singleton for noise
    activities: np.ndarray        # per-person group activity; -1 means noise
    actions: np.ndarray           # individual activity labels, used only by T1
    refs: np.ndarray


def integer_labels(a):
    a = np.asarray(a)
    if not np.isfinite(a).all() or not np.equal(a, np.rint(a)).all():
        raise ValueError("Labels must be finite integers")
    return a.astype(np.int64)


def _sensors(x, cfg):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3 or x.shape[1] != 200 or max(cfg.sensor_columns) >= x.shape[2]:
        raise ValueError(f"Expected [people,200,channels], got {x.shape}")
    x = x[:, :, list(cfg.sensor_columns)].copy()
    if not np.isfinite(x).all():
        raise ValueError("Non-finite sensor readings; refusing silent imputation")
    return x


def make_ut_scenes(xs, labels, actions, rows, cfg, mapping, seed):
    """Construct scenes from partition-local templates, with reproducible provenance."""
    rng = np.random.default_rng(seed)
    by_class = {c: [int(i) for i in rows if int(labels[i]) == c] for c in mapping}
    available = [c for c, ids in by_class.items() if ids]
    if len(available) < 4:
        raise ValueError("UT construction requires at least four group classes per partition")
    output = []
    for _ in rows:
        k = int(rng.integers(3, 5))
        classes = rng.choice(available, k, replace=False)
        source_rows = [int(rng.choice(by_class[c])) for c in classes]
        # Keep every action role present in the source subgroup when subsampling.
        sizes = [max(2, len(np.unique(actions[i]))) for i in source_rows]
        if sum(sizes) > cfg.persons - 1:
            # Four heterogeneous subgroups may not fit; use three instead.
            classes, source_rows, sizes = classes[:3], source_rows[:3], sizes[:3]
        if sum(sizes) > 9:
            raise ValueError("Source subgroup roles do not fit in a 10-person scene")
        while sum(sizes) < 9:
            choices = [j for j, size in enumerate(sizes) if size < 5]
            sizes[int(rng.choice(choices))] += 1
        signals, gids, activities, indiv, refs = [], [], [], [], []
        for g, (c, row, size) in enumerate(zip(classes, source_rows, sizes)):
            y = integer_labels(actions[row])
            selected = [int(rng.choice(np.flatnonzero(y == a))) for a in np.unique(y)]
            remaining = [p for p in range(5) if p not in selected]
            selected += rng.choice(remaining, size - len(selected), replace=False).tolist()
            signals.extend(_sensors(xs[row], cfg)[selected])
            gids.extend([g] * size)
            activities.extend([mapping[int(c)]] * size)
            indiv.extend(y[selected])
            refs.extend((row, p) for p in selected)
        row = int(rng.choice(rows))
        candidates = [p for p in range(5) if (row, p) not in refs]
        if not candidates:
            row = int(rng.choice([i for i in rows if i not in source_rows]))
            candidates = list(range(5))
        p = int(rng.choice(candidates))
        signals.append(_sensors(xs[row], cfg)[p])
        gids.append(len(classes))
        activities.append(-1)
        indiv.append(int(actions[row][p]))
        refs.append((row, p))
        order = rng.permutation(cfg.persons)
        output.append(Scene(np.array(signals)[order], np.array(gids)[order],
                            np.array(activities)[order], np.array(indiv)[order],
                            np.array(refs)[order]))
    return output


def overlap_audit(partitions):
    """Detect exact repeated person windows and aligned half-window reuse."""
    seen = {}
    for split, scenes in partitions.items():
        whole, halves = set(), set()
        for scene in scenes:
            for x in scene.x:
                whole.add(hashlib.sha256(x.tobytes()).hexdigest())
                for half in np.split(x, 2):
                    # Constant half-windows are not informative evidence of reused sources.
                    if np.max(np.std(half, axis=0)) > 1e-6:
                        halves.add(hashlib.sha256(half.tobytes()).hexdigest())
        seen[split] = (whole, halves)
    report = {}
    keys = list(seen)
    for i, left in enumerate(keys):
        for right in keys[i+1:]:
            report[f"{left}__{right}"] = {
                "identical_person_windows": len(seen[left][0] & seen[right][0]),
                "identical_nonconstant_half_windows": len(seen[left][1] & seen[right][1]),
            }
    return report
