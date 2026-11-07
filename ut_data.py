"""Signal-only templates and partition-local, explicitly synthetic group scenes."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scene_data import integer_labels, make_ut_scenes, overlap_audit


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def window_keys(x):
    keys = set()
    for person in x:
        keys.add(digest(person))
        for half in np.array_split(person, 2):
            if np.max(np.std(half, axis=0)) > 1e-6:
                keys.add(digest(half))
    return keys


def split_rows(x, y):
    """Blocked archive-order split, then purge observable source reuse."""
    parts = {name: [] for name in ("train", "val", "test")}
    for c in np.unique(y):
        rows = np.flatnonzero(y == c)
        a, b = int(len(rows) * .62), int(len(rows) * .8)
        if a < 5 or b-a < 7 or len(rows)-b < 3:
            raise ValueError("Insufficient per-class windows for blocked, purged splitting")
        parts["train"].extend(rows[:a])
        parts["val"].extend(rows[a+2:b-2])
        parts["test"].extend(rows[b:])
    cache = [window_keys(scene) for scene in x]
    used = set()
    dropped = {}
    for name in ("test", "val", "train"):
        original = sorted(parts[name])
        kept = [i for i in original if not (cache[i] & used)]
        dropped[name] = [int(i) for i in sorted(set(original)-set(kept))]
        parts[name] = np.array(kept, dtype=np.int64)
        for i in kept:
            used.update(cache[i])
        if set(y[kept]) != set(y):
            raise ValueError(f"Source purging removed an entire class from {name}")
    audit = {}
    seen = {name: set().union(*(cache[i] for i in rows)) for name, rows in parts.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        audit[a+"__"+b] = len(seen[a] & seen[b])
    if any(audit.values()):
        raise AssertionError("Identical full/half signal windows cross data partitions")
    return parts, dropped, audit


def labeled_rows(rows, y, ratio, seed):
    if not 0 < ratio <= 1:
        raise ValueError("labeled_ratio must be in (0,1]")
    rng = np.random.default_rng(seed)
    selected = []
    for c in np.unique(y):
        bucket = rows[y[rows] == c]
        if not len(bucket):
            raise ValueError("Every training class needs at least one available template")
        selected.extend(rng.permutation(bucket)[:max(1, int(np.ceil(len(bucket)*ratio)))])
    return np.array(sorted(selected), dtype=np.int64)


def prepare(path, ratio, seed, out):
    with np.load(path, allow_pickle=False) as data:
        raw = data["reshaped_segments"]
        y = integer_labels(data["segments_labels1"])
        actions = integer_labels(data["segments_labels2"])
    if raw.shape[1:] != (5, 200, 17) or actions.shape != (len(raw), 5):
        raise ValueError("This adapter expects the audited five-person UT template archive")
    if y.shape != (len(raw),) or (actions < 0).any() or (actions > 9).any():
        raise ValueError("Expected one template label per row and individual actions 0-9")
    x = raw[..., :6].copy()
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite sensor data")
    classes = np.unique(y)
    if len(classes) > 10:
        raise ValueError("The released classifier supports at most ten activity classes")
    if not np.array_equal(classes, np.arange(len(classes))):
        raise ValueError("Expected contiguous zero-based template labels")
    parts, dropped, audit = split_rows(x, y)
    labeled = labeled_rows(parts["train"], y, ratio, seed)
    train_signal = x[parts["train"]].astype(np.float64)
    mean = train_signal.mean(axis=(0, 1, 2)).astype(np.float32)
    std = train_signal.std(axis=(0, 1, 2)).clip(1e-5).astype(np.float32)
    cfg = SimpleNamespace(sensor_columns=tuple(range(6)), persons=10)
    mapping = {int(c): int(c) for c in classes}
    scenes = {name: make_ut_scenes(raw, y, actions, labeled if name == "train" else rows,
                                 cfg, mapping, seed+i+100)
              for i, (name, rows) in enumerate(parts.items())}
    scene_audit = overlap_audit(scenes)
    if any(any(v.values()) for v in scene_audit.values()):
        raise AssertionError("Constructed group scenes reuse held-out signal windows")
    manifest = {
        "source": str(Path(path).resolve()), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "sensor_columns": list(range(6)), "physical_axis_unit_and_location": "unverified",
        "excluded_columns": list(range(6, 17)), "labeled_ratio": ratio,
        "rows": {k: v.tolist() for k, v in parts.items()}, "labeled_rows": labeled.tolist(),
        "purged_rows": dropped, "overlap_counts_full_and_nonconstant_half": audit,
        "constructed_scene_overlap": scene_audit,
        "class_counts": {k: np.bincount(y[v], minlength=len(classes)).tolist() for k, v in parts.items()},
        "labeled_class_counts": np.bincount(y[labeled], minlength=len(classes)).tolist(),
        "mean_train_only": mean.tolist(), "std_train_only": std.tolist(),
        "scene_counts": {k: len(v) for k, v in scenes.items()},
        "scene_construction": "Synthetic 10-person mixtures of 3-4 distinct-class template fragments plus one noise person. Membership is assigned by construction, never by individual action equality.",
        "limitations": ["Archive order is not a verified timestamp or subject ID.",
                       "Zero exact full/half-window overlap does not prove subject or session independence.",
                       "The source has no multi-subgroup membership annotations; mixed-scene results are synthetic.",
                       "Training-only SSL and supervised stages may reuse training windows; this is not the paper's 30% sampling / 80-20 protocol.",
                       "Only this archive is used; other two archives share source signals and are not independent tests."]}
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out)/"data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for split, values in scenes.items():
        np.savez_compressed(Path(out)/f"{split}_scene_provenance.npz",
                            refs=np.stack([s.refs for s in values]),
                            groups=np.stack([s.groups for s in values]),
                            activities=np.stack([s.activities for s in values]))
    return SimpleNamespace(x=x, y=y, actions=actions, parts=parts, labeled=labeled,
                           mean=mean, std=std, scenes=scenes, classes=len(classes), manifest=manifest)
