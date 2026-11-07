# Training and testing

This is a standalone, public retraining workflow for the existing released
architecture. The original two model/inference files are unchanged. It exports
artifacts compatible with `BestM3.from_checkpoints` and does not require the
authors' local experiment folders or existing checkpoints.

**Scope:** the training schedule below is newly packaged from the existing
operators. It does not recreate the full historical experiment search,
supervised-backbone selection, augmentation budget or paper's low-label
protocol. In particular, successful execution or training to convergence is not
a guarantee of the paper's table values, 70% accuracy, or a particular ablation
ordering. Use the recorded configuration when reporting a new experiment.

## Commands

Install Python 3.11 and dependencies:

```bash
python -m pip install -r requirements.txt
```

Prepare the audited UT archive using the command in `DATA_FORMAT.md`, or supply
compatible pre-split scene archives. Train without reading test data:

```bash
python train.py --train data/ut/train.npz --val data/ut/val.npz --output runs/m3 --device cpu
```

Use `--device cuda` for a compatible CUDA environment. After training, run a
held-out evaluation separately:

```bash
python evaluate.py --data data/ut/test.npz --checkpoints runs/m3 --output runs/m3_test --device cpu
```

Both commands refuse existing output directories. Evaluation writes
`metrics.json` and `predictions.npz`. It rejects sources and exact full/half
windows found in training or validation. Do not use test results to tune
hyperparameters, select epochs or repeatedly select better runs.

## Training stages

| Stage | Default epochs | Objective and selection |
| --- | --- | --- |
| Individual representation pretraining | 20 | Contrastive loss (temperature 0.2), next-20-step MSE from the first 180 steps, and four-way speed classification. Last epoch, train data only. |
| Supervised backbone warmup | 10 | Per-person group-activity CE + 0.1 individual-action CE, train data only. Last epoch. This is an explicit additional training assumption, not an unchanged paper protocol. |
| M3 query model | 30 | Existing Hungarian-matched group/class/member objective, auxiliary behavior/action and relation losses, plus 0.25 balanced direct relation BCE. Selected by the mean of validation membership AP and behavior accuracy. |
| Relation confidence | 10 | Balanced off-diagonal membership BCE. Selected by validation membership AP. |
| Group classifier | 20 | Member-count-weighted group targets with 0.02 label smoothing. Selected by validation behavior accuracy with the inference mixture fixed at 0.75 prior + 0.25 classifier. |

The encoder is frozen after backbone warmup. Relation and classifier features
use predicted groups, not ground-truth groups. Classifier normalization uses
valid training group features only. Self-supervision uses only training signals.
All stages use AdamW (learning rate 0.001, weight decay 0.001), cosine scheduling
(minimum learning rate 0.00001), gradient clipping at norm 5, and seed 2024 by
default. `--help` lists adjustable budgets, batch size, learning rate and device.
The default batch size is 32; CPU training may be slow on large collections.

The speed rates are 0.5, 1, 1.5 and 2 with reflected boundaries. Augmentation
combines jitter, scaling, magnitude warping and time warping. These and the
forecast horizon are explicit implementation choices where the paper description
alone does not uniquely specify executable code.

## Outputs and provenance

`runs/m3/` contains the five files required for inference:

```text
backbone.pt
query.pt
relation.pt
classifier.pt
classifier_normalization.npz
```

It also contains training-only pretraining checkpoints, `config.json`,
`history.json`, `validation_metrics.json`, and `selection_provenance.json`.
Configuration records source code hashes, data file hashes, budgets, versions,
and the split overlap audit. Provenance is needed by `evaluate.py`; do not publish
it without reviewing whether your source IDs contain participant identifiers.
Do not upload checkpoints from untrusted sources: checkpoint loading uses Python
pickle internally.

## Metric definitions

All reported values are fractions in [0,1], not percentages:

- `membership_AP_global_off_diagonal`: pooled average precision over all directed
  off-diagonal person pairs in all scenes, computed with scikit-learn. It is not
  group-set IoU mAP and is not a per-scene AP average.
- `segmentation_pair_*`: pairwise precision, recall and F1 from the predicted
  partition. These are not group-set detection F1 scores.
- `recognition_*`: accuracy and fixed-ten-class macro precision/recall/F1 over
  people whose ground-truth behavior is not noise; a predicted noise label is an
  error for a non-noise target. Absent classes contribute zero to fixed-class
  macro scores. These definitions must be stated when comparing datasets.

This evaluation command does not output group-set IoU-AP or IOU-AUC and must not
be used to label pairwise scores as those metrics. No metrics are fabricated to
match a target paper table.

## Tests and smoke run

```bash
python -m unittest discover -s tests -p "test_*.py"
python tests/make_smoke_data.py --output data/smoke
python train.py --train data/smoke/train.npz --val data/smoke/val.npz --output runs/smoke --smoke
python evaluate.py --data data/smoke/test.npz --checkpoints runs/smoke --output runs/smoke_test
```

Smoke data are random generated fixtures, not research data. `--smoke` uses one
epoch per stage and at most eight scenes per split. Its metrics are only a check
that gradients, export, loading and evaluation execute; they are not experimental
results. For a real run, omit `--smoke` and use real, legally accessible data with
verified labels, provenance and an appropriate split protocol.
