# MLS-GSAR

**Group Segmentation and Activity Recognition Method Based on a Multi-task
Lightweight Self-supervised Framework**

This repository provides PyTorch model definitions and a Python inference API for
MLS-GSAR. It takes wearable sensor windows as input and returns group assignments,
member activity predictions, and pairwise membership scores.

## Repository contents

| File | Purpose |
| --- | --- |
| `best_m3_model.py` | Signal encoder, local/global attention, query-based group prediction, relation confidence model, and group activity classifier. |
| `best_m3_predict.py` | Checkpoint loading, normalization, feature extraction, group decoding, and end-to-end inference through `BestM3`. |

The model names in these files are Python identifiers. This repository does not
currently contain training scripts, pretrained checkpoints, or an evaluation CLI.
Installing the dependencies alone is therefore not sufficient to reproduce the
paper's experiments or obtain meaningful predictions with trained weights.

## Installation

Use Python 3.11 and run these commands from the repository directory:

```bash
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS
source .venv/bin/activate
```

Install the dependencies used for the local inference check:

```bash
python -m pip install torch==2.1.2 numpy==1.26.3 scikit-learn==1.6.1 threadpoolctl==3.6.0
```

The code was checked locally with Python 3.11.7 and PyTorch 2.1.2+cu121. For CUDA,
install a PyTorch build compatible with your GPU and driver. CPU inference is
supported and is used by default in the example below.

Check the imports and instantiate the architecture without checkpoints:

```bash
python -c "from best_m3_model import SignalEncoder, M3Query, RelationConfidence, GroupClassifier; from best_m3_predict import BestM3; print('Imports OK'); print('Query model parameters:', sum(p.numel() for p in M3Query().parameters()))"
```

This is an architecture check, not trained-model inference or an accuracy test.

## Input and output

`BestM3.predict(signals)` expects a finite numeric NumPy array with shape:

```text
[batch_size, 10, 200, 6]
```

- Each scene contains exactly 10 people, with 200 time steps and 6 sensor channels
  per person. The batch must be nonempty.
- At 50 Hz, 200 time steps represent a 4-second window.
- Channel order, units, window construction, and person grouping must match those
  used to train the supplied checkpoints. Channel meanings cannot be inferred
  from an arbitrary NPZ filename.
- Provide unnormalized signals: the loader obtains signal statistics from the
  backbone checkpoint, and `encode()` applies normalization and clipping.
- Do not include activity labels, group labels, or other target-derived metadata
  in the six signal channels. Do not use held-out data to fit normalization.

The returned dictionary contains:

| Key | Shape | Meaning |
| --- | --- | --- |
| `groups` | `[batch_size, 10]` | Predicted query/group ID for each person. IDs are local to each scene and need not be consecutive. |
| `activities` | `[batch_size, 10]` | Activity ID per person, shared within a predicted group. IDs 0-9 follow the training label mapping; `-1` denotes the noise class, internally encoded as 10. |
| `affinity` | `[batch_size, 10, 10]` | Symmetric pairwise membership scores in [0, 1]. They are scores, not ground-truth relations or guaranteed calibrated probabilities. |

The semantic activity-name mapping is not included in these two source files.
Use the mapping associated with the training data. Do not interpret group IDs as
activity labels. Group assignments are decoded from the query predictions, not
by thresholding the returned affinity matrix.

## Required checkpoints

Obtain the following mutually compatible artifacts before running inference.
The names below are suggested local filenames, not downloadable files bundled
with this repository:

```text
checkpoints/
    backbone.pt
    query.pt
    relation.pt
    classifier.pt
    classifier_normalization.npz
```

| Artifact | Required format |
| --- | --- |
| Backbone | PyTorch dictionary containing `model` with `indiv_module.*` state-dict keys and `normalization` with six-element `mean` and `std` arrays. |
| Query | PyTorch dictionary whose `model` entry is a compatible `M3Query` state dict. |
| Relation | PyTorch dictionary whose `model` entry is a compatible `RelationConfidence` state dict. |
| Classifier | PyTorch dictionary whose `model` entry is a compatible `GroupClassifier` state dict. |
| Classifier normalization | NPZ file with `mean` and `std`, each of shape `(771,)`. |

Standard deviations must be positive. Use statistics fitted on the training
partition and the matching checkpoints. Renaming an unrelated checkpoint does
not make it compatible. No public weight download is specified here.

**Security:** load only trusted checkpoints. The loader uses
`torch.load(..., weights_only=False)`, which can execute code when loading a
malicious serialized file.

## Run inference

Prepare `signals.npy` in the input format above and place the matching artifacts
under `checkpoints/`. Run this Python code from the repository directory:

```python
from pathlib import Path

import numpy as np

from best_m3_predict import BestM3

weights = Path("checkpoints")
model = BestM3.from_checkpoints(
    backbone=weights / "backbone.pt",
    query=weights / "query.pt",
    relation=weights / "relation.pt",
    classifier=weights / "classifier.pt",
    normalization=weights / "classifier_normalization.npz",
    device="cpu",  # Use "cuda" only with an available, compatible CUDA setup.
)

signals = np.load("signals.npy", allow_pickle=False)
predictions = model.predict(signals)
np.savez_compressed("predictions.npz", **predictions)

for name, values in predictions.items():
    print(name, values.shape, values.dtype)
```

There is no command-line parser: `python best_m3_predict.py` does not run training
or inference. Use the Python API above. No automatic loader from the released
dataset archives to ten-person scenes is included in this repository.

For cached features, `model.encode(signals)` returns `[batch_size, 10, 257]`
components: 128 temporal features, 128 CNN features, and one signal-derived
K-means cluster ID (0-3). Pass these matching components to
`model.predict_components(components)`. The cluster ID is not an annotation.

## Implementation overview

1. `SignalEncoder` extracts multi-scale features with channel attention, pools
   them into temporal tokens, and applies a Transformer encoder.
2. Signal-derived K-means clusters provide four attention partitions.
   `LocalGlobalFeatures` combines intra-partition and inter-partition attention.
3. `M3Query` uses contextual features and six learned queries to estimate
   memberships and activities. One query-class output denotes an inactive query.
4. `RelationConfidence` refines pairwise membership scores. `GroupClassifier`
   aggregates mean, maximum, standard deviation, group size, cohesion, and
   external affinity into 771-dimensional group features.
5. The inference wrapper combines group classifier probabilities with the
   activity prior using the fixed weights in the source code.

The wrapper enables deterministic PyTorch settings and a fixed K-means seed.
These settings do not guarantee bitwise identity across all hardware, software
versions, devices, or batching arrangements.

## Data and paper

- Author-provided dataset page: <https://www.kaggle.com/datasets/codeusea/mls-gsar>
- Paper title: *MLS-GSAR: Group Segmentation and Activity Recognition Method
  Based on a Multi-task Lightweight Self-supervised Framework*.

Consult the dataset documentation and original-source permissions before using
or redistributing data. A dataset license does not automatically license the
code. No code license is declared by this README; consult a repository LICENSE
file if one is added. No publication DOI is asserted here.
