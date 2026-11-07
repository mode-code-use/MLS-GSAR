# Training and testing

This is a standalone, public retraining workflow for the existing released
architecture. The original two model/inference files are unchanged. It exports
artifacts compatible with `BestM3.from_checkpoints` and does not require the
authors' local experiment folders or existing checkpoints.

## Commands

Install Python 3.11 and dependencies:

```bash
python -m pip install -r requirements.txt
```

Prepare UT-Group using the command in `README.md`, or supply compatible pre-split
scene archives. Train without reading test data:

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
