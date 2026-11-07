from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from dataset import load_scenes, audit_splits
from best_m3_model import M3Query
from training_loss import batched_joint_loss
from workflow import metrics
from make_smoke_data import make_split
from ut_data import prepare


class WorkflowTests(unittest.TestCase):
    def test_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'data.npz'
            data = make_split(2, 'a', 20)
            np.savez_compressed(p, **data)
            loaded = load_scenes(p)
            self.assertEqual(loaded['signals'].shape, (2, 10, 200, 6))
            data['activities'] = data['activities'].astype(float)
            np.savez_compressed(p, **data)
            with self.assertRaises(ValueError):
                load_scenes(p)

    def test_leakage_checks(self):
        a, b = make_split(2, 'a', 20), make_split(2, 'b', 21)
        audit_splits({'train': a, 'val': b})
        b['source_ids'][0, 0] = a['source_ids'][0, 0]
        with self.assertRaises(ValueError):
            audit_splits({'train': a, 'val': b})
        b = make_split(2, 'b', 21)
        b['signals'][0, 0] = a['signals'][0, 0]
        with self.assertRaises(ValueError):
            audit_splits({'train': a, 'val': b})

    def test_loss_backward(self):
        torch.set_num_threads(2)
        model = M3Query()
        x = torch.randn(2, 10, 257)
        x[..., 256] = torch.arange(10) % 4
        data = make_split(2, 'a', 20)
        outputs = model(x)
        loss = batched_joint_loss(outputs[:4], *(torch.as_tensor(data[k]) for k in ('groups', 'activities', 'actions')))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(model.queries.grad).all())

    def test_ut_adapter(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'raw.npz'
            rng = np.random.default_rng(40)
            np.savez_compressed(path, reshaped_segments=rng.normal(size=(200, 5, 200, 17)).astype(np.float32),
                                segments_labels1=np.repeat(np.arange(4), 50),
                                segments_labels2=np.tile(np.arange(5) % 2, (200, 1)))
            result = prepare(path, 1., 2024, Path(folder) / 'prepared')
            self.assertEqual(set(result.scenes), {'train', 'val', 'test'})
            self.assertTrue(all(len(v) for v in result.scenes.values()))
            self.assertEqual(result.scenes['train'][0].x.shape, (10, 200, 6))
            self.assertEqual(set(result.manifest['sensor_columns']), set(range(6)))

    def test_metric_definitions(self):
        data = make_split(2, 'a', 20)
        same = data['groups'][:, :, None] == data['groups'][:, None, :]
        prediction = {'groups': data['groups'], 'activities': data['activities'], 'affinity': same.astype(float)}
        result = metrics(data, prediction)
        self.assertEqual(result['membership_AP_global_off_diagonal'], 1.)
        self.assertEqual(result['recognition_accuracy'], 1.)
        self.assertEqual(result['segmentation_pair_f1'], 1.)
        # Only classes 0,1,2 occur: macro scores use the documented fixed 10 classes.
        self.assertAlmostEqual(result['recognition_macro_f1'], .3)


if __name__ == '__main__':
    unittest.main()
