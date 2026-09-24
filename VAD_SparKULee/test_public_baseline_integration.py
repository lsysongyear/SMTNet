"""CPU-only native integration tests, without data loading or Trainer.fit."""
import inspect
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import yaml
from speech_code.models.my_modules.classification_module import ClassificationModule
from speech_code.models.my_modules.utils import modules_from_config

torch.set_num_threads(1)
SMALL = {
    'pnpl_cnn_tcn': dict(input_dim=4, n_subjects=3, model_dim=8, tcn_layers=4, dropout=0.1),
}


def classifier(name, params):
    kwargs = dict(model_config={name: params}, n_classes=2,
                  optimizer_config={'name': 'adamw', 'config': {'lr': 0.001}},
                  loss_config={'config': {'weight': [1.0, 1.0]}},
                  margin_weight=[0.1, 0.1])
    if 'sfreq' in inspect.signature(ClassificationModule).parameters:
        kwargs['sfreq'] = 100
    return ClassificationModule(**kwargs)


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.x = torch.randn(3, 4, 24)
        self.ids = torch.tensor([1, 0, 1])

    def test_actual_sweep_models(self):
        sweep = yaml.safe_load((ROOT / 'configs/speech/my_run/model_sweep.yaml').read_text())
        for name in SMALL:
            params = sweep[name]
            model = modules_from_config({name: params})[0].eval()
            with torch.no_grad():
                output = model(torch.randn(2, params['input_dim'], 24),
                               torch.tensor([0, params['n_subjects'] - 1]))
            self.assertEqual(tuple(output.shape), (2, 24))
            self.assertTrue(torch.isfinite(output).all())
            self.assertEqual(len(model.spatial_projection.projections), params['n_subjects'])

    def test_native_subject_metadata_and_postprocessing(self):
        cases = dict(SMALL)
        cases['cnn_lstm'] = dict(input_channels=4, n_subjects=3, num_kernels=2,
                                 kernel_time=3, lstm_hidden=2, output_dim=1)
        for name, params in cases.items():
            module = classifier(name, params).eval()
            ids = module._extract_ids((self.x, torch.zeros(3, 24), {'subject_id': self.ids}))
            if isinstance(ids, tuple):
                ids = ids[0]
            torch.testing.assert_close(ids, self.ids)
            logits = module.modules_list[0](self.x, subject_ids=ids)
            if logits.ndim == 3:
                logits = logits.squeeze(-1)
            expected = module.smoother(logits.unsqueeze(1))
            if hasattr(module, 'scale_factor'):
                expected = F.interpolate(expected, size=int(24 * module.scale_factor),
                                         mode='linear', align_corners=False)
            expected = expected.squeeze(1).sigmoid()
            actual = module(self.x, subject_ids=ids)
            torch.testing.assert_close(actual, expected)
            self.assertTrue(torch.isfinite(actual).all())
            self.assertTrue(((actual >= 0) & (actual <= 1)).all())

    def test_subject_gradient_isolation(self):
        for name, params in SMALL.items():
            model = classifier(name, params).eval()
            output = model(self.x, self.ids)
            model.loss_mse(output, torch.zeros_like(output)).backward()
            projections = model.modules_list[0].spatial_projection.projections
            for index in [0, 1]:
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                    for p in projections[index].parameters()))
            self.assertTrue(all(p.grad is None or p.grad.count_nonzero() == 0
                                for p in projections[2].parameters()))

    def test_checkpoint_reload(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            for name, params in SMALL.items():
                model = classifier(name, params).eval()
                path = Path(directory) / (name + '.ckpt')
                torch.save({'state_dict': model.state_dict(), 'hyper_parameters': dict(model.hparams),
                            'pytorch-lightning_version': pl.__version__}, path)
                restored = ClassificationModule.load_from_checkpoint(path, map_location='cpu').eval()
                torch.testing.assert_close(model(self.x, self.ids), restored(self.x, self.ids))


if __name__ == '__main__':
    unittest.main(verbosity=2)
