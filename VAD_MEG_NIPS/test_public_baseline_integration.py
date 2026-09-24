"""CPU-only smoke tests; no datasets, trainer, optimizer steps, or Slurm jobs."""

import copy
from pathlib import Path
import tempfile
import unittest

import pytorch_lightning as pl
import torch
import yaml

from speech_code.models.my_modules.classification_module import ClassificationModule
from speech_code.models.my_modules.utils import modules_from_config


ROOT = Path(__file__).resolve().parent
MODEL_KEYS = ("pnpl_cnn_tcn",)


class PublicBaselineIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        with (ROOT / "configs/speech/my_run/model_sweep.yaml").open(encoding="utf-8") as f:
            cls.sweep = yaml.safe_load(f)
        with (ROOT / "configs/speech/my_run/config.yaml").open(encoding="utf-8") as f:
            cls.config = yaml.safe_load(f)

    def make_module(self, key):
        return ClassificationModule(
            model_config={key: self.sweep[key]}, n_classes=2,
            optimizer_config=copy.deepcopy(self.config["optimizer"]),
            loss_config=copy.deepcopy(self.config["loss"]),
            margin_weight=list(self.config["general"]["margin_weight"]),
        )

    def test_config_and_factory(self):
        for key in MODEL_KEYS:
            with self.subTest(model=key):
                params = self.sweep[key]
                self.assertEqual(params["input_dim"], 204)
                self.assertEqual(params["n_subjects"], 1)
                model = modules_from_config({key: params})[0]
                self.assertTrue(model.requires_subject_ids)
                self.assertEqual(len(model.spatial_projection.projections), 1)
                self.assertEqual(model.model.conv.out_channels, params["model_dim"])
                self.assertEqual(len(model.model.tcn), params["tcn_layers"])
                self.assertEqual(model.model.conv_dropout.p, params["dropout"])

    def test_native_window_and_postprocessing(self):
        x = torch.randn(1, 204, 1200)
        ids = torch.zeros(1, dtype=torch.long)
        for key in MODEL_KEYS:
            with self.subTest(model=key), torch.inference_mode():
                module = self.make_module(key).eval()
                backbone = module.modules_list[0]
                raw = backbone(x)
                self.assertEqual(raw.shape, (1, 1200))
                torch.testing.assert_close(backbone(x, ids), raw)
                expected = module.sigmoid(module.smoother(raw.unsqueeze(1)).squeeze(1))
                actual = module(x)
                torch.testing.assert_close(actual, expected)
                torch.testing.assert_close(module(x, ids), expected)
                self.assertTrue(torch.isfinite(actual).all())
                self.assertTrue(((actual >= 0) & (actual <= 1)).all())
                labels = torch.randint(0, 2, actual.shape).float()
                torch.testing.assert_close(module.loss_mse(actual, labels), (actual - labels).square().mean())

    def test_projection_gradients(self):
        for key in MODEL_KEYS:
            with self.subTest(model=key):
                module = self.make_module(key).train()
                prediction = module(torch.randn(2, 204, 32))
                labels = torch.randint(0, 2, prediction.shape).float()
                module.loss_mse(prediction, labels).backward()
                projection = module.modules_list[0].spatial_projection
                for parameter in projection.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_checkpoint_reload(self):
        for key in MODEL_KEYS:
            with self.subTest(model=key), tempfile.TemporaryDirectory() as tmp:
                module = self.make_module(key).eval()
                path = Path(tmp) / "smoke.ckpt"
                torch.save({
                    "state_dict": module.state_dict(),
                    "hyper_parameters": dict(module.hparams),
                    "pytorch-lightning_version": pl.__version__,
                }, path)
                restored = ClassificationModule.load_from_checkpoint(path, map_location="cpu").eval()
                x = torch.randn(1, 204, 32)
                with torch.inference_mode():
                    torch.testing.assert_close(restored(x), module(x))

    def test_existing_baseline_unchanged(self):
        module = self.make_module("cnn_lstm").eval()
        x = torch.randn(1, 204, 32)
        with torch.inference_mode():
            raw = module.modules_list[0](x).squeeze(-1)
            expected = module.sigmoid(module.smoother(raw.unsqueeze(1)).squeeze(1))
            torch.testing.assert_close(module(x), expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
