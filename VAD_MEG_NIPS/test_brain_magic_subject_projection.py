"""CPU-only checks for subject projection without changing the original model."""

import inspect
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("WANDB_MODE", "disabled")

import pytorch_lightning as pl
import torch
from torch import nn

from speech_code.models.my_modules.BrainNetwork import Brain_Magic_speech
from speech_code.models.my_modules.BrainNetworkSubject import SubjectProjectedBrainMagic
from speech_code.models.my_modules.classification_module import ClassificationModule
from speech_code.models.my_modules.utils import modules_from_config


torch.set_num_threads(1)
CORE = dict(input_dim=4, attention_dim=8, output_dim=1, kernel_size=5,
            depthwise_kernel=15, dropout=0.0)


def classifier(parameters):
    kwargs = dict(
        model_config={"brain_magic_speech": parameters}, n_classes=2,
        optimizer_config={"name": "adamw", "config": {"lr": 0.001}},
        loss_config={"config": {"weight": [1.0, 1.0]}},
        margin_weight=[0.1, 0.1],
    )
    if "sfreq" in inspect.signature(ClassificationModule).parameters:
        kwargs["sfreq"] = 100
    return ClassificationModule(**kwargs)


class SubjectProjectedBrainMagicTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(87)
        self.x = torch.randn(3, 4, 32)
        self.ids = torch.tensor([1, 0, 1])

    def test_defaults_match_original(self):
        original = inspect.signature(Brain_Magic_speech).parameters
        adapted = inspect.signature(SubjectProjectedBrainMagic).parameters
        for name in CORE:
            self.assertEqual(adapted[name].default, original[name].default)
        self.assertEqual(adapted["n_subjects"].default, 1)

    def test_factory_preserves_legacy_selection(self):
        legacy = modules_from_config({"brain_magic_speech": dict(CORE)})[0]
        adapted = modules_from_config(
            {"brain_magic_speech": dict(CORE, n_subjects=3)})[0]
        self.assertIs(type(legacy), Brain_Magic_speech)
        self.assertIs(type(adapted), SubjectProjectedBrainMagic)
        self.assertTrue(adapted.requires_subject_ids)
        self.assertEqual(len(adapted.spatial_projection.projections), 3)

    def test_unchanged_core_structure_and_exact_output(self):
        adapted = SubjectProjectedBrainMagic(**CORE, n_subjects=3).eval()
        original = Brain_Magic_speech(**CORE).eval()
        original.load_state_dict(adapted.model.state_dict(), strict=True)
        self.assertEqual(list(original.state_dict()), list(adapted.model.state_dict()))
        self.assertEqual(len(adapted.model.feature_encoder), 5)
        self.assertEqual(len(adapted.model.short_conv_block.layers), 12)
        self.assertEqual(adapted.model.lstm.num_layers, 1)
        self.assertTrue(adapted.model.lstm.bidirectional)
        shared_mlp = [m for m in adapted.model.spatial_attention if isinstance(m, nn.Linear)]
        self.assertEqual([(m.in_features, m.out_features) for m in shared_mlp],
                         [(4, 4), (4, 16), (16, 8)])
        with torch.no_grad():
            projected = adapted.spatial_projection(self.x, self.ids)
            expected = original(projected)
            actual = adapted(self.x, self.ids)
        self.assertEqual(tuple(actual.shape), (3, 32))
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_subject_order_routing_and_required_ids(self):
        model = SubjectProjectedBrainMagic(**CORE, n_subjects=3).eval()
        order = torch.tensor([2, 0, 1])
        with torch.no_grad():
            actual = model(self.x, self.ids)
            shuffled = model(self.x[order], self.ids[order])
            individual = torch.cat([
                model(self.x[i:i + 1], self.ids[i:i + 1]) for i in range(3)
            ])
        torch.testing.assert_close(shuffled, actual[order], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(individual, actual, rtol=1e-5, atol=1e-6)
        with self.assertRaises(ValueError):
            model(self.x)
        with self.assertRaises(ValueError):
            model(self.x, torch.tensor([0, 3, 0]))
        single = SubjectProjectedBrainMagic(**CORE, n_subjects=1).eval()
        with torch.no_grad():
            torch.testing.assert_close(
                single(self.x), single(self.x, torch.zeros(3, dtype=torch.long)),
                rtol=0, atol=0,
            )

    def test_projection_gradient_isolation(self):
        model = SubjectProjectedBrainMagic(**CORE, n_subjects=3).eval()
        output = model(self.x, self.ids)
        output.square().mean().backward()
        for index in (0, 1):
            parameters = model.spatial_projection.projections[index].parameters()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                for p in parameters))
        self.assertTrue(all(p.grad is None
                            for p in model.spatial_projection.projections[2].parameters()))
        self.assertIsNotNone(model.model.final_conv2.weight.grad)
        self.assertTrue(torch.isfinite(model.model.final_conv2.weight.grad).all())

    def test_output_is_raw_logits(self):
        model = SubjectProjectedBrainMagic(**CORE, n_subjects=3).eval()
        with torch.no_grad():
            model.model.final_conv2.weight.zero_()
            model.model.final_conv2.bias.fill_(-2.0)
            actual = model(self.x, self.ids)
        torch.testing.assert_close(actual, torch.full((3, 32), -2.0), rtol=0, atol=0)

    def test_legacy_and_projected_lightning_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            for n_subjects in (None, 3):
                with self.subTest(n_subjects=n_subjects):
                    parameters = dict(CORE)
                    if n_subjects is not None:
                        parameters["n_subjects"] = n_subjects
                    module = classifier(parameters).eval()
                    path = Path(directory) / ("legacy.ckpt" if n_subjects is None else "projected.ckpt")
                    torch.save({
                        "state_dict": module.state_dict(),
                        "hyper_parameters": dict(module.hparams),
                        "pytorch-lightning_version": pl.__version__,
                    }, path)
                    loaded = ClassificationModule.load_from_checkpoint(path, map_location="cpu").eval()
                    self.assertEqual(list(module.state_dict()), list(loaded.state_dict()))
                    if n_subjects is None:
                        self.assertIs(type(loaded.modules_list[0]), Brain_Magic_speech)
                    else:
                        self.assertIs(type(loaded.modules_list[0]), SubjectProjectedBrainMagic)
                    ids = self.ids if n_subjects is not None else None
                    with torch.no_grad():
                        expected = module(self.x, subject_ids=ids)
                        actual = loaded(self.x, subject_ids=ids)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertTrue(((actual >= 0) & (actual <= 1)).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
