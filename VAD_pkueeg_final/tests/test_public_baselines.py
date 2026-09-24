"""Small CPU-only integration checks; no dataset loading or training jobs."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('WANDB_MODE', 'disabled')

import torch
import torch.nn.functional as F
import yaml
import pytorch_lightning as pl

from speech_code.models.my_modules.classification_module import ClassificationModule
from speech_code.models.my_modules.utils import modules_from_config
from speech_code.models.my_modules.CNNTCN import CNNTCN, ExactLargeDilationConv1d

torch.set_num_threads(1)
SMALL = {
    'pnpl_cnn_tcn': dict(input_dim=4, n_subjects=3, model_dim=8, tcn_layers=4, dropout=0.2),
}


def classifier(name, parameters, sfreq=250):
    return ClassificationModule(
        model_config={name: parameters}, n_classes=2,
        optimizer_config={'name': 'adamw', 'config': {'lr': 0.001}},
        loss_config={'config': {'weight': [1.0, 1.0]}},
        margin_weight=[0.1, 0.1], sfreq=sfreq,
    )


class PublicBaselineTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        self.x = torch.randn(3, 4, 24)
        self.ids = torch.tensor([1, 0, 1])

    def test_factory_defaults(self):
        sweep = yaml.safe_load((ROOT / 'configs/speech/my_run/model_sweep.yaml').read_text())
        for name in SMALL:
            model = modules_from_config({name: sweep[name]})[0].eval()
            with torch.no_grad():
                output = model(torch.randn(2, 57, 16), torch.tensor([0, 24]))
            self.assertEqual(tuple(output.shape), (2, 16))
            self.assertTrue(torch.isfinite(output).all())
            self.assertEqual(len(model.spatial_projection.projections), 25)

    def test_subject_routing_and_gradients(self):
        for name, parameters in SMALL.items():
            model = modules_from_config({name: parameters})[0].eval()
            y = model(self.x, self.ids)
            expected = torch.cat([model(self.x[i:i+1], self.ids[i:i+1]) for i in range(3)])
            torch.testing.assert_close(y, expected, rtol=1e-5, atol=1e-6)
            y.square().mean().backward()
            for index in [0, 1]:
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                    for p in model.spatial_projection.projections[index].parameters()))
            self.assertTrue(all(p.grad is None or p.grad.count_nonzero() == 0
                                for p in model.spatial_projection.projections[2].parameters()))
            with self.assertRaises(ValueError):
                model(self.x)
            with self.assertRaises(ValueError):
                model(self.x, torch.tensor([0, 3, 0]))

    def test_shared_postprocessing_and_subject_metadata(self):
        cases = dict(SMALL)
        cases['cnn_lstm'] = dict(input_channels=4, n_subjects=3, num_kernels=2,
                                 kernel_time=3, lstm_hidden=2, output_dim=1)
        for name, parameters in cases.items():
            module = classifier(name, parameters, sfreq=100).eval()
            subject, _ = module._extract_ids((self.x, torch.zeros(3, 60),
                                              {'subject': ['02', '01', '02']}))
            torch.testing.assert_close(subject, self.ids)
            logits = module.modules_list[0](self.x, subject_ids=self.ids)
            if logits.ndim == 3:
                logits = logits.squeeze(-1)
            expected = F.interpolate(module.smoother(logits.unsqueeze(1)), size=60,
                                     mode='linear', align_corners=False).squeeze(1).sigmoid()
            actual = module(self.x, subject_ids=subject)
            torch.testing.assert_close(actual, expected)
            self.assertEqual(tuple(actual.shape), (3, 60))
            self.assertTrue(((actual >= 0) & (actual <= 1)).all())

    def test_lightning_checkpoint_reload(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            for name, parameters in SMALL.items():
                module = classifier(name, parameters).eval()
                path = Path(tmp) / (name + '.ckpt')
                torch.save({'state_dict': module.state_dict(),
                            'hyper_parameters': dict(module.hparams),
                            'pytorch-lightning_version': pl.__version__}, path)
                loaded = ClassificationModule.load_from_checkpoint(path, map_location='cpu').eval()
                torch.testing.assert_close(module(self.x, self.ids), loaded(self.x, self.ids))

    def test_large_dilation_forward_and_backward(self):
        a = ExactLargeDilationConv1d(2, 3, 3, padding=8, dilation=8)
        b = torch.nn.Conv1d(2, 3, 3, padding=8, dilation=8)
        b.load_state_dict(a.state_dict())
        x = torch.randn(2, 2, 4, requires_grad=True)
        z = x.detach().clone().requires_grad_()
        torch.testing.assert_close(a(x), b(z))
        a(x).square().sum().backward()
        b(z).square().sum().backward()
        torch.testing.assert_close(x.grad, z.grad)
        torch.testing.assert_close(a.weight.grad, b.weight.grad)

    def test_parameter_validation(self):
        with self.assertRaises(ValueError):
            CNNTCN(model_dim=0)
        with self.assertRaises(ValueError):
            CNNTCN(tcn_layers=0)
        with self.assertRaises(ValueError):
            CNNTCN(dropout=1.0)

    def test_submit_script_with_mock_sbatch(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            tmp = Path(tmp)
            shutil.copytree(ROOT / 'configs', tmp / 'configs')
            shutil.copy2(ROOT / 'submit_pretrain_per_sub_5t5v.sh', tmp)
            capture = tmp / 'calls.txt'
            env = dict(os.environ, SBATCH_CAPTURE=str(capture))
            command = '''sbatch() { printf '%s\\n' "$*" >> "$SBATCH_CAPTURE"; printf 'DRYRUN_ONLY\\n'; }
export -f sbatch
bash submit_pretrain_per_sub_5t5v.sh --models pnpl_cnn_tcn --seed 545'''
            process = subprocess.run(['bash', '-c', command], cwd=tmp, env=env,
                                     text=True, capture_output=True, timeout=60)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            calls = capture.read_text().splitlines()
            self.assertEqual(len(calls), 1)
            for call in calls:
                self.assertIn('--array=0-0', call)
                self.assertIn('SPLIT_SEED=545', call)
                self.assertIn('job_pretrain_per_sub_5t5v.slurm', call)
            snapshots = list((tmp / 'log/per_sub_5t5v/pretrain').glob('snapshot_*'))
            self.assertEqual(len(snapshots), 1)
            for folder in snapshots:
                config = yaml.safe_load((folder / 'config.yaml').read_text())
                name = config['general']['model_name']
                self.assertIn(name, SMALL)
                space = yaml.safe_load((folder / 'search-space.yaml').read_text())
                self.assertTrue(all(len(v) == 1 for v in space.values()))
                import ast
                pairs = [(ast.literal_eval(k), v[0]) for k, v in space.items()]
                segment = re.sub(r'seed-\d+_', '', '_'.join(f'{k[-1]}-{v}' for k, v in pairs))
                self.assertLessEqual(len(segment.encode()), 255, segment)

    @unittest.skipUnless(os.environ.get('PUBLIC_BASELINE_REFERENCE_ROOT'), 'optional pinned search reference')
    def test_pinned_search_numerical_equivalence(self):
        sys.path.insert(0, os.environ['PUBLIC_BASELINE_REFERENCE_ROOT'])
        from spatial_models import SubjectProjectedCNNTCN
        pairs = [(CNNTCN, SubjectProjectedCNNTCN, SMALL['pnpl_cnn_tcn'])]
        for current, old, parameters in pairs:
            torch.manual_seed(93)
            reference = old(**parameters)
            torch.manual_seed(93)
            integrated = current(**parameters)
            self.assertEqual(list(reference.state_dict()), list(integrated.state_dict()))
            for key, value in reference.state_dict().items():
                torch.testing.assert_close(value, integrated.state_dict()[key], rtol=0, atol=0)
            for training in [False, True]:
                reference.train(training).zero_grad(set_to_none=True)
                integrated.train(training).zero_grad(set_to_none=True)
                torch.manual_seed(204)
                expected = reference(self.x, self.ids)
                torch.manual_seed(204)
                actual = integrated(self.x, self.ids)
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                expected.square().sum().backward()
                actual.square().sum().backward()
                for key, parameter in reference.named_parameters():
                    other = dict(integrated.named_parameters())[key]
                    if parameter.grad is not None:
                        torch.testing.assert_close(parameter.grad, other.grad, rtol=1e-5, atol=1e-6)


if __name__ == '__main__':
    unittest.main(verbosity=2)
