import unittest

from speech_code.train_v1 import _pretrain_result_base


class PretrainResultPathTest(unittest.TestCase):
    def test_does_not_truncate_run_named_parent(self):
        checkpoint_dir = (
            "/project/results/run_20260906/candidate_00/model/config/"
            "pretrain/run0:seed1_fold0/checkpoints"
        )
        self.assertEqual(
            _pretrain_result_base(checkpoint_dir),
            "/project/results/run_20260906/candidate_00/model/config",
        )


if __name__ == "__main__":
    unittest.main()
