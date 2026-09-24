# PKU EEG Training-Data Scaling

This project is an isolated copy of the PKU EEG VAD training code for the
training-data scaling experiment. It does not import code from or write results
to `VAD_pkueeg_final`.

## Fixed protocol

- Model: `brain_magic_speech_v7`
- Split seed: `545`
- Subset seed: `1001` is retained only for path compatibility and has no sampling effect.
- Model seed: `1`
- Paper training scales: `5, 10, 20, 50, 80, 100` percent. The scale parser
  also accepts the historical exploratory 40% and 60% settings, but the
  release submit script does not run them.
- Train, validation, and test story pools are fixed. Validation and test data
  remain at 100% for every scale.
- For every subject, each scale keeps the chronological prefix of the training
  recording duration, then creates 12-second windows.
- Train windows use a 6-second stride; validation/test windows remain non-overlapping.
- Duration prefixes are nested and may end inside the final story.
- Each subject selects its checkpoint and threshold independently by validation
  macro accuracy, then uses that fixed pair on the subject's test stories.
- The primary metric is test macro accuracy averaged across 25 subjects.

## Validate the plan

```bash
python train_scaling.py --scale 0.05 --dry-run
python -m unittest tests/test_scaling.py
bash submit_scaling.sh --dry-run
```

## Submit the six paper scales

```bash
bash submit_scaling.sh
```

The command submits one six-element SLURM array. Each element gets its own
result directory and stores `config.yaml`, `scaling_manifest.json`, checkpoints,
per-subject predictions, per-subject metrics, and a completion status.

Submit from this release project directory. The Slurm job uses
`SLURM_SUBMIT_DIR` as its code root and `PYTHON` (default `python3`) as the
interpreter. Activate the training environment first or set `PYTHON` to its
absolute executable path. Raw data and labels are not distributed here;
update `configs/scaling/config.yaml` if the original Huairou paths are not
available.

## Aggregate completed results

```bash
python -m scaling.aggregate_results
```

This writes CSV/JSON summaries, a macro-accuracy scaling curve, and a
per-subject boxplot under
`results/scaling/summary_duration_prefix_train_only_seed1001_model1`.
