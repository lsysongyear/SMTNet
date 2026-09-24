# VAD SEM4Lang Scaling

Independent training-data scaling experiments for `brain_magic_speech_v7` on
SEM4Lang MEG. The source project `VAD_SEM4Lang` and its results are not modified.

## Protocol

- Subjects: all 12 SEM4Lang subjects.
- Split strategy: `per_sub_5t5v` with `split_seed=5`.
- Fixed validation stories: `9, 10, 26, 32, 56`.
- Fixed test stories: `36, 38, 44, 50, 55`.
- Full training pool: the remaining 50 stories.
- Paper scales: `5%, 10%, 20%, 50%, 80%, 100%`. The parser also accepts
  historical exploratory 40% and 60% settings, but the release submit script
  does not run them.
- Scaling unit: each subject's chronological training recording duration.
  Cropping happens before 12-second windows are generated and may end inside the
  final story. Validation and test sets remain complete at every scale.
- Subset seed `1001` is retained only for path compatibility and has no sampling effect.
- Model seed: `1`.
- Model: only the complete `brain_magic_speech_v7` with `8/19` blocks.
- Checkpointing: top five checkpoints by validation loss.
- Final selection: independently for each subject, select `(checkpoint, threshold)`
  by validation macro accuracy and apply both unchanged to the fixed test set.
- Primary metric: mean per-subject test macro accuracy.

The data and label paths point to the original `VAD_SEM4Lang` data directories;
all logs, checkpoints, manifests, and summaries are written inside this release
project. Update `configs/scaling/config.yaml` if those paths are unavailable.

## Validate

```bash
python3 -m unittest discover -s tests -v

bash submit_scaling.sh --dry-run

python3 train_scaling.py --scale 1.0 --split-seed 5 --subset-seed 1001 \
  --model-seed 1 --dry-run
```

## Submit

```bash
bash submit_scaling.sh
```

The six-task SLURM array writes independent runs below:

```text
results/scaling/brain_magic_speech_v7/protocol_duration_prefix_train_only/split_seed_5/
  subset_seed_1001/model_seed_1/scale_XXX/run_slurm_JOBID/
```

Submit from this release project directory. The Slurm job uses
`SLURM_SUBMIT_DIR` as its code root and `PYTHON` (default `python3`) as the
interpreter. Activate the training environment first or set `PYTHON` to its
absolute executable path. Raw data and labels are not distributed here.

Completed scales are skipped automatically. Use `--rerun-completed` only when an
intentional repeat is required.

## Aggregate

```bash
python3 scaling/aggregate_results.py
```

The aggregation writes CSV/JSON summaries, a scaling curve, a per-subject
boxplot, and a 100%-versus-original-baseline comparison.
