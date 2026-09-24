# VAD MEG NIPS Scaling

Independent training-data scaling experiments for `brain_magic_speech_v7` on
LibriBrain MEG. The source project `VAD_MEG_NIPS` is not modified.

## Protocol

- Original split repetitions: `split_seed=42,43,44,45,46`.
- Within each split seed, validation and test sets contain 9 fixed run keys each.
- Full training pool: 72 run keys.
- Paper scales: `5%, 10%, 20%, 50%, 80%, 100%`. The parser also accepts
  historical exploratory 40% and 60% settings, but the release submit script
  does not run them.
- Scaling unit: the chronological training recording-duration prefix. Cropping
  happens before 12-second windows are generated and may end inside a run.
- Validation and test runs remain complete at every scale.
- Model seed: `42`, matching the original v7 baseline.
- Model: only the complete `brain_magic_speech_v7` (`8/19` blocks).
- Checkpointing: top five checkpoints by validation loss.
- Final selection: select `(checkpoint, threshold)` by validation macro accuracy,
  then apply both unchanged to the test set.
- Primary metric: test macro accuracy.
- Each 100% run is checked against the corresponding original v7 split result.

The subset seed remains fixed at `1001` only for path compatibility; it no longer
affects sampling. The five split seeds reproduce the original MEG experiment's
data-partition repetitions.

## Validate

```bash
python3 -m unittest discover -s tests -v

./submit_scaling.sh --dry-run

python3 scaling/verify_baseline.py \
  --old-config /path/to/old/run4:seed42/config.yaml
```

## Submit

```bash
./submit_scaling.sh
```

The submit script creates one six-task array per missing split seed and skips
completed runs. Results are written below:

```text
results/scaling/brain_magic_speech_v7/protocol_duration_prefix_train_only/split_seed_XX/
  subset_seed_1001/model_seed_42/scale_XXX/run_slurm_JOBID/
```

Submit from this release project directory. The Slurm job uses
`SLURM_SUBMIT_DIR` as its code root and `PYTHON` (default `python3`) as the
interpreter. Activate the training environment first or set `PYTHON` to its
absolute executable path. Raw data are not distributed here; update
`configs/scaling/config.yaml` if the original Huairou path is not available.

## Aggregate

```bash
python3 scaling/aggregate_results.py
```

The aggregation writes both per-split results and the mean/standard deviation
across the five split seeds.
