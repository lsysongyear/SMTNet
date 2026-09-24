# VAD SParKULee Scaling

Independent training-data scaling experiments for `brain_magic_speech_v7` on
SParKULee.

## Protocol

- Subjects: `001-003,005-016,018-024,026` (23 subjects used by the v7 baseline).
- Paper split seed: `84` (candidate 18).
- Split protocol: `shared_story`. The 23 subjects share one validation story
  (`audiobook-5-1`) and one test story (`audiobook-5-3`); neither can appear in
  any subject's training set.
- Paper scales: `5%, 10%, 20%, 50%, 80%, 100%`. The parser also accepts
  historical exploratory 40% and 60% settings, but the release submit script
  does not run them.
- Scaling unit: each subject's chronological training recording duration. The
  prefix is cropped before 12-second windows are generated and may end inside
  the final trial. Validation and test trials remain complete at every scale.
- Subset seed `1001` is retained only for path compatibility and has no sampling effect.
- Model seed: `1`.
- Model: only the complete `brain_magic_speech_v7`.
- Checkpointing: top five checkpoints by global validation loss.
- Final selection: independently for each subject, select `(checkpoint, threshold)`
  by validation macro accuracy, then apply both unchanged to the test trial.
- Primary metric: mean per-subject test macro accuracy.

The split is globally story-disjoint, not merely disjoint in `(subject, story)`
pairs. Candidate 18 was selected from a story-pair search using test-set macro
accuracy; therefore its reported test score is test-informed and should not be
interpreted as an untouched holdout estimate. The training script asserts the
paper validation/test story pair whenever split seed 84 is used.

## Validate

```bash
python3 -m unittest discover -s tests -v

./submit_scaling.sh --dry-run
```

## Submit

```bash
./submit_scaling.sh
```

The six array tasks write independent runs below:

```text
results/scaling/brain_magic_speech_v7/protocol_duration_prefix_train_only/
  protocol_shared_story/split_seed_84/subset_seed_1001/model_seed_1/
  scale_XXX/run_slurm_JOBID/
```

Submit from this release project directory. The Slurm job uses
`SLURM_SUBMIT_DIR` as its code root and `PYTHON` (default `python3`) as the
interpreter. Activate the training environment first or set `PYTHON` to its
absolute executable path. Raw data and labels are not distributed here;
update `configs/scaling/config.yaml` if the original Huairou paths are not
available.

## Aggregate

```bash
python3 scaling/aggregate_results.py
```
