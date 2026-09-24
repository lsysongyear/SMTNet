# CNN+TCN

This model uses the project's normal training, loss, postprocessing, subject
evaluation, Slurm submission, and result-directory code. No manifest or separate
search project is required. Existing models and results are unchanged.

## Edit Parameters and Submit

Edit the `pnpl_cnn_tcn` entry in
`configs/speech/my_run/model_sweep.yaml`. This entry is one complete parameter
combination; several parameters may be changed together. Values here must be
scalars, not search lists. Keep `input_dim: 57` and `n_subjects: 25` for PKU-EEG.

From the project root on Huairou:

```bash
bash submit_pretrain_per_sub_5t5v.sh --models pnpl_cnn_tcn --seed 545
```

This is the existing submission script: it snapshots the current YAML files and invokes
`sbatch job_pretrain_per_sub_5t5v.slurm`. Subsequent YAML edits do not change an
already submitted job's parameter snapshot. `--seed` is the data split seed;
the existing model-sweep branch uses model seed 1. By default all 25 subjects
are trained jointly and evaluated individually. Running the script without
`--models` selects **all** entries, including the old models.

For the project's existing Cartesian search interface, put the chosen model in
`configs/speech/my_run/config.yaml`, match `general.model_name`, put parameter
lists in `configs/speech/my_run/search-space.yaml`, then submit with `--models ""`.
Do not use the obsolete `design.py` from the stopped search for this project.
Do not run training directly on the login node.

## Parameters

- Current configured CNN+TCN: `model_dim=100`, `tcn_layers=6`, `dropout=0.1`.
  These YAML values override the model class's constructor defaults. `model_dim` changes
  both the native input-projection width and the TCN width. Dropout does not
  disable the original BatchNorm inside each TCN block.
- The model retains independent subject-indexed
  `Linear(C,4C) -> GELU -> Linear(4C,C) -> GELU` projections before the native
  backbone input projection. Missing IDs are rejected for multi-subject data.
- The CNN uses an algebraically equivalent large-dilation optimization.
  This is an implementation optimization, not a new layer.

## Results

The existing training entry point saves results under:

```text
results/speech-detection/model:<model>-dataset:eeg_speech_v1/
  split:per_sub_5t5v/<parameter-string>/pretrain/runN:seed1_fold0/
```

The model key is `pnpl_cnn_tcn`. Different parameter
combinations get different parameter directories; repeated runs use the normal
incrementing run number. The project's existing per-subject evaluation also
writes `sub-XX/pretrain_eval/` and `pretrain_eval_summary.json` at the existing
parameter-directory level. Its summary represents the latest evaluation, not
a separate immutable summary for each repeat. The checkpoint path in each
subject's results identifies the evaluated run. Follow the original project's
convention when archiving repeated experiments.

No code changes were made to data splitting, preprocessing, MSE, AdamW, batch
size, epoch budget, smoothing, threshold selection, or metric calculation.
The model returns logits; the shared ClassificationModule applies the one
existing smoothing/interpolation/sigmoid pipeline. Model source and shared
helper source are copied into the run directory alongside its configuration.

## Source Provenance

- PNPL Parameter team CNN+TCN: OpenTSLab/
  MEG-to-Speech-Detection-for-the-LibriBrain-Competition-NeurIPS-2025,
  commit `931e9e93404a53e8ff7200b16e2c8e00eafe3481`, classes `CNN_TCN` and
  `TCNBlock`. The source commit contains no explicit LICENSE; this adaptation
  is retained for research reproduction. Obtain permission before distributing
  derived source.

This is a subject-projected adaptation under the project's shared training and
evaluation protocol, not an exact reproduction of the competition submission
or the original authors' training protocol.
