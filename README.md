# SMTNet: Speech Detection from EEG and MEG

This release collects the code used for the four-dataset experiments in
"Detecting the Onset and Offset of Neural Speech Processing from EEG/MEG
Recordings." SMTNet is the name used in the paper; its earlier MEBM-Speech
architecture is described in the [original preprint](https://arxiv.org/abs/2603.02255).
The code covers the main model, five comparison models, three ablations, and
training-data scaling experiments. `paper_results.csv` transcribes Tables 1
and 2 of the paper; `scaling_results.csv` transcribes the means printed on its
scaling figure. These CSVs are reference values, not substitutes for the
per-subject or per-split evaluation outputs produced by the scripts.

## Project layout

The release retains the eight experiment projects rather than merging their
dataset-specific preprocessing and evaluation code:

| Dataset in paper | Main experiments | Scaling experiments |
| --- | --- | --- |
| PKU-EEG | `VAD_pkueeg_final/` | `VAD_pkueeg_scaling/` |
| SparrKULee | `VAD_SparKULee/` | `VAD_SParKULee_scaling/` |
| SMN4Lang | `VAD_SEM4Lang/` | `VAD_SEM4Lang_scaling/` |
| LibriBrain | `VAD_MEG_NIPS/` | `VAD_MEG_NIPS_scaling/` |

Each main project contains the dataset-specific loading, preprocessing,
configuration, training, checkpoint selection, and testing code used for its
paper table. The corresponding scaling project keeps the original validation
and test sets and varies only the amount of training recording. The paper's
training submission entry points are `submit_pretrain_per_sub_5t5v.sh` for
PKU-EEG, `submit_pretrain_random_trial.sh` for SparrKULee, `submit.sh` for
LibriBrain, `submit_train.sh` for SMN4Lang, and `submit_scaling.sh` in each
scaling project. Review the configurations those scripts reference before
submission. Submit training through Slurm on the cluster rather than running
a long training process on a login node. Paths, partitions, account names,
and raw data locations in site-specific scripts must be set for your
environment. Some retained source scripts have historical exploratory
defaults; the paper protocol in this README and the release's reproduction
wrapper take precedence when reproducing the reported results.
Do not assume that an unmodified legacy default selects a paper run.
`SOURCE_MANIFEST.csv` records source paths and snapshot hashes;
`tools/snapshot_sources.py` is the snapshotting utility, not a training
entry point.

## Reproduce the paper runs

On a Slurm cluster, first inspect the commands without submitting anything:

```bash
bash reproduce_paper.sh --dataset all --experiment all
```

After configuring input data and the Python environment, submit one dataset
and experiment at a time, for example:

```bash
PYTHON=/path/to/python bash reproduce_paper.sh --dataset spar --experiment tables --submit
PYTHON=/path/to/python bash reproduce_paper.sh --dataset spar --experiment scaling --submit
```

Valid dataset selectors are `pku`, `spar`, `sem`, `libri`, and `all`;
experiment selectors are `tables`, `scaling`, and `all`. The wrapper selects
only the model keys and seeds used in the paper, not every historical entry
in `model_sweep.yaml`. `--submit` is required for any job submission. It does
not generate results from the reference CSVs. Slurm `sbatch` must be available,
and the partition, GPU request, and time limit in each job file must fit the
target cluster. The `PYTHON` environment variable overrides the interpreter
in release job scripts; otherwise they use `python3` on `PATH`.

The historical `VAD_MEG_NIPS/environment.yml` records the original Python
3.10 package environment, including PyTorch, MNE, Lightning, NumPy, SciPy,
PyYAML, and the dataset libraries. It is a platform-specific snapshot rather
than a guaranteed portable lockfile. Audio labeling additionally needs
FFmpeg and a local checkout of Silero VAD. The latter was an archive without
Git metadata on the source cluster, so its exact upstream revision is not
recoverable from this release.

Before submitting, check the `data_path` and `label_path` entries in each
main project's `configs/speech/my_run/config.yaml` and the corresponding
scaling project's `configs/scaling/config.yaml`. They currently name the
read-only processed data locations used on Huairou. On a different system,
point them to your own dataset and label directories. All released training
job files run from their own project directory and write results and logs
there, not into the original source projects. The raw/processed recordings,
labels, and weights are not included in Git. The reference CSVs contain only
published aggregate values.

The preprocessing and label-generation entry points are:

| Dataset | Signals | Labels |
| --- | --- | --- |
| PKU-EEG | `VAD_pkueeg_final/speech_code/utils.py` performs the training-time channel alignment, downsampling, and normalization; `speech_code/data_process.py` is an optional cache builder. | `VAD_pkueeg_final/Label_VAD/extract_vad_segments.py` |
| SparrKULee | `VAD_SparKULee/SparKULee/preprocess/extract_eeg_250hz_bp.py` | `VAD_SparKULee/Label_VAD/extract_sparkulee_vad_segments.py` |
| SMN4Lang | `VAD_SEM4Lang/SEM4Lang/preprocess_meg.py` | `VAD_SEM4Lang/Label_VAD/extract_vad_segments.py` |
| LibriBrain | `VAD_MEG_NIPS/speech_code/data_process.py` produces gradiometer arrays from the dataset's HDF5/events files; run it only on a writable data copy. | Official dataset event timestamps, parsed during loading. |

The preprocessing commands accept or document dataset paths separately from
their release-local outputs. The optional legacy PKU cache builder copied
into multiple project trees is **not** part of the other datasets' paper
pipelines. For the paper results, use the processed-data paths recorded in
the configurations and verify their contents before training.

The configuration identifiers behind the paper tables are
`brain_magic_speech_v7` (SMTNet), `brain_magic_no_subject_attn` (w/o Spatial
Projection), `brain_magic_no_short_conv` (w/o MultiScale Conv),
`brain_magic_no_feature_encoder` (w/o BrainMagick Block), `cnn_lstm`,
`dilated_conv`, `eeg_conformer`, `awavenet`, and `pnpl_cnn_tcn` (CNN+TCN).
Names in the paper are presentation labels; these identifiers are the model
keys to select in code. The single-subject LibriBrain experiments omit the
spatial-projection ablation.

The four datasets, derived recordings, model weights, and cluster job outputs
are **not** redistributed here. Obtain each dataset under its own terms and
set the input paths in the project configurations before running. The
`results/` directories generated by a run are outputs, not inputs to train or
evaluate a model.

## Paper protocol

- The task is framewise binary speech/silence detection from aligned EEG or
  MEG recordings. PKU-EEG, SparrKULee, and SMN4Lang labels are generated from
  their stimulus audio with [Silero VAD](https://github.com/snakers4/silero-vad).
  LibriBrain uses the dataset's official speech timestamps instead.
- Split recordings at the trial level. Form 12-second windows with a 6-second
  stride (50% overlap) for training. Validation and test windows are
  nonoverlapping. For multi-subject data, held-out stimuli/trials are kept
  consistently separated across subjects; training pools the subjects, while
  validation calibration and test reporting are per subject.
- The paper's shared training settings are batch size 64, AdamW at learning
  rate `1e-3`, up to 100 epochs, mean-squared-error loss, and early stopping
  after five epochs without improvement in global validation loss. Retain the
  five checkpoints with the lowest global validation loss.
- For each subject, choose a checkpoint from those five and a speech
  probability threshold from `0.01, 0.02, ..., 0.99` by maximum validation
  Macro_Acc. Fix both choices before evaluating that subject's test set. In
  this binary task, `Macro_Acc = (speech recall + silence recall) / 2`.
- The baselines in Table 1 use the same subject-indexed spatial projection
  front end and the same split, optimizer, checkpoint, and threshold-selection
  protocol as SMTNet. Table 2 removes one named component at a time. The
  spatial-projection ablation is not applicable to single-subject LibriBrain.

The paper configurations use the following split/model seeds:

| Dataset | Split selection | Model seed | Notes |
| --- | --- | ---: | --- |
| PKU-EEG | `split_seed=545` | 1 | Subject-level results are summarized across 25 subjects. |
| SparrKULee | `candidate_18`, `split_seed=84` | 1 | Shared validation story `audiobook-5-1`; shared test story `audiobook-5-3`; 23 retained subjects. |
| SMN4Lang | `split_seed=5` | 1 | Results are summarized across 12 subjects. |
| LibriBrain | `split_seed=42,43,44,45,46` | 42 | Single subject; mean and standard deviation across five splits. |

**Test-informed selection disclosure.** The SparrKULee held-out story pair
`candidate_18` was selected after comparing test Macro_Acc across candidate
validation/test story pairs. Thus its reported test value is exploratory and
test-informed, not an unbiased estimate from a once-only held-out test set.
This selection must be disclosed when citing or comparing the result. The
other datasets' reported means and standard deviations use the aggregation
specified above; do not interpret the SparrKULee result as a prospectively
chosen split.

## Training-data scaling

The scaling percentages are **recording-duration percentages**, not percentages
of already-generated windows or trial counts. For each subject independently,
take the chronological prefix of that subject's *training* recordings in the
original trial order. A prefix may stop partway through a trial. Segment only
that retained duration into 12-second windows with a 6-second stride. The
tested proportions are 5%, 10%, 20%, 50%, 80%, and 100%; they form nested
training sets. The complete validation and test recordings, windows, and
labels are unchanged at every proportion. The baseline full-data run and the
100% scaling run should use the same effective split, preprocessing, model,
and evaluation settings; compare their saved configurations before claiming
equivalence.

The scaling figure annotates only mean Macro_Acc to two decimal places. Its
shaded bands denote one standard deviation, but the figure does not provide
exact numerical standard deviations, so `scaling_results.csv` deliberately
leaves that field empty. Use the underlying run summaries for higher-precision
values or uncertainty analyses.

## Data and implementation attribution

- [PKU-EEG on OpenNeuro, ds008834 v1.0.0](https://openneuro.org/datasets/ds008834/versions/1.0.0)
- [SparrKULee dataset description](https://doi.org/10.3390/data9080094)
- [SMN4Lang dataset description](https://doi.org/10.1038/s41597-022-01708-5)
- [LibriBrain dataset description](https://arxiv.org/abs/2506.02098)
- [BrainMagick](https://doi.org/10.1038/s42256-023-00714-5) and the
  [MEBM-Speech preprint](https://arxiv.org/abs/2603.02255), from which SMTNet
  derives its architecture
- [Parameter Team's public CNN+TCN implementation](https://github.com/OpenTSLab/MEG-to-Speech-Detection-for-the-LibriBrain-Competition-NeurIPS-2025),
  adapted as a comparison model; it is cited as a repository, not as an
  independently verified peer-reviewed paper
- [CNN-LSTM/CLDNN](https://doi.org/10.1109/ICASSP.2015.7178838),
  [dilated convolution](https://arxiv.org/abs/1511.07122),
  [EEG Conformer](https://doi.org/10.1109/TNSRE.2022.3230250), and
  [adapted WaveNet](https://doi.org/10.1109/ICASSP49357.2023.10095420),
  which motivate the remaining comparison models
- [Silero VAD](https://github.com/snakers4/silero-vad), used to generate
  training labels for the three datasets noted above

See the paper for full model definitions, scientific context, and references.
Third-party software and dataset licenses remain with their respective
authors; this release does not grant rights to any external dataset or weight
file.
