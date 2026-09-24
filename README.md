# SMTNet: Speech Detection from EEG and MEG

This repository contains the training, evaluation, and data-scaling code for
"Detecting the Onset and Offset of Neural Speech Processing from EEG/MEG
Recordings." SMTNet is implemented under the model key
`brain_magic_speech_v7`.

## Repository layout

| Dataset | Model comparison and ablation | Training-data scaling |
| --- | --- | --- |
| PKU-EEG | `VAD_pkueeg_final/` | `VAD_pkueeg_scaling/` |
| SparrKULee | `VAD_SparKULee/` | `VAD_SParKULee_scaling/` |
| SMN4Lang | `VAD_SEM4Lang/` | `VAD_SEM4Lang_scaling/` |
| LibriBrain | `VAD_MEG_NIPS/` | `VAD_MEG_NIPS_scaling/` |

Each model-comparison project has a `configs/speech/my_run/config.yaml`
data/training configuration, a `model_sweep.yaml` with the paper's model
parameters, a submission script, and a Slurm job file. Each scaling project
has `configs/scaling/config.yaml`, `train_scaling.py`,
`submit_scaling.sh`, `job_scaling.slurm`, and the `scaling/` package.
`paper_results.csv` and `scaling_results.csv` record the aggregate
values reported in the paper for comparison with generated results.

## Environment and data

Use Python 3.10 with PyTorch, PyTorch Lightning, MNE, NumPy, SciPy, PyYAML,
scikit-learn, and the dataset readers. The package versions used for the
LibriBrain project are recorded in `VAD_MEG_NIPS/environment.yml`. Audio
label extraction uses FFmpeg and [Silero VAD](https://github.com/snakers4/silero-vad).
Training jobs require a Slurm cluster with GPU nodes. Set `PYTHON` to the
Python executable available on the compute nodes.

The repository does not contain the datasets or generated signal arrays,
speech labels, and checkpoints. Obtain the four datasets under their
respective access terms. Set the `data_path` and `label_path` values in
the eight project configuration files to the corresponding prepared data
locations. For LibriBrain, configure the `data_path` entries for the
dataset's HDF5 recordings and event files. Adjust the `#SBATCH` resources
in the job files for the target cluster.

| Dataset | Signal preparation | Speech-label source |
| --- | --- | --- |
| PKU-EEG | `VAD_pkueeg_final/speech_code/utils.py` and `preprocess/ch_transform.py` | `VAD_pkueeg_final/Label_VAD/extract_vad_segments.py` |
| SparrKULee | `VAD_SparKULee/SparKULee/preprocess/extract_eeg_250hz_bp.py` | `VAD_SparKULee/Label_VAD/extract_sparkulee_vad_segments.py` |
| SMN4Lang | `VAD_SEM4Lang/SEM4Lang/preprocess_meg.py` | `VAD_SEM4Lang/Label_VAD/extract_vad_segments.py` |
| LibriBrain | `VAD_MEG_NIPS/speech_code/data_process.py` | Dataset event timestamps |

The preprocessing scripts accept data and output paths as arguments or
configuration variables. The corresponding configurations identify the
expected processed-file locations and dataset-specific preprocessing tags.

## Submit the paper experiments

From the repository root, inspect the selected commands:

```bash
bash reproduce_paper.sh --dataset all --experiment all
```

Submit a dataset's model-comparison and scaling jobs:

```bash
PYTHON=/path/to/python bash reproduce_paper.sh --dataset pku --experiment tables --submit
PYTHON=/path/to/python bash reproduce_paper.sh --dataset pku --experiment scaling --submit
```

`--dataset` accepts `pku`, `spar`, `sem`, `libri`, or `all`.
`--experiment` accepts `tables`, `scaling`, or `all`. Without
`--submit`, the wrapper prints commands without launching jobs. The
submission scripts write run-specific configuration snapshots and submit
Slurm arrays. Logs and results are written under the corresponding project.

The model-comparison jobs include SMTNet, CNN-LSTM, DilatedConv,
EEGConformer, AWaveNet, CNN+TCN, and component ablations. Their model keys
are `brain_magic_speech_v7`, `cnn_lstm`, `dilated_conv`,
`eeg_conformer`, `awavenet`, `pnpl_cnn_tcn`,
`brain_magic_no_subject_attn`, `brain_magic_no_short_conv`, and
`brain_magic_no_feature_encoder`. LibriBrain uses the eight applicable
keys, excluding the subject-projection ablation. The model dimensions and
depths are specified in each project's `model_sweep.yaml`.

## Evaluation protocol

Signals and speech labels are segmented into 12-second windows. Training
windows use a 6-second stride; validation and test windows are
nonoverlapping. Multi-subject models pool training windows across subjects,
then select a checkpoint and a threshold for each subject from validation
Macro_Acc. Threshold candidates run from 0.01 to 0.99 in steps of 0.01.
The selected checkpoint and threshold are applied to that subject's test
set. For binary speech/silence labels,
`Macro_Acc = (speech recall + silence recall) / 2`.

| Dataset | Paper split | Model seed | Aggregation |
| --- | --- | ---: | --- |
| PKU-EEG | `split_seed=545` | 1 | 25 subjects |
| SparrKULee | `candidate_18`, selected by comparing candidate test Macro_Acc; `split_seed=84`; validation `audiobook-5-1`; test `audiobook-5-3` | 1 | 23 subjects |
| SMN4Lang | `split_seed=5` | 1 | 12 subjects |
| LibriBrain | `split_seed=42,43,44,45,46` | 42 | Five splits |

The SparrKULee submitter verifies the common validation/test story pair
against the prepared recordings before submitting jobs. The shared model
settings include batch size 64, AdamW with learning rate `1e-3`,
mean-squared-error loss, and up to 100 epochs. The training code retains
the five checkpoints with the lowest global validation loss for
subject-specific checkpoint selection.

## Training-data scaling

Scaling uses 5%, 10%, 20%, 50%, 80%, and 100% of each subject's training
recording duration. Each subset is a chronological prefix in trial order;
a prefix may end inside a trial. Window segmentation is applied after the
prefix is selected. The validation and test sets remain fixed across
proportions. Each scaling project provides
`scaling/aggregate_results.py` to summarize completed runs.

## Sources

- [PKU-EEG](https://openneuro.org/datasets/ds008834/versions/1.0.0)
- [SparrKULee](https://doi.org/10.3390/data9080094)
- [SMN4Lang](https://doi.org/10.1038/s41597-022-01708-5)
- [LibriBrain](https://arxiv.org/abs/2506.02098)
- [MEBM-Speech architecture](https://arxiv.org/abs/2603.02255)
- [CNN+TCN implementation](https://github.com/OpenTSLab/MEG-to-Speech-Detection-for-the-LibriBrain-Competition-NeurIPS-2025)
