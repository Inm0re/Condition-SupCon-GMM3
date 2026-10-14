# Condition-SupCon–GMM3

Code for **Condition-aware contrastive representation and multi-component density
modelling for normal-only bearing anomaly detection**.

This is a **code-only** release. It contains no original or preprocessed datasets,
train/calibration/test arrays, pretrained weights, normalization statistics,
Gaussian banks, calibration parameters or restricted industrial files. Prepare
your own public data, train locally, and then use the generated local artifacts.

## Method

Normal-only refers to health-state learning: normal signals alone are used for
representation learning, density fitting and calibration. Operating-condition
metadata still supervise training and route inference. The method supports only
conditions represented during normal fitting and calibration.

The shared CNN encoder returns a 64-dimensional embedding. Each condition has
three Ledoit–Wolf Gaussian components. Scoring takes the minimum prior-adjusted
component negative log-likelihood, including the covariance determinant and
component-prior terms; it is not a soft-mixture likelihood. Normal calibration
determines median–IQR alignment and a pooled q95 threshold for the primary cohorts.

See [protocols](docs/protocols.md) for recorded split boundaries, paired-training
rules, lifecycle threshold scope and independence limitations.

## Install

Python 3.12 or later is required; Python 3.12.3 is used for release verification.
Install PyTorch for your CPU/CUDA platform first, then:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

The numerical versions in `requirements.txt` record the inspected execution
environment. CPU smoke checks do not imply bitwise GPU retraining equivalence.
Paderborn archive preparation additionally requires `unrar` on your PATH.

## Prepare public data

Obtain public raw inputs yourself. All preprocessing writes to a **new output
directory** and refuses to overwrite an existing dataset. No private-data builder
is included. Dataset data retain their upstream licenses independently of this
software license.

```bash
# HUST fixed-speed .xls text files (X/Y/Z channels, 262144 samples per source).
python preprocessing/prepare.py --dataset hust --raw /your/hust/raw --output data/hust
# The HUST array folder is data/hust/windows_L2048_H256.

# Paderborn .rar bearing archives in one directory. The source builder retains
# archive provenance and both current channels plus vibration. REAL evaluation
# explicitly excludes artificial-fault categories after frozen scoring.
python preprocessing/prepare.py --dataset paderborn_real --raw /your/paderborn/raw --output data/paderborn_real

# SEU bearingset directory: two health and eight fault .csv source files.
python preprocessing/prepare.py --dataset seu_dds --raw /your/seu/bearingset --output data/seu_dds

# XJTU raw root containing Bearing1_1 and Bearing2_2 CSV directories.
python preprocessing/prepare.py --dataset xjtu_sy --raw /your/xjtu/raw --output data/xjtu_sy
```

Alternatively, point training at your own arrays in the documented layout:
`train_x.npy`, `train_condition.npy`, `val_x.npy`, `val_condition.npy`,
`test_x.npy`, `test_condition.npy`, `test_label.npy`, and source/fault metadata.
HUST uses `train.npy` and `test.npy` instead of the `_x` names. Arrays use
`[windows, time, channels]`; identifiers are integer arrays. Test labels are
opened only for evaluation after model/reference fitting.

The XJTU raw builder supplies a portable 16-window float16 interface. Previously
prepared inputs may differ in quantization/preprocessing; use the same input
recipe when comparing freshly trained models to archived results.

## Train and evaluate

```bash
python -m csgmm.cli --config configs/hust.json --data data/hust/windows_L2048_H256 --output outputs/hust
python -m csgmm.cli --config configs/paderborn_real.json --data data/paderborn_real --output outputs/paderborn_real
python -m csgmm.cli --config configs/seu_dds.json --data data/seu_dds --output outputs/seu_dds
python -m csgmm.lifecycle --data data/xjtu_sy --output outputs/xjtu_sy
```

By default the five seeds are 35001–35005 and training lasts 200 steps. Existing
run directories are not overwritten. Use `--stage train` and `--stage evaluate`
for separate primary-cohort stages. `--seed 35001` runs one seed. The lifecycle
entry point preserves bearing-wise window-calibration q95 and q90 record scoring;
these statistical scales are stated explicitly in the generated protocol.

Artifacts created locally include `model.pt`, `normal_reference.joblib`,
`training_history.csv`, scores and metrics. They are ignored by Git and are not
part of this release. Load artifact files only from trusted local training runs.

The public primary training code preserves the archived legacy pairing for HUST
and Paderborn and group-restricted pairing for SEU. Fresh retraining is necessary
because archived checkpoints are intentionally not distributed. The documented
auxiliary self/same-condition pair ambiguity remains in the original recipe.

## Factorial variants and sensitivity

`configs/variants.json` defines all nine original variants. For example:

```bash
python -m csgmm.cli --config configs/paderborn_real.json --data data/paderborn_real --output outputs/ablation --variant A2
python -m csgmm.report --runs outputs --output outputs/summary
```

Run each variant and each seed in a separate output folder. To evaluate K, q or
temperature settings, copy the JSON configuration, change the requested field
and use a new output directory; do not select a configuration from test-fault
metrics while describing selection as normal-only. Report five-seed SD as
algorithmic variability, not physical replications. Public-only summaries do not
reproduce the paper's four-cohort macro including the restricted industrial cohort.

## Local inference

First train. There are no downloadable pretrained artifacts:

```bash
python -m csgmm.infer --windows data/paderborn_real/test_x.npy \
  --conditions data/paderborn_real/test_condition.npy \
  --artifact outputs/paderborn_real/seed_35001/H --output outputs/inference.npz
```

Inputs must use the same preprocessing and channel order as training. The CLI
rejects unsupported condition identifiers; metadata errors that pass these
checks remain a limitation. This packaging check is not an unseen-condition
generalization experiment.

## Physical-bearing uncertainty

The six-fold Paderborn recipe retains all four known conditions, uses four healthy
bearings for fitting, one for calibration and one for normal testing, and evaluates
the same 14 real-damaged bearings. It resamples physical units with shared damaged
identity across folds, not individual windows:

```bash
CSGMM_RAW=/your/paderborn/raw CSGMM_DATA=data/paderborn_real \
CSGMM_OUTPUT=outputs/paderborn_sixfold python -m csgmm.bearing_holdout
```

This is a separate six-fold training run, not required to train the main model.
The 5000-draw intervals are conditional on fitted models and the observed bearing
composition; they do not include retraining or new-campaign uncertainty.

## Third-party content and scope

The adapted encoder originates from PaAno. Its upstream MIT notice is retained
in [third_party](third_party/PaAno_LICENSE.txt). Other source methods are separate
baselines rather than components of the proposed detector. No upstream datasets
or private industrial preprocessing are redistributed. This release enables the
public proposed-model and factorial recipes; it does not turn the unavailable
industrial cohort into a publicly reproducible experiment.
