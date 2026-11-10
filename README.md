# SynPhen-CL

**SynPhen-CL: Phenotype Prediction of Synonymous Mutations via Label–Tabular Contrastive Learning**

SynPhen-CL is a deep-learning framework for predicting phenotype associations of synonymous variants. It combines numerical variant features with semantic representations of phenotype labels to model label dependencies and variant–phenotype interactions.

The implementation predicts 24 outputs: a benign-associated output and 23 top-level Human Phenotype Ontology (HPO) phenotype categories. A general pathogenicity score can be derived as \(1 - P(\mathrm{benign})\).

## Overview

SynPhen-CL contains three main learnable components:

1. **Table Feature Extractor (TFE).** An FT-Transformer encodes numerical variant features and learns interactions between features.
2. **Label–Table Fusion Transformer (LTFT).** Self-attention models relationships among phenotype labels, while cross-attention lets each label representation attend to the variant's tabular feature tokens.
3. **Label–Table Contrastive Learning (LTCL).** A contrastive objective aligns each variant representation with label prototypes according to the sample's training labels.

The model uses two complementary sources of label semantics:

- **CODER:** a 768-dimensional medical concept representation.
- **cui2vec:** a 500-dimensional clinical concept representation.

The two representations are concatenated into a 1,268-dimensional label prior for each of the 24 outputs. During inference, the tabular branch and label-aware branch produce probabilities that can be combined by a weighted linear fusion. The fusion weight is selected using the validation set in the default training workflow.

## Project structure

```text
synphen-cl/
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── .gitignore
├── synphen_cl/
│   ├── config.py                    # Central paths and hyperparameters
│   ├── model.py                     # TFE, LTFT, and prediction heads
│   ├── losses.py                    # Classification and contrastive losses
│   ├── data.py                      # CSV loading and preprocessing artifacts
│   ├── training.py                  # Training loops and early stopping
│   ├── evaluation.py                # Metrics, probability fusion, and output files
│   ├── train.py                     # Training/evaluation entry point
│   ├── predict.py                   # Inference entry point
│   ├── utils.py                     # Reproducibility utilities
│   └── embeddings/
│       ├── phenotypes.py            # Canonical 24-label definitions and order
│       ├── generate_coder_embeddings.py
│       └── generate_cui2vec_embeddings.py
├── data/
│   ├── README.md                    # Data availability and input schema
│   ├── raw/
│   │   ├── README.md
│   │   ├── train_selected.csv       # Local input; not distributed
│   │   └── test_selected.csv        # Local input; not distributed
│   ├── embeddings/
│   │   ├── coder_phenotype_embeddings.npy
│   │   ├── cui2vec_phenotype_embeddings_avg.npy
│   │   └── phenotype_labels.txt
│   └── external/
│       └── README.md                # Optional cui2vec source file instructions
├── models/
│   └── README.md                    # Optional local CODER checkpoint instructions
├── outputs/
│   └── .gitkeep
└── tests/
    └── test_smoke.py
```

## Data availability and licensing

The `data/raw/` directory is intended for locally prepared input files. See [`data/README.md`](data/README.md) for the complete schema and label specification.

## Environment setup

Python 3.10 or later is recommended.

```bash
conda create -n synphen-cl python=3.10
conda activate synphen-cl
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For GPU training, install a PyTorch build compatible with your CUDA driver and hardware. The default device is selected automatically; it can also be set explicitly with `SYNPHEN_DEVICE`.

To install the development/test dependencies:

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Prepare the data

Place the following files in the indicated locations:

- `data/raw/train_selected.csv`
- `data/raw/test_selected.csv`

Each training/evaluation CSV must contain numerical feature columns and a `Label` column. The released schema omits `sequence_id`. The selected feature set described in the manuscript contains 800 numerical features; if using a different feature-selection result, document the change and keep the feature schema consistent across splits.

The `Label` column contains zero-based label indices separated by a pipe (`|`). Index `0` denotes the benign-associated output; indices `1`–`23` correspond to the 23 top-level HPO phenotype categories. Samples can have multiple phenotype labels. The canonical label order is recorded in `data/embeddings/phenotype_labels.txt` and defined in `synphen_cl/embeddings/phenotypes.py`.

For additional details, including prediction-only inputs and licensing notes, see [`data/README.md`](data/README.md).

## Train and evaluate

Run from the repository root:

```bash
python -m synphen_cl.train
```

The training entry point loads the configured training and test files, creates a training/validation split, fits preprocessing on the training partition, trains the model with early stopping, selects the fusion weight on the validation set, and evaluates predictions on the configured test set.

The default output directory is `outputs/training/`. Key artifacts include:

| File | Description |
|---|---|
| `v2p_fttransformer_dlcl_final.pt` | Trained model state dictionary |
| `feature_scaler.pkl` | Fitted feature scaler used by the saved checkpoint |
| `feature_columns.txt` | Feature names in the exact training order |
| `run_config.csv` | Hyperparameters, selected epoch, validation score, and fusion weight |
| `summary.csv` | Aggregate test metrics for the prediction branches and fused predictions |
| `per_label_metrics_by_branch.csv` | Per-label metrics for each branch |
| `proba_ft_tabular.csv` | Probabilities from the tabular branch |
| `proba_label.csv` | Probabilities from the label-aware branch |
| `proba_fused_mean.csv` | Mean-fusion probabilities |
| `proba_fused_weighted.csv` | Validation-selected weighted-fusion probabilities |
| `pr_curve_micro.png` | Micro-averaged precision–recall curves |
| `fusion_weight_search.csv` | Validation-set fusion-weight search results |

The saved scaler and feature order are required for consistent inference. Do not fit a new scaler on an evaluation or prediction set.

## Predict on a new dataset

After training has created the model checkpoint and preprocessing artifacts, run:

```bash
python -m synphen_cl.predict \
    --input path/to/new_features.csv \
    --output outputs/new_prediction
```

The input must contain all feature columns used during training. The code aligns the input columns to the saved training order and raises an error if required columns are missing.

- **Evaluation mode:** include `Label` to calculate label-wise and aggregate metrics.
- **Prediction-only mode:** omit `Label`; the program writes probabilities without calculating metrics that require ground-truth labels.
- `sequence_id` is not required. If omitted, row-based placeholder IDs are written with the prediction outputs; they must not be interpreted as real variant identifiers.

Typical prediction outputs include `proba_ft_tabular.csv`, `proba_label.csv`, `proba_fused_mean.csv`, `proba_fused_weighted.csv`, and, for prediction-only inputs, `all_label_proba_prediction.csv`, `phenotype_proba_prediction.csv`, and `pathogenicity_proba_prediction.csv`.

## Configuration

Model architecture, preprocessing, loss weights, training strategy, and default paths are centralized in `synphen_cl/config.py`.

Common environment-variable overrides include:

| Variable | Purpose |
|---|---|
| `SYNPHEN_TRAIN_CSV` | Training CSV path |
| `SYNPHEN_TEST_CSV` | Test/evaluation CSV path |
| `SYNPHEN_PREDICT_CSV` | Default prediction input path |
| `SYNPHEN_TRAIN_OUTPUT` | Training output directory |
| `SYNPHEN_PREDICT_OUTPUT` | Prediction output directory |
| `SYNPHEN_SEED` | Random seed; default `42` |
| `SYNPHEN_DEVICE` | Runtime device, for example `cpu` or `cuda` |
| `SYNPHEN_CODER_NPY` | CODER label-embedding array path |
| `SYNPHEN_CUI2VEC_NPY` | cui2vec label-embedding array path |

For example, on Linux/macOS:

```bash
SYNPHEN_DEVICE=cpu SYNPHEN_SEED=42 python -m synphen_cl.train
```

On Windows PowerShell:

```powershell
$env:SYNPHEN_DEVICE = "cpu"
$env:SYNPHEN_SEED = "42"
python -m synphen_cl.train
```

## Regenerate label embeddings (optional)

The provided embedding arrays are sufficient for the default training workflow. Regeneration is only needed if you want to reproduce the label embeddings from their source resources.

### CODER

Place a compatible local CODER model checkpoint under `models/coder/`, then run:

```bash
python -m synphen_cl.embeddings.generate_coder_embeddings
```

The model checkpoint is not redistributed with this repository. The script uses local model files and writes a 24 × 768 embedding array.

### cui2vec

Place the original pretrained cui2vec vector file at `data/external/cui2vec_pretrained.csv`, then run:

```bash
python -m synphen_cl.embeddings.generate_cui2vec_embeddings
```

The script writes a 24 × 500 embedding array. The original cui2vec source file is not included.

Both scripts use the same canonical label order. Do not reorder the labels independently of the embedding rows.

## Reproducibility and implementation notes

- The default random seed is `42`; exact results can still vary with hardware, CUDA, and library versions.
- The default feature scaler is a `QuantileTransformer` with a normal output distribution.
- The scaler is fitted on training data and reused for validation, test, and inference data.
- Early stopping and fusion-weight selection use validation data in the default workflow.
- The checkpoint stores model parameters, not the Python model architecture or fitted preprocessing objects; the scaler and feature-column list are saved separately.
- Review `synphen_cl/config.py` and record any deviations from the default configuration when reporting results.

## Citation

If you use SynPhen-CL, please cite the associated manuscript:

> **SynPhen-CL: Phenotype Prediction of Synonymous Mutations via Label–Tabular Contrastive Learning.**




