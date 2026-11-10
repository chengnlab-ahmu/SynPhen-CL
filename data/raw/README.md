# Data and Input File Specification

This directory contains the input locations and label-embedding artifacts used by SynPhen-CL. **Raw training and test CSV files are not distributed with the public code package.** Place authorized local copies in the paths below before running the model.

## Required and optional files

| Path (relative to repository root) | Status | Description |
|---|---|---|
| `data/raw/train_selected.csv` | Required for training | Training samples, numerical feature columns, and `Label` |
| `data/raw/test_selected.csv` | Required for the default train/evaluation workflow | Held-out test samples, numerical feature columns, and `Label` |
| `data/embeddings/coder_phenotype_embeddings.npy` | Required | CODER label embeddings; expected shape `(24, 768)` |
| `data/embeddings/cui2vec_phenotype_embeddings_avg.npy` | Required | cui2vec label embeddings; expected shape `(24, 500)` |
| `data/embeddings/phenotype_labels.txt` | Included reference | Canonical label index and label-name mapping |
| `data/external/cui2vec_pretrained.csv` | Optional; only for regenerating embeddings | Original pretrained cui2vec concept vectors |
| `models/coder/` | Optional; only for regenerating embeddings | Local CODER model and tokenizer files |

The two label-embedding arrays supplied with this project can be used directly for training; the original CODER checkpoint and cui2vec source file are only needed if you want to regenerate them. Verify the relevant third-party terms before redistributing any derived embeddings or source resources.

## CSV schema

Each row represents one sample. For the selected feature set described in the manuscript, each CSV contains 800 numerical feature columns and a `Label` column. The feature-selection procedure can result in a different feature count in other experiments; if the feature set changes, record the change and ensure the training, test, and inference schemas remain compatible.

### Identifier handling and HGMD licensing

**The `sequence_id` column has been removed from released data files to comply with HGMD copyright and licensing requirements.** It is not required by the SynPhen-CL code.

When `sequence_id` is absent, the loader creates row-based placeholders (`sample_0`, `sample_1`, …). These are technical placeholders, not variant identifiers. If you are authorized to retain a mapping between a sample and its original record, keep that mapping in a secure local file outside the public repository. Preserve the row order when using that private mapping to interpret output predictions.

Do not add restricted identifiers or private sample metadata to the public CSV files.

### Required columns

- **Numerical feature columns:** all selected input features must be numeric. The training file defines the feature schema used by the model.
- **`Label`:** required for training and for evaluation against known labels. It may be omitted only for prediction-only inputs passed to `python -m synphen_cl.predict`.
- **`sequence_id`:** intentionally omitted from released data; it is optional for local workflows and not required by the model.

The training and test files must contain the same set of feature column names. The code aligns test and inference columns to the feature order saved from training; missing required features cause an error. For reproducibility, do not change the selected feature set between training and evaluation.

## Label encoding

`Label` contains zero-based integer label indices separated by a pipe character (`|`). For example:

```text
4|16|17
```

This represents a sample associated with labels 4, 16, and 17. Use the canonical order in `phenotype_labels.txt` exactly; do not reorder label indices or embedding rows.

| Index | Label |
|---:|---|
| 0 | Benign |
| 1 | Cellular |
| 2 | Blood or blood-forming tissues |
| 3 | Head or neck |
| 4 | Limbs |
| 5 | Metabolism or homeostasis |
| 6 | Prenatal development or birth |
| 7 | Breast |
| 8 | Cardiovascular |
| 9 | Digestive |
| 10 | Ear |
| 11 | Endocrine |
| 12 | Eye |
| 13 | Genitourinary |
| 14 | Immune |
| 15 | Integument |
| 16 | Musculoskeletal |
| 17 | Nervous |
| 18 | Respiratory |
| 19 | Thoracic cavity |
| 20 | Voice |
| 21 | Constitutional symptom |
| 22 | Growth |
| 23 | Neoplasm |

Index `0` is the benign-associated output. Indices `1`–`23` are the 23 top-level HPO phenotype categories. The general pathogenicity score reported by the inference utility is derived as `1 - P(benign)`.

## Prepare local data

1. Place `train_selected.csv` and `test_selected.csv` in `data/raw/`.
2. Ensure that both files use the same feature names and label encoding.
3. Keep the original sample-to-record mapping, if one is permitted and needed, outside the repository and preserve its row order.
4. Confirm that `.gitignore` is active. CSV files under `data/raw/` and `data/external/` are ignored by default.

Example layout:

```text
data/
├── README.md
├── raw/
│   ├── README.md
│   ├── train_selected.csv       
│   └── test_selected.csv        
├── embeddings/
│   ├── coder_phenotype_embeddings.npy
│   ├── cui2vec_phenotype_embeddings_avg.npy
│   └── phenotype_labels.txt
└── external/
    ├── README.md
    └── cui2vec_pretrained.csv   
```

## Prediction-only data

For inference on samples without known labels, provide the same numerical feature columns but omit `Label`:

```bash
python -m synphen_cl.predict \
    --input path/to/new_features.csv \
    --output outputs/new_prediction
```

The saved scaler and `feature_columns.txt` from the training run are reused. Do not refit or replace the scaler using the new prediction data. If `sequence_id` is absent, output rows will use generated placeholders; they cannot be used to recover original variant identifiers without a separately maintained local mapping.

