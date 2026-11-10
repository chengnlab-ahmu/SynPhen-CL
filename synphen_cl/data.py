"""Dataset loading, feature preprocessing, and semantic-prior utilities."""
import os
import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from .config import *
from .utils import seed_worker

def load_csv_data(path: str, feature_cols=None):
    df = pd.read_csv(path, low_memory=False)
    available_cols = [c for c in df.columns if c not in [ID_COL, LABEL_COL]]
    if feature_cols is None:
        feat_cols = available_cols
    else:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"{path} is missing required feature columns: {missing[:10]}")
        feat_cols = list(feature_cols)
    if LABEL_COL not in df.columns:
        raise ValueError(f"Training/evaluation CSV must contain a '{LABEL_COL}' column: {path}")

    X = np.nan_to_num(df[feat_cols].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    Y = np.zeros((len(df), N_TOTAL_LABELS), dtype=np.float32)

    for i, lab_str in enumerate(df[LABEL_COL].astype(str)):
        for tok in lab_str.split(LABEL_SEP):
            tok = tok.strip()
            if tok.isdigit() and 0 <= int(tok) < N_TOTAL_LABELS:
                Y[i, int(tok)] = 1.0

    if ID_COL in df.columns:
        ids = df[ID_COL].astype(str).values
    else:
        ids = np.array([f"sample_{i}" for i in range(len(df))], dtype=object)

    print(
        f"  {os.path.basename(path)}: {X.shape[0]} 样本 x {X.shape[1]} 特征 | "
        f"良性={int(Y[:, 0].sum())}  致病={int((Y[:, 0] == 0).sum())}"
    )
    return X, Y, ids, feat_cols

def build_feature_scaler(X_train):
    if not USE_FEATURE_SCALER or SCALER_TYPE == "none":
        return None
    if SCALER_TYPE == "standard":
        return StandardScaler()
    if SCALER_TYPE == "quantile":
        return QuantileTransformer(
            n_quantiles=min(QUANTILE_MAX_QUANTILES, X_train.shape[0]),
            output_distribution="normal",
            random_state=SEED,
        )
    raise ValueError(f"Unknown SCALER_TYPE={SCALER_TYPE}")

def apply_feature_scaler(X_tr, X_val, X_test):
    scaler = build_feature_scaler(X_tr)
    if scaler is None:
        return X_tr.astype(np.float32), X_val.astype(np.float32), X_test.astype(np.float32), None
    X_tr_s = scaler.fit_transform(X_tr).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)
    return X_tr_s, X_val_s, X_test_s, scaler

def load_label_prior(coder_path, cui2vec_path):
    print(f"[Prior] loading {coder_path}")
    coder = np.load(coder_path).astype(np.float32)
    print(f"        CODER shape: {coder.shape}")

    print(f"[Prior] loading {cui2vec_path}")
    cui2v = np.load(cui2vec_path).astype(np.float32)
    print(f"        cui2vec shape: {cui2v.shape}")

    assert coder.shape[0] == cui2v.shape[0] == N_TOTAL_LABELS

    coder = coder / (np.linalg.norm(coder, axis=1, keepdims=True) + 1e-12)
    cui2v = cui2v / (np.linalg.norm(cui2v, axis=1, keepdims=True) + 1e-12)

    prior = np.concatenate([coder, cui2v], axis=1)
    print(f"[Prior] concatenated label prior: {prior.shape}")
    return prior

def _make_loaders(X_tr, Y_tr, X_val=None, Y_val=None, shuffle=True):
    g = torch.Generator()
    g.manual_seed(SEED)

    train_ds = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(Y_tr, dtype=torch.float32),
    )
    train_ld = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        worker_init_fn=seed_worker,
        drop_last=False,
        generator=g,
    )

    if X_val is None:
        return train_ld, None

    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(Y_val, dtype=torch.float32),
    )
    val_ld = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        worker_init_fn=seed_worker,
        drop_last=False,
    )
    return train_ld, val_ld


def save_preprocessing_artifacts(scaler, feature_columns, output_dir):
    """Persist the exact fitted scaler and feature order used by the checkpoint."""
    os.makedirs(output_dir, exist_ok=True)
    scaler_path = os.path.join(output_dir, "feature_scaler.pkl")
    with open(scaler_path, "wb") as handle:
        pickle.dump(scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)
    columns_path = os.path.join(output_dir, "feature_columns.txt")
    with open(columns_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(map(str, feature_columns)) + "\n")
    print(f"[Artifacts] scaler saved: {scaler_path}")
    print(f"[Artifacts] feature order saved: {columns_path}")
    return scaler_path, columns_path


def read_feature_columns(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature-column file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        columns = [line.strip() for line in handle if line.strip()]
    if not columns:
        raise ValueError(f"Feature-column file is empty: {path}")
    return columns


def load_saved_scaler(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Saved scaler not found: {path}. Train with this project version to create it; "
            "the fitted preprocessing object cannot safely be guessed during inference."
        )
    with open(path, "rb") as handle:
        return pickle.load(handle)


def transform_with_saved_scaler(X, scaler):
    if scaler is None:
        return X.astype(np.float32)
    return scaler.transform(X).astype(np.float32)


def load_csv_data_for_inference(path, feature_cols=None):
    """Load data with optional labels; align columns to training order."""
    df = pd.read_csv(path, low_memory=False)
    ids = df[ID_COL].astype(str).values if ID_COL in df.columns else np.array(
        [f"sample_{i}" for i in range(len(df))], dtype=object
    )
    has_label = LABEL_COL in df.columns
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in [ID_COL, LABEL_COL]]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path} is missing {len(missing)} feature columns required by training. "
            f"First missing columns: {missing[:10]}"
        )
    X = np.nan_to_num(df[list(feature_cols)].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    Y = None
    if has_label:
        Y = np.zeros((len(df), N_TOTAL_LABELS), dtype=np.float32)
        for i, lab_str in enumerate(df[LABEL_COL].astype(str)):
            for tok in lab_str.split(LABEL_SEP):
                tok = tok.strip()
                if tok.isdigit() and 0 <= int(tok) < N_TOTAL_LABELS:
                    Y[i, int(tok)] = 1.0
        print(f"  {os.path.basename(path)}: {len(df)} samples x {X.shape[1]} features | Label column found")
    else:
        print(f"  {os.path.basename(path)}: {len(df)} samples x {X.shape[1]} features | prediction only (no Label column)")
    return X, Y, ids, list(feature_cols)
