"""Prediction probabilities, metrics, plots, and result serialization."""
import os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score, f1_score, roc_auc_score,
    precision_recall_curve, precision_score, recall_score, hamming_loss,
)
from torch.utils.data import DataLoader, TensorDataset
from .config import *
from .utils import seed_worker

@torch.no_grad()
def predict_proba(model, X, device=DEVICE, fuse="mean", tab_weight=DEFAULT_FUSION_WEIGHT):
    """Return fused, FT-tabular, label probabilities."""
    model.eval()
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32))
    ld = DataLoader(
        ds,
        batch_size=INFERENCE_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        worker_init_fn=seed_worker,
    )

    p_tab, p_lab = [], []
    for (xb,) in ld:
        xb = xb.to(device)
        logits_tab, logits_lab, _, _ = model(xb)
        p_tab.append(torch.sigmoid(logits_tab).cpu().numpy())
        p_lab.append(torch.sigmoid(logits_lab).cpu().numpy())

    p_tab = np.concatenate(p_tab, axis=0)
    p_lab = np.concatenate(p_lab, axis=0)

    if fuse == "tab":
        p_fused = p_tab
    elif fuse == "lab":
        p_fused = p_lab
    elif fuse == "geom_mean":
        p_fused = np.sqrt(np.clip(p_tab * p_lab, 0, 1))
    elif fuse == "weighted":
        p_fused = tab_weight * p_tab + (1.0 - tab_weight) * p_lab
    else:
        p_fused = DEFAULT_FUSION_WEIGHT * p_tab + (1.0 - DEFAULT_FUSION_WEIGHT) * p_lab
    return p_fused, p_tab, p_lab

def safe_average_precision_score(y, p):
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return average_precision_score(y, p)

def safe_roc_auc_score(y, p):
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return roc_auc_score(y, p)

def evaluate(Y_true, Y_proba, name, label_names=None):
    if label_names is None:
        label_names = [f"label_{i}" for i in range(Y_true.shape[1])]

    rows = []
    for j, ln in enumerate(label_names):
        y = Y_true[:, j]
        p = Y_proba[:, j]
        ap = safe_average_precision_score(y, p)
        roc = safe_roc_auc_score(y, p)
        f1 = f1_score(y, (p >= 0.5).astype(int), zero_division=0)
        rows.append({"label": ln, "n_pos": int(y.sum()), "AP": ap, "ROC_AUC": roc, "F1@0.5": f1})

    df = pd.DataFrame(rows)
    y_path = 1 - Y_true[:, 0]
    p_path = 1 - Y_proba[:, 0]
    path_ap = safe_average_precision_score(y_path, p_path)
    path_roc = safe_roc_auc_score(y_path, p_path)

    valid = ~df["AP"].isna()
    valid_roc = ~df["ROC_AUC"].isna()
    summary = {
        "model": name,
        "micro_AP": average_precision_score(Y_true.ravel(), Y_proba.ravel()),
        "macro_AP": df.loc[valid, "AP"].mean(),
        "macro_ROC": df.loc[valid_roc, "ROC_AUC"].mean(),
        "macro_F1": df["F1@0.5"].mean(),
        "pathogenicity_AP": path_ap,
        "pathogenicity_ROC": path_roc,
        "macro_AP_pheno_only": df.loc[1:, :].loc[valid.loc[1:], "AP"].mean(),
        "macro_ROC_pheno_only": df.loc[1:, :].loc[valid_roc.loc[1:], "ROC_AUC"].mean(),
    }
    return df, summary

def get_micro_ap(Y_true, P):
    return average_precision_score(Y_true.ravel(), P.ravel())

def get_macro_ap_pheno_only(Y_true, P):
    scores = []
    for j in range(1, Y_true.shape[1]):
        y = Y_true[:, j]
        if y.sum() == 0 or y.sum() == len(y):
            continue
        scores.append(average_precision_score(y, P[:, j]))
    return float(np.mean(scores)) if scores else float("nan")

def search_best_fusion_weight(Y_val, p_tab_val, p_lab_val):
    best_w = 1.0
    best_score = -1.0
    rows = []
    for w in np.linspace(0, 1, FUSION_SEARCH_STEPS):
        P = w * p_tab_val + (1.0 - w) * p_lab_val
        if FUSION_METRIC == "micro_ap":
            score = get_micro_ap(Y_val, P)
        else:
            score = get_macro_ap_pheno_only(Y_val, P)
        rows.append({"tab_weight": w, f"val_{FUSION_METRIC}": score})
        if score > best_score:
            best_score = score
            best_w = w
    pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, "fusion_weight_search.csv"), index=False)
    return best_w, best_score

def plot_pr_micro(Y_true, predictions_dict, save_path):
    plt.figure(figsize=(7, 6))
    for name, P in predictions_dict.items():
        prec, rec, _ = precision_recall_curve(Y_true.ravel(), P.ravel())
        ap = average_precision_score(Y_true.ravel(), P.ravel())
        plt.plot(rec, prec, label=f"{name}  (micro-AP={ap:.4f})", lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Micro-averaged PR curve")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def _label_columns(n_labels=N_LABELS):
    return [f"HPO_{j}" for j in range(n_labels)]

def _save_probability_table(ids, P, path, columns=None):
    """Save an ID + probability matrix table."""
    if columns is None:
        columns = _label_columns(P.shape[1])
    df = pd.DataFrame(P, columns=columns)
    df.insert(0, ID_COL, ids)
    df.to_csv(path, index=False, float_format="%.6f")
    return path

def build_per_label_metrics(Y_true, Y_proba, label_names=None):
    """
    Per-label AP table. Baseline is the positive prevalence of each label,
    i.e. the AP of a random ranking in expectation.
    """
    if label_names is None:
        label_names = _label_columns(Y_true.shape[1])

    Y_bin = (Y_proba >= 0.5).astype(int)
    rows = []
    for j, label in enumerate(label_names):
        y = Y_true[:, j]
        p = Y_proba[:, j]
        n_pos = int(y.sum())
        baseline = float(y.mean())
        ap = safe_average_precision_score(y, p)
        roc = safe_roc_auc_score(y, p)
        f1 = f1_score(y, Y_bin[:, j], zero_division=0)
        rows.append({
            "label": label,
            "label_index": j,
            "label_type": "benign" if j == 0 else "phenotype",
            "n_pos": n_pos,
            "baseline_prevalence": baseline,
            "AP": ap,
            "AP_lift_vs_baseline": (ap / baseline if baseline > 0 and not np.isnan(ap) else float("nan")),
            "ROC_AUC": roc,
            "F1@0.5": f1,
        })
    return pd.DataFrame(rows)

def print_phenotype_ap_baseline(df_per):
    """Print HPO_1..HPO_23 AP and baseline like the XGBoost 01_04 output style."""
    cols = ["label", "n_pos", "baseline_prevalence", "AP", "AP_lift_vs_baseline", "F1@0.5"]
    df_show = df_per[df_per["label_type"] == "phenotype"][cols].copy()
    print("\n每个表型的 AP 与 baseline（baseline = 该表型阳性率）:")
    with pd.option_context(
        "display.float_format", "{:.4f}".format,
        "display.max_rows", None,
        "display.width", 160,
    ):
        print(df_show.to_string(index=False))

def save_0104_style_outputs(ids, Y_true, Y_proba, output_dir, prefix="final"):
    """
    Save outputs in the 01_04 style using a selected probability matrix.

    Files:
      - proba_label_test02.csv:        ID + HPO_0..HPO_23 probabilities
      - phenotype_proba.csv:        ID + HPO_1..HPO_23 probabilities
      - pathogenicity_proba.csv:    ID + benign probability and 1-HPO_0
      - per_label_metrics.csv:      AP / baseline / ROC / F1 for all labels
      - per_phenotype_ap_baseline.csv: AP / baseline for HPO_1..HPO_23
      - evaluation_summary_0104_style.csv: overall micro/macro/pathogenicity metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    label_cols = _label_columns(Y_proba.shape[1])

    path_all = os.path.join(output_dir, "proba_label_test02.csv")
    _save_probability_table(ids, Y_proba, path_all, label_cols)

    path_pheno = os.path.join(output_dir, "phenotype_proba.csv")
    _save_probability_table(ids, Y_proba[:, 1:], path_pheno, label_cols[1:])

    patho = (1.0 - Y_proba[:, 0]).astype(np.float32)
    df_patho = pd.DataFrame({
        ID_COL: ids,
        "HPO_0_benign_prob": np.round(Y_proba[:, 0], 6),
        "pathogenicity_prob": np.round(patho, 6),
    })
    path_patho = os.path.join(output_dir, "pathogenicity_proba.csv")
    df_patho.to_csv(path_patho, index=False)

    df_per = build_per_label_metrics(Y_true, Y_proba, label_cols)
    path_per = os.path.join(output_dir, "per_label_metrics.csv")
    df_per.to_csv(path_per, index=False, float_format="%.6f")

    df_pheno_metrics = df_per[df_per["label_type"] == "phenotype"].copy()
    path_pheno_metrics = os.path.join(output_dir, "per_phenotype_ap_baseline.csv")
    df_pheno_metrics.to_csv(path_pheno_metrics, index=False, float_format="%.6f")

    Y_bin = (Y_proba >= 0.5).astype(int)
    y_path = (Y_true[:, 0] == 0).astype(int)
    summary = {
        "prediction_source": prefix,
        "micro_AP_all_24_labels": average_precision_score(Y_true.ravel(), Y_proba.ravel()),
        "micro_AP_baseline_all_24_labels": float(Y_true.ravel().mean()),
        "macro_AP_all_24_labels": float(df_per["AP"].mean(skipna=True)),
        "macro_AP_pheno_only": float(df_pheno_metrics["AP"].mean(skipna=True)),
        "macro_baseline_pheno_only": float(df_pheno_metrics["baseline_prevalence"].mean(skipna=True)),
        "hamming_loss@0.5": hamming_loss(Y_true, Y_bin),
        "micro_F1@0.5": f1_score(Y_true, Y_bin, average="micro", zero_division=0),
        "macro_F1@0.5": f1_score(Y_true, Y_bin, average="macro", zero_division=0),
        "micro_precision@0.5": precision_score(Y_true, Y_bin, average="micro", zero_division=0),
        "micro_recall@0.5": recall_score(Y_true, Y_bin, average="micro", zero_division=0),
        "pathogenicity_AP": safe_average_precision_score(y_path, patho),
        "pathogenicity_baseline": float(y_path.mean()),
        "pathogenicity_ROC_AUC": safe_roc_auc_score(y_path, patho),
        "mean_HPO_0_benign_prob": float(Y_proba[:, 0].mean()),
        "mean_pathogenicity_prob": float(patho.mean()),
        "n_pathogenicity_prob_gt_0.5": int((patho > 0.5).sum()),
        "n_samples": int(len(Y_true)),
    }
    path_summary = os.path.join(output_dir, "evaluation_summary_0104_style.csv")
    pd.DataFrame([summary]).to_csv(path_summary, index=False, float_format="%.6f")

    print(f"\n{'=' * 72}")
    print(f"01_04-style output summary ({prefix})")
    print(f"{'=' * 72}")
    print(f"  Micro AP 全24标签:           {summary['micro_AP_all_24_labels']:.6f}")
    print(f"  Micro baseline 全24标签:     {summary['micro_AP_baseline_all_24_labels']:.6f}")
    print(f"  Macro AP 全24标签:           {summary['macro_AP_all_24_labels']:.6f}")
    print(f"  Macro AP 表型1-23:           {summary['macro_AP_pheno_only']:.6f}")
    print(f"  Macro baseline 表型1-23:     {summary['macro_baseline_pheno_only']:.6f}")
    print(f"  Pathogenicity AP:            {summary['pathogenicity_AP']:.6f}")
    print(f"  Pathogenicity baseline:      {summary['pathogenicity_baseline']:.6f}")
    print(f"  Pathogenicity ROC-AUC:       {summary['pathogenicity_ROC_AUC']:.6f}")
    print(f"  HPO_0 均值（良性概率）:       {summary['mean_HPO_0_benign_prob']:.4f}")
    print(f"  致病概率均值/中位数/>0.5:    {patho.mean():.4f} / {np.median(patho):.4f} / {(patho > 0.5).sum()}/{len(patho)}")

    print_phenotype_ap_baseline(df_per)

    print(f"\n输出文件汇总 → {output_dir}")
    for fname in [
        "proba_label_test02.csv",
        "phenotype_proba.csv",
        "pathogenicity_proba.csv",
        "per_label_metrics.csv",
        "per_phenotype_ap_baseline.csv",
        "evaluation_summary_0104_style.csv",
    ]:
        fp = os.path.join(output_dir, fname)
        if os.path.exists(fp):
            print(f"  {fname:<42} {os.path.getsize(fp) / 1024:>8.1f} KB")

    return {
        "all_label_proba": path_all,
        "phenotype_proba": path_pheno,
        "pathogenicity_proba": path_patho,
        "per_label_metrics": path_per,
        "per_phenotype_metrics": path_pheno_metrics,
        "summary": path_summary,
    }

def plot_pr_micro_with_baseline(Y_true, predictions_dict, save_path):
    """Micro PR curve with baseline prevalence line, matching 01_04 style."""
    y_t = Y_true.ravel()
    baseline = float(y_t.mean())
    plt.figure(figsize=(7, 6))
    for name, P in predictions_dict.items():
        prec, rec, _ = precision_recall_curve(y_t, P.ravel())
        ap = average_precision_score(y_t, P.ravel())
        plt.plot(rec, prec, label=f"{name}  (micro-AP={ap:.4f})", lw=2)
    plt.axhline(baseline, lw=1, linestyle="--", label=f"Baseline prevalence={baseline:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Micro-averaged Precision–Recall Curve")
    plt.legend(loc="lower left")
    plt.xlim([0, 1])
    plt.ylim([0, 1.02])
    plt.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
