"""Train SynPhen-CL and evaluate on the configured held-out test set.

Run from the project root: python -m synphen_cl.train
"""
import os
import time
import numpy as np
import pandas as pd
import torch
from .config import *
from .utils import set_global_seed
from .data import load_csv_data, load_label_prior, apply_feature_scaler, save_preprocessing_artifacts
from .training import train_model, train_model_final
from .evaluation import (
    predict_proba, evaluate, search_best_fusion_weight, _label_columns,
    _save_probability_table, save_0104_style_outputs, plot_pr_micro_with_baseline,
)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    set_global_seed(SEED)
    print("=" * 72)
    print("V2P-DLCL v6  (FT-Transformer + Label-table Fusion + Early Weak Global-Prototype CL)")
    print("=" * 72)
    print(f"[Device] {DEVICE}")
    print(
        f"[Config] SEED={SEED}, LR={LR}, ETA_CL={ETA_CL}, CL_WARMUP_START={CL_WARMUP_START_EPOCH}, "
        f"CL_WARMUP_EPOCHS={CL_WARMUP_EPOCHS}, MIN_EPOCHS={MIN_EPOCHS_BEFORE_EARLY_STOP}, EARLY_STOP_METRIC={EARLY_STOP_METRIC}, "
        f"USE_PHASE2_RETRAIN={USE_PHASE2_RETRAIN}, LABEL_PRIOR=CODER+cui2vec(trainable), "
        f"SCALER_TYPE={SCALER_TYPE}"
    )

    print("\n[1] Load tabular data")
    X_train_full_raw, Y_train_full, train_ids_full, feat_cols = load_csv_data(TRAIN_CSV)
    X_test_raw, Y_test, test_ids, _ = load_csv_data(TEST_CSV, feature_cols=feat_cols)
    print(f"  # features: {len(feat_cols)}   # labels: {N_LABELS}")

    print("[2] Load label semantic prior (CODER + cui2vec)")
    print("    External label prior is always enabled and trainable in this version.")
    label_prior = load_label_prior(CODER_NPY, CUI2VEC_NPY)

    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(X_train_full_raw))
    n_val = max(1, int(len(idx) * VAL_FRAC))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    X_tr_raw, Y_tr = X_train_full_raw[tr_idx], Y_train_full[tr_idx]
    X_val_raw, Y_val = X_train_full_raw[val_idx], Y_train_full[val_idx]
    print(f"\n[3] Train/val split: {len(tr_idx)} / {len(val_idx)}")

    print("\n[3.1] Feature preprocessing")
    X_tr, X_val, X_test, scaler = apply_feature_scaler(X_tr_raw, X_val_raw, X_test_raw)
    scaler_final = scaler
    if scaler is None:
        print("  scaler: none")
    else:
        print(f"  scaler: {scaler.__class__.__name__}")

    print("\n" + "=" * 72)
    print("Phase 1 — Train with early stopping")
    print("=" * 72)
    t0 = time.time()
    model_best, best_val_score, best_epoch = train_model(
        X_tr, Y_tr, X_val, Y_val, label_prior=label_prior, n_outputs=N_LABELS
    )
    print(
        f"  Phase-1 time: {time.time() - t0:.1f}s  "
        f"best_val_{EARLY_STOP_METRIC}={best_val_score:.4f}  best_epoch={best_epoch}"
    )

    if USE_PHASE2_RETRAIN:
        print("\n" + "=" * 72)
        print("Phase 2 — Optional full-train retraining")
        print("=" * 72)
        # Refit preprocessing on the full training set for phase-2.
        X_train_full, X_val_phase2, X_test_phase2, scaler_final = apply_feature_scaler(
            X_train_full_raw, X_val_raw, X_test_raw
        )
        t0 = time.time()
        model_final = train_model_final(
            X_train_full, Y_train_full, label_prior=label_prior,
            n_epochs=best_epoch, n_outputs=N_LABELS,
        )
        X_test_eval = X_test_phase2
        X_val_eval = X_val_phase2
        print(f"  Phase-2 time: {time.time() - t0:.1f}s")
    else:
        print("\n[Phase 2] Skipped. Use Phase-1 best validation model for test.")
        model_final = model_best
        X_test_eval = X_test
        X_val_eval = X_val

    print("\n[4] Validation prediction for fusion weight search")
    _, p_tab_val, p_lab_val = predict_proba(model_final, X_val_eval, fuse="mean")

    if USE_WEIGHTED_FUSION:
        best_w, best_score = search_best_fusion_weight(Y_val, p_tab_val, p_lab_val)
        print(
            f"  [fusion] best ft_weight on val = {best_w:.2f}, "
            f"val_{FUSION_METRIC} = {best_score:.4f}"
        )
    else:
        best_w = DEFAULT_FUSION_WEIGHT

    print("\n[5] Inference on test set")
    proba_mean, proba_tab, proba_lab = predict_proba(model_final, X_test_eval, fuse="mean")
    proba_weighted, _, _ = predict_proba(
        model_final, X_test_eval, fuse="weighted", tab_weight=best_w
    )

    print("\n" + "=" * 72)
    print("Evaluate")
    print("=" * 72)

    df_tab, sum_tab = evaluate(Y_test, proba_tab, "FTTransformer-DLCL_tabular")
    df_lab, sum_lab = evaluate(Y_test, proba_lab, "FTTransformer-DLCL_label")
    df_mean, sum_mean = evaluate(Y_test, proba_mean, "FTTransformer-DLCL_fused_mean")
    df_weighted, sum_weighted = evaluate(
        Y_test, proba_weighted, f"FTTransformer-DLCL_fused_weighted_w{best_w:.2f}"
    )

    summary = pd.DataFrame([sum_tab, sum_lab, sum_mean, sum_weighted])
    print("\n=== Summary (test set) ===")
    with pd.option_context(
        "display.float_format", "{:.4f}".format,
        "display.max_columns", None,
        "display.width", 240,
    ):
        print(summary.to_string(index=False))

    merged = df_tab.rename(columns={
        "AP": "AP_ft",
        "ROC_AUC": "ROC_ft",
        "F1@0.5": "F1_ft",
    }).merge(
        df_lab.rename(columns={
            "AP": "AP_lab",
            "ROC_AUC": "ROC_lab",
            "F1@0.5": "F1_lab",
        })[["label", "AP_lab", "ROC_lab", "F1_lab"]],
        on="label",
    ).merge(
        df_mean.rename(columns={
            "AP": "AP_mean",
            "ROC_AUC": "ROC_mean",
            "F1@0.5": "F1_mean",
        })[["label", "AP_mean", "ROC_mean", "F1_mean"]],
        on="label",
    ).merge(
        df_weighted.rename(columns={
            "AP": "AP_weighted",
            "ROC_AUC": "ROC_weighted",
            "F1@0.5": "F1_weighted",
        })[["label", "AP_weighted", "ROC_weighted", "F1_weighted"]],
        on="label",
    )

    # Keep the original four-branch comparison table, but save it under a separate name.
    merged.to_csv(os.path.join(OUTPUT_DIR, "per_label_metrics_by_branch.csv"), index=False)
    summary.to_csv(os.path.join(OUTPUT_DIR, "summary.csv"), index=False)

    # Save branch-specific probabilities with sequence_id and HPO_* column names.
    out_cols = _label_columns(N_LABELS)
    _save_probability_table(test_ids, proba_tab, os.path.join(OUTPUT_DIR, "proba_ft_tabular.csv"), out_cols)
    _save_probability_table(test_ids, proba_lab, os.path.join(OUTPUT_DIR, "proba_label.csv"), out_cols)
    _save_probability_table(test_ids, proba_mean, os.path.join(OUTPUT_DIR, "proba_fused_mean.csv"), out_cols)
    _save_probability_table(test_ids, proba_weighted, os.path.join(OUTPUT_DIR, "proba_fused_weighted.csv"), out_cols)

    # 01_04-style outputs use the final selected probability matrix.
    # If weighted fusion is enabled, this is proba_weighted; otherwise best_w=0.5.
    final_name = f"fused_weighted_w{best_w:.2f}" if USE_WEIGHTED_FUSION else "fused_mean"
    final_proba = proba_weighted if USE_WEIGHTED_FUSION else proba_mean
    save_0104_style_outputs(
        ids=test_ids,
        Y_true=Y_test,
        Y_proba=final_proba,
        output_dir=OUTPUT_DIR,
        prefix=final_name,
    )

    plot_pr_micro_with_baseline(
        Y_test,
        {
            "ft_tabular": proba_tab,
            "label": proba_lab,
            "fused_mean": proba_mean,
            "fused_weighted": proba_weighted,
        },
        os.path.join(OUTPUT_DIR, "pr_curve_micro.png"),
    )

    save_preprocessing_artifacts(scaler_final, feat_cols, OUTPUT_DIR)

    torch.save(
        model_final.state_dict(),
        os.path.join(OUTPUT_DIR, "v2p_fttransformer_dlcl_final.pt"),
    )

    config = {
        "SEED": SEED,
        "VAL_FRAC": VAL_FRAC,
        "USE_FEATURE_SCALER": USE_FEATURE_SCALER,
        "USE_PHASE2_RETRAIN": USE_PHASE2_RETRAIN,
        "DETERMINISTIC_ALGORITHMS": True,
        "LR": LR,
        "LABEL_PRIOR_SOURCE": "CODER+cui2vec",
        "LABEL_PRIOR_IS_TRAINABLE": True,
        "LABEL_PRIOR_LR": LABEL_PRIOR_LR,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "BATCH_SIZE": BATCH_SIZE,
        "NUM_EPOCHS": NUM_EPOCHS,
        "PATIENCE": PATIENCE,
        "MIN_EPOCHS_BEFORE_EARLY_STOP": MIN_EPOCHS_BEFORE_EARLY_STOP,
        "LR_SCHEDULER_PATIENCE": LR_SCHEDULER_PATIENCE,
        "FT_D_TOKEN": FT_D_TOKEN,
        "FT_N_LAYERS": FT_N_LAYERS,
        "FT_N_HEADS": FT_N_HEADS,
        "FT_D_FFN_FACTOR": FT_D_FFN_FACTOR,
        "FT_ATTN_DROPOUT": FT_ATTN_DROPOUT,
        "FT_FFN_DROPOUT": FT_FFN_DROPOUT,
        "D_MODEL": D_MODEL,
        "N_LABEL_FUSION_LAYERS": N_LABEL_FUSION_LAYERS,
        "ETA_CL": ETA_CL,
        "CL_TEMPERATURE": CL_TEMPERATURE,
        "CL_WARMUP_START_EPOCH": CL_WARMUP_START_EPOCH,
        "CL_WARMUP_EPOCHS": CL_WARMUP_EPOCHS,
        "CONTRAST_INCLUDE_BENIGN": CONTRAST_INCLUDE_BENIGN,
        "EARLY_STOP_METRIC": EARLY_STOP_METRIC,
        "ALPHA_CE": ALPHA_CE,
        "BETA_FOCAL": BETA_FOCAL,
        "SCALER_TYPE": SCALER_TYPE,
        "FUSION_METRIC": FUSION_METRIC,
        "USE_PHASE2_RETRAIN": USE_PHASE2_RETRAIN,
        "best_epoch": best_epoch,
        f"best_val_{EARLY_STOP_METRIC}": best_val_score,
        "best_fusion_ft_weight": best_w,
    }
    pd.Series(config).to_csv(os.path.join(OUTPUT_DIR, "run_config.csv"))

    print(f"\n[Done] Results written to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
