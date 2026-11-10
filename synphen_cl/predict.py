"""Run inference with a trained SynPhen-CL checkpoint.

Examples:
  python -m synphen_cl.predict
  python -m synphen_cl.predict --input path/to/new_features.csv --output outputs/new_run
"""
import argparse
import os
import pandas as pd
import numpy as np
import torch
from .config import *
from .utils import set_global_seed
from .data import load_label_prior, read_feature_columns, load_saved_scaler, transform_with_saved_scaler, load_csv_data_for_inference
from .model import _build_model
from .evaluation import (
    predict_proba, evaluate, _label_columns, _save_probability_table,
    save_0104_style_outputs, plot_pr_micro_with_baseline,
)


def save_prediction_only_outputs(ids, Y_proba, output_dir, prefix="prediction"):
    os.makedirs(output_dir, exist_ok=True)
    label_cols = _label_columns(Y_proba.shape[1])
    path_all = os.path.join(output_dir, f"all_label_proba_{prefix}.csv")
    _save_probability_table(ids, Y_proba, path_all, label_cols)
    path_pheno = os.path.join(output_dir, f"phenotype_proba_{prefix}.csv")
    _save_probability_table(ids, Y_proba[:, 1:], path_pheno, label_cols[1:])
    path_patho = os.path.join(output_dir, f"pathogenicity_proba_{prefix}.csv")
    df_patho = pd.DataFrame({
        ID_COL: ids,
        "HPO_0_benign_prob": np.round(Y_proba[:, 0], 6),
        "pathogenicity_prob": np.round(1.0 - Y_proba[:, 0], 6),
    })
    df_patho.to_csv(path_patho, index=False)
    print(f"[Saved prediction-only outputs] {output_dir}")
    for path in (path_all, path_pheno, path_patho):
        print(f"  {os.path.basename(path)}")


def main(input_path=None, output_dir=None):
    set_global_seed(SEED)
    input_path = input_path or PREDICT_CSV
    output_dir = output_dir or PREDICTION_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(OUTPUT_DIR, "v2p_fttransformer_dlcl_final.pt")
    run_config_path = os.path.join(OUTPUT_DIR, "run_config.csv")
    scaler_path = os.path.join(OUTPUT_DIR, "feature_scaler.pkl")
    feature_columns_path = os.path.join(OUTPUT_DIR, "feature_columns.txt")

    print("=" * 72)
    print("SynPhen-CL inference")
    print("=" * 72)
    print(f"[Device] {DEVICE}")
    print(f"[Input] {input_path}")
    print(f"[Checkpoint] {model_path}")
    print(f"[Output] {output_dir}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}. Run python -m synphen_cl.train first.")
    if not os.path.exists(run_config_path):
        raise FileNotFoundError(f"Training config not found: {run_config_path}")

    run_config = pd.read_csv(run_config_path, header=None, index_col=0).iloc[:, 0].to_dict()
    best_w = float(run_config.get("best_fusion_ft_weight", DEFAULT_FUSION_WEIGHT))
    feat_cols = read_feature_columns(feature_columns_path)
    scaler = load_saved_scaler(scaler_path)
    X_raw, Y_true, ids, _ = load_csv_data_for_inference(input_path, feature_cols=feat_cols)
    X = transform_with_saved_scaler(X_raw, scaler)

    label_prior = load_label_prior(CODER_NPY, CUI2VEC_NPY)
    model = _build_model(d_in=len(feat_cols), label_prior=label_prior, n_outputs=N_LABELS, device=DEVICE)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval()

    p_mean, p_tab, p_lab = predict_proba(model, X, fuse="mean")
    p_weighted, _, _ = predict_proba(model, X, fuse="weighted", tab_weight=best_w)
    out_cols = _label_columns(N_LABELS)
    _save_probability_table(ids, p_tab, os.path.join(output_dir, "proba_ft_tabular.csv"), out_cols)
    _save_probability_table(ids, p_lab, os.path.join(output_dir, "proba_label.csv"), out_cols)
    _save_probability_table(ids, p_mean, os.path.join(output_dir, "proba_fused_mean.csv"), out_cols)
    _save_probability_table(ids, p_weighted, os.path.join(output_dir, "proba_fused_weighted.csv"), out_cols)

    if Y_true is not None:
        rows = []
        for name, probs in [
            ("tabular", p_tab), ("label", p_lab), ("fused_mean", p_mean), (f"fused_weighted_w{best_w:.2f}", p_weighted)
        ]:
            _, summary = evaluate(Y_true, probs, name)
            rows.append(summary)
        pd.DataFrame(rows).to_csv(os.path.join(output_dir, "summary.csv"), index=False)
        save_0104_style_outputs(ids=ids, Y_true=Y_true, Y_proba=p_weighted if USE_WEIGHTED_FUSION else p_mean,
                                output_dir=output_dir, prefix=f"fused_weighted_w{best_w:.2f}")
        plot_pr_micro_with_baseline(Y_true, {
            "ft_tabular": p_tab, "label": p_lab, "fused_mean": p_mean, "fused_weighted": p_weighted
        }, os.path.join(output_dir, "pr_curve_micro.png"))
    else:
        save_prediction_only_outputs(ids, p_weighted if USE_WEIGHTED_FUSION else p_mean, output_dir)

    pd.Series({
        "input_path": input_path, "model_path": model_path, "best_fusion_ft_weight": best_w,
        "n_features": len(feat_cols), "n_samples": len(X), "has_label": Y_true is not None,
    }).to_csv(os.path.join(output_dir, "inference_config.csv"))
    print(f"[Done] Prediction outputs written to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SynPhen-CL prediction")
    parser.add_argument("--input", default=None, help="Input CSV; Label column is optional")
    parser.add_argument("--output", default=None, help="Prediction output directory")
    args = parser.parse_args()
    main(args.input, args.output)
