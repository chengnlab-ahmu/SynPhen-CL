"""Training loops for SynPhen-CL."""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from .config import *
from .data import _make_loaders
from .losses import compute_pos_weight, FocalDiceLoss, LabelSeqContrastiveLoss, get_current_eta_cl, _compute_total_loss
from .model import _build_model
from .evaluation import get_micro_ap, get_macro_ap_pheno_only

def make_optimizer(model):
    # Do not apply weight decay to tokenizer parameters, LayerNorm, or biases.
    def needs_wd(name):
        return all(x not in name for x in ["tokenizer", "norm", "bias"])

    # label_prior is always trainable in this version and uses a small dedicated LR.
    if hasattr(model, "label_prior") and isinstance(model.label_prior, nn.Parameter):
        label_prior_param_ids = {id(model.label_prior)}
    else:
        label_prior_param_ids = set()

    params_wd, params_no_wd = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in label_prior_param_ids:
            continue
        (params_wd if needs_wd(name) else params_no_wd).append(p)

    groups = [
        {"params": params_wd, "lr": LR, "weight_decay": WEIGHT_DECAY},
        {"params": params_no_wd, "lr": LR, "weight_decay": 0.0},
    ]
    if label_prior_param_ids:
        groups.append({"params": [model.label_prior], "lr": LABEL_PRIOR_LR, "weight_decay": 0.0})
    return torch.optim.AdamW(groups)

def train_model(X_tr, Y_tr, X_val, Y_val, label_prior, n_outputs,
                device=DEVICE):
    d_in = X_tr.shape[1]
    pos_weight = compute_pos_weight(Y_tr)
    pos_weight_t = torch.tensor(pos_weight, dtype=torch.float32, device=device)

    print("  [pos_weight] min={:.3f} max={:.3f} mean={:.3f}".format(
        pos_weight.min(), pos_weight.max(), pos_weight.mean()
    ))

    train_ld, val_ld = _make_loaders(X_tr, Y_tr, X_val, Y_val)
    model = _build_model(d_in, label_prior, n_outputs, device)
    opt = make_optimizer(model)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=LR_SCHEDULER_FACTOR, patience=LR_SCHEDULER_PATIENCE
    )

    focal_dice_fn = FocalDiceLoss().to(device)
    contrastive_fn = LabelSeqContrastiveLoss(temperature=CL_TEMPERATURE).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"  [model] trainable params: {n_params / 1e6:.2f}M  "
        f"(FT-d={FT_D_TOKEN} layers={FT_N_LAYERS} heads={FT_N_HEADS})"
    )

    best_score = -float("inf")
    best_epoch = 1
    best_state = None
    bad = 0
    history = []

    for ep in range(NUM_EPOCHS):
        epoch_1based = ep + 1
        eta_now = get_current_eta_cl(epoch_1based)

        model.train()
        tl = 0.0
        comp_sum = {}
        n_seen = 0

        for xb, yb in train_ld:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad(set_to_none=True)
            logits_tab, logits_lab, S_cl, label_proto = model(xb)
            loss, comp = _compute_total_loss(
                logits_tab, logits_lab, S_cl, label_proto, yb,
                focal_dice_fn, contrastive_fn, pos_weight_t,
                eta_cl=eta_now,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            opt.step()

            tl += loss.item() * xb.size(0)
            n_seen += xb.size(0)
            for k, v in comp.items():
                comp_sum[k] = comp_sum.get(k, 0.0) + v * xb.size(0)

        tl /= n_seen
        comp_avg = {k: v / n_seen for k, v in comp_sum.items()}

        model.eval()
        vl_cls = 0.0
        n_val = 0
        p_tab_val, p_lab_val = [], []
        with torch.no_grad():
            for xb, yb in val_ld:
                xb = xb.to(device)
                yb = yb.to(device)
                logits_tab, logits_lab, S_cl, label_proto = model(xb)

                # Validation classification loss excludes CL because CL is a
                # training regularizer and is not used at inference time.
                loss_cls, _ = _compute_total_loss(
                    logits_tab, logits_lab, S_cl, label_proto, yb,
                    focal_dice_fn, contrastive_fn, pos_weight_t,
                    eta_cl=0.0,
                )
                vl_cls += loss_cls.item() * xb.size(0)
                n_val += xb.size(0)
                p_tab_val.append(torch.sigmoid(logits_tab).cpu().numpy())
                p_lab_val.append(torch.sigmoid(logits_lab).cpu().numpy())

        vl_cls /= n_val
        p_tab_val = np.concatenate(p_tab_val, axis=0)
        p_lab_val = np.concatenate(p_lab_val, axis=0)
        p_val = DEFAULT_FUSION_WEIGHT * (p_tab_val + p_lab_val)

        if EARLY_STOP_METRIC == "micro_ap":
            val_score = get_micro_ap(Y_val, p_val)
        else:
            val_score = get_macro_ap_pheno_only(Y_val, p_val)

        # Scheduler minimizes, so use 1 - AP-like score.
        sched.step(1.0 - val_score)

        improved = val_score > best_score + EARLY_STOP_MIN_DELTA
        if improved:
            best_score = val_score
            best_epoch = epoch_1based
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1

        lr_now = opt.param_groups[0]["lr"]
        row = {
            "epoch": epoch_1based,
            "train_loss": tl,
            "val_cls_loss_no_cl": vl_cls,
            f"val_{EARLY_STOP_METRIC}": val_score,
            "lr": lr_now,
            **comp_avg,
        }
        history.append(row)

        print(
            f"  ep {epoch_1based:>3d}/{NUM_EPOCHS}  "
            f"train={tl:.4f}  val_cls={vl_cls:.4f}  "
            f"val_{EARLY_STOP_METRIC}={val_score:.4f}  "
            f"lr={lr_now:.1e}  eta_cl={eta_now:.4f}  "
            f"[sce={comp_avg['sce']:.3f} lce={comp_avg['lce']:.3f} "
            f"sml={comp_avg['sml']:.3f} cl={comp_avg['cl']:.3f}]  "
            f"{'*' if improved else ''}"
        )

        if epoch_1based >= MIN_EPOCHS_BEFORE_EARLY_STOP and bad >= PATIENCE:
            print(f"  early stop @ ep {epoch_1based}  (best={best_epoch})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    pd.DataFrame(history).to_csv(os.path.join(OUTPUT_DIR, "training_history_phase1.csv"), index=False)
    return model, best_score, best_epoch

def train_model_final(X_full, Y_full, label_prior, n_epochs, n_outputs,
                      device=DEVICE):
    d_in = X_full.shape[1]
    pos_weight = compute_pos_weight(Y_full)
    pos_weight_t = torch.tensor(pos_weight, dtype=torch.float32, device=device)

    train_ld, _ = _make_loaders(X_full, Y_full, None, None)
    model = _build_model(d_in, label_prior, n_outputs, device)
    opt = make_optimizer(model)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=LR_SCHEDULER_FACTOR, patience=LR_SCHEDULER_PATIENCE
    )

    focal_dice_fn = FocalDiceLoss().to(device)
    contrastive_fn = LabelSeqContrastiveLoss(temperature=CL_TEMPERATURE).to(device)

    best_train = float("inf")
    best_state = None

    for ep in range(n_epochs):
        model.train()
        tl = 0.0
        n_seen = 0

        for xb, yb in train_ld:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad(set_to_none=True)
            logits_tab, logits_lab, S_cl, label_proto = model(xb)
            eta_now = get_current_eta_cl(ep + 1)
            loss, _ = _compute_total_loss(
                logits_tab, logits_lab, S_cl, label_proto, yb,
                focal_dice_fn, contrastive_fn, pos_weight_t,
                eta_cl=eta_now,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            opt.step()

            tl += loss.item() * xb.size(0)
            n_seen += xb.size(0)

        tl /= n_seen
        sched.step(tl)

        if tl < best_train:
            best_train = tl
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"  [final] ep {ep + 1:>3d}/{n_epochs}  train_loss={tl:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model
