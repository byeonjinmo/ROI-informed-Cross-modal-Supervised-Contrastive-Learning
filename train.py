"""
Multimodal Depression Classification
T1 MRI (MedicalNet 3D ResNet) + rs-fMRI (GNN)

Leakage-free 5-fold evaluation over the FULL dataset (in-house or SRPBS),
matching manuscript Sec. 3.5 (participant-level stratified, 70:10:20 per fold):
  - Outer stratified K-fold: every subject is in exactly one test fold.
  - Inner split of the remaining subjects: inner-train (model + feature
    normalizer fit) / inner-val (early stopping + Youden threshold).
  - The outer test fold is evaluated once, with the frozen model and the
    threshold chosen on inner-val. Reported metrics are test-fold metrics
    (mean +/- SD across folds) and pooled out-of-fold metrics with bootstrap CI.
"""

import argparse
import copy
import json
import os
from datetime import datetime

# CUDA deterministic settings (set before other imports)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONHASHSEED'] = '42'

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

# Model imports
from models.resnet3d import MedicalNetResNet18, Simple3DCNN, load_medicalnet_pretrained
from models.gnn import GNNBackbone
from models.multimodal_fusion import MultimodalFusion, SingleModalityModel

# Shared utilities
from utils import (set_seed, fit_feature_normalizer, compute_metrics, find_optimal_threshold,
                   compute_statistical_tests, build_dataset, make_nested_splits,
                   summarize_oof)

REPORT_METRICS = ["auc", "aupr", "accuracy", "balanced_accuracy", "sensitivity", "specificity",
                  "ppv", "npv", "f1", "f1_weighted", "kappa", "mcc"]


def train_epoch(model, loader, device, criterion, optimizer,
                use_contrastive: bool = False, contrastive_weight: float = 0.1,
                supervised_contrastive: bool = False) -> float:
    """Train for one epoch with optional contrastive loss."""
    model.train()
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        if use_contrastive:
            contrastive_labels = batch.y.long() if supervised_contrastive else None
            logits, contrastive_loss = model(batch, return_contrastive=True,
                                              contrastive_labels=contrastive_labels)
            cls_loss = criterion(logits, batch.y)
            if contrastive_loss is not None:
                loss = cls_loss + contrastive_weight * contrastive_loss
            else:
                loss = cls_loss
        else:
            logits = model(batch)
            loss = criterion(logits, batch.y)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs

    n_samples = max(len(loader.dataset), 1)
    return total_loss / n_samples


def train_epoch_contrastive_only(model, loader, device, optimizer,
                                  supervised_contrastive: bool = False) -> float:
    """Stage A: Contrastive pretraining only (no classification loss)."""
    model.train()
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        contrastive_labels = batch.y.long() if supervised_contrastive else None

        _, contrastive_loss = model(batch, return_contrastive=True,
                                     contrastive_labels=contrastive_labels)

        if contrastive_loss is not None:
            contrastive_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += contrastive_loss.item() * batch.num_graphs

    n_samples = max(len(loader.dataset), 1)
    return total_loss / n_samples


def evaluate(model, loader, device) -> tuple:
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            prob = torch.sigmoid(logits)
            y_prob.extend(prob.cpu().tolist())
            y_true.extend(batch.y.cpu().tolist())
    return np.array(y_true), np.array(y_prob)


def main():
    parser = argparse.ArgumentParser(description="Multimodal Depression Classification")

    # Data paths
    parser.add_argument("--label-path", default="./data/label.csv")
    parser.add_argument("--outputs-root", default="./data/outputs")
    parser.add_argument("--t1-root", default="./data/T1_MNI")
    parser.add_argument("--adj-name", default="ADJ_abs_dens10.csv")
    parser.add_argument("--medicalnet-ckpt", default="./pretrain/resnet_18_23dataset.pth")

    # Model settings
    parser.add_argument("--modality", choices=["both", "t1", "fmri"], default="both",
                        help="Which modality to use (t1, fmri, or both for multimodal)")
    parser.add_argument("--t1-model", choices=["resnet", "simple"], default="resnet",
                        help="T1 backbone: resnet (MedicalNet pretrained) or simple (vanilla CNN)")
    parser.add_argument("--gnn-model", choices=["gcn", "sage", "gat"], default="gat")
    parser.add_argument("--gnn-hidden", type=int, default=256)
    parser.add_argument("--gat-heads", type=int, default=4)
    parser.add_argument("--fusion-hidden", type=int, default=256)
    parser.add_argument("--fusion-type",
                        choices=["concat", "gated", "attn", "attn_cls", "cross_attn", "cross_attn_uni", "ot"],
                        default="attn")
    parser.add_argument("--dropout", type=float, default=0.25)

    # OT (Sinkhorn) fusion settings
    parser.add_argument("--ot-eps", type=float, default=0.1,
                        help="Sinkhorn regularization (lower = sharper matching)")
    parser.add_argument("--ot-iters", type=int, default=30,
                        help="Number of Sinkhorn iterations")
    parser.add_argument("--ot-proj-dim", type=int, default=128,
                        help="OT projection dimension for cost matrix")
    parser.add_argument("--ot-row-normalize", action="store_true",
                        help="Row-normalize transport plan for stability")
    parser.add_argument("--atlas-path", default="./data/atlas/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii",
                        help="Path to Schaefer200 atlas for ROI pooling")
    parser.add_argument("--ot-t1-layer", choices=["layer2", "layer3"], default="layer2",
                        help="Which T1 CNN layer to extract features from")

    # Contrastive learning settings
    parser.add_argument("--use-contrastive", action="store_true",
                        help="Enable CLIP-style contrastive learning")
    parser.add_argument("--pretrain-contrastive", action="store_true",
                        help="Stage A: pretrain with contrastive loss only")
    parser.add_argument("--pretrain-epochs", type=int, default=20,
                        help="Number of epochs for contrastive pretraining")
    parser.add_argument("--contrastive-weight", type=float, default=0.1,
                        help="Weight for contrastive loss in joint training")
    parser.add_argument("--contrastive-tau", type=float, default=0.07,
                        help="Temperature for InfoNCE loss")
    parser.add_argument("--contrastive-queue", action="store_true",
                        help="Use queue memory for small batch negatives")
    parser.add_argument("--contrastive-queue-size", type=int, default=256,
                        help="Queue size for additional negatives")
    parser.add_argument("--supervised-contrastive", action="store_true",
                        help="Use supervised contrastive (same class = positive)")
    parser.add_argument("--contrastive-mode", choices=["unsupervised", "supervised", "hybrid"],
                        default="unsupervised",
                        help="Contrastive mode: unsupervised, supervised, or hybrid")
    parser.add_argument("--diagonal-weight", type=float, default=2.0,
                        help="Weight for diagonal (same subject) in hybrid mode")

    parser.add_argument("--fusion-dropout", type=float, default=0.3)
    parser.add_argument("--modality-dropout", type=float, default=0.1)

    # Training settings
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr-t1", type=float, default=5e-5)
    parser.add_argument("--lr-gnn", type=float, default=1e-4)
    parser.add_argument("--lr-fusion", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)

    # CV settings
    parser.add_argument("--folds", type=int, default=5,
                        help="Outer CV folds over the full dataset (each fold = held-out test)")
    parser.add_argument("--inner-val-size", type=float, default=0.125,
                        help="Fraction of each outer-train split used as inner validation "
                             "(early stopping + threshold selection)")
    parser.add_argument("--n-bootstrap", type=int, default=10000,
                        help="Bootstrap resamples for pooled out-of-fold CI")
    parser.add_argument("--seed", type=int, default=42)

    # Output
    parser.add_argument("--save-dir", default="./results")

    # Threshold
    parser.add_argument("--fixed-threshold", type=float, default=None,
                        help="Use fixed threshold instead of optimizing on inner-val (e.g., 0.5)")

    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("="*70)
    print("Multimodal Depression Classification")
    print("T1 (MedicalNet ResNet18) + rs-fMRI (GNN)")
    print(f"Nested {args.folds}-fold CV over full dataset "
          f"(inner val = {args.inner_val_size:.0%} of outer-train)")
    print("="*70)
    print(f"Device: {device}")

    # Load data (all subjects are used; none are left out unused)
    print("\nLoading multimodal data...")
    graphs, labels, subject_ids = build_dataset(
        args.label_path, args.outputs_root, args.t1_root, args.adj_name,
        modality=args.modality, return_ids=True
    )

    if not graphs:
        raise RuntimeError("No data loaded!")

    labels_array = np.array(labels)
    print(f"Total samples: {len(graphs)}")
    print(f"Class 0 (Normal): {sum(labels_array == 0)}")
    print(f"Class 1 (High-risk): {sum(labels_array == 1)}")

    os.makedirs(args.save_dir, exist_ok=True)

    fold_results = []
    # Out-of-fold predictions: each subject is predicted exactly once, by the
    # model whose outer test fold contains it.
    oof_prob = np.full(len(graphs), np.nan)
    oof_pred = np.full(len(graphs), -1, dtype=int)
    oof_fold = np.zeros(len(graphs), dtype=int)

    splits = make_nested_splits(labels_array, n_folds=args.folds,
                                inner_val_size=args.inner_val_size, seed=args.seed)
    for fold, train_idx, val_idx, test_idx in splits:
        print(f"\n{'-'*70}")
        print(f"Fold {fold}/{args.folds}")
        print(f"{'-'*70}")

        train_graphs_raw = [graphs[i] for i in train_idx]
        val_graphs_raw = [graphs[i] for i in val_idx]
        test_graphs_raw = [graphs[i] for i in test_idx]

        for name, idx in [("Train", train_idx), ("Val", val_idx), ("Test", test_idx)]:
            n_pos = int(labels_array[idx].sum())
            print(f"  {name:<5} {len(idx)} (pos={n_pos}, neg={len(idx) - n_pos}, "
                  f"ratio={n_pos / len(idx) * 100:.1f}%)")

        # Normalize graph features (fit on inner-train only; applied to val/test)
        norm_save_path = os.path.join(args.save_dir, f"feat_norm_fold{fold}.npz")
        normalizer = fit_feature_normalizer(train_graphs_raw, skip_cols=(1,), save_path=norm_save_path)
        train_graphs = [normalizer(g) for g in train_graphs_raw]
        val_graphs = [normalizer(g) for g in val_graphs_raw]
        test_graphs = [normalizer(g) for g in test_graphs_raw]

        # Save ReHo mean per ROI (inner-train only) for external validation
        reho_per_subject = torch.stack([g.x[:, 0] for g in train_graphs_raw])
        reho_mean_per_roi = reho_per_subject.mean(dim=0).numpy()
        np.save(os.path.join(args.save_dir, f"train_reho_mean_200_fold{fold}.npy"), reho_mean_per_roi)

        # Count classes for pos_weight
        y_train = torch.tensor([g.y.item() for g in train_graphs])
        pos = (y_train == 1).sum().item()
        neg = (y_train == 0).sum().item()

        train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_graphs, batch_size=args.batch_size, shuffle=False)

        # Build model based on modality
        if args.modality == "t1":
            if args.t1_model == "simple":
                t1_backbone = Simple3DCNN(dropout=args.dropout)
                print("[T1-only] Using Simple3DCNN (no pretrain)")
            else:
                t1_backbone = MedicalNetResNet18(dropout=args.dropout)
                load_medicalnet_pretrained(t1_backbone, args.medicalnet_ckpt)
                t1_backbone.freeze_bn()
                print("[T1-only] Using MedicalNet ResNet-18 (pretrained)")

            model = SingleModalityModel(
                backbone=t1_backbone,
                modality="t1",
                hidden=args.fusion_hidden,
                dropout=args.fusion_dropout
            ).to(device)
            param_groups = [
                {"params": model.backbone.parameters(),
                 "lr": args.lr_t1, "weight_decay": args.weight_decay},
                {"params": model.classifier.parameters(),
                 "lr": args.lr_fusion, "weight_decay": args.weight_decay},
            ]
            use_contrastive_training = False

        elif args.modality == "fmri":
            gnn = GNNBackbone(
                in_dim=train_graphs[0].num_node_features,
                hidden=args.gnn_hidden,
                model_type=args.gnn_model,
                dropout=args.dropout,
                heads=args.gat_heads
            )
            model = SingleModalityModel(
                backbone=gnn,
                modality="fmri",
                hidden=args.fusion_hidden,
                dropout=args.fusion_dropout
            ).to(device)
            param_groups = [
                {"params": model.backbone.parameters(),
                 "lr": args.lr_gnn, "weight_decay": args.weight_decay},
                {"params": model.classifier.parameters(),
                 "lr": args.lr_fusion, "weight_decay": args.weight_decay},
            ]
            use_contrastive_training = False

        else:  # both - multimodal
            gnn = GNNBackbone(
                in_dim=train_graphs[0].num_node_features,
                hidden=args.gnn_hidden,
                model_type=args.gnn_model,
                dropout=args.dropout,
                heads=args.gat_heads
            )

            t1_backbone = MedicalNetResNet18(dropout=args.dropout)
            load_medicalnet_pretrained(t1_backbone, args.medicalnet_ckpt)
            t1_backbone.freeze_bn()

            # Determine contrastive mode
            if args.supervised_contrastive:
                effective_contrastive_mode = args.contrastive_mode if args.contrastive_mode != "unsupervised" else "supervised"
            else:
                effective_contrastive_mode = args.contrastive_mode

            model = MultimodalFusion(
                gnn=gnn,
                t1=t1_backbone,
                fusion_hidden=args.fusion_hidden,
                fusion_dropout=args.fusion_dropout,
                fusion_type=args.fusion_type,
                modality_dropout=args.modality_dropout,
                ot_eps=args.ot_eps,
                ot_iters=args.ot_iters,
                ot_proj_dim=args.ot_proj_dim,
                ot_row_normalize=args.ot_row_normalize,
                atlas_path=args.atlas_path,
                ot_t1_layer=args.ot_t1_layer,
                use_contrastive=args.use_contrastive or args.pretrain_contrastive,
                contrastive_tau=args.contrastive_tau,
                contrastive_queue=args.contrastive_queue,
                contrastive_queue_size=args.contrastive_queue_size,
                contrastive_mode=effective_contrastive_mode,
                diagonal_weight=args.diagonal_weight,
            ).to(device)

            param_groups = [
                {"params": model.t1.parameters(),
                 "lr": args.lr_t1, "weight_decay": args.weight_decay},
                {"params": model.gnn.parameters(),
                 "lr": args.lr_gnn, "weight_decay": args.weight_decay},
                {"params": model.classifier.parameters(),
                 "lr": args.lr_fusion, "weight_decay": args.weight_decay},
            ]

            if args.fusion_type == "ot":
                param_groups.extend([
                    {"params": model.sinkhorn_ot.parameters(),
                     "lr": args.lr_fusion, "weight_decay": args.weight_decay},
                    {"params": model.ot_fusion_mlp.parameters(),
                     "lr": args.lr_fusion, "weight_decay": args.weight_decay},
                ])

            if args.use_contrastive or args.pretrain_contrastive:
                param_groups.append({
                    "params": model.contrastive.parameters(),
                    "lr": args.lr_fusion, "weight_decay": args.weight_decay
                })

            use_contrastive_training = args.use_contrastive or args.pretrain_contrastive

        # Loss with class weight
        pos_weight = torch.tensor([neg / max(pos, 1)], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.AdamW(param_groups)

        # Stage A: Contrastive Pretraining (optional, multimodal only)
        if args.pretrain_contrastive and args.modality == "both":
            if effective_contrastive_mode == "hybrid":
                mode_str = f"Hybrid (diag_weight={args.diagonal_weight})"
            elif effective_contrastive_mode == "supervised" or args.supervised_contrastive:
                mode_str = "Supervised"
            else:
                mode_str = "Unsupervised"

            use_labels_for_contrastive = effective_contrastive_mode in ["supervised", "hybrid"] or args.supervised_contrastive

            print(f"\n  [Stage A] {mode_str} Contrastive Pretraining ({args.pretrain_epochs} epochs)")
            for epoch in range(1, args.pretrain_epochs + 1):
                contrast_loss = train_epoch_contrastive_only(
                    model, train_loader, device, optimizer,
                    supervised_contrastive=use_labels_for_contrastive
                )
                if epoch % 5 == 0:
                    print(f"    Pretrain Epoch {epoch}: Contrastive Loss={contrast_loss:.4f}")
            print("  [Stage A] Contrastive pretraining complete")

        # Stage B: Supervised Finetuning
        if args.pretrain_contrastive and args.modality == "both":
            print(f"\n  [Stage B] Supervised Finetuning ({args.epochs} epochs)")

        best_state = None
        best_score = -1e9
        patience_counter = 0

        for epoch in range(1, args.epochs + 1):
            use_joint_contrastive = (use_contrastive_training and
                                     args.use_contrastive and not args.pretrain_contrastive)
            use_labels_for_joint = (effective_contrastive_mode in ["supervised", "hybrid"]
                                    if args.modality == "both" else False) or args.supervised_contrastive
            loss = train_epoch(
                model, train_loader, device, criterion, optimizer,
                use_contrastive=use_joint_contrastive,
                contrastive_weight=args.contrastive_weight,
                supervised_contrastive=use_labels_for_joint
            )

            y_val_true, y_val_prob = evaluate(model, val_loader, device)
            val_metrics = compute_metrics(y_val_true, y_val_prob)

            score = val_metrics["auc"] if not np.isnan(val_metrics["auc"]) else val_metrics["accuracy"]

            if score > best_score + 1e-4:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 10 == 0:
                print(f"    Epoch {epoch}: Loss={loss:.4f}, Val AUC={val_metrics['auc']:.4f}")

            if patience_counter >= args.patience:
                print(f"    Early stopping at epoch {epoch}")
                break

        # Load best model
        if best_state:
            model.load_state_dict(best_state)

        # Threshold selected on inner-val only (never on the test fold)
        y_val_true, y_val_prob = evaluate(model, val_loader, device)
        if args.fixed_threshold is not None:
            thr = args.fixed_threshold
        else:
            thr = find_optimal_threshold(y_val_true, y_val_prob)
        val_metrics = compute_metrics(y_val_true, y_val_prob, threshold=thr)

        # Single final evaluation on the untouched outer test fold
        y_test_true, y_test_prob = evaluate(model, test_loader, device)
        test_metrics = compute_metrics(y_test_true, y_test_prob, threshold=thr)

        oof_prob[test_idx] = y_test_prob
        oof_pred[test_idx] = (y_test_prob >= thr).astype(int)
        oof_fold[test_idx] = fold

        print(f"\n  Fold {fold} (threshold={thr:.3f} from inner-val, val AUC={val_metrics['auc']:.4f})")
        print(f"  Test: AUC={test_metrics['auc']:.4f}, AUPR={test_metrics['aupr']:.4f}, BAcc={test_metrics['balanced_accuracy']:.4f}")
        print(f"        Sens={test_metrics['sensitivity']:.4f}, Spec={test_metrics['specificity']:.4f}")
        print(f"        PPV={test_metrics['ppv']:.4f}, NPV={test_metrics['npv']:.4f}")
        print(f"        Kappa={test_metrics['kappa']:.4f}, MCC={test_metrics['mcc']:.4f}")

        fold_result = {"fold": fold, "threshold": thr,
                       "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
                       "val_auc": val_metrics["auc"]}  # model-selection score, for reference only
        fold_result.update({f"test_{k}": test_metrics[k] for k in REPORT_METRICS})
        fold_results.append(fold_result)

        # Save fold model
        torch.save(best_state, os.path.join(args.save_dir, f"fold_{fold}_model.pt"))

    assert not np.isnan(oof_prob).any(), "Some subjects never appeared in a test fold"

    # Results Summary (test folds only)
    print(f"\n{'='*100}")
    print(f"NESTED {args.folds}-FOLD CV: OUTER TEST-FOLD RESULTS")
    print(f"{'='*100}")

    print(f"\n{'Fold':<6}{'AUC':<8}{'AUPR':<8}{'BAcc':<8}{'Sens':<8}{'Spec':<8}{'PPV':<8}{'NPV':<8}{'Kappa':<8}{'MCC':<8}")
    print("-" * 96)
    table_cols = ["auc", "aupr", "balanced_accuracy", "sensitivity", "specificity", "ppv", "npv", "kappa", "mcc"]
    for r in fold_results:
        print(f"{r['fold']:<6}" + "".join(f"{r[f'test_{k}']:<8.4f}" for k in table_cols))

    summary = {}
    for k in REPORT_METRICS:
        vals = [r[f"test_{k}"] for r in fold_results]
        summary[k] = (float(np.mean(vals)), float(np.std(vals)))

    print("-" * 96)
    print(f"{'Mean':<6}" + "".join(f"{summary[k][0]:<8.4f}" for k in table_cols))
    print(f"{'Std':<6}" + "".join(f"{summary[k][1]:<8.4f}" for k in table_cols))

    # Pooled out-of-fold metrics over all subjects
    pooled = summarize_oof(labels_array, oof_prob, oof_pred, n_boot=args.n_bootstrap, seed=args.seed)

    display_names = {"auc": "AUC", "aupr": "AUPR", "accuracy": "Accuracy",
                     "balanced_accuracy": "Balanced Acc", "sensitivity": "Sensitivity",
                     "specificity": "Specificity", "ppv": "PPV (Precision)", "npv": "NPV",
                     "f1": "F1-Score", "f1_weighted": "F1 (weighted)",
                     "kappa": "Cohen's Kappa", "mcc": "MCC"}

    print(f"\n{'='*100}")
    print(f"SUMMARY (leak-free; N={len(graphs)} subjects, each predicted once as test)")
    print(f"{'='*100}")
    print(f"  {'Metric':<18}{'Test folds mean+/-SD':<24}{'Pooled OOF':<12}")
    for k in REPORT_METRICS:
        print(f"  {display_names[k]:<18}{summary[k][0]:.4f} +/- {summary[k][1]:.4f}      {pooled[k]:.4f}")
    print(f"  Pooled AUC 95% CI (bootstrap): [{pooled['auc_ci95'][0]:.4f}, {pooled['auc_ci95'][1]:.4f}]")
    print(f"  Pooled AUPR 95% CI (bootstrap): [{pooled['aupr_ci95'][0]:.4f}, {pooled['aupr_ci95'][1]:.4f}]")
    print(f"{'='*100}")

    # Statistical validation
    stat_tests = compute_statistical_tests(fold_results, prefix="test")
    print(f"\n{'='*100}")
    print("STATISTICAL VALIDATION (One-sample t-test on test folds, H0: metric = chance level)")
    print(f"{'='*100}")
    print(f"  {'Metric':<20} {'Mean+/-SD':<18} {'t-stat':>8} {'p-value':>10} {'Sig.':>6}")
    print(f"  {'-'*62}")
    for metric_key, test_result in stat_tests.items():
        k = metric_key[len("test_"):]
        name = display_names[k]
        mean, std = summary[k]
        p = test_result["p_value"]
        if p < 0.001:
            sig = "***"
        elif p < 0.01:
            sig = "**"
        elif p < 0.05:
            sig = "*"
        else:
            sig = "n.s."
        print(f"  {name:<20} {mean:.4f}+/-{std:.4f}    {test_result['t_statistic']:>8.2f} {p:>10.4f} {sig:>6}")
    print(f"  {'-'*62}")
    print(f"  Significance: *** p<0.001, ** p<0.01, * p<0.05, n.s. not significant")
    print(f"  Note: df={args.folds - 1} ({args.folds}-fold CV), one-sided test (H1: metric > chance)")
    print(f"{'='*100}")

    # Save results. cv_<metric>_mean/std are OUTER TEST-FOLD statistics
    # (key names kept for compatibility with scripts/ that read results.json).
    results = {
        "config": vars(args),
        "evaluation": "nested_cv_full_dataset",
        "n_subjects": len(graphs),
        "cv_results": fold_results,
        "pooled_oof": {k: (list(v) if isinstance(v, tuple) else float(v)) for k, v in pooled.items()},
        "statistical_tests": stat_tests,
        "timestamp": datetime.now().isoformat()
    }
    for k in REPORT_METRICS:
        results[f"cv_{k}_mean"], results[f"cv_{k}_std"] = summary[k]

    with open(os.path.join(args.save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    pd.DataFrame({
        "subject_id": subject_ids,
        "fold": oof_fold,
        "y_true": labels_array,
        "y_prob": oof_prob,
        "y_pred": oof_pred,
    }).to_csv(os.path.join(args.save_dir, "oof_predictions.csv"), index=False)

    print(f"\nResults saved to: {args.save_dir}")

    return results


if __name__ == "__main__":
    main()
