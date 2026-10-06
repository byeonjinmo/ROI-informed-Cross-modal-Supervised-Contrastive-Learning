"""
Single Modality Depression Classification
T1 MRI only OR rs-fMRI only
Leakage-free nested CV over the full dataset (same protocol as train.py):
outer stratified K-fold = held-out test; inner-val (from outer-train) for
early stopping + threshold; metrics reported on outer test folds only.

Usage:
    # T1 with MedicalNet ResNet-18 (pretrained)
    python train_single_modality.py --modality t1 --t1-model resnet --save-dir results_t1_resnet

    # T1 with Simple 3D CNN (no pretrain)
    python train_single_modality.py --modality t1 --t1-model simple --save-dir results_t1_simple

    # fMRI with GAT
    python train_single_modality.py --modality fmri --gnn-model gat --save-dir results_fmri_gat
"""

import argparse
import copy
import json
import os
from datetime import datetime

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONHASHSEED'] = '42'

import numpy as np
import torch
import pandas as pd
import torch.nn as nn
from torch_geometric.loader import DataLoader

from models.resnet3d import MedicalNetResNet18, Simple3DCNN, load_medicalnet_pretrained
from models.gnn import GNNBackbone
from models.multimodal_fusion import SingleModalityModel

# Shared utilities
from utils import (set_seed, fit_feature_normalizer, compute_metrics,
                   find_optimal_threshold, build_dataset, make_nested_splits,
                   summarize_oof)

REPORT_METRICS = ["auc", "aupr", "accuracy", "balanced_accuracy", "sensitivity", "specificity",
                  "ppv", "npv", "f1", "f1_weighted", "kappa", "mcc"]


def train_epoch(model, loader, device, criterion, optimizer) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        loss = criterion(logits, batch.y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / max(len(loader.dataset), 1)


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
    parser = argparse.ArgumentParser(description="Single Modality Depression Classification")

    # Data paths
    parser.add_argument("--label-path", default="./data/label.csv")
    parser.add_argument("--outputs-root", default="./data/outputs")
    parser.add_argument("--t1-root", default="./data/T1_MNI")
    parser.add_argument("--adj-name", default="ADJ_abs_dens10.csv")
    parser.add_argument("--medicalnet-ckpt", default="./pretrain/resnet_18_23dataset.pth")

    # Model settings
    parser.add_argument("--modality", choices=["t1", "fmri"], required=True)
    parser.add_argument("--t1-model", choices=["resnet", "simple"], default="resnet")
    parser.add_argument("--gnn-model", choices=["gcn", "sage", "gat"], default="gat")
    parser.add_argument("--gnn-hidden", type=int, default=256)
    parser.add_argument("--gat-heads", type=int, default=4)
    parser.add_argument("--fusion-hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--fusion-dropout", type=float, default=0.3)

    # Training settings
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-classifier", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)

    # CV settings
    parser.add_argument("--folds", type=int, default=5,
                        help="Outer CV folds over the full dataset (each fold = held-out test)")
    parser.add_argument("--inner-val-size", type=float, default=0.125,
                        help="Fraction of outer-train used for early stopping + threshold")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--fixed-threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)

    # Output
    parser.add_argument("--save-dir", default="./results_single")

    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("="*70)
    print(f"Single Modality Depression Classification: {args.modality.upper()}")
    print("="*70)

    # Load data
    graphs, labels, subject_ids = build_dataset(
        args.label_path, args.outputs_root, args.t1_root, args.adj_name,
        modality=args.modality, return_ids=True
    )

    if not graphs:
        raise RuntimeError("No data loaded!")

    labels_array = np.array(labels)
    print(f"Total: {len(graphs)} (0={sum(labels_array==0)}, 1={sum(labels_array==1)})")

    os.makedirs(args.save_dir, exist_ok=True)

    fold_results = []
    oof_prob = np.full(len(graphs), np.nan)
    oof_pred = np.full(len(graphs), -1, dtype=int)
    oof_fold = np.zeros(len(graphs), dtype=int)

    splits = make_nested_splits(labels_array, n_folds=args.folds,
                                inner_val_size=args.inner_val_size, seed=args.seed)
    for fold, train_idx, val_idx, test_idx in splits:
        print(f"\n{'-'*70}\nFold {fold}/{args.folds}\n{'-'*70}")
        print(f"  Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

        train_graphs_raw = [graphs[i] for i in train_idx]
        val_graphs_raw = [graphs[i] for i in val_idx]
        test_graphs_raw = [graphs[i] for i in test_idx]

        normalizer = fit_feature_normalizer(
            train_graphs_raw, skip_cols=(1,),
            save_path=os.path.join(args.save_dir, f"feat_norm_fold{fold}.npz"))
        train_graphs = [normalizer(g) for g in train_graphs_raw]
        val_graphs = [normalizer(g) for g in val_graphs_raw]
        test_graphs = [normalizer(g) for g in test_graphs_raw]

        y_train = torch.tensor([g.y.item() for g in train_graphs])
        pos = (y_train == 1).sum().item()
        neg = (y_train == 0).sum().item()

        train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_graphs, batch_size=args.batch_size, shuffle=False)

        # Build model
        if args.modality == "t1":
            if args.t1_model == "simple":
                backbone = Simple3DCNN(dropout=args.dropout)
            else:
                backbone = MedicalNetResNet18(dropout=args.dropout)
                load_medicalnet_pretrained(backbone, args.medicalnet_ckpt)
                backbone.freeze_bn()
        else:  # fmri
            backbone = GNNBackbone(
                in_dim=train_graphs[0].num_node_features,
                hidden=args.gnn_hidden,
                model_type=args.gnn_model,
                dropout=args.dropout,
                heads=args.gat_heads
            )

        model = SingleModalityModel(
            backbone=backbone,
            modality=args.modality,
            hidden=args.fusion_hidden,
            dropout=args.fusion_dropout
        ).to(device)

        param_groups = [
            {"params": model.backbone.parameters(), "lr": args.lr, "weight_decay": args.weight_decay},
            {"params": model.classifier.parameters(), "lr": args.lr_classifier, "weight_decay": args.weight_decay},
        ]

        pos_weight = torch.tensor([neg / max(pos, 1)], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(param_groups)

        best_state, best_score, patience_counter = None, -1e9, 0

        for epoch in range(1, args.epochs + 1):
            loss = train_epoch(model, train_loader, device, criterion, optimizer)
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
                print(f"  Epoch {epoch}: Loss={loss:.4f}, Val AUC={val_metrics['auc']:.4f}")
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

        if best_state:
            model.load_state_dict(best_state)

        # Threshold from inner-val; single evaluation on the outer test fold
        y_val_true, y_val_prob = evaluate(model, val_loader, device)
        thr = (args.fixed_threshold if args.fixed_threshold is not None
               else find_optimal_threshold(y_val_true, y_val_prob))
        val_auc = compute_metrics(y_val_true, y_val_prob, threshold=thr)["auc"]

        y_test_true, y_test_prob = evaluate(model, test_loader, device)
        test_metrics = compute_metrics(y_test_true, y_test_prob, threshold=thr)
        oof_prob[test_idx] = y_test_prob
        oof_pred[test_idx] = (y_test_prob >= thr).astype(int)
        oof_fold[test_idx] = fold

        print(f"\n  Fold {fold} Test: AUC={test_metrics['auc']:.4f}, Sens={test_metrics['sensitivity']:.4f}, "
              f"Spec={test_metrics['specificity']:.4f} (thr={thr:.3f} from inner-val)")

        fold_result = {"fold": fold, "threshold": thr, "val_auc": val_auc}
        fold_result.update({f"test_{k}": test_metrics[k] for k in REPORT_METRICS})
        fold_results.append(fold_result)

        torch.save(best_state, os.path.join(args.save_dir, f"fold_{fold}_model.pt"))

    # Summary
    print(f"\n{'='*70}")
    print(f"RESULTS: {args.modality.upper()} only")
    print(f"{'='*70}")
    assert not np.isnan(oof_prob).any(), "Some subjects never appeared in a test fold"
    pooled = summarize_oof(labels_array, oof_prob, oof_pred, n_boot=args.n_bootstrap, seed=args.seed)

    results = {
        "config": vars(args),
        "evaluation": "nested_cv_full_dataset",
        "n_subjects": len(graphs),
        "cv_results": fold_results,
        "pooled_oof": {k: (list(v) if isinstance(v, tuple) else float(v)) for k, v in pooled.items()},
        "timestamp": datetime.now().isoformat()
    }
    for k in REPORT_METRICS:
        vals = [r[f"test_{k}"] for r in fold_results]
        results[f"cv_{k}_mean"], results[f"cv_{k}_std"] = float(np.mean(vals)), float(np.std(vals))
        print(f"  {k:<18} test folds {np.mean(vals):.4f} +/- {np.std(vals):.4f}   pooled OOF {pooled[k]:.4f}")
    print(f"  Pooled AUC 95% CI: [{pooled['auc_ci95'][0]:.4f}, {pooled['auc_ci95'][1]:.4f}]")

    with open(os.path.join(args.save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    pd.DataFrame({"subject_id": subject_ids, "fold": oof_fold, "y_true": labels_array,
                  "y_prob": oof_prob, "y_pred": oof_pred}
                 ).to_csv(os.path.join(args.save_dir, "oof_predictions.csv"), index=False)

    print(f"Results saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
