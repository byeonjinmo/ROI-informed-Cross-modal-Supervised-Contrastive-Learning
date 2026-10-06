"""
External Validation for Multimodal Depression Classification
Evaluate trained model on SRPBS_OPEN dataset

Note: External data has 6 features (no ReHo), padded to 7 features
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import remove_self_loops, to_undirected

try:
    import nibabel as nib
except ImportError as e:
    nib = None
    _nib_err = e

from models.resnet3d import MedicalNetResNet18, load_medicalnet_pretrained
from models.gnn import GNNBackbone
from models.multimodal_fusion import MultimodalFusion

# Shared utilities
from sklearn.metrics import roc_auc_score
from utils import compute_metrics, find_optimal_threshold, bootstrap_ci



def load_train_normalizer(npz_path: str):
    """Load normalizer statistics saved during training."""
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Train normalizer not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    mean = torch.tensor(data['mean'], dtype=torch.float)
    std = torch.tensor(data['std'], dtype=torch.float)
    std = torch.clamp(std, min=1e-6)
    skip_cols = tuple(data['skip_cols'].tolist()) if 'skip_cols' in data else (1,)

    print(f"[Normalizer] Loaded from {npz_path}")
    print(f"  Mean shape: {mean.shape}, Std shape: {std.shape}")
    print(f"  Skip cols: {skip_cols}")

    mask = torch.ones(mean.shape[0], dtype=torch.bool)
    for idx in skip_cols:
        if 0 <= idx < mask.numel():
            mask[idx] = False

    def transform(graph: Data) -> Data:
        g = graph.clone()
        x = g.x.clone()
        if not torch.isfinite(x).all():
            fill = mean.unsqueeze(0).expand_as(x)
            x = torch.where(torch.isfinite(x), x, fill)
        x[:, mask] = (x[:, mask] - mean[mask]) / std[mask]
        g.x = x
        return g

    return transform


def _nanmean_std(stacked: torch.Tensor, eps: float = 1e-6) -> tuple:
    finite = torch.isfinite(stacked)
    count = finite.sum(dim=0).clamp(min=1)
    summed = torch.where(finite, stacked, torch.zeros_like(stacked)).sum(dim=0)
    mean = summed / count
    diff = torch.where(finite, stacked - mean, torch.zeros_like(stacked))
    var = (diff ** 2).sum(dim=0) / count
    std = torch.sqrt(var)
    std = torch.clamp(std, min=eps)
    return mean, std


def fit_feature_normalizer_robust(graphs, skip_cols=(1,), eps: float = 1e-6):
    """Fit z-score normalizer with NaN/Inf-safe stats."""
    stacked = torch.cat([g.x for g in graphs], dim=0)
    mean, std = _nanmean_std(stacked, eps=eps)

    mask = torch.ones(mean.shape[0], dtype=torch.bool)
    for idx in skip_cols:
        if 0 <= idx < mask.numel():
            mask[idx] = False

    def transform(graph: Data) -> Data:
        g = graph.clone()
        x = g.x.clone()
        if not torch.isfinite(x).all():
            fill = mean.unsqueeze(0).expand_as(x)
            x = torch.where(torch.isfinite(x), x, fill)
        x[:, mask] = (x[:, mask] - mean[mask]) / std[mask]
        g.x = x
        return g

    return transform


def report_feature_health(graphs, col_names, tag: str) -> bool:
    stacked = torch.cat([g.x for g in graphs], dim=0)
    nan_counts = torch.isnan(stacked).sum(dim=0)
    inf_counts = torch.isinf(stacked).sum(dim=0)
    has_issue = bool((nan_counts > 0).any() or (inf_counts > 0).any())
    if has_issue:
        print(f"\n[Feature health - {tag}]")
        for i, name in enumerate(col_names):
            if nan_counts[i] > 0 or inf_counts[i] > 0:
                print(f"  {name}: NaN={int(nan_counts[i])}, Inf={int(inf_counts[i])}")
    return has_issue


def sanitize_graph_features(graphs, fill_values=None):
    if fill_values is None:
        stacked = torch.cat([g.x for g in graphs], dim=0)
        fill_values, _ = _nanmean_std(stacked)
    for i, g in enumerate(graphs):
        if not torch.isfinite(g.x).all():
            x = g.x.clone()
            fill = fill_values.unsqueeze(0).expand_as(x)
            x = torch.where(torch.isfinite(x), x, fill)
            g.x = x
            graphs[i] = g
    return graphs



def load_external_labels(file_path: str, label_mode: str = "bdi", bdi_threshold: int = 14) -> dict:
    """Load labels from sup2.tsv or other label files."""
    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    elif file_path.endswith('.tsv'):
        df = pd.read_csv(file_path, sep='\t')
    else:
        df = pd.read_csv(file_path)

    labels = {}
    for _, row in df.iterrows():
        sid = row['participant_id']
        if label_mode == "bdi":
            bdi = row.get('BDI-II', None)
            if bdi is None or pd.isna(bdi):
                continue
            labels[sid] = 1 if int(bdi) >= bdi_threshold else 0
        else:
            diag = int(row['diag'])
            if diag == 0:
                labels[sid] = 0
            elif diag == 2:
                labels[sid] = 1

    n_pos = sum(v == 1 for v in labels.values())
    n_neg = sum(v == 0 for v in labels.values())
    print(f"[Labels] Mode={label_mode}, Loaded {len(labels)} subjects (0={n_neg}, 1={n_pos})")
    return labels



def load_external_graph(subject_id, base_dir, adj_name,
                        reho_fill_mode="zeros", train_reho_mean=None,
                        verify_adj=False, adj_log_count=0, max_adj_logs=5):
    """Load external rs-fMRI graph with ReHo padding options."""
    subj_dir = os.path.join(base_dir, subject_id, "atlas_only_schaefer200")
    node_activity_path = os.path.join(subj_dir, "NODE_local_activity.csv")
    node_time_path = os.path.join(subj_dir, "NODE_time_stats.csv")
    adj_path = os.path.join(subj_dir, adj_name)

    required = [node_activity_path, node_time_path, adj_path]
    if not all(os.path.exists(p) for p in required):
        return None, None

    node_activity = np.loadtxt(node_activity_path, delimiter=",", skiprows=1)
    node_time = np.loadtxt(node_time_path, delimiter=",")

    if node_activity.shape[0] != node_time.shape[0]:
        return None, None

    num_rois = node_activity.shape[0]
    num_activity_cols = node_activity.shape[1] if node_activity.ndim > 1 else 1

    if num_activity_cols == 3:
        features = np.concatenate([node_activity, node_time], axis=1)
    elif num_activity_cols == 2:
        if reho_fill_mode == "train_mean" and train_reho_mean is not None:
            reho_col = train_reho_mean.reshape(-1, 1)
            if reho_col.shape[0] != num_rois:
                reho_col = np.zeros((num_rois, 1))
        else:
            reho_col = np.zeros((num_rois, 1))
        features = np.concatenate([reho_col, node_activity, node_time], axis=1)
    else:
        return None, None

    adj = np.loadtxt(adj_path, delimiter=",")
    rows, cols = np.nonzero(adj)
    edge_index = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = to_undirected(edge_index, num_nodes=num_rois)

    data = Data(
        x=torch.tensor(features, dtype=torch.float),
        edge_index=edge_index,
        y=torch.tensor([0.0], dtype=torch.float)
    )
    return data, None


def load_t1_volume_external(subject_id: str, t1_root: str, eps: float = 1e-6):
    """Load and normalize external T1 MRI volume"""
    if nib is None:
        raise ImportError(f"nibabel required: {_nib_err}")

    t1_path = os.path.join(t1_root, f"{subject_id}_MNI2mm.nii.gz")
    if not os.path.exists(t1_path):
        return None

    img = nib.load(t1_path)
    data = img.get_fdata().astype(np.float32)

    mask = data != 0
    if mask.any():
        mu = data[mask].mean()
        sigma = data[mask].std()
    else:
        mu = data.mean()
        sigma = data.std()
    sigma = max(sigma, eps)
    data = (data - mu) / sigma
    data = np.clip(data, -5.0, 5.0)

    tensor = torch.from_numpy(data)[None, None, ...].contiguous()
    return tensor


def build_external_dataset(label_path, outputs_root, t1_root, adj_name,
                           reho_fill_mode="zeros", train_reho_mean_path=None,
                           label_mode="bdi", bdi_threshold=14):
    """Build external validation dataset"""
    labels = load_external_labels(label_path, label_mode=label_mode, bdi_threshold=bdi_threshold)

    train_reho_mean = None
    if reho_fill_mode == "train_mean":
        if train_reho_mean_path and os.path.exists(train_reho_mean_path):
            train_reho_mean = np.load(train_reho_mean_path)
            print(f"[ReHo] Loaded train mean from {train_reho_mean_path}")
        else:
            reho_fill_mode = "zeros"

    graphs, y = [], []
    missing = []

    for sid, lab in labels.items():
        g, _ = load_external_graph(
            sid, outputs_root, adj_name,
            reho_fill_mode=reho_fill_mode,
            train_reho_mean=train_reho_mean,
        )
        t1 = load_t1_volume_external(sid, t1_root)

        if g is None or t1 is None:
            missing.append(sid)
            continue

        g.y = torch.tensor([float(lab)], dtype=torch.float)
        g.t1 = t1
        graphs.append(g)
        y.append(lab)

    print(f"\n[Dataset] Loaded {len(graphs)} subjects, skipped {len(missing)}")
    return graphs, y



def evaluate(model, loader, device):
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


def fold_normalizer_path(model_path) -> Path:
    """fold_3_model.pt -> feat_norm_fold3.npz (saved by train.py next to the model)."""
    model_path = Path(model_path)
    fold_id = model_path.stem.replace("fold_", "").replace("_model", "")
    return model_path.parent / f"feat_norm_fold{fold_id}.npz"


def build_loader(graphs_raw, model_path, args):
    """Normalize with the statistics of the fold that trained this model."""
    norm_path = args.train_normalizer or fold_normalizer_path(model_path)
    if os.path.exists(norm_path):
        transform = load_train_normalizer(str(norm_path))
    else:
        print(f"[WARN] Train normalizer not found ({norm_path}); fitting on external data (NOT RECOMMENDED)")
        transform = fit_feature_normalizer_robust(graphs_raw, skip_cols=(1,))
    graphs = [transform(g) for g in graphs_raw]
    return DataLoader(graphs, batch_size=args.batch_size, shuffle=False)


def load_train_thresholds(results_path, model_paths):
    """Thresholds chosen on inner-val during training (results.json from train.py)."""
    with open(results_path, "r") as f:
        cv_results = json.load(f)["cv_results"]
    by_fold = {int(r["fold"]): float(r["threshold"]) for r in cv_results}
    thresholds = []
    for p in model_paths:
        fold_id = Path(p).stem.replace("fold_", "").replace("_model", "")
        if fold_id.isdigit() and int(fold_id) in by_fold:
            thresholds.append(by_fold[int(fold_id)])
    return thresholds or list(by_fold.values())


def evaluate_ensemble(model_paths, graphs_raw, device, args):
    """Run inference with multiple checkpoints and average probabilities."""
    y_true = None
    all_probs = []

    for idx, model_path in enumerate(model_paths, start=1):
        print(f"[Ensemble] Loading model {idx}/{len(model_paths)}: {model_path}")
        model = load_model(args, device, model_path=str(model_path))
        loader = build_loader(graphs_raw, model_path, args)
        y_true_i, y_prob_i = evaluate(model, loader, device)
        if y_true is None:
            y_true = y_true_i
        all_probs.append(y_prob_i)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    mean_probs = np.mean(np.stack(all_probs, axis=0), axis=0)
    return y_true, mean_probs


def load_model(args, device, num_node_features=7, model_path=None):
    """Load trained model from checkpoint"""
    ckpt_path = model_path or args.model_path

    gnn = GNNBackbone(
        in_dim=num_node_features,
        hidden=args.gnn_hidden,
        model_type=args.gnn_model,
        dropout=args.dropout,
        heads=args.gat_heads
    )

    t1_backbone = MedicalNetResNet18(dropout=args.dropout)
    if args.medicalnet_ckpt and os.path.exists(args.medicalnet_ckpt):
        load_medicalnet_pretrained(t1_backbone, args.medicalnet_ckpt)
    t1_backbone.freeze_bn()

    model = MultimodalFusion(
        gnn=gnn,
        t1=t1_backbone,
        fusion_hidden=args.fusion_hidden,
        fusion_dropout=args.fusion_dropout,
        fusion_type=args.fusion_type,
        modality_dropout=0.0,
        atlas_path=args.atlas_path,
        use_contrastive=args.use_contrastive,
        contrastive_tau=args.contrastive_tau,
    ).to(device)

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"[Model] Loaded from: {ckpt_path}")
    return model


def main():
    parser = argparse.ArgumentParser(description="External Validation - SRPBS_OPEN")

    # External data paths
    parser.add_argument("--external-root", default="./data/external")
    parser.add_argument("--label-file", default="sup2.tsv")
    parser.add_argument("--label-mode", choices=["bdi", "diag"], default="bdi")
    parser.add_argument("--bdi-threshold", type=int, default=14)
    parser.add_argument("--adj-name", default="ADJ_abs_dens10.csv")

    # Trained model
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--model-paths", nargs="+", default=None,
                        help="List of model checkpoints for ensemble")
    parser.add_argument("--model-dir", default=None,
                        help="Directory containing fold models")
    parser.add_argument("--model-glob", default="fold_*_model.pt")
    parser.add_argument("--medicalnet-ckpt", default="./pretrain/resnet_18_23dataset.pth")

    # Model architecture
    parser.add_argument("--gnn-model", default="gat")
    parser.add_argument("--gnn-hidden", type=int, default=256)
    parser.add_argument("--gat-heads", type=int, default=4)
    parser.add_argument("--fusion-hidden", type=int, default=256)
    parser.add_argument("--fusion-type", default="attn")
    parser.add_argument("--fusion-dropout", type=float, default=0.4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--use-contrastive", action="store_true")
    parser.add_argument("--contrastive-tau", type=float, default=0.05)
    parser.add_argument("--atlas-path",
                        default="./data/atlas/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii")

    # Train normalizer
    parser.add_argument("--train-normalizer", default=None,
                        help="Force one feat_norm npz for all models. Default: use "
                             "feat_norm_fold{k}.npz next to each fold_{k}_model.pt")
    parser.add_argument("--train-results", default=None,
                        help="results.json from train.py; the decision threshold is the mean of "
                             "the inner-val thresholds (default: <model-dir>/results.json)")

    # ReHo fill mode
    parser.add_argument("--reho-fill-mode", choices=["zeros", "train_mean"], default="zeros")
    parser.add_argument("--train-reho-mean", default=None,
                        help="Path to train_reho_mean_200.npy")

    # Inference
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=None)

    # Output
    parser.add_argument("--save-dir", default="./results_external")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("EXTERNAL VALIDATION - SRPBS_OPEN Dataset")
    print("=" * 70)
    print(f"Device: {device}")

    # Resolve model paths
    model_paths = []
    if args.model_paths:
        model_paths = args.model_paths
    elif args.model_dir:
        model_dir = Path(args.model_dir)
        model_paths = sorted(model_dir.glob(args.model_glob))
        if not model_paths:
            raise FileNotFoundError(f"No models found in {model_dir} with pattern {args.model_glob}")
    else:
        if not args.model_path:
            raise ValueError("Provide --model-path, --model-paths, or --model-dir.")
        model_paths = [args.model_path]

    print(f"Models: {len(model_paths)}")

    # Build paths
    label_path = os.path.join(args.external_root, args.label_file)
    outputs_root = os.path.join(args.external_root, "outputs")
    t1_root = os.path.join(args.external_root, "T1_MNI")

    # Load external dataset
    graphs, labels = build_external_dataset(
        label_path, outputs_root, t1_root, args.adj_name,
        reho_fill_mode=args.reho_fill_mode,
        train_reho_mean_path=args.train_reho_mean,
        label_mode=args.label_mode,
        bdi_threshold=args.bdi_threshold
    )

    if not graphs:
        raise RuntimeError("No external data loaded!")

    labels_array = np.array(labels)
    print(f"\nTotal: {len(graphs)} (0={sum(labels_array==0)}, 1={sum(labels_array==1)})")

    # Check feature health
    col_names = ['ReHo', 'ALFF_z', 'fALFF', 'mean', 'std', 'skew', 'kurt']
    if report_feature_health(graphs, col_names, tag="before normalization"):
        graphs = sanitize_graph_features(graphs)

    # Inference (each model normalized with its own fold's training statistics)
    y_true, y_prob = evaluate_ensemble(model_paths, graphs, device, args)

    # Handle NaN/Inf
    if np.any(np.isnan(y_prob)) or np.any(np.isinf(y_prob)):
        valid_mask = ~(np.isnan(y_prob) | np.isinf(y_prob))
        y_true = y_true[valid_mask]
        y_prob = y_prob[valid_mask]

    # Threshold: fixed before seeing external labels (no tuning on external data)
    if args.threshold is not None:
        threshold, threshold_source = args.threshold, "user-specified"
    else:
        results_path = args.train_results or (
            os.path.join(args.model_dir, "results.json") if args.model_dir else None)
        if results_path and os.path.exists(results_path):
            thresholds = load_train_thresholds(results_path, model_paths)
            threshold = float(np.mean(thresholds))
            threshold_source = f"mean inner-val threshold from {results_path}"
        else:
            threshold, threshold_source = 0.5, "default 0.5 (train results.json not found)"

    # Metrics
    metrics = compute_metrics(y_true, y_prob, threshold=threshold)
    auc_ci = bootstrap_ci(y_true, y_prob, roc_auc_score)
    # Youden-optimal threshold on the external labels is optimistic (label leakage);
    # reported for reference only, never as the main result.
    oracle_thr = find_optimal_threshold(y_true, y_prob)
    oracle_metrics = compute_metrics(y_true, y_prob, threshold=oracle_thr)

    print("\n" + "=" * 70)
    print("EXTERNAL VALIDATION RESULTS")
    print("=" * 70)
    print(f"Threshold: {threshold:.3f} ({threshold_source})")
    print(f"  AUC 95% CI:   [{auc_ci[0]:.4f}, {auc_ci[1]:.4f}]")
    print(f"  AUC:          {metrics['auc']:.4f}")
    print(f"  AUPR:         {metrics['aupr']:.4f}")
    print(f"  Balanced Acc: {metrics['balanced_accuracy']:.4f}")
    print(f"  Sensitivity:  {metrics['sensitivity']:.4f}")
    print(f"  Specificity:  {metrics['specificity']:.4f}")
    print(f"  F1:           {metrics['f1']:.4f}")
    print(f"  Kappa:        {metrics['kappa']:.4f}")
    print(f"  MCC:          {metrics['mcc']:.4f}")
    print(f"  [Reference only, optimistic] Youden on external labels: thr={oracle_thr:.3f}, "
          f"BAcc={oracle_metrics['balanced_accuracy']:.4f}")
    print("=" * 70)

    # Save
    os.makedirs(args.save_dir, exist_ok=True)
    results = {
        "model_paths": [str(p) for p in model_paths],
        "ensemble": len(model_paths) > 1,
        "num_samples": len(graphs),
        "threshold": float(threshold),
        "threshold_source": threshold_source,
        "metrics": {k: float(v) if not isinstance(v, int) else v for k, v in metrics.items()},
        "auc_ci95": list(auc_ci),
        "oracle_reference_only": {
            "threshold": float(oracle_thr),
            "balanced_accuracy": float(oracle_metrics["balanced_accuracy"]),
            "sensitivity": float(oracle_metrics["sensitivity"]),
            "specificity": float(oracle_metrics["specificity"]),
        },
        "timestamp": datetime.now().isoformat()
    }

    with open(os.path.join(args.save_dir, "external_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    predictions_df = pd.DataFrame({
        "y_true": y_true,
        "y_prob": y_prob,
        "y_pred": (y_prob >= threshold).astype(int)
    })
    predictions_df.to_csv(os.path.join(args.save_dir, "predictions.csv"), index=False)

    print(f"Results saved to: {args.save_dir}")
    return metrics


if __name__ == "__main__":
    main()
