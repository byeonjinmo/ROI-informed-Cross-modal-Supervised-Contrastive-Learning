"""
Shared utilities for training, evaluation, and external validation.

Includes:
- Seed setting and reproducibility
- Label / graph / T1 volume loading
- Feature normalization
- Metric computation and threshold optimization
- Statistical testing
"""

import os
import random
from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             recall_score, confusion_matrix,
                             average_precision_score, cohen_kappa_score,
                             matthews_corrcoef)
from scipy import stats
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, to_undirected

try:
    import nibabel as nib
except ImportError as e:
    nib = None
    _nib_err = e


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_labels(label_path: str) -> dict:
    labels = {}
    with open(label_path, "r", encoding="utf-8") as f:
        _ = f.readline()  # Skip header
        for line in f:
            if not line.strip():
                continue
            sid, lab = line.strip().split(",")
            labels[sid] = int(lab)
    return labels


def load_subject_graph(subject_id: str, base_dir: str, adj_name: str) -> Data | None:
    """Load rs-fMRI graph for subject (Schaefer200 atlas)."""
    subj_dir = os.path.join(base_dir, subject_id, "atlas_only_schaefer200")
    node_activity_path = os.path.join(subj_dir, "NODE_local_activity.csv")
    node_time_path = os.path.join(subj_dir, "NODE_time_stats.csv")
    adj_path = os.path.join(subj_dir, adj_name)

    required = [node_activity_path, node_time_path, adj_path]
    if not all(os.path.exists(p) for p in required):
        return None

    node_activity = np.loadtxt(node_activity_path, delimiter=",", skiprows=1)
    node_time = np.loadtxt(node_time_path, delimiter=",")

    if node_activity.shape[0] != node_time.shape[0]:
        raise ValueError(f"Row mismatch for {subject_id}")

    features = np.concatenate([node_activity, node_time], axis=1)
    adj = np.loadtxt(adj_path, delimiter=",")
    rows, cols = np.nonzero(adj)
    edge_index = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = to_undirected(edge_index, num_nodes=features.shape[0])

    data = Data(
        x=torch.tensor(features, dtype=torch.float),
        edge_index=edge_index,
        y=torch.tensor([0.0], dtype=torch.float),
    )
    return data


def load_t1_volume(subject_id: str, t1_root: str, eps: float = 1e-6) -> torch.Tensor | None:
    """Load and normalize T1 MRI volume."""
    if nib is None:
        raise ImportError(f"nibabel required: {_nib_err}")

    candidates = [
        f"{subject_id}_T1_MNI2mm_brain.nii.gz",
        f"{subject_id}_T1_brain_MNI_flirt.nii.gz",
    ]
    t1_path = None
    for name in candidates:
        p = os.path.join(t1_root, name)
        if os.path.exists(p):
            t1_path = p
            break
    if t1_path is None:
        return None

    img = nib.load(t1_path)
    data = img.get_fdata().astype(np.float32)

    # Z-score normalization on non-zero voxels
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


# ---------------------------------------------------------------------------
# Feature normalization
# ---------------------------------------------------------------------------

def fit_feature_normalizer(graphs: Sequence[Data], skip_cols: Sequence[int] = (1,),
                           eps: float = 1e-6, save_path: str = None):
    """Fit z-score normalizer on graph node features.

    Args:
        graphs: List of graph Data objects
        skip_cols: Column indices to skip normalization (default: (1,) for ALFF_z)
        eps: Small value for numerical stability
        save_path: If provided, save normalizer stats to this path (.npz)

    Returns:
        transform: Function to normalize graph features
    """
    stacked = torch.cat([g.x for g in graphs], dim=0)
    mean = stacked.mean(dim=0)
    std = stacked.std(dim=0)
    std = torch.clamp(std, min=eps)

    mask = torch.ones(mean.shape[0], dtype=torch.bool)
    for idx in skip_cols:
        if 0 <= idx < mask.numel():
            mask[idx] = False

    if save_path is not None:
        np.savez(save_path,
                 mean=mean.numpy(),
                 std=std.numpy(),
                 skip_cols=np.array(skip_cols))
        print(f"[Normalizer] Saved to {save_path}")

    def transform(graph: Data) -> Data:
        g = graph.clone()
        x = g.x.clone()
        x[:, mask] = (x[:, mask] - mean[mask]) / std[mask]
        g.x = x
        return g

    return transform


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                    threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)

    try:
        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
        aupr = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    except Exception:
        auc = aupr = float("nan")

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    sens = recall_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
    spec = tn / max(tn + fp, 1)

    balanced_acc = (sens + spec) / 2
    ppv = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)

    try:
        kappa = cohen_kappa_score(y_true, y_pred)
    except Exception:
        kappa = 0.0

    try:
        mcc = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_pred)) > 1 else 0.0
    except Exception:
        mcc = 0.0

    return {
        "auc": auc,
        "aupr": aupr,
        "accuracy": acc,
        "balanced_accuracy": balanced_acc,
        "f1": f1,
        "f1_weighted": f1_weighted,
        "sensitivity": sens,
        "specificity": spec,
        "ppv": ppv,
        "npv": npv,
        "kappa": kappa,
        "mcc": mcc,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn
    }


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find threshold that maximizes Youden's J (Sensitivity + Specificity - 1)."""
    thresholds = np.linspace(0.1, 0.9, 81)

    best_j = -1
    best_t = 0.5

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        j = sens + spec - 1

        if j > best_j:
            best_j = j
            best_t = t

    return best_t


def compute_statistical_tests(fold_results: list, prefix: str = "test") -> dict:
    """One-sample t-test: test if each metric is significantly above chance level."""
    chance_levels = {
        "auc": 0.5,
        "aupr": 0.5,
        "balanced_accuracy": 0.5,
        "sensitivity": 0.5,
        "specificity": 0.5,
        "f1": 0.0,
        "kappa": 0.0,
        "mcc": 0.0,
    }
    metrics_to_test = {f"{prefix}_{k}": v for k, v in chance_levels.items()}
    results = {}
    for metric, chance in metrics_to_test.items():
        values = [r[metric] for r in fold_results]
        t_stat, p_value = stats.ttest_1samp(values, chance)
        p_one_sided = p_value / 2 if t_stat > 0 else 1.0 - p_value / 2
        results[metric] = {
            "t_statistic": float(t_stat),
            "p_value": float(p_one_sided),
            "significant": p_one_sided < 0.05
        }
    return results


# ---------------------------------------------------------------------------
# Leakage-free evaluation (nested CV over the full dataset)
# ---------------------------------------------------------------------------

def make_nested_splits(labels: np.ndarray, n_folds: int = 5, inner_val_size: float = 0.125,
                       seed: int = 42):
    """Outer stratified K-fold over ALL subjects + inner train/val split.

    Every subject appears in exactly one outer test fold. Within each outer
    fold, the training portion is further split into inner-train (fit model and
    normalizer) and inner-val (early stopping + threshold selection). The outer
    test fold is never touched until the final, single evaluation.

    Yields (fold, train_idx, val_idx, test_idx) as indices into `labels`.
    """
    from sklearn.model_selection import StratifiedKFold, train_test_split

    labels = np.asarray(labels)
    all_idx = np.arange(len(labels))
    outer = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (dev_idx, test_idx) in enumerate(outer.split(all_idx, labels), 1):
        train_idx, val_idx = train_test_split(
            dev_idx, test_size=inner_val_size,
            stratify=labels[dev_idx], random_state=seed + fold,
        )
        assert not (set(train_idx) | set(val_idx)) & set(test_idx)
        yield fold, np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)


def bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray, metric_fn,
                 n_boot: int = 2000, alpha: float = 0.05, seed: int = 42) -> tuple:
    """Stratified percentile bootstrap CI of a metric on pooled predictions."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    pos = np.where(y_true == 1)[0]
    neg = np.where(y_true == 0)[0]
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True),
                              rng.choice(neg, len(neg), replace=True)])
        vals.append(metric_fn(y_true[idx], y_prob[idx]))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def summarize_oof(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray,
                  n_boot: int = 2000, seed: int = 42) -> dict:
    """Pooled out-of-fold metrics. Each prediction comes from a model (and a
    threshold) that never saw that subject, so the pooled metrics are leak-free."""
    m = compute_metrics(y_true, y_prob, threshold=0.5)
    # Overwrite threshold-dependent metrics with per-fold-threshold predictions
    m_pred = compute_metrics(y_true, y_pred.astype(float), threshold=0.5)
    for k in ["accuracy", "balanced_accuracy", "f1", "f1_weighted", "sensitivity",
              "specificity", "ppv", "npv", "kappa", "mcc", "tp", "fp", "tn", "fn"]:
        m[k] = m_pred[k]
    m["auc_ci95"] = bootstrap_ci(y_true, y_prob, roc_auc_score, n_boot=n_boot, seed=seed)
    m["aupr_ci95"] = bootstrap_ci(y_true, y_prob, average_precision_score, n_boot=n_boot, seed=seed)
    return m


# ---------------------------------------------------------------------------
# Dataset building
# ---------------------------------------------------------------------------

def build_dataset(label_path: str, outputs_root: str, t1_root: str,
                  adj_name: str, modality: str = "both", return_ids: bool = False):
    """Build dataset loading graphs and optionally T1 volumes."""
    labels = load_labels(label_path)
    graphs, y = [], []
    missing = []

    for sid, lab in labels.items():
        g = load_subject_graph(sid, outputs_root, adj_name)
        t1 = load_t1_volume(sid, t1_root) if modality in ["t1", "both"] else None

        if g is None:
            missing.append(sid)
            continue

        if modality in ["t1", "both"] and t1 is None:
            missing.append(sid)
            continue

        g.y = torch.tensor([float(lab)], dtype=torch.float)
        if t1 is not None:
            g.t1 = t1
        graphs.append(g)
        y.append(lab)

    if missing:
        print(f"Skipped {len(missing)} subjects: {missing[:5]}...")

    if return_ids:
        missing_set = set(missing)
        ids = [sid for sid in labels if sid not in missing_set]
        return graphs, y, ids
    return graphs, y
