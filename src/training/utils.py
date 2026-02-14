"""
Training utility functions: metrics, checkpointing, data splitting, and sampling.
"""

import os
import datetime
import json

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
from scipy import stats
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.model_selection import StratifiedKFold, train_test_split


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def calculate_metrics(y_true, y_pred, y_scores):
    """Compute classification metrics including accuracy, AUC, sensitivity, etc.

    Args:
        y_true (np.ndarray): Ground-truth labels.
        y_pred (np.ndarray): Predicted labels.
        y_scores (np.ndarray): Prediction scores (probability of positive class).

    Returns:
        dict: Dictionary of metric names to values (all in percentage).
    """
    metrics = {}
    metrics["accuracy"] = (y_true == y_pred).mean() * 100

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics["precision"] = report["weighted avg"]["precision"] * 100
    metrics["recall"] = report["weighted avg"]["recall"] * 100
    metrics["f1_score"] = report["weighted avg"]["f1-score"] * 100

    unique_classes = np.unique(y_true)
    if len(unique_classes) == 2:
        TP = ((y_pred == 1) & (y_true == 1)).sum()
        TN = ((y_pred == 0) & (y_true == 0)).sum()
        FP = ((y_pred == 1) & (y_true == 0)).sum()
        FN = ((y_pred == 0) & (y_true == 1)).sum()

        metrics["TP"] = TP
        metrics["FP"] = FP
        metrics["FN"] = FN
        metrics["TN"] = TN

        metrics["sensitivity"] = (TP / (TP + FN) if (TP + FN) > 0 else 0) * 100
        metrics["specificity"] = (TN / (TN + FP) if (TN + FP) > 0 else 0) * 100
        metrics["ppv"] = (TP / (TP + FP) if (TP + FP) > 0 else 0) * 100
        metrics["npv"] = (TN / (TN + FN) if (TN + FN) > 0 else 0) * 100

        try:
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            metrics["auc"] = auc(fpr, tpr) * 100
        except ValueError:
            metrics["auc"] = 0
    else:
        metrics["TP"] = 0
        metrics["FP"] = 0
        metrics["FN"] = 0
        metrics["TN"] = 0
        metrics["sensitivity"] = 0
        metrics["specificity"] = 0
        metrics["ppv"] = 0
        metrics["npv"] = 0
        metrics["auc"] = 0

    return metrics


def calculate_confidence_interval(values, confidence=0.95):
    """Compute the confidence interval for a list of metric values.

    Args:
        values (list[float]): Metric values across folds.
        confidence (float): Confidence level (default 0.95).

    Returns:
        tuple: (lower_bound, upper_bound) of the confidence interval.
    """
    if len(values) < 2:
        return 0, 0
    mean = np.mean(values)
    sem = stats.sem(values)
    ci = stats.t.interval(confidence, len(values) - 1, loc=mean, scale=sem)
    return ci[0], ci[1]


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def convert_to_serializable(obj):
    """Recursively convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, epoch, metrics, fold, is_best, checkpoint_dir):
    """Save a training checkpoint and optionally the best model.

    Args:
        model: PyTorch model.
        optimizer: Optimizer state.
        epoch (int): Current epoch number.
        metrics (dict): Validation metrics.
        fold (int): Current fold number.
        is_best (bool): Whether this is the best model so far.
        checkpoint_dir (str): Directory to save checkpoints.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "fold": fold,
        "date": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
    }
    checkpoint_path = os.path.join(
        checkpoint_dir, f"checkpoint_fold{fold}_epoch{epoch}.pth"
    )
    torch.save(checkpoint, checkpoint_path)

    if is_best:
        best_path = os.path.join(
            checkpoint_dir,
            f'best_model_fold{fold}_acc{metrics["accuracy"]:.2f}.pth',
        )
        torch.save(checkpoint, best_path)

        serializable_metrics = convert_to_serializable(metrics)
        metrics_path = os.path.join(checkpoint_dir, f"best_metrics_fold{fold}.json")
        with open(metrics_path, "w") as f:
            json.dump(serializable_metrics, f, indent=4)


def cleanup_checkpoints(checkpoint_dir, fold, best_acc):
    """Remove non-best epoch checkpoints for a given fold.

    Args:
        checkpoint_dir (str): Checkpoint directory.
        fold (int): Fold number.
        best_acc (float): Best accuracy (unused, kept for API compatibility).
    """
    for file in os.listdir(checkpoint_dir):
        if f"fold{fold}_epoch" in file and "best" not in file:
            os.remove(os.path.join(checkpoint_dir, file))


def save_final_best_model(best_fold_metrics, checkpoint_dir):
    """Save the overall best model across all folds.

    Args:
        best_fold_metrics (dict): Mapping from fold number to best metrics.
        checkpoint_dir (str): Checkpoint directory.
    """
    best_fold = max(
        best_fold_metrics.keys(), key=lambda k: best_fold_metrics[k]["accuracy"]
    )
    best_metrics = best_fold_metrics[best_fold]
    best_model_path = os.path.join(
        checkpoint_dir,
        f'best_model_fold{best_fold}_acc{best_metrics["accuracy"]:.2f}.pth',
    )

    if os.path.exists(best_model_path):
        final_path = os.path.join(
            checkpoint_dir,
            f'final_best_model_acc{best_metrics["accuracy"]:.2f}.pth',
        )
        checkpoint = torch.load(best_model_path, weights_only=False)
        torch.save(checkpoint, final_path)

        serializable_metrics = convert_to_serializable(best_metrics)
        final_metrics_path = os.path.join(checkpoint_dir, "final_best_metrics.json")
        with open(final_metrics_path, "w") as f:
            json.dump(
                {"best_fold": best_fold, "metrics": serializable_metrics}, f, indent=4
            )


# ---------------------------------------------------------------------------
# Weighted sampling for class imbalance
# ---------------------------------------------------------------------------

def create_weighted_sampler(labels):
    """Create a WeightedRandomSampler to oversample the minority class.

    Args:
        labels (list[int]): Training labels.

    Returns:
        WeightedRandomSampler: Sampler with inverse-frequency weights.
    """
    class_counts = np.bincount(labels)
    print(f"Class counts: {class_counts}")

    class_weights = 1.0 / class_counts
    class_weights = np.nan_to_num(class_weights, nan=0.0, posinf=0.0)

    if np.any(class_weights == 0):
        class_weights[class_weights == 0] = 0.0001

    class_weights = class_weights / class_weights.sum()

    weights = [class_weights[label] for label in labels]
    weights = torch.DoubleTensor(weights)

    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    print(f"Created weighted sampler with weights: {class_weights}")

    return sampler


# ---------------------------------------------------------------------------
# Subject-level data splitting (prevents data leakage)
# ---------------------------------------------------------------------------

def extract_subject_id(filename):
    """Extract the subject ID from a DICOM filename.

    Assumes filenames follow the pattern ``P239170000001_1.dcm``, where
    the part before the first underscore is the subject ID.

    Args:
        filename (str): DICOM filename.

    Returns:
        str: Subject identifier.
    """
    base_name = os.path.splitext(filename)[0]
    if "_" in base_name:
        return base_name.split("_")[0]
    return base_name


def verify_no_subject_leakage(train_indices, val_indices, test_indices, dataset):
    """Verify that no subject appears in more than one data split.

    Args:
        train_indices (list[int]): Training set indices.
        val_indices (list[int]): Validation set indices.
        test_indices (list[int]): Test set indices.
        dataset: Dataset with ``image_files`` attribute.

    Returns:
        tuple: (train_subject_count, val_subject_count, test_subject_count).
    """

    def get_subject_ids_from_indices(indices):
        subject_ids = set()
        for idx in indices:
            img_path = dataset.image_files[idx]
            filename = os.path.basename(img_path)
            subject_id = extract_subject_id(filename)
            subject_ids.add(subject_id)
        return subject_ids

    train_subjects = get_subject_ids_from_indices(train_indices)
    val_subjects = get_subject_ids_from_indices(val_indices)
    test_subjects = get_subject_ids_from_indices(test_indices)

    train_val_overlap = train_subjects.intersection(val_subjects)
    train_test_overlap = train_subjects.intersection(test_subjects)
    val_test_overlap = val_subjects.intersection(test_subjects)

    if train_val_overlap:
        print(f"WARNING: Train-val subject overlap: {train_val_overlap}")
    if train_test_overlap:
        print(f"WARNING: Train-test subject overlap: {train_test_overlap}")
    if val_test_overlap:
        print(f"WARNING: Val-test subject overlap: {val_test_overlap}")

    if not any([train_val_overlap, train_test_overlap, val_test_overlap]):
        print("PASSED: No subject leakage detected.")

    return len(train_subjects), len(val_subjects), len(test_subjects)


def subject_level_split(dataset, test_size=0, num_folds=5, random_state=42):
    """Split the dataset at the subject level to prevent data leakage.

    First separates a held-out test set, then creates stratified K-fold
    splits on the remaining subjects.

    Args:
        dataset: Dataset with ``image_files`` and ``labels`` attributes.
        test_size (float): Fraction of subjects reserved for testing.
        num_folds (int): Number of cross-validation folds.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: (fold_splits, test_indices, train_val_labels)
            - fold_splits: list of (train_indices, val_indices) per fold.
            - test_indices: list of image indices for the test set.
            - train_val_labels: labels for the train+val set.
    """
    # Build subject-to-indices mapping
    subject_ids = []
    subject_labels = []
    subject_to_indices = {}

    for idx in range(len(dataset)):
        img_path = dataset.image_files[idx]
        filename = os.path.basename(img_path)
        subject_id = extract_subject_id(filename)

        if subject_id not in subject_to_indices:
            subject_to_indices[subject_id] = []
            subject_ids.append(subject_id)
            subject_labels.append(dataset.labels[idx])

        subject_to_indices[subject_id].append(idx)

    print(f"Total unique subjects: {len(subject_ids)}")

    for subject_id in subject_ids[:5]:
        print(
            f"Subject {subject_id} has {len(subject_to_indices[subject_id])} images"
        )

    # Subject-level test split
    train_val_subjects, test_subjects = train_test_split(
        subject_ids,
        test_size=test_size,
        random_state=random_state,
        stratify=subject_labels,
    )

    # Map subjects back to image indices
    train_val_indices = []
    test_indices = []

    for subject_id in train_val_subjects:
        train_val_indices.extend(subject_to_indices[subject_id])
    for subject_id in test_subjects:
        test_indices.extend(subject_to_indices[subject_id])

    train_val_labels = [dataset.labels[idx] for idx in train_val_indices]

    # Stratified K-Fold on the train+val subjects
    train_val_subject_labels = [
        subject_labels[subject_ids.index(sid)] for sid in train_val_subjects
    ]

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state)

    fold_splits = []
    for fold, (train_subject_indices, val_subject_indices) in enumerate(
        skf.split(train_val_subjects, train_val_subject_labels)
    ):
        train_subjects_fold = [train_val_subjects[i] for i in train_subject_indices]
        val_subjects_fold = [train_val_subjects[i] for i in val_subject_indices]

        train_indices = []
        val_indices = []

        for subject_id in train_subjects_fold:
            train_indices.extend(subject_to_indices[subject_id])
        for subject_id in val_subjects_fold:
            val_indices.extend(subject_to_indices[subject_id])

        fold_splits.append((train_indices, val_indices))

        # Verify no subject leakage for this fold
        print(f"\nFold {fold + 1} verification:")
        train_count, val_count, test_count = verify_no_subject_leakage(
            train_indices, val_indices, test_indices, dataset
        )
        print(
            f"Train subjects: {train_count}, Val subjects: {val_count}, "
            f"Test subjects: {test_count}"
        )

    return fold_splits, test_indices, train_val_labels
