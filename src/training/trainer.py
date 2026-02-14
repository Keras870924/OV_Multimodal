"""
Main training loop for the multimodal ovarian cancer classifier.

Implements 5-fold cross-validation with subject-level splitting,
weighted sampling, CutMix augmentation, multi-task loss, and
early stopping.
"""

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from ..config import (
    MEAN,
    STD,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    PATIENCE,
    NUM_FOLDS,
    TEST_SIZE,
    RANDOM_SEED,
    IMAGE_SIZE,
    NUM_WORKERS,
    BERT_MODEL_NAME,
    BERT_MAX_LENGTH,
    DATA_DIR,
    METADATA_PATH,
    LOG_DIR,
    MODEL_NAME,
)
from ..data import (
    BertTextProcessor,
    MultimodalDataset,
    MultimodalTransformSubset,
    AutoROICropWithAspectRatio,
    UltrasoundSpeckleNoise,
    UltrasoundShadowAugmentation,
    UltrasoundEnhancementAugmentation,
)
from ..models import MultimodalModel, FocalLoss, improved_multi_task_loss, cutmix
from .evaluation import evaluate_multimodal_model, ensemble_predict_multimodal
from .utils import (
    calculate_confidence_interval,
    save_checkpoint,
    cleanup_checkpoints,
    save_final_best_model,
    create_weighted_sampler,
    extract_subject_id,
    verify_no_subject_leakage,
    subject_level_split,
)


def train_multimodal_model():
    """Execute the full training pipeline with 5-fold cross-validation.

    Steps:
        1. Set random seeds for reproducibility.
        2. Load and preprocess clinical metadata with BertTextProcessor.
        3. Build the multimodal dataset from DICOM images and metadata.
        4. Perform subject-level train/val/test splitting.
        5. Train each fold with weighted sampling, CutMix, and early stopping.
        6. Evaluate using ensemble prediction on the held-out test set.
        7. Save all results and metrics to CSV and JSON files.
    """
    # -------------------------------------------------------------------------
    # 1. Reproducibility
    # -------------------------------------------------------------------------
    seed = RANDOM_SEED
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Using {torch.cuda.device_count()} GPU(s): {device}")
    else:
        print(f"Using device: {device}")
        return

    print(
        f"Using {NUM_WORKERS} workers for data loading "
        f"(single process mode to avoid shared memory issues)"
    )

    # -------------------------------------------------------------------------
    # 2. Clinical data preprocessing
    # -------------------------------------------------------------------------
    metadata = pd.read_csv(METADATA_PATH)
    text_processor = BertTextProcessor(
        metadata,
        bert_model_name=BERT_MODEL_NAME,
        max_length=BERT_MAX_LENGTH,
    )
    text_processor.fit()
    text_feature_size = text_processor.get_feature_size()

    checkpoint_dir = os.path.join(LOG_DIR, MODEL_NAME)
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {checkpoint_dir}")

    # -------------------------------------------------------------------------
    # 3. Image transforms
    # -------------------------------------------------------------------------
    train_transform = transforms.Compose(
        [
            AutoROICropWithAspectRatio(target_size=IMAGE_SIZE, margin=15),
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=(-15, 15)),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
                fill=0,
            ),
            transforms.ToTensor(),
            UltrasoundSpeckleNoise(noise_variance=0.03),
            UltrasoundShadowAugmentation(shadow_intensity=0.4, shadow_size=0.15),
            UltrasoundEnhancementAugmentation(
                enhancement_intensity=1.2, enhancement_size=0.15
            ),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )

    test_transform = transforms.Compose(
        [
            AutoROICropWithAspectRatio(target_size=IMAGE_SIZE, margin=15),
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )

    # -------------------------------------------------------------------------
    # 4. Dataset and splitting
    # -------------------------------------------------------------------------
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory does not exist: {DATA_DIR}")
        return

    class_names = [
        d
        for d in sorted(os.listdir(DATA_DIR))
        if os.path.isdir(os.path.join(DATA_DIR, d))
        and not d.startswith(".")
        and d not in ["__pycache__", ".git", ".ipynb_checkpoints"]
    ]
    print(f"Found classes: {class_names}")

    dataset = MultimodalDataset(
        data_dir=DATA_DIR,
        metadata_path=METADATA_PATH,
        image_transform=None,
        text_processor=text_processor,
    )
    total_size = len(dataset)
    print(f"Total dataset size: {total_size}")

    for i, class_name in enumerate(class_names):
        count = sum(1 for label in dataset.labels if label == i)
        print(f"Class {class_name}: {count} images")

    fold_splits, test_indices, train_val_labels = subject_level_split(
        dataset, test_size=TEST_SIZE, num_folds=NUM_FOLDS, random_state=seed
    )

    # Create test set
    test_dataset = MultimodalTransformSubset(
        dataset, test_indices, transform=test_transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    # Log test set class distribution
    all_labels = dataset.labels
    test_labels = [all_labels[idx] for idx in test_indices]
    test_counts = {}
    for i, class_name in enumerate(class_names):
        test_counts[class_name] = test_labels.count(i)

    print("\nTest set class distribution:")
    for class_name, count in test_counts.items():
        print(f"  {class_name}: {count} images")

    # Final verification: no subject leakage in test set
    print("\n=== Final test set verification ===")
    test_subject_ids = set()
    for idx in test_indices:
        img_path = dataset.image_files[idx]
        filename = os.path.basename(img_path)
        subject_id = extract_subject_id(filename)
        test_subject_ids.add(subject_id)
    print(f"Test set contains {len(test_subject_ids)} unique subjects")

    # -------------------------------------------------------------------------
    # 5. Cross-validation training
    # -------------------------------------------------------------------------
    best_fold_metrics = {}
    train_counts_per_fold = []
    val_counts_per_fold = []
    num_classes = len(class_names)

    for fold, (train_indices, val_indices) in enumerate(fold_splits):
        print(f"\n{'='*60}")
        print(f"Training Fold {fold + 1}/{NUM_FOLDS}")
        print(f"{'='*60}")

        # Verify no subject leakage for this fold
        print(f"Fold {fold + 1} detailed verification:")
        train_subject_count, val_subject_count, test_subject_count = (
            verify_no_subject_leakage(
                train_indices, val_indices, test_indices, dataset
            )
        )
        print(f"  Train subjects: {train_subject_count}")
        print(f"  Val subjects: {val_subject_count}")
        print(f"  Test subjects: {test_subject_count}")

        train_subset = MultimodalTransformSubset(
            dataset, train_indices, transform=train_transform
        )
        val_subset = MultimodalTransformSubset(
            dataset, val_indices, transform=test_transform
        )

        # Create weighted sampler for class imbalance
        train_labels = [all_labels[idx] for idx in train_indices]
        sampler = create_weighted_sampler(train_labels)

        train_loader = DataLoader(
            train_subset,
            batch_size=BATCH_SIZE,
            sampler=sampler,
            num_workers=NUM_WORKERS,
            pin_memory=False,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=False,
        )

        # Log class distributions
        train_counts = {}
        for i, class_name in enumerate(class_names):
            train_counts[class_name] = train_labels.count(i)
        train_counts_per_fold.append(train_counts)

        val_labels = [all_labels[idx] for idx in val_indices]
        val_counts = {}
        for i, class_name in enumerate(class_names):
            val_counts[class_name] = val_labels.count(i)
        val_counts_per_fold.append(val_counts)

        print("Training set class distribution:")
        for class_name, count in train_counts.items():
            print(f"  {class_name}: {count} images")
        print("Validation set class distribution:")
        for class_name, count in val_counts.items():
            print(f"  {class_name}: {count} images")

        # Check batch balance with weighted sampler
        print("Checking batch balance with weighted sampler...")
        batch_class_counts = np.zeros(len(class_names), dtype=int)
        num_batches_to_check = min(10, len(train_loader))
        for i, (_, _, batch_labels) in enumerate(train_loader):
            if i >= num_batches_to_check:
                break
            for cls in range(len(class_names)):
                batch_class_counts[cls] += (batch_labels == cls).sum().item()

        print("Class distribution in sampled batches:")
        for i, class_name in enumerate(class_names):
            print(f"  {class_name}: {batch_class_counts[i]} samples")

        # Initialize model, optimizer, scheduler
        model = MultimodalModel(
            num_classes, swin_weight=0.5, text_features=text_feature_size
        ).to(device)

        focal_loss = FocalLoss(gamma=2.0, alpha=0.25)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
        )

        best_val_acc = 0.0
        best_val_loss = float("inf")
        early_stop_counter = 0

        # ----- Epoch loop -----
        for epoch in range(NUM_EPOCHS):
            model.train()
            running_loss = 0.0

            for i, (images, text_features, labels) in enumerate(train_loader):
                images = images.to(device)
                text_features = text_features.to(device)
                labels = labels.to(device)

                # Apply CutMix with 50% probability
                if np.random.random() < 0.5:
                    images, mixed_text, labels_a, labels_b, lam = cutmix(
                        images, text_features, labels, beta=1.0, device=device
                    )
                    combined_out, swin_out, densenet_out, text_out = model(
                        images, mixed_text
                    )
                    loss_a = improved_multi_task_loss(
                        combined_out, swin_out, densenet_out, text_out, labels_a
                    )
                    loss_b = improved_multi_task_loss(
                        combined_out, swin_out, densenet_out, text_out, labels_b
                    )
                    loss = lam * loss_a + (1 - lam) * loss_b
                else:
                    combined_out, swin_out, densenet_out, text_out = model(
                        images, text_features
                    )
                    loss = improved_multi_task_loss(
                        combined_out, swin_out, densenet_out, text_out, labels
                    )
                    # Add Focal Loss as auxiliary loss on the combined output
                    loss += focal_loss(combined_out, labels) * 0.2

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                running_loss += loss.item()
                if (i + 1) % 10 == 0:
                    print(
                        f"Epoch [{epoch + 1}/{NUM_EPOCHS}], "
                        f"Step [{i + 1}/{len(train_loader)}], "
                        f"Loss: {running_loss / 10:.4f}"
                    )
                    running_loss = 0.0

            # Validation
            val_metrics = evaluate_multimodal_model(
                model, val_loader, nn.CrossEntropyLoss(), device
            )
            print(
                f'Epoch [{epoch + 1}/{NUM_EPOCHS}] - '
                f'Val Loss: {val_metrics["loss"]:.4f}, '
                f'Accuracy: {val_metrics["accuracy"]:.2f}%'
            )

            # Log learnable fusion weights
            if not isinstance(model, nn.DataParallel):
                weight = torch.sigmoid(model.swin_weight.data)
                print(
                    f"Current Swin Weight: {weight.item():.4f}, "
                    f"DenseNet Weight: {1 - weight.item():.4f}"
                )

            # Check for improvement
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_val_acc = val_metrics["accuracy"]
                early_stop_counter = 0
                is_best = True
                best_fold_metrics[fold + 1] = val_metrics
                print(
                    f"Best model saved with loss: {best_val_loss:.4f} "
                    f"and accuracy: {best_val_acc:.2f}%"
                )
            else:
                early_stop_counter += 1
                is_best = False
                print(f"Early stopping counter: {early_stop_counter}/{PATIENCE}")

            scheduler.step(val_metrics["accuracy"])

            # Save checkpoint
            model_to_save = (
                model.module if isinstance(model, nn.DataParallel) else model
            )
            save_checkpoint(
                model_to_save,
                optimizer,
                epoch + 1,
                val_metrics,
                fold + 1,
                is_best,
                checkpoint_dir,
            )

            if early_stop_counter >= PATIENCE:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

        cleanup_checkpoints(checkpoint_dir, fold + 1, best_val_acc)
        print(f"Fold {fold + 1} finished! Best validation accuracy: {best_val_acc:.2f}%")

    save_final_best_model(best_fold_metrics, checkpoint_dir)

    # -------------------------------------------------------------------------
    # 6. Validation summary
    # -------------------------------------------------------------------------
    metrics_names = [
        "accuracy",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "auc",
        "precision",
        "recall",
        "f1_score",
    ]
    val_summary = {}
    for metric in metrics_names:
        values = [m[metric] for m in best_fold_metrics.values()]
        mean_value = np.mean(values)
        ci_low, ci_high = calculate_confidence_interval(values)
        val_summary[f"{metric}_mean"] = mean_value
        val_summary[f"{metric}_ci_low"] = ci_low
        val_summary[f"{metric}_ci_high"] = ci_high

    # -------------------------------------------------------------------------
    # 7. Ensemble prediction on test set
    # -------------------------------------------------------------------------
    print("\nPerforming ensemble prediction with all fold best models...")
    ensemble_models = []
    for fold in best_fold_metrics.keys():
        best_model_path = os.path.join(
            checkpoint_dir,
            f'best_model_fold{fold}_acc{best_fold_metrics[fold]["accuracy"]:.2f}.pth',
        )
        if os.path.exists(best_model_path):
            model = MultimodalModel(
                num_classes, text_features=text_feature_size
            ).to(device)
            checkpoint = torch.load(best_model_path, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            ensemble_models.append(model)
        else:
            print(f"Warning: Best model not found at {best_model_path} for fold {fold}")

    if not ensemble_models:
        print("Error: No models available for ensemble prediction")
        return

    test_metrics, test_labels_arr, test_probabilities, test_image_paths = (
        ensemble_predict_multimodal(
            ensemble_models, test_loader, device, dataset=test_dataset
        )
    )
    print(
        f"Ensemble test metrics: Accuracy: {test_metrics['accuracy']:.2f}%, "
        f"Loss: {test_metrics['loss']:.4f}"
    )
    print(
        f"Sensitivity: {test_metrics['sensitivity']:.2f}%, "
        f"Specificity: {test_metrics['specificity']:.2f}%"
    )

    # Save per-sample prediction probabilities
    prob_df = pd.DataFrame(
        test_probabilities,
        columns=[f"Prob_Class_{i}" for i in range(num_classes)],
    )
    prob_df["Image_Path"] = test_image_paths
    prob_df["True_Label"] = test_labels_arr
    prob_df["Predicted_Label"] = np.argmax(test_probabilities, axis=1)
    prob_csv_path = os.path.join(checkpoint_dir, "test_predictions_probabilities.csv")
    prob_df.to_csv(prob_csv_path, index=False)
    print(f"Per-sample prediction probabilities saved to '{prob_csv_path}'")

    # -------------------------------------------------------------------------
    # 8. Summary report
    # -------------------------------------------------------------------------
    print("\n=== Dataset Class Counts ===")
    for fold_idx in range(NUM_FOLDS):
        print(f"\nFold {fold_idx + 1}:")
        print("Training set class distribution:")
        for class_name, count in train_counts_per_fold[fold_idx].items():
            print(f"  {class_name}: {count} images")
        print("Validation set class distribution:")
        for class_name, count in val_counts_per_fold[fold_idx].items():
            print(f"  {class_name}: {count} images")

    print("\nTest set class distribution:")
    for class_name, count in test_counts.items():
        print(f"  {class_name}: {count} images")

    # Build results table
    results = []
    results.append(
        {
            "Dataset": "Validation",
            "TP": 0,
            "FP": 0,
            "FN": 0,
            "TN": 0,
            "Accuracy": val_summary["accuracy_mean"],
            "CI_Lower": val_summary["accuracy_ci_low"],
            "CI_Upper": val_summary["accuracy_ci_high"],
            "Sensitivity": val_summary["sensitivity_mean"],
            "CI_Lower_Sensitivity": val_summary["sensitivity_ci_low"],
            "CI_Upper_Sensitivity": val_summary["sensitivity_ci_high"],
            "Specificity": val_summary["specificity_mean"],
            "CI_Lower_Specificity": val_summary["specificity_ci_low"],
            "CI_Upper_Specificity": val_summary["specificity_ci_high"],
            "ppv": val_summary["ppv_mean"],
            "CI_Lower_ppv": val_summary["ppv_ci_low"],
            "CI_Upper_ppv": val_summary["ppv_ci_high"],
            "npv": val_summary["npv_mean"],
            "CI_Lower_npv": val_summary["npv_ci_low"],
            "CI_Upper_npv": val_summary["npv_ci_high"],
            "auc": val_summary["auc_mean"],
            "CI_Lower_auc": val_summary["auc_ci_low"],
            "CI_Upper_auc": val_summary["auc_ci_high"],
            "precision": val_summary["precision_mean"],
            "recall": val_summary["recall_mean"],
            "f1_score": val_summary["f1_score_mean"],
            "CI_Lower_f1_score": val_summary["f1_score_ci_low"],
            "CI_Upper_f1_score": val_summary["f1_score_ci_high"],
        }
    )

    results.append(
        {
            "Dataset": "Test (Ensemble)",
            "TP": test_metrics.get("TP", 0),
            "FP": test_metrics.get("FP", 0),
            "FN": test_metrics.get("FN", 0),
            "TN": test_metrics.get("TN", 0),
            "Accuracy": test_metrics["accuracy"],
            "CI_Lower": test_metrics["accuracy"],
            "CI_Upper": test_metrics["accuracy"],
            "Sensitivity": test_metrics["sensitivity"],
            "CI_Lower_Sensitivity": test_metrics["sensitivity"],
            "CI_Upper_Sensitivity": test_metrics["sensitivity"],
            "Specificity": test_metrics["specificity"],
            "CI_Lower_Specificity": test_metrics["specificity"],
            "CI_Upper_Specificity": test_metrics["specificity"],
            "ppv": test_metrics["ppv"],
            "CI_Lower_ppv": test_metrics["ppv"],
            "CI_Upper_ppv": test_metrics["ppv"],
            "npv": test_metrics["npv"],
            "CI_Lower_npv": test_metrics["npv"],
            "CI_Upper_npv": test_metrics["npv"],
            "auc": test_metrics["auc"],
            "CI_Lower_auc": test_metrics["auc"],
            "CI_Upper_auc": test_metrics["auc"],
            "precision": test_metrics["precision"],
            "recall": test_metrics["recall"],
            "f1_score": test_metrics["f1_score"],
            "CI_Lower_f1_score": test_metrics["f1_score"],
            "CI_Upper_f1_score": test_metrics["f1_score"],
        }
    )

    results_csv_path = os.path.join(
        checkpoint_dir, "model_evaluation_results_oversampling.csv"
    )
    df = pd.DataFrame(results)
    df.to_csv(results_csv_path, index=False)
    print(f"\nResults saved to '{results_csv_path}'")

    # Subject-level statistics
    print("\n=== Subject-Level Statistics ===")
    for fold_idx in range(NUM_FOLDS):
        train_indices, val_indices = fold_splits[fold_idx]

        train_subjects = set()
        for idx in train_indices:
            img_path = dataset.image_files[idx]
            filename = os.path.basename(img_path)
            subject_id = extract_subject_id(filename)
            train_subjects.add(subject_id)

        val_subjects = set()
        for idx in val_indices:
            img_path = dataset.image_files[idx]
            filename = os.path.basename(img_path)
            subject_id = extract_subject_id(filename)
            val_subjects.add(subject_id)

        print(f"\nFold {fold_idx + 1}:")
        print(
            f"  Train: {len(train_subjects)} subjects, {len(train_indices)} images"
        )
        print(
            f"  Val: {len(val_subjects)} subjects, {len(val_indices)} images"
        )

    test_subjects = set()
    for idx in test_indices:
        img_path = dataset.image_files[idx]
        filename = os.path.basename(img_path)
        subject_id = extract_subject_id(filename)
        test_subjects.add(subject_id)

    print(
        f"\nTest: {len(test_subjects)} subjects, {len(test_indices)} images"
    )
