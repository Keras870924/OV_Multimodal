"""
Model evaluation and ensemble prediction functions.
"""

import numpy as np
import torch
import torch.nn as nn

from .utils import calculate_metrics


def evaluate_multimodal_model(model, data_loader, criterion, device):
    """Evaluate a single multimodal model on a data loader.

    Args:
        model: Trained MultimodalModel.
        data_loader: DataLoader for evaluation data.
        criterion: Loss function.
        device: Torch device.

    Returns:
        dict: Evaluation metrics including loss, accuracy, sensitivity, etc.
    """
    model.eval()
    all_labels = []
    all_predictions = []
    all_scores = []
    total_loss = 0.0

    with torch.no_grad():
        for images, text_features, labels in data_loader:
            images = images.to(device)
            text_features = text_features.to(device)
            labels = labels.to(device)

            outputs, swin_outputs, densenet_outputs, text_outputs = model(
                images, text_features
            )
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Use the combined (fused) output for predictions
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            if probabilities.shape[1] == 2:
                all_scores.extend(probabilities[:, 1].cpu().numpy())
            else:
                all_scores.extend(probabilities.max(dim=1)[0].cpu().numpy())

    metrics = calculate_metrics(
        np.array(all_labels), np.array(all_predictions), np.array(all_scores)
    )
    metrics["loss"] = total_loss / len(data_loader)

    return metrics


def ensemble_predict_multimodal(
    models, data_loader, device, class_weights=None, dataset=None
):
    """Perform ensemble prediction using soft voting across multiple models.

    Averages the softmax probabilities from all models and selects the
    class with the highest average probability.

    Args:
        models (list): List of trained MultimodalModel instances.
        data_loader: DataLoader for evaluation data.
        device: Torch device.
        class_weights (Tensor, optional): Class weights for loss computation.
        dataset: Subset dataset for retrieving image paths.

    Returns:
        tuple: (metrics, all_labels, all_probabilities, all_image_paths)
    """
    all_labels = []
    all_probabilities = []
    all_image_paths = []
    total_loss = 0.0
    criterion = (
        nn.CrossEntropyLoss(weight=class_weights)
        if class_weights is not None
        else nn.CrossEntropyLoss()
    )

    with torch.no_grad():
        for batch_idx, (images, text_features, labels) in enumerate(data_loader):
            images = images.to(device)
            text_features = text_features.to(device)
            labels = labels.to(device)

            # Collect predictions from each model
            batch_probabilities = []
            for model in models:
                model.eval()
                outputs, _, _, _ = model(images, text_features)
                probabilities = torch.softmax(outputs, dim=1)
                batch_probabilities.append(probabilities)

            # Soft voting: average probabilities across models
            avg_probabilities = torch.mean(torch.stack(batch_probabilities), dim=0)
            _, predicted = torch.max(avg_probabilities, 1)

            # Compute loss using averaged log-probabilities
            loss = criterion(avg_probabilities.log(), labels)
            total_loss += loss.item()

            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(avg_probabilities.cpu().numpy())

            # Retrieve image paths for this batch
            start_idx = batch_idx * data_loader.batch_size
            end_idx = min(
                (batch_idx + 1) * data_loader.batch_size, len(dataset)
            )
            batch_image_paths = [
                dataset.dataset.image_files[dataset.indices[idx]]
                for idx in range(start_idx, end_idx)
            ]
            all_image_paths.extend(batch_image_paths)

    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    all_predictions = np.argmax(all_probabilities, axis=1)
    all_scores = (
        all_probabilities[:, 1]
        if all_probabilities.shape[1] == 2
        else np.max(all_probabilities, axis=1)
    )

    metrics = calculate_metrics(all_labels, all_predictions, all_scores)
    metrics["loss"] = total_loss / len(data_loader)

    return metrics, all_labels, all_probabilities, all_image_paths
