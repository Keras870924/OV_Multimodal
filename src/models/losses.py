"""
Loss functions and data augmentation strategies for training.

Includes Focal Loss for class imbalance, a dynamic multi-task loss for
multi-branch training, and CutMix augmentation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance in classification.

    Down-weights well-classified examples and focuses training on hard,
    misclassified samples.

    Reference:
        Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.

    Args:
        gamma (float): Focusing parameter; higher values focus more on hard examples.
        alpha (float or list): Balancing factor per class.
        reduction (str): Reduction mode ('mean', 'sum', or 'none').
    """

    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            if isinstance(self.alpha, (list, tuple)):
                alpha = torch.tensor(self.alpha).to(inputs.device)
                batch_alpha = alpha[targets]
                loss = batch_alpha * loss
            else:
                pos_mask = (targets == 1).float()
                neg_mask = 1 - pos_mask
                loss = (
                    self.alpha * pos_mask * loss
                    + (1 - self.alpha) * neg_mask * loss
                )

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def improved_multi_task_loss(
    combined_out,
    swin_out,
    densenet_out,
    text_out,
    labels,
    alpha=0.7,
    beta=0.15,
    gamma=0.1,
    delta=0.05,
):
    """Compute multi-task loss with dynamic per-branch weight adjustment.

    Adjusts auxiliary branch weights based on whether the main (combined)
    model already predicts correctly: if the main model is correct but a
    branch is wrong, the branch loss weight is decreased, and vice versa.

    Args:
        combined_out (Tensor): Logits from the fused classifier (B, C).
        swin_out (Tensor): Logits from the Swin auxiliary classifier (B, C).
        densenet_out (Tensor): Logits from the DenseNet auxiliary classifier (B, C).
        text_out (Tensor): Logits from the text auxiliary classifier (B, C).
        labels (Tensor): Ground-truth labels (B,).
        alpha (float): Base weight for the combined branch.
        beta (float): Base weight for the Swin branch.
        gamma (float): Base weight for the DenseNet branch.
        delta (float): Base weight for the text branch.

    Returns:
        Tensor: Scalar total loss.
    """
    criterion = nn.CrossEntropyLoss(reduction="none")

    # Per-sample loss for each branch
    loss_combined = criterion(combined_out, labels)
    loss_swin = criterion(swin_out, labels)
    loss_densenet = criterion(densenet_out, labels)
    loss_text = criterion(text_out, labels)

    # Per-sample correctness for each branch
    _, pred_combined = torch.max(combined_out.data, 1)
    _, pred_swin = torch.max(swin_out.data, 1)
    _, pred_densenet = torch.max(densenet_out.data, 1)
    _, pred_text = torch.max(text_out.data, 1)

    correct_combined = (pred_combined == labels).float()
    correct_swin = (pred_swin == labels).float()
    correct_densenet = (pred_densenet == labels).float()
    correct_text = (pred_text == labels).float()

    # Dynamic weight adjustment: reduce branch weight when main model is
    # already correct but the branch is wrong, and vice versa
    dynamic_beta = torch.where(
        correct_combined > correct_swin,
        torch.ones_like(correct_swin) * (beta * 0.8),
        torch.ones_like(correct_swin) * (beta * 1.2),
    )
    dynamic_gamma = torch.where(
        correct_combined > correct_densenet,
        torch.ones_like(correct_densenet) * (gamma * 0.8),
        torch.ones_like(correct_densenet) * (gamma * 1.2),
    )
    dynamic_delta = torch.where(
        correct_combined > correct_text,
        torch.ones_like(correct_text) * (delta * 0.8),
        torch.ones_like(correct_text) * (delta * 1.2),
    )

    # Weighted loss computation
    weighted_loss_combined = (alpha * loss_combined).mean()
    weighted_loss_swin = (dynamic_beta * loss_swin).mean()
    weighted_loss_densenet = (dynamic_gamma * loss_densenet).mean()
    weighted_loss_text = (dynamic_delta * loss_text).mean()

    total_loss = (
        weighted_loss_combined
        + weighted_loss_swin
        + weighted_loss_densenet
        + weighted_loss_text
    )

    return total_loss


def cutmix(images, text_features, labels, beta=1.0, device=None):
    """Apply CutMix augmentation to a batch of images and text features.

    Randomly cuts a rectangular region from one image and pastes it onto
    another, with proportional label mixing.

    Reference:
        Yun et al., "CutMix: Regularization Strategy to Train Strong Classifiers
        with Localizable Features", ICCV 2019.

    Args:
        images (Tensor): Batch of images (B, C, H, W).
        text_features (Tensor): Batch of text features (B, D).
        labels (Tensor): Batch of labels (B,).
        beta (float): Beta distribution parameter for mixing ratio.
        device: Target torch device.

    Returns:
        tuple: (mixed_images, mixed_text, labels_a, labels_b, lam)
    """
    batch_size = images.size(0)
    lam = np.random.beta(beta, beta)

    # Random permutation for mixing pairs
    indices = torch.randperm(batch_size).to(device)
    shuffled_images = images[indices]
    shuffled_text = text_features[indices]
    shuffled_labels = labels[indices]

    # Compute cut region dimensions
    _, h, w = images.size()[1:]
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(h * cut_ratio)
    cut_w = int(w * cut_ratio)

    # Random center point for the cut
    cx = np.random.randint(w)
    cy = np.random.randint(h)

    # Compute bounding box with clipping
    x1 = np.clip(cx - cut_w // 2, 0, w)
    y1 = np.clip(cy - cut_h // 2, 0, h)
    x2 = np.clip(cx + cut_w // 2, 0, w)
    y2 = np.clip(cy + cut_h // 2, 0, h)

    # Paste cut region from shuffled images
    images[:, :, y1:y2, x1:x2] = shuffled_images[:, :, y1:y2, x1:x2]

    # Recompute lambda based on actual cut area
    lam = 1 - ((x2 - x1) * (y2 - y1) / (h * w))

    # Mix text features proportionally
    mixed_text = text_features * lam + shuffled_text * (1 - lam)

    return images, mixed_text, labels, shuffled_labels, lam
