"""
Training entry point for the Ovarian Cancer Multimodal Classifier.

Usage:
    python train.py

This script launches the full 5-fold cross-validation training pipeline
with subject-level splitting, weighted sampling, CutMix augmentation,
multi-task loss, and ensemble evaluation on the held-out test set.
"""

from src.training import train_multimodal_model

if __name__ == "__main__":
    train_multimodal_model()
