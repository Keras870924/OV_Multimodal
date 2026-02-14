"""
PyTorch Dataset classes for multimodal ultrasound data.

Provides datasets that pair DICOM ultrasound images with clinical text
features for training and inference of the multimodal classifier.
"""

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .dicom_utils import read_dicom_file


class MultimodalDataset(Dataset):
    """Dataset combining ultrasound DICOM images with clinical text features.

    Scans the data directory for class subdirectories, matches DICOM files
    with clinical metadata by ID, and provides (image, text_features, label)
    tuples.

    Args:
        data_dir (str): Root directory containing class subdirectories.
        metadata_path (str): Path to the clinical data CSV file.
        image_transform: Optional torchvision transform for images.
        text_processor (BertTextProcessor): Fitted text feature processor.
    """

    def __init__(self, data_dir, metadata_path, image_transform=None, text_processor=None):
        self.data_dir = data_dir
        self.transform = image_transform

        # Load clinical metadata
        self.metadata = pd.read_csv(metadata_path)

        if text_processor is None:
            raise ValueError("A fitted TextProcessor instance is required.")

        self.text_processor = text_processor
        self.text_features = self.text_processor.transform(self.metadata)

        self.image_files = []
        self.labels = []
        self.feature_indices = []

        # Discover valid class directories
        valid_classes = [
            d
            for d in sorted(os.listdir(data_dir))
            if os.path.isdir(os.path.join(data_dir, d))
            and not d.startswith(".")
            and d not in ["__pycache__", ".git"]
        ]
        class_to_idx = {class_name: idx for idx, class_name in enumerate(valid_classes)}

        # Map each metadata entry to its corresponding DICOM file
        for idx, row in self.metadata.iterrows():
            img_id = row["ID"]
            for class_name in valid_classes:
                class_dir = os.path.join(data_dir, class_name)
                for img_name in os.listdir(class_dir):
                    if img_name == f"{img_id}.dcm":
                        img_path = os.path.join(class_dir, img_name)
                        try:
                            img = read_dicom_file(img_path)
                            if img is not None:
                                self.image_files.append(img_path)
                                self.labels.append(class_to_idx[class_name])
                                self.feature_indices.append(idx)
                        except Exception as e:
                            print(f"Warning: Skipping invalid DICOM file {img_path}: {e}")

        print(f"Total dataset size: {len(self.image_files)} DICOM files")

        if len(self.image_files) == 0:
            print("Warning: No valid DICOM files found in the directory structure!")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = read_dicom_file(img_path)
        label = self.labels[idx]
        text_features = torch.FloatTensor(self.text_features[self.feature_indices[idx]])

        if self.transform:
            image = self.transform(image)

        return image, text_features, label


class MultimodalTransformSubset(Dataset):
    """A subset wrapper that applies a specific transform to dataset samples.

    Allows different transforms (e.g., train vs. validation augmentation)
    to be applied to different subsets of the same base dataset.

    Args:
        dataset (MultimodalDataset): The base dataset.
        indices (list[int]): Indices into the base dataset for this subset.
        transform: Optional torchvision transform to apply.
    """

    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img_path = self.dataset.image_files[real_idx]
        image = read_dicom_file(img_path)
        label = self.dataset.labels[real_idx]
        text_features = self.dataset.text_features[
            self.dataset.feature_indices[real_idx]
        ]

        if self.transform:
            image = self.transform(image)

        return image, torch.FloatTensor(text_features), label
