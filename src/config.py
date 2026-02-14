"""
Configuration constants for the Ovarian Cancer Multimodal Classifier.

This module centralizes all hyperparameters, file paths, and normalization
constants used throughout training and inference.
"""

# ImageNet normalization parameters
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Training hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 150
LEARNING_RATE = 0.00005
WEIGHT_DECAY = 1e-4
PATIENCE = 30
NUM_FOLDS = 5
TEST_SIZE = 0.15
RANDOM_SEED = 42
IMAGE_SIZE = (224, 224)
NUM_WORKERS = 0

# BERT configuration
BERT_MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
BERT_MAX_LENGTH = 128

# Data paths
DATA_DIR = "UltrasoundImage"
METADATA_PATH = "Clinical_data.csv"
LOG_DIR = "log"
MODEL_NAME = "multimodal_ov"
