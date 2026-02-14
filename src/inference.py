"""
Inference pipeline for the multimodal ovarian cancer classifier.

Loads pre-trained fold models and performs ensemble prediction on new
ultrasound images with accompanying clinical data.
"""

import os

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image

from .config import (
    MEAN,
    STD,
    IMAGE_SIZE,
    BERT_MODEL_NAME,
    BERT_MAX_LENGTH,
    METADATA_PATH,
    LOG_DIR,
)
from .data.dicom_utils import read_dicom_file
from .data.text_processor import BertTextProcessor
from .data.transforms import AutoROICropWithAspectRatio
from .models.multimodal_model import MultimodalModel


class MultimodalInference:
    """Ensemble inference engine for the multimodal ovarian cancer classifier.

    Loads all 5 fold models and the fitted text processor on initialization,
    then provides a ``predict()`` method for single-sample inference.

    Args:
        model_dir (str): Directory containing ``best_model_fold{1..5}.pth``.
        metadata_path (str): Path to ``Clinical_data.csv`` (needed to fit
            the text processor's route encoder and age scaler).
        device (torch.device, optional): Inference device. Defaults to CUDA
            if available, otherwise CPU.
    """

    CLASS_NAMES = ["Benign", "Cancer"]

    def __init__(self, model_dir=LOG_DIR, metadata_path=METADATA_PATH, device=None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model_dir = model_dir

        # Fit the text processor on the training metadata so that the
        # route encoder and age scaler have the correct parameters
        print("Loading clinical data and fitting text processor...")
        metadata = pd.read_csv(metadata_path)
        self.text_processor = BertTextProcessor(
            metadata,
            bert_model_name=BERT_MODEL_NAME,
            max_length=BERT_MAX_LENGTH,
        )
        self.text_processor.fit()
        self.text_feature_size = self.text_processor.get_feature_size()

        # Image transform (same as test-time transform during training)
        self.transform = transforms.Compose(
            [
                AutoROICropWithAspectRatio(target_size=IMAGE_SIZE, margin=15),
                transforms.Resize(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD),
            ]
        )

        # Load all fold models
        print("Loading ensemble models...")
        self.models = self._load_models()
        print(f"Loaded {len(self.models)} models on {self.device}")

    def _load_models(self):
        """Load all 5 fold best models from disk.

        Returns:
            list[MultimodalModel]: Loaded models in eval mode.
        """
        models = []
        for fold in range(1, 6):
            model_path = os.path.join(self.model_dir, f"best_model_fold{fold}.pth")
            if os.path.exists(model_path):
                model = MultimodalModel(
                    num_classes=2, text_features=self.text_feature_size
                )
                checkpoint = torch.load(
                    model_path, map_location=self.device, weights_only=False
                )
                model.load_state_dict(checkpoint["model_state_dict"])
                model.to(self.device)
                model.eval()
                models.append(model)
                print(f"  Loaded fold {fold} model from {model_path}")
            else:
                print(f"  Warning: Model not found at {model_path}")
        return models

    def predict(self, image, age, transducer_type, report_text):
        """Run ensemble prediction on a single sample.

        Args:
            image: PIL Image, or file path (str) to a DICOM/PNG/JPG image.
            age (float): Patient age in years.
            transducer_type (str): Ultrasound transducer type
                (``'Trandabdominal'`` or ``'Transvaginal'``).
            report_text (str): Free-text ultrasound report.

        Returns:
            dict: Prediction results containing:
                - ``prediction`` (str): Class name ('Benign' or 'Cancer').
                - ``confidence`` (float): Probability of the predicted class.
                - ``probabilities`` (dict): Per-class probabilities.
                - ``fold_predictions`` (list[dict]): Per-fold probabilities.
        """
        # Handle image input (path or PIL Image)
        if isinstance(image, str):
            if image.lower().endswith(".dcm"):
                image = read_dicom_file(image)
            else:
                image = Image.open(image).convert("RGB")

        if image is None:
            raise ValueError("Failed to load the provided image.")

        # Apply test-time image transform
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Build a single-row DataFrame for clinical feature extraction
        clinical_df = pd.DataFrame(
            {
                "UltrasoundRoute": [transducer_type],
                "REPORT_TEXT": [report_text],
                "AGE": [float(age)],
            }
        )
        text_features = self.text_processor.transform(clinical_df)
        text_tensor = torch.FloatTensor(text_features).to(self.device)

        # Collect predictions from each fold model
        fold_probabilities = []
        with torch.no_grad():
            for model in self.models:
                outputs, _, _, _ = model(image_tensor, text_tensor)
                probs = torch.softmax(outputs, dim=1)
                fold_probabilities.append(probs.cpu().numpy()[0])

        # Ensemble: average softmax probabilities across folds
        fold_probabilities = np.array(fold_probabilities)
        ensemble_probs = fold_probabilities.mean(axis=0)
        prediction_idx = int(np.argmax(ensemble_probs))

        return {
            "prediction": self.CLASS_NAMES[prediction_idx],
            "confidence": float(ensemble_probs[prediction_idx]),
            "probabilities": {
                name: float(ensemble_probs[i])
                for i, name in enumerate(self.CLASS_NAMES)
            },
            "fold_predictions": [
                {
                    "fold": i + 1,
                    "probabilities": {
                        name: float(fold_probabilities[i][j])
                        for j, name in enumerate(self.CLASS_NAMES)
                    },
                }
                for i in range(len(fold_probabilities))
            ],
        }
