"""
Clinical text feature extraction using Bio_ClinicalBERT.

Processes structured clinical data (transducer type, patient age) and
free-text ultrasound reports into a unified feature vector using
pre-trained BERT embeddings.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from transformers import BertTokenizer, BertModel


class BertTextProcessor:
    """Extract clinical features using Bio_ClinicalBERT for report text encoding.

    Combines one-hot encoded transducer route, standardized age, and
    BERT [CLS] token embeddings of ultrasound reports into a single
    feature vector.

    Args:
        df (pd.DataFrame): Training DataFrame used to fit encoders.
        route_col (str): Column name for the ultrasound transducer route.
        report_col (str): Column name for the free-text ultrasound report.
        age_col (str): Column name for patient age.
        bert_model_name (str): HuggingFace model identifier for BERT.
        max_length (int): Maximum token sequence length for BERT.
    """

    def __init__(
        self,
        df,
        route_col="UltrasoundRoute",
        report_col="REPORT_TEXT",
        age_col="AGE",
        bert_model_name="emilyalsentzer/Bio_ClinicalBERT",
        max_length=128,
    ):
        self.df = df
        self.route_col = route_col
        self.report_col = report_col
        self.age_col = age_col

        # Label encoder for transducer route
        self.route_encoder = LabelEncoder()

        # Standard scaler for age normalization
        self.age_scaler = StandardScaler()

        # BERT model and tokenizer
        self.bert_model_name = bert_model_name
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.model = BertModel.from_pretrained(bert_model_name)
        self.max_length = max_length
        self.bert_dim = self.model.config.hidden_size  # Typically 768

    def fit(self):
        """Fit the route encoder and age scaler on the training data.

        Returns:
            BertTextProcessor: self, for method chaining.
        """
        # Fit the route label encoder
        self.route_encoder.fit(self.df[self.route_col].fillna("Unknown"))

        # Fit the age standard scaler
        self.age_scaler.fit(self.df[self.age_col].values.reshape(-1, 1))

        # Set BERT to evaluation mode (pre-trained, no fitting needed)
        self.model.eval()

        return self

    def transform(self, df):
        """Transform a DataFrame into a numerical feature matrix.

        Args:
            df (pd.DataFrame): DataFrame with route, report, and age columns.

        Returns:
            np.ndarray: Feature matrix of shape (n_samples, feature_size).
        """
        # Encode transducer route as one-hot
        route_encoded = self.route_encoder.transform(
            df[self.route_col].fillna("Unknown")
        )
        route_one_hot = pd.get_dummies(route_encoded, prefix="route")

        # Standardize age
        age_scaled = self.age_scaler.transform(
            df[self.age_col].values.reshape(-1, 1)
        )

        # Extract BERT embeddings for report text
        report_embeddings = self._get_bert_embeddings(
            df[self.report_col].fillna("")
        )

        # Concatenate all features: [route_one_hot | age | bert_embedding]
        num_route_features = len(self.route_encoder.classes_)
        features = np.zeros((len(df), num_route_features + 1 + self.bert_dim))

        features[:, :num_route_features] = route_one_hot
        features[:, num_route_features : num_route_features + 1] = age_scaled
        features[:, num_route_features + 1 :] = report_embeddings

        return features

    def _get_bert_embeddings(self, reports):
        """Extract BERT [CLS] token embeddings for a series of text reports.

        Args:
            reports: Iterable of report text strings.

        Returns:
            np.ndarray: Embedding matrix of shape (n_reports, bert_dim).
        """
        embeddings = np.zeros((len(reports), self.bert_dim))

        with torch.no_grad():
            for i, report in enumerate(reports):
                encoded_input = self.tokenizer(
                    report,
                    return_tensors="pt",
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                )

                outputs = self.model(**encoded_input)

                # Use the [CLS] token embedding as the text representation
                cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings[i] = cls_embedding

        return embeddings

    def get_feature_size(self):
        """Return the total feature vector dimensionality.

        Returns:
            int: Number of features (route_classes + 1 age + bert_dim).
        """
        return len(self.route_encoder.classes_) + 1 + self.bert_dim
