"""
Multimodal classification model combining Swin Transformer, DenseNet121,
and BERT-based clinical text features for ovarian cancer diagnosis.

Architecture:
    - Swin-T branch: extracts visual features (768-dim)
    - DenseNet121 branch: extracts complementary visual features (1024-dim)
    - Text branch: projects BERT clinical features through MLP (128-dim)
    - Fusion layer: concatenates all features and classifies via MLP
"""

import torch
import torch.nn as nn
from torchvision.models import swin_t, Swin_T_Weights
from torchvision.models import densenet121, DenseNet121_Weights


class MultimodalModel(nn.Module):
    """Multimodal model fusing Swin Transformer, DenseNet121, and clinical text.

    Uses a late-fusion strategy where each modality is processed independently
    before concatenation and joint classification. Auxiliary classifiers on
    individual branches enable multi-task learning.

    Args:
        num_classes (int): Number of output classes.
        swin_weight (float): Initial learnable weight for the Swin branch.
        text_features (int): Dimensionality of the input text feature vector.
    """

    def __init__(self, num_classes, swin_weight=0.5, text_features=None):
        super(MultimodalModel, self).__init__()

        # Swin Transformer branch (ImageNet pre-trained)
        self.swin = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        swin_features = self.swin.head.in_features  # 768
        self.swin.head = nn.Identity()  # Remove classification head

        # DenseNet121 branch (ImageNet pre-trained)
        self.densenet = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        densenet_features = self.densenet.classifier.in_features  # 1024
        self.densenet.classifier = nn.Identity()  # Remove classification head

        # Text feature input size (route one-hot + age + BERT embedding)
        self.text_feature_size = text_features

        # Swin feature processing layers
        self.swin_dropout = nn.Dropout(0.5)
        self.swin_bn = nn.BatchNorm1d(swin_features)

        # DenseNet feature processing layers
        self.densenet_dropout = nn.Dropout(0.5)
        self.densenet_bn = nn.BatchNorm1d(densenet_features)

        # Text feature MLP: projects BERT features to a compact representation
        # Uses GELU activation and LayerNorm to stay consistent with BERT
        self.text_fc = nn.Sequential(
            nn.Linear(self.text_feature_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
        )

        # Feature fusion layer
        combined_features = swin_features + densenet_features + 128
        self.fusion_layer = nn.Sequential(
            nn.Linear(combined_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Main classifier
        self.classifier = nn.Linear(256, num_classes)

        # Auxiliary branch classifiers (for multi-task training)
        self.swin_classifier = nn.Linear(swin_features, num_classes)
        self.densenet_classifier = nn.Linear(densenet_features, num_classes)
        self.text_classifier = nn.Linear(128, num_classes)

        # Learnable fusion weight
        self.swin_weight = nn.Parameter(
            torch.tensor([swin_weight]), requires_grad=True
        )

    def forward(self, image, text_features):
        """Forward pass through all branches and fusion.

        Args:
            image (Tensor): Batch of images, shape (B, 3, 224, 224).
            text_features (Tensor): Batch of text features, shape (B, text_feature_size).

        Returns:
            tuple: (combined_out, swin_out, densenet_out, text_out)
                Each is a logit tensor of shape (B, num_classes).
        """
        # Swin Transformer branch
        swin_feat = self.swin(image)
        swin_feat = self.swin_dropout(swin_feat)
        if swin_feat.size(0) > 1:  # Skip BatchNorm for single-sample batches
            swin_feat = self.swin_bn(swin_feat)
        swin_out = self.swin_classifier(swin_feat)

        # DenseNet branch
        densenet_feat = self.densenet(image)
        densenet_feat = self.densenet_dropout(densenet_feat)
        if densenet_feat.size(0) > 1:  # Skip BatchNorm for single-sample batches
            densenet_feat = self.densenet_bn(densenet_feat)
        densenet_out = self.densenet_classifier(densenet_feat)

        # Text feature branch (BERT embeddings through MLP)
        text_feat = self.text_fc(text_features)
        text_out = self.text_classifier(text_feat)

        # Late fusion: concatenate all branch features
        combined_feat = torch.cat([swin_feat, densenet_feat, text_feat], dim=1)
        fused_feat = self.fusion_layer(combined_feat)
        combined_out = self.classifier(fused_feat)

        return combined_out, swin_out, densenet_out, text_out
