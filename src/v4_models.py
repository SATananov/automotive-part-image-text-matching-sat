from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

NUMBER_OF_RELATION_CLASSES = 3
NUMBER_OF_PART_CATEGORIES_V4 = 50
IMAGE_FEATURE_DIMENSION = 128
TEXT_FEATURE_DIMENSION = 128


class ResNet18ImageEncoderV4(nn.Module):
    """ResNet18 image encoder with an optional frozen pretrained backbone."""

    def __init__(
        self,
        *,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        output_dimension: int = IMAGE_FEATURE_DIMENSION,
    ) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        feature_dimension = int(backbone.fc.in_features)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.projection = nn.Sequential(
            nn.Linear(feature_dimension, output_dimension),
            nn.ReLU(),
            nn.Dropout(0.20),
        )
        self._backbone_frozen = False
        self.set_backbone_trainable(not freeze_backbone)

    @property
    def backbone_frozen(self) -> bool:
        return self._backbone_frozen

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = trainable
        self._backbone_frozen = not trainable
        if self._backbone_frozen:
            self.backbone.eval()

    def train(self, mode: bool = True) -> "ResNet18ImageEncoderV4":
        super().train(mode)
        if self._backbone_frozen:
            self.backbone.eval()
        return self

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.backbone(image)
        return self.projection(features)


class TextEncoderV4(nn.Module):
    def __init__(
        self,
        text_dimension: int,
        output_dimension: int = TEXT_FEATURE_DIMENSION,
    ) -> None:
        super().__init__()
        if text_dimension < 2:
            raise ValueError("Text feature dimension must be at least two")
        self.network = nn.Sequential(
            nn.Linear(text_dimension, 256),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(256, output_dimension),
            nn.ReLU(),
            nn.Dropout(0.10),
        )

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        return self.network(text)


class MultimodalRelationModelV4(nn.Module):
    """Pretrained ResNet18 + text MLP with two helper category tasks."""

    def __init__(
        self,
        text_dimension: int,
        *,
        pretrained_image_encoder: bool = True,
        freeze_image_backbone: bool = True,
        number_of_part_categories: int = NUMBER_OF_PART_CATEGORIES_V4,
    ) -> None:
        super().__init__()
        if number_of_part_categories != NUMBER_OF_PART_CATEGORIES_V4:
            raise ValueError("Dataset V4 expects exactly 50 part categories")
        self.image_encoder = ResNet18ImageEncoderV4(
            pretrained=pretrained_image_encoder,
            freeze_backbone=freeze_image_backbone,
        )
        self.text_encoder = TextEncoderV4(text_dimension)
        combined_dimension = IMAGE_FEATURE_DIMENSION + TEXT_FEATURE_DIMENSION
        self.relation_head = nn.Sequential(
            nn.Linear(combined_dimension, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 96),
            nn.ReLU(),
            nn.Linear(96, NUMBER_OF_RELATION_CLASSES),
        )
        self.image_category_head = nn.Linear(
            IMAGE_FEATURE_DIMENSION, number_of_part_categories
        )
        self.text_category_head = nn.Linear(
            TEXT_FEATURE_DIMENSION, number_of_part_categories
        )

    def forward(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_features = self.image_encoder(image)
        text_features = self.text_encoder(text)
        relation_logits = self.relation_head(
            torch.cat([image_features, text_features], dim=1)
        )
        return (
            relation_logits,
            self.image_category_head(image_features),
            self.text_category_head(text_features),
        )


class MultimodalRelationModelV4NoAuxiliary(nn.Module):
    """Relation-only ablation using the same shared encoders and relation head."""

    def __init__(
        self,
        text_dimension: int,
        *,
        pretrained_image_encoder: bool = True,
        freeze_image_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.image_encoder = ResNet18ImageEncoderV4(
            pretrained=pretrained_image_encoder,
            freeze_backbone=freeze_image_backbone,
        )
        self.text_encoder = TextEncoderV4(text_dimension)
        self.relation_head = nn.Sequential(
            nn.Linear(IMAGE_FEATURE_DIMENSION + TEXT_FEATURE_DIMENSION, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 96),
            nn.ReLU(),
            nn.Linear(96, NUMBER_OF_RELATION_CLASSES),
        )

    def forward(self, image: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        image_features = self.image_encoder(image)
        text_features = self.text_encoder(text)
        return self.relation_head(torch.cat([image_features, text_features], dim=1))


def count_trainable_parameters_v4(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def count_total_parameters_v4(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
