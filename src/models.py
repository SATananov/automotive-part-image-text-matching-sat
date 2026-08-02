from __future__ import annotations

import torch
from torch import nn

NUMBER_OF_RELATION_CLASSES = 3
NUMBER_OF_PART_CATEGORIES = 8


class TextRelationMLP(nn.Module):
    def __init__(self, text_dimension: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(text_dimension, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, NUMBER_OF_RELATION_CLASSES),
        )

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        return self.network(text)


class ImageEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48, 48),
            nn.ReLU(),
            nn.Dropout(0.15),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.projection(self.features(image))


class ImageRelationCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = ImageEncoder()
        self.classifier = nn.Linear(48, NUMBER_OF_RELATION_CLASSES)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(image))


class MultimodalRelationCNN(nn.Module):
    """CNN + text MLP with auxiliary image/text category objectives."""

    def __init__(
        self,
        text_dimension: int,
        number_of_part_categories: int = NUMBER_OF_PART_CATEGORIES,
    ) -> None:
        super().__init__()
        if number_of_part_categories < 2:
            raise ValueError("At least two part categories are required")
        self.image_encoder = ImageEncoder()
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dimension, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.relation_head = nn.Sequential(
            nn.Linear(48 + 32, 96),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, NUMBER_OF_RELATION_CLASSES),
        )
        self.image_category_head = nn.Linear(48, number_of_part_categories)
        self.text_category_head = nn.Linear(32, number_of_part_categories)

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


class MultimodalRelationCNNNoAuxiliary(nn.Module):
    """Controlled relation-only ablation with the same shared architecture."""

    def __init__(self, text_dimension: int) -> None:
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dimension, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.relation_head = nn.Sequential(
            nn.Linear(48 + 32, 96),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, NUMBER_OF_RELATION_CLASSES),
        )

    def forward(self, image: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        image_features = self.image_encoder(image)
        text_features = self.text_encoder(text)
        return self.relation_head(torch.cat([image_features, text_features], dim=1))


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def count_total_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
