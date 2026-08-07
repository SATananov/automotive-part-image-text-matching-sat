from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.v4_data import LABELS, load_relations_v4


@dataclass
class TrainingResultV4:
    name: str
    best_epoch: int
    accuracy: float
    macro_f1: float
    confusion_matrix: list[list[int]]
    history: list[dict[str, float | int]]


def set_seed_v4(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_development_relations_v4() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load only the two splits that are allowed during model development."""
    train = load_relations_v4("train")
    validation = load_relations_v4("validation")
    return train, validation


class TextOnlyClassifierV4(nn.Module):
    def __init__(self, text_dimension: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(text_dimension, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, len(LABELS)),
        )

    def forward(self, image: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        del image
        return self.network(text)


class ImageOnlyClassifierV4(nn.Module):
    def __init__(self, image_dimension: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(image_dimension, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, len(LABELS)),
        )

    def forward(self, image: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        del text
        return self.network(image)


class MultimodalClassifierV4(nn.Module):
    def __init__(
        self,
        image_dimension: int,
        text_dimension: int,
        *,
        auxiliary_heads: bool,
        number_of_categories: int = 50,
    ) -> None:
        super().__init__()
        self.auxiliary_heads = auxiliary_heads
        self.image_encoder = nn.Sequential(
            nn.Linear(image_dimension, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
        )
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dimension, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
        )
        self.relation_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, len(LABELS)),
        )
        if auxiliary_heads:
            self.image_category_head = nn.Linear(128, number_of_categories)
            self.text_category_head = nn.Linear(128, number_of_categories)

    def forward(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_features = self.image_encoder(image)
        text_features = self.text_encoder(text)
        relation = self.relation_head(torch.cat([image_features, text_features], dim=1))
        if not self.auxiliary_heads:
            return relation
        return (
            relation,
            self.image_category_head(image_features),
            self.text_category_head(text_features),
        )


def _make_loader(
    image_features: np.ndarray,
    text_features: np.ndarray,
    relation_labels: np.ndarray,
    image_categories: np.ndarray,
    text_categories: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(image_features, dtype=np.float32)),
        torch.from_numpy(np.asarray(text_features, dtype=np.float32)),
        torch.from_numpy(np.asarray(relation_labels, dtype=np.int64)),
        torch.from_numpy(np.asarray(image_categories, dtype=np.int64)),
        torch.from_numpy(np.asarray(text_categories, dtype=np.int64)),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _relation_logits(output: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    return output


def evaluate_classifier_v4(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, list[list[int]]]:
    model.eval()
    truth: list[int] = []
    predicted: list[int] = []
    with torch.no_grad():
        for image, text, relation, _, _ in loader:
            image = image.to(device)
            text = text.to(device)
            logits = _relation_logits(model(image, text))
            truth.extend(relation.numpy().tolist())
            predicted.extend(logits.argmax(dim=1).cpu().numpy().tolist())
    accuracy = float(accuracy_score(truth, predicted))
    macro_f1 = float(f1_score(truth, predicted, average="macro"))
    matrix = confusion_matrix(truth, predicted, labels=list(range(len(LABELS))))
    return accuracy, macro_f1, matrix.astype(int).tolist()


def train_classifier_v4(
    name: str,
    model: nn.Module,
    train_arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    validation_arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    auxiliary_weight: float = 0.15,
) -> tuple[TrainingResultV4, dict[str, torch.Tensor]]:
    if epochs < 1:
        raise ValueError("epochs must be at least one")

    train_loader = _make_loader(*train_arrays, batch_size=batch_size, shuffle=True)
    validation_loader = _make_loader(
        *validation_arrays, batch_size=batch_size, shuffle=False
    )
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_state = copy.deepcopy(model.state_dict())
    best_f1 = -1.0
    best_epoch = 1
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        sample_count = 0
        for image, text, relation, image_category, text_category in train_loader:
            image = image.to(device)
            text = text.to(device)
            relation = relation.to(device)
            image_category = image_category.to(device)
            text_category = text_category.to(device)

            optimizer.zero_grad()
            output = model(image, text)
            if isinstance(output, tuple):
                relation_logits, image_logits, text_logits = output
                loss = criterion(relation_logits, relation)
                loss = loss + auxiliary_weight * criterion(image_logits, image_category)
                loss = loss + auxiliary_weight * criterion(text_logits, text_category)
            else:
                loss = criterion(output, relation)
            loss.backward()
            optimizer.step()

            current = int(relation.shape[0])
            running_loss += float(loss.item()) * current
            sample_count += current

        accuracy, macro_f1, _ = evaluate_classifier_v4(
            model, validation_loader, device
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": running_loss / max(sample_count, 1),
                "validation_accuracy": accuracy,
                "validation_macro_f1": macro_f1,
            }
        )
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    accuracy, macro_f1, matrix = evaluate_classifier_v4(
        model, validation_loader, device
    )
    result = TrainingResultV4(
        name=name,
        best_epoch=best_epoch,
        accuracy=accuracy,
        macro_f1=macro_f1,
        confusion_matrix=matrix,
        history=history,
    )
    return result, best_state
