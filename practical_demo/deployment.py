from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from src.v4_data import LABELS, PROJECT_ROOT, image_transform_v4
from src.v4_training import MultimodalClassifierV4


DEMO_ROOT = Path(__file__).resolve().parent
MODEL_DIR = DEMO_ROOT / "models"

CLASSIFIER_PATH = (
    MODEL_DIR / "dataset_v4_deployment_classifier.pt"
)

TFIDF_PATH = (
    MODEL_DIR / "dataset_v4_deployment_tfidf.pkl"
)

METADATA_PATH = (
    MODEL_DIR / "dataset_v4_deployment_metadata.json"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def load_deployment_metadata() -> dict[str, Any]:
    if not METADATA_PATH.is_file():
        raise FileNotFoundError(
            f"Deployment metadata not found: {METADATA_PATH}"
        )

    metadata = json.loads(
        METADATA_PATH.read_text(encoding="utf-8")
    )

    required = {
        "status",
        "classifier_sha256",
        "tfidf_sha256",
        "image_features",
        "text_features",
        "labels",
        "locked_final_test_relations_loaded",
        "new_final_test_inference_performed",
        "post_test_tuning_performed",
    }

    missing = sorted(required - set(metadata))

    if missing:
        raise ValueError(
            f"Deployment metadata is missing fields: {missing}"
        )

    if (
        metadata["status"]
        != "PASS_DATASET_V4_DEPLOYMENT_BUNDLE_CREATED"
    ):
        raise ValueError(
            "Deployment bundle status is not PASS."
        )

    if metadata["locked_final_test_relations_loaded"] is not False:
        raise ValueError(
            "Deployment metadata unexpectedly reports "
            "locked final-test access."
        )

    if metadata["new_final_test_inference_performed"] is not False:
        raise ValueError(
            "Deployment bundle must not contain new final-test inference."
        )

    if metadata["post_test_tuning_performed"] is not False:
        raise ValueError(
            "Deployment bundle must not contain post-test tuning."
        )

    if tuple(metadata["labels"]) != tuple(LABELS):
        raise ValueError(
            "Deployment label order does not match Dataset V4."
        )

    return metadata


def verify_deployment_artifacts(
    metadata: dict[str, Any],
) -> None:
    for path in (
        CLASSIFIER_PATH,
        TFIDF_PATH,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    classifier_sha = file_sha256(
        CLASSIFIER_PATH
    )

    tfidf_sha = file_sha256(
        TFIDF_PATH
    )

    if classifier_sha != metadata["classifier_sha256"]:
        raise ValueError(
            "Deployment classifier SHA-256 mismatch."
        )

    if tfidf_sha != metadata["tfidf_sha256"]:
        raise ValueError(
            "Deployment TF-IDF SHA-256 mismatch."
        )


class DatasetV4DeploymentPredictor:
    """Practical image-text relation predictor.

    This class loads the separately built Dataset V4
    deployment bundle. It does not load benchmark test
    relations and does not perform training.
    """

    def __init__(
        self,
        *,
        device: str | torch.device | None = None,
    ) -> None:
        self.metadata = load_deployment_metadata()
        verify_deployment_artifacts(self.metadata)

        if device is None:
            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        else:
            self.device = torch.device(device)

        with TFIDF_PATH.open("rb") as stream:
            # This pickle is a trusted project artifact
            # created locally by our deployment builder.
            self.vectorizer = pickle.load(stream)

        expected_text_dimension = int(
            self.metadata["text_features"]
        )

        actual_text_dimension = len(
            self.vectorizer.get_feature_names_out()
        )

        if actual_text_dimension != expected_text_dimension:
            raise ValueError(
                "TF-IDF feature dimension does not match metadata."
            )

        self.image_transform = image_transform_v4(
            training=False
        )

        weights = ResNet18_Weights.DEFAULT

        self.image_model = resnet18(
            weights=weights
        )

        self.image_model.fc = nn.Identity()

        self.image_model.eval()
        self.image_model.to(self.device)

        image_dimension = int(
            self.metadata["image_features"]
        )

        text_dimension = int(
            self.metadata["text_features"]
        )

        self.classifier = MultimodalClassifierV4(
            image_dimension,
            text_dimension,
            auxiliary_heads=True,
        )

        state = torch.load(
            CLASSIFIER_PATH,
            map_location="cpu",
            weights_only=True,
        )

        self.classifier.load_state_dict(
            state,
            strict=True,
        )

        self.classifier.eval()
        self.classifier.to(self.device)

    def extract_image_feature(
        self,
        image: Image.Image,
    ) -> np.ndarray:
        """Extract the same ResNet18 representation used by V4."""

        tensor = self.image_transform(
            image.convert("RGB")
        ).unsqueeze(0)

        with torch.no_grad():
            features = self.image_model(
                tensor.to(self.device)
            )

        feature = (
            features
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        if feature.shape != (
            1,
            int(self.metadata["image_features"]),
        ):
            raise ValueError(
                "Unexpected image feature shape: "
                f"{feature.shape}"
            )

        return feature

    def transform_text(
        self,
        description: str,
    ) -> tuple[np.ndarray, int]:
        description = description.strip()

        if not description:
            raise ValueError(
                "Description must not be empty."
            )

        matrix = (
            self.vectorizer
            .transform([description])
            .toarray()
            .astype(np.float32)
        )

        recognized_features = int(
            np.count_nonzero(matrix)
        )

        return matrix, recognized_features

    def predict(
        self,
        image: Image.Image,
        description: str,
    ) -> dict[str, Any]:
        image_features = self.extract_image_feature(
            image
        )

        text_features, recognized_features = (
            self.transform_text(description)
        )

        image_tensor = torch.from_numpy(
            image_features
        ).to(self.device)

        text_tensor = torch.from_numpy(
            text_features
        ).to(self.device)

        with torch.no_grad():
            output = self.classifier(
                image_tensor,
                text_tensor,
            )

            if isinstance(output, tuple):
                relation_logits = output[0]
            else:
                relation_logits = output

            scores = torch.softmax(
                relation_logits,
                dim=1,
            )[0]

        score_values = (
            scores
            .cpu()
            .numpy()
            .astype(float)
        )

        predicted_index = int(
            np.argmax(score_values)
        )

        predicted_label = LABELS[
            predicted_index
        ]

        label_scores = {
            label: float(score_values[index])
            for index, label in enumerate(LABELS)
        }

        warning = None

        if recognized_features == 0:
            warning = (
                "The description contains no TF-IDF features "
                "known to this deployment model. "
                "The result may rely mostly on the image."
            )

        return {
            "label": predicted_label,
            "model_score": float(
                score_values[predicted_index]
            ),
            "scores": label_scores,
            "recognized_text_features": (
                recognized_features
            ),
            "warning": warning,
            "device": str(self.device),
            "note": (
                "Softmax scores are model scores, "
                "not calibrated real-world probabilities."
            ),
        }

    def predict_path(
        self,
        image_path: str | Path,
        description: str,
    ) -> dict[str, Any]:
        path = Path(image_path)

        if not path.is_file():
            raise FileNotFoundError(path)

        with Image.open(path) as image:
            return self.predict(
                image,
                description,
            )
