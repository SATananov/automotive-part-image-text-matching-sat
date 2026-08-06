from __future__ import annotations

import torch

from src.v4_models import (
    MultimodalRelationModelV4,
    MultimodalRelationModelV4NoAuxiliary,
    ResNet18ImageEncoderV4,
    count_total_parameters_v4,
    count_trainable_parameters_v4,
)


def test_resnet18_v4_encoder_shape_and_freezing() -> None:
    encoder = ResNet18ImageEncoderV4(pretrained=False, freeze_backbone=True)
    encoder.train()
    assert encoder.backbone_frozen is True
    assert encoder.backbone.training is False
    assert all(not parameter.requires_grad for parameter in encoder.backbone.parameters())
    output = encoder(torch.zeros(2, 3, 224, 224))
    assert output.shape == (2, 128)


def test_multimodal_v4_output_shapes_without_network_download() -> None:
    model = MultimodalRelationModelV4(
        64,
        pretrained_image_encoder=False,
        freeze_image_backbone=True,
    )
    relation, image_category, text_category = model(
        torch.zeros(2, 3, 224, 224),
        torch.zeros(2, 64),
    )
    assert relation.shape == (2, 3)
    assert image_category.shape == (2, 50)
    assert text_category.shape == (2, 50)
    assert count_total_parameters_v4(model) > count_trainable_parameters_v4(model)


def test_v4_ablation_has_no_category_heads() -> None:
    model = MultimodalRelationModelV4NoAuxiliary(
        64,
        pretrained_image_encoder=False,
        freeze_image_backbone=True,
    )
    output = model(torch.zeros(2, 3, 224, 224), torch.zeros(2, 64))
    assert output.shape == (2, 3)
    assert not hasattr(model, "image_category_head")
    assert not hasattr(model, "text_category_head")
