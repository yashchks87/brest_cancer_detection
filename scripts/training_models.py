import math

import torch
from torch import nn
from torchvision import models


APPROACHES = ('simple', 'advanced', 'vit')
VIT_PATCH_SIZE = 16
VIT_PRETRAINED_SIZE = 224


def _validate_inputs(images: torch.Tensor, view_mask: torch.Tensor) -> None:
    if not isinstance(images, torch.Tensor) or images.ndim != 5:
        raise ValueError('Images must have shape [B, V, 3, H, W].')
    if images.shape[2] != 3 or any(size < 1 for size in images.shape):
        raise ValueError('Images must have three channels and nonempty dimensions.')
    if not images.is_floating_point():
        raise ValueError('Images must be floating-point tensors.')
    if not isinstance(view_mask, torch.Tensor) or view_mask.dtype != torch.bool:
        raise ValueError('View mask must be a boolean tensor.')
    if view_mask.ndim != 2 or view_mask.shape != images.shape[:2]:
        raise ValueError('View mask must have shape [B, V] matching images.')
    if view_mask.device != images.device:
        raise ValueError('Images and view mask must be on the same device.')
    if not view_mask.any(dim=1).all():
        raise ValueError('Every sample must contain at least one valid view.')


def _encode_views(encoder: nn.Module, images: torch.Tensor, training: bool,
                  view_chunk_size: int) -> torch.Tensor:
    if training:
        return encoder(images)
    return torch.cat([encoder(chunk) for chunk in images.split(view_chunk_size)], dim=0)


def _vit_pretrained_state() -> dict:
    return models.ViT_B_16_Weights.DEFAULT.get_state_dict(progress=False)


def _convnext_encoder(pretrained: bool) -> tuple[nn.Module, int]:
    weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    encoder = models.convnext_tiny(weights=weights)
    feature_dim = encoder.classifier[-1].in_features
    encoder.classifier[-1] = nn.Identity()
    return encoder, feature_dim


def _vit_encoder(pretrained: bool, image_size: int) -> tuple[nn.Module, int]:
    interpolated = pretrained and image_size != VIT_PRETRAINED_SIZE
    weights = models.ViT_B_16_Weights.DEFAULT if pretrained and not interpolated else None
    encoder = models.vit_b_16(weights=weights, image_size=image_size)
    if interpolated:
        state = models.vision_transformer.interpolate_embeddings(
            image_size=image_size, patch_size=VIT_PATCH_SIZE,
            model_state=dict(_vit_pretrained_state()), reset_heads=True,
        )
        missing, unexpected = encoder.load_state_dict(state, strict=False)
        if unexpected or any(not name.startswith('heads.') for name in missing):
            raise RuntimeError('Interpolated ViT weights do not match the requested image size.')
    encoder.heads = nn.Identity()
    return encoder, encoder.hidden_dim


class SimpleModel(nn.Module):
    def __init__(self, pretrained: bool, view_chunk_size: int):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.encoder = models.resnet18(weights=weights)
        self.encoder.fc = nn.Linear(self.encoder.fc.in_features, 1)
        self.view_chunk_size = view_chunk_size

    def forward(self, images: torch.Tensor, view_mask: torch.Tensor) -> torch.Tensor:
        _validate_inputs(images, view_mask)
        if not (view_mask.sum(dim=1) == 1).all():
            raise ValueError('Simple model requires exactly one valid view per sample.')
        valid_view = view_mask.to(torch.int64).argmax(dim=1)
        selected = images[torch.arange(images.shape[0], device=images.device), valid_view]
        return _encode_views(
            self.encoder, selected, self.training, self.view_chunk_size,
        ).squeeze(-1)


class MultiViewModel(nn.Module):
    def __init__(self, encoder: nn.Module, feature_dim: int, dropout: float, view_chunk_size: int):
        super().__init__()
        self.encoder = encoder
        self.attention_tanh = nn.Linear(feature_dim, 128)
        self.attention_sigmoid = nn.Linear(feature_dim, 128)
        self.attention_score = nn.Linear(128, 1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(feature_dim, 1))
        self.view_chunk_size = view_chunk_size

    def forward(self, images: torch.Tensor, view_mask: torch.Tensor) -> torch.Tensor:
        _validate_inputs(images, view_mask)
        encoded = _encode_views(
            self.encoder, images[view_mask], self.training, self.view_chunk_size,
        )
        features = encoded.new_zeros((*view_mask.shape, encoded.shape[-1]))
        features[view_mask] = encoded
        gated = torch.tanh(self.attention_tanh(features)) * torch.sigmoid(
            self.attention_sigmoid(features)
        )
        scores = self.attention_score(gated).squeeze(-1)
        attention = torch.softmax(scores.float().masked_fill(~view_mask, -torch.inf), dim=1)
        pooled = (features * attention.to(features.dtype).unsqueeze(-1)).sum(dim=1)
        return self.classifier(pooled).squeeze(-1)


def build_model(approach: str, pretrained: bool = True, dropout: float = 0.2,
                view_chunk_size: int = 8, image_size: int | None = None) -> nn.Module:
    if approach not in APPROACHES:
        raise ValueError('Approach must be simple, advanced, or vit.')
    if not isinstance(pretrained, bool):
        raise ValueError('Pretrained must be a boolean.')
    if (isinstance(dropout, bool) or not isinstance(dropout, (int, float))
            or not math.isfinite(dropout) or not 0 <= dropout < 1):
        raise ValueError('Dropout must be a finite number in [0, 1).')
    if type(view_chunk_size) is not int or view_chunk_size < 1:
        raise ValueError('View chunk size must be a positive integer.')
    if image_size is not None and (type(image_size) is not int or image_size < 64):
        raise ValueError('Image size must be an integer of at least 64, or None.')
    if approach == 'vit':
        if image_size is None:
            raise ValueError('The ViT approach requires an explicit image size.')
        if image_size % VIT_PATCH_SIZE:
            raise ValueError(f'ViT image size must be a multiple of {VIT_PATCH_SIZE}.')
        return MultiViewModel(*_vit_encoder(pretrained, image_size), float(dropout), view_chunk_size)
    if approach == 'simple':
        return SimpleModel(pretrained, view_chunk_size)
    return MultiViewModel(*_convnext_encoder(pretrained), float(dropout), view_chunk_size)
