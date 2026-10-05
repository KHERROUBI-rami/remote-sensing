"""6-channel MobileNetV3-Large classifier for EuroSAT multispectral imagery."""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision import models


class MobileNetV3Classifier(nn.Module):
    """ImageNet-pretrained MobileNetV3-Large adapted to ``num_bands`` input channels.

    Transfer-learning recipe
    ------------------------
    1. Start from the torchvision MobileNetV3-Large ``features`` backbone
       (ImageNet weights).
    2. Replace the first convolution so it accepts ``num_bands`` channels.
       The pretrained RGB filters are copied into the first three channels and
       the extra channels are initialised with the *mean* RGB filter, so the
       network starts from a sensible, scale-consistent initialisation.
    3. Replace the ImageNet head with a small MLP (960 -> 256 -> 128 -> classes)
       with heavy dropout for regularisation.

    Note: the parameter names (``encoder.*``, ``classifier.*``) must stay as they
    are to remain compatible with ``best_model.pth``.
    """

    def __init__(self, num_bands: int = 6, num_classes: int = 10, pretrained: bool = True):
        super().__init__()
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.mobilenet_v3_large(weights=weights)
        features = list(backbone.features.children())

        old_conv = features[0][0]  # Conv2d(3, 16, k=3, s=2, p=1, bias=False)
        new_conv = nn.Conv2d(
            num_bands, old_conv.out_channels,
            kernel_size=3, stride=2, padding=1, bias=False,
        )
        if pretrained:
            with torch.no_grad():
                new_conv.weight[:, :3] = old_conv.weight
                if num_bands > 3:
                    new_conv.weight[:, 3:] = old_conv.weight.mean(dim=1, keepdim=True)
        features[0][0] = new_conv

        self.encoder = nn.Sequential(*features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(960, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.67),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.47),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def load_checkpoint(
    path: str,
    device: torch.device | str = "cpu",
    num_bands: int = 6,
    num_classes: int = 10,
) -> MobileNetV3Classifier:
    """Build the model and load a ``state_dict`` checkpoint (eval mode).

    Handles checkpoints saved from ``nn.DataParallel`` (``module.`` prefix).
    ImageNet weights are *not* downloaded - the checkpoint overwrites them.
    """
    model = MobileNetV3Classifier(num_bands, num_classes, pretrained=False)
    state = torch.load(path, map_location=device)
    state = OrderedDict((k.replace("module.", "", 1), v) for k, v in state.items())
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
