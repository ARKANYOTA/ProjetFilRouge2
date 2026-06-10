from __future__ import annotations

import torch
import torch.nn as nn

from src.config.settings import Settings


# ---------------------------------------------------------------------------
# LeNet-5  (LeCun et al., 1998 — §II.B, Fig. 2)
# ---------------------------------------------------------------------------
class LeNet5(nn.Module):
    """LeNet-5 adapted from *Gradient-Based Learning Applied to Document
    Recognition* (LeCun et al., 1998).

    Original architecture (§II.B, p.7-8):
        C1: 6 feature maps, 5×5 conv
        S2: 2×2 average-pool (non-overlapping)
        C3: 16 feature maps, 5×5 conv
        S4: 2×2 average-pool
        C5: 120 feature maps, 5×5 conv
        F6: 84 fully-connected units
        Output: num_classes

    The paper uses ``tanh`` activations (§II.B, eq. 6: f(a) = A·tanh(S·a)).
    We use standard ``torch.tanh`` for simplicity.

    Adaptation for 224×224 input: an ``AdaptiveAvgPool2d(1)`` replaces the
    fixed spatial collapse after C5 so the network handles any input size.
    """

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        c_in = settings.in_channels
        n_cls = settings.num_classes

        # C1: 6 feature maps, 5×5 kernel
        self.c1 = nn.Conv2d(c_in, 6, kernel_size=5, padding=2)
        # S2: 2×2 average pool, non-overlapping (paper: "2×2 area")
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # C3: 16 feature maps, 5×5 kernel
        self.c3 = nn.Conv2d(6, 16, kernel_size=5)
        # S4: 2×2 average pool
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)
        # C5: 120 feature maps, 5×5 kernel
        self.c5 = nn.Conv2d(16, 120, kernel_size=5)
        # Adaptive pool to handle arbitrary spatial sizes
        self.pool = nn.AdaptiveAvgPool2d(1)
        # F6: 84 units (paper: "contains 84 units")
        self.f6 = nn.Linear(120, 84)
        # Output layer (softmax handled by CrossEntropyLoss)
        self.out = nn.Linear(84, n_cls)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.c1(x))
        x = self.s2(x)
        x = torch.tanh(self.c3(x))
        x = self.s4(x)
        x = torch.tanh(self.c5(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.tanh(self.f6(x))
        result: torch.Tensor = self.out(x)
        return result


# ---------------------------------------------------------------------------
# AlexNet  (Krizhevsky et al., 2012 — §3.5, Fig. 2)
# ---------------------------------------------------------------------------
class AlexNet(nn.Module):
    """AlexNet adapted from *ImageNet Classification with Deep Convolutional
    Neural Networks* (Krizhevsky et al., 2012).

    Original architecture (§3.5, p.4):
        Conv1: 96 kernels, 11×11, stride 4           + ReLU + MaxPool(3,2)
        Conv2: 256 kernels, 5×5, pad 2               + ReLU + MaxPool(3,2)
        Conv3: 384 kernels, 3×3, pad 1               + ReLU
        Conv4: 384 kernels, 3×3, pad 1               + ReLU
        Conv5: 256 kernels, 3×3, pad 1               + ReLU + MaxPool(3,2)
        FC6:   4096                                   + ReLU + Dropout(0.5)
        FC7:   4096                                   + ReLU + Dropout(0.5)
        FC8:   num_classes

    Key innovations:
    - ReLU activation (§3.1): "non-saturating nonlinearity f(x) = max(0,x)"
    - Overlapping pooling (§3.4): kernel 3, stride 2 (s < z)
    - Dropout 0.5 on FC layers (§4.2): "setting to zero the output of each
      hidden neuron with probability 0.5"

    Adaptation: channel counts halved (96→48, 256→128, etc.) because TILDA
    has only ~2300 images vs ImageNet's 1.2M.  Full-size AlexNet would
    massively overfit.
    """

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        c_in = settings.in_channels
        n_cls = settings.num_classes
        drop = settings.dropout_rate

        self.features = nn.Sequential(
            # Conv1: paper says 11×11, stride 4.  We keep it for 224×224 input.
            nn.Conv2d(c_in, 48, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # overlapping pooling §3.4
            # Conv2
            nn.Conv2d(48, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Conv3
            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Conv4
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Conv5
            nn.Conv2d(192, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=drop),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, n_cls),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        result: torch.Tensor = self.classifier(x)
        return result


# ---------------------------------------------------------------------------
# ResNet-18  (He et al., 2016 — Table 1, §3.2–3.3, Fig. 2,5)
# ---------------------------------------------------------------------------
class _BasicBlock(nn.Module):
    """Residual *BasicBlock* (Fig. 5 left, He et al. 2016).

    Two 3×3 conv layers with Batch Normalisation (§3.4: "BN right after each
    convolution and before activation").  Identity shortcut y = F(x) + x
    (Eq. 1); a 1×1 projection shortcut is used when dimensions change (Eq. 2).
    """

    expansion: int = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # shortcut connection — Eq. 1: y = F(x) + x
        result: torch.Tensor = self.relu(out)
        return result


class ResNet18(nn.Module):
    """ResNet-18 from *Deep Residual Learning for Image Recognition*
    (He et al., 2016).

    Architecture (Table 1, 18-layer column):
        conv1:   7×7, 64 filters, stride 2 → BN → ReLU → 3×3 max-pool stride 2
        conv2_x: 2 × BasicBlock(64)
        conv3_x: 2 × BasicBlock(128), first block stride 2
        conv4_x: 2 × BasicBlock(256), first block stride 2
        conv5_x: 2 × BasicBlock(512), first block stride 2
        Global average pooling → FC(num_classes)

    Key features:
    - Shortcut connections (§3.2): identity mapping y = F(x) + x
    - Projection shortcut (Eq. 2) when dimensions change: y = F(x) + Ws·x
    - Batch Norm after every conv (§3.4)
    - No dropout (§3.4: "We do not use dropout, following [16]")
    """

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        c_in = settings.in_channels
        n_cls = settings.num_classes
        self.in_planes = 64

        # conv1: "7×7, 64, stride 2" (Table 1)
        self.conv1 = nn.Conv2d(c_in, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # "3×3 max pool, stride 2" (Table 1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers (Table 1, 18-layer column)
        self.layer1 = self._make_layer(64, blocks=2, stride=1)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)

        # "the network ends with a global average pooling layer" (§3.3)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * _BasicBlock.expansion, n_cls)

        # Weight initialisation following He et al. [13] (§3.4)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        downsample: nn.Module | None = None
        if stride != 1 or self.in_planes != planes * _BasicBlock.expansion:
            # Projection shortcut (Eq. 2, option B)
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_planes,
                    planes * _BasicBlock.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * _BasicBlock.expansion),
            )

        layers: list[nn.Module] = [_BasicBlock(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes * _BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(_BasicBlock(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        result: torch.Tensor = self.fc(x)
        return result


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
_MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "lenet5": LeNet5,
    "alexnet": AlexNet,
    "resnet18": ResNet18,
}


def get_model(settings: Settings) -> nn.Module:
    """Instantiate a model by name and send it to the configured device."""
    name = settings.model_name.lower()
    if name not in _MODEL_REGISTRY:
        valid = ", ".join(sorted(_MODEL_REGISTRY))
        msg = f"Unknown model '{name}'. Choose from: {valid}"
        raise ValueError(msg)
    model = _MODEL_REGISTRY[name](settings)
    device = settings.resolve_device()
    return model.to(device)
