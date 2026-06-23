from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
from torchvision.ops import StochasticDepth

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


class _Bottleneck(nn.Module):
    """Residual *Bottleneck* block (Fig. 5 right, He et al. 2016).

    Three stacked convolutions — 1×1 (reduce) → 3×3 → 1×1 (expand by four) —
    each followed by Batch Normalisation, used by ResNet-50/101/152.  The 1×1
    layers restore and then expand the channel dimension so the 3×3 convolution
    operates on a reduced "bottleneck" width (§3.3, "Deeper Bottleneck
    Architectures").
    """

    expansion: int = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # shortcut connection — Eq. 1: y = F(x) + x
        result: torch.Tensor = self.relu(out)
        return result


class ResNet(nn.Module):
    """ResNet architecture from *Deep Residual Learning for Image Recognition*
    (He et al., 2016).

    Key features:
    - Shortcut connections (§3.2): identity mapping y = F(x) + x
    - Projection shortcut (Eq. 2) when dimensions change: y = F(x) + Ws·x
    - Batch Norm after every conv (§3.4)
    - No dropout (§3.4: "We do not use dropout, following [16]")

    The residual ``block`` (``_BasicBlock`` for ResNet-18/34, ``_Bottleneck`` for
    ResNet-50) determines the channel expansion factor of each stage.
    """

    def __init__(
        self,
        settings: Settings,
        layers: list[int],
        block: type[_BasicBlock | _Bottleneck] = _BasicBlock,
    ) -> None:
        super().__init__()
        c_in = settings.in_channels
        n_cls = settings.num_classes
        self.in_planes = 64
        self.block = block

        # conv1: "7×7, 64, stride 2"
        self.conv1 = nn.Conv2d(c_in, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # "3×3 max pool, stride 2"
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(64, blocks=layers[0], stride=1)
        self.layer2 = self._make_layer(128, blocks=layers[1], stride=2)
        self.layer3 = self._make_layer(256, blocks=layers[2], stride=2)
        self.layer4 = self._make_layer(512, blocks=layers[3], stride=2)

        # "the network ends with a global average pooling layer" (§3.3)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, n_cls)

        # Weight initialisation following He et al. [13] (§3.4)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        downsample: nn.Module | None = None
        if stride != 1 or self.in_planes != planes * self.block.expansion:
            # Projection shortcut (Eq. 2, option B)
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_planes,
                    planes * self.block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * self.block.expansion),
            )

        layers_: list[nn.Module] = [self.block(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes * self.block.expansion
        for _ in range(1, blocks):
            layers_.append(self.block(self.in_planes, planes))

        return nn.Sequential(*layers_)

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


def resnet18(settings: Settings) -> ResNet:
    return ResNet(settings, [2, 2, 2, 2])


def resnet34(settings: Settings) -> ResNet:
    return ResNet(settings, [3, 4, 6, 3])


def resnet50(settings: Settings) -> ResNet:
    return ResNet(settings, [3, 4, 6, 3], block=_Bottleneck)


# ---------------------------------------------------------------------------
# ConvNeXt  (Liu et al., 2022 — "A ConvNet for the 2020s")
# ---------------------------------------------------------------------------
class _LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for ``(N, C, H, W)`` tensors (ConvNeXt §2.3).

    Normalises over the channel dimension only, matching the official
    ``channels_first`` LayerNorm used in the ConvNeXt reference implementation.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[None, :, None, None] * x + self.bias[None, :, None, None]


class _ConvNeXtBlock(nn.Module):
    """ConvNeXt residual block (Liu et al., 2022 — Fig. 4, §2).

    Pipeline: depthwise 7×7 conv → LayerNorm → 1×1 conv expand ×4 → GELU →
    1×1 conv project → LayerScale → stochastic depth → residual add.

    The whole block stays in channels-first (NCHW) layout: the two pointwise
    "Linear" layers of the paper are implemented as equivalent 1×1 convolutions
    and a channel-wise :class:`_LayerNorm2d` is used.  This avoids the
    ``permute`` → ``nn.LayerNorm`` pattern, whose backward pass is broken on the
    Apple MPS backend ("view size is not compatible ..."), while remaining
    numerically identical on CPU/CUDA.
    """

    def __init__(self, dim: int, drop_path: float, layer_scale_init: float) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = _LayerNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)
        # LayerScale (Touvron et al. 2021): a learnable per-channel scale γ.
        self.gamma: nn.Parameter | None = (
            nn.Parameter(layer_scale_init * torch.ones(dim, 1, 1)) if layer_scale_init > 0 else None
        )
        self.drop_path = StochasticDepth(drop_path, mode="row")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        result: torch.Tensor = shortcut + self.drop_path(x)
        return result


class ConvNeXt(nn.Module):
    """ConvNeXt trained **from scratch** (no pretrained weights).

    Hierarchical design (§2): a patchify stem (4×4 stride-4 conv), four stages
    of :class:`_ConvNeXtBlock` separated by 2×2 stride-2 downsampling layers,
    global average pooling, a final LayerNorm and a linear classifier head.
    Depths, channel widths, stochastic-depth rate and LayerScale are all driven
    by :class:`Settings` so the network can be sized down for the small TILDA
    dataset (1,888 training images) to control overfitting.
    """

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        depths = settings.convnext_depths
        dims = settings.convnext_dims
        c_in = settings.in_channels
        n_cls = settings.num_classes

        # Downsampling layers: stem + 3 inter-stage downsamplers.
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(c_in, dims[0], kernel_size=4, stride=4),
            _LayerNorm2d(dims[0]),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    _LayerNorm2d(dims[i]),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                )
            )

        # Linear stochastic-depth decay across all blocks (§2.3, "we use ...
        # a stochastic depth rate that increases linearly with block depth").
        total_blocks = sum(depths)
        dp_rates = [float(r) for r in torch.linspace(0.0, settings.drop_path_rate, total_blocks)]
        self.stages = nn.ModuleList()
        cursor = 0
        for i in range(4):
            blocks = [
                _ConvNeXtBlock(dims[i], dp_rates[cursor + j], settings.layer_scale_init)
                for j in range(depths[i])
            ]
            self.stages.append(nn.Sequential(*blocks))
            cursor += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # applied to pooled features
        # Head structure is fixed (Dropout always present) so checkpoint keys are
        # stable regardless of the configured dropout probability.
        self.head = nn.Sequential(
            nn.Dropout(settings.head_dropout),
            nn.Linear(dims[-1], n_cls),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d | nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = x.mean([-2, -1])  # global average pooling → (N, C)
        x = self.norm(x)
        result: torch.Tensor = self.head(x)
        return result


def convnext(settings: Settings) -> ConvNeXt:
    return ConvNeXt(settings)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
_MODEL_REGISTRY: dict[str, Callable[[Settings], nn.Module]] = {
    "lenet5": LeNet5,
    "alexnet": AlexNet,
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "convnext": convnext,
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
