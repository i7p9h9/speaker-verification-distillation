"""Building blocks for ECAPA2.

The tensor convention in this module is ``(batch, channels, frequency, time)``
for the local feature extractor and ``(batch, channels, time)`` afterwards.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class Conv2dReluBn(nn.Sequential):
    """2-D convolution in the Conv -> ReLU -> BatchNorm order used by ECAPA2."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: tuple[int, int] = (1, 1),
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )


class FrequencyWiseSqueezeExcitation(nn.Module):
    """Frequency-wise squeeze-excitation (fwSE).

    Unlike a conventional SE block, the squeeze operation preserves frequency
    and averages over feature maps and time.  Consequently, one gate is learned
    for every frequency position.
    """

    def __init__(self, num_frequencies: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.num_frequencies = num_frequencies
        self.excitation = nn.Sequential(
            nn.Linear(num_frequencies, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, num_frequencies),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        descriptor = x.mean(dim=(1, 3))
        scale = self.excitation(descriptor).unsqueeze(1).unsqueeze(-1)
        return x * scale


class LocalFeatureExtractorBlock(nn.Module):
    """Three 2-D convolutions followed by fwSE and an optional residual path."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_frequencies: int,
        *,
        stride: tuple[int, int] = (1, 1),
        fwse_hidden_dim: int = 128,
        residual: bool = True,
        frequency_encoding: bool = True,
    ) -> None:
        super().__init__()
        if stride[1] != 1:
            raise ValueError("ECAPA2 must preserve the temporal resolution in its LFE")

        self.position = (
            nn.Parameter(torch.zeros(1, 1, num_frequencies, 1)) if frequency_encoding else None
        )
        self.body = nn.Sequential(
            Conv2dReluBn(in_channels, out_channels, stride=stride),
            Conv2dReluBn(out_channels, out_channels),
            Conv2dReluBn(out_channels, out_channels),
            FrequencyWiseSqueezeExcitation(
                (num_frequencies + stride[0] - 1) // stride[0],
                hidden_dim=fwse_hidden_dim,
            ),
        )

        self.residual = residual
        if residual and (in_channels != out_channels or stride != (1, 1)):
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch = x if self.position is None else x + self.position
        branch = self.body(branch)
        if self.residual:
            branch = branch + self.skip(x)
        return branch


class LocalFeatureExtractor(nn.Module):
    """The 20-block local, two-dimensional part of ECAPA2."""

    def __init__(
        self,
        num_frequencies: int,
        *,
        in_channels: int = 1,
        channels: tuple[int, ...] = (164, 164, 164, 192, 192),
        repeats: tuple[int, ...] = (3, 4, 4, 4, 5),
        fwse_hidden_dim: int = 128,
        frequency_encoding: bool = True,
    ) -> None:
        super().__init__()
        if len(channels) != len(repeats):
            raise ValueError("channels and repeats must have the same length")
        if any(repeat < 1 for repeat in repeats):
            raise ValueError("every LFE stage must contain at least one block")

        blocks: list[nn.Module] = []
        current_channels = in_channels
        frequencies = num_frequencies
        for stage, (out_channels, repeat) in enumerate(zip(channels, repeats)):
            stride = (1, 1) if stage == 0 else (2, 1)
            for block_index in range(repeat):
                block_stride = stride if block_index == 0 else (1, 1)
                # The first block is the stem. All subsequent blocks use a
                # projected or identity residual connection.
                residual = not (stage == 0 and block_index == 0)
                blocks.append(
                    LocalFeatureExtractorBlock(
                        current_channels,
                        out_channels,
                        frequencies,
                        stride=block_stride,
                        fwse_hidden_dim=fwse_hidden_dim,
                        residual=residual,
                        frequency_encoding=frequency_encoding,
                    )
                )
                if block_stride[0] > 1:
                    frequencies = (frequencies + block_stride[0] - 1) // block_stride[0]
                current_channels = out_channels

        self.blocks = nn.Sequential(*blocks)
        self.out_channels = current_channels
        self.out_frequencies = frequencies

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class Conv1dReluBn(nn.Sequential):
    """1-D convolution in the Conv -> ReLU -> BatchNorm order."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        *,
        dilation: int = 1,
    ) -> None:
        super().__init__(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=(kernel_size // 2) * dilation,
                dilation=dilation,
            ),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
        )


class Res2Conv1d(nn.Module):
    """Hierarchical Res2Net convolution used in the global extractor."""

    def __init__(self, channels: int, scale: int = 8, dilation: int = 2) -> None:
        super().__init__()
        if channels % scale:
            raise ValueError(f"channels ({channels}) must be divisible by scale ({scale})")
        self.scale = scale
        width = channels // scale
        self.convs = nn.ModuleList(
            [nn.Conv1d(width, width, 3, dilation=dilation) for _ in range(scale - 1)]
        )
        self.norms = nn.ModuleList([nn.BatchNorm1d(width) for _ in range(scale - 1)])
        self.padding = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunks = x.chunk(self.scale, dim=1)
        outputs = [chunks[0]]
        current: torch.Tensor | None = None
        for index, (conv, norm) in enumerate(zip(self.convs, self.norms), start=1):
            current = chunks[index] if current is None else chunks[index] + current
            current = F.pad(current, (self.padding, self.padding), mode="reflect")
            current = norm(F.relu(conv(current)))
            outputs.append(current)
        return torch.cat(outputs, dim=1)


class SqueezeExcitation1d(nn.Module):
    def __init__(self, channels: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.excitation = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.excitation(x.mean(dim=-1)).unsqueeze(-1)
        return x * scale


class GlobalFeatureExtractor(nn.Module):
    """TDNN/Res2Net global-frequency module from ECAPA2."""

    def __init__(
        self,
        in_channels: int,
        num_frequencies: int,
        *,
        hidden_channels: int = 1024,
        out_channels: int = 1536,
        scale: int = 8,
        se_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.pre = Conv1dReluBn(in_channels * num_frequencies, hidden_channels)
        self.residual = nn.Sequential(
            Conv1dReluBn(hidden_channels, hidden_channels),
            Res2Conv1d(hidden_channels, scale=scale, dilation=2),
            Conv1dReluBn(hidden_channels, hidden_channels),
            SqueezeExcitation1d(hidden_channels, hidden_dim=se_hidden_dim),
        )
        self.projection = nn.Sequential(
            nn.Conv1d(hidden_channels, out_channels, 1),
            nn.ReLU(),
        )
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(1, 2)
        x = self.pre(x)
        x = x + self.residual(x)
        return self.projection(x)


class ChannelDependentAttentiveStatisticsPooling(nn.Module):
    """CAS pooling with access to its pre-softmax channel attention scores."""

    def __init__(self, channels: int, hidden_channels: int = 128, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.attention = nn.Sequential(
            nn.Conv1d(channels, hidden_channels, 1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_channels),
            nn.Conv1d(hidden_channels, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # time = x.shape[-1]
        # mean = x.mean(dim=-1, keepdim=True).expand(-1, -1, time)
        # std = x.var(dim=-1, keepdim=True, unbiased=False).add(self.eps).sqrt()
        # std = std.expand(-1, -1, time)

        # attention_logits = self.attention(torch.cat((x, mean, std), dim=1))
        attention_logits = self.attention(x)
        weights = attention_logits.softmax(dim=-1)
        attentive_mean = (weights * x).sum(dim=-1)
        second_moment = (weights * x.square()).sum(dim=-1)
        attentive_std = (second_moment - attentive_mean.square()).clamp_min(self.eps).sqrt()
        return torch.cat((attentive_mean, attentive_std), dim=1), attention_logits


__all__ = [
    "ChannelDependentAttentiveStatisticsPooling",
    "FrequencyWiseSqueezeExcitation",
    "GlobalFeatureExtractor",
    "LocalFeatureExtractor",
    "LocalFeatureExtractorBlock",
    "Res2Conv1d",
]
