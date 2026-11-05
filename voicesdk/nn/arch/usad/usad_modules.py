# Reference: https://github.com/sooftware/conformer

import contextlib
import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel


def lengths_to_padding_mask(
    lengths: torch.Tensor, max_len: Optional[int] = None
) -> torch.Tensor:
    """Create padding mask from lengths.

    Args:
        lengths: A 1-D tensor of shape (B,).
        max_len: An integer. It will be automatically set to the max value of lengths
            if not given.

    Returns:
        A bool tensor of shape (B, max_len), where padded positions are indicated by True.
    """
    batch_size = lengths.size(0)
    max_len = lengths.max().item() if max_len is None else max_len
    seq_range = torch.arange(
        0, max_len, dtype=torch.long, device=lengths.device
    )
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    lengths_expand = lengths.unsqueeze(1).expand_as(seq_range_expand)
    padding_mask = seq_range_expand >= lengths_expand
    return padding_mask


class SamePad(nn.Module):
    def __init__(self, kernel_size, causal=False):
        super().__init__()
        if causal:
            self.remove = kernel_size - 1
        else:
            self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        if self.remove > 0:
            x = x[:, :, : -self.remove]
        return x


class TransposeLast(nn.Module):
    def __init__(self, deconstruct_idx=None, tranpose_dim=-2):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx
        self.tranpose_dim = tranpose_dim

    def forward(self, x):
        if self.deconstruct_idx is not None:
            x = x[self.deconstruct_idx]
        return x.transpose(self.tranpose_dim, -1)


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * inputs.sigmoid()


class GLU(nn.Module):
    def __init__(self, dim: int) -> None:
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class ResidualConnectionModule(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        module_factor: float = 1.0,
        input_factor: float = 1.0,
    ):
        super(ResidualConnectionModule, self).__init__()
        self.module = module
        self.module_factor = module_factor
        self.input_factor = input_factor

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return (self.module(inputs) * self.module_factor) + (
            inputs * self.input_factor
        )


class Linear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, bias: bool = True
    ) -> None:
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class View(nn.Module):
    def __init__(self, shape: tuple, contiguous: bool = False):
        super(View, self).__init__()
        self.shape = shape
        self.contiguous = contiguous

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.contiguous:
            x = x.contiguous()

        return x.view(*self.shape)


class Transpose(nn.Module):
    def __init__(self, shape: tuple):
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(*self.shape)


class FeedForwardModule(nn.Module):
    def __init__(
        self,
        encoder_dim: int = 512,
        expansion_factor: int = 4,
        dropout_p: float = 0.1,
        rms_norm: bool = False,
    ) -> None:
        super(FeedForwardModule, self).__init__()
        self.sequential = nn.Sequential(
            (
                nn.LayerNorm(encoder_dim)
                if not rms_norm
                else RMSNorm(encoder_dim)
            ),
            Linear(encoder_dim, encoder_dim * expansion_factor, bias=True),
            Swish(),
            nn.Dropout(p=dropout_p),
            Linear(encoder_dim * expansion_factor, encoder_dim, bias=True),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.sequential(inputs)


class DepthwiseConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
    ) -> None:
        super(DepthwiseConv1d, self).__init__()
        assert (
            out_channels % in_channels == 0
        ), "out_channels should be constant multiple of in_channels"
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.conv(inputs)


class PointwiseConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ) -> None:
        super(PointwiseConv1d, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.conv(inputs)


class ConformerConvModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 31,
        expansion_factor: int = 2,
        dropout_p: float = 0.1,
        rms_norm: bool = False,
    ) -> None:
        super(ConformerConvModule, self).__init__()
        assert (
            kernel_size - 1
        ) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding"
        assert (
            expansion_factor == 2
        ), "Currently, Only Supports expansion_factor 2"

        self.sequential = nn.Sequential(
            (
                nn.LayerNorm(in_channels)
                if not rms_norm
                else RMSNorm(in_channels)
            ),
            Transpose(shape=(1, 2)),
            PointwiseConv1d(
                in_channels,
                in_channels * expansion_factor,
                stride=1,
                padding=0,
                bias=True,
            ),
            GLU(dim=1),
            DepthwiseConv1d(
                in_channels,
                in_channels,
                kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
            ),
            nn.BatchNorm1d(in_channels),
            Swish(),
            PointwiseConv1d(
                in_channels, in_channels, stride=1, padding=0, bias=True
            ),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.sequential(inputs).transpose(1, 2)


class FramewiseConv2dSubampling(nn.Module):
    def __init__(self, out_channels: int, subsample_rate: int = 2) -> None:
        super(FramewiseConv2dSubampling, self).__init__()
        assert subsample_rate in {2, 4}, "subsample_rate should be 2 or 4"
        self.subsample_rate = subsample_rate
        self.cnn = nn.Sequential(
            nn.Conv2d(1, out_channels, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=(2 if subsample_rate == 4 else 1, 2),
                padding=(0 if subsample_rate == 4 else 1, 0),
            ),
            nn.ReLU(),
        )

    def forward(
        self, inputs: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # inputs: (B, T, C) -> (B, 1, T, C)
        if self.subsample_rate == 2 and inputs.shape[1] % 2 == 0:
            inputs = F.pad(inputs, (0, 0, 0, 1), "constant", 0)
        if self.subsample_rate == 4 and inputs.shape[1] % 4 < 3:
            inputs = F.pad(
                inputs, (0, 0, 0, 3 - inputs.shape[1] % 4), "constant", 0
            )
        outputs = self.cnn(inputs.unsqueeze(1))
        batch_size, channels, subsampled_lengths, sumsampled_dim = (
            outputs.size()
        )

        outputs = outputs.permute(0, 2, 1, 3)
        outputs = outputs.contiguous().view(
            batch_size, subsampled_lengths, channels * sumsampled_dim
        )

        if self.subsample_rate == 4:
            output_lengths = input_lengths >> 2
        else:
            output_lengths = input_lengths >> 1

        return outputs, output_lengths

    def get_out_dim(self, input_dim: int) -> int:
        # dummy input to get the output dimension
        with torch.no_grad():
            device = next(self.parameters()).device
            inputs = torch.zeros(1, 16, input_dim, device=device)
            input_lengths = torch.tensor([16], device=device)
            outputs, _ = self.forward(inputs, input_lengths)
        return outputs.size(-1)


class PatchwiseConv2dSubampling(nn.Module):
    def __init__(
        self,
        mel_dim: int,
        out_channels: int,
        patch_size_time: int = 16,
        patch_size_freq: int = 16,
    ) -> None:
        super(PatchwiseConv2dSubampling, self).__init__()

        self.mel_dim = mel_dim
        self.patch_size_time = patch_size_time
        self.patch_size_freq = patch_size_freq

        self.proj = nn.Conv2d(
            1,
            out_channels,
            kernel_size=(patch_size_time, patch_size_freq),
            stride=(patch_size_time, patch_size_freq),
            padding=0,
        )
        self.cnn = nn.Sequential(
            nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
        )

    @property
    def subsample_rate(self) -> int:
        return self.patch_size_time * self.patch_size_freq // self.mel_dim

    def forward(
        self, inputs: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert (
            inputs.shape[2] == self.mel_dim
        ), "inputs.shape[2] should be equal to mel_dim"

        # inputs: (B, Time, Freq) -> (B, 1, Time, Freq)
        outputs = self.proj(inputs.unsqueeze(1))
        outputs = self.cnn(outputs)
        # (B, channels, Time // patch_size_time, Freq // patch_size_freq)
        outputs = outputs.flatten(2, 3).transpose(1, 2)
        # (B, (Time // patch_size_time) * (Freq // patch_size_freq), channels)

        output_lengths = (
            input_lengths
            // self.patch_size_time
            * (self.mel_dim // self.patch_size_freq)
        )

        return outputs, output_lengths


class RelPositionalEncoding(nn.Module):
    def __init__(self, d_model: int) -> None:
        super(RelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.pe = None

    def extend_pe(self, x: torch.Tensor) -> None:
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return

        length = x.size(1)
        pe_positive = torch.zeros(length, self.d_model, device="cpu")
        pe_negative = torch.zeros(length, self.d_model, device="cpu")
        position = torch.arange(
            0, length, dtype=torch.float32, device="cpu"
        ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32, device="cpu")
            * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        self.extend_pe(x)
        assert self.pe is not None
        pos_emb = self.pe[
            :,
            self.pe.size(1) // 2
            - x.size(1)
            + 1 : self.pe.size(1) // 2
            + x.size(1),
        ]
        return pos_emb


class RelativeMultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 16,
        dropout_p: float = 0.1,
    ):
        super(RelativeMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model % num_heads should be zero."
        self.d_model = d_model
        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(self.d_head)

        self.query_proj = Linear(d_model, d_model)
        self.key_proj = Linear(d_model, d_model)
        self.value_proj = Linear(d_model, d_model)
        self.pos_proj = Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout_p)
        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)

        self.out_proj = Linear(d_model, d_model)

    @staticmethod
    def _relative_shift(pos_score: torch.Tensor) -> torch.Tensor:
        # pos_score: (B, H, T, 2T-1)
        B, H, T, L = pos_score.size()

        # Pad on the left of the last dimension: (B, H, T, 2T)
        pos_score = F.pad(pos_score, (1, 0))

        # Reshape to (B, H, 2T, T)
        pos_score = pos_score.view(B, H, L + 1, T)

        # Slice and reshape back to (B, H, T, 2T-1)
        pos_score = pos_score[:, :, 1:].view(B, H, T, L)

        # Keep only first T positions => (B, H, T, T)
        return pos_score[:, :, :, : (L // 2 + 1)]

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        pos_embedding: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        *,
        need_weights: bool = False,
        use_sdpa: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        - If need_weights=True: returns (output, attn) like your original code.
        - If need_weights=False: returns (output, None) and uses SDPA in eval for speed/memory.
        """
        B, Tq, _ = query.size()
        _, Tk, _ = key.size()

        # Project
        q = self.query_proj(query)  # (B, Tq, C)
        k = self.key_proj(key)  # (B, Tk, C)
        v = self.value_proj(value)  # (B, Tk, C)

        # Reshape to (B, H, T, Dh)
        q = q.view(B, Tq, self.num_heads, self.d_head).transpose(
            1, 2
        )  # (B,H,Tq,Dh)
        k = k.view(B, Tk, self.num_heads, self.d_head).transpose(
            1, 2
        )  # (B,H,Tk,Dh)
        v = v.view(B, Tk, self.num_heads, self.d_head).transpose(
            1, 2
        )  # (B,H,Tk,Dh)

        # Positional projection.
        # IMPORTANT: allow pos_embedding to be (1, 2T-1, C) and broadcast across batch.
        # pos_embedding expected length: 2Tq - 1 for self-attn.
        pB = pos_embedding.size(0)
        p = self.pos_proj(pos_embedding)  # (pB, L, C)
        p = p.view(pB, -1, self.num_heads, self.d_head).transpose(
            1, 2
        )  # (pB,H,L,Dh)

        # Compute position-based bias (scaled) to feed SDPA or add to scores
        # q_pos: (B,H,Tq,Dh), p^T: (pB,H,Dh,L) -> broadcast on pB if pB==1
        q_pos = q + self.v_bias.unsqueeze(0).unsqueeze(2)  # (B,H,Tq,Dh)
        pos_score = torch.matmul(q_pos, p.transpose(-2, -1))  # (B,H,Tq,L)
        pos_bias = self._relative_shift(pos_score)  # (B,H,Tq,Tq) for self-attn
        pos_bias = pos_bias.mul(
            1.0 / self.sqrt_dim
        )  # scale matches SDPA scaling

        if padding_mask is not None:
            # padding_mask: (B, T) -> (B, 1, 1, T) to broadcast with pos_bias (B, H, Tq, Tk)
            # This masks out key positions that are padded across all heads and queries
            if padding_mask.dtype != torch.bool:
                padding_mask = padding_mask.to(torch.bool)
            pos_bias = pos_bias.masked_fill(
                padding_mask[:, None, None, :], -1e9
            )

        if use_sdpa is None:
            use_sdpa = (not self.training) and (not need_weights)

        # ---- Fast inference path: no attention matrix materialized ----
        if use_sdpa:
            # Content term uses u_bias
            q_content = q + self.u_bias.unsqueeze(0).unsqueeze(
                2
            )  # (B,H,Tq,Dh)

            with sdpa_kernel(
                [
                    SDPBackend.FLASH_ATTENTION,
                    SDPBackend.EFFICIENT_ATTENTION,
                    SDPBackend.MATH,
                ]
            ):
                out = F.scaled_dot_product_attention(
                    q_content,  # (B,H,Tq,Dh)
                    k,  # (B,H,Tk,Dh)
                    v,  # (B,H,Tk,Dh)
                    attn_mask=pos_bias,  # (B,H,Tq,Tk) additive bias
                    dropout_p=0.0,  # dropout disabled in inference
                    is_causal=False,
                )  # (BH, Tq, Dh)

            out = out.transpose(1, 2).contiguous().view(B, Tq, self.d_model)

            return self.out_proj(out), None

        # ---- Reference path (training / if you need attn weights): matches your math ----
        q_content = q + self.u_bias.unsqueeze(0).unsqueeze(2)  # (B,H,Tq,Dh)
        content_score = torch.matmul(
            q_content, k.transpose(-2, -1)
        )  # (B,H,Tq,Tk)
        content_score = content_score.mul(1.0 / self.sqrt_dim)

        score = content_score + pos_bias  # already scaled

        attn = F.softmax(score, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, v)  # (B,H,Tq,Dh)
        context = (
            context.transpose(1, 2).contiguous().view(B, Tq, self.d_model)
        )

        return self.out_proj(context), attn


class MultiHeadedSelfAttentionModule(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout_p: float = 0.1,
        rms_norm: bool = False,
    ):
        super(MultiHeadedSelfAttentionModule, self).__init__()
        self.positional_encoding = RelPositionalEncoding(d_model)
        self.layer_norm = (
            nn.LayerNorm(d_model) if not rms_norm else RMSNorm(d_model)
        )
        self.attention = RelativeMultiHeadAttention(
            d_model, num_heads, dropout_p
        )
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(
        self,
        inputs: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        pos_embedding: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if pos_embedding is None:
            pos_embedding = self.positional_encoding(inputs)

        inputs = self.layer_norm(inputs)
        outputs, attn = self.attention(
            inputs,
            inputs,
            inputs,
            pos_embedding=pos_embedding,
            padding_mask=padding_mask,
        )

        return self.dropout(outputs), attn, pos_embedding


class ConformerBlock(nn.Module):
    def __init__(
        self,
        encoder_dim: int = 512,
        attention_type: str = "mhsa",
        num_attention_heads: int = 8,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        feed_forward_dropout_p: float = 0.1,
        attention_dropout_p: float = 0.1,
        conv_dropout_p: float = 0.1,
        conv_kernel_size: int = 31,
        half_step_residual: bool = True,
        transformer_style: bool = False,
        usad_v2: bool = False,
        pre_norm: bool = False,
        rms_norm: bool = False,
    ):
        super(ConformerBlock, self).__init__()

        self.transformer_style = transformer_style
        self.attention_type = attention_type
        self.usad_v2 = usad_v2
        self.pre_norm = pre_norm

        if half_step_residual and not transformer_style:
            self.feed_forward_residual_factor = 0.5
        else:
            self.feed_forward_residual_factor = 1

        assert (
            attention_type == "mhsa"
        ), "Only 'mhsa' attention is supported in this implementation."
        attention = MultiHeadedSelfAttentionModule(
            d_model=encoder_dim,
            num_heads=num_attention_heads,
            dropout_p=attention_dropout_p,
            rms_norm=rms_norm,
        )

        self.ffn_1 = FeedForwardModule(
            encoder_dim=encoder_dim,
            expansion_factor=feed_forward_expansion_factor,
            dropout_p=feed_forward_dropout_p,
            rms_norm=rms_norm,
        )
        self.attention = attention
        if not transformer_style:
            self.conv = ConformerConvModule(
                in_channels=encoder_dim,
                kernel_size=conv_kernel_size,
                expansion_factor=conv_expansion_factor,
                dropout_p=conv_dropout_p,
                rms_norm=rms_norm,
            )
            self.ffn_2 = FeedForwardModule(
                encoder_dim=encoder_dim,
                expansion_factor=feed_forward_expansion_factor,
                dropout_p=feed_forward_dropout_p,
                rms_norm=rms_norm,
            )
        self.layernorm = (
            (
                nn.LayerNorm(encoder_dim)
                if not rms_norm
                else RMSNorm(encoder_dim)
            )
            if not pre_norm
            else nn.Identity()
        )

    def forward_attention(
        self,
        x: torch.Tensor,
        pos_embedding: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        attn_out, attn, pos_embedding = self.attention(
            x, pos_embedding=pos_embedding, padding_mask=padding_mask
        )
        return attn_out, attn, pos_embedding

    def forward_legacy(
        self,
        x: torch.Tensor,
        pos_embedding: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
        # FFN 1
        ffn_1_out = self.ffn_1(x)
        x = ffn_1_out * self.feed_forward_residual_factor + x

        # Attention
        attn_out, attn, pos_embedding = self.forward_attention(
            x, pos_embedding, padding_mask
        )
        x = attn_out + x

        if self.transformer_style:
            x = self.layernorm(x)
            return x, {
                "ffn_1": ffn_1_out,
                "attn": attn,
                "conv": None,
                "ffn_2": None,
            }

        # Convolution
        conv_out = self.conv(x)
        x = conv_out + x

        # FFN 2
        ffn_2_out = self.ffn_2(x)
        x = ffn_2_out * self.feed_forward_residual_factor + x
        x = self.layernorm(x)

        other = {
            "ffn_1": ffn_1_out,
            "attn": attn,
            "conv": conv_out,
            "ffn_2": ffn_2_out,
            "pos_embedding": pos_embedding,
        }

        return x, other

    def forward_transformer(
        self,
        x: torch.Tensor,
        pos_embedding: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
        # Attention
        attn_out, attn, pos_embedding = self.forward_attention(
            x, pos_embedding, padding_mask
        )
        x = attn_out + x

        # FFN
        ffn_out = self.ffn_1(x)
        x = ffn_out * self.feed_forward_residual_factor + x

        x = self.layernorm(x)
        return x, {
            "ffn_1": ffn_out,
            "attn": attn,
            "conv": None,
            "ffn_2": None,
            "pos_embedding": pos_embedding,
        }

    def forward_conformer(
        self,
        x: torch.Tensor,
        pos_embedding: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
        # FFN 1
        ffn_1_out = self.ffn_1(x)
        x = ffn_1_out * self.feed_forward_residual_factor + x

        # Attention
        attn_out, attn, pos_embedding = self.forward_attention(
            x, pos_embedding, padding_mask
        )
        x = attn_out + x

        # Convolution
        conv_out = self.conv(x)
        x = conv_out + x

        # FFN 2
        ffn_2_out = self.ffn_2(x)
        x = ffn_2_out * self.feed_forward_residual_factor + x
        x = self.layernorm(x)

        other = {
            "ffn_1": ffn_1_out,
            "attn": attn,
            "conv": conv_out,
            "ffn_2": ffn_2_out,
            "pos_embedding": pos_embedding,
        }

        return x, other

    def forward(
        self,
        x: torch.Tensor,
        pos_embedding: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
        if not self.usad_v2:
            return self.forward_legacy(x, pos_embedding, padding_mask)

        if self.transformer_style:
            return self.forward_transformer(x, pos_embedding, padding_mask)

        return self.forward_conformer(x, pos_embedding, padding_mask)


class ConformerEncoder(nn.Module):
    def __init__(self, cfg):
        super(ConformerEncoder, self).__init__()

        self.cfg = cfg
        self.framewise_subsample = None
        self.patchwise_subsample = None
        self.framewise_in_proj = None
        self.patchwise_in_proj = None
        assert (
            cfg.use_framewise_subsample or cfg.use_patchwise_subsample
        ), "At least one subsampling method should be used"
        if cfg.use_framewise_subsample:
            self.framewise_subsample = FramewiseConv2dSubampling(
                out_channels=cfg.conv_subsample_channels,
                subsample_rate=cfg.conv_subsample_rate,
            )
            self.framewise_in_proj = nn.Sequential(
                Linear(
                    self.framewise_subsample.get_out_dim(cfg.input_dim),
                    cfg.encoder_dim,
                ),
                nn.Dropout(p=cfg.input_dropout_p),
            )
        if cfg.use_patchwise_subsample:
            self.patchwise_subsample = PatchwiseConv2dSubampling(
                mel_dim=cfg.input_dim,
                out_channels=cfg.conv_subsample_channels,
                patch_size_time=cfg.patch_size_time,
                patch_size_freq=cfg.patch_size_freq,
            )
            self.patchwise_in_proj = nn.Sequential(
                Linear(
                    cfg.conv_subsample_channels,
                    cfg.encoder_dim,
                ),
                nn.Dropout(p=cfg.input_dropout_p),
            )
            assert not cfg.use_framewise_subsample or (
                cfg.conv_subsample_rate
                == self.patchwise_subsample.subsample_rate
            ), (
                f"conv_subsample_rate ({cfg.conv_subsample_rate}) != patchwise_subsample.subsample_rate"
                f"({self.patchwise_subsample.subsample_rate})"
            )

        self.framewise_norm, self.patchwise_norm = None, None
        if getattr(cfg, "subsample_normalization", False):
            if cfg.use_framewise_subsample:
                self.framewise_norm = (
                    nn.LayerNorm(cfg.encoder_dim)
                    if not getattr(cfg, "rms_norm", False)
                    else RMSNorm(cfg.encoder_dim)
                )
            if cfg.use_patchwise_subsample:
                self.patchwise_norm = (
                    nn.LayerNorm(cfg.encoder_dim)
                    if not getattr(cfg, "rms_norm", False)
                    else RMSNorm(cfg.encoder_dim)
                )

        self.conv_pos = None
        self.conv_pos_post_ln = None
        if cfg.conv_pos:
            num_pos_layers = cfg.conv_pos_depth
            k = max(3, cfg.conv_pos_width // num_pos_layers)
            self.conv_pos = nn.Sequential(
                TransposeLast(),
                *[
                    nn.Sequential(
                        nn.Conv1d(
                            cfg.encoder_dim,
                            cfg.encoder_dim,
                            kernel_size=k,
                            padding=k // 2,
                            groups=cfg.conv_pos_groups,
                        ),
                        SamePad(k),
                        TransposeLast(),
                        nn.LayerNorm(
                            cfg.encoder_dim, elementwise_affine=False
                        ),
                        TransposeLast(),
                        nn.GELU(),
                    )
                    for _ in range(num_pos_layers)
                ],
                TransposeLast(),
            )
            self.conv_pos_post_ln = (
                (
                    nn.LayerNorm(cfg.encoder_dim)
                    if not getattr(cfg, "rms_norm", False)
                    else RMSNorm(cfg.encoder_dim)
                )
                if not getattr(cfg, "pre_norm", False)
                else nn.Identity()
            )

        self.layers = nn.ModuleList(
            [
                ConformerBlock(
                    encoder_dim=cfg.encoder_dim,
                    attention_type=cfg.attention_type,
                    num_attention_heads=cfg.num_attention_heads,
                    feed_forward_expansion_factor=cfg.feed_forward_expansion_factor,
                    conv_expansion_factor=cfg.conv_expansion_factor,
                    feed_forward_dropout_p=cfg.feed_forward_dropout_p,
                    attention_dropout_p=cfg.attention_dropout_p,
                    conv_dropout_p=cfg.conv_dropout_p,
                    conv_kernel_size=cfg.conv_kernel_size,
                    half_step_residual=cfg.half_step_residual,
                    transformer_style=getattr(cfg, "transformer_style", False),
                    usad_v2=getattr(cfg, "usad_v2", False),
                    pre_norm=getattr(cfg, "pre_norm", False),
                    rms_norm=getattr(cfg, "rms_norm", False),
                )
                for _ in range(cfg.num_layers)
            ]
        )
        self.layerdrop_p = getattr(cfg, "layerdrop_p", 0.0)

        if cfg.attention_type == "mhsa" and len(self.layers) > 0:
            # Share positional encoding across layers
            shared_pos = None
            for layer in self.layers:
                if isinstance(layer.attention, MultiHeadedSelfAttentionModule):
                    if shared_pos is None:
                        shared_pos = layer.attention.positional_encoding
                    else:
                        layer.attention.positional_encoding = shared_pos
            if shared_pos is not None:
                # precompute positional encodings
                # expecting most mel inputs to be fewer than 2000 frames (20 seconds)
                max_len = 2000 // cfg.conv_subsample_rate
                shared_pos.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def count_parameters(self) -> int:
        """Count parameters of encoder"""
        return sum([p.numel() for p in self.parameters() if p.requires_grad])

    def update_dropout(self, dropout_p: float) -> None:
        """Update dropout probability of encoder"""
        for name, child in self.named_children():
            if isinstance(child, nn.Dropout):
                child.p = dropout_p

    def forward(
        self,
        inputs: torch.Tensor,
        input_lengths: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        *,
        return_hidden: bool = False,
        freeze_input_layers: bool = False,
        target_layer: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, List[torch.Tensor]]]:
        if input_lengths is None:
            input_lengths = torch.full(
                (inputs.size(0),),
                inputs.size(1),
                dtype=torch.long,
                device=inputs.device,
            )

        with (
            torch.no_grad() if freeze_input_layers else contextlib.ExitStack()
        ):
            frame_feat, patch_feat = None, None
            frame_lengths, patch_lengths = None, None
            if self.framewise_subsample is not None:
                assert self.framewise_in_proj is not None
                frame_feat, frame_lengths = self.framewise_subsample(
                    inputs, input_lengths
                )
                frame_feat = self.framewise_in_proj(frame_feat)
                if self.framewise_norm is not None:
                    frame_feat = self.framewise_norm(frame_feat)

            if self.patchwise_subsample is not None:
                assert self.patchwise_in_proj is not None
                patch_feat, patch_lengths = self.patchwise_subsample(
                    inputs, input_lengths
                )
                patch_feat = self.patchwise_in_proj(patch_feat)
                if self.patchwise_norm is not None:
                    patch_feat = self.patchwise_norm(patch_feat)

            assert frame_feat is not None or patch_feat is not None
            assert frame_lengths is not None or patch_lengths is not None

            if frame_feat is not None and patch_feat is not None:
                assert frame_lengths is not None and patch_lengths is not None
                min_len = min(frame_feat.size(1), patch_feat.size(1))
                frame_feat = frame_feat[:, :min_len]
                patch_feat = patch_feat[:, :min_len]

                features = frame_feat + patch_feat
                output_lengths = (
                    frame_lengths
                    if frame_lengths.max().item() < patch_lengths.max().item()
                    else patch_lengths
                )
            elif frame_feat is not None:
                features = frame_feat
                output_lengths = frame_lengths
            else:
                features = patch_feat
                output_lengths = patch_lengths

            assert features is not None
            assert output_lengths is not None

        # Positional encoding with convolutional layers
        if self.conv_pos is not None and self.conv_pos_post_ln is not None:
            pos = self.conv_pos(features)
            if not self.training:
                features = features.add_(pos)
            else:
                features = features + pos
            features = self.conv_pos_post_ln(features)

        # Create padding mask for attention
        if padding_mask is not None:
            # downsample to match features length
            input_len = padding_mask.size(1)
            feat_len = features.size(1)
            factor = input_len / feat_len
            indices = (
                torch.arange(feat_len, device=padding_mask.device) * factor
            ).long()
            padding_mask = padding_mask.index_select(1, indices)
        else:
            # create from output_lengths
            padding_mask = lengths_to_padding_mask(
                output_lengths, max_len=features.size(1)
            )

        layer_results = defaultdict(list)
        outputs = features
        other = {}
        for i, layer in enumerate(self.layers):
            if (
                self.training
                and self.layerdrop_p > 0
                and torch.rand(1).item() < self.layerdrop_p
            ):
                continue
            outputs, other = layer(
                outputs,
                pos_embedding=other.get("pos_embedding"),
                padding_mask=padding_mask,
            )
            if return_hidden:
                layer_results["hidden_states"].append(outputs)
                for k, v in other.items():
                    layer_results[k].append(v)

            if target_layer is not None and i + 1 == target_layer:
                break

        return outputs, output_lengths, layer_results
