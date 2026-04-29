import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.init import kaiming_normal_, orthogonal_
import numpy as np
from torch.distributions.utils import broadcast_all, probs_to_logits, logits_to_probs, lazy_property, clamp_probs
import torch.nn.functional as F
import math
from model.encoders import InertialBlock, ImageBlock


def _tensor_stats(name: str, tensor: torch.Tensor) -> str:
    tensor = tensor.detach()
    return (
        f"{name}: shape={tuple(tensor.shape)} "
        f"min={tensor.min().item():.6f} "
        f"max={tensor.max().item():.6f} "
        f"mean={tensor.mean().item():.6f}"
    )


def _check_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise ValueError(f"Non-finite tensor detected. {_tensor_stats(name, tensor)}")


class AttentionBlock(nn.Module):
    def __init__(self, input_dim=1024, embed_dim=1024, num_heads=4, dropout=0.1):
        super().__init__()

        self.l1 = nn.Identity() if input_dim == embed_dim else nn.Linear(input_dim, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(embed_dim)

        self.l2 = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.l1(x)
        _check_finite("attn_l1", x)
        t = x.size(1)
        causal_mask = torch.triu(
            torch.ones(t, t, device=x.device, dtype=torch.bool),
            diagonal=1
        )

        attn_input = self.norm1(x)
        attn_out, _ = self.attn(
            attn_input,
            attn_input,
            attn_input,
            attn_mask=causal_mask,
            need_weights=False
        )
        _check_finite("attn_out", attn_out)
        x = x + self.dropout(attn_out)
        _check_finite("attn_residual", x)

        x = x + self.l2(self.norm2(x))
        _check_finite("ffn_residual", x)
        return x


class CausalFusionModel(nn.Module):
    def __init__(
        self,
        encoder_dim: int = 512,
        token_dim: int = 1024,
        num_heads: int = 4,
        num_layers: int = 4,
        dropout: float = 0.1,
        output_dim: int = 6,
        max_sequence_length: int = 128,
    ):
        super().__init__()

        self.imu_encoder = InertialBlock(fc2_dims=encoder_dim)
        self.img_encoder = ImageBlock(fc2_dims=encoder_dim)

        if token_dim != 2 * encoder_dim:
            raise ValueError(
                f"token_dim must equal 2 * encoder_dim so the fused timestep token is [imu, img]. "
                f"Got token_dim={token_dim}, encoder_dim={encoder_dim}"
            )

        self.token_dim = token_dim
        self.positional_embedding = nn.Parameter(torch.zeros(1, max_sequence_length, token_dim))
        self.embedding_dropout = nn.Dropout(dropout)

        self.attention_blocks = nn.ModuleList([
            AttentionBlock(
                input_dim=token_dim,
                embed_dim=token_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(token_dim)
        self.head = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim, output_dim)
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="leaky_relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                nn.init.xavier_uniform_(module.in_proj_weight)
                if module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)
                nn.init.xavier_uniform_(module.out_proj.weight)
                if module.out_proj.bias is not None:
                    nn.init.zeros_(module.out_proj.bias)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        nn.init.normal_(self.positional_embedding, mean=0.0, std=0.02)

    def forward(self, imu: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        if imu.ndim == 3:
            imu = imu.unsqueeze(1)
        if img.ndim == 4:
            img = img.unsqueeze(1)

        if imu.ndim != 4 or img.ndim != 5:
            raise ValueError(
                f"Expected imu [B, T, W, C] or [B, W, C] and img [B, T, C, H, W] or [B, C, H, W], "
                f"got imu {tuple(imu.shape)} and img {tuple(img.shape)}"
            )

        batch_size, sequence_length = imu.shape[:2]
        if sequence_length > self.positional_embedding.size(1):
            raise ValueError(
                f"Sequence length {sequence_length} exceeds max_sequence_length "
                f"{self.positional_embedding.size(1)}"
            )

        imu_token = self.imu_encoder(imu.reshape(batch_size * sequence_length, *imu.shape[2:]))
        _check_finite("imu_token_flat", imu_token)
        img_token = self.img_encoder(img.reshape(batch_size * sequence_length, *img.shape[2:]))
        _check_finite("img_token_flat", img_token)

        imu_token = imu_token.reshape(batch_size, sequence_length, -1)
        img_token = img_token.reshape(batch_size, sequence_length, -1)
        _check_finite("imu_token", imu_token)
        _check_finite("img_token", img_token)

        x = torch.cat((imu_token, img_token), dim=-1)
        _check_finite("fused_token_concat", x)
        x = x + self.positional_embedding[:, :sequence_length]
        x = self.embedding_dropout(x)
        _check_finite("fused_token_with_pos", x)

        for block_idx, block in enumerate(self.attention_blocks):
            x = block(x)
            _check_finite(f"attention_block_{block_idx}", x)

        x = self.final_norm(x)
        _check_finite("final_norm", x)
        x = self.head(x)
        _check_finite("head_output", x)
        return x
