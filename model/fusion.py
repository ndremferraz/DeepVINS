import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.init import kaiming_normal_, orthogonal_
import numpy as np
from torch.distributions.utils import broadcast_all, probs_to_logits, logits_to_probs, lazy_property, clamp_probs
import torch.nn.functional as F
import math
from model.encoders import InertialBlock, ImageBlock


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
        x = x + self.dropout(attn_out)

        x = x + self.l2(self.norm2(x))
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
        img_token = self.img_encoder(img.reshape(batch_size * sequence_length, *img.shape[2:]))

        imu_token = imu_token.reshape(batch_size, sequence_length, -1)
        img_token = img_token.reshape(batch_size, sequence_length, -1)

        x = torch.cat((imu_token, img_token), dim=-1)
        x = x + self.positional_embedding[:, :sequence_length]
        x = self.embedding_dropout(x)

        for block in self.attention_blocks:
            x = block(x)

        x = self.final_norm(x)
        return self.head(x)
