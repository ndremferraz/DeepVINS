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
    def __init__(self, input_dim=1024, embed_dim=512, num_heads=4, dropout=0.1):
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
        embed_dim: int = 512,
        num_heads: int = 4,
        num_layers: int = 4,
        dropout: float = 0.1,
        output_dim: int = 6
    ):
        super().__init__()

        self.imu_encoder = InertialBlock(fc2_dims=embed_dim)
        self.img_encoder = ImageBlock(fc2_dims=embed_dim)

        self.positional_embedding = nn.Parameter(torch.zeros(1, 2, embed_dim))
        self.embedding_dropout = nn.Dropout(dropout)

        self.attention_blocks = nn.ModuleList([
            AttentionBlock(
                input_dim=embed_dim,
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, output_dim)
        )

    def forward(self, imu: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        imu_token = self.imu_encoder(imu)
        img_token = self.img_encoder(img)

        x = torch.stack((imu_token, img_token), dim=1)
        x = x + self.positional_embedding[:, :x.size(1)]
        x = self.embedding_dropout(x)

        for block in self.attention_blocks:
            x = block(x)

        fused = self.final_norm(x[:, -1])
        return self.head(fused)

