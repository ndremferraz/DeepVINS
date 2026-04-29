import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.init import kaiming_normal_, orthogonal_
import numpy as np
from torch.distributions.utils import broadcast_all, probs_to_logits, logits_to_probs, lazy_property, clamp_probs
import torch.nn.functional as F
import math


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


class InertialBlock(nn.Module):

    def __init__(self, 
        input_dims: int = 6,
        l1_filters: int = 16 , l2_filters: int = 16,
        l3_dims: int = 128,
        l4_dims: int = 128,
        fc1_dims: int = 256,
        fc2_dims: int = 512
    ):

        super().__init__()

        self.input_dims = input_dims
        l1_size = input_dims * l1_filters
        l2_size = l1_size * l2_filters

        # Depthwise convolution is applied with the intent to filter each channel individually
        self.l1 = nn.Sequential(
            nn.Conv1d(
                in_channels=input_dims,
                out_channels=l1_size,
                kernel_size=3,
                padding=1,
                groups=input_dims
            ),
            nn.GroupNorm(num_groups=input_dims, num_channels=l1_size),
            nn.LeakyReLU(0.01)    
        )
        self.l2 = nn.Sequential(
            nn.Conv1d(
                in_channels=l1_size,
                out_channels=l2_size,
                kernel_size=3,
                padding=1,
                groups=l1_size
            ),
            nn.GroupNorm(num_groups=l1_size, num_channels=l2_size),
            nn.LeakyReLU(0.01)    
        )
        
        # 1D convolution along time axis across different channels
        self.l3 = nn.Sequential(
            nn.Conv1d(
                in_channels=l2_size,
                out_channels=l3_dims,
                kernel_size=3,
                padding=1
            ),
            nn.GroupNorm(num_groups=8, num_channels=l3_dims),
            nn.LeakyReLU(0.01)
        )
        self.l4 = nn.Sequential(
            nn.Conv1d(
                in_channels=l3_dims,
                out_channels=l4_dims,
                kernel_size=3,
                padding=1
            ),
            nn.GroupNorm(num_groups=8, num_channels=l4_dims),
            nn.LeakyReLU(0.01)
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(l4_dims, fc1_dims)
        self.fc1_residual = nn.Linear(l2_size, fc1_dims)
        self.fc1_norm = nn.LayerNorm(fc1_dims)
        self.fc1_activation = nn.LeakyReLU(0.01)

        self.fc2 = nn.Sequential(
            nn.Linear(fc1_dims, fc2_dims),
            nn.LayerNorm(fc2_dims),
            nn.LeakyReLU(0.01)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected IMU input with shape [B, C, T] or [B, T, C], got {tuple(x.shape)}")

        if x.size(1) != self.input_dims and x.size(-1) == self.input_dims:
            x = x.transpose(1, 2)

        # Standardize each IMU window independently to reduce sensitivity
        # to raw sensor scale and dataset-specific offsets.
        x = x - x.mean(dim=-1, keepdim=True)
        x = x / x.std(dim=-1, keepdim=True).clamp_min(1e-6)
        _check_finite("imu_input_norm", x)

        x = self.l1(x)
        _check_finite("imu_l1", x)
        x = self.l2(x)
        _check_finite("imu_l2", x)
        residual = self.pool(x).squeeze(-1)
        _check_finite("imu_residual", residual)
        x = self.l3(x)
        _check_finite("imu_l3", x)
        x = self.l4(x)
        _check_finite("imu_l4", x)
        x = self.pool(x).squeeze(-1)
        _check_finite("imu_pool", x)
        x = self.fc1(x) + self.fc1_residual(residual)
        _check_finite("imu_fc1_sum", x)
        x = self.fc1_norm(x)
        _check_finite("imu_fc1_norm", x)
        x = self.fc1_activation(x)
        _check_finite("imu_fc1_act", x)
        x = self.fc2(x)
        _check_finite("imu_fc2", x)
        return x
    

class ImageBlock(nn.Module):

    def __init__(self, 
        i_channels: int = 2,
        l1_filters: int = 16, l2_filters: int = 32,
        l3_dims: int = 64, l4_dims: int = 128,
        l5_dims: int = 256, l6_dims: int = 256,
        fc1_dims: int = 512, fc2_dims: int = 512
    ):

        super().__init__()

        self.output_dims = fc2_dims
        l1_size = i_channels * l1_filters
        l2_size = l1_size

        # Two depthwise stages let each image develop independent low-level features
        # before the later layers start mixing the pair together.
        self.l1 = nn.Sequential(
            nn.Conv2d(
                in_channels=i_channels,
                out_channels=l1_size,
                kernel_size=5,
                stride=2,
                padding=2,
                groups=i_channels
            ),
            nn.BatchNorm2d(l1_size),
            nn.LeakyReLU(0.01)
        )
        self.l2 = nn.Sequential(
            nn.Conv2d(
                in_channels=l1_size,
                out_channels=l2_size,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=l1_size
            ),
            nn.BatchNorm2d(l2_size),
            nn.LeakyReLU(0.01)
        )
        self.l3 = nn.Sequential(
            nn.Conv2d(
                in_channels=l2_size,
                out_channels=l2_filters,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(l2_filters),
            nn.LeakyReLU(0.01)
        )
        self.l4 = nn.Sequential(
            nn.Conv2d(
                in_channels=l2_filters,
                out_channels=l3_dims,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(l3_dims),
            nn.LeakyReLU(0.01)
        )
        self.l5 = nn.Sequential(
            nn.Conv2d(
                in_channels=l3_dims,
                out_channels=l4_dims,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(l4_dims),
            nn.LeakyReLU(0.01)
        )
        self.l6 = nn.Sequential(
            nn.Conv2d(
                in_channels=l4_dims,
                out_channels=l5_dims,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(l5_dims),
            nn.LeakyReLU(0.01)
        )
        self.l7 = nn.Sequential(
            nn.Conv2d(
                in_channels=l5_dims,
                out_channels=l6_dims,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(l6_dims),
            nn.LeakyReLU(0.01)
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(l6_dims, fc1_dims)
        self.fc1_residual = nn.Linear(l2_size, fc1_dims)
        self.fc1_norm = nn.LayerNorm(fc1_dims)
        self.fc1_activation = nn.LeakyReLU(0.01)

        self.fc2 = nn.Sequential(
            nn.Linear(fc1_dims, fc2_dims),
            nn.LayerNorm(fc2_dims),
            nn.LeakyReLU(0.01)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.l1(x)
        x = self.l2(x)
        residual = self.pool(x).flatten(1)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.pool(x).flatten(1)
        x = self.fc1(x) + self.fc1_residual(residual)
        x = self.fc1_norm(x)
        x = self.fc1_activation(x)
        x = self.fc2(x)
        return x
