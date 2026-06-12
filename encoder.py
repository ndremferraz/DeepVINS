import torch 
import torch.nn as nn


# Image Encoder following format close to FlowNetSimple
# However, input images are grayscale and 2x752x480
# 7x7 kernel, stride 2, padding 3, output 64 channels
# Max pooling with kernel size 3, stride 2, padding 1
# 5x5 kernel, stride 2, padding 2, output 128 channels
# Max pooling with kernel size 3, stride 2, padding 1
# 5x5 kernel, stride 2, padding 2, output 256 channels
# 3x3 kernel, stride 2, padding 1, output 512 channels
# 3x3 kernel, stride 2, padding 1, output 512 channels
# Flatten and linear layer to output 512 features

class ImageEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.l1 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01)
        )
        self.p1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.l2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01)
        )
        self.p2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.l3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01)
        )
        self.l4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.01)
        )
        self.l5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.01)
        )
        self.l6 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*6*4, 512),
        )

    def forward(self, x):
        batch_size, context_len = x.shape[:2]

        x = x.reshape(-1, 2, 752, 480)
        x = self.l1(x)
        x = self.p1(x)
        x = self.l2(x)
        x = self.p2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = x.reshape(batch_size, context_len, -1) 
        return x


# Inertial Encoder using simple 1D convolutional layers
# Inputs of shape (batch_size, context_len, 70) 
# Reshape 70 to 10x7, then apply 1D conv 
# Following what is described in this paper: https://arxiv.org/pdf/2205.06187
# After the inertial encoder 
class InertialEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.l1 = nn.Sequential(
            nn.Conv1d(7, 64, kernel_size=3, padding='same'),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.01)
        )
        self.l2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding='same'),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.01)
        )
        self.l3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding='same'),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.01)
        )
        self.l4 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*10, 512),
        )


    def forward(self, x):
        # Input shape: (batch_size, context_len, 70) -> (batch_size, context_len, 7, 10)
        batch_size, context_len = x.shape[:2]

        x = x.reshape(-1, 10, 7)
        x = x.transpose(-1, -2)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)  
        x = self.l4(x)
        x = x.reshape(batch_size, context_len, -1) 
        return x