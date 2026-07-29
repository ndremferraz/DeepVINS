import torch 
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import EurocMavDataset
from loss import PoseSequenceLoss
from decoder import CausalFusionModel
from train import Trainer


EPOCHS = 10
CONTEXT_LEN = 12
LEARNING_RATE = 1e-5
LEARNING_RATE_STEP_SIZE = 20
LEARNING_RATE_GAMMA = 0.5
MAX_GRAD_NORM = 1



SEQ_LIST = ['vicon_room1/V1_01_easy', 'vicon_room1/V1_02_medium', 'vicon_room1/V1_03_difficult']
dataset = EurocMavDataset(seq_path_list=SEQ_LIST, context_len=CONTEXT_LEN+1)
data_loader = DataLoader(dataset, 4)

model = CausalFusionModel(context_length=CONTEXT_LEN)
trainer = Trainer(
    model=model,
    train_data=data_loader,
    val_data=data_loader,
    checkpoint_path='checkpoints/checkpoint.pth',
    metrics_path='metrics/metrics.json',
    total_epochs=EPOCHS,
    lr=LEARNING_RATE,
    lr_step_size=LEARNING_RATE_STEP_SIZE,
    lr_gamma=LEARNING_RATE_GAMMA,
    max_grad_norm=MAX_GRAD_NORM
)

trainer.train(loss_fn=PoseSequenceLoss())