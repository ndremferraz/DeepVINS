import torch 
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import EurocMavDataset
from loss import PoseSequenceLoss
from decoder import CausalFusionModel
from train import Trainer

import matplotlib.pyplot as plt

EPOCHS = 20
CONTEXT_LEN = 12
LEARNING_RATE = 1e-5
LEARNING_RATE_STEP_SIZE = 20
LEARNING_RATE_GAMMA = 0.5
MAX_GRAD_NORM = 1

VALID_INTERVAL = 20

TRAIN_CSVS = ['vicon_room1/V1_01_easy/mav0/train_data.csv', 'vicon_room1/V1_02_medium/mav0/train_data.csv', 'vicon_room1/V1_03_difficult/mav0/train_data.csv']
VALID_CSVS = ['vicon_room1/V1_01_easy/mav0/val_data.csv', 'vicon_room1/V1_02_medium/mav0/val_data.csv', 'vicon_room1/V1_03_difficult/mav0/val_data.csv']

IMG_FOLDER = ['vicon_room1/V1_01_easy/mav0/cam0/data', 'vicon_room1/V1_02_medium/mav0/cam0/data', 'vicon_room1/V1_03_difficult/mav0/cam0/data']

train_dataset = EurocMavDataset(data_file_path=TRAIN_CSVS, image_folder=IMG_FOLDER, context_len=CONTEXT_LEN+1)
valid_dataset = EurocMavDataset(data_file_path=VALID_CSVS, image_folder=IMG_FOLDER, context_len=CONTEXT_LEN+1)

train_loader = DataLoader(train_dataset, 4, shuffle=True)
valid_loader = DataLoader(valid_dataset, 4, shuffle=True)

model = CausalFusionModel(context_length=CONTEXT_LEN)
trainer = Trainer(
    model=model,
    train_data=train_loader,
    val_data=valid_loader,
    checkpoint_path='best.pth',
    total_epochs=EPOCHS,
    lr=LEARNING_RATE,
    lr_step_size=LEARNING_RATE_STEP_SIZE,
    lr_gamma=LEARNING_RATE_GAMMA,
    max_grad_norm=MAX_GRAD_NORM
)

train_results, val_results = trainer.train(val_interval=20 ,loss_fn=PoseSequenceLoss())

train_x = range(len(train_results))

val_x = range(0, len(train_results), 20)  

plt.plot(train_x, train_results, label="Train Loss", color="green")
plt.plot(val_x, val_results, label="Validation Loss", color="red", marker="o")

plt.xlabel("Training Iteration")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.show()