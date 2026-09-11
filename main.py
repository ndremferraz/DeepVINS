import torch 
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import EurocMavDataset
from loss import PoseSequenceLoss
from decoder import CausalFusionModel
from train import Trainer

import matplotlib.pyplot as plt

EPOCHS = 1
LEARNING_RATE = 1e-5
LEARNING_RATE_STEP_SIZE = 20
LEARNING_RATE_GAMMA = 0.5
MAX_GRAD_NORM = 1

SEQUENCE_LEN = 8        

VALID_INTERVAL = 20

DATASET_ROOT = './DeepVINS datasets/'

TRAIN_CSVS = [
    f'{DATASET_ROOT}MH_01_easy/mav0/train_data.csv',
    f'{DATASET_ROOT}MH_02_easy/mav0/train_data.csv',
    f'{DATASET_ROOT}MH_03_medium/mav0/train_data.csv',
    f'{DATASET_ROOT}MH_04_difficult/mav0/train_data.csv',
    f'{DATASET_ROOT}MH_05_difficult/mav0/train_data.csv',
    f'{DATASET_ROOT}V1_01_easy/mav0/train_data.csv',
    f'{DATASET_ROOT}V1_02_medium/mav0/train_data.csv',
    f'{DATASET_ROOT}V1_03_difficult/mav0/train_data.csv',
    f'{DATASET_ROOT}V2_02_medium/mav0/train_data.csv',
    f'{DATASET_ROOT}V2_03_difficult/mav0/train_data.csv'
]

VALID_CSVS = [
    f'{DATASET_ROOT}MH_01_easy/mav0/val_data.csv',
    f'{DATASET_ROOT}MH_02_easy/mav0/val_data.csv',
    f'{DATASET_ROOT}MH_03_medium/mav0/val_data.csv',
    f'{DATASET_ROOT}MH_04_difficult/mav0/val_data.csv',
    f'{DATASET_ROOT}MH_05_difficult/mav0/val_data.csv',
    f'{DATASET_ROOT}V1_01_easy/mav0/val_data.csv',
    f'{DATASET_ROOT}V1_02_medium/mav0/val_data.csv',
    f'{DATASET_ROOT}V1_03_difficult/mav0/val_data.csv',
    f'{DATASET_ROOT}V2_02_medium/mav0/val_data.csv',
    f'{DATASET_ROOT}V2_03_difficult/mav0/val_data.csv'
]

IMG_FOLDERS = [cv.replace('train_data.csv', 'cam0/data/') for cv in TRAIN_CSVS]

train_dataset = EurocMavDataset(data_file_path=TRAIN_CSVS, image_folder=IMG_FOLDERS, sequence_length=SEQUENCE_LEN)
valid_dataset = EurocMavDataset(data_file_path=VALID_CSVS, image_folder=IMG_FOLDERS, sequence_length=SEQUENCE_LEN)

train_loader = DataLoader(train_dataset, 4, shuffle=True)
valid_loader = DataLoader(valid_dataset, 4, shuffle=True)


model = CausalFusionModel(context_length=SEQUENCE_LEN)
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

train_results, val_results = trainer.train(val_interval=VALID_INTERVAL, loss_fn=PoseSequenceLoss())

train_x = range(len(train_results))

val_x = range(0, len(val_results) * VALID_INTERVAL, VALID_INTERVAL)  

plt.plot(train_x, train_results, label="Train Loss", color="green")
plt.plot(val_x, val_results, label="Validation Loss", color="red", marker="o")

plt.xlabel("Training Iteration")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.show()

