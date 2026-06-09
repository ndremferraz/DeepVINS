import torch 
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class EurocMavDataset(Dataset):
    def __init__(self, seq_path_list: list, context_len=64):
        
        self.seq_path_list = seq_path_list
        self.data = []
        for seq_path in seq_path_list:
            df = pd.read_csv(f'{seq_path}/mav0/input_data.csv')
            self.data.append(df)
            print(len(df))
            print(len(df) // context_len)

        self.context_len = context_len
        self.sequence_lengths = [len(df) // context_len for df in self.data]

    def __len__(self):
        return sum(self.sequence_lengths)

    def __getitem__(self, idx):
        
        seq_idx = 0
        while idx >= self.sequence_lengths[seq_idx]:
            idx -= self.sequence_lengths[seq_idx]
            seq_idx += 1

        start_idx = idx * self.context_len
        end_idx = start_idx + self.context_len

        img_names = self.data[seq_idx]['filename'].iloc[start_idx:end_idx].values
        img_names = img_names.tolist()

        image_data = []
        for img_name in img_names:
            img_path = f'{self.seq_path_list[seq_idx]}/mav0/cam0/data/{img_name}'
            img = Image.open(img_path).convert('L')
            image_data.append(np.array(img))
        image_data = np.stack(image_data, axis=0)
        image_data = torch.from_numpy(image_data).float() / 255.0

        # The IMU data starts from column index 2 and goes up to index 72 (exclusive), which gives us 70 columns of IMU data.
        # I hardcoded these indices based on the structure of the input_data.csv file created in data_proc.py.
        # I wanted to save time
        imu_data = self.data[seq_idx].iloc[start_idx:end_idx, 2:72].to_numpy()
        imu_data = torch.tensor(imu_data, dtype=torch.float32)

        gt_data = self.data[seq_idx].iloc[start_idx:end_idx, 72:].to_numpy()
        gt_data = torch.tensor(gt_data, dtype=torch.float32)

        return image_data, imu_data, gt_data