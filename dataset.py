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

        self.context_len = context_len
        self.sequence_lengths = [len(df) // context_len for df in self.data]

    def __len__(self):
        return sum(self.sequence_lengths)


    # Returns examples that have n = context_length - 1, 
    # Because of the way we use subsequent frames
    def __getitem__(self, idx):
        
        seq_idx = 0
        while idx >= self.sequence_lengths[seq_idx]:
            idx -= self.sequence_lengths[seq_idx]
            seq_idx += 1

        start_idx = idx * self.context_len
        end_idx = start_idx + self.context_len

        img_names = self.data[seq_idx]['filename'].iloc[start_idx:end_idx].values
        img_names = img_names.tolist()

        img_data = []

        for i in range(len(img_names) - 1):
            
            frame1_path = f'{self.seq_path_list[seq_idx]}/mav0/cam0/data/{img_names[i]}'
            frame2_path = f'{self.seq_path_list[seq_idx]}/mav0/cam0/data/{img_names[i+1]}'
            
            frame1_img = Image.open(frame1_path).convert('L')
            frame2_img = Image.open(frame2_path).convert('L')

            frame1_img = np.array(frame1_img)
            frame2_img = np.array(frame2_img)

            img_data.append(np.stack((frame1_img,frame2_img)))

        img_data = np.stack(img_data)
        img_data = torch.from_numpy(img_data).float() / 255.0

        # The IMU data starts from column index 2 and goes up to index 72 (exclusive), which gives us 70 columns of IMU data.
        # I hardcoded these indices based on the structure of the input_data.csv file created in data_proc.py.
        # I wanted to save time
        imu_data = self.data[seq_idx].iloc[start_idx:(end_idx - 1), 2:72].to_numpy()
        imu_data = torch.tensor(imu_data, dtype=torch.float32)

        pose_data = self.data[seq_idx].iloc[start_idx:end_idx, 72:].to_numpy()
        
        input_pose = torch.tensor(pose_data[:-1])
        output_pose = torch.tensor(pose_data[1:])

        return img_data, imu_data, input_pose, output_pose
    
