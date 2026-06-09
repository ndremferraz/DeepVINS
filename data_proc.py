import pandas as pd 
import numpy as np 
import os 

IMU_PER_IMG = 10

SEQ_LIST = ['vicon_room1/V1_01_easy', 'vicon_room1/V1_02_medium', 'vicon_room1/V1_03_difficult']

# Reads the IMU and image data for each sequence into a single dataframe that contains already alligned imu + img data.
for seq in SEQ_LIST:

    imu_df = pd.read_csv(f'{seq}/mav0/imu0/data.csv')
    img_nm_df = pd.read_csv(f'{seq}/mav0/cam0/data.csv')
    gt_df = pd.read_csv(f'{seq}/mav0/vicon0/data.csv')
    
    imu_arr = imu_df.to_numpy()

    if len(imu_arr) % IMU_PER_IMG != 0:
        print(f"Warning: Number of IMU readings in {seq} is not a multiple of {IMU_PER_IMG}. Truncating extra readings.")
        imu_arr = imu_arr[:len(imu_arr) // IMU_PER_IMG * IMU_PER_IMG]

    imu_arr = imu_arr.reshape(imu_arr.shape[0]//IMU_PER_IMG, IMU_PER_IMG * imu_arr.shape[1])
    imu_df = pd.DataFrame(imu_arr)

    input_df = pd.concat([img_nm_df, imu_df], axis=1)

    input_df['#timestamp [ns]'] = input_df['#timestamp [ns]'].astype('float64')
    gt_df['#timestamp [ns]'] = gt_df['#timestamp [ns]'].astype('float64')
    
    input_df = pd.merge_asof(input_df.dropna(subset=['#timestamp [ns]']), gt_df.dropna(subset=['#timestamp [ns]']), on='#timestamp [ns]', direction='nearest')

    imu_data = input_df.iloc[:, 2:72]
    print(imu_data.head())

    input_df.to_csv(f'{seq}/mav0/input_data.csv', index=False)








