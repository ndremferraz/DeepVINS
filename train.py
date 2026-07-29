import torch
import torch.nn as nn
import time

from torch.utils.data import DataLoader

class Trainer:

    def __init__(
        self, 
        model: nn.Module, 
        train_data: DataLoader,
        val_data: DataLoader,
        checkpoint_path: str,
        metrics_path: str,
        total_epochs: int,
        lr: float,
        lr_step_size: int,
        lr_gamma: float,
        max_grad_norm: float):

        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.checkpoint_path = checkpoint_path
        self.metrics_path = metrics_path

        self.total_epochs = total_epochs

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=lr_step_size, gamma=lr_gamma)
        self.max_grad_norm = max_grad_norm

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _run_batch(self, imu_batch, img_batch, input_pose, target_pose, loss_fn: nn.Module):

        self.model.train()

        imu_batch = imu_batch.to(self.device)
        img_batch = img_batch.to(self.device)
        input_pose = input_pose.to(self.device)
        target_pose = target_pose.to(self.device)

        self.optimizer.zero_grad()

        predict = self.model(img_batch,imu_batch)
        loss, metrics = loss_fn(predict, input_pose, target_pose)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return metrics


    def train(self, loss_fn: nn.Module):

        self.model.to(self.device)

        for epoch in range(self.total_epochs):

            print(f"Epoch {epoch + 1}/{self.total_epochs}")

            for img_batch, imu_batch, input_pose, target_pose in self.train_data:

                metrics = self._run_batch(imu_batch, img_batch, input_pose, target_pose, loss_fn)

                print(f"Train Metrics: {metrics}")















    


        



        






