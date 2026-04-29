from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader, IterableDataset


def load_pt_dataset(path: str | Path) -> dict[str, torch.Tensor]:
    path = Path(path)
    payload = torch.load(path, map_location="cpu")

    if not isinstance(payload, dict):
        raise TypeError(
            f"Unsupported dataset format in {path}. Expected a dict, got {type(payload)}."
        )

    images = payload["images"]
    imu = payload["imus"]
    target = payload["gts"]

    if not (torch.is_tensor(images) and torch.is_tensor(imu) and torch.is_tensor(target)):
        raise TypeError(f"Dataset {path} must contain tensor keys 'images', 'imus', and 'gts'.")

    num_examples = images.shape[0]
    if imu.shape[0] != num_examples or target.shape[0] != num_examples:
        raise ValueError(
            f"Mismatched example counts in {path}: "
            f"images={images.shape[0]}, imu={imu.shape[0]}, target={target.shape[0]}"
        )

    return {
        "img": images,
        "imu": imu,
        "target": target,
        "dataset_name": path.stem,
    }


class EurocMavBatchDataset(IterableDataset):
    def __init__(
        self,
        dataset_files: list[str | Path],
        batch_size: int,
        shuffle_files: bool = True,
        shuffle_examples: bool = True,
    ):
        super().__init__()
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if not dataset_files:
            raise ValueError("dataset_files cannot be empty.")

        self.dataset_files = [Path(path) for path in dataset_files]
        self.batch_size = batch_size
        self.shuffle_files = shuffle_files
        self.shuffle_examples = shuffle_examples

    def __iter__(self):
        dataset_files = list(self.dataset_files)
        if self.shuffle_files:
            random.shuffle(dataset_files)

        for dataset_file in dataset_files:
            dataset = load_pt_dataset(dataset_file)
            num_examples = dataset["img"].shape[0]

            indices = torch.arange(num_examples)
            if self.shuffle_examples:
                indices = indices[torch.randperm(num_examples)]

            usable_examples = (num_examples // self.batch_size) * self.batch_size
            indices = indices[:usable_examples]

            for start in range(0, usable_examples, self.batch_size):
                batch_indices = indices[start : start + self.batch_size]
                yield {
                    "img": dataset["img"][batch_indices],
                    "imu": dataset["imu"][batch_indices],
                    "target": dataset["target"][batch_indices],
                    "dataset_name": dataset["dataset_name"],
                }


def build_euroc_loader(
    dataset_files: list[str | Path],
    batch_size: int,
    shuffle_files: bool = True,
    shuffle_examples: bool = True,
) -> DataLoader:
    dataset = EurocMavBatchDataset(
        dataset_files=dataset_files,
        batch_size=batch_size,
        shuffle_files=shuffle_files,
        shuffle_examples=shuffle_examples,
    )
    return DataLoader(dataset, batch_size=None)
