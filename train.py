from pathlib import Path

import torch

from dataset_utils import build_euroc_loader
from loss_utill import PoseSequenceLoss
from model.fusion import CausalFusionModel


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 1.0
LR_DECAY_STEP = 20
LR_DECAY_GAMMA = 0.5

EMBED_DIM = 512
NUM_HEADS = 4
NUM_LAYERS = 4
DROPOUT = 0.1
OUTPUT_DIM = 7

CHECKPOINT_DIR = Path("checkpoints")
DATASET_DIR = Path(".")

# Adjust this split if you want a different validation sequence.
TRAIN_DATASET_FILES = [
    "MH_01_easy_dataset.pt",
    "MH_02_easy_dataset.pt",
    "MH_03_medium_dataset.pt",
    "MH_04_difficult_dataset.pt",
    "V1_02_medium_dataset.pt",
    "V1_03_difficult_dataset.pt",
    "V2_01_easy_dataset.pt",
    "V2_02_medium_dataset.pt",
]
VAL_DATASET_FILES = [
    "V2_03_difficult_dataset.pt",
]

def get_dataloaders():
    train_files = [DATASET_DIR / filename for filename in TRAIN_DATASET_FILES]
    val_files = [DATASET_DIR / filename for filename in VAL_DATASET_FILES]

    train_loader = build_euroc_loader(
        dataset_files=train_files,
        batch_size=BATCH_SIZE,
        shuffle_files=True,
        shuffle_examples=True,
    )
    val_loader = build_euroc_loader(
        dataset_files=val_files,
        batch_size=BATCH_SIZE,
        shuffle_files=False,
        shuffle_examples=False,
    )
    return train_loader, val_loader


def parse_loss_output(loss_output):
    if torch.is_tensor(loss_output):
        return loss_output, {"loss": float(loss_output.detach().item())}

    loss, metrics = loss_output
    parsed_metrics = {"loss": float(loss.detach().item())}
    for key, value in metrics.items():
        parsed_metrics[key] = float(value.detach().item()) if torch.is_tensor(value) else float(value)
    return loss, parsed_metrics


def train_one_epoch(model, loader, loss_fn, optimizer):
    model.train()
    metric_sums = {}
    total_examples = 0

    for batch in loader:
        imu = batch["imu"].to(DEVICE)
        img = batch["img"].to(DEVICE)
        target = batch["target"].to(DEVICE)
        batch_size = int(target.shape[0])

        prediction = model(imu, img)
        loss_output = loss_fn(prediction, target, batch)
        loss, metrics = parse_loss_output(loss_output)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()

        total_examples += batch_size
        for key, value in metrics.items():
            metric_sums[key] = metric_sums.get(key, 0.0) + value * batch_size

    return {key: value / total_examples for key, value in metric_sums.items()}


def validate(model, loader, loss_fn):
    model.eval()
    metric_sums = {}
    total_examples = 0

    with torch.no_grad():
        for batch in loader:
            imu = batch["imu"].to(DEVICE)
            img = batch["img"].to(DEVICE)
            target = batch["target"].to(DEVICE)
            batch_size = int(target.shape[0])

            prediction = model(imu, img)
            loss_output = loss_fn(prediction, target, batch)
            _, metrics = parse_loss_output(loss_output)

            total_examples += batch_size
            for key, value in metrics.items():
                metric_sums[key] = metric_sums.get(key, 0.0) + value * batch_size

    return {key: value / total_examples for key, value in metric_sums.items()}


def save_checkpoint(model, optimizer, epoch, best_val_loss, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def main():
    model = CausalFusionModel(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        output_dim=OUTPUT_DIM,
    ).to(DEVICE)

    loss_fn = PoseSequenceLoss().to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=LR_DECAY_STEP,
        gamma=LR_DECAY_GAMMA,
    )

    train_loader, val_loader = get_dataloaders()
    best_val_loss = float("inf")

    print(f"device: {DEVICE}")
    print(f"train files: {TRAIN_DATASET_FILES}")
    print(f"val files: {VAL_DATASET_FILES}")

    for epoch in range(EPOCHS):
        train_metrics = train_one_epoch(model, train_loader, loss_fn, optimizer)
        val_metrics = validate(model, val_loader, loss_fn)

        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"train: {train_metrics}")
        print(f"val: {val_metrics}")
        print(f"lr: {optimizer.param_groups[0]['lr']:.8f}")

        val_loss = val_metrics["loss"]

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model,
                optimizer,
                epoch,
                best_val_loss,
                CHECKPOINT_DIR / "best.pt",
            )

        save_checkpoint(
            model,
            optimizer,
            epoch,
            best_val_loss,
            CHECKPOINT_DIR / "latest.pt",
        )

        scheduler.step()


if __name__ == "__main__":
    main()
