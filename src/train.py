from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import random_split, DataLoader
from transforms import train_transforms, test_transforms
from torchvision.datasets import CIFAR10
from configs import SEED, B, WORKERS
from train_functions import train
from pathlib import Path
from model import VisionTransformer
import argparse
import torch.nn as nn
import torch
import time

def main():
    # add command line argument to resume training at a certain checkpoint
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from."
    )
    args = parser.parse_args()

    # load complete CIFAR10 training dataset
    train_dataset = CIFAR10(
        root="../data",
        train=True,
        transform=train_transforms,
        download=True
    )

    # split training dataset into train and validation sets
    generator_ = torch.Generator().manual_seed(SEED)
    train_dataset, val_dataset = random_split(
        train_dataset, [0.8, 0.2], generator=generator_
    )

    # create train and validation dataloaders
    train_dl = DataLoader(train_dataset, batch_size=B, shuffle=True,
                          num_workers=WORKERS, pin_memory=True, persistent_workers=True)
    val_dl = DataLoader(val_dataset, batch_size=B, num_workers=WORKERS,
                        pin_memory=True, persistent_workers=True)

    # use GPU instead of CPU, if possible
    device = "cuda" if torch.cuda.is_available() else "cpu"

    start_epoch = 0
    num_epochs = 100

    model = VisionTransformer().to(device, non_blocking=True)
    loss_fn = nn.CrossEntropyLoss.to(device, non_blocking=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = CosineAnnealingLR(optimizer, num_epochs, 1e-4)

    if args.resume is not None:
        checkpoint = torch.load(args.resume)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]

        print(f"Resuming from epoch {start_epoch}")

    # 3. training loop
    checkpoint_dir = Path("../outputs/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, num_epochs):
        print(f"Starting Epoch {epoch+1}")

        # track time it takes for one epoch to complete
        start_time = time.perf_counter()

        # a. train model
        # train_loss = train(model, loss_fn, optimizer, train_dl, device)
        scheduler.step()

        # b. evaluate performance on validation set

        # c. save checkpoints
        checkpoint = {
            "epoch": epoch+1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }

        torch.save(checkpoint, checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth")

        # d. display statistics
        print(f"(Epoch {epoch+1}) Training: Loss - {train_loss:.4f} "
              f"| Validation: Loss - {val_loss:.4f} Accuracy - {val_accuracy}")

        # e. display epoch duration
        end_time = time.perf_counter()
        print(f"Epoch took {(end_time - start_time):.4f} seconds to complete.")

if __name__ == "__main__":
    main()