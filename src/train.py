from torch.optim.lr_scheduler import CosineAnnealingLR
from configs import SEED, BATCH_SIZE, WORKERS, device
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import CIFAR10
from transforms import train_transforms
from inference_functions import evaluate
from model import VisionTransformer
from train_functions import train
from pathlib import Path
import argparse
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
    train_dl = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=WORKERS, pin_memory=True, persistent_workers=True
    )
    val_dl = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=WORKERS,
        pin_memory=True, persistent_workers=True
    )

    start_epoch = 0
    num_epochs = 100

    model = VisionTransformer().to(device, non_blocking=True)
    loss_fn = torch.nn.CrossEntropyLoss().to(device, non_blocking=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, num_epochs, 5e-5)

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
        train_loss = train(model, loss_fn, optimizer, train_dl, device)
        scheduler.step()

        # b. evaluate performance on validation set
        val_accuracy, val_loss = evaluate(model, val_dl, device, loss_fn)

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
              f"| Validation: Loss - {val_loss:.4f} Accuracy - {val_accuracy:.4f}")

        # e. display epoch duration
        end_time = time.perf_counter()
        print(f"Epoch took {(end_time - start_time):.4f} seconds to complete.")

if __name__ == "__main__":
    main()