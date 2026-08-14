from configs import BATCH_SIZE, WORKERS, device
from inference_functions import evaluate
from torchvision.datasets import CIFAR10
from transforms import test_transforms
from model import VisionTransformer
from pathlib import Path
import torch

test_dataset = CIFAR10(
    root="../data",
    train=False,
    transform=test_transforms,
    download=True
)

test_dl = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=WORKERS,
                     shuffle=False, pin_memory=True, persistent_workers=True)

# load checkpoint
checkpoint_path = Path("../outputs/checkpoints/..")
checkpoint = torch.load(checkpoint_path)

# load trained model's parameters into new model
model = Vision_Transformer().load_state_dict(checkpoint["model_state_dict"])

test_accuracy = evaluate(model, test_dl, device)

print(f"Model Test Accuracy - {(test_accuracy * 100):.2f}%")