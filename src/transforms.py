from torchvision.transforms import v2
import torch

CIFAR_MEAN = (0.49145, 0.48219, 0.44658)
CIFAR_STD = (0.24687, 0.24332, 0.26137)

train_transforms = v2.Compose([
    v2.RandomCrop(size=32, padding=4),
    v2.RandomRotation(degrees=45),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ColorJitter(
        brightness=[0.6, 1.4],
        contrast=[0.6, 1.4],
        saturation=[0.6, 1.4]
    ),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD)
])

test_transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD)
])