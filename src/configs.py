import torch

# use GPU instead of CPU, if possible
device = "cuda" if torch.cuda.is_available() else "cpu"

# patch embedding dimensions
D = 256

# image width and height
W = 32
H = 32

# image patch width/height
P = 4

# total number of image patch embeddings
N = int((W / P) * (H / P))

# number of images per batch for training/inference
BATCH_SIZE = 128

# dropout probability
dropout_P = 0.0

# number of heads
HEADS = 4

# number of transformer layers in the transformer encoder
LAYERS = 4

# number of classes in the classification dataset
CLASSES = 10

# seed for generators
SEED = 7

# CPU worker processes count
WORKERS = 4