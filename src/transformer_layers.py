import torch.nn.functional as F
from configs import H
import torch.nn as nn
import torch

class MultiHeadSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
