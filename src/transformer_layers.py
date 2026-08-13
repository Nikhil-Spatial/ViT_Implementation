from src.configs import HEADS, D, N, dropout_P, LAYERS
import torch.nn.functional as F
import torch.nn as nn
import torch
import math

class SelfAttentionHead(nn.Module):
    def __init__(self):
        super().__init__()

        # dimensions of each token in the query, key, and value tensors
        self.D_qkv = D // HEADS

        # Q, K, and V transformations are purely linear
        self.Q_linear_transform = nn.Linear(D, self.D_qkv, bias=False)
        self.K_linear_transform = nn.Linear(D, self.D_qkv, bias=False)
        self.V_linear_transform = nn.Linear(D, self.D_qkv, bias=False)

    def forward(self, x):
        # query, key, and value tensors
        Q = self.Q_linear_transform(x)
        K = self.K_linear_transform(x)
        V = self.V_linear_transform(x)

        # compute attention scores:
        # softmax(matmul(Q, transpose(K)) / sqrt(D_qkv))
        attention_scores = F.softmax(
            (Q @ K.transpose(-2, -1)) / math.sqrt(self.D_qkv),
            dim=-1
        )

        # compute head output
        return attention_scores @ V

class MultiHeadSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()

        # list of self-attention heads
        self.self_attention_heads = nn.ModuleList(
            [SelfAttentionHead() for _ in range(HEADS)]
        )

        # apply linear transformation to vertically concatenated head outputs
        self.output_linear_transform = nn.Linear(D, D, bias=False)

    def forward(self, x):
        # compute head outputs
        head_outputs = [self_attention_head(x) for self_attention_head
                        in self.self_attention_heads]

        # vertically concatenate the head outputs
        output = torch.cat(head_outputs, dim=-1)

        # apply linear transformation
        return self.output_linear_transform(output)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear_transform_1 = nn.Linear(D, D)
        self.dropout_1 = nn.Dropout(dropout_P)

        self.linear_transform_2 = nn.Linear(D, D)
        self.dropout_2 = nn.Dropout(dropout_P)

    def forward(self, x):
        # affine transformation -> dropout -> GeLU activation
        # -> affine transformation -> dropout
        x = F.gelu(self.dropout_1(self.linear_transform_1(x)))

        return self.dropout_2(self.linear_transform_2(x))

class TransformerLayer(nn.Module):
    def __init__(self):
        super().__init__()

        # layer normalization layers
        self.layer_norm_1 = nn.LayerNorm(D)
        self.layer_norm_2 = nn.LayerNorm(D)

        # multi-head self-attention layer
        self.multi_head_self_attention = MultiHeadSelfAttention()

        # multi-layered perceptron (MLP)
        self.mlp = MLP()

    def forward(self, x):
        # keep original input for residual connection
        pre_norm_x = x

        # layer normalize input -> multi-head self-attention residual connection
        x = self.layer_norm_1(x)
        x = self.multi_head_self_attention(x) + pre_norm_x

        # layer normalize input -> mlp residual connection
        pre_norm_x = x

        x = self.layer_norm_2(x)
        return self.mlp(x) + pre_norm_x

class TransformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # module list of transformer layers
        self.transformer_layers = nn.ModuleList(
            [TransformerLayer() for _ in range(LAYERS)]
        )
    def forward(self, x):
        # feed patch embeddings to the transformer layers
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x)

        return x