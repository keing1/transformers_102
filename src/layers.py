import torch as t
from torch import nn
from typing import Tuple

class LayerNorm(nn.Module):
    def __init__(self, data_shape: Tuple[int, ...], eps: float=1e-5, includes_bias: bool=True):
        super().__init__()
        self.data_shape = data_shape
        self.eps = eps
        self.includes_bias = includes_bias

        self.weight = nn.Parameter(t.ones(self.data_shape))
        if self.includes_bias:
            self.bias = nn.Parameter(t.zeros(self.data_shape))
    
    def forward(self, x: t.Tensor) -> t.Tensor:
        # Calcualte mean and variance over all dimensions but the first
        agg_dims = tuple(range(1, len(x.shape)))
        x_mean = t.mean(x, dim=agg_dims, keepdim=True)
        x_var = t.var(x, dim=agg_dims, keepdim=True, correction=0)

        normed_x = (x - x_mean)/(t.sqrt(x_var+self.eps))

        if self.includes_bias:
            out_x = normed_x * self.weight + self.bias
        else:
            out_x = normed_x * self.weight
        return out_x

class RMSNorm(nn.Module):
    def __init__(self, data_shape: Tuple[int, ...], eps: float=0):
        super().__init__()
        self.data_shape = data_shape
        self.eps = eps

        self.weight = nn.Parameter(t.ones(self.data_shape))
    
    def forward(self, x: t.Tensor) -> t.Tensor:
        # Calculate mean and variance over all dimensions but the first
        agg_dims = tuple(range(1, len(x.shape)))

        x_ms = t.mean(x ** 2, dim=agg_dims, keepdim=True)
        normed_x = x/t.sqrt(x_ms+self.eps)
        return normed_x * self.weight
    

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Using default pytorch initialization
        self.weight = nn.Parameter(t.randn(num_embeddings, embedding_dim))

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.weight[x]

class Linear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, includes_bias: bool=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.includes_bias = includes_bias

        # Using default pytorch initialization
        bound_val = 1/self.in_dim**0.5
        self.weight = nn.Parameter(t.rand(self.out_dim, self.in_dim) * 2*bound_val - bound_val)
        if self.includes_bias:
            self.bias = nn.Parameter(t.rand(self.out_dim) * 2*bound_val - bound_val)
    
    def forward(self, x: t.Tensor) -> t.Tensor:
        x = t.einsum('oi,...i-> ...o', self.weight, x)
        if self.includes_bias:
            x += self.bias
        return x
        