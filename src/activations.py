import torch as t
from scipy.stats import norm

def relu(x: t.Tensor) -> t.Tensor:
    return t.maximum(x, t.zeros(x.shape))

def gelu(x: t.Tensor) -> t.Tensor:
    return norm.cdf(x)

def swish(x: t.Tensor, beta: float=1) -> t.Tensor:
    return x / (1 + t.exp(-beta * x))