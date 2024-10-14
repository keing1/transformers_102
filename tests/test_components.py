import unittest
import torch as t
import torch.nn.functional as F
import sys
import os

from src import activations, layers, blocks

class TestTransformerComponents(unittest.TestCase):
    def test_activation_functions(self):
        x = t.randn(100)
        x2 = x.reshape((10,10))
        assert t.allclose(activations.relu(x), F.relu(x))
        assert t.allclose(activations.relu(x2), F.relu(x2))

        assert t.allclose(activations.gelu(x), F.gelu(x))
        assert t.allclose(activations.gelu(x2), F.gelu(x2))

        assert t.allclose(activations.swish(x), F.silu(x))
        assert t.allclose(activations.swish(x2), F.silu(x2))

    def test_normalizers(self):
        x = t.randn(100)
        x2 = x.reshape((10,10))

        W1 = t.nn.Parameter(t.randn(100))
        b1 = t.nn.Parameter(t.randn(100))
        W2 = t.nn.Parameter(t.randn(10))
        b2 = t.nn.Parameter(t.randn(10))

        ln1 = layers.LayerNorm(x.shape[-1:])
        torch_ln1 = t.nn.LayerNorm(x.shape[-1:])
        ln2 = layers.LayerNorm(x2.shape[-1:])
        torch_ln2 = t.nn.LayerNorm(x2.shape[-1:])
        
        ln1.weight, ln1.bias = W1, b1
        torch_ln1.weight, torch_ln1.bias = W1, b1
        ln2.weight, ln2.bias = W2, b2
        torch_ln2.weight, torch_ln2.bias = W2, b2

        assert t.allclose(ln1(x), torch_ln1(x))
        assert t.allclose(ln2(x2), torch_ln2(x2))

        rn1 = layers.RMSNorm(x.shape[-1:])
        torch_rn1 = t.nn.RMSNorm(x.shape[-1:])
        rn2 = layers.RMSNorm(x2.shape[-1:])
        torch_rn2 = t.nn.RMSNorm(x2.shape[-1:])
        
        rn1.weight, rn1.bias = W1, b1
        torch_rn1.weight, torch_rn1.bias = W1, b1
        rn2.weight, rn2.bias = W2, b2
        torch_rn2.weight, torch_rn2.bias = W2, b2

        assert t.allclose(rn1(x), torch_rn1(x))
        assert t.allclose(rn2(x2), torch_rn2(x2))


    def test_embedding(self):
        pass

    def test_linear(self):
        pass

    def test_attention_block(self):
        pass

    def test_mlp_blocks(self):
        pass

    def test_transformer_block(self):
        pass

if __name__ == '__main__':
    unittest.main()