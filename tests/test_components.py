import unittest
import torch as t
import torch.nn.functional as F
import sys
import os

from src import activations

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

        W1
        W2
        b1
        b2

        pass

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