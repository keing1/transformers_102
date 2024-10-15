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

        assert t.allclose(ln1.weight, torch_ln1.weight)
        assert t.allclose(ln2.weight, torch_ln2.weight)
        
        ln1.weight, ln1.bias = W1, b1
        torch_ln1.weight, torch_ln1.bias = W1, b1
        ln2.weight, ln2.bias = W2, b2
        torch_ln2.weight, torch_ln2.bias = W2, b2

        assert t.allclose(ln1(x), torch_ln1(x), atol=1e-7)
        assert t.allclose(ln2(x2), torch_ln2(x2), atol=1e-7)

        rn1 = layers.RMSNorm(x.shape[-1:])
        torch_rn1 = t.nn.RMSNorm(x.shape[-1:])
        rn2 = layers.RMSNorm(x2.shape[-1:])
        torch_rn2 = t.nn.RMSNorm(x2.shape[-1:])

        assert t.allclose(rn1.weight, torch_rn1.weight)
        assert t.allclose(rn2.weight, torch_rn2.weight)
        
        rn1.weight, rn1.bias = W1, b1
        torch_rn1.weight, torch_rn1.bias = W1, b1
        rn2.weight, rn2.bias = W2, b2
        torch_rn2.weight, torch_rn2.bias = W2, b2

        assert t.allclose(rn1(x), torch_rn1(x))
        assert t.allclose(rn2(x2), torch_rn2(x2))


    def test_embedding(self):
        num_embeddings = 10
        embedding_dim = 5

        W = t.nn.Parameter(t.randn((num_embeddings, embedding_dim)))

        emb = layers.Embedding(num_embeddings, embedding_dim)
        torch_emb = t.nn.Embedding(num_embeddings, embedding_dim)

        emb.weight = W
        torch_emb.weight = W

        tokens = t.tensor([8,6,2,9])
        tokens2 = t.tensor([[5,9], [7,2]])

        assert t.allclose(emb(tokens), torch_emb(tokens))
        assert t.allclose(emb(tokens2), torch_emb(tokens2))


    def test_linear(self):
        in_dim = 5
        out_dim = 10
        W = t.nn.Parameter(t.randn((out_dim, in_dim)))
        b = t.nn.Parameter(t.randn((out_dim,)))

        lin = layers.Linear(in_dim, out_dim)
        torch_lin = t.nn.Linear(in_dim, out_dim)

        lin.weight, lin.bias = W, b
        torch_lin.weight, torch_lin.bias = W, b

        x = t.arange(5).float()
        x2 = t.arange(20).reshape((4,5)).float()

        assert t.allclose(lin(x), torch_lin(x), atol=1e-7)
        assert t.allclose(lin(x2), torch_lin(x2), atol=1e-7)

    def test_attention_block(self):
        pass

    def test_mlp_blocks(self):
        pass

    def test_transformer_block(self):
        pass

if __name__ == '__main__':
    unittest.main()