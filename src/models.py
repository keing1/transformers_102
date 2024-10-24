import torch as t

from src import layers, blocks

class GPT2Embedding(t.nn.Module):
    def __init__(self, num_token_embeddings: int, embedding_dim: int, context_length: int):
        super().__init__()
        self.num_token_embeddings = num_token_embeddings
        self.embedding_dim = embedding_dim
        self.context_length = context_length
        self.position_embedding_layer = layers.Embedding(self.context_length, self.embedding_dim)
        self.token_embedding_layer = layers.Embedding(self.num_token_embeddings, self.embedding_dim)
    
    def forward(self, x: t.Tensor) -> t.Tensor:
        pos_tensor = t.arange(x.shape[-1])
        x_pos = self.position_embedding_layer(pos_tensor)
        x_tok = self.token_embedding_layer(x)
        return x_tok + x_pos

class GPT2SmallModel(t.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_tokens = 50257
        self.hidden_size = 768
        self.num_layers = 12
        self.num_heads = 12
        self.activation_function = 'gelu'
        self.context_length = 1024
        self.dropout_rate = 0.1

        self.init_dropout_layer = layers.Dropout(self.dropout_rate)
        self.embedding_layer = GPT2Embedding(self.num_tokens, self.hidden_size, self.context_length)
        self.transformer_blocks = t.nn.ModuleList([blocks.TransformerDecoderBlock(self.hidden_size, self.num_heads, 4*self.hidden_size, 'mlpblock', self.activation_function, 'layer_norm', use_pre_norm=True, dropout_rate=self.dropout_rate) for _ in range(self.num_layers)])
        self.final_ln = layers.LayerNorm((self.hidden_size,))
    
    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.embedding_layer(x)
        x = self.init_dropout_layer(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.final_ln(x)

        # Tied unembedding
        x = t.einsum('th, ...h -> ...t', self.embedding_layer.token_embedding_layer.weight, x)

        return x

class Llama7BModel(t.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_tokens = 32000
        self.hidden_size = 4096
        self.num_layers = 32
        self.num_heads = 32
        self.activation_function = 'swish'
        self.context_length = 2048

        self.embedding_layer = layers.Embedding(self.num_tokens, self.hidden_size)
        self.transformer_blocks = t.nn.ModuleList([blocks.TransformerDecoderBlock(self.hidden_size, self.num_heads, 8*self.hidden_size//3, 'glublock', self.activation_function, 'rms_norm', use_pre_norm=True, rotary_embedding=True) for _ in range(self.num_layers)])
        self.final_rn = layers.RMSNorm((self.hidden_size,))
        self.unembedding_layer = layers.Linear(4096, 32000, includes_bias=False)
        
    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.embedding_layer(x)
        for block in self.transformer_blocks:
            x = block(x)
        
        x = self.final_rn(x)
        x = self.unembedding_layer(x)
        return x