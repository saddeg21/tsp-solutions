import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import MultiHeadAttention

class EncoderLayer(nn.Module):
    """
    @d_ff: usually 4*d_model
    """
    def __init__(self, d_model, num_heads, d_ff):
        super(EncoderLayer, self).__init__()
        
        self.attention = MultiHeadAttention(num_heads, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
    
    def forward(self, x, mask=None):
        attn_output, _ = self.attention(x, x, x, mask) # Because Q=K=V=x
        x = self.norm1(x + attn_output)

        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)

        return x

class Encoder(nn.Module):
    # for our example input_dim=2
    # d_model=128, num_heads=8, d_ff=512, num_layers=3
    def __init__(self, input_dim, d_model, num_heads, d_ff, num_layers):
        super(Encoder, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])
    
    def forward(self, x, mask=None):
        # x => (batch_size, seq_len, 2)
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        for layer in self.layers:
            x = layer(x, mask)
        return x  # (batch_size, seq_len, d_model)