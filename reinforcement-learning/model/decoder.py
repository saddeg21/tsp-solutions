import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import MultiHeadAttention

class Decoder(nn.Module):
    def __init__(self, d_model, num_heads):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        # we used 3*d_model beccause grap_embedding + first city embedding + last city embedding
        self.context_projection = nn.Linear(3*d_model, d_model)
        self.attention = MultiHeadAttention(num_heads, d_model)
        self.project_output = nn.Linear(d_model, d_model)
    
    def forward(self, encoder_output, mask, first_city_idx, current_city_idx):
        # first, get city embeddings mean
        graph_embedding = encoder_output.mean(dim=1)  # (batch_size, d_model)

        # get first city embedding and last city embedding
        idx = first_city_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.d_model)
        first_embedding = torch.gather(encoder_output, 1, idx).squeeze(1)  # (batch_size, d_model)

        idx = current_city_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.d_model)
        current_embedding = torch.gather(encoder_output, 1, idx).squeeze(1)  # (batch_size, d_model)

        # context oluştur
        context = torch.cat([graph_embedding, first_embedding, current_embedding], dim=-1)  # (batch_size, 3*d_model)
        context = self.context_projection(context)  # (batch_size, d_model)

        # use context as query, encoder_output as key and value
        context = context.unsqueeze(1)  # (batch_size, 1, d_model
        
        #prepare mask for attention
        attention_mask = mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)

        attn_output, _ = self.attention(context, encoder_output, encoder_output, attention_mask)  # (batch_size, 1, d_model)
        
        # find probs
        query = self.project_output(attn_output)  # (batch_size, 1, d_model)
        scores = torch.matmul(query, encoder_output.transpose(-2, -1)).squeeze(1)  # (batch_size, seq_len)
        scores = scores / (self.d_model ** 0.5) # scaling

        scores = scores.masked_fill(mask == 0, -1e9)  # apply mask
        probs = F.softmax(scores, dim=-1)  # (batch_size, seq_len

        return probs