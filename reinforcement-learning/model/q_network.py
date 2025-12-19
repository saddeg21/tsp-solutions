import math
import torch
import torch.nn as nn
from .encoder import Encoder

class QNetwork(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, d_ff, num_layers):
        super(QNetwork, self).__init__()
        self.encoder = Encoder(input_dim, d_model, num_heads, d_ff, num_layers)
        self.context_proj = nn.Linear(2*d_model, d_model)

    def forward(self, coordinates, mask, current_city):
        """
        coordinates: (B, N, 2)
        mask: (B, N) with 1 = unvisited, 0 = visited
        current_city: (B,)
        returns: q_values (B, N)
        """
        enc = self.encoder(coordinates, mask)  # (B, N, d_model)
        graph_emb = enc.mean(dim=1)  # (B, d_model)

        idx = current_city.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, enc.size(-1))  # (B, 1, d_model)
        current_city_emb = enc.gather(enc, 1, idx).squeeze(1) # (B, d_model)

        context = torch.cat([graph_emb, current_city_emb], dim=-1)  # (B, 2*d_model)
        query = self.context_proj(context).unsqueeze(1)  # (B, 1, d_model)

        scores = torch.matmul(query, enc.transpose(-2, -1)).squeeze(1)  # (B, N)
        scores = scores / math.sqrt(enc.size(-1))
        scores = scores.masked_fill(mask == 0, -1e9))  # Mask visited cities

        return scores  # (B, N)

    @torch.no_grad()
    def act(self, coordinates, mask, current_city, epsilon):
        """
        coordinates: (B, N, 2)
        mask: (B, N) with 1 = unvisited, 0 = visited
        current_city: (B,)
        epsilon: float
        returns: actions (B,)
        """
        q_values = self.forward(coordinates, mask, current_city)  # (B, N)
        batch_size, num_cities = q_values.size()

        actions = []

        for b in range(batch_size):
            available = torch.nonzero(mask[b]>0, as_tuple=False).squeeze(-1)

            if len(available) == 0:
                actions.append(torch.tensor(0, device=q_values.device))
                continue

            if torch.rand(1).item() < epsilon:
                idx = torch.randint(0, len(available), (1,), device=q_values.device)
                actions.append(available[idx])
            else:
                actions.append(torch.argmax(q_values[b,available])) 
            
        return torch.stack(actions)  # (B,)