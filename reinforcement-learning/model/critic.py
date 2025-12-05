import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, d_model, hidden_dim=256):
        super(Critic, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, encoder_output):
        graph_embedding = encoder_output.mean(dim=1)  # (batch_size, d_model)
        
        value = self.model(graph_embedding).squeeze(-1)  # (batch_size,)

        return value
