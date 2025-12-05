from .encoder import Encoder
from .decoder import Decoder

import torch.nn as nn
import torch

class TSPModel(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, d_ff, num_layers):     
        super(TSPModel, self).__init__()
        self.encoder = Encoder(input_dim, d_model, num_heads, d_ff, num_layers)
        self.decoder = Decoder(d_model, num_heads)
    
    def forward(self, coordinates, greedy=False):
        encoder_output = self.encoder(coordinates)
        batch_size, seq_len, _ = encoder_output.size()

        mask = torch.ones(batch_size, seq_len).to(coordinates.device)  # Initially, all cities are unvisited

        tour = []
        log_probs = []

        # for first city, use 0 for simplicity
        first_city = torch.zeros(batch_size, dtype=torch.long).to(coordinates.device)
        current_city = first_city.clone()

        tour.append(first_city)

        mask = mask.scatter(1, first_city.unsqueeze(1), 0)  # Mark first city as visited

        for _ in range(seq_len - 1): # first city already visited
            probs = self.decoder(encoder_output, mask, first_city, current_city)  # (batch_size, seq_len)

            # Select a city
            if greedy:
                # pick the most probable city
                selected_city = torch.argmax(probs, dim=-1)  # (batch_size,)
            else:
                # sampling
                selected_city = torch.multinomial(probs, num_samples=1).squeeze(1)  # (batch_size,)

            # log probs for rl
            log_prob = torch.log(probs.gather(1, selected_city.unsqueeze(1)).squeeze(1))
            log_probs.append(log_prob)

            # add tour
            tour.append(selected_city)

            mask = mask.scatter(1, selected_city.unsqueeze(1), 0)  # Mark selected city as visited

            current_city = selected_city
        
        tour = torch.stack(tour, dim=1)  # (batch_size, seq_len)
        log_probs = torch.stack(log_probs, dim=1)  # (batch_size,

        return tour, log_probs