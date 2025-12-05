import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from env import calculate_tour_length

def train_epoch(model, optimizer, dataloader, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        coordinates = batch.to(device)  # (batch_size, num_cities, 2)

        # With sampling, create a tour
        tour, log_probs = model(coordinates, greedy=False)  # (batch_size, num_cities), (batch_size, num_cities-1)

        # with greed, create baseline tour
        with torch.no_grad():
            baseline_tour, _ = model(coordinates, greedy=True)  # (batch_size, num_cities), (batch_size, num_cities-1)

        # calculate tour lengths
        tour_length = calculate_tour_length(tour, coordinates)  # (batch_size,)
        baseline_length = calculate_tour_length(baseline_tour, coordinates)  # (batch_size,)

        # advantage calculation
        advantage = (tour_length - baseline_length).detach()  # (batch_size,)

        # Reinforce loss
        loss = (advantage * log_probs.sum(dim=1)).mean()  # scalar

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)