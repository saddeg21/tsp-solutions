import torch
import torch.nn as nn
import torch.optim as optim
from env import calculate_tour_length

def train_epoch_actor_critic(actor, critic, actor_optimizer, critic_optimizer, dataloader, device):
    
    actor.train()
    critic.train()
    
    total_actor_loss = 0
    total_critic_loss = 0

    for batch in dataloader:
        coordinates = batch.to(device)  # (batch_size, num_cities, 2)

        tour, log_probs, encoder_output = actor.forward_with_cache(coordinates, greedy=False)  # (batch_size, num_cities), (batch_size, num_cities-1)

        value = critic(encoder_output)  # (batch_size,)

        tour_length = calculate_tour_length(tour, coordinates)  # (batch_size,)

        advantage = (tour_length - value).detach()  # (batch_size,)

        actor_loss = (advantage * log_probs.sum(dim=1)).mean()  # scalar

        critic_loss = nn.MSELoss()(value, tour_length.detach())  # scalar

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        total_actor_loss += actor_loss.item()
        total_critic_loss += critic_loss.item()
    
    return total_actor_loss / len(dataloader), total_critic_loss / len(dataloader)