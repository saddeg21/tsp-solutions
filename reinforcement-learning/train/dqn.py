import torch
import torch.nn as nn
from env import calculate_tour_length

def gather_city(coords, idx):
    return torch.gather(coords, 1, idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 2)).squeeze(1)

def step_reward(coords, current_city, next_city, last_step):
    curr = gather_city(coords, current_city)
    nxt = gather_city(coords, next_city)
    reward = -torch.norm(curr - nxt, dim=-1)
    if last_step:
        first = gather_city(coords, torch.zeros_like(next_city))
        reward = reward - torch.norm(nxt - first, dim=-1)
    return reward

def build_greedy_tour(q_net, coords, device):
    q_net.eval()
    with torch.no_grad():
        coords = coords.to(device)
        B, N, _ = coords.shape
        mask = torch.ones(B, N, device=device)
        current = torch.zeros(B, dtype=torch.long, device=device)
        mask.scatter_(1, current.unsqueeze(1), 0)
        tour = [current]

        for t in range(N-1):
            q_vals = q_net(coords, mask, current)
            actions = torch.argmax(q_vals, dim=-1)
            actions = actions.masked_fill(mask.gather(1, actions.unsqueeze(1)).squeeze(1) == 0, 0)
            mask = mask.scatter(1, actions.unsqueeze(1), 0)
            current = actions
            tour.append(actions)
        
        tour = torch.stack(tour, dim=-1)
        length = calculate_tour_length(tour, coords)
    return tour, length

def train_epoch_dqn(policy_net, target_net, optimizer, dataloader, replay_buffer, eps_scheduler, device, gamma=0.99, batch_size=64, warmup=2000, target_update=1000):
    policy_net.train()
    total_loss, updates = 0.0, 0

    for coords in dataloader:
        coords = coords.to(device)
        B, N, _ = coords.shape
        mask = torch.ones(B, N, device=device)
        current = torch.zeros(B, dtype=torch.long, device=device)
        mask.scatter_(1, current.unsqueeze(1), 0)

        for t in range(N-1):
            epsilon = eps_scheduler.epsilon
            actions = policy_net.act(coords, mask, current, epsilon)

            next_mask = mask.scatter(1, actions.unsqueeze(1), 0)
            last_step = (t == N - 2)
            rewards = step_reward(coords, current, actions, last_step)
            dones = torch.full((B,), last_step, dtype=torch.bool, device=device)

            for i in range(B):
                replay_buffer.push(
                    coords[i],
                    mask[i],
                    current[i],
                    actions[i],
                    rewards[i].item(),
                    next_mask[i],
                    actions[i],
                    dones[i].item(),
                )
            
            mask = next_mask
            current = actions
            eps_scheduler.step()

            if (len(replay_buffer) < warmup):
                (
                    c_b,
                    m_b,
                    curr_b,
                    a_b,
                    r_b,
                    next_m_b,
                    next_curr_b,
                    done_b,
                ) = replay_buffer.sample(batch_size)

                q_pred = policy_net(c_b, m_b, curr_b).gather(1, a_b.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    next_q = target_net(c_b, next_m_b, next_curr_b).max(dim=1).values
                    target = r_b + gamma * next_q * (~done_b)
                
                loss = nn.functional.smooth_l1_loss(q_pred, target)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                updates += 1

                if updates % target_update == 0:
                    target_net.load_state_dict(policy_net.state_dict())
        
    avg_loss = total_loss / updates if updates > 0 else 0.0
    return avg_loss