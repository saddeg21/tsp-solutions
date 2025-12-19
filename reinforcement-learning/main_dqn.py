import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from env import TSPDataset
from model import QNetwork
from model.replay_buffer import ReplayBuffer, EpsilonScheduler
from train.dqn import train_epoch_dqn, build_greedy_tour

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    num_cities = 20
    d_model = 128
    num_heads = 8
    d_ff = 512
    num_layers = 3
    train_size = 20000
    batch_size = 64
    replay_capacity = 50000
    lr = 1e-4
    num_epochs = 50
    gamma = 0.99
    warmup = 2000
    target_update = 500

    # Data
    train_dataset = TSPDataset(num_sample=train_size, num_cities=num_cities)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Models
    policy_net = QNetwork(input_dim=2, d_model=d_model, num_heads=num_heads, d_ff=d_ff, num_layers=num_layers).to(device)
    target_net = QNetwork(input_dim=2, d_model=d_model, num_heads=num_heads, d_ff=d_ff, num_layers=num_layers).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(replay_capacity, device)
    eps_sched = EpsilonScheduler(eps_start=1.0, eps_end=0.05, decay_steps=50000)

    best_len = float("inf")

    for epoch in range(1, num_epochs + 1):
        loss = train_epoch_dqn(
            policy_net,
            target_net,
            optimizer,
            train_loader,
            replay_buffer,
            eps_sched,
            device,
            gamma=gamma,
            batch_size=batch_size,
            warmup=warmup,
            target_update=target_update,
        )

        # quick eval on a fresh batch
        val_coords = next(iter(train_loader))
        val_mean = build_greedy_tour(policy_net, val_coords, device)

        print(f"Epoch {epoch}: loss={loss:.4f} | greedy mean tour length={val_mean:.4f}")

        if val_mean < best_len:
            best_len = val_mean
            torch.save(policy_net.state_dict(), "best_dqn.pth")
            print("  ✓ New best model saved")

if __name__ == "__main__":
    main()
