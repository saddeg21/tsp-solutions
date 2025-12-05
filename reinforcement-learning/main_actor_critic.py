import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model import TSPModel, CriticModel
from env import TSPDataset, calculate_tour_length
from train import train_epoch_actor_critic

def validate(model, dataloader, device):
    model.eval()
    total_length = 0
    
    with torch.no_grad():
        for batch in dataloader:
            coordinates = batch.to(device)
            tour, _, _ = model.forward_with_cache(coordinates, greedy=True)
            tour_length = calculate_tour_length(tour, coordinates)
            total_length += tour_length.sum().item()
    
    return total_length / len(dataloader.dataset)

def main():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    num_cities = 20
    d_model = 128
    num_heads = 8
    d_ff = 512
    num_layers = 3
    batch_size = 512
    num_epochs = 100
    actor_lr = 1e-4
    critic_lr = 1e-3  # Critic genelde daha yüksek lr
    train_size = 100000
    val_size = 10000

    # Datasets
    train_dataset = TSPDataset(train_size, num_cities)
    val_dataset = TSPDataset(val_size, num_cities)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Models
    actor = TSPModel(
        input_dim=2,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers
    ).to(device)

    critic = Critic(d_model=d_model, hidden_dim=256).to(device)

    # Optimizers
    actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)

    # Training
    best_val_length = float('inf')
    
    for epoch in range(num_epochs):
        actor_loss, critic_loss = train_epoch_actor_critic(
            actor, critic, actor_optimizer, critic_optimizer, train_loader, device
        )
        
        val_length = validate(actor, val_loader, device)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}")
        print(f"  Validation Tour Length: {val_length:.4f}")
        
        # Save best model
        if val_length < best_val_length:
            best_val_length = val_length
            torch.save({
                'actor': actor.state_dict(),
                'critic': critic.state_dict(),
            }, "best_actor_critic_model.pth")
            print(f"  ✓ New best model saved!")
        
        print()

    print(f"Training complete! Best validation tour length: {best_val_length:.4f}")


if __name__ == "__main__":
    main()