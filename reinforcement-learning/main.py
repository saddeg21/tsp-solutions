import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model import TSPModel
from env import TSPDataset, calculate_tour_length
from train import train_epoch

import os

def main():
    # Pick a device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    num_cities = 20
    d_model = 128
    num_heads = 8
    d_ff = 512
    num_layers = 3
    batch_size = 512
    num_epochs = 100
    lr = 1e-4
    
    # Data path
    data_dir = f"./data/tsp_{num_cities}"
    train_path = f"{data_dir}/train.pt"
    val_path = f"{data_dir}/val.pt"

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print(f"Dataset not found at {data_dir}")
        print("Please generate dataset first:")
        print(f"  python data/generate_dataset.py --num_cities {num_cities}")
        return
    
    # Create datasets and dataloaders
    train_dataset = TSPDataset(file_path=train_path)
    val_dataset = TSPDataset(file_path=val_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # model creation
    model = TSPModel(input_dim=2, d_model=d_model, num_heads=num_heads, d_ff=d_ff, num_layers=num_layers)
    model = model.to(device)

    # create optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_length = float('inf')

    # training loop
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, optimizer, train_loader, device)
        val_length = validate(model, val_loader, device)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Validation Tour Length: {val_length:.4f}")

        #checkpoint logic
        if val_length < best_val_length:
            best_val_length = val_length
            torch.save(model.state_dict(), "best_tsp_model_reinforce.pth")
            print(f"  New best model saved!")
        
        print()
    
    print("Training complete.")

    # Save the trained model
    #torch.save(model.state_dict(), "tsp_model.pth")
    
    

def validate(model, dataloader, device):
    model.eval()
    total_length = 0
    
    with torch.no_grad():
        for batch in dataloader:
            coordinates = batch.to(device)
            tour, _ = model(coordinates, greedy=True)
            tour_length = calculate_tour_length(tour, coordinates)
            total_length += tour_length.sum().item()
    
    return total_length / len(dataloader.dataset)

if __name__ == "__main__":
    main()