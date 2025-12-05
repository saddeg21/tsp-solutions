import torch
import os
import argparse

def generate_tsp_dataset(num_samples, num_cities, seed=42):
    torch.manual_seed(seed)
    coordinates = torch.rand(num_samples, num_cities, 2)
    return coordinates

def save_dataset(coordinates, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(coordinates, save_path)
    print(f"Dataset saved to {save_path}")
    print(f"Dataset shape: {coordinates.shape}")

def load_dataset(load_path):
    coordinates = torch.load(load_path)
    print(f"Dataset loaded from {load_path}")
    print(f"Dataset shape: {coordinates.shape}")
    return coordinates

def main():
    parser = argparse.ArgumentParser(description="Generate or Load TSP Dataset")
    parser.add_argument("--num_cities", type=int, default=20, help="Number of cities")
    parser.add_argument("--train_size", type=int, default=70000, help="Training set size")
    parser.add_argument("--val_size", type=int, default=10000, help="Validation set size")
    parser.add_argument("--test_size", type=int, default=20000, help="Test set size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_dir", type=str, default="./data/", help="Directory to save datasets")

    args = parser.parse_args()

    print(f"Generating TSP{args.num_cities} datasets...")
    print(f"  Train size: {args.train_size}")
    print(f"  Val size: {args.val_size}")
    print(f"  Test size: {args.test_size}")
    print(f"  Seed: {args.seed}")
    print()

    train_data = generate_tsp_dataset(args.train_size, args.num_cities, args.seed)
    save_dataset(train_data, os.path.join(args.save_dir, f"tsp{args.num_cities}_train.pt"))

    val_data = generate_tsp_dataset(args.val_size, args.num_cities, args.seed)
    save_dataset(val_data, os.path.join(args.save_dir, f"tsp{args.num_cities}_val.pt"))

    test_data = generate_tsp_dataset(args.test_size, args.num_cities, args.seed)
    save_dataset(test_data, os.path.join(args.save_dir, f"tsp{args.num_cities}_test.pt"))

    print("Generation complete.")

if __name__ == "__main__":
    main()