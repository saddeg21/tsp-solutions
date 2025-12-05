"""
TSP Model Test Script

Bu script eğitilmiş modeli test eder ve sonuçları görselleştirir.
"""

import torch
from torch.utils.data import DataLoader

from model import TSPModel
from env import TSPDataset, calculate_tour_length
from utils.visualization import (
    plot_tour, 
    plot_multiple_tours, 
    plot_comparison,
    plot_metrics_dashboard,
    plot_length_vs_index,
    calculate_metrics,
    print_metrics_table
)


def load_model(model_path, device, **model_kwargs):
    """Kaydedilmiş modeli yükler."""
    model = TSPModel(**model_kwargs).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model yüklendi: {model_path}")
    return model


def test_model(model, test_loader, device):
    model.eval()

    all_tours = []
    all_coords = []
    all_lengths = []
    
    with torch.no_grad():
        for batch in test_loader:
            coordinates = batch.to(device)
            
            #Greedy gives best prediction
            tour, _ = model(coordinates, greedy=True)
            tour_length = calculate_tour_length(tour, coordinates)
            
            all_tours.extend(tour.cpu())
            all_coords.extend(coordinates.cpu())
            all_lengths.extend(tour_length.cpu().tolist())
    
    avg_length = sum(all_lengths) / len(all_lengths)
    
    return avg_length, all_tours, all_coords, all_lengths


def evaluate_and_visualize(model, test_loader, device, num_visualize=6):
    """
    Modeli değerlendirir ve örnek sonuçları görselleştirir.
    """
    print("=" * 60)
    print(" TSP Model Değerlendirmesi")
    print("=" * 60)
    
    avg_length, all_tours, all_coords, all_lengths = test_model(model, test_loader, device)
    
    # Detaylı metrikler
    metrics = calculate_metrics(all_lengths)
    print_metrics_table(metrics, title="Test Set Performance Metrics")
    
    # En iyi ve en kötü sonuçları bul
    sorted_indices = sorted(range(len(all_lengths)), key=lambda i: all_lengths[i])
    
    # Görselleştirme için örnekler seç
    best_indices = sorted_indices[:3]
    worst_indices = sorted_indices[-3:]
    
    print(f"\n--- En İyi 3 Çözüm ---")
    for i, idx in enumerate(best_indices):
        print(f"  {i+1}. Tur uzunluğu: {all_lengths[idx]:.4f}")
    
    print(f"\n--- En Kötü 3 Çözüm ---")
    for i, idx in enumerate(worst_indices):
        print(f"  {i+1}. Tur uzunluğu: {all_lengths[idx]:.4f}")
    
    # Görselleştirme
    print(f"\n--- Görselleştirmeler ---")
    
    # 1. Metrics Dashboard
    print("\n1. Metrics Dashboard oluşturuluyor...")
    plot_metrics_dashboard(all_lengths, save_path="metrics_dashboard.png")
    
    # 2. Tour Length vs Index
    print("\n2. Length vs Index grafiği oluşturuluyor...")
    plot_length_vs_index(all_lengths, save_path="length_vs_index.png")
    
    # 3. En iyi çözümler
    best_coords = [all_coords[i] for i in best_indices]
    best_tours = [all_tours[i] for i in best_indices]
    best_titles = [f"En İyi {i+1}\nUzunluk: {all_lengths[idx]:.4f}" 
                   for i, idx in enumerate(best_indices)]
    
    print("\n3. En iyi 3 çözüm görselleştiriliyor...")
    plot_multiple_tours(best_coords, best_tours, best_titles, save_path="best_solutions.png")
    
    # 4. En kötü çözümler
    worst_coords = [all_coords[i] for i in worst_indices]
    worst_tours = [all_tours[i] for i in worst_indices]
    worst_titles = [f"En Kötü {i+1}\nUzunluk: {all_lengths[idx]:.4f}" 
                    for i, idx in enumerate(worst_indices)]
    
    print("\n4. En kötü 3 çözüm görselleştiriliyor...")
    plot_multiple_tours(worst_coords, worst_tours, worst_titles, save_path="worst_solutions.png")
    
    return metrics


def run_single_example(model, num_cities, device):
    """
    Tek bir örnek üzerinde modeli çalıştırır ve görselleştirir.
    """
    # Rastgele koordinatlar oluştur
    coordinates = torch.rand(1, num_cities, 2).to(device)
    
    # Model ile çöz
    model.eval()
    with torch.no_grad():
        tour, _ = model(coordinates, greedy=True)
    
    # Görselleştir
    plot_tour(
        coordinates[0], 
        tour[0], 
        title=f"TSP Çözümü ({num_cities} şehir)",
        save_path="single_example.png"
    )
    
    # Tur uzunluğunu hesapla
    tour_length = calculate_tour_length(tour, coordinates)
    print(f"Tur uzunluğu: {tour_length.item():.4f}")


def main():
    # Device seçimi
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Hyperparameters (main.py ile aynı olmalı)
    num_cities = 20
    d_model = 128
    num_heads = 8
    d_ff = 512
    num_layers = 3
    
    # Test dataset
    data_dir = f"./data/tsp_{num_cities}"
    test_path = f"{data_dir}/test.pt"
    model_path = "best_tsp_model_reinforce.pth"

    if not os.path.exists(test_path):
        print(f"Test dataset not found at {test_path}")
        print("Please generate dataset first:")
        print(f"  python data/generate_dataset.py --num_cities {num_cities}")
        return
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please train the model first:")
        print(f"  python main.py")
        return
        
    test_dataset = TSPDataset(file_path=test_path)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Test dataset loaded: {len(test_dataset)} samples")
    print(f"Batch size: {batch_size}\n")
    
    # Model oluştur
    model = TSPModel(
        input_dim=2,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded from {model_path}\n")
    
    rint("Testing model...")
    test_lengths, test_tours, test_coords = test_model(model, test_loader, device)
    
    # Compute metrics
    metrics = compute_metrics(test_lengths)
    
    print("\n" + "="*50)
    print("TEST RESULTS")
    print("="*50)
    print(f"Mean Tour Length:   {metrics['mean']:.4f}")
    print(f"Std Dev:            {metrics['std']:.4f}")
    print(f"Median:             {metrics['median']:.4f}")
    print(f"Min:                {metrics['min']:.4f}")
    print(f"Max:                {metrics['max']:.4f}")
    print(f"Q1 (25%):           {metrics['q1']:.4f}")
    print(f"Q3 (75%):           {metrics['q3']:.4f}")
    print(f"IQR:                {metrics['iqr']:.4f}")
    print("="*50 + "\n")
    
    # Plot best solutions
    print("Generating visualizations...")
    
    # En iyi 3 çözüm
    best_indices = torch.argsort(test_lengths)[:3]
    best_lengths = test_lengths[best_indices]
    best_tours = test_tours[best_indices]
    best_coords = test_coords[best_indices]
    
    plot_multiple_tours(
        best_coords.numpy(),
        best_tours.numpy(),
        best_lengths.numpy(),
        title="Best 3 Solutions",
        save_path="best_solutions.png"
    )
    
    # En kötü 3 çözüm
    worst_indices = torch.argsort(test_lengths)[-3:]
    worst_lengths = test_lengths[worst_indices]
    worst_tours = test_tours[worst_indices]
    worst_coords = test_coords[worst_indices]
    
    plot_multiple_tours(
        worst_coords.numpy(),
        worst_tours.numpy(),
        worst_lengths.numpy(),
        title="Worst 3 Solutions",
        save_path="worst_solutions.png"
    )
    
    # Single example
    example_idx = 0
    plot_tour(
        test_coords[example_idx].numpy(),
        test_tours[example_idx].numpy(),
        length=test_lengths[example_idx].item(),
        title=f"Example Solution (Tour Length: {test_lengths[example_idx]:.4f})",
        save_path="single_example.png"
    )
    
    print("Visualizations saved:")
    print("  - best_solutions.png")
    print("  - worst_solutions.png")
    print("  - single_example.png")


if __name__ == "__main__":
    main()
