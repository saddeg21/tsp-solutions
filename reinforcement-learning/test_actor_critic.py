"""
TSP Actor-Critic Model Test Script

Bu script eğitilmiş actor-critic modelini test eder ve sonuçları görselleştirir.
"""

import os
import torch
from torch.utils.data import DataLoader

from model import TSPModel, Critic
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


def load_models(model_path, device, **model_kwargs):
    """Kaydedilmiş actor ve critic modellerini yükler."""
    actor = TSPModel(**model_kwargs).to(device)
    critic = Critic(d_model=model_kwargs['d_model'], hidden_dim=256).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    actor.load_state_dict(checkpoint['actor'])
    critic.load_state_dict(checkpoint['critic'])
    
    actor.eval()
    critic.eval()
    
    print(f"Actor-Critic modelleri yüklendi: {model_path}")
    return actor, critic


def test_model(actor, critic, test_loader, device):
    """Actor-critic modelini test eder."""
    actor.eval()
    critic.eval()

    all_tours = []
    all_coords = []
    all_lengths = []
    all_values = []
    
    with torch.no_grad():
        for batch in test_loader:
            coordinates = batch.to(device)
            
            # Greedy ile en iyi tahmini al
            tour, encoder_output, _ = actor.forward_with_cache(coordinates, greedy=True)
            tour_length = calculate_tour_length(tour, coordinates)
            
            # Critic'in değer tahminini al
            value = critic(encoder_output)
            
            all_tours.extend(tour.cpu())
            all_coords.extend(coordinates.cpu())
            all_lengths.extend(tour_length.cpu().tolist())
            all_values.extend(value.cpu().tolist())
    
    avg_length = sum(all_lengths) / len(all_lengths)
    avg_value = sum(all_values) / len(all_values)
    
    return avg_length, avg_value, all_tours, all_coords, all_lengths, all_values


def evaluate_and_visualize(actor, critic, test_loader, device):
    """
    Actor-Critic modelini değerlendirir ve örnek sonuçları görselleştirir.
    """
    print("=" * 60)
    print(" TSP Actor-Critic Model Değerlendirmesi")
    print("=" * 60)
    
    avg_length, avg_value, all_tours, all_coords, all_lengths, all_values = test_model(
        actor, critic, test_loader, device
    )
    
    # Detaylı metrikler
    metrics = calculate_metrics(all_lengths)
    print_metrics_table(metrics, title="Test Set Performance Metrics")
    
    print(f"\nOrtalama Critic Value Tahmini: {avg_value:.4f}")
    print(f"Ortalama Tur Uzunluğu: {avg_length:.4f}")
    
    # Value tahminleri ile gerçek tur uzunlukları arasındaki korelasyon
    import numpy as np
    correlation = np.corrcoef(all_values, all_lengths)[0, 1]
    print(f"Value-Length Korelasyonu: {correlation:.4f}")
    
    # En iyi ve en kötü sonuçları bul
    sorted_indices = sorted(range(len(all_lengths)), key=lambda i: all_lengths[i])
    
    # Görselleştirme için örnekler seç
    best_indices = sorted_indices[:3]
    worst_indices = sorted_indices[-3:]
    
    print(f"\n--- En İyi 3 Çözüm ---")
    for i, idx in enumerate(best_indices):
        print(f"  {i+1}. Tur uzunluğu: {all_lengths[idx]:.4f}, Value: {all_values[idx]:.4f}")
    
    print(f"\n--- En Kötü 3 Çözüm ---")
    for i, idx in enumerate(worst_indices):
        print(f"  {i+1}. Tur uzunluğu: {all_lengths[idx]:.4f}, Value: {all_values[idx]:.4f}")
    
    # Görselleştirme
    print(f"\n--- Görselleştirmeler ---")
    
    # 1. Metrics Dashboard
    print("\n1. Metrics Dashboard oluşturuluyor...")
    plot_metrics_dashboard(all_lengths, save_path="ac_metrics_dashboard.png")
    
    # 2. Tour Length vs Index
    print("\n2. Length vs Index grafiği oluşturuluyor...")
    plot_length_vs_index(all_lengths, save_path="ac_length_vs_index.png")
    
    # 3. Value vs Length Scatter Plot
    print("\n3. Value vs Length scatter plot oluşturuluyor...")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.scatter(all_values, all_lengths, alpha=0.5)
    plt.xlabel('Critic Value Prediction')
    plt.ylabel('Actual Tour Length')
    plt.title(f'Value Predictions vs Actual Lengths (Correlation: {correlation:.4f})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ac_value_vs_length.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   Kaydedildi: ac_value_vs_length.png")
    
    # 4. En iyi çözümler
    best_coords = [all_coords[i] for i in best_indices]
    best_tours = [all_tours[i] for i in best_indices]
    best_titles = [f"En İyi {i+1}\nUzunluk: {all_lengths[idx]:.4f}\nValue: {all_values[idx]:.4f}" 
                   for i, idx in enumerate(best_indices)]
    
    print("\n4. En iyi 3 çözüm görselleştiriliyor...")
    plot_multiple_tours(best_coords, best_tours, best_titles, save_path="ac_best_solutions.png")
    
    # 5. En kötü çözümler
    worst_coords = [all_coords[i] for i in worst_indices]
    worst_tours = [all_tours[i] for i in worst_indices]
    worst_titles = [f"En Kötü {i+1}\nUzunluk: {all_lengths[idx]:.4f}\nValue: {all_values[idx]:.4f}" 
                    for i, idx in enumerate(worst_indices)]
    
    print("\n5. En kötü 3 çözüm görselleştiriliyor...")
    plot_multiple_tours(worst_coords, worst_tours, worst_titles, save_path="ac_worst_solutions.png")
    
    return metrics, all_values, all_lengths


def run_single_example(actor, critic, num_cities, device):
    """
    Tek bir örnek üzerinde modeli çalıştırır ve görselleştirir.
    """
    # Rastgele koordinatlar oluştur
    coordinates = torch.rand(1, num_cities, 2).to(device)
    
    # Model ile çöz
    actor.eval()
    critic.eval()
    
    with torch.no_grad():
        tour, encoder_output, _ = actor.forward_with_cache(coordinates, greedy=True)
        value = critic(encoder_output)
    
    # Tur uzunluğunu hesapla
    tour_length = calculate_tour_length(tour, coordinates)
    
    # Görselleştir
    plot_tour(
        coordinates[0], 
        tour[0], 
        title=f"TSP Actor-Critic Çözümü ({num_cities} şehir)\nUzunluk: {tour_length.item():.4f}, Value: {value.item():.4f}",
        save_path="ac_single_example.png"
    )
    
    print(f"Tur uzunluğu: {tour_length.item():.4f}")
    print(f"Critic value tahmini: {value.item():.4f}")


def main():
    # Device seçimi
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Hyperparameters (main_actor_critic.py ile aynı olmalı)
    num_cities = 20
    d_model = 128
    num_heads = 8
    d_ff = 512
    num_layers = 3
    batch_size = 512
    
    # Test dataset
    data_dir = f"./data/tsp_{num_cities}"
    test_path = f"{data_dir}/test.pt"
    model_path = "best_actor_critic_model.pth"

    if not os.path.exists(test_path):
        print(f"Test dataset not found at {test_path}")
        print("Please generate dataset first:")
        print(f"  python data/generate_dataset.py --num_cities {num_cities}")
        return
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please train the model first:")
        print(f"  python main_actor_critic.py")
        return
        
    test_dataset = TSPDataset(file_path=test_path)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Test dataset loaded: {len(test_dataset)} samples")
    print(f"Batch size: {batch_size}\n")
    
    # Modelleri yükle
    actor, critic = load_models(
        model_path,
        device,
        input_dim=2,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers
    )
    
    print("Testing actor-critic models...\n")
    
    # Değerlendirme ve görselleştirme
    metrics, all_values, all_lengths = evaluate_and_visualize(actor, critic, test_loader, device)
    
    print("\n" + "="*60)
    print(" Tüm görselleştirmeler tamamlandı!")
    print("="*60)
    print("\nOluşturulan dosyalar:")
    print("  - ac_metrics_dashboard.png")
    print("  - ac_length_vs_index.png")
    print("  - ac_value_vs_length.png")
    print("  - ac_best_solutions.png")
    print("  - ac_worst_solutions.png")
    
    # Bonus: Tek bir örnek çözüm
    print("\n" + "="*60)
    print(" Tek Örnek Test")
    print("="*60)
    run_single_example(actor, critic, num_cities, device)
    print("\nTek örnek görselleştirmesi kaydedildi: ac_single_example.png")


if __name__ == "__main__":
    main()
