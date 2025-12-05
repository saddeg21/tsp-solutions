"""
TSP Model Test Script

Bu script eğitilmiş modeli test eder ve sonuçları görselleştirir.
"""

import torch
from torch.utils.data import DataLoader

from model import TSPModel
from env import TSPDataset, calculate_tour_length
from utils.visualization import plot_tour, plot_multiple_tours, plot_comparison


def load_model(model_path, device, **model_kwargs):
    """Kaydedilmiş modeli yükler."""
    model = TSPModel(**model_kwargs).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model yüklendi: {model_path}")
    return model


def test_model(model, test_loader, device):
    """
    Modeli test veriseti üzerinde değerlendirir.
    
    Returns:
        avg_length: Ortalama tur uzunluğu
        all_tours: Tüm turlar
        all_coords: Tüm koordinatlar
        all_lengths: Tüm tur uzunlukları
    """
    model.eval()
    all_tours = []
    all_coords = []
    all_lengths = []
    
    with torch.no_grad():
        for batch in test_loader:
            coordinates = batch.to(device)
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
    print("=" * 50)
    print("Model Değerlendirmesi")
    print("=" * 50)
    
    avg_length, all_tours, all_coords, all_lengths = test_model(model, test_loader, device)
    
    print(f"\nTest Sonuçları:")
    print(f"  - Test örnek sayısı: {len(all_lengths)}")
    print(f"  - Ortalama tur uzunluğu: {avg_length:.4f}")
    print(f"  - En kısa tur: {min(all_lengths):.4f}")
    print(f"  - En uzun tur: {max(all_lengths):.4f}")
    print(f"  - Standart sapma: {torch.tensor(all_lengths).std().item():.4f}")
    
    # En iyi ve en kötü sonuçları bul
    sorted_indices = sorted(range(len(all_lengths)), key=lambda i: all_lengths[i])
    
    # Görselleştirme için örnekler seç
    # En iyi 3 ve en kötü 3
    best_indices = sorted_indices[:3]
    worst_indices = sorted_indices[-3:]
    
    print(f"\n--- En İyi 3 Çözüm ---")
    for i, idx in enumerate(best_indices):
        print(f"  {i+1}. Tur uzunluğu: {all_lengths[idx]:.4f}")
    
    print(f"\n--- En Kötü 3 Çözüm ---")
    for i, idx in enumerate(worst_indices):
        print(f"  {i+1}. Tur uzunluğu: {all_lengths[idx]:.4f}")
    
    # Görselleştir
    print(f"\n--- Görselleştirme ---")
    
    # En iyi çözümler
    best_coords = [all_coords[i] for i in best_indices]
    best_tours = [all_tours[i] for i in best_indices]
    best_titles = [f"En İyi {i+1}\nUzunluk: {all_lengths[idx]:.4f}" 
                   for i, idx in enumerate(best_indices)]
    
    print("En iyi 3 çözüm görselleştiriliyor...")
    plot_multiple_tours(best_coords, best_tours, best_titles, save_path="best_solutions.png")
    
    # En kötü çözümler
    worst_coords = [all_coords[i] for i in worst_indices]
    worst_tours = [all_tours[i] for i in worst_indices]
    worst_titles = [f"En Kötü {i+1}\nUzunluk: {all_lengths[idx]:.4f}" 
                    for i, idx in enumerate(worst_indices)]
    
    print("En kötü 3 çözüm görselleştiriliyor...")
    plot_multiple_tours(worst_coords, worst_tours, worst_titles, save_path="worst_solutions.png")
    
    return avg_length


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
    test_size = 1000
    batch_size = 100
    
    print(f"\nTest dataseti oluşturuluyor ({test_size} örnek)...")
    test_dataset = TSPDataset(num_samples=test_size, num_cities=num_cities)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Model oluştur
    model = TSPModel(
        input_dim=2,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers
    ).to(device)
    
    # Eğitilmiş model varsa yükle
    import os
    model_path = "tsp_model.pth"
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Eğitilmiş model yüklendi: {model_path}")
    else:
        print(f"Uyarı: Eğitilmiş model bulunamadı ({model_path})")
        print("Rastgele ağırlıklarla test ediliyor...")
    
    # Değerlendir ve görselleştir
    evaluate_and_visualize(model, test_loader, device)
    
    # Tek örnek çalıştır
    print("\n" + "=" * 50)
    print("Tek Örnek Testi")
    print("=" * 50)
    run_single_example(model, num_cities, device)


if __name__ == "__main__":
    main()
