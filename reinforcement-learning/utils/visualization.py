import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_tour(coordinates, tour, title=None, save_path=None, ax=None):
    """
    Tek bir TSP turunu görselleştirir.
    
    Args:
        coordinates: Şehir koordinatları (num_cities, 2) - tensor veya numpy
        tour: Ziyaret sırası (num_cities,) - tensor veya numpy
        title: Grafik başlığı (opsiyonel)
        save_path: Kaydetme yolu (opsiyonel)
        ax: Matplotlib axis (opsiyonel, subplot için)
    """
    # Tensor'ları numpy'a çevir
    if isinstance(coordinates, torch.Tensor):
        coordinates = coordinates.cpu().numpy()
    if isinstance(tour, torch.Tensor):
        tour = tour.cpu().numpy()
    
    # Tur sırasına göre koordinatları al
    ordered_coords = coordinates[tour]
    
    # Başlangıca dönüş için ilk şehri sona ekle
    ordered_coords = np.vstack([ordered_coords, ordered_coords[0]])
    
    # Yeni figure oluştur veya verilen axis'i kullan
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Tur çizgilerini çiz
    ax.plot(ordered_coords[:, 0], ordered_coords[:, 1], 'b-', linewidth=1.5, alpha=0.7)
    
    # Tüm şehirleri çiz
    ax.scatter(coordinates[:, 0], coordinates[:, 1], c='blue', s=50, zorder=5)
    
    # Başlangıç şehrini farklı renkte göster
    start_city = coordinates[tour[0]]
    ax.scatter(start_city[0], start_city[1], c='green', s=150, marker='*', zorder=10, label='Start')
    
    # Şehir numaralarını ekle
    for i, (x, y) in enumerate(coordinates):
        ax.annotate(str(i), (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
    
    # Tur uzunluğunu hesapla
    total_length = 0
    for i in range(len(tour)):
        next_i = (i + 1) % len(tour)
        city1 = coordinates[tour[i]]
        city2 = coordinates[tour[next_i]]
        total_length += np.sqrt(np.sum((city1 - city2) ** 2))
    
    # Başlık
    if title:
        ax.set_title(f"{title}\nTur Uzunluğu: {total_length:.4f}")
    else:
        ax.set_title(f"TSP Tur - Uzunluk: {total_length:.4f}")
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    # Kaydet veya göster
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Grafik kaydedildi: {save_path}")
    
    return ax


def plot_multiple_tours(coordinates_list, tours_list, titles=None, save_path=None, cols=3):
    """
    Birden fazla TSP turunu yan yana görselleştirir.
    
    Args:
        coordinates_list: Koordinat listesi
        tours_list: Tur listesi
        titles: Başlık listesi (opsiyonel)
        save_path: Kaydetme yolu (opsiyonel)
        cols: Sütun sayısı
    """
    n = len(coordinates_list)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, (coords, tour) in enumerate(zip(coordinates_list, tours_list)):
        title = titles[i] if titles else f"Problem {i+1}"
        plot_tour(coords, tour, title=title, ax=axes[i])
    
    # Boş subplot'ları gizle
    for i in range(n, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Grafik kaydedildi: {save_path}")
    
    plt.show()


def plot_training_curve(losses, val_lengths=None, save_path=None):
    """
    Eğitim sürecini görselleştirir.
    
    Args:
        losses: Epoch başına loss listesi
        val_lengths: Epoch başına validation tur uzunlukları (opsiyonel)
        save_path: Kaydetme yolu (opsiyonel)
    """
    epochs = range(1, len(losses) + 1)
    
    if val_lengths:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    else:
        fig, ax1 = plt.subplots(figsize=(8, 5))
    
    # Loss grafiği
    ax1.plot(epochs, losses, 'b-', linewidth=2, label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Validation grafiği
    if val_lengths:
        ax2.plot(epochs, val_lengths, 'g-', linewidth=2, label='Validation Tour Length')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Average Tour Length')
        ax2.set_title('Validation Performance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Grafik kaydedildi: {save_path}")
    
    plt.show()


def plot_comparison(coordinates, model_tour, optimal_tour=None, save_path=None):
    """
    Model çözümünü optimal çözümle karşılaştırır.
    
    Args:
        coordinates: Şehir koordinatları
        model_tour: Model tarafından bulunan tur
        optimal_tour: Optimal tur (varsa)
        save_path: Kaydetme yolu (opsiyonel)
    """
    if optimal_tour is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        plot_tour(coordinates, model_tour, title="Model Çözümü", ax=ax1)
        plot_tour(coordinates, optimal_tour, title="Optimal Çözüm", ax=ax2)
    else:
        fig, ax = plt.subplots(figsize=(8, 8))
        plot_tour(coordinates, model_tour, title="Model Çözümü", ax=ax)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Grafik kaydedildi: {save_path}")
    
    plt.show()
