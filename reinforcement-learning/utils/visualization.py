import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats


def calculate_metrics(tour_lengths):
    """
    Tur uzunlukları için detaylı metrikler hesaplar.
    
    Args:
        tour_lengths: Tur uzunluklarının listesi
        
    Returns:
        dict: Metrikler sözlüğü
    """
    lengths = np.array(tour_lengths)
    
    metrics = {
        'count': len(lengths),
        'mean': np.mean(lengths),
        'std': np.std(lengths),
        'min': np.min(lengths),
        'max': np.max(lengths),
        'median': np.median(lengths),
        'q1': np.percentile(lengths, 25),
        'q3': np.percentile(lengths, 75),
        'iqr': np.percentile(lengths, 75) - np.percentile(lengths, 25),
        'variance': np.var(lengths),
        'range': np.max(lengths) - np.min(lengths),
        'cv': np.std(lengths) / np.mean(lengths) * 100,  # Coefficient of variation (%)
    }
    
    # 95% confidence interval
    ci = stats.t.interval(0.95, len(lengths)-1, loc=np.mean(lengths), scale=stats.sem(lengths))
    metrics['ci_lower'] = ci[0]
    metrics['ci_upper'] = ci[1]
    
    return metrics


def print_metrics_table(metrics, title="Model Performance Metrics"):
    """
    Metrikleri güzel bir tablo formatında yazdırır.
    """
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)
    print(f"  {'Metric':<25} {'Value':>15}")
    print("-" * 60)
    print(f"  {'Sample Count':<25} {metrics['count']:>15}")
    print(f"  {'Mean Tour Length':<25} {metrics['mean']:>15.4f}")
    print(f"  {'Std Deviation':<25} {metrics['std']:>15.4f}")
    print(f"  {'Variance':<25} {metrics['variance']:>15.4f}")
    print(f"  {'Coefficient of Var (%)':<25} {metrics['cv']:>15.2f}")
    print("-" * 60)
    print(f"  {'Minimum':<25} {metrics['min']:>15.4f}")
    print(f"  {'25th Percentile (Q1)':<25} {metrics['q1']:>15.4f}")
    print(f"  {'Median (Q2)':<25} {metrics['median']:>15.4f}")
    print(f"  {'75th Percentile (Q3)':<25} {metrics['q3']:>15.4f}")
    print(f"  {'Maximum':<25} {metrics['max']:>15.4f}")
    print("-" * 60)
    print(f"  {'Range':<25} {metrics['range']:>15.4f}")
    print(f"  {'IQR':<25} {metrics['iqr']:>15.4f}")
    print(f"  {'95% CI Lower':<25} {metrics['ci_lower']:>15.4f}")
    print(f"  {'95% CI Upper':<25} {metrics['ci_upper']:>15.4f}")
    print("=" * 60)


def plot_metrics_dashboard(tour_lengths, save_path=None):
    """
    Kapsamlı metrik dashboard'u oluşturur.
    
    Args:
        tour_lengths: Tur uzunluklarının listesi
        save_path: Kaydetme yolu (opsiyonel)
    """
    lengths = np.array(tour_lengths)
    metrics = calculate_metrics(tour_lengths)
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Histogram
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.hist(lengths, bins=30, color='steelblue', edgecolor='white', alpha=0.7)
    ax1.axvline(metrics['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {metrics['mean']:.4f}")
    ax1.axvline(metrics['median'], color='green', linestyle='--', linewidth=2, label=f"Median: {metrics['median']:.4f}")
    ax1.set_xlabel('Tour Length')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Tour Lengths')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Box Plot
    ax2 = fig.add_subplot(2, 3, 2)
    bp = ax2.boxplot(lengths, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    ax2.set_ylabel('Tour Length')
    ax2.set_title('Box Plot of Tour Lengths')
    ax2.grid(True, alpha=0.3)
    
    # Outlier'ları işaretle
    q1, q3 = metrics['q1'], metrics['q3']
    iqr = metrics['iqr']
    outliers = lengths[(lengths < q1 - 1.5*iqr) | (lengths > q3 + 1.5*iqr)]
    ax2.text(1.15, metrics['median'], f"Median: {metrics['median']:.4f}", fontsize=9)
    ax2.text(1.15, q1, f"Q1: {q1:.4f}", fontsize=9)
    ax2.text(1.15, q3, f"Q3: {q3:.4f}", fontsize=9)
    
    # 3. CDF (Cumulative Distribution Function)
    ax3 = fig.add_subplot(2, 3, 3)
    sorted_lengths = np.sort(lengths)
    cdf = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)
    ax3.plot(sorted_lengths, cdf, color='steelblue', linewidth=2)
    ax3.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax3.axvline(metrics['median'], color='green', linestyle='--', alpha=0.5)
    ax3.fill_between(sorted_lengths, 0, cdf, alpha=0.3)
    ax3.set_xlabel('Tour Length')
    ax3.set_ylabel('Cumulative Probability')
    ax3.set_title('Cumulative Distribution Function (CDF)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Violin Plot
    ax4 = fig.add_subplot(2, 3, 4)
    parts = ax4.violinplot(lengths, positions=[1], showmeans=True, showmedians=True)
    parts['bodies'][0].set_facecolor('lightblue')
    parts['bodies'][0].set_alpha(0.7)
    ax4.set_ylabel('Tour Length')
    ax4.set_title('Violin Plot of Tour Lengths')
    ax4.set_xticks([1])
    ax4.set_xticklabels(['Model'])
    ax4.grid(True, alpha=0.3)
    
    # 5. Metrics Table (Text)
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.axis('off')
    
    table_data = [
        ['Metric', 'Value'],
        ['Mean', f"{metrics['mean']:.4f}"],
        ['Std Dev', f"{metrics['std']:.4f}"],
        ['Min', f"{metrics['min']:.4f}"],
        ['Max', f"{metrics['max']:.4f}"],
        ['Median', f"{metrics['median']:.4f}"],
        ['Q1', f"{metrics['q1']:.4f}"],
        ['Q3', f"{metrics['q3']:.4f}"],
        ['IQR', f"{metrics['iqr']:.4f}"],
        ['CV (%)', f"{metrics['cv']:.2f}"],
        ['95% CI', f"[{metrics['ci_lower']:.4f}, {metrics['ci_upper']:.4f}]"],
    ]
    
    table = ax5.table(cellText=table_data, loc='center', cellLoc='center',
                       colWidths=[0.4, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Header'ı renklendir
    table[(0, 0)].set_facecolor('#4472C4')
    table[(0, 1)].set_facecolor('#4472C4')
    table[(0, 0)].set_text_props(color='white', fontweight='bold')
    table[(0, 1)].set_text_props(color='white', fontweight='bold')
    
    ax5.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)
    
    # 6. Percentile Bar Chart
    ax6 = fig.add_subplot(2, 3, 6)
    percentiles = [10, 25, 50, 75, 90]
    percentile_values = [np.percentile(lengths, p) for p in percentiles]
    bars = ax6.bar([f'{p}th' for p in percentiles], percentile_values, color='steelblue', edgecolor='white')
    ax6.set_xlabel('Percentile')
    ax6.set_ylabel('Tour Length')
    ax6.set_title('Tour Length by Percentile')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Bar değerlerini üstüne yaz
    for bar, val in zip(bars, percentile_values):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('TSP Model Performance Dashboard', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Dashboard kaydedildi: {save_path}")
    
    plt.show()
    
    return metrics


def plot_length_vs_index(tour_lengths, save_path=None):
    """
    Tur uzunluklarını index'e göre çizer (trend analizi için).
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.plot(tour_lengths, 'b-', alpha=0.5, linewidth=0.5)
    ax.scatter(range(len(tour_lengths)), tour_lengths, c='steelblue', s=10, alpha=0.5)
    
    # Moving average
    window = min(50, len(tour_lengths) // 10)
    if window > 1:
        moving_avg = np.convolve(tour_lengths, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(tour_lengths)), moving_avg, 'r-', linewidth=2, 
                label=f'Moving Avg (window={window})')
    
    ax.axhline(np.mean(tour_lengths), color='green', linestyle='--', 
               label=f'Mean: {np.mean(tour_lengths):.4f}')
    
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Tour Length')
    ax.set_title('Tour Length vs Sample Index')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Grafik kaydedildi: {save_path}")
    
    plt.show()


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
