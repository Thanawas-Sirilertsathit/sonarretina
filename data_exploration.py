import matplotlib.pyplot as plt
import numpy as np
import csv
import glob
import os
from collections import Counter

# Define distance bins for 3 levels
DISTANCE_BINS = {
    0: (0, 2.5),      # near
    1: (2.5, 5.0),    # medium
    2: (5.0, 10)      # far
}

def get_distance_class(distance):
    for cls, (min_d, max_d) in DISTANCE_BINS.items():
        if min_d <= distance < max_d:
            return cls
    return 2  # default to far (last class)

def load_all_distances(csv_files):
    """Load all distance values from CSV files."""
    distances = []
    classes = []
    
    for csv_file in csv_files:
        with open(csv_file) as f:
            for row in csv.reader(f):
                try:
                    dist = float(row[5])  # column 5 is distance
                    distances.append(dist)
                    cls = get_distance_class(dist)
                    classes.append(cls)
                except (ValueError, IndexError):
                    continue
    
    return np.array(distances), np.array(classes)

def plot_distance_distribution(distances, classes):
    """Plot distance distribution and class counts."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogram of distances
    ax1.hist(distances, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_title('Distance Distribution')
    ax1.set_xlabel('Distance (meters)')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)
    
    # Add bin boundaries
    for cls, (min_d, max_d) in DISTANCE_BINS.items():
        ax1.axvline(min_d, color='red', linestyle='--', alpha=0.7, label=f'Bin {cls}' if cls == 0 else "")
        if cls < 2:
            ax1.axvline(max_d, color='red', linestyle='--', alpha=0.7)
    
    # Class count bar chart
    class_counts = Counter(classes)
    class_names = ['Near (0-2.5m)', 'Medium (2.5-5m)', 'Far (5-10m)']
    counts = [class_counts.get(i, 0) for i in range(3)]
    
    bars = ax2.bar(range(3), counts, color='green', alpha=0.7, edgecolor='black')
    ax2.set_title('Class Distribution')
    ax2.set_xlabel('Distance Class')
    ax2.set_ylabel('Number of Samples')
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(class_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01, 
                f'{count:,}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('data_distribution_3classes.png', dpi=150, bbox_inches='tight')
    plt.show()

def print_statistics(distances, classes):
    """Print statistical summary."""
    print("="*60)
    print("DATA DISTRIBUTION STATISTICS")
    print("="*60)
    print(f"Total samples: {len(distances):,}")
    print(f"Distance range: {distances.min():.3f} - {distances.max():.3f} meters")
    print(f"Mean distance: {distances.mean():.3f} meters")
    print(f"Median distance: {np.median(distances):.3f} meters")
    print(f"Standard deviation: {distances.std():.3f} meters")
    print()
    
    class_counts = Counter(classes)
    class_names = ['Near (0-2.5m)', 'Medium (2.5-5m)', 'Far (5-10m)']
    
    print("Class Distribution:")
    for i in range(3):
        count = class_counts.get(i, 0)
        percentage = (count / len(classes)) * 100
        print(f"  Class {i} ({class_names[i]}): {count:,} samples ({percentage:.1f}%)")
    
    print()
    print("Class balance:")
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    for cls, count in sorted_classes:
        print(f"  Class {cls}: {count:,} samples")

if __name__ == '__main__':
    # Find all CSV files
    csv_files = glob.glob('dcase/zenodo_upload/meta_data/*.csv')
    print(f"Found {len(csv_files)} CSV files")
    
    # Load data
    distances, classes = load_all_distances(csv_files)
    print(f"Loaded {len(distances)} distance measurements")
    
    # Print statistics
    print_statistics(distances, classes)
    
    # Plot distributions
    plot_distance_distribution(distances, classes)
    print("\nDistribution plot saved as 'data_distribution_3classes.png'")