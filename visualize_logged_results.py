#!/usr/bin/env python3
"""
Visualize dormancy analysis results from logged CSV files.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import numpy as np


def load_and_aggregate_results(csv_path):
    """Load CSV and compute aggregate statistics per step."""
    df = pd.read_csv(csv_path)
    
    # Convert step to numeric (handle 'N/A' values)
    df['step'] = pd.to_numeric(df['step'], errors='coerce')
    
    # Remove rows without valid step numbers
    df = df.dropna(subset=['step'])
    df['step'] = df['step'].astype(int)
    
    # Compute aggregate statistics per step
    agg_stats = df.groupby('step').agg({
        'pct_dormant': ['mean', 'min', 'max', 'std'],
        'num_dormant': 'sum',
        'num_neurons': 'sum',
    }).reset_index()
    
    # Flatten column names
    agg_stats.columns = ['step', 'avg_dormancy', 'min_dormancy', 'max_dormancy', 
                         'std_dormancy', 'total_dormant', 'total_neurons']
    
    return df, agg_stats


def plot_dormancy_over_steps(agg_stats, save_path=None):
    """Plot average, min, and max dormancy over training steps."""
    plt.figure(figsize=(12, 7))
    
    # Plot average line
    plt.plot(agg_stats['step'], agg_stats['avg_dormancy'], 
             color='steelblue', linewidth=2.5, label='Average Dormancy', 
             marker='o', markersize=8)
    
    # Fill between min and max
    plt.fill_between(agg_stats['step'], 
                     agg_stats['min_dormancy'], 
                     agg_stats['max_dormancy'], 
                     alpha=0.3, color='steelblue', label='Min-Max Range')
    
    plt.title('Dormancy Percentage Over Training Steps', fontsize=14, fontweight='bold')
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Percentage of Dormant Neurons (%)', fontsize=12)
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot to: {save_path}")
    
    plt.show()


def plot_layer_heatmap(df, max_layers=20, save_path=None):
    """Plot heatmap of dormancy across layers and steps."""
    # Pivot to get layer x step matrix
    # Take only first N layers for readability
    unique_layers = df['layer_name'].unique()[:max_layers]
    df_subset = df[df['layer_name'].isin(unique_layers)]
    
    pivot = df_subset.pivot_table(
        values='pct_dormant',
        index='layer_name',
        columns='step',
        aggfunc='mean'
    )
    
    plt.figure(figsize=(14, 8))
    sns.heatmap(pivot, cmap='YlOrRd', annot=True, fmt='.1f', 
                cbar_kws={'label': 'Dormancy %'})
    plt.title(f'Dormancy Heatmap: First {max_layers} Layers', fontsize=14, fontweight='bold')
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Layer', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved heatmap to: {save_path}")
    
    plt.show()


def plot_layer_evolution(df, num_layers=5, save_path=None):
    """Plot dormancy evolution for specific layers."""
    # Select layers with most variation
    layer_variance = df.groupby('layer_name')['pct_dormant'].var().sort_values(ascending=False)
    top_layers = layer_variance.head(num_layers).index
    
    plt.figure(figsize=(12, 7))
    
    for layer in top_layers:
        layer_data = df[df['layer_name'] == layer].sort_values('step')
        plt.plot(layer_data['step'], layer_data['pct_dormant'], 
                marker='o', label=layer, linewidth=2)
    
    plt.title(f'Dormancy Evolution: Top {num_layers} Most Variable Layers', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Dormancy Percentage (%)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved evolution plot to: {save_path}")
    
    plt.show()


def print_summary_stats(df, agg_stats):
    """Print summary statistics."""
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print(f"\nTotal checkpoints analyzed: {len(agg_stats)}")
    print(f"Total layers per checkpoint: {len(df['layer_name'].unique())}")
    print(f"Total records: {len(df)}")
    
    print(f"\nDormancy Range:")
    print(f"  Overall average: {df['pct_dormant'].mean():.2f}%")
    print(f"  Overall min: {df['pct_dormant'].min():.2f}%")
    print(f"  Overall max: {df['pct_dormant'].max():.2f}%")
    print(f"  Overall std: {df['pct_dormant'].std():.2f}%")
    
    print(f"\nPer-Step Averages:")
    print(agg_stats[['step', 'avg_dormancy', 'min_dormancy', 'max_dormancy']].to_string(index=False))
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize dormancy analysis results from CSV log files"
    )
    parser.add_argument(
        "csv_file",
        type=str,
        help="Path to CSV file with logged results"
    )
    parser.add_argument(
        "--save-prefix",
        type=str,
        default=None,
        help="Prefix for saved plot files (e.g., 'pythia_')"
    )
    parser.add_argument(
        "--max-layers-heatmap",
        type=int,
        default=20,
        help="Maximum number of layers to show in heatmap"
    )
    parser.add_argument(
        "--num-layers-evolution",
        type=int,
        default=5,
        help="Number of layers to show in evolution plot"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't show plots interactively (only save)"
    )
    
    args = parser.parse_args()
    
    print(f"Loading results from: {args.csv_file}")
    df, agg_stats = load_and_aggregate_results(args.csv_file)
    
    print_summary_stats(df, agg_stats)
    
    # Generate save paths if prefix provided
    save_main = f"{args.save_prefix}dormancy_over_steps.png" if args.save_prefix else None
    save_heatmap = f"{args.save_prefix}dormancy_heatmap.png" if args.save_prefix else None
    save_evolution = f"{args.save_prefix}layer_evolution.png" if args.save_prefix else None
    
    print("\nGenerating visualizations...\n")
    
    # Main plot
    print("1. Dormancy over training steps...")
    plot_dormancy_over_steps(agg_stats, save_path=save_main)
    
    # Heatmap
    print("\n2. Layer dormancy heatmap...")
    plot_layer_heatmap(df, max_layers=args.max_layers_heatmap, save_path=save_heatmap)
    
    # Layer evolution
    print("\n3. Layer evolution plot...")
    plot_layer_evolution(df, num_layers=args.num_layers_evolution, save_path=save_evolution)
    
    print("\n✓ All visualizations complete!")


if __name__ == "__main__":
    main()
