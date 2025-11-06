import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_dormancy_over_steps(csv_path):
    """
    Plot dormancy metrics over training steps with min/max as shaded channel.
    
    Args:
        csv_path: Path to CSV file with columns ['step', 'avg_dormancy', 'min_dormancy', 'max_dormancy']
    """
    # Load the data
    data = pd.read_csv(csv_path)
    
    # Create the plot
    plt.figure(figsize=(12, 7))
    
    # Plot the average dormancy line
    plt.plot(data['step'], data['avg_dormancy'], 
             color='steelblue', linewidth=2, label='Average Dormancy', marker='o')
    
    # Fill between min and max to show the range
    plt.fill_between(data['step'], data['min_dormancy'], data['max_dormancy'], 
                     alpha=0.3, color='steelblue', label='Min-Max Range')
    
    plt.title('Dormancy Percentage Over Training Steps (Pythia-70m-dedup)', fontsize=14, fontweight='bold')
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Percentage of Dormant Neurons (%)', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_avg_dormancy_over_steps(data):
    """
    Plot average dormancy percentage over training steps.
    
    Args:
        dormancy_df: DataFrame with columns ['step', 'layer', 'pct_dormant']
    """
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x='step', y='pct_dormant', hue='layer', marker='o')
    plt.title('Average Dormancy Percentage Over Training Steps')
    plt.xlabel('Training Step')
    plt.ylabel('Percentage of Dormant Neurons (%)')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    # Quick visualization of Pythia-70m-dedup results
    csv_path = 'first_results_Pythia-70m-dedup'
    plot_dormancy_over_steps(csv_path)