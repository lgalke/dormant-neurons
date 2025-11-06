from dormancy import compute_dormant_neurons, compute_dormant_neurons_per_layer, track_dormancy_over_time
import numpy as np
# Example 1: Single layer analysis
# Assume you have activations from a language model layer
activations = np.random.randn(1000, 512)  # 1000 samples, 512 neurons
activations[activations < 0] = 0  # ReLU-like activation

results = compute_dormant_neurons(activations, threshold=0.1, return_scores=True)
print(f"Dormant neurons: {results['num_dormant']} ({results['pct_dormant']:.1f}%)")
print(f"Dormant neuron indices: {results['dormant_indices'][:10]}...")  # First 10

# Example 2: Multiple layers
layer_activations = {
    'layer_0': np.random.randn(1000, 512),
    'layer_1': np.random.randn(1000, 512),
    'layer_2': np.random.randn(1000, 512),
}

multi_layer_results = compute_dormant_neurons_per_layer(layer_activations, threshold=0.1)
for layer, stats in multi_layer_results.items():
    print(f"{layer}: {stats['pct_dormant']:.1f}% dormant")

# Example 3: Track over training checkpoints
checkpoints = [np.random.randn(500, 512) for _ in range(10)]
dormancy_over_time = track_dormancy_over_time(checkpoints, threshold=0.1)
print(f"Dormancy evolution: {dormancy_over_time}")