import numpy as np
from typing import Dict, Tuple, Optional

def compute_dormant_neurons(
    activations: np.ndarray,
    threshold: float = 0.025,
    return_scores: bool = False
) -> Dict:
    """
    Compute dormant neuron statistics from activation data.
    
    Based on "The Dormant Neuron Phenomenon in Deep Reinforcement Learning"
    (Sokar et al., 2023)
    
    Args:
        activations: Array of shape (num_samples, num_neurons) containing
                    neuron activations across a dataset
        threshold: Dormancy threshold τ. Neurons with normalized score ≤ τ
                  are considered dormant. Common values: 0 (strict), 0.025, 0.1
        return_scores: If True, return individual neuron scores
    
    Returns:
        Dictionary containing:
            - 'num_dormant': Number of dormant neurons
            - 'pct_dormant': Percentage of dormant neurons
            - 'dormant_indices': Indices of dormant neurons
            - 'scores': Normalized neuron scores (if return_scores=True)
    """
    # Compute mean absolute activation for each neuron
    mean_abs_activations = np.mean(np.abs(activations), axis=0)

    # DEBUG
    print(f"[DEBUG] Activations shape: {activations.shape}")
    print(f"[DEBUG] Mean abs acts: min={mean_abs_activations.min():.6f}, max={mean_abs_activations.max():.6f}, mean={mean_abs_activations.mean():.6f}")
    
    
    # Normalize scores so they sum to 1
    # This makes comparison across layers possible
    # scores = mean_abs_activations / (np.mean(mean_abs_activations) * len(mean_abs_activations))
    scores = mean_abs_activations / np.mean(mean_abs_activations)

     # DEBUG
    print(f"[DEBUG] Scores: min={scores.min():.6f}, max={scores.max():.6f}, sum={scores.sum():.6f}")
    print(f"[DEBUG] Threshold: {threshold}, Num below: {(scores <= threshold).sum()}/{len(scores)}")
    
    # Identify dormant neurons
    dormant_mask = scores <= threshold
    dormant_indices = np.where(dormant_mask)[0]
    
    results = {
        'num_dormant': int(np.sum(dormant_mask)),
        'pct_dormant': float(100 * np.mean(dormant_mask)),
        'dormant_indices': dormant_indices,
    }
    
    if return_scores:
        results['scores'] = scores
    
    return results


def compute_dormant_neurons_per_layer(
    activations_dict: Dict[str, np.ndarray],
    threshold: float = 0.025
) -> Dict[str, Dict]:
    """
    Compute dormant neurons for multiple layers.
    
    Args:
        activations_dict: Dictionary mapping layer names to activation arrays
                         Each array should be shape (num_samples, num_neurons)
        threshold: Dormancy threshold τ
    
    Returns:
        Dictionary mapping layer names to dormancy statistics
    """
    results = {}
    for layer_name, activations in activations_dict.items():
        results[layer_name] = compute_dormant_neurons(activations, threshold)
    
    return results


# Example usage with helpful utilities
def track_dormancy_over_time(
    activations_list: list,
    threshold: float = 0.025,
    window_size: Optional[int] = None
) -> np.ndarray:
    """
    Track percentage of dormant neurons over time/training.
    
    Args:
        activations_list: List of activation arrays, one per checkpoint
        threshold: Dormancy threshold
        window_size: If provided, use rolling window of recent samples
    
    Returns:
        Array of dormancy percentages over time
    """
    dormancy_pcts = []
    
    for activations in activations_list:
        if window_size is not None and len(activations) > window_size:
            # Use only recent samples
            activations = activations[-window_size:]
        
        result = compute_dormant_neurons(activations, threshold)
        dormancy_pcts.append(result['pct_dormant'])
    
    return np.array(dormancy_pcts)