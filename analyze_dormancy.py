import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from typing import Dict, List, Optional, Union
from tqdm import tqdm
import argparse


class ActivationCapture:
    """Hook to capture activations from model layers."""
    
    def __init__(self):
        self.activations = {}
        self.hooks = []
    
    def get_activation(self, name: str, do_abs: bool = True):
        """Create a hook function for a specific layer.
        do_abs: Whether to take absolute value of activations, as in dormancy paper.
        This is important because it needs to be done before averaging over sequence length."""
        def hook(model, input, output):
            # Handle different output types
            if isinstance(output, tuple):
                output = output[0]
            
            # Extract activation values
            if isinstance(output, torch.Tensor):

                # For MLP/feedforward layers: (batch, seq_len, hidden_dim)
                # We'll take mean over sequence dimension
                activation = output.detach().cpu()

                # DEBUG: Print shape and stats
                # print(f"[DEBUG] {name}: shape={activation.shape}, mean={activation.mean():.4f}, std={activation.std():.4f}, min={activation.min():.4f}, max={activation.max():.4f}")
                
                # Average over sequence length dimension
                if activation.dim() == 3:
                    if do_abs:
                        activation = torch.abs(activation).mean(dim=1)  # (batch, hidden_dim)
                    else:
                        activation = activation.mean(dim=1)  # (batch, hidden_dim)
                elif activation.dim() == 2:
                    if do_abs:
                        activation = torch.abs(activation)  # (batch, hidden_dim)
                else:
                    return  # Skip unexpected shapes
                
                if name not in self.activations:
                    self.activations[name] = []
                self.activations[name].append(activation)
        
        return hook
    
    def register_hooks(self, model, layer_pattern: str = "mlp", debug: bool = False):
        """
        Register hooks on model layers.
        
        Args:
            model: The model to hook
            layer_pattern: Pattern to match layer names. Options:
                - "mlp": Hook MLP/feedforward layers (after activation)
                - "mlp_out": Hook MLP output (after final projection)
                - "attention": Hook attention output
                - "all": Hook all major layers
            debug: If True, print all layer names being considered
        """
        hooked_layers = []
        
        for name, module in model.named_modules():
            should_hook = False
            hook_name = name
            
            if debug:
                print(f"Examining: {name} - {type(module).__name__}")
            
            # GPT-2 / GPT-style models
            if "gpt" in model.__class__.__name__.lower():
                if layer_pattern == "mlp":
                    # Hook after the activation function in MLP
                    # In GPT-2: transformer.h.{i}.mlp.act (after GELU activation)
                    # Or hook c_fc (first projection) which is after activation
                    if ".mlp.c_fc" in name or ".mlp.act" in name:
                        should_hook = True
                elif layer_pattern == "mlp_out":
                    # Hook after final MLP projection
                    if ".mlp.c_proj" in name:
                        should_hook = True
                elif layer_pattern == "attention":
                    if ".attn.c_proj" in name:
                        should_hook = True
                elif layer_pattern == "all":
                    if ".mlp.c_proj" in name or ".attn.c_proj" in name:
                        should_hook = True
            
            # BERT-style models
            elif "bert" in model.__class__.__name__.lower():
                if layer_pattern == "mlp":
                    # Hook intermediate layer (after activation)
                    if "intermediate.dense" in name or "output.dense" in name:
                        should_hook = True
                elif layer_pattern == "attention":
                    if "attention.output.dense" in name:
                        should_hook = True
                elif layer_pattern == "all":
                    if ("intermediate" in name or "attention.output" in name) and "dense" in name:
                        should_hook = True
            
            # T5-style models
            elif "t5" in model.__class__.__name__.lower():
                if layer_pattern == "mlp":
                    if "DenseReluDense.wo" in name or "DenseReluDense.wi" in name:
                        should_hook = True
                elif layer_pattern == "attention":
                    if "attention.o" in name:
                        should_hook = True
                elif layer_pattern == "all":
                    if "DenseReluDense.wo" in name or "attention.o" in name:
                        should_hook = True
            
            # LLaMA / OPT / Generic transformer models
            else:
                # Generic patterns that work for many models
                if layer_pattern == "mlp":
                    if any(x in name.lower() for x in [".mlp.", ".feed_forward.", ".ffn."]):
                        # Hook on Linear layers within MLP
                        if isinstance(module, torch.nn.Linear):
                            should_hook = True
                elif layer_pattern == "attention":
                    if "attn" in name.lower() and isinstance(module, torch.nn.Linear):
                        should_hook = True
                elif layer_pattern == "all":
                    if any(x in name.lower() for x in [".mlp.", ".attn.", ".attention."]):
                        if isinstance(module, torch.nn.Linear):
                            should_hook = True
            
            # Register hook if this layer matches
            # if should_hook and isinstance(module, torch.nn.Linear):
            if should_hook:
                hook = module.register_forward_hook(self.get_activation(hook_name))
                self.hooks.append(hook)
                hooked_layers.append(hook_name)
                if debug:
                    print(f"  ✓ HOOKED: {hook_name}")
        
        print(f"\nRegistered {len(hooked_layers)} hooks")
        if len(hooked_layers) > 0:
            print(f"First few hooked layers: {hooked_layers[:3]}")
            print(f"Last few hooked layers: {hooked_layers[-3:]}")
        else:
            print("⚠️  WARNING: No hooks registered! Try setting debug=True")
    
    def clear(self):
        """Clear stored activations."""
        self.activations = {}
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_stacked_activations(self) -> Dict[str, np.ndarray]:
        """Stack all captured activations into numpy arrays."""
        stacked = {}
        for name, acts in self.activations.items():
            if acts:
                stacked[name] = torch.cat(acts, dim=0).numpy()
        return stacked


def analyze_model_dormancy(
    model_name: str,
    dataset_name: str,
    dataset_config: Optional[str] = None,
    dataset_split: str = "train",
    text_column: str = "text",
    num_samples: int = 500,
    max_length: int = 512,
    threshold: float = 0.1,
    layer_pattern: str = "mlp",
    batch_size: int = 8,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    debug: bool = False,
    revision: Optional[str] = None  # ADD THIS
) -> Dict:
    """
    Analyze dormant neurons in a HuggingFace model.
    
    Args:
        model_name: HuggingFace model identifier (e.g., "gpt2", "bert-base-uncased")
        dataset_name: HuggingFace dataset identifier (e.g., "wikitext", "c4")
        dataset_config: Dataset configuration (e.g., "wikitext-2-raw-v1")
        dataset_split: Dataset split to use
        text_column: Name of the text column in the dataset
        num_samples: Number of samples to process
        max_length: Maximum sequence length
        threshold: Dormancy threshold (τ)
        layer_pattern: Which layers to analyze ("mlp", "mlp_out", "attention", or "all")
        batch_size: Batch size for inference
        device: Device to run on
        debug: Enable debug output
    
    Returns:
        Dictionary with dormancy statistics per layer
    """
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model = AutoModel.from_pretrained(model_name, revision=revision)
    model.to(device)
    model.eval()
    
    print(f"Model type: {type(model).__name__}")
    if revision:
        print(f"  Revision: {revision}")
    
    # Set padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading dataset: {dataset_name}")
    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, split=dataset_split)
    else:
        dataset = load_dataset(dataset_name, split=dataset_split)
    
    # Take subset
    if len(dataset) > num_samples:
        dataset = dataset.select(range(num_samples))
    
    # Setup activation capture
    capture = ActivationCapture()
    capture.register_hooks(model, layer_pattern=layer_pattern, debug=debug)
    
    if len(capture.hooks) == 0:
        print("\n⚠️  ERROR: No hooks were registered!")
        print("Try running with --debug to see available layers")
        print(f"For {type(model).__name__}, you might need a different --layer-pattern")
        return {}
    
    print(f"\nProcessing {len(dataset)} samples in batches of {batch_size}...")
    
    # Process dataset in batches
    processed_samples = 0
    with torch.no_grad():
        for i in tqdm(range(0, len(dataset), batch_size)):
            batch = dataset[i:i+batch_size]
            
            # Handle different dataset formats
            if text_column in batch:
                texts = batch[text_column]
            elif 'sentence' in batch:
                texts = batch['sentence']
            else:
                # Take first string column
                texts = [str(v) for v in list(batch.values())[0]]
            
            # Filter out empty texts
            texts = [t for t in texts if t and len(t.strip()) > 0]
            if not texts:
                continue
            
            # Tokenize
            inputs = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(device)
            
            # Forward pass (activations captured by hooks)
            try:
                outputs = model(**inputs)
                processed_samples += len(texts)
            except Exception as e:
                print(f"Warning: Error processing batch {i}: {e}")
                continue
    
    print(f"Processed {processed_samples} samples")
    
    # Get stacked activations
    print("\nComputing dormancy statistics...")
    activations_dict = capture.get_stacked_activations()
    
    if not activations_dict:
        print("Warning: No activations captured! Check layer_pattern or enable debug mode.")
        capture.remove_hooks()
        return {}
    
    print(f"Captured activations from {len(activations_dict)} layers")
    for name, acts in list(activations_dict.items())[:3]:
        print(f"  {name}: shape {acts.shape}")
    
    # Import the dormancy computation function
    from dormancy_utils import compute_dormant_neurons_per_layer
    
    # Compute dormancy
    results = compute_dormant_neurons_per_layer(activations_dict, threshold=threshold)
    
    # Cleanup
    capture.remove_hooks()
    
    return results


def print_dormancy_report(results: Dict, verbose: bool = True):
    """Print a formatted report of dormancy statistics."""
    print("\n" + "="*70)
    print("DORMANCY ANALYSIS REPORT")
    print("="*70)
    
    if not results:
        print("No results to display.")
        return
    
    # Calculate summary statistics
    total_dormant = sum(stats['num_dormant'] for stats in results.values())
    
    print(f"\nOverall Statistics:")
    print(f"  Total layers analyzed: {len(results)}")
    print(f"  Average dormancy: {np.mean([s['pct_dormant'] for s in results.values()]):.2f}%")
    print(f"  Min dormancy: {np.min([s['pct_dormant'] for s in results.values()]):.2f}%")
    print(f"  Max dormancy: {np.max([s['pct_dormant'] for s in results.values()]):.2f}%")
    
    print(f"\nPer-Layer Statistics:")
    print(f"{'Layer Name':<50} {'Dormant':<15} {'%':<10}")
    print("-"*70)
    
    # Sort by dormancy percentage
    sorted_results = sorted(results.items(), 
                          key=lambda x: x[1]['pct_dormant'], 
                          reverse=True)
    
    for layer_name, stats in sorted_results:
        # Shorten long layer names
        display_name = layer_name if len(layer_name) <= 47 else layer_name[:44] + "..."
        print(f"{display_name:<50} {stats['num_dormant']:<15} {stats['pct_dormant']:>6.2f}%")
    
    if verbose:
        print("\n" + "="*70)
        print("Most dormant layers:")
        for layer_name, stats in sorted_results[:3]:
            print(f"\n{layer_name}:")
            print(f"  {stats['num_dormant']} dormant neurons ({stats['pct_dormant']:.2f}%)")
            if len(stats['dormant_indices']) > 0:
                print(f"  First few dormant indices: {stats['dormant_indices'][:10].tolist()}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze dormant neurons in HuggingFace language models"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        help="HuggingFace model name (e.g., 'gpt2', 'bert-base-uncased')"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext",
        help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default=None,
        help="Dataset configuration (e.g., 'wikitext-2-raw-v1')"
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Name of text column in dataset"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=500,
        help="Number of samples to analyze"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Dormancy threshold (0, 0.025, or 0.1)"
    )
    parser.add_argument(
        "--layer-pattern",
        type=str,
        default="mlp",
        choices=["mlp", "mlp_out", "attention", "all"],
        help="Which layers to analyze"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )

    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Model revision/checkpoint (e.g., 'step3000' for Pythia)"
    )


    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output to see all layers"
    )
    
    args = parser.parse_args()
    
    results = analyze_model_dormancy(
        model_name=args.model,
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        text_column=args.text_column,
        num_samples=args.num_samples,
        threshold=args.threshold,
        layer_pattern=args.layer_pattern,
        batch_size=args.batch_size,
        max_length=args.max_length,
        debug=args.debug,
        revision=args.revision
    )
    
    print_dormancy_report(results, verbose=True)


if __name__ == "__main__":
    main()