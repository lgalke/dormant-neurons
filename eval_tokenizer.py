"""
Tokenizer Evaluation Script
Calculates metrics for tokenizer performance on a dataset, following arXiv 2508.04796:
- Fertility Rate: Average number of tokens per word (whitespace-delimited)
- Compression Ratio: Average number of words per token (per-text averaging)
- Vocabulary Utilization: Percentage of vocabulary used
- Token Distribution Gini: Measure of inequality in token usage distribution
"""

import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset
from collections import Counter
from typing import Dict, List, Tuple
from tqdm import tqdm


def tokenize_texts(texts: List[str], tokenizer) -> List[Tuple[str, List[int]]]:
    """
    Tokenize all texts once and return with original texts.

    Args:
        texts: List of text strings to tokenize
        tokenizer: HuggingFace tokenizer

    Returns:
        List of tuples (original_text, token_ids)
    """
    tokenized = []
    for text in tqdm(texts, desc="Tokenizing", unit="text"):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        tokenized.append((text, tokens))
    return tokenized


def calculate_fertility_rate(tokenized_texts: List[Tuple[str, List[int]]]) -> float:
    """
    Calculate fertility rate: average number of tokens per word (whitespace-delimited).
    Lower is better (more efficient).

    As defined in arXiv 2508.04796: Fertility(T) = Σ|τ(b)| / Σ|b|_u
    where u is the normalization unit (words).

    Args:
        tokenized_texts: List of (text, token_ids) tuples
    """
    total_tokens = 0
    total_words = 0

    for text, tokens in tokenized_texts:
        total_tokens += len(tokens)
        total_words += len(text.split())

    return total_tokens / total_words if total_words > 0 else 0


def calculate_compression_ratio(tokenized_texts: List[Tuple[str, List[int]]]) -> float:
    """
    Calculate compression ratio: average number of words per token across all texts.
    Higher is better (more efficient).

    As defined in arXiv 2508.04796: CR(D;τ) = (1/|D|) Σ |b|_u / |τ(b)|
    where u is the normalization unit (words) and the average is taken per text.

    Args:
        tokenized_texts: List of (text, token_ids) tuples
    """
    if not tokenized_texts:
        return 0.0

    ratios = []
    for text, tokens in tokenized_texts:
        num_words = len(text.split())
        num_tokens = len(tokens)
        if num_tokens > 0:
            ratios.append(num_words / num_tokens)

    return np.mean(ratios) if ratios else 0.0


def calculate_vocabulary_utilization(tokenized_texts: List[Tuple[str, List[int]]], vocab_size: int) -> Tuple[float, int, int]:
    """
    Calculate vocabulary utilization: percentage of vocabulary used.
    Returns (utilization_percentage, unique_tokens_used, vocab_size)

    Args:
        tokenized_texts: List of (text, token_ids) tuples
        vocab_size: Size of the tokenizer vocabulary
    """
    all_tokens = []

    for _, tokens in tokenized_texts:
        all_tokens.extend(tokens)

    unique_tokens = len(set(all_tokens))
    utilization = (unique_tokens / vocab_size) * 100 if vocab_size > 0 else 0

    return utilization, unique_tokens, vocab_size


def calculate_token_distribution_gini(tokenized_texts: List[Tuple[str, List[int]]]) -> float:
    """
    Calculate Gini coefficient for token frequency distribution.
    0 = perfect equality, 1 = perfect inequality.
    Measures how evenly tokens are used across the dataset.

    NOTE: This measures vocabulary distribution inequality, NOT cross-lingual fairness
    as defined in arXiv 2508.04796. The paper's Gini measures tokenization cost
    inequality across different languages in a parallel corpus.

    Args:
        tokenized_texts: List of (text, token_ids) tuples
    """
    all_tokens = []

    for _, tokens in tokenized_texts:
        all_tokens.extend(tokens)

    if not all_tokens:
        return 0.0

    # Count token frequencies
    token_counts = Counter(all_tokens)

    # Get sorted array of counts
    counts = np.array(sorted(token_counts.values()))

    if len(counts) == 0:
        return 0.0

    # Calculate Gini coefficient
    n = len(counts)
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * counts)) / (n * np.sum(counts)) - (n + 1) / n

    return gini


def evaluate_tokenizer(model_name: str, dataset_name: str, dataset_split: str = "train",
                       num_samples: int = 1000, text_field: str = "text") -> Dict[str, float]:
    """
    Evaluate a tokenizer on a dataset and return all metrics.

    Args:
        model_name: HuggingFace model name for tokenizer
        dataset_name: HuggingFace dataset name
        dataset_split: Dataset split to use
        num_samples: Number of samples to evaluate on
        text_field: Name of the text field in the dataset

    Returns:
        Dictionary with all metrics
    """
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Loading dataset: {dataset_name} ({dataset_split} split)")
    dataset = load_dataset(dataset_name, split=dataset_split)

    # Take a subset
    if num_samples is not None and len(dataset) > num_samples:
        dataset = dataset.select(range(num_samples))

    print(f"Processing {len(dataset)} samples...")
    texts = [sample[text_field] for sample in dataset if sample[text_field]]

    print("Tokenizing texts (once)...")
    tokenized_texts = tokenize_texts(texts, tokenizer)

    print("Calculating metrics...")

    # Calculate all metrics using pre-tokenized data
    fertility_rate = calculate_fertility_rate(tokenized_texts)
    compression_ratio = calculate_compression_ratio(tokenized_texts)
    vocab_util, unique_tokens, vocab_size = calculate_vocabulary_utilization(tokenized_texts, tokenizer.vocab_size)
    gini = calculate_token_distribution_gini(tokenized_texts)

    results = {
        "fertility_rate": fertility_rate,
        "compression_ratio": compression_ratio,
        "vocabulary_utilization_%": vocab_util,
        "unique_tokens_used": unique_tokens,
        "vocab_size": vocab_size,
        "gini_coefficient": gini,
        "num_samples": len(texts),
    }

    return results


def print_results(results: Dict[str, float], model_name: str):
    """Pretty print the evaluation results."""
    print("\n" + "="*60)
    print(f"Tokenizer Evaluation Results: {model_name}")
    print("="*60)
    print(f"Samples evaluated: {results['num_samples']}")
    print(f"\nFertility Rate: {results['fertility_rate']:.4f} tokens/word")
    print(f"  (Lower is better - fewer tokens per word)")
    print(f"\nCompression Ratio: {results['compression_ratio']:.4f} words/token")
    print(f"  (Higher is better - more words per token)")
    print(f"\nVocabulary Utilization: {results['vocabulary_utilization_%']:.2f}%")
    print(f"  ({results['unique_tokens_used']:,} / {results['vocab_size']:,} tokens used)")
    print(f"\nToken Distribution Gini: {results['gini_coefficient']:.4f}")
    print(f"  (0 = equal token usage, 1 = unequal token usage)")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Example usage with Danish dataset
    # MODEL_NAME = "mistralai/Mistral-Nemo-Base-2407"  # Replace with your desired model
    DATASET_NAME = "danish-foundation-models/danish-dynaword"
    NUM_SAMPLES = 1000

    # # Evaluate the tokenizer
    # results = evaluate_tokenizer(
    #     model_name=MODEL_NAME,
    #     dataset_name=DATASET_NAME,
    #     dataset_split="train",
    #     num_samples=NUM_SAMPLES,
    #     text_field="text"  # Adjust if the dataset uses a different field name
    # )

    # # Print results
    # print_results(results, MODEL_NAME)

    # Example: Compare multiple tokenizers
    models_to_compare = [
        "google/gemma-7b",
        "deepseek-ai/DeepSeek-R1",
        "PleIAs/Pleias-RAG-1B",
        "common-pile/comma-v0.1-1t",
        "allenai/FlexOlmo-7x7B-1T",
        "mistralai/Mistral-Nemo-Base-2407",
    ]
    
    for model in models_to_compare:
        results = evaluate_tokenizer(model, DATASET_NAME, dataset_split="train", num_samples=NUM_SAMPLES, text_field="text")
        print_results(results, model)