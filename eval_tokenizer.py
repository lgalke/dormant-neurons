"""
Tokenizer Evaluation Script
Calculates metrics for tokenizer performance on a dataset, following arXiv 2508.04796:
- Fertility Rate: Average number of tokens per word (whitespace-delimited)
- Compression Ratio: Average number of words per token (per-text averaging)
- Vocabulary Utilization: Percentage of vocabulary used
- Token Distribution Gini: Measure of inequality in token usage distribution
"""

import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizer
from datasets import Dataset, load_dataset
from collections import Counter
from pyarrow import compute as pc
from typing import Dict, List, Tuple
from tqdm import tqdm
import typer


def tokenize_texts(dataset: Dataset, tokenizer: PreTrainedTokenizer, num_proc: int) -> Dataset:
    """
    Tokenize all texts once and return with original texts.

    Args:
        dataset: HuggingFace dataset
        tokenizer: HuggingFace tokenizer
        num_proc: number of processes to use for mapping

    Returns:
        dataset with a new field with the tokens
    """
    def tokenize_batch(batch: dict) -> dict:
        batch["tokens"] = tokenizer(batch["text"], padding="do_not_pad", add_special_tokens=False)["input_ids"]
        batch["unique_tokens"] = [list(set(tokens)) for tokens in batch["tokens"]]
        batch["num_tokens"] = [len(tokens) for tokens in batch["tokens"]]
        batch["ratios"] = [num_words / num_tokens for num_words, num_tokens in zip(batch["num_words"], batch["num_tokens"])]
        return batch
    dataset = dataset.map(
        tokenize_batch,
        batched=True,
        batch_size=1+len(dataset) // num_proc,
        num_proc=num_proc,
    )
    return dataset


def calculate_fertility_rate(dataset: Dataset) -> float:
    """
    Calculate fertility rate: average number of tokens per word (whitespace-delimited).
    Lower is better (more efficient).

    As defined in arXiv 2508.04796: Fertility(T) = Σ|τ(b)| / Σ|b|_u
    where u is the normalization unit (words).

    Args:
        dataset: HuggingFace dataset with tokenized texts
    """
    total_tokens = pc.sum(dataset.with_format("arrow")["num_tokens"]).as_py()
    total_words  = pc.sum(dataset.with_format("arrow")["num_words"]).as_py()

    return total_tokens / total_words if total_words > 0 else 0


def calculate_compression_ratio(dataset: Dataset, num_proc: int) -> float:
    """
    Calculate compression ratio: average number of words per token across all texts.
    Higher is better (more efficient).

    As defined in arXiv 2508.04796: CR(D;τ) = (1/|D|) Σ |b|_u / |τ(b)|
    where u is the normalization unit (words) and the average is taken per text.

    Args:
        dataset: HuggingFace dataset with tokenized texts
    """
    return pc.mean(dataset.with_format("arrow")["ratios"]).as_py()


def calculate_vocabulary_utilization(dataset: Dataset, vocab_size: int) -> Tuple[float, int, int]:
    """
    Calculate vocabulary utilization: percentage of vocabulary used.
    Returns (utilization_percentage, unique_tokens_used, vocab_size)

    Args:
        dataset: HuggingFace dataset with tokenized texts
        vocab_size: Size of the tokenizer vocabulary
    """
    unique_tokens = len(pc.unique(pc.list_flatten(dataset.with_format("arrow")["unique_tokens"])))
    utilization = (unique_tokens / vocab_size) * 100

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


def evaluate_tokenizer(model_name: str, dataset: Dataset, num_proc: int) -> Dict[str, float]:
    """
    Evaluate a tokenizer on a dataset and return all metrics.

    Args:
        model_name: HuggingFace model name for tokenizer
        dataset: HuggingFace dataset with text and tokenized fields
        num_proc: number of processes to use for mapping
    Returns:
        Dictionary with all metrics
    """
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = 2**30

    print("Tokenizing texts (once) ...")
    dataset = tokenize_texts(dataset=dataset, tokenizer=tokenizer, num_proc=num_proc)

    # Calculate all metrics using pre-tokenized data
    pbar = tqdm(total=3, desc="Calculating metrics")
    fertility_rate = calculate_fertility_rate(dataset=dataset)
    pbar.update(1)
    compression_ratio = calculate_compression_ratio(dataset=dataset, num_proc=num_proc)
    pbar.update(1)
    vocab_util, unique_tokens, vocab_size = calculate_vocabulary_utilization(dataset=dataset, vocab_size=tokenizer.vocab_size)
    pbar.update(1)
    pbar.close()
    #gini = calculate_token_distribution_gini(tokenized_texts)

    results = {
        "fertility_rate": fertility_rate,
        "compression_ratio": compression_ratio,
        "vocabulary_utilization_%": vocab_util,
        "unique_tokens_used": unique_tokens,
        "vocab_size": vocab_size,
        #"gini_coefficient": gini,
        "num_samples": len(dataset  ),
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
    #print(f"\nToken Distribution Gini: {results['gini_coefficient']:.4f}")
    #print(f"  (0 = equal token usage, 1 = unequal token usage)")
    print("="*60 + "\n")


def prepare_dataset(dataset_name: str, dataset_split: str, num_samples: int, text_field: str, num_proc: int):
    print(f"{dataset_name=}, {dataset_split=}, {num_samples=}, {text_field=}, {num_proc=}")
    print(f"Loading dataset: {dataset_name} ({dataset_split} split)")
    dataset = load_dataset(dataset_name, split=dataset_split)
    print(len(dataset  ), "samples loaded.")
    dataset = dataset.remove_columns([col for col in dataset.column_names if col != text_field])
    print(len(dataset  ), "samples remaining after filtering.")
    print("Shuffling deterministically")
    dataset = dataset.shuffle(seed=42)
    if num_samples and len(dataset) > num_samples:
        print(f"Subsampling to {num_samples} samples")
        dataset = dataset.select(range(num_samples))
    dataset = dataset.flatten_indices(num_proc=num_proc)
    print(f"Filtering out empty texts")
    def filter_empty(batch: dict) -> bool:
        return [bool(text and text.strip()) for text in batch[text_field]]
    dataset = dataset.filter(filter_empty, batched=True, batch_size=1+len(dataset) // num_proc, num_proc=num_proc)
    print(f"Splitting texts into words")
    def split_into_words(batch: dict) -> dict:
        batch["words"] = [sample.split() for sample in batch["text"]]
        batch["num_words"] = [len(sample) for sample in batch["words"]]
        return batch
    dataset = dataset.map(split_into_words, batched=True, batch_size=1+len(dataset) // num_proc, num_proc=num_proc)
    return dataset

def main(
    models_to_compare: List[str] = typer.Argument([
        "google/gemma-7b",
        "deepseek-ai/DeepSeek-R1",
        "PleIAs/Pleias-RAG-1B",
        "common-pile/comma-v0.1-1t",
        "allenai/FlexOlmo-7x7B-1T",
        "mistralai/Mistral-Nemo-Base-2407",
    ]),
    dataset_name: str = "danish-foundation-models/danish-dynaword",
    dataset_split: str = "train",
    num_samples: int = 1000,
    text_field: str = "text",
    num_proc: int = 64,
):
    print(f"Loading and preparing dataset: {dataset_name} ({dataset_split} split)")
    dataset = prepare_dataset(dataset_name=dataset_name, dataset_split=dataset_split, num_samples=num_samples, text_field=text_field, num_proc=num_proc)
    print(len(dataset  ), "samples ready for evaluation.\n")
    for model_name in models_to_compare:
        results = evaluate_tokenizer(model_name=model_name, dataset=dataset, num_proc=num_proc)
        print_results(results=results, model_name=model_name)

if __name__ == "__main__":
    typer.run(main)
