# dormant-neurons

## Quick Start: Analyzing Pythia Checkpoints

### Run the Full Analysis

To analyze all Pythia-70m-dedup checkpoints from the first_results file:

```bash
./analyze_pythia_checkpoints.sh
```

This will:
- Analyze 10 different training checkpoints (steps 1000 through 143000)
- Log all results to `pythia_70m_full_analysis.csv`
- Take approximately 30-60 minutes depending on your hardware

### Visualize the Results

Once the analysis is complete, visualize the results:

```bash
python visualize_logged_results.py pythia_70m_full_analysis.csv --save-prefix pythia_
```

This creates three visualizations:
1. **Dormancy over training steps** - Shows average, min, and max dormancy
2. **Layer heatmap** - Shows dormancy per layer across checkpoints
3. **Layer evolution** - Tracks the most variable layers over time

## Logging Functionality

The `analyze_dormancy.py` script now includes comprehensive logging capabilities that write all analysis results to a CSV file with the following information:

### Logged Information

- **timestamp**: When the analysis was run
- **model**: Model name (e.g., "EleutherAI/pythia-70m-dedup")
- **revision**: Model checkpoint/revision (e.g., "step1000")
- **step**: Training step number
- **threshold**: Dormancy threshold used
- **layer_name**: Full name of the analyzed layer
- **layer_index**: Sequential index of the layer
- **num_neurons**: Total number of neurons in the layer
- **num_dormant**: Number of dormant neurons
- **pct_dormant**: Percentage of dormant neurons
- **avg_activation**: Average activation value across the layer
- **max_activation**: Maximum activation value in the layer

### Usage Examples

**Basic usage with automatic log file:**

```bash
python analyze_dormancy.py \
    --model gpt2 \
    --dataset wikitext \
    --dataset-config wikitext-2-raw-v1 \
    --num-samples 500
```

This will create a log file like `dormancy_results_gpt2_20251106_143022.csv`

**With custom log file and append mode:**

```bash
python analyze_dormancy.py \
    --model EleutherAI/pythia-70m-dedup \
    --revision step1000 \
    --step 1000 \
    --dataset pile \
    --num-samples 500 \
    --log-file pythia_analysis.csv \
    --log-mode append
```

**Analyzing multiple checkpoints (append to same file):**

```bash
for step in 1000 14000 46000 78000; do
    python analyze_dormancy.py \
        --model EleutherAI/pythia-70m-dedup \
        --revision step${step} \
        --step ${step} \
        --num-samples 500 \
        --log-file pythia_checkpoints.csv \
        --log-mode append
done
```

### Log File Format

The CSV file will have the following structure:

```csv
timestamp,model,revision,step,threshold,layer_name,layer_index,num_neurons,num_dormant,pct_dormant,avg_activation,max_activation
2025-11-06T14:30:22,gpt2,N/A,N/A,0.1,transformer.h.0.mlp.c_fc,0,3072,245,7.9753,0.123456,2.345678
2025-11-06T14:30:22,gpt2,N/A,N/A,0.1,transformer.h.1.mlp.c_fc,1,3072,312,10.1563,0.098765,1.876543
...
```

### Command-Line Arguments

- `--log-file`: Specify output CSV file path (optional, auto-generated if not provided)
- `--log-mode`: Choose 'append' (default) or 'write' mode
- `--step`: Specify training step number for logging
- `--revision`: Specify model revision/checkpoint

