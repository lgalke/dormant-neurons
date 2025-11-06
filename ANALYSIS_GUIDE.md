# Pythia Dormancy Analysis Guide

## Overview

This guide will help you analyze dormant neurons across multiple Pythia-70m-dedup checkpoints and visualize the results.

## Files Created

1. **`analyze_pythia_checkpoints.sh`** - Main analysis script
   - Analyzes 10 training checkpoints (1000 to 143000)
   - Logs all results to CSV
   - Takes ~30-60 minutes to run

2. **`quick_analysis_test.sh`** - Quick test script
   - Tests only 2 checkpoints with reduced samples
   - Verifies everything works before full analysis
   - Takes ~5-10 minutes

3. **`visualize_logged_results.py`** - Visualization script
   - Creates 3 types of plots from logged results
   - Can save plots as PNG files
   - Prints summary statistics

## Quick Start

### Step 1: Test the Setup (Optional but Recommended)

```bash
./quick_analysis_test.sh
```

This will:
- Analyze 2 checkpoints (step1000, step14000)
- Use only 100 samples per checkpoint
- Create `test_pythia_analysis.csv`
- Verify everything is working

### Step 2: Run Full Analysis

```bash
./analyze_pythia_checkpoints.sh
```

This will:
- Analyze 10 checkpoints from the Pythia training
- Use 500 samples per checkpoint
- Create `pythia_70m_full_analysis.csv`
- Take approximately 30-60 minutes

**What happens during analysis:**
- Downloads/loads model checkpoints from HuggingFace
- Processes 500 samples from the Pile dataset per checkpoint
- Computes dormancy metrics for each layer
- Appends results to CSV file

### Step 3: Visualize Results

```bash
python visualize_logged_results.py pythia_70m_full_analysis.csv --save-prefix pythia_
```

This creates:
1. `pythia_dormancy_over_steps.png` - Overall trend plot
2. `pythia_dormancy_heatmap.png` - Layer-wise heatmap
3. `pythia_layer_evolution.png` - Evolution of variable layers

## Understanding the Results

### CSV Format

Each row in the output CSV contains:
- `timestamp` - When the analysis ran
- `model` - Model name
- `revision` - Checkpoint (e.g., "step1000")
- `step` - Training step number
- `threshold` - Dormancy threshold (0.1)
- `layer_name` - Full layer name
- `layer_index` - Sequential layer number
- `num_neurons` - Total neurons in layer
- `num_dormant` - Number of dormant neurons
- `pct_dormant` - Percentage dormant
- `avg_activation` - Average activation magnitude
- `max_activation` - Maximum activation magnitude

### Key Metrics

- **Average Dormancy**: Mean dormancy % across all layers at a checkpoint
- **Min/Max Dormancy**: Range showing layer variation
- **Layer Evolution**: How specific layers change over training

## Customization

### Modify the Analysis

Edit `analyze_pythia_checkpoints.sh` to change:

```bash
NUM_SAMPLES=500      # Increase for more accurate results
THRESHOLD=0.1        # Try 0.025 for stricter dormancy
BATCH_SIZE=8         # Increase if you have more GPU memory
STEPS=(...)          # Add/remove checkpoints to analyze
```

### Different Visualizations

The visualization script has options:

```bash
# Show more layers in heatmap
python visualize_logged_results.py results.csv --max-layers-heatmap 30

# Track more layers in evolution plot
python visualize_logged_results.py results.csv --num-layers-evolution 10

# Save plots without showing
python visualize_logged_results.py results.csv --save-prefix plots/ --no-show
```

## Troubleshooting

### Out of Memory
- Reduce `BATCH_SIZE` in the analysis script
- Reduce `NUM_SAMPLES`
- Use CPU instead of GPU (slower): add `--device cpu`

### Dataset Download Issues
- The Pile dataset is large; first run may take time
- Ensure you have stable internet connection
- Check HuggingFace authentication if needed

### Model Checkpoint Not Found
- Verify checkpoint names match Pythia naming: "step1000", "step14000", etc.
- Check https://huggingface.co/EleutherAI/pythia-70m-dedup for available revisions

## Expected Timeline

- **Quick test**: ~5-10 minutes
- **Full analysis (10 checkpoints)**: ~30-60 minutes
- **Visualization**: ~1-2 minutes

## Next Steps

After getting results, you can:
1. Compare with the original `first_results_Pythia-70m-dedup`
2. Analyze different thresholds (0, 0.025, 0.1)
3. Test on different model sizes (pythia-160m, pythia-410m, etc.)
4. Investigate specific layers showing high/low dormancy
5. Correlate dormancy with model performance metrics

## Questions or Issues?

Check the logs in the terminal output for detailed error messages and progress updates.
