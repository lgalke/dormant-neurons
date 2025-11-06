#!/bin/bash

# Analysis script for Pythia-70m-dedup across multiple training checkpoints
# This script analyzes dormancy at various training steps

set -e  # Exit on error

# Configuration
MODEL="EleutherAI/pythia-70m-deduped"
DATASET="Salesforce/wikitext"
DATASET_CONFIG="wikitext-2-raw-v1"
NUM_SAMPLES=1000
THRESHOLD=0.5
LOG_FILE="pythia_70m_full_analysis.csv"
BATCH_SIZE=8

# Training steps to analyze (from first_results_Pythia-70m-dedup)
STEPS=(1000 7000 14000 28000 46000 78000 94000 110000 126000 133000 138000 143000)

echo "=================================================="
echo "Pythia-70m-dedup Dormancy Analysis"
echo "=================================================="
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Samples per checkpoint: $NUM_SAMPLES"
echo "Threshold: $THRESHOLD"
echo "Output file: $LOG_FILE"
echo "Steps to analyze: ${STEPS[@]}"
echo "=================================================="
echo ""

# Check if log file exists and warn user
if [ -f "$LOG_FILE" ]; then
    echo "⚠️  WARNING: Log file '$LOG_FILE' already exists!"
    echo "   The script will APPEND new results to the existing file."
    echo "   This allows you to add missing checkpoints without losing existing data."
    echo ""
    echo "   If you want to start fresh, delete the file first:"
    echo "   rm $LOG_FILE"
    echo ""
    read -p "   Press Enter to continue with APPEND mode, or Ctrl+C to cancel..."
    echo ""
    LOG_MODE="append"
else
    echo "Creating new log file: $LOG_FILE"
    LOG_MODE="write"
fi

# Process all steps
for step in "${STEPS[@]}"; do
    echo "=================================================="
    echo "Analyzing checkpoint: step$step"
    echo "=================================================="
    
    python analyze_dormancy.py \
        --model "$MODEL" \
        --revision "step${step}" \
        --step "$step" \
        --dataset "$DATASET" \
        --dataset-config "$DATASET_CONFIG" \
        --num-samples "$NUM_SAMPLES" \
        --threshold "$THRESHOLD" \
        --batch-size "$BATCH_SIZE" \
        --log-file "$LOG_FILE" \
        --log-mode "$LOG_MODE"
    
    echo ""
    echo "✓ Checkpoint step$step completed!"
    echo ""
    
    # After first iteration, always use append mode
    LOG_MODE="append"
done

echo "=================================================="
echo "Analysis Complete!"
echo "=================================================="
echo "Results saved to: $LOG_FILE"
echo ""
echo "Quick statistics:"
wc -l "$LOG_FILE"
echo ""
echo "You can now visualize the results or analyze the CSV file."
echo "=================================================="
