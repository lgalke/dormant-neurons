#!/bin/bash

# Quick test script - analyzes just 2 checkpoints to verify everything works
# Run this before the full analysis to catch any errors early

set -e

echo "=================================================="
echo "Quick Test: Analyzing 2 Pythia Checkpoints"
echo "=================================================="

MODEL="EleutherAI/pythia-70m-dedup"
DATASET="EleutherAI/pile"
NUM_SAMPLES=100  # Reduced for quick test
THRESHOLD=0.1
LOG_FILE="test_pythia_analysis.csv"

echo ""
echo "Configuration:"
echo "  Model: $MODEL"
echo "  Dataset: $DATASET"
echo "  Samples: $NUM_SAMPLES (reduced for testing)"
echo "  Checkpoints: step1000, step14000"
echo "  Output: $LOG_FILE"
echo ""

# Test checkpoint 1
echo "Testing checkpoint: step1000"
python analyze_dormancy.py \
    --model "$MODEL" \
    --revision "step1000" \
    --step 1000 \
    --dataset "$DATASET" \
    --num-samples "$NUM_SAMPLES" \
    --threshold "$THRESHOLD" \
    --log-file "$LOG_FILE" \
    --log-mode write

echo ""
echo "✓ First checkpoint successful!"
echo ""

# Test checkpoint 2
echo "Testing checkpoint: step14000"
python analyze_dormancy.py \
    --model "$MODEL" \
    --revision "step14000" \
    --step 14000 \
    --dataset "$DATASET" \
    --num-samples "$NUM_SAMPLES" \
    --threshold "$THRESHOLD" \
    --log-file "$LOG_FILE" \
    --log-mode append

echo ""
echo "✓ Second checkpoint successful!"
echo ""
echo "=================================================="
echo "Test Complete!"
echo "=================================================="
echo ""
echo "Results in: $LOG_FILE"
wc -l "$LOG_FILE"
echo ""
echo "If this looks good, you can run the full analysis with:"
echo "  ./analyze_pythia_checkpoints.sh"
echo ""
echo "Or visualize these test results with:"
echo "  python visualize_logged_results.py $LOG_FILE"
echo "=================================================="
