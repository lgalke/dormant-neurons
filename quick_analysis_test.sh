#!/bin/bash

# Quick test script - analyzes just 2 checkpoints to verify everything works
# Run this before the full analysis to catch any errors early

set -e


MODEL="common-pile/comma-v0.1-1t"
DATASET="danish-foundation-models/danish-dynaword"
NUM_SAMPLES=1000  # Reduced for quick test
THRESHOLD=0.1
LOG_FILE="dynaword.csv"

echo "=================================================="
echo "Quick Test: ${MODEL} on ${DATASET}"
echo "=================================================="

echo ""
echo "Configuration:"
echo "  Model: $MODEL"
echo "  Dataset: $DATASET"
echo "  Samples: $NUM_SAMPLES (reduced for testing)"
echo "  Output: $LOG_FILE"
echo ""

# Test checkpoint
echo "Testing: $MODEL"
python analyze_dormancy.py \
    --model "$MODEL" \
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
