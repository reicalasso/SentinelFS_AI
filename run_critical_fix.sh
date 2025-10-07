#!/bin/bash
# Quick Start Script for Critical Fix
# Runs the fixed training pipeline with comprehensive diagnostics

echo "=========================================="
echo "SentinelFS AI - Critical Fix Quick Start"
echo "=========================================="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ ERROR: Python 3 not found"
    exit 1
fi

echo "✓ Python 3 found: $(python3 --version)"

# Check if required packages are installed
echo ""
echo "Checking dependencies..."

python3 -c "import torch; print('✓ PyTorch:', torch.__version__)" 2>/dev/null || echo "❌ PyTorch not installed"
python3 -c "import numpy; print('✓ NumPy:', numpy.__version__)" 2>/dev/null || echo "❌ NumPy not installed"
python3 -c "import sklearn; print('✓ scikit-learn:', sklearn.__version__)" 2>/dev/null || echo "❌ scikit-learn not installed"
python3 -c "import matplotlib; print('✓ matplotlib:', matplotlib.__version__)" 2>/dev/null || echo "❌ matplotlib not installed (will install)"

# Check if matplotlib is missing
if ! python3 -c "import matplotlib" 2>/dev/null; then
    echo ""
    echo "Installing missing dependency: matplotlib..."
    pip install matplotlib>=3.7.0
fi

# Check CUDA availability
echo ""
echo "Checking CUDA availability..."
python3 -c "import torch; print('✓ CUDA Available:', torch.cuda.is_available()); print('  Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')" 2>/dev/null

# Create output directories
echo ""
echo "Creating output directories..."
mkdir -p ./metrics
mkdir -p ./diagnostics
mkdir -p ./models/production
echo "✓ Directories created"

# Run the fixed training script
echo ""
echo "=========================================="
echo "Running Fixed Training Pipeline"
echo "=========================================="
echo ""
echo "This will:"
echo "  1. Generate realistic test data with verified threat signatures"
echo "  2. Perform adversarial validation to check distribution match"
echo "  3. Train model with real GPU monitoring"
echo "  4. Calibrate threshold using ROC/PR curves"
echo "  5. Test with comprehensive diagnostics"
echo "  6. Generate detailed reports and visualizations"
echo ""
echo "Expected runtime: 5-15 minutes (depending on hardware)"
echo ""
read -p "Press Enter to continue or Ctrl+C to abort..."

python3 train_rm_rtx5060_fixed.py

# Check if training was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Training Complete!"
    echo "=========================================="
    echo ""
    echo "📊 Output Files:"
    echo "  - ./metrics/roc_curve_*.png - ROC curve analysis"
    echo "  - ./metrics/pr_curve_*.png - Precision-Recall curve"
    echo "  - ./metrics/threshold_metrics_*.png - Threshold optimization"
    echo "  - ./diagnostics/predictions_*.json - Detailed prediction logs"
    echo "  - ./diagnostics/score_distribution_*.png - Score distributions"
    echo "  - ./models/production/sentinelfs_fixed.pt - Saved model"
    echo ""
    echo "📖 Review Results:"
    echo "  1. Check detection rate (should be >90%)"
    echo "  2. Verify GPU utilization was >0%"
    echo "  3. Confirm adversarial validation AUC <0.65"
    echo "  4. Review score separation between threats/benign"
    echo ""
    echo "📋 Next Steps:"
    echo "  - Read CRITICAL_FIX_DOCUMENTATION.md for details"
    echo "  - Review diagnostic outputs in ./diagnostics/"
    echo "  - Consider shadow mode deployment"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "❌ Training Failed"
    echo "=========================================="
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check if all dependencies are installed: pip install -r requirements.txt"
    echo "  2. Verify CUDA is available if using GPU"
    echo "  3. Check error logs above"
    echo "  4. Ensure sufficient disk space for outputs"
    echo ""
    exit 1
fi
