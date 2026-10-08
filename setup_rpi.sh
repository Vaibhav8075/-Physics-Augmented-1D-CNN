#!/bin/bash
# =========================================================================
# Raspberry Pi 5 (8GB) Setup & Installation Script for Industrial Fault AI
# =========================================================================

echo "========================================================================="
echo " Installing Industrial Bearing AI on Raspberry Pi 5 (8GB RAM)"
echo " Processor: Broadcom BCM2712 Quad-Core Cortex-A76 @ 2.4 GHz"
echo "========================================================================="

# 1. Update system packages
sudo apt update && sudo apt install -y python3-pip python3-numpy git

# 2. Install the edge runtime (all rpi_edge_diagnostic.py needs is the ONNX model)
pip3 install onnxruntime psutil --break-system-packages

# Optional: the Streamlit dashboard additionally needs PyTorch and Streamlit.
if [ "$1" == "--with-dashboard" ]; then
    pip3 install torch streamlit matplotlib seaborn scikit-learn scipy --break-system-packages
fi

echo ""
echo "========================================================================="
echo " [OK] Installation Complete! You can now run:"
echo " 1. Terminal Real-Time Diagnostic Test:  python3 rpi_edge_diagnostic.py"
echo " 2. Web Dashboard (needs --with-dashboard): streamlit run app.py --server.port 8501"
echo "========================================================================="
