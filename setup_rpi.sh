#!/bin/bash
# =========================================================================
# Raspberry Pi 5 (8GB) Setup & Installation Script for Industrial Fault AI
# =========================================================================

echo "========================================================================="
echo " Installing Industrial Bearing AI on Raspberry Pi 5 (8GB RAM)"
echo " Processor: Broadcom BCM2712 Quad-Core Cortex-A76 @ 2.4 GHz"
echo "========================================================================="

# 1. Update system packages
sudo apt update && sudo apt install -y python3-pip python3-numpy python3-matplotlib python3-scipy git

# 2. Install lightweight edge runtime
pip3 install onnxruntime streamlit torch torchvision --break-system-packages

echo ""
echo "========================================================================="
echo " [OK] Installation Complete! You can now run:"
echo " 1. Terminal Real-Time Diagnostic Test:  python3 rpi_edge_diagnostic.py"
echo " 2. Web Visual Diagnostic Dashboard:    streamlit run app.py --server.port 8501"
echo "========================================================================="
