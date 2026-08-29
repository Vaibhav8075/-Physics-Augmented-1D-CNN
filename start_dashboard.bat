@echo off
title Industrial Fault AI - Live Diagnostic Console
cd /d "C:\Users\vibhu\industrial_fault_ai"
echo =========================================================================
echo Starting Physics-Augmented 1D-CNN Real-Time Diagnostic Dashboard...
echo =========================================================================
python -m streamlit run app.py
pause
