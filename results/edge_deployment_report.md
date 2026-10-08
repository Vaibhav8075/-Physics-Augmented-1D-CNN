# Edge Deployment Benchmark

## 1. Summary

Measured on `Intel64 Family 6 Model 183 Stepping 1, GenuineIntel`. ONNX Runtime uses 1 thread; PyTorch/TorchScript use default threading. Batch size 1, 50 warm-up runs, 1,000 timed runs. Accuracy is on the full impaired 3317-window test set. The fastest ONNX variant here is **ONNX Runtime INT8 Static** at 0.293 ms.

## 2. Quantitative Performance Matrix

| Runtime Engine | Precision | Model Size (KB) | Mean Latency (ms) | P95 (ms) | Throughput (windows/s) | Test Accuracy | Acc. vs ORT FP32 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| PyTorch FP32 | FP32 | 401.1 | 1.688 | 1.878 | 592 | 85.80% | +0.00 pp |
| TorchScript | FP32 | 441.5 | 1.431 | 1.564 | 699 | 85.80% | +0.00 pp |
| ONNX Runtime FP32 | FP32 | 386.8 | 0.419 | 0.493 | 2,388 | 85.80% | +0.00 pp |
| ONNX Runtime INT8 Dynamic | INT8 | 120.4 | 5.206 | 5.567 | 192 | 85.80% | +0.00 pp |
| ONNX Runtime INT8 Static | INT8 | 132.5 | 0.293 | 0.393 | 3,413 | 85.80% | +0.00 pp |

## 3. Real-Time Feasibility

- **Sampling window**: 1024 points @ 12 kHz = 85.33 ms.
- **ONNX Runtime FP32**: 0.419 ms = 0.49% of the window period; 1.00x the FP32 latency.
- **ONNX Runtime INT8 Dynamic**: 5.206 ms = 6.10% of the window period; 12.43x the FP32 latency.
- **ONNX Runtime INT8 Static**: 0.293 ms = 0.34% of the window period; 0.70x the FP32 latency.
- These are host-PC figures. Run `rpi_edge_diagnostic.py` on the target board for edge numbers.
