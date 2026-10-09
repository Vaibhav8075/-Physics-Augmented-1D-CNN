# Edge Deployment Benchmark

## 1. Summary

Measured on `Intel64 Family 6 Model 183 Stepping 1, GenuineIntel`. ONNX Runtime uses 1 thread; PyTorch/TorchScript use default threading. Batch size 1, 50 warm-up runs, 1,000 timed runs. Accuracy is on the full impaired 1894-window test set. The fastest ONNX variant here is **ONNX Runtime INT8 Static** at 0.296 ms.

## 2. Quantitative Performance Matrix

| Runtime Engine | Precision | Model Size (KB) | Mean Latency (ms) | P95 (ms) | Throughput (windows/s) | Test Accuracy | Acc. vs ORT FP32 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| PyTorch FP32 | FP32 | 401.1 | 2.430 | 2.712 | 412 | 75.13% | +0.00 pp |
| TorchScript | FP32 | 441.5 | 1.888 | 2.140 | 530 | 75.13% | +0.00 pp |
| ONNX Runtime FP32 | FP32 | 386.8 | 0.421 | 0.486 | 2,375 | 75.13% | +0.00 pp |
| ONNX Runtime INT8 Dynamic | INT8 | 120.4 | 5.308 | 5.856 | 188 | 75.13% | +0.00 pp |
| ONNX Runtime INT8 Static | INT8 | 132.5 | 0.296 | 0.399 | 3,377 | 75.13% | +0.00 pp |

## 3. Real-Time Feasibility

- **Sampling window**: 1024 points @ 12 kHz = 85.33 ms.
- **ONNX Runtime FP32**: 0.421 ms = 0.49% of the window period; 1.00x the FP32 latency.
- **ONNX Runtime INT8 Dynamic**: 5.308 ms = 6.22% of the window period; 12.61x the FP32 latency.
- **ONNX Runtime INT8 Static**: 0.296 ms = 0.35% of the window period; 0.70x the FP32 latency.
- These are host-PC figures. Run `rpi_edge_diagnostic.py` on the target board for edge numbers.
