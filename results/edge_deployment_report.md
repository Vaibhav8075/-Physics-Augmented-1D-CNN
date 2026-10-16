# Edge Deployment Benchmark

## 1. Summary

Measured on `Intel64 Family 6 Model 183 Stepping 1, GenuineIntel`. ONNX Runtime uses 1 thread; PyTorch/TorchScript use default threading. Batch size 1, 50 warm-up runs, 1,000 timed runs. Accuracy is on the full impaired 1894-window test set. The fastest ONNX variant here is **ONNX Runtime INT8 Static** at 0.299 ms.

## 2. Quantitative Performance Matrix

| Runtime Engine | Precision | Model Size (KB) | Mean Latency (ms) | P95 (ms) | Throughput (windows/s) | Test Accuracy | Acc. vs ORT FP32 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| PyTorch FP32 | FP32 | 401.1 | 1.887 | 2.209 | 530 | 79.30% | +0.00 pp |
| TorchScript | FP32 | 441.5 | 1.521 | 1.906 | 657 | 79.30% | +0.00 pp |
| ONNX Runtime FP32 | FP32 | 386.8 | 0.434 | 0.521 | 2,305 | 79.30% | +0.00 pp |
| ONNX Runtime INT8 Dynamic | INT8 | 120.4 | 5.347 | 5.772 | 187 | 78.88% | -0.42 pp |
| ONNX Runtime INT8 Static | INT8 | 132.5 | 0.299 | 0.408 | 3,347 | 79.09% | -0.21 pp |

## 3. Real-Time Feasibility

- **Sampling window**: 1024 points @ 12 kHz = 85.33 ms.
- **ONNX Runtime FP32**: 0.434 ms = 0.51% of the window period; 1.00x the FP32 latency.
- **ONNX Runtime INT8 Dynamic**: 5.347 ms = 6.27% of the window period; 12.33x the FP32 latency.
- **ONNX Runtime INT8 Static**: 0.299 ms = 0.35% of the window period; 0.69x the FP32 latency.
- These are host-PC figures. Run `rpi_edge_diagnostic.py` on the target board for edge numbers.
