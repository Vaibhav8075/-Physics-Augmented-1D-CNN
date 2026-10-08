import os
import sys
import time
import platform

# Set utf-8 encoding for standard streams
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (quantize_dynamic, quantize_static, QuantType, QuantFormat,
                                      CalibrationDataReader)

import config
from models import PhysicsAugmentedCalibratedCNN

RESULTS_DIR = config.RESULTS_DIR
PROCESSED_DIR = config.PROCESSED_DIR
WINDOW_MS = config.WINDOW_SIZE / config.SAMPLING_RATE_HZ * 1000.0


class ExportablePhysicsCNN(nn.Module):
    """Wrapper to output a clean, standard (logits, temporal_attention) tuple for ONNX."""
    def __init__(self, model):
        super(ExportablePhysicsCNN, self).__init__()
        self.model = model

    def forward(self, x):
        calibrated_logits, attn_info, _ = self.model(x, return_calibrated=True)
        return calibrated_logits, attn_info["temporal_attention"]


class ValidationWindowReader(CalibrationDataReader):
    """Feeds validation windows to static INT8 calibration (never test data)."""
    def __init__(self, X, input_name, n=256):
        idx = np.random.default_rng(0).choice(len(X), size=min(n, len(X)), replace=False)
        self.batches = iter([{input_name: X[i:i + 1]} for i in idx])

    def get_next(self):
        return next(self.batches, None)


def write_sample_windows(X_test, y_test, per_class=5):
    """Small stratified labelled sample for the Raspberry Pi demo (committed to git)."""
    rng = np.random.default_rng(0)
    idx = np.concatenate([rng.choice(np.where(y_test == c)[0], per_class, replace=False)
                          for c in np.unique(y_test)])
    rng.shuffle(idx)
    path = os.path.join(RESULTS_DIR, "sample_windows.npz")
    np.savez_compressed(path, X=X_test[idx].astype(np.float32), y=y_test[idx])
    print(f"[SAVED] Edge demo sample windows -> {path}")


def time_fn(fn, runs=1000, warmup=50):
    for _ in range(warmup):
        fn()
    lat = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        lat.append((time.perf_counter() - t0) * 1000.0)
    return lat


def bench_ort(path, X_eval, y_eval, dummy_np):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1  # single-thread numbers are closest to one small edge CPU core
    sess = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    lat = time_fn(lambda: sess.run(None, {name: dummy_np}))
    logits = sess.run(None, {name: X_eval})[0]
    return lat, np.mean(np.argmax(logits, axis=1) == y_eval) * 100.0


@torch.no_grad()
def bench_torch(model, X_eval, y_eval, dummy_input):
    lat = time_fn(lambda: model(dummy_input))
    preds = [torch.argmax(model(torch.from_numpy(X_eval[i:i + 256]))[0], dim=1).numpy()
             for i in range(0, len(X_eval), 256)]
    return lat, np.mean(np.concatenate(preds) == y_eval) * 100.0


def export_and_benchmark():
    print("=" * 75)
    print(" INDUSTRIAL FAULT AI: EDGE ONNX & INT8 QUANTIZATION BENCHMARK")
    print("=" * 75)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Load Pre-trained PyTorch Model
    model_path = os.path.join(RESULTS_DIR, "physics_model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}. Train the model first.")

    model = PhysicsAugmentedCalibratedCNN(in_channels=2, num_classes=4)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    export_model = ExportablePhysicsCNN(model).eval()

    X_test = np.load(os.path.join(PROCESSED_DIR, "X_test.npy")).astype(np.float32)
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))
    X_val = np.load(os.path.join(PROCESSED_DIR, "X_val.npy")).astype(np.float32)
    write_sample_windows(X_test, y_test)

    dummy_input = torch.randn(1, 2, config.WINDOW_SIZE, dtype=torch.float32)

    # 2. EXPORT TO ONNX (FP32)
    onnx_fp32_path = os.path.join(RESULTS_DIR, "physics_model_fp32.onnx")
    print(f"\n[1/4] Exporting to ONNX (FP32) -> {onnx_fp32_path}...")
    torch.onnx.export(
        export_model,
        dummy_input,
        onnx_fp32_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
        input_names=["sensor_stream"],
        output_names=["fault_logits", "temporal_attention"],
        dynamic_axes={
            "sensor_stream": {0: "batch_size"},
            "fault_logits": {0: "batch_size"},
            "temporal_attention": {0: "batch_size"}
        }
    )
    onnx.checker.check_model(onnx.load(onnx_fp32_path))
    print("  [OK] ONNX FP32 model verified successfully.")

    # 3. EXPORT TO TORCHSCRIPT (C++ Embedded / LibTorch)
    ts_path = os.path.join(RESULTS_DIR, "physics_model_traced.pt")
    print(f"\n[2/4] Exporting to TorchScript -> {ts_path}...")
    traced_script = torch.jit.trace(export_model, dummy_input)
    traced_script.save(ts_path)
    print("  [OK] TorchScript traced model saved successfully.")

    # 4. INT8 QUANTIZATION: dynamic (weights only) and static QDQ (weights + activations)
    onnx_int8_path = os.path.join(RESULTS_DIR, "physics_model_int8.onnx")
    onnx_int8_static_path = os.path.join(RESULTS_DIR, "physics_model_int8_static.onnx")
    print(f"\n[3/4] INT8 dynamic quantization -> {onnx_int8_path}...")
    quantize_dynamic(model_input=onnx_fp32_path, model_output=onnx_int8_path, weight_type=QuantType.QInt8)
    print(f"      INT8 static QDQ quantization (calibrated on validation windows) -> {onnx_int8_static_path}...")
    quantize_static(
        model_input=onnx_fp32_path,
        model_output=onnx_int8_static_path,
        calibration_data_reader=ValidationWindowReader(X_val, "sensor_stream"),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
    )
    print("  [OK] INT8 models generated.")

    # 5. BENCHMARK: batch-1 latency, size, full-test-set accuracy
    print("\n[4/4] Benchmarking (batch size 1, 50 warm-up + 1,000 timed runs each)...")
    dummy_np = dummy_input.numpy()
    measured = [
        ("PyTorch FP32", model_path, *bench_torch(export_model, X_test, y_test, dummy_input)),
        ("TorchScript", ts_path, *bench_torch(traced_script, X_test, y_test, dummy_input)),
        ("ONNX Runtime FP32", onnx_fp32_path, *bench_ort(onnx_fp32_path, X_test, y_test, dummy_np)),
        ("ONNX Runtime INT8 Dynamic", onnx_int8_path, *bench_ort(onnx_int8_path, X_test, y_test, dummy_np)),
        ("ONNX Runtime INT8 Static", onnx_int8_static_path, *bench_ort(onnx_int8_static_path, X_test, y_test, dummy_np)),
    ]
    results = [{"Format": fmt, "Size_KB": os.path.getsize(path) / 1024.0, "Mean_Latency_ms": np.mean(lat),
                "P95_ms": np.percentile(lat, 95), "FPS": 1000.0 / np.mean(lat), "Accuracy": acc}
               for fmt, path, lat, acc in measured]

    print("\n" + "=" * 100)
    print(f"{'Runtime Engine':<28} | {'Size':<10} | {'Latency':<12} | {'P95':<12} | {'Throughput':<18} | {'Accuracy':<8}")
    print("=" * 100)
    for r in results:
        print(f"{r['Format']:<28} | {r['Size_KB']:.1f} KB | {r['Mean_Latency_ms']:.3f} ms | {r['P95_ms']:.3f} ms | "
              f"{r['FPS']:,.0f} windows/s | {r['Accuracy']:.2f}%")
    print("=" * 100)

    # 6. VISUAL EDGE BENCHMARK PLOT
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    names = [r["Format"].replace("ONNX Runtime ", "ORT\n") for r in results]
    colors = ["#64748b", "#0284c7", "#0ea5e9", "#10b981", "#059669"]
    panels = [("Mean_Latency_ms", "Batch-1 Latency (ms) [Lower is Better]", "{:.3f}"),
              ("Size_KB", "Model Footprint (KB) [Lower is Better]", "{:.0f}"),
              ("FPS", "Throughput (Windows/Sec) [Higher is Better]", "{:,.0f}")]
    for ax, (key, title, label_fmt) in zip(axes, panels):
        vals = [r[key] for r in results]
        bars = ax.bar(names, vals, color=colors, width=0.55)
        ax.set_title(title, fontweight="bold", fontsize=11)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), label_fmt.format(v),
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "edge_benchmark_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\n[SAVED] Edge Benchmark Chart -> {plot_path}")

    # 7. MARKDOWN REPORT (all statements derived from the measurements)
    fp32 = results[2]
    best = min(results[2:], key=lambda r: r["Mean_Latency_ms"])
    report_path = os.path.join(RESULTS_DIR, "edge_deployment_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Edge Deployment Benchmark\n\n")
        f.write("## 1. Summary\n\n")
        f.write(f"Measured on `{platform.processor() or platform.machine()}`. ONNX Runtime uses 1 thread; "
                f"PyTorch/TorchScript use default threading. Batch size 1, 50 warm-up runs, 1,000 timed runs. "
                f"Accuracy is on the full impaired {len(y_test)}-window test set. "
                f"The fastest ONNX variant here is **{best['Format']}** at {best['Mean_Latency_ms']:.3f} ms.\n\n")
        f.write("## 2. Quantitative Performance Matrix\n\n")
        f.write("| Runtime Engine | Precision | Model Size (KB) | Mean Latency (ms) | P95 (ms) | Throughput (windows/s) | Test Accuracy | Acc. vs ORT FP32 |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for r in results:
            f.write(f"| {r['Format']} | {'INT8' if 'INT8' in r['Format'] else 'FP32'} | {r['Size_KB']:.1f} | "
                    f"{r['Mean_Latency_ms']:.3f} | {r['P95_ms']:.3f} | {r['FPS']:,.0f} | {r['Accuracy']:.2f}% | "
                    f"{r['Accuracy'] - fp32['Accuracy']:+.2f} pp |\n")
        f.write("\n## 3. Real-Time Feasibility\n\n")
        f.write(f"- **Sampling window**: {config.WINDOW_SIZE} points @ {config.SAMPLING_RATE_HZ // 1000} kHz "
                f"= {WINDOW_MS:.2f} ms.\n")
        for r in results[2:]:
            f.write(f"- **{r['Format']}**: {r['Mean_Latency_ms']:.3f} ms = "
                    f"{r['Mean_Latency_ms'] / WINDOW_MS * 100:.2f}% of the window period; "
                    f"{r['Mean_Latency_ms'] / fp32['Mean_Latency_ms']:.2f}x the FP32 latency.\n")
        f.write("- These are host-PC figures. Run `rpi_edge_diagnostic.py` on the target board for edge numbers.\n")
    print(f"[SAVED] Edge Deployment Markdown Report -> {report_path}")


if __name__ == "__main__":
    export_and_benchmark()
