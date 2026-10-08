import os
import sys
import time
import platform
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("[ERROR] ONNX Runtime is not installed. Run: pip install onnxruntime")
    sys.exit(1)

try:
    import psutil
except ImportError:
    psutil = None

CLASS_NAMES = ["Normal (Healthy)", "Inner Race Fault", "Ball Fault", "Outer Race Fault"]
# Static (QDQ) INT8: ~18x faster than the dynamic INT8 model on x86 with identical accuracy
MODEL_PATH = "results/physics_model_int8_static.onnx"
# Small labelled set of impaired 3 HP test windows, written by export_edge.py and kept in git
SAMPLES_PATH = "results/sample_windows.npz"
SAMPLING_WINDOW_MS = 1024 / 12000 * 1000.0  # 1024 points @ 12 kHz
NUM_THREADS = 4


def rss_mb():
    return psutil.Process().memory_info().rss / 1e6 if psutil else float("nan")


def run_pi_edge_test():
    print("=" * 75)
    print(" EDGE BEARING FAULT INFERENCE (ONNX Runtime)")
    print(f" Host: {platform.machine()} | {platform.processor() or platform.platform()}")
    print("=" * 75)

    for path in (MODEL_PATH, SAMPLES_PATH):
        if not os.path.exists(path):
            print(f"[ERROR] {path} not found. Run export_edge.py on the training machine and copy results/ over.")
            return

    mem_before = rss_mb()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = NUM_THREADS
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(MODEL_PATH, sess_options=opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    samples = np.load(SAMPLES_PATH)
    X, y = samples["X"].astype(np.float32), samples["y"]

    for _ in range(20):
        session.run(None, {input_name: X[:1]})

    print(f"\n[OK] Model: {MODEL_PATH} ({os.path.getsize(MODEL_PATH)/1024.0:.1f} KB)")
    if psutil:
        print(f"[OK] Process memory after model load: {rss_mb():.1f} MB (+{rss_mb() - mem_before:.1f} MB for the session)")

    print("\n[STREAM] Processing labelled 3 HP test windows one at a time:")
    print("-" * 75)
    print(f"{'Sample #':<10} | {'True Condition':<20} | {'Edge Prediction':<20} | {'Latency':<10} | {'Status'}")
    print("-" * 75)

    latencies, correct = [], 0
    for i in range(len(X)):
        t0 = time.perf_counter()
        logits, _ = session.run(None, {input_name: X[i:i + 1]})
        latency_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(latency_ms)
        pred = int(np.argmax(logits[0]))
        correct += pred == y[i]
        status = "[OK] MATCH" if pred == y[i] else "[X] MISMATCH"
        print(f"#{i+1:<9} | {CLASS_NAMES[y[i]]:<20} | {CLASS_NAMES[pred]:<20} | {latency_ms:.3f} ms  | {status}")

    mean_lat = float(np.mean(latencies))
    print("-" * 75)
    print("\n[EDGE BENCHMARK SUMMARY]")
    print(f"  • Mean / P95 latency:       {mean_lat:.3f} / {np.percentile(latencies, 95):.3f} ms per 1024-pt window")
    print(f"  • Physical sensor window:   {SAMPLING_WINDOW_MS:.2f} ms")
    print(f"  • Real-time budget used:    {mean_lat / SAMPLING_WINDOW_MS * 100.0:.2f}% of wall-clock time "
          f"({NUM_THREADS} threads)")
    print(f"  • Sample accuracy:          {correct / len(X) * 100.0:.1f}% on {len(X)} windows "
          f"(small sample; see results/metrics.json for full-test-set figures)\n")
    print("=" * 75)


if __name__ == "__main__":
    run_pi_edge_test()
