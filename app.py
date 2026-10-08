import os
import json
import time
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from models import Baseline1DCNN, PhysicsAugmentedCalibratedCNN
import config
from common import add_real_world_impairments

st.set_page_config(
    page_title="Industrial Fault AI Diagnostic Console",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 26px;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 2px;
    }
    .sub-title {
        font-size: 14px;
        color: #475569;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
    }
    .badge-healthy {
        background-color: #dcfce7;
        color: #15803d;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 16px;
    }
    .badge-fault {
        background-color: #fee2e2;
        color: #b91c1c;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

CLASS_NAMES = config.CLASS_NAMES
RESULTS_DIR = config.RESULTS_DIR
PROCESSED_DIR = config.PROCESSED_DIR
TEST_SNR_DB = config.IMPAIRMENTS["test"]["snr_db"]

@st.cache_resource
def load_models():
    base_model = Baseline1DCNN()
    phys_model = PhysicsAugmentedCalibratedCNN()

    base_path = os.path.join(RESULTS_DIR, "baseline_model.pt")
    phys_path = os.path.join(RESULTS_DIR, "physics_model.pt")

    if os.path.exists(base_path):
        base_model.load_state_dict(torch.load(base_path, map_location="cpu"))
    if os.path.exists(phys_path):
        phys_model.load_state_dict(torch.load(phys_path, map_location="cpu"))

    base_model.eval()
    phys_model.eval()
    return base_model, phys_model

@st.cache_data
def load_test_data():
    # Clean windows: impairments are injected below from the sidebar controls
    x_path = os.path.join(PROCESSED_DIR, "X_test_clean.npy")
    y_path = os.path.join(PROCESSED_DIR, "y_test.npy")
    if os.path.exists(x_path) and os.path.exists(y_path):
        X_test = np.load(x_path)
        y_test = np.load(y_path)
        return X_test, y_test
    return None, None

@st.cache_data
def load_metrics():
    path = os.path.join(RESULTS_DIR, "metrics.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None

base_model, phys_model = load_models()
X_test, y_test = load_test_data()

# Header
st.markdown('<div class="main-title">⚙️ Cyber-Physical Diagnostic Console: Physics-Augmented 1D-CNN</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Live Bearing Vibration Diagnostics under Factory Noise, Baseline Drift & Cross-Load Domain Shifts | Vaibhav Goel</div>', unsafe_allow_html=True)

if X_test is None or y_test is None:
    st.error("Processed test data not found. Please run `preprocess_data.py` first.")
    st.stop()

# Sidebar
st.sidebar.header("🕹️ Sensor Stream Controls")
class_filter = st.sidebar.selectbox("Filter Target Condition:", ["Any (Random)", "Normal (Healthy)", "Inner Race Fault", "Ball Fault", "Outer Race Fault"])

if class_filter == "Any (Random)":
    candidate_indices = np.arange(len(y_test))
else:
    target_cls_idx = CLASS_NAMES.index(class_filter)
    candidate_indices = np.where(y_test == target_cls_idx)[0]

sample_idx = st.sidebar.selectbox("Select Sample Index:", candidate_indices, index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("🏭 Factory Noise & Drift Injection")
add_noise = st.sidebar.checkbox(f"Inject Colored Factory Noise ({TEST_SNR_DB:.0f} dB SNR)", value=True)
add_drift = st.sidebar.checkbox("Inject Non-Linear Thermal Drift", value=True)
add_spikes = st.sidebar.checkbox("Inject Transient Electrical Spikes", value=True)
noise_seed = st.sidebar.number_input("Noise Seed", min_value=0, value=0, step=1,
                                     help="Same seed = same impairment, so the view is stable across reruns.")

# Process signal
raw_sample = X_test[sample_idx].copy()
true_label = y_test[sample_idx]

if add_noise or add_drift or add_spikes:
    impaired_sample = add_real_world_impairments(
        raw_sample, 
        snr_db=TEST_SNR_DB if add_noise else 100.0,
        drift_prob=1.0 if add_drift else 0.0,
        impulse_prob=1.0 if add_spikes else 0.0,
        rng=np.random.default_rng(int(noise_seed) * 100003 + int(sample_idx))
    )
else:
    impaired_sample = raw_sample

# Run Inference
input_tensor = torch.from_numpy(impaired_sample).unsqueeze(0).float()

with torch.no_grad():
    phys_model(input_tensor)  # warm-up so the displayed latency excludes first-call overhead
    t0 = time.perf_counter()
    p_logits, p_attn, p_res = phys_model(input_tensor)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    p_probs = F.softmax(p_logits, dim=1).numpy()[0]
    p_pred = np.argmax(p_probs)

with torch.no_grad():
    b_logits, _, _ = base_model(input_tensor)
    b_probs = F.softmax(b_logits, dim=1).numpy()[0]
    b_pred = np.argmax(b_probs)

# Main Display
col1, col2, col3, col4 = st.columns(4)

with col1:
    is_correct = (p_pred == true_label)
    badge_style = "badge-healthy" if p_pred == 0 else "badge-fault"
    st.markdown(f"**Ground Truth:**\n### {CLASS_NAMES[true_label]}")

with col2:
    st.markdown(f"**Patent Model Prediction:**\n### <span class='{badge_style}'>{CLASS_NAMES[p_pred]}</span>", unsafe_allow_html=True)

with col3:
    st.metric("Calibrated Confidence", f"{p_probs[p_pred]*100:.1f}%", f"{'✔ Match' if is_correct else '✖ Mismatch'}")

with col4:
    st.metric("Inference Latency (this PC)", f"{latency_ms:.2f} ms")

st.markdown("---")

# Visual Tabs
tab1, tab2, tab3 = st.tabs(["📊 Live Explainability & Kinematics", "🔬 Baseline vs Patent Comparison", "📑 Benchmark & Confusion Matrix"])

with tab1:
    st.subheader("Physics Kinematic Residual & Temporal Attention Attribution")
    
    fig, axes = plt.subplots(3, 1, figsize=(13, 6.5), sharex=True)
    plt.subplots_adjust(hspace=0.25)
    
    # 1. Raw sensor stream
    axes[0].plot(impaired_sample[0], color="#0284c7", linewidth=1.1, label="Drive End (DE) Feed with Industrial Drift & Noise")
    axes[0].set_title(f"Input Sensor Stream: 1024-point Sliding Window @ {config.SAMPLING_RATE_HZ // 1000} kHz (Sample #{sample_idx})", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Normalized amplitude", fontsize=10)
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # 2. Kinematic residual
    if p_res is not None:
        res_sig = p_res[0, 0].numpy()
        axes[1].plot(res_sig, color="#e11d48", linewidth=1.1, label="Kinematic Residual r(t) = x(t) - Smooth(x(t)) [Drift Eliminated]")
    else:
        axes[1].plot(impaired_sample[0], color="#e11d48", label="Signal")
    axes[1].set_ylabel("Residual", fontsize=10)
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # 3. Temporal Attention Saliency
    t_attn = p_attn["temporal_attention"][0, 0].numpy()
    t_attn_full = np.interp(np.linspace(0, 1, len(impaired_sample[0])), np.linspace(0, 1, len(t_attn)), t_attn)
    axes[2].plot(t_attn_full, color="#16a34a", linewidth=1.8, label="Learned Temporal Attention (Localized Impact Bursts)")
    axes[2].fill_between(range(len(t_attn_full)), 0, t_attn_full, color="#16a34a", alpha=0.25)
    axes[2].set_ylabel("Attention", fontsize=10)
    axes[2].set_xlabel("Time Samples (Points)", fontsize=10, fontweight="bold")
    axes[2].legend(loc="upper right", fontsize=9)
    axes[2].grid(True, linestyle="--", alpha=0.5)

    st.pyplot(fig)
    plt.close(fig)

with tab2:
    st.subheader("Model Diagnostic Comparison (Standard 1D-CNN vs Proposed Architecture)")
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("#### Standard Baseline 1D-CNN")
        st.markdown(f"**Predicted:** `{CLASS_NAMES[b_pred]}` (Confidence: `{b_probs[b_pred]*100:.1f}%`)")
        fig_b, ax_b = plt.subplots(figsize=(6, 3.5))
        ax_b.barh(CLASS_NAMES, b_probs * 100, color="#64748b")
        ax_b.set_xlim(0, 100)
        ax_b.set_xlabel("Probability (%)")
        st.pyplot(fig_b)
        plt.close(fig_b)

    with c2:
        st.markdown("#### Proposed Physics-Augmented Calibrated CNN")
        st.markdown(f"**Predicted:** `{CLASS_NAMES[p_pred]}` (Confidence: `{p_probs[p_pred]*100:.1f}%`)")
        fig_p, ax_p = plt.subplots(figsize=(6, 3.5))
        ax_p.barh(CLASS_NAMES, p_probs * 100, color="#0284c7")
        ax_p.set_xlim(0, 100)
        ax_p.set_xlabel("Probability (%)")
        st.pyplot(fig_p)
        plt.close(fig_p)

with tab3:
    metrics = load_metrics()
    if metrics is None:
        st.info("Run `python train_and_evaluate.py` to generate results/metrics.json.")
    else:
        proto = metrics["protocol"]
        st.subheader(f"Measured Benchmark: Unseen {proto['test_loads']} HP Load + "
                     f"{proto['impairments']['test']['snr_db']:.0f} dB Plant Noise "
                     f"(mean ± std over {len(proto['seeds'])} seeds)")
        rows = {"Metric": ["Accuracy (%)", "Macro-F1 (%)", "False Alarm Rate (%)", "Missed Fault Rate (%)",
                           "ECE (%)", "CPU Latency, batch 1 (ms)"]}
        for name, label in [("baseline", "Standard 1D-CNN"), ("physics", "PAC-1DCNN (Proposed)")]:
            m = metrics["models"][name]
            cell = lambda k: f"{m['test'][k]['mean']:.2f} ± {m['test'][k]['std']:.2f}"
            rows[label] = [cell("accuracy"), cell("macro_f1"), cell("false_alarm_rate"),
                           cell("missed_fault_rate"), cell("ece"), f"{m['cpu_latency_batch1']['mean_ms']:.3f}"]
        st.table(rows)

    col_cm, col_saliency = st.columns(2)
    with col_cm:
        cm_file = os.path.join(RESULTS_DIR, "patent_figure_confusion_matrix.png")
        if os.path.exists(cm_file):
            st.image(cm_file, caption="Confusion Matrix under 3 HP Load + Plant Noise", use_container_width=True)
    with col_saliency:
        sal_file = os.path.join(RESULTS_DIR, "patent_figure_explainability.png")
        if os.path.exists(sal_file):
            st.image(sal_file, caption="Patent Saliency & Attribution Proof", use_container_width=True)
