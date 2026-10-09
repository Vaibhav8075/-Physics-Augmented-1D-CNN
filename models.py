import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------------------
# 1. BASELINE MODEL: Standard Data-Driven 1D-CNN
# -------------------------------------------------------------
class Baseline1DCNN(nn.Module):
    """
    highpass_input=True puts the same fixed residual filter as the PAC model in front
    of the network, so the filter's effect can be measured without the other components.
    """
    def __init__(self, in_channels=2, num_classes=4, highpass_input=False):
        super(Baseline1DCNN, self).__init__()
        self.highpass = PhysicalHarmonicResidualFilter(in_channels, filter_size=11) if highpass_input else None
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=15, stride=2, padding=7)
        self.bn1 = nn.BatchNorm1d(32)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

        # Post-hoc temperature so the baseline gets the same calibration treatment
        self.register_buffer("temperature", torch.ones(1))

    def forward(self, x, return_calibrated=True):
        if self.highpass is not None:
            x, _ = self.highpass(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)
        logits = self.fc(x)
        if return_calibrated:
            logits = logits / self.temperature.clamp(min=1e-3)
        return logits, None, None


class WideKernelCNN(nn.Module):
    """
    WDCNN-style baseline (Zhang et al., Sensors 2017): a wide, strided first-layer kernel
    followed by small 3-tap convolutions with max-pooling. Layer sizes follow our reading of
    that design adapted to 1024-sample windows; they have not been checked against the
    paper's architecture table.
    """
    def __init__(self, in_channels=2, num_classes=4):
        super(WideKernelCNN, self).__init__()
        def block(c_in, c_out, k, s, p):
            return [nn.Conv1d(c_in, c_out, kernel_size=k, stride=s, padding=p),
                    nn.BatchNorm1d(c_out), nn.ReLU(), nn.MaxPool1d(2)]
        self.features = nn.Sequential(
            *block(in_channels, 16, 64, 16, 24),  # 1024 -> 64 -> 32
            *block(16, 32, 3, 1, 1),              # -> 16
            *block(32, 64, 3, 1, 1),              # -> 8
            *block(64, 64, 3, 1, 1),              # -> 4
            *block(64, 64, 3, 1, 0),              # -> 2 -> 1
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(64, 100), nn.ReLU(), nn.Linear(100, num_classes))
        self.register_buffer("temperature", torch.ones(1))

    def forward(self, x, return_calibrated=True):
        logits = self.fc(self.features(x))
        if return_calibrated:
            logits = logits / self.temperature.clamp(min=1e-3)
        return logits, None, None


# -------------------------------------------------------------
# 2. NOVEL PATENT ARCHITECTURE: Physics-Augmented Calibrated CNN
# -------------------------------------------------------------
class PhysicalHarmonicResidualFilter(nn.Module):
    """
    Patent Claim Component 1:
    Kinematic baseline estimator that separates low-frequency operational/drift
    baseline from high-frequency transient fault impact bursts.
    """
    def __init__(self, in_channels=2, filter_size=11):
        super(PhysicalHarmonicResidualFilter, self).__init__()
        # Fixed moving-average smoothing kernel (represents physical shaft speed baseline)
        weight = torch.ones(in_channels, 1, filter_size) / filter_size
        self.register_buffer('smooth_kernel', weight)
        self.padding = filter_size // 2
        self.in_channels = in_channels

    def forward(self, x):
        # x: (B, 2, L)
        baseline = F.conv1d(x, self.smooth_kernel, padding=self.padding, groups=self.in_channels)
        residual = x - baseline  # Isolates high-frequency impact harmonics & eliminates sensor drift
        return residual, baseline


class SensorTemporalAttention(nn.Module):
    """
    Patent Claim Component 2:
    Channel and Temporal Self-Attention block that maps latent activations
    back to physical sensor channels (Drive End vs Fan End) and specific time windows,
    generating an explainable attribution heatmap for operators.
    """
    def __init__(self, in_channels, reduction=8):
        super(SensorTemporalAttention, self).__init__()
        self.channel_fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_channels, max(1, in_channels // reduction)),
            nn.ReLU(),
            nn.Linear(max(1, in_channels // reduction), in_channels),
            nn.Sigmoid()
        )
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(in_channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        c_weights = self.channel_fc(x).unsqueeze(-1) # (B, C, 1)
        x_chan = x * c_weights
        
        t_weights = self.temporal_conv(x_chan)        # (B, 1, L)
        x_out = x_chan * t_weights
        
        return x_out, (c_weights, t_weights)


class PhysicsAugmentedCalibratedCNN(nn.Module):
    """
    Full proposed model. The use_* flags exist only for the ablation study;
    with all flags on, the module layout matches saved checkpoints.
    Calibration is post-hoc (common.calibrate_temperature), so it needs no flag:
    compare return_calibrated=True vs False on the same trained network.
    """
    def __init__(self, in_channels=2, num_classes=4,
                 use_residual_filter=True, use_dual_stream=True, use_attention=True):
        super(PhysicsAugmentedCalibratedCNN, self).__init__()
        self.use_residual_filter = use_residual_filter
        self.use_dual_stream = use_dual_stream
        self.use_attention = use_attention

        # Novel Component 1: Physical Harmonic Residual Separator
        self.residual_filter = PhysicalHarmonicResidualFilter(in_channels=in_channels, filter_size=11)

        if use_dual_stream:
            # Novel Component 2: Dual-Stream Multi-Scale Conv (Residual + Raw Features)
            # Stream 1: High-frequency fault impact extractor (on physics residual)
            self.stream_residual = nn.Sequential(
                nn.Conv1d(in_channels, 32, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU()
            )
            # Stream 2: Wide receptive field envelope extractor (on raw signal)
            self.stream_envelope = nn.Sequential(
                nn.Conv1d(in_channels, 32, kernel_size=15, stride=2, padding=7),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=15, stride=2, padding=7),
                nn.BatchNorm1d(64),
                nn.ReLU()
            )
            # Feature Fusion & Dimensionality Reduction (64 + 64 = 128 channels)
            self.fusion = nn.Sequential(
                nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU()
            )
        else:
            # Ablation: single-scale stream with the same output shape (128, L/8)
            self.single_stream = nn.Sequential(
                nn.Conv1d(in_channels, 32, kernel_size=15, stride=2, padding=7),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU()
            )

        # Novel Component 3: Sensor & Temporal Explainability Attention
        if use_attention:
            self.attention = SensorTemporalAttention(in_channels=128)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

        # Novel Component 4: Temperature, fitted post-hoc on validation NLL (not trained)
        self.register_buffer("temperature", torch.ones(1))

    def forward(self, x, return_calibrated=True):
        # 1. Physical residual extraction
        if self.use_residual_filter:
            residual, _ = self.residual_filter(x)
        else:
            residual = x

        # 2. Feature extraction
        if self.use_dual_stream:
            feat_res = self.stream_residual(residual)
            feat_env = self.stream_envelope(x)
            fused = torch.cat([feat_res, feat_env], dim=1) # (B, 128, L/4)
            features = self.fusion(fused)                  # (B, 128, L/8)
        else:
            features = self.single_stream(residual)

        # 3. Sensor & Temporal Attention
        if self.use_attention:
            attended, (c_weights, t_weights) = self.attention(features)
        else:
            attended, c_weights, t_weights = features, None, None

        # 4. Pooling & Classification
        pooled = self.pool(attended).squeeze(-1)
        raw_logits = self.fc(pooled)

        # 5. Output Calibration
        logits = raw_logits / self.temperature.clamp(min=1e-3) if return_calibrated else raw_logits

        attn_info = {
            "channel_attention": c_weights,
            "temporal_attention": t_weights,
            "residual_norm": torch.norm(residual, dim=-1)
        }

        return logits, attn_info, residual if self.use_residual_filter else None
