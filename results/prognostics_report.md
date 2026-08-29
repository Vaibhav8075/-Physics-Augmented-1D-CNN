# Industrial Prognostics & Health Index Trajectory Report

## 1. Executive Summary

The Physics-Augmented 1D-CNN was extended with a **Continuous Health Degradation Index ($HI(t) \in [0, 100\%]$)** and Remaining Useful Life (RUL) estimation engine. The system successfully detected incipient bearing surface spalling at **t = 313.1 hours**, providing a **25.3-hour maintenance advisory window** before catastrophic mechanical failure at **t = 338.4 hours**.

## 2. Mathematical Health Formulation

$$HI(t) = \left[ 0.70 \cdot P(\text{Normal}|\mathbf{x}(t)) + 0.30 \cdot \left(1 - \frac{E_{\text{residual}}}{E_{\text{total}}}\right) \right] \times 100\%$$

## 3. Operational Lifespan Milestones

- **Normal Steady State**: 0 to 313.1 operating hours ($HI > 70\%$).
- **Incipient Fault Advisory**: Triggered at t = 313.1 hours ($20\% < HI \le 70\%$).
- **Critical Failure EOL**: t = 338.4 hours ($HI \le 20\%$).
- **Actionable RUL Buffer**: **25.3 hours** available for scheduling replacement parts without unplanned line stoppage.
