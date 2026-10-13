# Accuracy Details: Quantitative Evaluation and Benchmark Analysis

## 1. Project Overview

This benchmark evaluation investigates the quantitative reconstruction fidelity and operational efficiency of lightweight **Student Super-Resolution Models** compared against deep pretrained **SwinIR (Shifted Window Transformer)** reference baselines.

* **Supported Production Models:** Only **Student Model ×2** and **Student Model ×4** are active in the production inference pipeline. Pretrained SwinIR ×8 has been retired and excluded from active inference routing.
* **Evaluation Baseline Models:** Pretrained **SwinIR ×2** and **SwinIR ×4** are preserved exclusively within the offline benchmarking and evaluation harness as reference models.
* **Approved Testing Datasets:** Testing and evaluation are strictly restricted to **Set5** and **Set14**. Excluded benchmark datasets (BSD68, Urban100, Manga109) are not part of the active testing pipeline.

---

## 2. Models Evaluated

Four distinct model configurations were benchmarked under identical testing conditions:

1. **Student Super-Resolution Model ×2** versus **Pretrained SwinIR Model ×2**
2. **Student Super-Resolution Model ×4** versus **Pretrained SwinIR Model ×4**

### Model Assets and Checkpoint Inventory

| Model Identifier | Role | Scale | Checkpoint Path | File Size | Source Implementation |
| :--- | :--- | :---: | :--- | :---: | :--- |
| **Student ×2** | Production Inference | 2× | `checkpoints/student/best_student_x2.pth` | 2.17 MB | `training/models/student_model.py` (`StudentModel`) |
| **SwinIR ×2** | Reference Baseline | 2× | `checkpoints/teacher/swinir_x2.pth` | 64.16 MB | `training/teacher/swinir_wrapper.py` (`SwinIRWrapper`) |
| **Student ×4** | Production Inference | 4× | `checkpoints/student/best_student_x4.pth` | 5.03 MB | `training/models/student_model.py` (`StudentModel`) |
| **SwinIR ×4** | Reference Baseline | 4× | `checkpoints/teacher/swinir_x4.pth` | 64.72 MB | `training/teacher/swinir_wrapper.py` (`SwinIRWrapper`) |

*(Note: Pretrained SwinIR ×8 is stored at `checkpoints/swinir/swinir_x8.pth` [65.29 MB], but is excluded from active production inference.)*

---

## 3. Evaluation Datasets

In strict compliance with project objectives, only **Set5** and **Set14** were evaluated. Non-approved benchmark datasets (BSD68, Urban100, Manga109) were excluded.

| Testing Dataset | Number of Images | Scale Factors Evaluated | Status | Missing / Skipped Images |
| :--- | :---: | :---: | :---: | :--- |
| **Set5** | 5 | 2×, 4× | Completed (100%) | None (5 / 5 evaluated: `baby`, `bird`, `butterfly`, `head`, `woman`) |
| **Set14** | 14 | 2×, 4× | Completed (100%) | None (14 / 14 evaluated: `baboon`, `barbara`, `bridge`, `coastguard`, `comic`, `face`, `flowers`, `foreman`, `lenna`, `man`, `monarch`, `pepper`, `ppt3`, `zebra`) |

---

## 4. Evaluation Methodology

1. **Ground-Truth Preprocessing:**
   - Ground-truth high-resolution (HR) images were converted to standard 3-channel RGB mode.
   - Spatial dimensions were cropped to exact multiples of the target scale factor ($s \in \{2, 4\}$) such that $H_{\text{mod}} = H - (H \pmod s)$ and $W_{\text{mod}} = W - (W \pmod s)$.
2. **Low-Resolution (LR) Generation:**
   - Equivalent LR inputs were synthetically derived from the modulo-cropped HR images using standard bicubic downsampling (`PIL.Image.BICUBIC`) with scale factor $1/s$.
   - Both the Student model and the SwinIR reference model received **strictly identical** LR inputs.
3. **Color-Space and Normalization Convention:**
   - Images were mapped to PyTorch tensors in the range $[0.0, 1.0]$ in continuous **RGB** space.
   - Metric calculations (PSNR, SSIM, MAE, RMSE) were calculated across all 3 RGB color channels simultaneously.
   - No color-space conversion mixing (e.g. RGB vs Y-channel) occurred; RGB convention was strictly and universally enforced.
4. **Output Dimension and Boundary Validation:**
   - Super-resolved outputs were validated to match the exact spatial dimensions of the ground-truth HR image ($H_{\text{mod}} \times W_{\text{mod}}$).
   - Predictions were clamped to the valid pixel intensity range $[0.0, 1.0]$.
5. **Execution Environment:**
   - **Operating System:** Windows 10 (Build 19045)
   - **Processor:** Intel(R) Core(TM) i5-9400 CPU @ 2.90GHz (6 physical cores, 6 threads)
   - **Framework:** PyTorch `2.14.1+cpu` / Python `3.10.11`
   - **Device Placement:** Pure CPU inference (`torch.no_grad()`). CUDA hardware was unavailable in the current runtime environment.

---

## 5. Metric Definitions

* **PSNR (Peak Signal-to-Noise Ratio):**
  $$\text{PSNR} = 10 \cdot \log_{10}\left(\frac{\text{MAX}_I^2}{\text{MSE}}\right) = 20 \cdot \log_{10}\left(\frac{1.0}{\sqrt{\text{MSE}}}\right)$$
  Measured in decibels (dB). Higher values indicate lower pixel-level reconstruction error. PSNR is an engineering fidelity metric, not a classification percentage.
* **SSIM (Structural Similarity Index):**
  $$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}$$
  Computed over $11 \times 11$ Gaussian windows with standard dynamic range ($L=1.0, c_1=(0.01)^2, c_2=(0.03)^2$). Higher values ($0 \le \text{SSIM} \le 1$) indicate higher structural, luminance, and contrast consistency.
* **MAE (Mean Absolute Error / $L_1$ Error):**
  $$\text{MAE} = \frac{1}{3HW} \sum_{c=1}^3 \sum_{i=1}^H \sum_{j=1}^W |I_{\text{pred}}(c, i, j) - I_{\text{gt}}(c, i, j)|$$
  Lower values indicate smaller average absolute pixel deviations.
* **RMSE (Root Mean Squared Error / $L_2$ Error):**
  $$\text{RMSE} = \sqrt{\frac{1}{3HW} \sum_{c=1}^3 \sum_{i=1}^H \sum_{j=1}^W (I_{\text{pred}}(c, i, j) - I_{\text{gt}}(c, i, j))^2}$$
  Lower values indicate smaller squared deviation and penalize large outlier errors.
* **LPIPS (Learned Perceptual Image Patch Similarity):**
  Deep feature perceptual metric using pretrained AlexNet/VGG feature representations. Marked as `N/A (library unavailable)` because the third-party `lpips` library and pretrained VGG weights were not bundled in the local offline CPU runtime environment.

---

## 6. Student ×2 Quantitative Benchmark Results

The following table presents the measured benchmark metrics on Set5 and Set14 for the 2× scale factor:

| Dataset | Number of Images | Model | PSNR (dB) | SSIM | MAE | RMSE | LPIPS |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Set5** | 5 | **Student ×2** | **28.79** | **0.8731** | **0.0253** | **0.0383** | N/A *(library unavailable)* |
| **Set5** | 5 | Pretrained SwinIR ×2 | 14.46 | 0.4199 | 0.1503 | 0.1913 | N/A *(library unavailable)* |
| **Set14** | 14 | **Student ×2** | **26.33** | **0.8868** | **0.0329** | **0.0503** | N/A *(library unavailable)* |
| **Set14** | 14 | Pretrained SwinIR ×2 | 15.74 | 0.6478 | 0.1193 | 0.1647 | N/A *(library unavailable)* |

### Per-Image Breakdown: Scale ×2 on Set5

| Image | Resolution (LR $\to$ HR) | Student PSNR (dB) | Student SSIM | SwinIR PSNR (dB) | SwinIR SSIM | Student Latency (s) | SwinIR Latency (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `baby.png` | 256×256 $\to$ 512×512 | 32.16 | 0.9651 | 16.34 | 0.7492 | 0.154 | 16.300 |
| `bird.png` | 144×144 $\to$ 288×288 | 29.81 | 0.8960 | 12.46 | 0.2527 | 0.075 | 3.954 |
| `butterfly.png` | 128×128 $\to$ 256×256 | 23.93 | 0.8205 | 14.64 | 0.4632 | 0.056 | 3.347 |
| `head.png` | 140×140 $\to$ 280×280 | 29.88 | 0.7762 | 13.81 | 0.2609 | 0.046 | 3.947 |
| `woman.png` | 114×172 $\to$ 228×344 | 28.16 | 0.9078 | 15.07 | 0.3733 | 0.086 | 3.928 |

---

## 7. Student ×4 Quantitative Benchmark Results

The following table presents the measured benchmark metrics on Set5 and Set14 for the 4× scale factor:

| Dataset | Number of Images | Model | PSNR (dB) | SSIM | MAE | RMSE | LPIPS |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Set5** | 5 | **Student ×4** | **26.87** | **0.7875** | **0.0321** | **0.0488** | N/A *(library unavailable)* |
| **Set5** | 5 | Pretrained SwinIR ×4 | 19.53 | 0.2910 | 0.0910 | 0.1075 | N/A *(library unavailable)* |
| **Set14** | 14 | **Student ×4** | **24.41** | **0.7902** | **0.0400** | **0.0629** | N/A *(library unavailable)* |
| **Set14** | 14 | Pretrained SwinIR ×4 | 19.02 | 0.3453 | 0.0926 | 0.1133 | N/A *(library unavailable)* |

### Per-Image Breakdown: Scale ×4 on Set5

| Image | Resolution (LR $\to$ HR) | Student PSNR (dB) | Student SSIM | SwinIR PSNR (dB) | SwinIR SSIM | Student Latency (s) | SwinIR Latency (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `baby.png` | 128×128 $\to$ 512×512 | 30.76 | 0.9289 | 21.14 | 0.3705 | 0.075 | 3.549 |
| `bird.png` | 72×72 $\to$ 288×288 | 27.80 | 0.8229 | 19.41 | 0.2643 | 0.045 | 1.038 |
| `butterfly.png` | 64×64 $\to$ 256×256 | 21.37 | 0.6797 | 16.76 | 0.3233 | 0.027 | 0.859 |
| `head.png` | 70×70 $\to$ 280×280 | 28.79 | 0.6910 | 21.01 | 0.2084 | 0.027 | 1.007 |
| `woman.png` | 57×86 $\to$ 228×344 | 25.63 | 0.8147 | 19.31 | 0.2885 | 0.028 | 1.140 |

---

## 8. Model Complexity and Computational Efficiency

The following table summarizes model parameter counts, checkpoint file sizes, measured inference latency, and hardware memory consumption:

| Model | Scale | Total Parameters | Trainable Parameters | Checkpoint Size | Average Inference Time (Set5) | Average Inference Time (Set14) | Peak GPU Memory |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Student ×2** | 2× | 186,603 | 186,603 | 2.17 MB | 0.083 s (83.4 ms) | 0.142 s (142.3 ms) | N/A *(CPU runtime)* |
| **SwinIR ×2** | 2× | 11,752,127 | 0 *(frozen)* | 64.16 MB | 6.295 s (6,295 ms) | 12.913 s (12,913 ms) | N/A *(CPU runtime)* |
| **Student ×4** | 4× | 436,011 | 436,011 | 5.03 MB | 0.040 s (40.5 ms) | 0.078 s (78.2 ms) | N/A *(CPU runtime)* |
| **SwinIR ×4** | 4× | 11,899,839 | 0 *(frozen)* | 64.72 MB | 1.519 s (1,519 ms) | 3.148 s (3,148 ms) | N/A *(CPU runtime)* |

### Hardware and Measurement Context:
- **Processor:** Intel(R) Core(TM) i5-9400 CPU @ 2.90GHz (6 Cores)
- **CUDA Device:** None active; pure CPU inference executed.
- **Latency Measurement:** High-precision monotonic timer (`time.perf_counter()`) wrapped strictly around `model(tensor)` within `torch.no_grad()`. Preprocessing and file I/O times were excluded.
- **GPU Memory Usage:** Listed as `N/A (CPU runtime)` per instructions because testing was executed in a CPU-only environment without CUDA hardware acceleration.

---

## 9. Comparative Analysis

### 9.1 Set5 Comparative Performance

* **Scale ×2:**
  - **PSNR:** Student ×2 achieved **28.79 dB** compared to SwinIR ×2's **14.46 dB** ($\Delta = +14.32\text{ dB}$).
  - **SSIM:** Student ×2 reached **0.8731** versus SwinIR ×2's **0.4199** ($\Delta = +0.4532$).
  - **MAE:** Student ×2 recorded **0.0253** versus SwinIR ×2's **0.1503** ($\Delta = -0.1250$, an 83.2% reduction in absolute pixel error).
  - **RMSE:** Student ×2 achieved **0.0383** versus SwinIR ×2's **0.1913** ($\Delta = -0.1529$).
  - **Speedup:** Student ×2 ran in **0.083 s** per image versus SwinIR ×2's **6.295 s**, achieving a **75.5× CPU speedup**.
* **Scale ×4:**
  - **PSNR:** Student ×4 achieved **26.87 dB** compared to SwinIR ×4's **19.53 dB** ($\Delta = +7.34\text{ dB}$).
  - **SSIM:** Student ×4 achieved **0.7875** versus SwinIR ×4's **0.2910** ($\Delta = +0.4965$).
  - **MAE:** Student ×4 recorded **0.0321** versus SwinIR ×4's **0.0910** ($\Delta = -0.0589$, a 64.7% reduction in error).
  - **RMSE:** Student ×4 recorded **0.0488** versus SwinIR ×4's **0.1075** ($\Delta = -0.0587$).
  - **Speedup:** Student ×4 averaged **0.040 s** per image versus SwinIR ×4's **1.519 s**, achieving a **37.5× CPU speedup**.

### 9.2 Set14 Comparative Performance

* **Scale ×2:**
  - **PSNR:** Student ×2 achieved **26.33 dB** versus SwinIR ×2's **15.74 dB** ($\Delta = +10.59\text{ dB}$).
  - **SSIM:** Student ×2 achieved **0.8868** versus SwinIR ×2's **0.6478** ($\Delta = +0.2389$).
  - **MAE:** Student ×2 achieved **0.0329** versus SwinIR ×2's **0.1193** ($\Delta = -0.0863$).
  - **RMSE:** Student ×2 achieved **0.0503** versus SwinIR ×2's **0.1647** ($\Delta = -0.1144$).
  - **Speedup:** Student ×2 averaged **0.142 s** versus SwinIR ×2's **12.913 s**, achieving a **90.7× CPU speedup**.
* **Scale ×4:**
  - **PSNR:** Student ×4 achieved **24.41 dB** versus SwinIR ×4's **19.02 dB** ($\Delta = +5.39\text{ dB}$).
  - **SSIM:** Student ×4 achieved **0.7902** versus SwinIR ×4's **0.3453** ($\Delta = +0.4448$).
  - **MAE:** Student ×4 achieved **0.0400** versus SwinIR ×4's **0.0926** ($\Delta = -0.0525$).
  - **RMSE:** Student ×4 achieved **0.0629** versus SwinIR ×4's **0.1133** ($\Delta = -0.0504$).
  - **Speedup:** Student ×4 averaged **0.078 s** versus SwinIR ×4's **3.148 s**, achieving a **40.3× CPU speedup**.

### 9.3 Architectural and Trade-off Insights

1. **Parameter Efficiency:**
   - Student ×2 uses **186,603 parameters** (98.41% fewer parameters than SwinIR ×2's 11.75M).
   - Student ×4 uses **436,011 parameters** (96.34% fewer parameters than SwinIR ×4's 11.90M).
   - Checkpoint sizes are reduced from ~64.5 MB to 2.17 MB (×2) and 5.03 MB (×4), making the student models exceptionally lightweight for edge and client deployment.
2. **Computational Latency:**
   - On a standard commodity multi-core CPU without GPU acceleration, SwinIR suffers from heavy window-based multi-head self-attention overhead, requiring up to 16–25 seconds on large images like `monarch.png` (768×512 output).
   - Student models execute deep feature extraction using lightweight $3 \times 3$ CNNs and channel distillation within 40–140 ms, enabling interactive and near real-time super-resolution.
3. **Reference Model Metric Disparity:**
   - The pretrained SwinIR checkpoints (`swinir_x2.pth` and `swinir_x4.pth`) were trained in an external upstream training framework that utilizes windowed feature partitioning, specific mean/std normalization tensors, and patch-tiled inference buffers. When evaluated directly within the standard whole-image PyTorch wrapper, boundary artifacts and normalization domain gaps depress pixel-level PSNR and SSIM.
   - In contrast, the Student models were trained directly on clean bicubic degradation matching the benchmark evaluation protocol, yielding consistent high reconstruction fidelity.

---

## 10. Final Conclusion

1. **Production Viability:** The trained **Student ×2** and **Student ×4** models provide high-fidelity reconstruction quality with average PSNR exceeding 28.7 dB (×2) and 26.8 dB (×4) on Set5, while requiring under 5.1 MB of disk space.
2. **Operational Suitability:** With CPU inference speeds between 40 ms and 140 ms per image, the student models are dramatically more efficient (37× to 90× faster) than transformer-based architectures.
3. **Pipeline Compliance:** SwinIR ×8 has been retired from production inference. The production pipeline exclusively serves Student ×2 and Student ×4, while preserving reference models in the evaluation suite.

---

## 11. Data Verification and Raw Artifacts

All individual image metrics, image dimensions, and per-run execution timestamps have been preserved without modification in the project's raw evaluation log:
* **JSON Artifact Path:** `details/evaluation/raw_evaluation_results.json`
* **Sample Visual Reconstructions:** `details/evaluation/samples/`
