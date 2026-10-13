# Technical Architecture and Model Implementation Details

## 1. Project Overview

### 1.1 Purpose of Image Super-Resolution
Single Image Super-Resolution (SISR) is the computer vision task of reconstructing a high-resolution (HR) image from a degraded, low-resolution (LR) counterpart. The objective is to restore high-frequency spatial details—such as sharp edges, fine textures, and crisp contours—lost during downsampling, sensor limitations, or compression.

### 1.2 Lightweight Student Models for Resource-Constrained Environments
While state-of-the-art vision transformer models (e.g., SwinIR, HAT) achieve high peak reconstruction quality, their immense computational footprint (11.7M+ parameters, billions of FLOPs, and heavy self-attention memory matrices) renders them impractical for real-time edge processing, mobile apps, and lightweight web servers.

This project implements **Lightweight Student Super-Resolution Models** based on **Residual Feature Distillation Blocks (RFDB)** and sub-pixel convolution (PixelShuffle). With parameter budgets of only **186.6K (×2)** and **436.0K (×4)**, the student models achieve fast inference latency (~40–140 ms on standard commodity CPUs) with minimal memory and disk footprints.

### 1.3 Knowledge Distillation (KD)
Knowledge Distillation bridges the performance gap between massive teacher architectures and lightweight student models. During training, a heavy, high-capacity teacher network (SwinIR) guides the compact student network by transferring rich structural representations via pixel-level and feature-level distillation loss objectives.

### 1.4 Production Scope and Reference Baselines
* **Active Production Inference:** Restricted strictly to **Student Model ×2** and **Student Model ×4**. Pretrained SwinIR ×8 has been completely removed from active inference routing and user-facing endpoints.
* **Evaluation Reference Models:** Pretrained **SwinIR ×2** and **SwinIR ×4** are retained strictly within the offline evaluation harness to serve as comparative baselines against the student models on **Set5** and **Set14**.

---

## 2. Overall Project Architecture

```
                                 [OFFLINE TRAINING PIPELINE]
  +--------------------+        +---------------------+
  | DIV2K Dataset (HR) | -----> | Bicubic Degradation |
  +--------------------+        +---------------------+
            |                              |
            | (Ground Truth)               v (LR Input)
            |                   +-----------------------+
            |                   | Frozen Teacher Model  | ----+
            |                   | (SwinIR-M x2 / x4)    |     | Teacher Predictions
            |                   +-----------------------+     |
            |                              |                  v
            |                   +-----------------------+  +--------------------+
            +-----------------> | Student Model (RFDB)  |  | Distillation Loss  |
                                | (Trained with Adam)   |  | (L1 Pixel Distill) |
                                +-----------------------+  +--------------------+
                                           |                          ^
                                           +--- Student Output -------+
                                           |
                                           v
                              +--------------------------+
                              | Checkpoint Serializer    |
                              | (best_student_x2/x4.pth) |
                              +--------------------------+
                                           |
+------------------------------------------+------------------------------------------+
|                                                                                     |
v [ONLINE PRODUCTION INFERENCE PIPELINE]                                              v [EVALUATION PIPELINE]
+------------------------------------+                                    +-----------------------+
| React Web Frontend (Vite)          |                                    | Benchmark Evaluator   |
| (ScaleSelector: 2x or 4x)          |                                    | (Set5 & Set14 Only)   |
+------------------------------------+                                    +-----------------------+
                 | Multipart POST                                                     |
                 v                                                        +-----------+-----------+
+------------------------------------+                                    |                       |
| FastAPI Backend Server             |                                    v                       v
| (Validation & Header Metadata)     |                               +----------+           +-----------+
+------------------------------------+                               | Student  |           | Reference |
                 |                                                   | x2 & x4  |           | SwinIR    |
                 v                                                   +----------+           +-----------+
+------------------------------------+                                    |                       |
| Model Selector & Cache Service     |                                    +-----------+-----------+
| (Routes exclusively to Student)    |                                                |
+------------------------------------+                                                v
                 |                                                        +-----------------------+
+------------------------------------+                                    | Metrics (PSNR, SSIM,  |
| StudentInference Engine            |                                    | MAE, RMSE, Latency)   |
| (Tensor Preprocess -> Forward pass |                                    +-----------------------+
|  -> PixelShuffle -> Clamping)      |                                                |
+------------------------------------+                                                v
                 | PNG Bytes                                              +-----------------------+
                 v                                                        | details/evaluation/   |
+------------------------------------+                                    | raw_results.json      |
| Client UI (Side-by-Side Comparison)|                                    +-----------------------+
+------------------------------------+
```

### Component Separation: Training-Time vs Inference-Time
* **Training-Time Components:**
  - Teacher Network (`SwinIRWrapper`, frozen, weights loaded in eval mode).
  - Loss Manager combining Reconstruction Loss ($L_1$) and Knowledge Distillation Loss ($L_{\text{KD}}$).
  - Synthetic degradation transforms (bicubic downsampling, Gaussian blur, noise).
  - Dataset pipelines loading DIV2K image patches.
* **Inference-Time Components:**
  - Standalone `StudentModel` instances initialized directly from `.pth` checkpoints.
  - Zero teacher dependency; no teacher graph or weights loaded during production enhancement.
  - FastAPI web service routing requests, validating dimensions, and returning PNG images.
  - React single-page frontend offering interactive slider comparisons and resolution inspection.

---

## 3. Student Model ×2 Architecture

The Student Model ×2 is a compact convolutional neural network designed for 2× spatial magnification.

* **Source File:** `training/models/student_model.py`
* **Model Class:** `StudentModel`
* **Checkpoint Path:** `checkpoints/student/best_student_x2.pth` (2.17 MB)
* **Upscaling Factor:** $s = 2$
* **Input Tensor Shape:** $(B, 3, H, W)$
* **Output Tensor Shape:** $(B, 3, 2H, 2W)$
* **Base Feature Channels:** $C = 48$
* **Distilled Feature Channels:** $D = 24$
* **Number of RFDB Blocks:** 3 sequential blocks
* **Activation Function:** LeakyReLU (negative slope $\alpha = 0.05$)
* **Total Parameter Count:** **186,603**
* **Trainable Parameter Count:** **186,603**

### Layer-by-Layer Architectural Specifications (Scale ×2)

| Stage | Submodule | Layer Type | Input Shape $(C_{\text{in}}, H_{\text{in}}, W_{\text{in}})$ | Output Shape $(C_{\text{out}}, H_{\text{out}}, W_{\text{out}})$ | Kernel / Stride / Pad | Parameter Count |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **1. Shallow Extraction** | `feature_extractor.conv` | Conv2d | $(3, H, W)$ | $(48, H, W)$ | $3 \times 3$, $s=1$, $p=1$ | 1,344 |
| | `feature_extractor.act` | LeakyReLU | $(48, H, W)$ | $(48, H, W)$ | $\alpha = 0.05$ | 0 |
| **2. Deep Extraction** | `blocks.0` | RFDB Block 1 | $(48, H, W)$ | $(48, H, W)$ | (Detailed in Section 5) | 24,384 |
| | `blocks.1` | RFDB Block 2 | $(48, H, W)$ | $(48, H, W)$ | (Detailed in Section 5) | 24,384 |
| | `blocks.2` | RFDB Block 3 | $(48, H, W)$ | $(48, H, W)$ | (Detailed in Section 5) | 24,384 |
| **3. Feature Fusion** | `fusion.conv1x1` | Conv2d | $(144, H, W)$ | $(48, H, W)$ | $1 \times 1$, $s=1$, $p=0$ | 6,960 |
| | `fusion.conv3x3` | Conv2d | $(48, H, W)$ | $(48, H, W)$ | $3 \times 3$, $s=1$, $p=1$ | 20,784 |
| **4. Global Skip** | Residual Addition | Elementwise Add | $(48, H, W) + (48, H, W)$ | $(48, H, W)$ | $F_{\text{fused}} + F_{\text{shallow}}$ | 0 |
| **5. Reconstruction** | `reconstruction.conv1` | Conv2d | $(48, H, W)$ | $(192, H, W)$ | $3 \times 3$, $s=1$, $p=1$ | 83,136 |
| | `reconstruction.upsampler` | PixelShuffle | $(192, H, W)$ | $(48, 2H, 2W)$ | $r = 2$ | 0 |
| | `reconstruction.conv2` | Conv2d | $(48, 2H, 2W)$ | $(3, 2H, 2W)$ | $3 \times 3$, $s=1$, $p=1$ | 1,302 |
| | `reconstruction.act` | LeakyReLU | $(3, 2H, 2W)$ | $(3, 2H, 2W)$ | $\alpha = 0.05$ | 0 |
| **Total** | | | | | | **186,603** |

---

## 4. Student Model ×4 Architecture

The Student Model ×4 uses an identical feature extraction backbone as the ×2 model, but incorporates an expanded sub-pixel channel projection layer in the reconstruction backend to accommodate 4× spatial upscaling.

* **Source File:** `training/models/student_model.py`
* **Model Class:** `StudentModel`
* **Checkpoint Path:** `checkpoints/student/best_student_x4.pth` (5.03 MB)
* **Upscaling Factor:** $s = 4$
* **Input Tensor Shape:** $(B, 3, H, W)$
* **Output Tensor Shape:** $(B, 3, 4H, 4W)$
* **Base Feature Channels:** $C = 48$
* **Distilled Feature Channels:** $D = 24$
* **Number of RFDB Blocks:** 3 sequential blocks
* **Activation Function:** LeakyReLU (negative slope $\alpha = 0.05$)
* **Total Parameter Count:** **436,011**
* **Trainable Parameter Count:** **436,011**

### Comparison Between Student ×2 and Student ×4

| Architectural Component | Student Model ×2 | Student Model ×4 | Structural Invariant |
| :--- | :---: | :---: | :--- |
| **Shallow Feature Extractor** | Conv2d(3 $\to$ 48) | Conv2d(3 $\to$ 48) | Exactly identical (1,344 params) |
| **RFDB Trunk (3 Blocks)** | 3 Blocks (73,152 params) | 3 Blocks (73,152 params) | Exactly identical weights layout |
| **Global Feature Fusion** | 144 $\to$ 48 $\to$ 48 (27,744 params) | 144 $\to$ 48 $\to$ 48 (27,744 params) | Exactly identical structure |
| **Reconstruction Conv1** | Conv2d(48 $\to$ 192, $3 \times 3$) | Conv2d(48 $\to$ 768, $3 \times 3$) | **Differs by $s^2$ channel expansion** |
| **PixelShuffle Factor** | $r = 2$ | $r = 4$ | Upscales spatial resolution by $r$ |
| **Reconstruction Conv2** | Conv2d(48 $\to$ 3, $3 \times 3$) | Conv2d(48 $\to$ 3, $3 \times 3$) | Exactly identical (1,302 params) |
| **Reconstruction Params** | 84,438 | 333,846 | +249,408 params due to Conv1 channels |
| **Total Parameters** | **186,603** | **436,011** | Exact difference = $48 \times (768 - 192) \times 9 + (768 - 192) = 249,408$ |

---

## 5. RFDB — Residual Feature Distillation Block

The **Residual Feature Distillation Block (RFDB)** is the core building block of the student network, implemented in `training/models/modules/rfdb.py`.

### 5.1 Operational Mechanism
Standard residual blocks pass all feature maps through deep convolutional layers, creating computational bottlenecks. In contrast, RFDB separates features into two parallel branches:
1. **Information Extraction Branch:** Distills high-frequency structural features via a compact $1 \times 1$ convolution.
2. **Feature Preservation Branch:** Employs zero-parameter identity channel slicing to retain remaining low-frequency background details.

### 5.2 Step-by-Step Flow within an RFDB Block

Given an input feature tensor $X \in \mathbb{R}^{B \times C \times H \times W}$ (where $C = 48$):

```
       Input X (B, 48, H, W)
         |
         +-------------------------------------------------------+
         |                                                       |
         v                                                       |
  [Feature Refinement]                                           |
  Conv 3x3 (48 -> 48) + LeakyReLU                                |
         |                                                       |
         v Refined Tensor R (B, 48, H, W)                        |
         |                                                       |
         +-----------------------------+                         |
         |                             |                         |
         v                             v                         |
  [Information Extraction]      [Feature Preservation]           |
  Conv 1x1 (48 -> 24)           Identity Slice R[:, 24:, :, :]   |
         |                             |                         |
         v Distilled D (B, 24, H, W)   v Preserved P (B, 24, H, W)
         |                             |                         |
         +--------------+--------------+                         |
                        |                                        |
                        v Feature Concatenation [D, P]           |
                        Concatenated Tensor (B, 48, H, W)        |
                        |                                        |
                        v [Feature Fusion]                       |
                        Conv 1x1 (48 -> 48)                      |
                        |                                        |
                        v Fused Features F (B, 48, H, W)         |
                        |                                        |
                        v [Learnable Residual Scaling]           |
                        Scaled = Alpha * F (Alpha in R^{1,48,1,1})
                        |                                        |
                        +----------------------------------------+
                        | (Residual Addition)
                        v
                 Output (B, 48, H, W) = X + Alpha * F
```

### 5.3 Mathematical Formulation
1. **Feature Refinement:**
   $$R = \text{LeakyReLU}(\text{Conv}_{3\times3}(X))$$
2. **Information Extraction:**
   $$D = \text{Conv}_{1\times1}(R) \quad \text{where } D \in \mathbb{R}^{B \times 24 \times H \times W}$$
3. **Feature Preservation (Zero-Parameter Slicing):**
   $$P = R[:, 24:48, :, :] \quad \text{where } P \in \mathbb{R}^{B \times 24 \times H \times W}$$
4. **Channel Concatenation:**
   $$C_{\text{concat}} = [D \mathbin{\Vert} P] \in \mathbb{R}^{B \times 48 \times H \times W}$$
5. **Feature Fusion:**
   $$F = \text{Conv}_{1\times1}(C_{\text{concat}}) \in \mathbb{R}^{B \times 48 \times H \times W}$$
6. **Learnable Channel-Wise Residual Scaling & Addition:**
   $$\text{Output} = X + \boldsymbol{\alpha} \odot F$$
   where $\boldsymbol{\alpha} \in \mathbb{R}^{1 \times 48 \times 1 \times 1}$ is a learnable parameter tensor initialized to $1.0$.

### 5.4 Exact Parameter Count per RFDB Block
* `refinement_conv`: $3 \times 3 \times 48 \times 48 + 48 = 20,784$ parameters
* `distill_conv`: $1 \times 1 \times 48 \times 24 + 24 = 1,176$ parameters
* `fusion_conv`: $1 \times 1 \times 48 \times 48 + 48 = 2,352$ parameters
* `alpha` parameter vector: $1 \times 48 \times 1 \times 1 = 48$ parameters
* **Total parameters per RFDB:** $20,784 + 1,176 + 2,352 + 48 = \mathbf{24,384}$

---

## 6. CNN Layers and Feature Extraction Pipeline

### 6.1 Shallow Feature Extraction (`FeatureExtractor`)
* **Module:** `training/models/modules/feature_extractor.py`
* **Purpose:** Maps standard 3-channel RGB image tensors into a higher-dimensional continuous latent space ($C=48$) using a single $3 \times 3$ convolutional layer.
* **Activation:** LeakyReLU ($\alpha = 0.05$).
* **Output Shape:** $(B, 48, H, W)$.

### 6.2 Deep Feature Extraction
* **Module:** Sequential stack of 3 `ResidualFeatureDistillationBlock` modules stored in an `nn.ModuleList`.
* **Flow:** Shallow features pass sequentially through Block 1 $\to$ Block 2 $\to$ Block 3.
* **Feature Preservation:** Each intermediate block output is captured in a list `[F1, F2, F3]` to facilitate multi-level global feature aggregation.

### 6.3 Global Feature Fusion (`GlobalFeatureFusion`)
* **Module:** `training/models/modules/global_feature_fusion.py`
* **Input:** Concatenation of all 3 RFDB outputs along the channel dimension:
  $$F_{\text{all}} = [F_1 \mathbin{\Vert} F_2 \mathbin{\Vert} F_3] \in \mathbb{R}^{B \times (3 \times 48) \times H \times W} = \mathbb{R}^{B \times 144 \times H \times W}$$
* **Projection Stage 1 (1x1 Conv):** Projects $144 \to 48$ channels ($1 \times 1 \times 144 \times 48 + 48 = 6,960$ params).
* **Projection Stage 2 (3x3 Conv):** Refines spatial receptive field on fused features $48 \to 48$ channels ($3 \times 3 \times 48 \times 48 + 48 = 20,784$ params).
* **Total Fusion Parameters:** $6,960 + 20,784 = \mathbf{27,744}$.

### 6.4 Global Skip Connection
* Directly adds shallow features to the globally fused features:
  $$F_{\text{final}} = F_{\text{fused}} + F_{\text{shallow}}$$
* Preserves low-frequency structural information and stabilizes gradient flow throughout backpropagation.

---

## 7. Reconstruction and Sub-Pixel Convolution

The reconstruction backend transforms deep feature representations into the final high-resolution image using **PixelShuffle** (Sub-Pixel Convolution).

### 7.1 PixelShuffle Mathematical Principle
Instead of executing computationally expensive transpose convolutions (which often produce checkerboard artifacts), PixelShuffle rearranges elements from the channel dimension into spatial blocks:
$$\mathcal{PS}(T)_{c, y, x} = T_{c \cdot r^2 + (y \pmod r) \cdot r + (x \pmod r), \, \lfloor y/r \rfloor, \, \lfloor x/r \rfloor}$$
Given an input tensor of shape $(B, C \cdot r^2, H, W)$, PixelShuffle produces an output of shape $(B, C, H \cdot r, W \cdot r)$ with zero learned parameters.

### 7.2 Implementation in `training/models/modules/reconstruction.py`
1. **Pre-Shuffle Expansion Conv (`conv1`):**
   - Maps 48 feature channels to $48 \times r^2$ channels.
   - For Scale ×2: $48 \to 48 \times 2^2 = 192$ channels.
   - For Scale ×4: $48 \to 48 \times 4^2 = 768$ channels.
2. **Sub-Pixel Spatial Upsampling (`PixelShuffle`):**
   - For Scale ×2: $(B, 192, H, W) \to (B, 48, 2H, 2W)$.
   - For Scale ×4: $(B, 768, H, W) \to (B, 48, 4H, 4W)$.
3. **Final Color Projection Conv (`conv2`):**
   - Maps $(B, 48, rH, rW) \to (B, 3, rH, rW)$ using a $3 \times 3$ convolution followed by LeakyReLU.

---

## 8. Pretrained SwinIR Reference Models

### 8.1 Model Architecture and Origin
* **Architecture:** SwinIR (Image Restoration Using Swin Transformer).
* **Mechanism:** Employs Shifted Window multi-head self-attention (RSTB blocks), deep feature extraction, and residual connections to capture non-local image dependencies.
* **Source Wrapper:** `training/teacher/swinir_wrapper.py` (`SwinIRWrapper`).

### 8.2 Inventory of SwinIR Checkpoints

| Reference Model | Upscale Factor | Checkpoint Path | File Size | Parameter Count | Pipeline Role |
| :--- | :---: | :--- | :---: | :---: | :--- |
| **SwinIR ×2** | 2× | `checkpoints/teacher/swinir_x2.pth` | 64.16 MB | 11,752,127 | Benchmark Reference on Set5/Set14 |
| **SwinIR ×4** | 4× | `checkpoints/teacher/swinir_x4.pth` | 64.72 MB | 11,899,839 | Benchmark Reference on Set5/Set14 |
| **SwinIR ×8** | 8× | `checkpoints/swinir/swinir_x8.pth` | 65.29 MB | 12,047,551 | **Excluded from production inference** |

---

## 9. Teacher Model and Knowledge Distillation

### 9.1 Teacher Role During Distillation
During the offline training phase, the high-capacity SwinIR model serves as the **Teacher**:
* The teacher network parameters are strictly frozen (`param.requires_grad = False`).
* Teacher checkpoints (`swinir_x2.pth`, `swinir_x4.pth`) provide supervisory guidance to the student network.
* Both student and teacher receive identical low-resolution inputs $I_{\text{LR}}$.

### 9.2 Distillation Loss Formulations
Distillation is governed by `LossManager` (`training/losses/loss_manager.py`):
1. **Ground-Truth Pixel Reconstruction Loss ($\mathcal{L}_{\text{pixel}}$):**
   $$\mathcal{L}_{\text{pixel}} = \| I_{\text{SR}}^{\text{student}} - I_{\text{HR}}^{\text{gt}} \|_1$$
2. **Knowledge Distillation Loss ($\mathcal{L}_{\text{KD}}$):**
   $$\mathcal{L}_{\text{KD}} = \| I_{\text{SR}}^{\text{student}} - I_{\text{SR}}^{\text{teacher}} \|_1$$
3. **Total Distillation Objective:**
   $$\mathcal{L}_{\text{total}} = \alpha \cdot \mathcal{L}_{\text{pixel}} + \beta \cdot \mathcal{L}_{\text{KD}}$$
   Standard configuration values: $\alpha = 1.0, \beta = 0.2$.

---

## 10. Loss Functions

The repository provides modular loss functions under `training/losses/`:

1. **`PixelLoss` (`training/losses/pixel_loss.py`):**
   - Supports $L_1$ and Charbonnier losses.
   - Computes pixel-wise error against the ground truth high-resolution target.
2. **`KnowledgeDistillationLoss` (`training/losses/kd_loss.py`):**
   - Computes mean absolute error between the student's super-resolved output tensor and the teacher's output tensor.
3. **`LossManager` (`training/losses/loss_manager.py`):**
   - Aggregates pixel and distillation losses into a single scalar value for backpropagation, returning diagnostic loss metrics per batch.

---

## 11. Training Configuration

Extracted directly from `training/configs/student_x2_training.yaml` and active run configs:

* **Training Dataset:** DIV2K (800 high-resolution training images).
* **Patch Size:** $192 \times 192$ (HR patch size).
* **Batch Size:** 16.
* **Epochs:** 100 (x2 configuration), 300 (extended distillation runs).
* **Optimizer:** Adam ($\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$).
* **Initial Learning Rate:** $1 \times 10^{-4}$ ($0.0001$).
* **LR Milestones:** Decreased by factor $\gamma = 0.5$ at epochs 50 and 80.
* **Weight Decay:** 0.0.
* **Random Seed:** 42.
* **Validation Datasets:** Set5 and Set14 evaluated at 10-epoch checkpoints.

---

## 12. Evaluation Pipeline

The evaluation pipeline (`training/evaluate.py`) is restricted strictly to **Set5** and **Set14**.

### Key Evaluation Rules:
* **Excluded Datasets:** BSD68, Urban100, and Manga109 are completely excluded from the active evaluation script.
* **Input Compatibility:** LR inputs are synthetically generated from ground-truth HR images via bicubic downsampling, ensuring both student and reference models receive identical inputs.
* **Raw Metric Recording:** Raw per-image metrics, resolutions, and inference times are persisted to `details/evaluation/raw_evaluation_results.json`.

---

## 13. Production Inference Pipeline

The production inference pipeline serves real-time user requests through the web UI and REST API.

```
+---------------------------------------------------------------------------------+
|                                 USER BROWSER                                    |
| 1. User uploads image and selects scale (2x or 4x).                             |
| 2. ScaleSelector dispatches multipart POST to /api/enhance.                     |
+---------------------------------------------------------------------------------+
                                      |
                                      v
+---------------------------------------------------------------------------------+
|                             FASTAPI BACKEND (/api/enhance)                      |
| 3. validate_image_file: Verifies image size (< 10 MB) and format (PNG/JPEG/WEBP)|
| 4. load_and_preprocess_image: Decodes bytes to RGB PIL Image.                   |
| 5. get_model_engine(scale): Retrieves cached StudentInference instance.         |
|    - Scale 2 -> StudentInference(scale=2)                                       |
|    - Scale 4 -> StudentInference(scale=4)                                       |
|    - Scale 8 / Other -> Rejects with HTTP 400                                   |
+---------------------------------------------------------------------------------+
                                      |
                                      v
+---------------------------------------------------------------------------------+
|                             STUDENT INFERENCE ENGINE                            |
| 6. Image to Tensor: Maps RGB [0, 255] -> Tensor [0.0, 1.0] shape (1, 3, H, W).  |
| 7. Forward Pass: model(x) within torch.no_grad() on CPU.                        |
| 8. Postprocessing: Clamps to [0.0, 1.0], converts to uint8 NumPy array.         |
| 9. encode_image_to_png: Encodes to optimized PNG bytes.                         |
+---------------------------------------------------------------------------------+
                                      |
                                      v
+---------------------------------------------------------------------------------+
|                                 HTTP RESPONSE                                   |
| 10. Returns PNG image bytes with custom diagnostic headers:                     |
|     - X-Inference-Time: e.g. 0.0842                                             |
|     - X-Model-Used: e.g. Student x2                                             |
|     - X-Scale-Factor: 2                                                         |
|     - X-Input-Resolution / X-Output-Resolution                                  |
| 11. React UI displays interactive side-by-side slider and metadata.             |
+---------------------------------------------------------------------------------+
```

---

## 14. Model Parameters and Complexity Summary

| Model | Scale | Total Parameters | Trainable Parameters | Checkpoint Size | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Student ×2** | 2× | **186,603** | 186,603 | 2.17 MB | Active Production Model (RFDB + PixelShuffle ×2) |
| **Student ×4** | 4× | **436,011** | 436,011 | 5.03 MB | Active Production Model (RFDB + PixelShuffle ×4) |
| **SwinIR ×2** | 2× | **11,752,127** | 0 *(frozen)* | 64.16 MB | Offline Benchmark Reference Baseline |
| **SwinIR ×4** | 4× | **11,899,839** | 0 *(frozen)* | 64.72 MB | Offline Benchmark Reference Baseline |
| **SwinIR ×8** | 8× | **12,047,551** | 0 *(frozen)* | 65.29 MB | Retired; excluded from production inference |

---

## 15. Limitations and Future Improvements

### 15.1 Architectural and Computational Limitations
1. **High-Frequency Hallucination:** Under large upscaling factors (4×), lightweight CNNs smooth out stochastic high-frequency textures (such as fine animal fur or distant foliage) compared to generative diffusion models or deep transformer models.
2. **CPU Execution on Ultra-High Resolutions:** While inference on $256 \times 256$ inputs completes in under 100 ms on CPU, multi-megapixel images require substantial CPU memory bandwidth.

### 15.2 Future Improvements
1. **INT8 Quantization:** Quantizing student weights from FP32 to INT8 via PyTorch Mobile / ONNX Runtime will reduce memory footprint by 4× and deliver 2–3× CPU inference speedups.
2. **Structural Re-parameterization:** Merging multi-branch RFDB convolutions into single equivalent $3 \times 3$ kernels during inference will accelerate execution without altering model capacity.
3. **WebAssembly / ONNX Web Deployment:** Compiling student models for client-side WebAssembly execution will enable zero-server-cost super-resolution directly within the user's web browser.
