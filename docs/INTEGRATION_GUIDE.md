# Image Super-Resolution Project Integration & Deployment Guide

## 1. Project Architecture Overview

This project provides an end-to-end deep learning framework for **Image Super-Resolution (SR)**, combining lightweight custom distillation models with high-capacity Transformer architectures:

- **Student x2 (2× Scale)**: Lightweight RFDB architecture with scale=2 for 2× resolution enhancement.
- **Student x4 (4× Scale)**: Lightweight RFDB architecture trained via Knowledge Distillation from SwinIR for 4× resolution enhancement.
- **SwinIR x8 (8× Scale)**: Pretrained SwinIR-M classical super-resolution transformer model for high-fidelity 8× resolution enhancement.
- **Automatic Model Selector**: Transparent model routing based strictly on requested output resolution scale.
- **FastAPI Backend**: Production REST API exposing image enhancement endpoints (`/enhance`).
- **Streamlit Frontend**: Interactive web interface for uploading images, selecting scale (2x, 4x, 8x), visual before/after comparison, and image downloading.

---

## 2. Model Routing & Checkpoint Architecture

The user interface exposes **only resolution scale options (2x, 4x, 8x)**. Model selection is handled automatically by `inference/model_selector.py`:

| Scale Selected | Routed Model | Model Type | Checkpoint Location |
| :--- | :--- | :--- | :--- |
| **2×** | Student x2 | Lightweight RFDB (scale=2) | `checkpoints/student/student_x2.pth` |
| **4×** | Student x4 | Lightweight RFDB (scale=4) | `checkpoints/student/student_x4.pth` |
| **8×** | SwinIR x8 | SwinIR-M Classical SR (scale=8) | `checkpoints/swinir/swinir_x8.pth` |

---

## 3. Checkpoint Management & Auto-Download

- **Student Checkpoints**:
  - `checkpoints/student/student_x4.pth`: Pretrained 100-epoch Student x4 model (~436,011 parameters).
  - `checkpoints/student/student_x2.pth`: Student x2 model (~186,603 parameters).
- **SwinIR Checkpoints**:
  - `checkpoints/swinir/swinir_x8.pth`: SwinIR-M DF2K x8 classical SR model (~11.8M parameters).
  - If missing locally, `SwinIRInference` automatically downloads the official weights from:
    `https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x8.pth`

---

## 4. Hardware Support & Window Padding

- **CPU / CUDA Execution**: Device selection defaults automatically to GPU (`cuda`) if available, falling back to `cpu`.
- **Window Padding for SwinIR**:
  - SwinIR uses window attention with `window_size = 8`.
  - Input images whose spatial dimensions are not divisible by 8 are padded using reflection padding: `pad_h = (8 - H % 8) % 8`, `pad_w = (8 - W % 8) % 8`.
  - After model forward execution, output tensors are cropped back to exact target scaled size `(scale * W) × (scale * H)`.

---

## 5. How to Train Student x2 Model

To launch isolated Student x2 model training:

```bash
# Verify configuration and architecture
python training/scripts/test_student_x2.py

# Run Student x2 training pipeline
python training/scripts/train_student_x2.py training/configs/student_x2_training.yaml
```

The best checkpoint will be exported to `checkpoints/student/student_x2.pth`.

---

## 6. How to Run Local Inference

```python
from inference import select_model
from PIL import Image

# 1. Load image
img = Image.open("input.png").convert("RGB")

# 2. Select scale (2, 4, or 8)
engine = select_model(scale=4, device="cpu")

# 3. Enhance image
enhanced_img = engine.enhance(img)
enhanced_img.save("output_x4.png")
```

---

## 7. How to Start Backend API Service

```bash
# Start FastAPI backend server on port 8000
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

Endpoints:
- `GET /health` -> Health check status
- `POST /enhance` -> Upload image file and form field `scale` (2, 4, or 8)

---

## 8. How to Start Frontend Application

```bash
# Run Streamlit Web Application
streamlit run frontend/app.py
```

Open browser at `http://localhost:8501`.
