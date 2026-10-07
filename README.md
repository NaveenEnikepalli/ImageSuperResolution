# Lightweight Image Super-Resolution Full-Stack Project
## React + Vite Frontend + FastAPI Backend + PyTorch Models

[![React](https://img.shields.io/badge/React-19.0%2B-61DAFB.svg)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-6.0%2B-646CFF.svg)](https://vitejs.dev/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.140%2B-009688.svg)](https://fastapi.tiangolo.com/)

Full-stack application for **"Lightweight Image Super-Resolution using Knowledge Distillation for Resource-Constrained Devices"**.

---

## 🌟 1. Overview

This project provides multi-scale AI image enhancement across three scale factors (**2×**, **4×**, and **8×**):
- **2× Scaling**: Uses a lightweight **Student RFDB model** (~186,603 parameters).
- **4× Scaling**: Uses a lightweight **Student RFDB model** (~436,011 parameters).
- **8× Scaling**: Uses the high-capacity **SwinIR-M Vision Transformer** model (~11.8M parameters).

---

## 🏗️ 2. System Architecture & Communication

The application separates UI presentation from backend model inference:
- **Frontend**: React + Vite SPA running on `http://localhost:5173`.
- **Backend**: FastAPI REST API server running on `http://127.0.0.1:8000`.
- **Communication**: Frontend sends multipart form requests (`POST /api/enhance`) to the FastAPI backend and receives binary PNG image output along with performance headers (`X-Inference-Time`, `X-Model-Used`, `X-Input-Resolution`, `X-Output-Resolution`, `X-Scale-Factor`).

```
  [React + Vite UI (Port 5173)]
           │
           │ HTTP POST /api/enhance (multipart/form-data)
           ▼
    [FastAPI API (Port 8000)]
           │
           ▼
 [Enhancement Service Layer] ──► [Model Cache Manager]
           │                              │
           ▼                              ▼
  [Image Preprocessing]        [Inference Routing Engine]
                                          │
                        ┌─────────────────┼─────────────────┐
                        ▼                 ▼                 ▼
                  [Student x2]      [Student x4]       [SwinIR x8]
```

---

## ⚡ 3. Model Routing Matrix

| Scale | Model Architecture | Parameters | Trained Checkpoint Path | Hardware Target |
| :---: | :---: | :---: | :---: | :---: |
| **2×** | Student RFDB Model (`StudentInference(scale=2)`) | ~186,603 | `checkpoints/student/best_student_x2.pth` | Resource-constrained |
| **4×** | Student RFDB Model (`StudentInference(scale=4)`) | ~436,011 | `checkpoints/student/best_student_x4.pth` | Resource-constrained |
| **8×** | SwinIR-M Transformer (`SwinIRInference(scale=8)`) | ~11,800,000 | `checkpoints/swinir/swinir_x8.pth` | High-definition synthesis |

*(SwinIR x4 is excluded from routing as it was used only for internal model benchmarking).*

---

## 📂 4. Project Directory Structure

```
ImageSuperResolution/
│
├── backend/                    # FastAPI REST API Backend
│   ├── venv/                   # Independent Backend Virtual Environment
│   ├── app/
│   │   ├── main.py             # FastAPI startup & CORS config
│   │   ├── config.py           # Backend limits & model metadata
│   │   ├── api/routes.py       # API endpoints (/api/health, /api/models, /api/enhance)
│   │   ├── schemas/            # Pydantic schemas
│   │   └── services/           # Enhancement, caching, & image preprocessing services
│   ├── tests/                  # Backend test suite
│   ├── main.py                 # Convenience backend entry point
│   ├── requirements.txt        # Backend dependencies
│   └── README.md               # Backend execution guide
│
├── frontend/                   # React + Vite Frontend Web App
│   ├── node_modules/           # Node.js dependencies
│   ├── public/                 # Static assets
│   ├── src/
│   │   ├── assets/             # Images & static assets
│   │   ├── components/         # Navbar, Footer, ImageUploader, ScaleSelector, etc.
│   │   ├── pages/              # Home & Enhance pages
│   │   ├── services/           # HTTP API client (api.js)
│   │   ├── App.jsx             # React app root component
│   │   ├── main.jsx            # Vite DOM mounting point
│   │   └── index.css           # Global design system & styles
│   ├── .env.example            # Frontend environment variable template
│   ├── eslint.config.js        # ESLint configuration
│   ├── package.json            # Node project configuration
│   ├── vite.config.js          # Vite build configuration
│   └── README.md               # Frontend execution guide
│
├── inference/                  # Inference pipeline & model selector
│   ├── model_selector.py       # Automatic scale-based model router
│   ├── student_inference.py    # RFDB Student inference engine
│   └── swinir_inference.py     # SwinIR Transformer inference engine
│
├── checkpoints/                # Pretrained model checkpoints
│   ├── student/                # best_student_x2.pth, best_student_x4.pth
│   └── swinir/                 # swinir_x8.pth
│
├── tests/                      # Project integration & unit test suite
├── .gitignore                  # Git ignore file
├── .env.example                # Environment variables template
└── README.md                   # Main documentation
```

---

## 🚀 5. Execution Instructions

To run the application, open **two separate terminal windows**:

### Step 1: Launch FastAPI Backend (Terminal 1)
```powershell
cd /d "D:\MINI PROJECT-ImageSuperResolution\ImageSuperResolution\backend"
.\venv\Scripts\activate
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```
- API Base: `http://127.0.0.1:8000`
- Swagger Docs: `http://127.0.0.1:8000/docs`

### Step 2: Launch React Frontend (Terminal 2)
```powershell
cd /d "D:\MINI PROJECT-ImageSuperResolution\ImageSuperResolution\frontend"
npm install
npm run dev
```
- Web App UI: `http://localhost:5173`
