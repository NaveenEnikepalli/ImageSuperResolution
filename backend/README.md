# PixelLift — FastAPI Backend Service

[![FastAPI](https://img.shields.io/badge/FastAPI-0.140%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Uvicorn](https://img.shields.io/badge/Uvicorn-0.30%2B-black.svg)](https://www.uvicorn.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)

FastAPI REST API backend service for the PixelLift project: Lightweight AI-Based Image Enhancement using Knowledge Distillation. It handles file validation, image preprocessing, persistent PyTorch model caching, automatic scale-based model routing, and super-resolution inference execution.

---

## 🏗️ Architecture

```
                  ┌───────────────────────────────┐
                  │    FastAPI Backend Server     │
                  │    (http://127.0.0.1:8000)    │
                  └───────────────┬───────────────┘
                                  │
                                  ▼
                  ┌───────────────────────────────┐
                  │    Enhancement Service Layer  │
                  └───────────────┬───────────────┘
                                  │
                  ┌───────────────┴───────────────┐
                  ▼                               ▼
      ┌──────────────────────┐        ┌──────────────────────┐
      │  Model Service Cache │        │ Image Preprocessing  │
      └──────────┬───────────┘        └──────────────────────┘
                 │
                 ▼
      ┌──────────────────────┐
      │    Model Selector    │
      └──────────┬───────────┘
                 │
      ┌──────────┼──────────────────────┐
      ▼          ▼                      ▼
┌──────────┐ ┌──────────┐    ┌────────────────────┐
│Student x2│ │Student x4│    │    SwinIR x8       │
│ (~186K)  │ │ (~436K)  │    │     (~11.8M)       │
└──────────┘ └──────────┘    └────────────────────┘
```

---

## 📡 API Endpoints

### 1. Health Status Endpoint
- **URL**: `GET /api/health` (also `GET /health`)
- **Response**:
  ```json
  {
    "status": "ok",
    "service": "ImageSuperResolution API",
    "supported_scales": [2, 4, 8]
  }
  ```

### 2. Available Models Endpoint
- **URL**: `GET /api/models`
- **Response**:
  ```json
  {
    "supported_scales": [2, 4, 8],
    "models": {
      "2": { "name": "Student x2", "type": "Lightweight RFDB Student", "scale": "2×" },
      "4": { "name": "Student x4", "type": "Lightweight RFDB Student", "scale": "4×" },
      "8": { "name": "SwinIR x8", "type": "SwinIR-M Transformer", "scale": "8×" }
    }
  }
  ```

### 3. Super-Resolution Enhancement Endpoint
- **URL**: `POST /api/enhance` (also `POST /enhance`)
- **Content-Type**: `multipart/form-data`
- **Form Parameters**:
  - `image` (or `file`): Uploaded image file (`PNG`, `JPG`, `JPEG`, `WEBP`)
  - `scale`: `2`, `4`, or `8` (default: `4`)
- **Response**: PNG binary image payload (`image/png`)
- **Response Headers**:
  - `X-Inference-Time`: Elapsed inference duration in seconds.
  - `X-Model-Used`: Underlying model architecture name.
  - `X-Input-Resolution`: Original image dimensions (\(W \times H\)).
  - `X-Output-Resolution`: Scaled output dimensions.
  - `X-Scale-Factor`: Applied resolution multiplier (`2`, `4`, or `8`).

---

## 🛠️ Environment Setup & Manual Execution

### Command Prompt (CMD)

```cmd
:: 1. Navigate to backend directory
cd /d "D:\MINI PROJECT-ImageSuperResolution\ImageSuperResolution\backend"

:: 2. Create Virtual Environment
python -m venv .venv

:: 3. Activate Virtual Environment
.venv\Scripts\activate

:: 4. Upgrade Pip & Install Backend Dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

:: 5. Launch FastAPI Backend Server
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

### PowerShell

```powershell
# 1. Navigate to backend directory
Set-Location "D:\MINI PROJECT-ImageSuperResolution\ImageSuperResolution\backend"

# 2. Set Execution Policy (if script execution is blocked)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 3. Create Virtual Environment
python -m venv .venv

# 4. Activate Virtual Environment
.\.venv\Scripts\Activate.ps1

# 5. Upgrade Pip & Install Backend Dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 6. Launch FastAPI Backend Server
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

---

## 🔗 Important URLs

- **API Base**: `http://127.0.0.1:8000`
- **Health Check**: `http://127.0.0.1:8000/api/health`
- **Interactive Swagger Documentation**: `http://127.0.0.1:8000/docs`
- **Redoc API Documentation**: `http://127.0.0.1:8000/redoc`

---

## 🔧 Troubleshooting

1. **Port 8000 Already in Use**:
   Check processes on port 8000:
   ```cmd
   netstat -ano | findstr :8000
   ```
   Kill process:
   ```cmd
   taskkill /PID <PID_NUMBER> /F
   ```

2. **Missing Module Error**:
   Ensure virtual environment is activated (`.venv\Scripts\activate`) and requirements are installed (`pip install -r requirements.txt`).
