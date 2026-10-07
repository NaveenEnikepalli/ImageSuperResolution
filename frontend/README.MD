# PixelLift Frontend (React + Vite)

A modern, high-performance React + Vite user interface for PixelLift: Lightweight AI-Based Image Enhancement using Knowledge Distillation.

---

## 📋 Prerequisites

- **Node.js**: v18.0.0 or higher
- **npm**: v9.0.0 or higher

---

## 🚀 Getting Started

### 1. Install Dependencies

Navigate to the `frontend/` directory and install the Node.js dependencies:

```powershell
cd /d "D:\MINI PROJECT-ImageSuperResolution\ImageSuperResolution\frontend"
npm install
```

### 2. Configure Environment Variables

The application uses environment variables for backend API routing. An example configuration is provided in `.env.example`:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

To customize the backend base URL, create or update `.env` in the `frontend/` directory.

---

## 🛠️ Development & Building

### Start Development Server

Launch the Vite local development server:

```powershell
npm run dev
```

The application will be accessible at `http://localhost:5173`.

### Code Linting

Run ESLint to verify code quality and style standards:

```powershell
npm run lint
```

### Production Build

Compile and bundle the production build artifact to the `dist/` directory:

```powershell
npm run build
```

---

## 🐍 Backend Execution Instructions

The FastAPI backend must be executed separately in a Python environment:

```powershell
cd /d "D:\MINI PROJECT-ImageSuperResolution\ImageSuperResolution\backend"
.\venv\Scripts\activate
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

The backend server runs on `http://127.0.0.1:8000`.

---

## 🧠 Model Routing & Supported Scales

- **2× Scale**: Student RFDB (`~186,603` parameters)
- **4× Scale**: Student RFDB (`~436,011` parameters)
- **8× Scale**: SwinIR-M Transformer (`~11.8`M parameters)
