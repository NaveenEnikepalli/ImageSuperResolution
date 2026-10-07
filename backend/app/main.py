"""FastAPI Main Entry Point for Image Super-Resolution Project.

Author: Antigravity
Purpose: Application startup configuration, middleware setup, CORS, and route aggregation.
"""

import sys
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add repository root and backend directory to sys.path
BACKEND_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = BACKEND_DIR.parent

for p in [str(ROOT_DIR), str(BACKEND_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from backend.app.api.routes import router as api_router
except ModuleNotFoundError:
    from app.api.routes import router as api_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("BackendServerMain")

app = FastAPI(
    title="PixelLift API",
    description="Backend API for PixelLift lightweight AI-based image enhancement using knowledge distillation.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable Cross-Origin Resource Sharing (CORS) for frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "X-Inference-Time",
        "X-Model-Used",
        "X-Input-Resolution",
        "X-Output-Resolution",
        "X-Scale-Factor",
        "Content-Disposition",
    ],
)

# Include API endpoints
app.include_router(api_router)


@app.get("/")
async def root():
    """Root endpoint returning API metadata."""
    return JSONResponse(
        content={
            "name": "PixelLift API",
            "version": "2.0.0",
            "status": "running",
            "docs": "/docs",
            "supported_scales": [2, 4, 8],
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
