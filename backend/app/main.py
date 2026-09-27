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

# Add repository root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.api.routes import router as api_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("BackendMain")

app = FastAPI(
    title="Image Super-Resolution API",
    description="High-Performance Deep Learning Image Super-Resolution (2x, 4x, 8x)",
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
)

# Include API endpoints
app.include_router(api_router)


@app.get("/")
async def root():
    """Root endpoint returning API metadata."""
    return JSONResponse(
        content={
            "name": "Image Super-Resolution API",
            "version": "2.0.0",
            "status": "running",
            "docs": "/docs",
            "supported_scales": ["2x", "4x", "8x"],
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
