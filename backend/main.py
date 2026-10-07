"""FastAPI Main Server Entry Point.

Author: Antigravity
Purpose: Primary entrypoint for launching FastAPI backend server.
"""

import sys
import logging
from pathlib import Path

# Add repository root and backend directory to sys.path
BACKEND_DIR = Path(__file__).resolve().parent
ROOT_DIR = BACKEND_DIR.parent

for p in [str(ROOT_DIR), str(BACKEND_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from backend.app.main import app
except ModuleNotFoundError:
    from app.main import app

if __name__ == "__main__":
    import uvicorn
    logging.info("Starting FastAPI backend server on 127.0.0.1:8000...")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
