# Dockerfile for PixelLift Image Super-Resolution API
# Optimized for lightweight production deployment on Render and Docker
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000 \
    MODEL_DIR=/app/checkpoints/student

WORKDIR /app

# Upgrade pip and pre-install CPU-optimized PyTorch to minimize Docker image size
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install backend dependencies
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy backend application source
COPY backend /app/backend

# Copy required inference engine and Student model architecture modules
COPY inference /app/inference
COPY training/models /app/training/models

# Copy required Student checkpoints (x2 and x4)
COPY checkpoints/student/best_student_x2.pth /app/checkpoints/student/best_student_x2.pth
COPY checkpoints/student/best_student_x4.pth /app/checkpoints/student/best_student_x4.pth

# Expose service port
EXPOSE 8000

# Start FastAPI server via Uvicorn with dynamic PORT support for Render
CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
