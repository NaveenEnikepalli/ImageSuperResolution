"""FastAPI API Routes for Super-Resolution Backend.

Author: Antigravity
Purpose: Define API endpoints for health status, model metadata, and super-resolution image enhancement.
"""

import io
import logging
from fastapi import APIRouter, File, Form, UploadFile, HTTPException, status
from fastapi.responses import StreamingResponse, JSONResponse

try:
    from backend.app.config import MODEL_METADATA, AVAILABLE_SCALES
    from backend.app.schemas.enhancement import HealthResponse, ModelsResponse
    from backend.app.services.enhancement_service import enhance_image
except ModuleNotFoundError:
    from app.config import MODEL_METADATA, AVAILABLE_SCALES
    from app.schemas.enhancement import HealthResponse, ModelsResponse
    from app.services.enhancement_service import enhance_image

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
@router.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check status endpoint."""
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "ok",
            "service": "PixelLift API",
            "supported_scales": AVAILABLE_SCALES,
        },
    )


@router.get("/api/models", response_model=ModelsResponse, tags=["Models"])
async def get_models_info():
    """Return metadata for user-facing super-resolution models without exposing filesystem paths."""
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "supported_scales": AVAILABLE_SCALES,
            "models": MODEL_METADATA,
        },
    )


@router.post("/enhance", tags=["Super Resolution"])
@router.post("/api/enhance", tags=["Super Resolution"])
@router.post("/api/v1/enhance", tags=["Super Resolution"])
async def enhance_image_endpoint(
    image: UploadFile = File(None, description="Target low-resolution image file (PNG, JPG, JPEG, WEBP)"),
    file: UploadFile = File(None, description="Target low-resolution image file (alias)"),
    scale: int = Form(4, description="Output resolution scale factor (choices: 2, 4, 8)"),
):
    """Super-resolution image enhancement endpoint."""
    upload_file = image or file
    if upload_file is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing image file payload. Please provide 'image' or 'file' form parameter.",
        )

    if scale not in (2, 4, 8):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported scale factor x{scale}. Supported scales are 2, 4, and 8.",
        )

    try:
        contents = await upload_file.read()
        if not contents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file payload is empty.",
            )

        result = enhance_image(
            file_bytes=contents,
            scale=scale,
            filename=upload_file.filename,
        )

        output_bytes = result["output_bytes"]
        in_res = f"{result['input_size'][0]}x{result['input_size'][1]}"
        out_res = f"{result['output_size'][0]}x{result['output_size'][1]}"
        elapsed_str = f"{result['inference_time']:.3f}"
        model_name = result["model_name"]

        response_filename = f"enhanced_x{scale}_{upload_file.filename or 'image.png'}"

        headers = {
            "Content-Disposition": f"inline; filename={response_filename}",
            "X-Inference-Time": elapsed_str,
            "X-Model-Used": model_name,
            "X-Input-Resolution": in_res,
            "X-Output-Resolution": out_res,
            "X-Scale-Factor": str(scale),
            "Access-Control-Expose-Headers": "X-Inference-Time, X-Model-Used, X-Input-Resolution, X-Output-Resolution, X-Scale-Factor",
        }

        return StreamingResponse(
            io.BytesIO(output_bytes),
            media_type="image/png",
            headers=headers,
        )

    except ValueError as e:
        logger.error(f"Validation error during image processing: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected backend exception during image enhancement: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Image enhancement processing failed: {str(e)}",
        )
