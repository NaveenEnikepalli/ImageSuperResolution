"""API Routes for Super-Resolution FastAPI Backend.

Author: Antigravity
Purpose: Define API endpoints for health status checks and image super-resolution enhancement.
"""

import io
import logging
from fastapi import APIRouter, File, Form, UploadFile, HTTPException, status
from fastapi.responses import StreamingResponse, JSONResponse

from backend.app.services.inference import process_image_enhancement

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", tags=["Health"])
async def health_check():
    """Health check status endpoint."""
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "healthy",
            "service": "ImageSuperResolution API",
            "supported_scales": [2, 4, 8],
        },
    )


@router.post("/enhance", tags=["Super Resolution"])
@router.post("/api/v1/enhance", tags=["Super Resolution"])
async def enhance_image(
    file: UploadFile = File(..., description="Target image file (PNG, JPG, JPEG, WEBP)"),
    scale: int = Form(4, description="Resolution scale factor (2, 4, or 8)"),
):
    """Super-resolution image enhancement endpoint.

    Args:
        file (UploadFile): Low-resolution image file payload.
        scale (int): Desired output resolution scale factor (choices: 2, 4, 8).

    Returns:
        StreamingResponse: Enhanced high-resolution PNG image response.
    """
    if scale not in (2, 4, 8):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported scale factor x{scale}. Supported scales are 2, 4, and 8.",
        )

    if not file.content_type or not file.content_type.startswith("image/"):
        logger.warning(f"File upload content type mismatch: {file.content_type}")

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file payload is empty.",
            )

        output_bytes = process_image_enhancement(image_bytes=contents, scale=scale)

        filename = f"enhanced_x{scale}.png"
        return StreamingResponse(
            io.BytesIO(output_bytes),
            media_type="image/png",
            headers={"Content-Disposition": f"inline; filename={filename}"},
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
