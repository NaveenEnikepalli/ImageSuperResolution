"""Pydantic Schemas for API Endpoints.

Author: Antigravity
Purpose: Define request and response schemas for FastAPI endpoints.
"""

from typing import List, Dict, Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check endpoint response schema."""

    status: str = Field(..., json_schema_extra={"example": "ok"})
    service: str = Field(..., json_schema_extra={"example": "ImageSuperResolution API"})
    supported_scales: List[int] = Field(..., json_schema_extra={"example": [2, 4, 8]})


class ModelInfo(BaseModel):
    """Information for a single super-resolution model scale."""

    name: str = Field(..., json_schema_extra={"example": "Student x4"})
    type: str = Field(..., json_schema_extra={"example": "Lightweight RFDB Student"})
    scale: str = Field(..., json_schema_extra={"example": "4×"})
    parameters: str = Field(..., json_schema_extra={"example": "~436,011"})
    description: str = Field(
        ...,
        json_schema_extra={
            "example": "Lightweight Student RFDB model for balanced 4x deep restoration."
        },
    )


class ModelsResponse(BaseModel):
    """Response schema listing available super-resolution models."""

    supported_scales: List[int] = Field(..., json_schema_extra={"example": [2, 4, 8]})
    models: Dict[int, ModelInfo]
