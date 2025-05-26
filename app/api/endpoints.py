from fastapi import APIRouter, Depends, HTTPException, status
from typing import Any, Dict
import logging

from app.models.schemas import (
    PredictionInput,
    PredictionOutput,
    ErrorResponse,
    HealthCheck
)
from app.services.prediction_service import PredictionService
from app.core.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()

# Initialize prediction service using the from_config class method
prediction_service = PredictionService.from_config()

@router.get("/health", response_model=HealthCheck, status_code=status.HTTP_200_OK)
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint to verify the API is running.
    
    Returns:
        HealthCheck: Status of the API and model
    """
    try:
        # Get model features will automatically load the model if needed
        features = prediction_service.get_model_features()
        
        return {
            "status": "healthy",
            "version": settings.app_version,
            "model_loaded": prediction_service.is_model_loaded,
            "model_features": features["features"]
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "version": settings.app_version,
            "model_loaded": False,
            "model_features": []
        }

@router.post(
    "/predict",
    response_model=PredictionOutput,
    responses={
        status.HTTP_200_OK: {"model": PredictionOutput},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"model": ErrorResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)
async def predict(input_data: PredictionInput) -> Dict[str, Any]:
    """
    Make a prediction using the trained model.
    
    Args:
        input_data (PredictionInput): Input features for prediction
        
    Returns:
        Dict[str, Any]: Prediction result with confidence scores
    """
    try:
        # Convert Pydantic model to dict for processing
        input_dict = input_data.dict()
        
        # Make prediction
        result = prediction_service.predict(input_dict)
        
        # Return the prediction result directly as it already matches the expected schema
        return result
        
    except HTTPException as he:
        # Re-raise HTTP exceptions
        raise he
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"detail": f"An error occurred during prediction: {str(e)}"}
        )

@router.get("/model/features", response_model=Dict[str, Any])
async def get_model_features() -> Dict[str, Any]:
    """
    Get the list of features expected by the model.
    
    Returns:
        Dict containing the list of features
    """
    try:
        features = prediction_service.get_model_features()
        return features
        
    except Exception as e:
        logger.error(f"Error getting model features: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"detail": f"Error getting model features: {str(e)}"}
        )
