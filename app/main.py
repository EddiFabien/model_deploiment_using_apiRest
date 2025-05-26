"""
Main FastAPI application module.
"""
import logging
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from app.core.config import get_settings
from app.api.endpoints import router as api_router

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('api_debug.log')
    ]
)
logger = logging.getLogger(__name__)

# Get application settings
settings = get_settings()

# Initialize FastAPI application
app = FastAPI(
    title=settings.app_name,
    description="API for predicting student options (L, S, OSE) based on academic performance and other factors.",
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only, restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix="/api/v1")

# Custom exception handler for validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors and return a clean error response."""
    errors = []
    for error in exc.errors():
        field = ".".join(str(loc) for loc in error["loc"] if loc != "body")
        errors.append({
            "field": field,
            "message": error["msg"],
            "type": error["type"]
        })
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "status": "error",
            "message": "Validation error",
            "errors": errors
        },
    )

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all other exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "message": "Internal server error",
            "detail": str(exc)
        },
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on application startup."""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    
    # Import the prediction service (this will initialize it)
    from app.services.prediction_service import prediction_service  # noqa: F401
    
    # The model is loaded during service initialization
    logger.info("Prediction service initialized")

# Root endpoint
@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint that provides API information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "redoc": "/redoc"
    }
