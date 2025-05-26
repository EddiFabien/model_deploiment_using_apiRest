#!/usr/bin/env python3
"""
Run the FastAPI application using Uvicorn.
"""
import uvicorn
from app.core.config import get_settings

def main():
    """Run the FastAPI application using Uvicorn."""
    settings = get_settings()
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info" if settings.debug else "warning",
    )

if __name__ == "__main__":
    main()
