"""
API Key Authentication middleware for FastAPI.

Simple API key validation for protecting endpoints.
"""
import os
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

# API key from environment variable (or default for development)
API_KEY = os.environ.get("RAG_API_KEY", "dev-api-key-change-in-production")
API_KEY_NAME = "X-API-Key"

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """
    Verify the API key from request header.
    
    Usage:
        @app.get("/protected")
        def protected_route(api_key: str = Depends(verify_api_key)):
            return {"message": "Access granted"}
    """
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required. Include 'X-API-Key' header."
        )
    
    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key"
        )
    
    return api_key


def get_api_key():
    """Get the current API key (for testing/debugging)"""
    return API_KEY
