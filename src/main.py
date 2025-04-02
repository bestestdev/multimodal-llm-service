#!/usr/bin/env python3
"""
Multimodal LLM Service main application.
This module initializes the FastAPI application and sets up routes.
"""

import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from api.chat import router as chat_router
from api.websocket import router as websocket_router
from utils.project import setup_logging, check_gpu_compatibility, print_startup_message

# Set up logging
setup_logging()

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Initialize FastAPI app
app = FastAPI(
    title="Multimodal LLM Service",
    description="A realtime voice/text communication service powered by Ultravox",
    version="0.1.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat_router, prefix="/api")
app.include_router(websocket_router, prefix="/ws")

# Serve static files for frontend
frontend_build_path = os.path.join(PROJECT_ROOT, "frontend", "build")
app.mount("/app", StaticFiles(directory=frontend_build_path, html=True), name="app")

@app.get("/")
async def root():
    """Root endpoint redirects to the API documentation."""
    return {"message": "Multimodal LLM Service API", "docs_url": "/docs"}

@app.on_event("startup")
async def startup_event():
    """Run initialization on startup."""
    print_startup_message()
    check_gpu_compatibility()

if __name__ == "__main__":
    # Run the application with uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True) 