"""
Chat API routes for handling text and audio requests.
"""

from typing import List, Dict, Any, Optional
import tempfile
import os
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from models.schemas import Message, ChatRequest, ChatResponse
from models.inference import ModelManager

router = APIRouter(tags=["chat"])

# Initialize model manager
model_manager = ModelManager()

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process a text chat request.
    
    Args:
        request: ChatRequest object containing messages history
        
    Returns:
        Response from the model
    """
    try:
        result = await model_manager.generate_text_response(
            messages=request.messages,
            system_prompt=request.system_prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@router.post("/chat/audio", response_model=ChatResponse)
async def chat_audio(
    audio: UploadFile = File(...),
    system_prompt: str = Form("You are a helpful assistant."),
    max_tokens: int = Form(1024),
    temperature: float = Form(0.7)
):
    """
    Process an audio chat request.
    
    Args:
        audio: Audio file upload
        system_prompt: System prompt for the model
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        
    Returns:
        Response from the model
    """
    try:
        # Save the uploaded audio file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_filename = temp_file.name
            audio_content = await audio.read()
            temp_file.write(audio_content)
        
        # Process the audio file
        try:
            result = await model_manager.generate_audio_response(
                audio_path=temp_filename,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return result
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}") 