"""
Model management and inference for the Ultravox model.
"""

import time
import base64
import asyncio
import tempfile
import os
from typing import List, Dict, Any, Optional, AsyncGenerator, Union
import numpy as np
import torch
import librosa
import transformers
from pydantic import BaseModel

from api.chat import Message, ChatResponse

class ModelManager:
    """Manager for loading and running the Ultravox model."""
    
    def __init__(self):
        """Initialize the model manager and load the model."""
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_loaded = False
        self.sampling_rate = 16000  # Required sampling rate for Ultravox
        
    async def load_model(self):
        """Load the Ultravox model asynchronously."""
        if self.model_loaded:
            return
        
        # This is run in a separate thread to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_model_sync)
        
    def _load_model_sync(self):
        """Synchronous model loading function."""
        if self.model_loaded:
            return
            
        try:
            # Load the model using transformers pipeline with trust_remote_code=True
            self.model = transformers.pipeline(
                model='fixie-ai/ultravox-v0_5-llama-3_1-8b',
                device=self.device,
                torch_dtype=torch.bfloat16,  # Use BF16 to save VRAM
                trust_remote_code=True
            )
            self.model_loaded = True
            print(f"Model loaded on {self.device}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
    async def generate_text_response(
        self, 
        messages: List[Message],
        system_prompt: str = "You are a helpful assistant.",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stream: bool = False
    ) -> Union[ChatResponse, AsyncGenerator[str, None]]:
        """
        Generate a response from text input.
        
        Args:
            messages: List of conversation messages
            system_prompt: System prompt for the model
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream the response
            
        Returns:
            Either a ChatResponse object or a streaming generator
        """
        # Ensure model is loaded
        await self.load_model()
        
        # Format the conversation history for the model
        turns = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]
        
        # Add user messages
        for msg in messages:
            turns.append({
                "role": msg.role,
                "content": msg.content
            })
        
        start_time = time.time()
        
        if stream:
            # Return a streaming generator
            return self._stream_text_response(turns, max_tokens, temperature)
        else:
            # Generate the full response at once
            response = await asyncio.to_thread(
                self.model,
                {"turns": turns},
                max_new_tokens=max_tokens,
                temperature=temperature
            )
            
            inference_time = time.time() - start_time
            
            return ChatResponse(
                response=response['generated_text'],
                total_tokens=len(response['generated_text'].split()),  # Approximate token count
                inference_time=inference_time
            )
    
    async def _stream_text_response(
        self,
        turns: List[Dict[str, str]],
        max_tokens: int,
        temperature: float
    ) -> AsyncGenerator[str, None]:
        """Stream the text response token by token."""
        # This is a simplified implementation as Ultravox may not natively support streaming
        # In a production implementation, you'd use the appropriate streaming method
        
        response = await asyncio.to_thread(
            self.model,
            {"turns": turns},
            max_new_tokens=max_tokens,
            temperature=temperature
        )
        
        # Simulate streaming by yielding words
        text = response['generated_text']
        words = text.split()
        
        for word in words:
            yield word + " "
            await asyncio.sleep(0.05)  # Small delay to simulate streaming
    
    async def generate_audio_response(
        self,
        audio_path: str,
        system_prompt: str = "You are a helpful assistant.",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stream: bool = False
    ) -> Union[ChatResponse, AsyncGenerator[str, None]]:
        """
        Generate a response from audio input.
        
        Args:
            audio_path: Path to the audio file
            system_prompt: System prompt for the model
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream the response
            
        Returns:
            Either a ChatResponse object or a streaming generator
        """
        # Ensure model is loaded
        await self.load_model()
        
        # Load and preprocess the audio
        audio, sr = await asyncio.to_thread(librosa.load, audio_path, sr=self.sampling_rate)
        
        # Format the conversation turns
        turns = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]
        
        start_time = time.time()
        
        if stream:
            # Return a streaming generator
            return self._stream_audio_response(audio, turns, max_tokens, temperature)
        else:
            # Generate the full response at once
            response = await asyncio.to_thread(
                self.model,
                {"audio": audio, "turns": turns, "sampling_rate": self.sampling_rate},
                max_new_tokens=max_tokens,
                temperature=temperature
            )
            
            inference_time = time.time() - start_time
            
            return ChatResponse(
                response=response['generated_text'],
                total_tokens=len(response['generated_text'].split()),  # Approximate token count
                inference_time=inference_time
            )
    
    async def _stream_audio_response(
        self,
        audio: np.ndarray,
        turns: List[Dict[str, str]],
        max_tokens: int,
        temperature: float
    ) -> AsyncGenerator[str, None]:
        """Stream the audio response token by token."""
        # Similar to text streaming, with audio input
        
        response = await asyncio.to_thread(
            self.model,
            {"audio": audio, "turns": turns, "sampling_rate": self.sampling_rate},
            max_new_tokens=max_tokens,
            temperature=temperature
        )
        
        # Simulate streaming by yielding words
        text = response['generated_text']
        words = text.split()
        
        for word in words:
            yield word + " "
            await asyncio.sleep(0.05)  # Small delay to simulate streaming
    
    async def generate_audio_response_from_base64(
        self,
        audio_base64: str,
        system_prompt: str = "You are a helpful assistant.",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stream: bool = False
    ) -> Union[ChatResponse, AsyncGenerator[str, None]]:
        """
        Generate a response from base64-encoded audio.
        
        Args:
            audio_base64: Base64-encoded audio data
            system_prompt: System prompt for the model
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream the response
            
        Returns:
            Either a ChatResponse object or a streaming generator
        """
        # Decode the base64 audio to a temporary file
        audio_bytes = base64.b64decode(audio_base64)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_filename = temp_file.name
            temp_file.write(audio_bytes)
        
        try:
            # Process the audio file
            return await self.generate_audio_response(
                audio_path=temp_filename,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream
            )
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_filename):
                os.unlink(temp_filename) 